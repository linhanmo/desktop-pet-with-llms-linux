import os
import sys
import json
import time
import io

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import glob
import re
import argparse


def _move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_move_to_cpu(x) for x in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj


def _bytes_to_u8_tensor(b: bytes) -> torch.Tensor:
    if not isinstance(b, (bytes, bytearray)):
        raise TypeError("需要 bytes/bytearray")
    if hasattr(torch, "frombuffer"):
        try:
            t = torch.frombuffer(memoryview(b), dtype=torch.uint8)
            return t.clone()
        except Exception:
            pass
    return torch.tensor(list(b), dtype=torch.uint8)


def _u8_tensor_to_bytes(t: torch.Tensor) -> bytes:
    if not isinstance(t, torch.Tensor):
        raise TypeError("需要 torch.Tensor")
    t = t.detach().cpu().to(torch.uint8)
    try:
        return t.numpy().tobytes()
    except Exception:
        return bytes(t.tolist())


def _save_state_dict_safetensors(state_dict: dict, path: str):
    try:
        from safetensors.torch import save_file
    except Exception as e:
        raise RuntimeError(
            "未安装 safetensors，无法保存 .safetensors 权重文件。请先安装: pip install safetensors"
        ) from e
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("state_dict 为空，无法保存 safetensors")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensors = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and isinstance(v, torch.Tensor):
            tensors[k] = _move_to_cpu(v).contiguous()
    if not tensors:
        raise ValueError("state_dict 中没有可保存的 Tensor")
    save_file(tensors, path)


def _save_checkpoint_safetensors(checkpoint_state: dict, path: str):
    try:
        from safetensors.torch import save_file
    except Exception as e:
        raise RuntimeError(
            "未安装 safetensors，无法保存 .safetensors 检查点文件。请先安装: pip install safetensors"
        ) from e

    if not isinstance(checkpoint_state, dict) or not checkpoint_state:
        raise ValueError("checkpoint_state 为空")
    weights = checkpoint_state.get("model_state_dict") or {}
    if not isinstance(weights, dict) or not weights:
        raise ValueError("model_state_dict 为空，无法保存检查点")

    meta_state = dict(checkpoint_state)
    meta_state.pop("model_state_dict", None)

    buf = io.BytesIO()
    torch.save(_move_to_cpu(meta_state), buf)
    meta_u8 = _bytes_to_u8_tensor(buf.getvalue())

    tensors = {"__checkpoint_state__": meta_u8}
    for k, v in weights.items():
        if isinstance(k, str) and isinstance(v, torch.Tensor):
            tensors[k] = _move_to_cpu(v).contiguous()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path)


def _load_checkpoint_safetensors(path: str) -> dict:
    try:
        from safetensors.torch import load_file
    except Exception as e:
        raise RuntimeError(
            "未安装 safetensors，无法加载 .safetensors 检查点文件。请先安装: pip install safetensors"
        ) from e
    tensors = load_file(path, device="cpu")
    meta_u8 = tensors.pop("__checkpoint_state__", None)
    if meta_u8 is None:
        raise ValueError("检查点缺少 __checkpoint_state__")
    meta_bytes = _u8_tensor_to_bytes(meta_u8)
    try:
        meta_state = torch.load(
            io.BytesIO(meta_bytes), map_location="cpu", weights_only=False
        )
    except TypeError:
        meta_state = torch.load(io.BytesIO(meta_bytes), map_location="cpu")
    if not isinstance(meta_state, dict):
        raise ValueError("检查点 meta 不是 dict")
    meta_state["model_state_dict"] = {k: v for k, v in tensors.items() if isinstance(k, str)}
    return meta_state

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from core.model import LlamaLite
    from core.config import (
        LLAMA_LITE_150M_CONFIG,
        LLAMA_LITE_450M_CONFIG,
        LLAMA_LITE_1_5B_CONFIG,
    )
    from chinese_tokenizer import ChineseTokenizer
    from core.dataset import (
        BinaryTokenIterableDataset,
        create_dataloader_v1,
        load_binary_dataset_meta,
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在 llm 目录下运行脚本，或检查目录结构。")
    sys.exit(1)

TRAIN_CONFIG = {
    "train_data_path": os.path.join(CURRENT_DIR, "data", "layer1", "train.bin"),
    "val_data_path": os.path.join(CURRENT_DIR, "data", "layer1", "val.bin"),
    "batch_size": 2,
    "gradient_accumulation_steps": 32,
    "mixed_precision": True,
    "precision_dtype": "bf16",
    "num_workers": 0 if os.name == "nt" else 4,
    "epoch_shards": 10,
    "shuffle_buffer_size": 8192,
    "seed": 1234,
    "max_seq_len": 1024,
    "epochs": 60,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 1000,
    "min_lr_ratio": 0.1,
    "log_interval": 10,
    "eval_interval": 500,
    "weight_save_dir": os.path.join(CURRENT_DIR, "checkpoints"),
    "model_size": "150m",
}


def get_lr_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    last_epoch: int = -1,
):

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = max(0.0, min(1.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_epoch
    )


@torch.no_grad()
def calculate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    dtype: torch.dtype,
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    steps = 0

    print(f"正在计算验证集困惑度 (共 {len(dataloader)} 个批次)...")

    for x, y in tqdm(dataloader, desc="Validating", leave=False):
        x = x.to(device=device, dtype=torch.long, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype)

        with autocast_ctx:
            logits = model(x)
            loss = F.cross_entropy(
                logits.flatten(0, 1), y.flatten(0, 1), reduction="sum"
            )

        total_loss += loss.item()
        total_tokens += y.numel()
        steps += 1

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    model.train()
    return avg_loss, perplexity


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None, 0

    candidates = []
    for cp in glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.safetensors")):
        match = re.search(r"model_epoch_(\d+)\.safetensors", cp)
        if match:
            candidates.append((int(match.group(1)), cp))
    if not candidates:
        return None, 0
    epoch, path = sorted(candidates)[-1]
    return path, epoch


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        default=os.environ.get(
            "LLM_MODEL_SIZE", TRAIN_CONFIG.get("model_size", "150m")
        ),
    )
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--weight_save_dir", type=str, default=None)
    args = parser.parse_args(argv)

    TRAIN_CONFIG["model_size"] = str(args.model_size).lower()
    if args.train_data_path:
        TRAIN_CONFIG["train_data_path"] = args.train_data_path
    if args.val_data_path:
        TRAIN_CONFIG["val_data_path"] = args.val_data_path
    if args.weight_save_dir:
        TRAIN_CONFIG["weight_save_dir"] = args.weight_save_dir

    os.makedirs(TRAIN_CONFIG["weight_save_dir"], exist_ok=True)
    print(f"===== LlamaLite 中文预训练脚本 =====")
    print(f"权重保存目录: {TRAIN_CONFIG['weight_save_dir']}")
    metrics_path = os.path.join(
        TRAIN_CONFIG["weight_save_dir"], "training_metrics.json"
    )
    plot_path = os.path.join(TRAIN_CONFIG["weight_save_dir"], "training_plot.png")
    training_history = []
    train_window_loss_sum = 0.0
    train_window_loss_count = 0
    consecutive_worsen = 0
    best_val_loss_seen = None

    def _safe_read_json(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _atomic_write_text(path: str, text: str):
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    def _atomic_write_json(path: str, obj):
        _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))

    def _append_metrics(record: dict):
        existing = _safe_read_json(metrics_path)
        records = []
        if isinstance(existing, dict) and isinstance(existing.get("records"), list):
            records = existing["records"]
        elif isinstance(existing, list):
            records = existing
        records.append(record)
        payload = {"records": records}
        _atomic_write_json(metrics_path, payload)

    def _assess_overfit(train_loss_recent, val_loss, val_ppl):
        nonlocal consecutive_worsen, best_val_loss_seen
        if val_loss is None or not isinstance(val_loss, (int, float)):
            return "none", ""
        if best_val_loss_seen is None or val_loss < best_val_loss_seen:
            best_val_loss_seen = val_loss
            consecutive_worsen = 0
            return "none", ""
        if len(training_history) >= 1:
            prev = training_history[-1]
            prev_val = prev.get("val_loss")
            prev_train = prev.get("train_loss_recent")
            if (
                isinstance(prev_val, (int, float))
                and isinstance(prev_train, (int, float))
                and isinstance(train_loss_recent, (int, float))
            ):
                if val_loss > prev_val and train_loss_recent < prev_train:
                    consecutive_worsen += 1
                elif val_loss <= prev_val:
                    consecutive_worsen = max(0, consecutive_worsen - 1)
        gap = None
        if isinstance(train_loss_recent, (int, float)):
            gap = val_loss - train_loss_recent
        if consecutive_worsen >= 3 or (gap is not None and gap > 0.25):
            return "high", (
                f"val_loss 连续走高={consecutive_worsen}, gap={gap:.4f}"
                if gap is not None
                else f"val_loss 连续走高={consecutive_worsen}"
            )
        if consecutive_worsen == 2 or (gap is not None and gap > 0.15):
            return "medium", (
                f"val_loss 连续走高={consecutive_worsen}, gap={gap:.4f}"
                if gap is not None
                else f"val_loss 连续走高={consecutive_worsen}"
            )
        if consecutive_worsen == 1 or (gap is not None and gap > 0.08):
            return "low", (
                f"val_loss 走高迹象, gap={gap:.4f}"
                if gap is not None
                else "val_loss 走高迹象"
            )
        return "none", ""

    def _write_plot_matplotlib(records, out_path: str):
        try:
            import matplotlib

            try:
                matplotlib.use("Agg", force=True)
            except Exception:
                pass
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"matplotlib 不可用，跳过图表生成: {e}")
            return False

        series = [
            (
                r.get("global_step"),
                r.get("train_loss_recent"),
                r.get("val_loss"),
                r.get("val_ppl"),
                r.get("epoch"),
            )
            for r in records
            if r.get("kind") in ("step_eval", "epoch_eval")
        ]
        xs = [s[0] for s in series if isinstance(s[0], (int, float))]
        if not xs:
            return False

        train_x = [
            s[0]
            for s in series
            if isinstance(s[0], (int, float)) and isinstance(s[1], (int, float))
        ]
        train_y = [
            s[1]
            for s in series
            if isinstance(s[0], (int, float)) and isinstance(s[1], (int, float))
        ]
        val_x = [
            s[0]
            for s in series
            if isinstance(s[0], (int, float)) and isinstance(s[2], (int, float))
        ]
        val_y = [
            s[2]
            for s in series
            if isinstance(s[0], (int, float)) and isinstance(s[2], (int, float))
        ]
        ppl_x = [
            s[0]
            for s in series
            if isinstance(s[0], (int, float)) and isinstance(s[3], (int, float))
        ]
        ppl_y = [
            s[3]
            for s in series
            if isinstance(s[0], (int, float)) and isinstance(s[3], (int, float))
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True, dpi=130)
        if train_x:
            ax1.plot(
                train_x,
                train_y,
                label="train_loss_recent",
                color="#1f77b4",
                linewidth=1.8,
            )
        if val_x:
            ax1.plot(val_x, val_y, label="val_loss", color="#d62728", linewidth=1.8)
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")

        if ppl_x:
            ax2.plot(ppl_x, ppl_y, label="val_ppl", color="#2ca02c", linewidth=1.8)
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("PPL")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

        last = records[-1] if records else {}
        fig.suptitle(
            f"epoch={last.get('epoch')}, step={last.get('global_step')}, val_ppl={last.get('val_ppl')}"
        )
        fig.tight_layout()
        tmp = f"{out_path}.tmp"
        try:
            fig.savefig(tmp)
            os.replace(tmp, out_path)
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass
        return True

    desired_epochs = int(TRAIN_CONFIG.get("epochs", 0) or 0)
    resume_checkpoint = {}
    resume_latest_checkpoint_path, resume_latest_epoch = find_latest_checkpoint(
        TRAIN_CONFIG["weight_save_dir"]
    )
    if resume_latest_checkpoint_path:
        try:
            resume_checkpoint = _load_checkpoint_safetensors(resume_latest_checkpoint_path)
        except Exception as e:
            print(f"检查点加载失败，将忽略检查点并从头开始训练: {e}")
            resume_latest_checkpoint_path = None
            resume_latest_epoch = 0
            resume_checkpoint = {}
        ckpt_cfg = (
            resume_checkpoint.get("config")
            if isinstance(resume_checkpoint, dict)
            else None
        )
        if isinstance(ckpt_cfg, dict):
            keep_keys = {
                "train_data_path",
                "val_data_path",
                "weight_save_dir",
                "model_size",
            }
            keep_values = {k: TRAIN_CONFIG.get(k) for k in keep_keys}
            for k in list(TRAIN_CONFIG.keys()):
                if k in ckpt_cfg:
                    TRAIN_CONFIG[k] = ckpt_cfg[k]
            for k, v in keep_values.items():
                if v is not None:
                    TRAIN_CONFIG[k] = v
            if desired_epochs > 0:
                TRAIN_CONFIG["epochs"] = max(
                    int(TRAIN_CONFIG.get("epochs", 0) or 0), desired_epochs
                )
            print("已从检查点恢复训练配置。")

    latest_checkpoint_path = resume_latest_checkpoint_path
    latest_epoch = resume_latest_epoch
    checkpoint = resume_checkpoint

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"使用设备: GPU ({gpu_name}, {vram:.1f} GB VRAM)")

        if vram < 7.5:
            print("警告: 显存小于 8GB，建议减小 batch_size 或使用混合精度。")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 GPU，将使用 CPU 训练 (速度极慢)。")

    use_amp = TRAIN_CONFIG["mixed_precision"] and torch.cuda.is_available()
    amp_dtype = torch.float16

    if use_amp and TRAIN_CONFIG["precision_dtype"] == "bf16":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("已启用 BFloat16 混合精度训练 (BF16)")
        else:
            print("警告: 当前设备不支持 BF16，回退到 FP16")
            amp_dtype = torch.float16
    else:
        print(f"混合精度训练 (AMP): {'开启 (FP16)' if use_amp else '关闭'}")

    print("正在加载分词器...")
    tokenizer = ChineseTokenizer()
    print(f"分词器词表大小: {tokenizer.vocab_size}")

    def resolve_data_path(path: str) -> str:
        if os.path.exists(path):
            return path
        if path.endswith(".bin"):
            alt = path[:-4] + ".pt"
            if os.path.exists(alt):
                return alt
        if path.endswith(".pt"):
            alt = path[:-3] + "bin"
            if os.path.exists(alt):
                return alt
        return path

    print("正在加载预处理数据集...")
    train_data_path = resolve_data_path(TRAIN_CONFIG["train_data_path"])
    val_data_path = resolve_data_path(TRAIN_CONFIG["val_data_path"])

    if os.path.exists(train_data_path) and train_data_path.endswith(".bin"):
        try:
            tm = load_binary_dataset_meta(train_data_path)
            print(
                f"训练集 meta: seq_len={tm.get('seq_len')}, num_samples={tm.get('num_samples')}, chunks_total={tm.get('chunks_total')}, texts_kept={tm.get('texts_kept')}, texts_skipped={tm.get('texts_skipped')}, duplicates={tm.get('duplicates')}"
            )
            sources = tm.get("sources") or {}
            for s, st in list(sources.items())[:5]:
                print(
                    f"源 {s}: files={st.get('files')}, lines={st.get('lines')}, kept={st.get('kept')}, skipped={st.get('skipped')}, dup={st.get('duplicates')}, chunks={st.get('chunks')}"
                )
        except Exception as e:
            print(f"训练集 meta 读取失败: {e}")
    if os.path.exists(val_data_path) and val_data_path.endswith(".bin"):
        try:
            vm = load_binary_dataset_meta(val_data_path)
            print(
                f"验证集 meta: seq_len={vm.get('seq_len')}, num_samples={vm.get('num_samples')}, chunks_total={vm.get('chunks_total')}, texts_kept={vm.get('texts_kept')}, texts_skipped={vm.get('texts_skipped')}, duplicates={vm.get('duplicates')}"
            )
        except Exception as e:
            print(f"验证集 meta 读取失败: {e}")

    if not os.path.exists(train_data_path):
        print(f"错误: 找不到训练数据 {train_data_path}")
        print("请先运行 chinese_pretrain_data.py 生成数据。")
        sys.exit(1)

    is_binary_train = isinstance(train_data_path, str) and train_data_path.endswith(
        ".bin"
    )

    train_input_ids = None
    train_target_ids = None
    train_meta = None

    if is_binary_train:
        train_meta = load_binary_dataset_meta(train_data_path)
        seq_len = int(train_meta["seq_len"])
        num_samples = int(train_meta["num_samples"])
        epoch_shards = max(1, int(TRAIN_CONFIG.get("epoch_shards", 1)))
        shard_size = (num_samples + epoch_shards - 1) // epoch_shards

        def build_train_loader_for_epoch(epoch_idx: int):
            shard_id = epoch_idx % epoch_shards
            start = shard_id * shard_size
            end = min(start + shard_size, num_samples)
            dataset = BinaryTokenIterableDataset(
                train_data_path,
                seq_len=seq_len,
                num_samples=num_samples,
                start_sample=start,
                end_sample=end,
                shuffle_buffer_size=int(TRAIN_CONFIG.get("shuffle_buffer_size", 0)),
                seed=int(TRAIN_CONFIG.get("seed", 1234)) + epoch_idx,
            )
            return DataLoader(
                dataset,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=False,
                drop_last=True,
                num_workers=TRAIN_CONFIG["num_workers"],
                pin_memory=True,
            )

        train_loader = build_train_loader_for_epoch(0)
    else:
        print(f"Loading data from {train_data_path}...")
        train_data_content = torch.load(
            train_data_path, weights_only=False, map_location="cpu"
        )
        train_input_ids = train_data_content["input_ids"]
        train_target_ids = train_data_content["target_ids"]

        train_loader = create_dataloader_v1(
            (train_input_ids, train_target_ids),
            batch_size=TRAIN_CONFIG["batch_size"],
            max_length=TRAIN_CONFIG["max_seq_len"],
            shuffle=True,
            drop_last=True,
            num_workers=TRAIN_CONFIG["num_workers"],
        )

    if not os.path.exists(val_data_path):
        print(f"错误: 找不到验证数据 {val_data_path}")
        print("请先运行 chinese_pretrain_data.py 生成数据。")
        sys.exit(1)

    val_loader = create_dataloader_v1(
        val_data_path,
        batch_size=TRAIN_CONFIG["batch_size"],
        max_length=TRAIN_CONFIG["max_seq_len"],
        shuffle=False,
        drop_last=False,
        num_workers=TRAIN_CONFIG["num_workers"],
    )

    try:
        print(f"训练集批次数: {len(train_loader)}")
    except Exception:
        pass
    try:
        print(f"验证集批次数: {len(val_loader)}")
    except Exception:
        pass

    print("正在初始化 LlamaLite 模型...")
    model_presets = {
        "150m": LLAMA_LITE_150M_CONFIG,
        "450m": LLAMA_LITE_450M_CONFIG,
        "1.5b": LLAMA_LITE_1_5B_CONFIG,
        "1_5b": LLAMA_LITE_1_5B_CONFIG,
        "1p5b": LLAMA_LITE_1_5B_CONFIG,
    }
    preset = model_presets.get(
        str(TRAIN_CONFIG.get("model_size") or "150m").lower(), LLAMA_LITE_150M_CONFIG
    )
    model_config = preset.copy()
    if model_config["vocab_size"] != tokenizer.vocab_size:
        print(
            f"警告: 配置文件词表大小 ({model_config['vocab_size']}) 与分词器 ({tokenizer.vocab_size}) 不一致，正在自动调整..."
        )
        model_config["vocab_size"] = tokenizer.vocab_size

    print("正在抽样检查训练数据，确保 Token ID 不越界...")
    try:
        if is_binary_train:
            check_count = min(
                int(train_meta["num_samples"]),
                max(1, int(TRAIN_CONFIG["batch_size"]) * 64),
            )
            check_dataset = BinaryTokenIterableDataset(
                train_data_path,
                seq_len=int(train_meta["seq_len"]),
                num_samples=int(train_meta["num_samples"]),
                start_sample=0,
                end_sample=check_count,
                shuffle_buffer_size=0,
                seed=int(TRAIN_CONFIG.get("seed", 1234)),
            )
            check_loader = DataLoader(
                check_dataset,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
        else:
            check_loader = create_dataloader_v1(
                (train_input_ids, train_target_ids),
                batch_size=TRAIN_CONFIG["batch_size"],
                max_length=TRAIN_CONFIG["max_seq_len"],
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )

        data_iter = iter(check_loader)
        check_batches = 3
        max_id_observed = -1
        for _ in range(check_batches):
            batch = next(data_iter)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    bx, by = batch
                else:
                    bx = batch[0]
                    by = None
            else:
                bx = batch
                by = None

            def max_id(t):
                if t is None:
                    return -1
                if isinstance(t, torch.Tensor):
                    return int(t.max().item())
                if isinstance(t, (list, tuple)):
                    m = -1
                    for item in t:
                        if isinstance(item, torch.Tensor):
                            m = max(m, int(item.max().item()))
                    return m
                return -1

            max_id_observed = max(max_id_observed, max_id(bx), max_id(by))

        print(
            f"Sample Max Token ID (over {check_batches} batches): {max_id_observed}, Vocab Size: {model_config['vocab_size']}"
        )
        if max_id_observed >= model_config["vocab_size"]:
            print(
                f"严重错误: 数据集中存在越界 Token ID {max_id_observed} >= {model_config['vocab_size']}"
            )
            print(
                "建议：使用当前分词器重新生成数据 (运行 chinese_pretrain_data.py)，或确保数据与分词器版本一致。"
            )
            sys.exit(1)
        else:
            print("数据抽样检查通过。")
    except StopIteration:
        print("数据检查时发生错误：数据加载器为空。")
    except Exception as e:
        print(f"数据检查时发生未知错误: {e}")

    def _load_state_dict_like(obj):
        if not isinstance(obj, dict):
            return None
        if "model_state_dict" in obj and isinstance(obj.get("model_state_dict"), dict):
            return obj["model_state_dict"]
        if "state_dict" in obj and isinstance(obj.get("state_dict"), dict):
            return obj["state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
        return None

    def _load_weights_from_path(path: str):
        if not path or not isinstance(path, str) or not os.path.exists(path):
            return None
        try:
            sd_obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            sd_obj = torch.load(path, map_location="cpu")
        state_dict = _load_state_dict_like(sd_obj)
        return state_dict

    model = LlamaLite(model_config).to(device)
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)

    if hasattr(torch, "compile"):
        print("暂时禁用 torch.compile 以避免 CUDAGraphs 张量覆盖问题。")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(TRAIN_CONFIG["beta1"], TRAIN_CONFIG["beta2"]),
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    if is_binary_train:
        num_samples = int(train_meta["num_samples"])
        epoch_shards = max(1, int(TRAIN_CONFIG.get("epoch_shards", 1)))
        shard_size = (num_samples + epoch_shards - 1) // epoch_shards
        batches_per_epoch = shard_size // int(TRAIN_CONFIG["batch_size"])
        total_steps = (
            batches_per_epoch
            * int(TRAIN_CONFIG["epochs"])
            // int(TRAIN_CONFIG["gradient_accumulation_steps"])
        )
    else:
        total_steps = (
            len(train_loader)
            * TRAIN_CONFIG["epochs"]
            // TRAIN_CONFIG["gradient_accumulation_steps"]
        )
    print(f"预计总训练步数 (Gradient Steps): {total_steps}")

    start_epoch = 0
    global_step = 0
    best_perplexity = float("inf")

    if latest_checkpoint_path and checkpoint:
        print(f"===== 发现检查点: {latest_checkpoint_path} =====")
        print(f"正在恢复训练状态 (Epoch {latest_epoch})...")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"优化器状态恢复失败，将重置优化器状态: {e}")
            start_epoch = int(checkpoint.get("epoch", latest_epoch) or 0)
            global_step = int(checkpoint.get("global_step", 0) or 0)
            best_perplexity = float(checkpoint.get("best_perplexity", float("inf")))
            print(
                f"成功恢复至 Epoch {start_epoch}, Step {global_step}, Best PPL {best_perplexity:.2f}"
            )
        else:
            print("检测到旧版本检查点格式，仅恢复模型权重，重置优化器状态。")
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
            start_epoch = latest_epoch
            if is_binary_train and train_meta is not None:
                num_samples = int(train_meta["num_samples"])
                epoch_shards = max(1, int(TRAIN_CONFIG.get("epoch_shards", 1)))
                shard_size = (num_samples + epoch_shards - 1) // epoch_shards
                batches_per_epoch = shard_size // int(TRAIN_CONFIG["batch_size"])
                global_step = (
                    batches_per_epoch
                    // int(TRAIN_CONFIG["gradient_accumulation_steps"])
                ) * latest_epoch
            else:
                global_step = (
                    len(train_loader) // TRAIN_CONFIG["gradient_accumulation_steps"]
                ) * latest_epoch

    if global_step - 1 >= 0:
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group.get("lr", TRAIN_CONFIG["learning_rate"])

    lr_scheduler = get_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        min_lr_ratio=TRAIN_CONFIG["min_lr_ratio"],
        last_epoch=global_step - 1,
    )

    if latest_checkpoint_path and "scheduler_state_dict" in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            print(f"学习率调度器状态恢复失败，将重置调度器状态: {e}")

    scaler_enabled = use_amp and (amp_dtype == torch.float16)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    if latest_checkpoint_path and "scaler_state_dict" in checkpoint and scaler_enabled:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    model.train()

    if start_epoch >= TRAIN_CONFIG["epochs"]:
        print("所有 Epoch 已完成，无需训练。")
        return

    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"]):
        print(f"\n===== Epoch {epoch + 1}/{TRAIN_CONFIG['epochs']} =====")
        epoch_loss = 0.0
        optimizer.zero_grad()
        batches_seen = 0

        if is_binary_train:
            epoch_train_loader = build_train_loader_for_epoch(epoch)
        else:
            epoch_train_loader = train_loader

        progress_bar = tqdm(epoch_train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device=device, dtype=torch.long, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)
            batches_seen += 1

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                )
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)

            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(0, 1))
                loss = loss / TRAIN_CONFIG["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            current_loss = loss.item() * TRAIN_CONFIG["gradient_accumulation_steps"]
            epoch_loss += current_loss

            if (batch_idx + 1) % TRAIN_CONFIG["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), TRAIN_CONFIG["grad_clip"]
                )

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                train_window_loss_sum += float(current_loss)
                train_window_loss_count += 1

                current_lr = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(
                    {
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{current_lr:.6f}",
                        "step": global_step,
                    }
                )

                if global_step % TRAIN_CONFIG["eval_interval"] == 0:
                    tqdm.write(f"\n[Step {global_step}] 开始验证...")
                    val_loss, val_ppl = calculate_perplexity(
                        model, val_loader, device, use_amp, amp_dtype
                    )
                    tqdm.write(f"验证集 Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")
                    train_loss_recent = None
                    if train_window_loss_count > 0:
                        train_loss_recent = (
                            train_window_loss_sum / train_window_loss_count
                        )
                    risk_level, risk_reason = _assess_overfit(
                        train_loss_recent, val_loss, val_ppl
                    )
                    record = {
                        "ts": time.time(),
                        "epoch": int(epoch + 1),
                        "global_step": int(global_step),
                        "kind": "step_eval",
                        "train_loss_recent": train_loss_recent,
                        "val_loss": (
                            float(val_loss)
                            if isinstance(val_loss, (int, float))
                            else None
                        ),
                        "val_ppl": (
                            float(val_ppl)
                            if isinstance(val_ppl, (int, float))
                            else None
                        ),
                        "lr": (
                            float(current_lr)
                            if isinstance(current_lr, (int, float))
                            else None
                        ),
                        "risk_level": risk_level,
                        "risk_reason": risk_reason,
                    }
                    training_history.append(record)
                    _append_metrics(record)
                    if risk_level != "none":
                        tqdm.write(f"⚠️ 过拟合风险[{risk_level}]: {risk_reason}")
                    train_window_loss_sum = 0.0
                    train_window_loss_count = 0

                    if val_ppl < best_perplexity:
                        best_perplexity = val_ppl
                        best_path = os.path.join(
                            TRAIN_CONFIG["weight_save_dir"], "best_model.safetensors"
                        )
                        _save_state_dict_safetensors(model.state_dict(), best_path)
                        tqdm.write(
                            f"✅ 新的最优模型已保存 (PPL: {best_perplexity:.2f})"
                        )

        avg_epoch_loss = epoch_loss / max(1, batches_seen)
        print(f"\nEpoch {epoch + 1} 结束，平均 Loss: {avg_epoch_loss:.4f}")

        val_loss, val_ppl = calculate_perplexity(
            model, val_loader, device, use_amp, amp_dtype
        )
        print(f"Epoch {epoch + 1} 验证集 PPL: {val_ppl:.2f}")
        risk_level, risk_reason = _assess_overfit(avg_epoch_loss, val_loss, val_ppl)
        epoch_record = {
            "ts": time.time(),
            "epoch": int(epoch + 1),
            "global_step": int(global_step),
            "kind": "epoch_eval",
            "train_loss_recent": float(avg_epoch_loss),
            "val_loss": float(val_loss) if isinstance(val_loss, (int, float)) else None,
            "val_ppl": float(val_ppl) if isinstance(val_ppl, (int, float)) else None,
            "lr": (
                float(lr_scheduler.get_last_lr()[0])
                if hasattr(lr_scheduler, "get_last_lr")
                else None
            ),
            "risk_level": risk_level,
            "risk_reason": risk_reason,
        }
        training_history.append(epoch_record)
        _append_metrics(epoch_record)
        if risk_level != "none":
            print(f"⚠️ 过拟合风险[{risk_level}]: {risk_reason}")
        try:
            existing = _safe_read_json(metrics_path)
            recs = (
                existing.get("records")
                if isinstance(existing, dict)
                else (existing if isinstance(existing, list) else None)
            )
            if isinstance(recs, list) and recs:
                if _write_plot_matplotlib(recs, plot_path):
                    print(f"已更新训练可视化图表: {plot_path}")
        except Exception as e:
            print(f"训练可视化图表更新失败: {e}")

        checkpoint_state = {
            "epoch": int(epoch + 1),
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler_enabled else None,
            "best_perplexity": float(best_perplexity),
            "config": TRAIN_CONFIG,
        }
        epoch_path = os.path.join(
            TRAIN_CONFIG["weight_save_dir"], f"model_epoch_{epoch + 1}.safetensors"
        )
        _save_checkpoint_safetensors(checkpoint_state, epoch_path)
        print(f"已保存 Epoch 检查点(safetensors): {epoch_path}")

    print(f"\n===== 训练结束 =====")
    print(f"最优验证集 PPL: {best_perplexity:.2f}")
    final_path = os.path.join(TRAIN_CONFIG["weight_save_dir"], "final_model.safetensors")
    _save_state_dict_safetensors(model.state_dict(), final_path)
    print(f"最终模型已保存: {final_path}")


if __name__ == "__main__":
    main()
