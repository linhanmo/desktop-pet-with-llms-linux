import os
import sys
import json
import time
import io

if os.environ.get("LLM_DEBUG_CUDA_LAUNCH_BLOCKING", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    if "CUDA_LAUNCH_BLOCKING" in os.environ:
        del os.environ["CUDA_LAUNCH_BLOCKING"]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import glob
import re
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
os.environ.setdefault("LLM_WSL_EXTENSION_NAME", "llm_wsl_extension")
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
try:
    _ext = __import__(os.environ.get("LLM_WSL_EXTENSION_NAME", "llm_wsl_extension"))
    rms_norm = getattr(_ext, "rms_norm", None)
    rope = getattr(_ext, "rope", None)
    ThreadManager = getattr(_ext, "ThreadManager", None)
    from custom_ops import patch_model, set_active_extension

    set_active_extension(_ext)

    print("成功加载自定义 WSL 扩展 (CUDA/C++ Thread)!")
    USE_CUSTOM_EXTENSION = True
except ImportError:
    print("未检测到自定义 WSL 扩展，将使用原生 PyTorch 实现。")
    USE_CUSTOM_EXTENSION = False
    _ext = None

_THREAD_MANAGER = None


def _configure_cpu_threads(cfg: dict):
    cpu_threads = int(cfg.get("cpu_threads") or (os.cpu_count() or 1))
    interop_threads = int(cfg.get("cpu_interop_threads") or 1)
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(k, str(cpu_threads))
    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass
    return cpu_threads


def _init_thread_manager(cfg: dict, ext_module, cpu_threads: int):
    global _THREAD_MANAGER
    if not cfg.get("enable_thread_manager", False):
        return None
    if ext_module is None or not hasattr(ext_module, "ThreadManager"):
        return None
    workers = int(cfg.get("thread_manager_workers") or max(1, cpu_threads // 2))
    try:
        _THREAD_MANAGER = ext_module.ThreadManager(workers)
        return _THREAD_MANAGER
    except Exception:
        _THREAD_MANAGER = None
        return None


TRAIN_CONFIG = {
    "train_data_path": os.path.join(CURRENT_DIR, "data", "layer1", "train.bin"),
    "val_data_path": os.path.join(CURRENT_DIR, "data", "layer1", "val.bin"),
    "batch_size": 2,
    "gradient_accumulation_steps": 32,
    "mixed_precision": True,
    "precision_dtype": "bf16",
    "num_workers": 4,
    "epoch_shards": 10,
    "shuffle_buffer_size": 8192,
    "seed": 1234,
    "cpu_threads": max(1, min((os.cpu_count() or 1), 8)),
    "cpu_interop_threads": 1,
    "enable_thread_manager": True,
    "thread_manager_workers": max(1, min(((os.cpu_count() or 1) // 2), 8)),
    "enable_custom_extension": True,
    "enable_torch_compile": True,
    "torch_compile_backend": "inductor",
    "torch_compile_mode": "default",
    "torch_compile_fullgraph": False,
    "torch_compile_dynamic": False,
    "use_deepspeed": True,
    "deepspeed_zero_stage": 2,
    "deepspeed_offload_optimizer": True,
    "deepspeed_offload_param": False,
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


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    else:
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
            tensors[k] = move_to_cpu(v).contiguous()
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
    torch.save(move_to_cpu(meta_state), buf)
    meta_u8 = _bytes_to_u8_tensor(buf.getvalue())

    tensors = {"__checkpoint_state__": meta_u8}
    for k, v in weights.items():
        if isinstance(k, str) and isinstance(v, torch.Tensor):
            tensors[k] = move_to_cpu(v).contiguous()

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
    cpu_threads = _configure_cpu_threads(TRAIN_CONFIG)
    _init_thread_manager(TRAIN_CONFIG, _ext, cpu_threads)

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
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    use_deepspeed = (
        bool(TRAIN_CONFIG.get("use_deepspeed", False)) and torch.cuda.is_available()
    )
    use_amp = (
        (not use_deepspeed)
        and TRAIN_CONFIG["mixed_precision"]
        and torch.cuda.is_available()
    )
    amp_dtype = torch.float16
    if use_amp and TRAIN_CONFIG["precision_dtype"] == "bf16":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("已启用 BFloat16 混合精度训练 (BF16)")
        else:
            print("警告: 当前设备不支持 BF16，回退到 FP16")
            amp_dtype = torch.float16
    if use_deepspeed:
        dtype_str = (
            "BF16"
            if TRAIN_CONFIG.get("precision_dtype") == "bf16"
            and torch.cuda.is_bf16_supported()
            else "FP16"
        )
        print(f"混合精度训练 (DeepSpeed {dtype_str}): 开启")
    else:
        print(f"混合精度训练 (AMP): {'开启 (FP16)' if use_amp else '关闭'}")
    if use_deepspeed:
        print("已启用 DeepSpeed 训练引擎")
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
    if not os.path.exists(train_data_path):
        print(f"错误: 找不到训练数据 {train_data_path}")
        print("请先运行 chinese_pretrain_data.py 生成数据。")
        sys.exit(1)
    is_binary_train = isinstance(train_data_path, str) and train_data_path.endswith(
        ".bin"
    )
    train_meta = None
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
                num_workers=0,
                pin_memory=True,
            )

        train_loader = build_train_loader_for_epoch(0)
    else:
        train_loader = create_dataloader_v1(
            train_data_path,
            batch_size=TRAIN_CONFIG["batch_size"],
            max_length=TRAIN_CONFIG["max_seq_len"],
            shuffle=True,
            drop_last=True,
            num_workers=TRAIN_CONFIG["num_workers"],
        )
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
    model_size = str(TRAIN_CONFIG.get("model_size", "150m")).lower()
    if model_size in ("150m", "0.15b", "150"):
        model_config = LLAMA_LITE_150M_CONFIG.copy()
    elif model_size in ("450m", "0.45b", "450"):
        model_config = LLAMA_LITE_450M_CONFIG.copy()
    elif model_size in ("1.5b", "1_5b", "1.5", "1500m", "1500"):
        model_config = LLAMA_LITE_1_5B_CONFIG.copy()
    else:
        print(f"警告: 未知 model_size={model_size}，将回退到 150m")
        model_config = LLAMA_LITE_150M_CONFIG.copy()
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
            check_loader = train_loader
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
    model = LlamaLite(model_config).to(device)

    def _extract_state_dict(obj):
        if isinstance(obj, dict) and "model_state_dict" in obj:
            return obj["model_state_dict"]
        if isinstance(obj, dict):
            has_any_tensor = any(isinstance(v, torch.Tensor) for v in obj.values())
            if has_any_tensor:
                return obj
        return None

    def _load_weights_from_path(path: str):
        loaded = None
        if isinstance(path, str) and path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except Exception as e:
                raise RuntimeError(
                    "未安装 safetensors，无法加载 .safetensors 权重文件。请先安装: pip install safetensors"
                ) from e
            loaded = load_file(path, device="cpu")
            sd = _extract_state_dict(loaded)
            if sd is None:
                raise ValueError(f"无法从权重文件解析 state_dict: {path}")
            return loaded, sd
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        sd = _extract_state_dict(loaded)
        if sd is None:
            raise ValueError(f"无法从权重文件解析 state_dict: {path}")
        return loaded, sd

    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)
    use_custom_extension = USE_CUSTOM_EXTENSION and bool(
        TRAIN_CONFIG.get("enable_custom_extension", False)
    )
    if use_custom_extension:
        model = patch_model(model, layer=1, ext=_ext)
    if (
        (not use_deepspeed)
        and bool(TRAIN_CONFIG.get("enable_torch_compile", False))
        and hasattr(torch, "compile")
    ):
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
        except Exception:
            pass
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = False
        except Exception:
            pass
        print("正在尝试启用 torch.compile...")
        try:
            compile_mode = TRAIN_CONFIG.get("torch_compile_mode", "default")
            if compile_mode == "default":
                compile_mode = None
            model = torch.compile(
                model,
                backend=str(TRAIN_CONFIG.get("torch_compile_backend", "inductor")),
                mode=compile_mode,
                fullgraph=bool(TRAIN_CONFIG.get("torch_compile_fullgraph", False)),
                dynamic=bool(TRAIN_CONFIG.get("torch_compile_dynamic", False)),
            )
            print("已启用 torch.compile")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"torch.compile 启用失败，将回退到 eager: {e}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if use_deepspeed and bool(TRAIN_CONFIG.get("deepspeed_offload_optimizer", True)):
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(
                optim_params,
                lr=TRAIN_CONFIG["learning_rate"],
                betas=(TRAIN_CONFIG["beta1"], TRAIN_CONFIG["beta2"]),
                weight_decay=TRAIN_CONFIG["weight_decay"],
            )
            print("已启用 DeepSpeedCPUAdam 优化器 (适配 ZeRO-Offload)")
        except ImportError:
            print(
                "警告: 无法导入 DeepSpeedCPUAdam，将使用原生 AdamW (可能导致 ZeRO-Offload 报错)"
            )
        except Exception as e:
            print(f"DeepSpeedCPUAdam 初始化失败: {e}")
    if optimizer is None:
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
    checkpoint = resume_checkpoint
    latest_checkpoint_path = resume_latest_checkpoint_path
    latest_epoch = resume_latest_epoch
    if latest_checkpoint_path and checkpoint:
        print(f"===== 发现检查点: {latest_checkpoint_path} =====")
        print(f"正在恢复训练状态 (Epoch {latest_epoch})...")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint["global_step"]
            best_perplexity = checkpoint.get("best_perplexity", float("inf"))
            if not use_deepspeed:
                opt_state = checkpoint.get("optimizer_state_dict", None)
                if isinstance(opt_state, dict) and "optimizer_state_dict" in opt_state:
                    opt_state = opt_state.get("optimizer_state_dict")
                if opt_state is not None:
                    try:
                        optimizer.load_state_dict(opt_state)
                    except Exception as e:
                        print(f"优化器状态恢复失败，将重置优化器状态: {e}")
            print(
                f"成功恢复至 Epoch {start_epoch}, Step {global_step}, Best PPL {best_perplexity:.2f}"
            )
        else:
            if checkpoint:
                print("检测到旧版本检查点格式，仅恢复模型权重，重置优化器状态。")
                if isinstance(checkpoint, dict):
                    legacy_sd = _extract_state_dict(checkpoint)
                    if isinstance(legacy_sd, dict):
                        model.load_state_dict(legacy_sd, strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                start_epoch = latest_epoch
                if is_binary_train:
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
    if (
        (not use_deepspeed)
        and latest_checkpoint_path
        and "scheduler_state_dict" in checkpoint
    ):
        sched_state = checkpoint.get("scheduler_state_dict", None)
        if isinstance(sched_state, dict) and "scheduler_state_dict" in sched_state:
            sched_state = sched_state.get("scheduler_state_dict")
        if sched_state is not None:
            try:
                lr_scheduler.load_state_dict(sched_state)
            except Exception as e:
                print(f"学习率调度器状态恢复失败，将重置调度器状态: {e}")
    deepspeed_enabled = False
    if use_deepspeed:
        try:
            import deepspeed

            zero_cfg = {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "ignore_unused_parameters": True,
            }
            ds_config = {
                "train_micro_batch_size_per_gpu": int(TRAIN_CONFIG["batch_size"]),
                "gradient_accumulation_steps": int(
                    TRAIN_CONFIG["gradient_accumulation_steps"]
                ),
                "gradient_clipping": float(TRAIN_CONFIG["grad_clip"]),
                "zero_optimization": zero_cfg,
                "steps_per_print": int(TRAIN_CONFIG.get("log_interval", 10)),
                "wall_clock_breakdown": False,
                "zero_force_ds_cpu_optimizer": False,
                "torch_ddp": {"find_unused_parameters": False},
            }
            if TRAIN_CONFIG["precision_dtype"] == "bf16":
                ds_config["bf16"] = {"enabled": True}
            else:
                ds_config["fp16"] = {"enabled": True, "loss_scale": 0}
            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                model_parameters=optim_params,
                config=ds_config,
            )
            deepspeed_enabled = True
        except Exception as e:
            print(f"DeepSpeed 初始化失败，将回退到原生训练: {e}")
            deepspeed_enabled = False
            if hasattr(model, "_forward_pre_hooks"):
                model._forward_pre_hooks.clear()
            if hasattr(model, "_forward_hooks"):
                model._forward_hooks.clear()
            if hasattr(model, "_backward_hooks"):
                model._backward_hooks.clear()
            print("已清理残留的 DeepSpeed hooks。")
    scaler = None
    scaler_enabled = False
    if not deepspeed_enabled:
        scaler_enabled = use_amp and (amp_dtype == torch.float16)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        if (
            latest_checkpoint_path
            and "scaler_state_dict" in checkpoint
            and scaler_enabled
        ):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    model.train()
    if start_epoch >= TRAIN_CONFIG["epochs"]:
        print("所有 Epoch 已完成，无需训练。")
        return
    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"]):
        print(f"\n===== Epoch {epoch + 1}/{TRAIN_CONFIG['epochs']} =====")
        epoch_loss = 0.0
        if deepspeed_enabled:
            model.zero_grad()
        else:
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
                if not deepspeed_enabled:
                    loss = loss / TRAIN_CONFIG["gradient_accumulation_steps"]
            current_loss = (
                loss.item() * TRAIN_CONFIG["gradient_accumulation_steps"]
                if not deepspeed_enabled
                else loss.item()
            )
            epoch_loss += current_loss
            if deepspeed_enabled:
                is_boundary = bool(model.is_gradient_accumulation_boundary())
                model.backward(loss)
                model.step()
                if is_boundary:
                    global_step += 1
                    train_window_loss_sum += float(current_loss)
                    train_window_loss_count += 1
                    current_lr = optimizer.param_groups[0]["lr"]
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
                            model, val_loader, device, False, amp_dtype
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
                            raw_model = (
                                model.module if hasattr(model, "module") else model
                            )
                            _save_state_dict_safetensors(raw_model.state_dict(), best_path)
                            tqdm.write(
                                f"✅ 新的最优模型已保存 (PPL: {best_perplexity:.2f})"
                            )
            else:
                scaler.scale(loss).backward()
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
                            try:
                                raw_model = (
                                    model.module if hasattr(model, "module") else model
                                )
                                _save_state_dict_safetensors(
                                    raw_model.state_dict(), best_path
                                )
                                tqdm.write(
                                    f"✅ 新的最优模型已保存 (PPL: {best_perplexity:.2f})"
                                )
                            except Exception as e:
                                tqdm.write(f"⚠️ 保存最优模型失败: {e}")
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
                float(optimizer.param_groups[0]["lr"])
                if deepspeed_enabled
                else float(lr_scheduler.get_last_lr()[0])
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
        raw_model = model.module if hasattr(model, "module") else model
        checkpoint_state = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": move_to_cpu(raw_model.state_dict()),
            "optimizer_state_dict": (
                move_to_cpu(optimizer.state_dict()) if not deepspeed_enabled else None
            ),
            "scheduler_state_dict": (
                move_to_cpu(lr_scheduler.state_dict())
                if not deepspeed_enabled
                else None
            ),
            "scaler_state_dict": (
                move_to_cpu(scaler.state_dict())
                if (scaler is not None and scaler_enabled)
                else None
            ),
            "best_perplexity": best_perplexity,
            "config": TRAIN_CONFIG,
        }
        epoch_path = os.path.join(
            TRAIN_CONFIG["weight_save_dir"],
            f"model_epoch_{epoch + 1}.safetensors",
        )
        try:
            _save_checkpoint_safetensors(checkpoint_state, epoch_path)
            print(f"已保存 Epoch 检查点(safetensors): {epoch_path}")
        except Exception as e:
            print(f"保存检查点失败 (Epoch {epoch+1}): {e}")
            try:
                _save_state_dict_safetensors(checkpoint_state["model_state_dict"], epoch_path)
                print(f"已降级保存仅模型权重(safetensors): {epoch_path}")
            except Exception as e2:
                print(f"降级保存也失败: {e2}")

    print(f"\n===== 训练结束 =====")
    print(f"最优验证集 PPL: {best_perplexity:.2f}")
    final_path = os.path.join(
        TRAIN_CONFIG["weight_save_dir"],
        "final_model.safetensors",
    )
    raw_model = model.module if hasattr(model, "module") else model
    try:
        _save_state_dict_safetensors(raw_model.state_dict(), final_path)
        print(f"最终模型已保存: {final_path}")
    except Exception as e:
        print(f"最终模型保存失败: {e}")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
