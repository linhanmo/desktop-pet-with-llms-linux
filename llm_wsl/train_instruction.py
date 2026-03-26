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
os.environ.setdefault("LLM_WSL_EXTENSION_NAME", "llm_wsl_extension_instruction")

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
    print("请确保在 llm_wsl 目录下运行脚本，或检查目录结构。")
    sys.exit(1)

try:
    _ext = __import__(
        os.environ.get("LLM_WSL_EXTENSION_NAME", "llm_wsl_extension_instruction")
    )
    rms_norm = getattr(_ext, "rms_norm", None)
    rope = getattr(_ext, "rope", None)
    ThreadManager = getattr(_ext, "ThreadManager", None)
    xentropy_forward = getattr(_ext, "xentropy_forward", None)
    xentropy_backward = getattr(_ext, "xentropy_backward", None)
    from custom_ops import patch_model, set_active_extension

    set_active_extension(_ext)

    print("成功加载自定义 WSL 扩展 (CUDA/C++ Thread)!")
    USE_CUSTOM_EXTENSION = True
except ImportError:
    print("未检测到自定义 WSL 扩展，将使用原生 PyTorch 实现。")
    USE_CUSTOM_EXTENSION = False
    _ext = None

TRAIN_CONFIG = {
    "train_data_path": os.path.join(CURRENT_DIR, "data", "layer2", "train.bin"),
    "val_data_path": os.path.join(CURRENT_DIR, "data", "layer2", "val.bin"),
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
    "epochs": 20,
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_steps": 200,
    "min_lr_ratio": 0.1,
    "log_interval": 10,
    "eval_interval": 500,
    "weight_save_dir": os.path.join(CURRENT_DIR, "checkpoints", "layer2"),
    "layer": 2,
    "model_size": "150m",
    "base_model_path": None,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_name": "default",
    "save_lora_only": True,
    "label_smoothing": 0.0,
    "z_loss": 0.0,
    "ignore_index": -100,
}

_THREAD_MANAGER = None


USE_FUSED_XENTROPY = callable(globals().get("xentropy_forward")) and callable(
    globals().get("xentropy_backward")
)


class _FusedXEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing, z_loss, ignore_index):
        out = xentropy_forward(
            logits, labels, float(smoothing), float(z_loss), int(ignore_index)
        )
        losses, logsumexp = out[0], out[1]
        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.smoothing = float(smoothing)
        ctx.z_loss = float(z_loss)
        ctx.ignore_index = int(ignore_index)
        return losses

    @staticmethod
    def backward(ctx, grad_losses):
        logits, logsumexp, labels = ctx.saved_tensors
        grad_logits = xentropy_backward(
            grad_losses,
            logits,
            logsumexp,
            labels,
            float(ctx.smoothing),
            float(ctx.z_loss),
            int(ctx.ignore_index),
        )
        return grad_logits, None, None, None, None


def _cross_entropy(logits, labels, reduction: str = "mean"):
    if USE_FUSED_XENTROPY and logits.is_cuda:
        losses = _FusedXEntropy.apply(
            logits,
            labels,
            float(TRAIN_CONFIG.get("label_smoothing", 0.0)),
            float(TRAIN_CONFIG.get("z_loss", 0.0)),
            int(TRAIN_CONFIG.get("ignore_index", -100)),
        )
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        return losses.mean()
    return F.cross_entropy(logits, labels, reduction=reduction)


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
    if isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
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
    kind = checkpoint_state.get("checkpoint_kind")
    weights_key = "lora_state_dict" if kind == "lora" else "model_state_dict"
    weights = checkpoint_state.get(weights_key) or {}
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{weights_key} 为空，无法保存检查点")

    meta_state = dict(checkpoint_state)
    meta_state.pop(weights_key, None)

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
    kind = meta_state.get("checkpoint_kind")
    weights_key = "lora_state_dict" if kind == "lora" else "model_state_dict"
    meta_state[weights_key] = {k: v for k, v in tensors.items() if isinstance(k, str)}
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
    for x, y in dataloader:
        x = x.to(device=device, dtype=torch.long, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype)
        with autocast_ctx:
            logits = model(x)
            loss = _cross_entropy(
                logits.flatten(0, 1), y.flatten(0, 1), reduction="sum"
            )
        total_loss += loss.item()
        total_tokens += y.numel()
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
        return None, 0, None
    candidates = []
    patterns = [
        ("model_epoch_*.safetensors", r"model_epoch_(\d+)\.safetensors", "full"),
        ("lora_epoch_*.safetensors", r"lora_epoch_(\d+)\.safetensors", "lora"),
    ]
    for glob_pat, rx, kind in patterns:
        for cp in glob.glob(os.path.join(checkpoint_dir, glob_pat)):
            match = re.search(rx, cp)
            if match:
                candidates.append((int(match.group(1)), kind, cp))
    if not candidates:
        return None, 0, None
    epoch, kind, path = sorted(candidates)[-1]
    return path, epoch, kind


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
    parser.add_argument(
        "--base_model_path", type=str, default=os.environ.get("LLM_BASE_MODEL_PATH")
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=int(
            os.environ.get("LLM_LORA_RANK", str(TRAIN_CONFIG.get("lora_rank", 8)))
        ),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=int(
            os.environ.get("LLM_LORA_ALPHA", str(TRAIN_CONFIG.get("lora_alpha", 16)))
        ),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=float(
            os.environ.get(
                "LLM_LORA_DROPOUT", str(TRAIN_CONFIG.get("lora_dropout", 0.0))
            )
        ),
    )
    parser.add_argument(
        "--lora_name",
        type=str,
        default=os.environ.get(
            "LLM_LORA_NAME", str(TRAIN_CONFIG.get("lora_name", "default"))
        ),
    )
    parser.add_argument("--save_lora_only", action="store_true", default=False)
    parser.add_argument("--full_state_dict", action="store_true", default=False)
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=float(
            os.environ.get(
                "LLM_LABEL_SMOOTHING", str(TRAIN_CONFIG.get("label_smoothing", 0.0))
            )
        ),
    )
    parser.add_argument(
        "--z_loss",
        type=float,
        default=float(
            os.environ.get("LLM_Z_LOSS", str(TRAIN_CONFIG.get("z_loss", 0.0)))
        ),
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=int(
            os.environ.get(
                "LLM_IGNORE_INDEX", str(TRAIN_CONFIG.get("ignore_index", -100))
            )
        ),
    )
    args = parser.parse_args(argv)

    TRAIN_CONFIG["layer"] = 2
    TRAIN_CONFIG["model_size"] = str(args.model_size).lower()
    TRAIN_CONFIG["label_smoothing"] = float(args.label_smoothing)
    TRAIN_CONFIG["z_loss"] = float(args.z_loss)
    TRAIN_CONFIG["ignore_index"] = int(args.ignore_index)
    if args.train_data_path:
        TRAIN_CONFIG["train_data_path"] = args.train_data_path
    if args.val_data_path:
        TRAIN_CONFIG["val_data_path"] = args.val_data_path
    if args.weight_save_dir:
        TRAIN_CONFIG["weight_save_dir"] = args.weight_save_dir
    TRAIN_CONFIG["base_model_path"] = args.base_model_path
    TRAIN_CONFIG["lora_rank"] = int(args.lora_rank)
    TRAIN_CONFIG["lora_alpha"] = int(args.lora_alpha)
    TRAIN_CONFIG["lora_dropout"] = float(args.lora_dropout)
    TRAIN_CONFIG["lora_name"] = str(args.lora_name)
    TRAIN_CONFIG["save_lora_only"] = bool(
        args.save_lora_only or (not args.full_state_dict)
    )
    cpu_threads = _configure_cpu_threads(TRAIN_CONFIG)
    _init_thread_manager(TRAIN_CONFIG, _ext, cpu_threads)

    if not TRAIN_CONFIG.get("base_model_path"):
        layer1_ckpt_dir = os.path.join(CURRENT_DIR, "checkpoints")
        for fn in ("best_model.safetensors", "final_model.safetensors"):
            p = os.path.join(layer1_ckpt_dir, fn)
            if os.path.exists(p):
                TRAIN_CONFIG["base_model_path"] = p
                break
        if not TRAIN_CONFIG.get("base_model_path"):
            latest_p, _, kind = find_latest_checkpoint(layer1_ckpt_dir)
            if latest_p and kind == "full":
                TRAIN_CONFIG["base_model_path"] = latest_p

    os.makedirs(TRAIN_CONFIG["weight_save_dir"], exist_ok=True)
    print("===== LlamaLite Instruction 微调脚本 (WSL Layer2) =====")
    print(f"权重保存目录: {TRAIN_CONFIG['weight_save_dir']}")
    metrics_path = os.path.join(
        TRAIN_CONFIG["weight_save_dir"], "training_metrics.json"
    )
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
            return (
                "high",
                (
                    f"val_loss 连续走高={consecutive_worsen}, gap={gap:.4f}"
                    if gap is not None
                    else f"val_loss 连续走高={consecutive_worsen}"
                ),
            )
        if consecutive_worsen == 2 or (gap is not None and gap > 0.15):
            return (
                "medium",
                (
                    f"val_loss 连续走高={consecutive_worsen}, gap={gap:.4f}"
                    if gap is not None
                    else f"val_loss 连续走高={consecutive_worsen}"
                ),
            )
        if consecutive_worsen == 1 or (gap is not None and gap > 0.08):
            return (
                "low",
                (
                    f"val_loss 走高迹象, gap={gap:.4f}"
                    if gap is not None
                    else "val_loss 走高迹象"
                ),
            )
        return "none", ""

    desired_epochs = int(TRAIN_CONFIG.get("epochs", 0) or 0)
    resume_checkpoint = {}
    resume_latest_checkpoint_path, resume_latest_epoch, resume_latest_kind = (
        find_latest_checkpoint(TRAIN_CONFIG["weight_save_dir"])
    )
    if resume_latest_checkpoint_path:
        try:
            resume_checkpoint = _load_checkpoint_safetensors(
                resume_latest_checkpoint_path
            )
        except Exception as e:
            print(f"检查点加载失败，将忽略检查点并从头开始训练: {e}")
            resume_latest_checkpoint_path = None
            resume_latest_epoch = 0
            resume_latest_kind = None
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
                "layer",
                "model_size",
                "base_model_path",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_name",
                "save_lora_only",
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
            TRAIN_CONFIG["layer"] = 2
            print("已从检查点恢复训练配置。")
        if (
            isinstance(resume_checkpoint, dict)
            and "checkpoint_kind" not in resume_checkpoint
        ):
            is_pure_state_dict = bool(resume_checkpoint) and all(
                isinstance(v, torch.Tensor) for v in resume_checkpoint.values()
            )
            if not is_pure_state_dict:
                resume_checkpoint["checkpoint_kind"] = resume_latest_kind

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"使用设备: GPU ({gpu_name}, {vram:.1f} GB VRAM)")
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
        sys.exit(1)

    is_binary_train = isinstance(train_data_path, str) and train_data_path.endswith(
        ".bin"
    )
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

    model_size = str(TRAIN_CONFIG.get("model_size", "150m")).lower()
    if model_size in ("150m", "0.15b", "150"):
        model_config = LLAMA_LITE_150M_CONFIG.copy()
    elif model_size in ("450m", "0.45b", "450"):
        model_config = LLAMA_LITE_450M_CONFIG.copy()
    elif model_size in ("1.5b", "1_5b", "1.5", "1500m", "1500"):
        model_config = LLAMA_LITE_1_5B_CONFIG.copy()
    else:
        model_config = LLAMA_LITE_150M_CONFIG.copy()

    if model_config["vocab_size"] != tokenizer.vocab_size:
        model_config["vocab_size"] = tokenizer.vocab_size

    def _load_state_dict_like(obj):
        if not isinstance(obj, dict):
            return None
        if "model_state_dict" in obj and isinstance(obj.get("model_state_dict"), dict):
            return obj["model_state_dict"]
        if "lora_state_dict" in obj and isinstance(obj.get("lora_state_dict"), dict):
            return obj["lora_state_dict"]
        if "state_dict" in obj and isinstance(obj.get("state_dict"), dict):
            return obj["state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
        return None

    def _load_weights_from_path(path: str):
        if not path or not isinstance(path, str) or not os.path.exists(path):
            return None, None
        if path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except Exception as e:
                raise RuntimeError(
                    "未安装 safetensors，无法加载 .safetensors 权重文件。请先安装: pip install safetensors"
                ) from e
            obj = load_file(path, device="cpu")
            if isinstance(obj, dict) and "__checkpoint_state__" in obj:
                try:
                    obj = dict(obj)
                    obj.pop("__checkpoint_state__", None)
                except Exception:
                    pass
            sd = _load_state_dict_like(obj)
            return None, sd
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        sd = _load_state_dict_like(obj)
        kind = None
        if isinstance(obj, dict):
            kind = obj.get("checkpoint_kind")
        return kind, sd

    model = LlamaLite(model_config).to(device)

    checkpoint_kind = None
    if isinstance(resume_checkpoint, dict):
        checkpoint_kind = resume_checkpoint.get("checkpoint_kind")
    if not checkpoint_kind:
        checkpoint_kind = resume_latest_kind

    effective_base_model_path = TRAIN_CONFIG.get("base_model_path")
    if checkpoint_kind == "lora" and isinstance(resume_checkpoint, dict):
        effective_base_model_path = (
            resume_checkpoint.get("base_model_path") or effective_base_model_path
        )

    if effective_base_model_path:
        try:
            _, base_sd = _load_weights_from_path(str(effective_base_model_path))
            if isinstance(base_sd, dict) and base_sd:
                model.load_state_dict(base_sd, strict=False)
                print(f"已加载 base_model_path: {effective_base_model_path}")
        except Exception as e:
            print(f"base 权重加载失败: {e}")

    lora_cfg = None
    if checkpoint_kind == "lora" and isinstance(resume_checkpoint, dict):
        lora_cfg = resume_checkpoint.get("lora")
    if isinstance(lora_cfg, dict) and isinstance(lora_cfg.get("adapters"), dict):
        adapters = lora_cfg.get("adapters") or {}
        active = lora_cfg.get("active_adapters")
        if isinstance(active, list) and active:
            name = str(active[0])
        else:
            name = str(
                next(iter(adapters.keys()), TRAIN_CONFIG.get("lora_name", "default"))
            )
        c = adapters.get(name) if isinstance(adapters, dict) else None
        if isinstance(c, dict):
            lora_cfg = {
                "rank": int(c.get("rank", TRAIN_CONFIG.get("lora_rank", 8))),
                "alpha": int(c.get("alpha", TRAIN_CONFIG.get("lora_alpha", 16))),
                "dropout": float(
                    c.get("dropout", TRAIN_CONFIG.get("lora_dropout", 0.0))
                ),
                "name": name,
            }
    if not isinstance(lora_cfg, dict):
        lora_cfg = {
            "rank": int(TRAIN_CONFIG.get("lora_rank", 8)),
            "alpha": int(TRAIN_CONFIG.get("lora_alpha", 16)),
            "dropout": float(TRAIN_CONFIG.get("lora_dropout", 0.0)),
            "name": str(TRAIN_CONFIG.get("lora_name", "default")),
        }

    try:
        model.enable_lora(
            rank=int(lora_cfg.get("rank", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            name=str(lora_cfg.get("name", "default")),
        )
        model.freeze_base_model()
    except Exception as e:
        print(f"LoRA 初始化失败: {e}")

    def _get_lora_state_dict(m):
        if hasattr(m, "get_lora_state_dict"):
            try:
                return m.get_lora_state_dict()
            except Exception:
                pass
        return {k: v for k, v in m.state_dict().items() if ".lora_adapters." in k}

    def _get_lora_config(m):
        if hasattr(m, "get_lora_config"):
            try:
                return m.get_lora_config()
            except Exception:
                pass
        return {
            "rank": int(TRAIN_CONFIG.get("lora_rank", 8)),
            "alpha": int(TRAIN_CONFIG.get("lora_alpha", 16)),
            "dropout": float(TRAIN_CONFIG.get("lora_dropout", 0.0)),
            "name": str(TRAIN_CONFIG.get("lora_name", "default")),
        }

    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)

    use_custom_extension = USE_CUSTOM_EXTENSION and bool(
        TRAIN_CONFIG.get("enable_custom_extension", False)
    )
    if use_custom_extension:
        model = patch_model(model, layer=2, ext=_ext)

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
            print(f"torch.compile 启用失败，将回退到 eager: {e}")

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
        except Exception as e:
            print(f"DeepSpeedCPUAdam 初始化失败: {e}")
            optimizer = None
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

    start_epoch = 0
    global_step = 0
    best_perplexity = float("inf")
    checkpoint = resume_checkpoint
    latest_checkpoint_path = resume_latest_checkpoint_path
    latest_epoch = resume_latest_epoch

    if latest_checkpoint_path and checkpoint:
        ckpt_kind = None
        if isinstance(checkpoint, dict):
            ckpt_kind = checkpoint.get("checkpoint_kind")
        if not ckpt_kind:
            ckpt_kind = resume_latest_kind
        if ckpt_kind == "lora" and isinstance(checkpoint, dict):
            raw_model = model.module if hasattr(model, "module") else model
            lora_sd = checkpoint.get("lora_state_dict")
            if not isinstance(lora_sd, dict):
                lora_sd = {
                    k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)
                }
            if isinstance(lora_sd, dict):
                raw_model.load_state_dict(lora_sd, strict=False)
            start_epoch = int(checkpoint.get("epoch", latest_epoch) or 0)
            global_step = int(checkpoint.get("global_step", 0) or 0)
            best_perplexity = float(checkpoint.get("best_perplexity", float("inf")))
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            start_epoch = int(checkpoint.get("epoch", latest_epoch) or 0)
            global_step = int(checkpoint.get("global_step", 0) or 0)
            best_perplexity = float(checkpoint.get("best_perplexity", float("inf")))

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

    if (not use_deepspeed) and latest_checkpoint_path and isinstance(checkpoint, dict):
        sched_state = checkpoint.get("scheduler_state_dict", None)
        if isinstance(sched_state, dict) and "scheduler_state_dict" in sched_state:
            sched_state = sched_state.get("scheduler_state_dict")
        if isinstance(sched_state, dict):
            try:
                lr_scheduler.load_state_dict(sched_state)
            except Exception:
                pass

    deepspeed_enabled = False
    if use_deepspeed:
        try:
            import deepspeed

            zero_cfg = {
                "stage": int(TRAIN_CONFIG.get("deepspeed_zero_stage", 2)),
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
            print("已启用 DeepSpeed 训练引擎")
        except Exception as e:
            print(f"DeepSpeed 初始化失败，将回退到原生训练: {e}")
            deepspeed_enabled = False

    scaler = None
    scaler_enabled = False
    if not deepspeed_enabled:
        scaler_enabled = use_amp and (amp_dtype == torch.float16)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        if latest_checkpoint_path and isinstance(checkpoint, dict) and scaler_enabled:
            if "scaler_state_dict" in checkpoint:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception:
                    pass

    model.train()
    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"]):
        epoch_loss = 0.0
        if deepspeed_enabled:
            model.zero_grad()
        else:
            optimizer.zero_grad()

        if is_binary_train:
            epoch_train_loader = build_train_loader_for_epoch(epoch)
        else:
            epoch_train_loader = train_loader

        progress_bar = tqdm(epoch_train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device=device, dtype=torch.long, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                )
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
            with autocast_ctx:
                logits = model(x)
                loss = _cross_entropy(
                    logits.flatten(0, 1), y.flatten(0, 1), reduction="mean"
                )
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

            if (deepspeed_enabled and model.is_gradient_accumulation_boundary()) or (
                (not deepspeed_enabled)
                and ((batch_idx + 1) % TRAIN_CONFIG["gradient_accumulation_steps"] == 0)
            ):
                train_window_loss_sum += float(current_loss)
                train_window_loss_count += 1
                current_lr = (
                    float(optimizer.param_groups[0]["lr"])
                    if deepspeed_enabled
                    else float(lr_scheduler.get_last_lr()[0])
                )
                progress_bar.set_postfix(
                    {
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{current_lr:.6f}",
                        "step": global_step,
                    }
                )

                if global_step % TRAIN_CONFIG["eval_interval"] == 0:
                    val_loss, val_ppl = calculate_perplexity(
                        model,
                        val_loader,
                        device,
                        False if deepspeed_enabled else use_amp,
                        amp_dtype,
                    )
                    train_loss_recent = (
                        (train_window_loss_sum / train_window_loss_count)
                        if train_window_loss_count > 0
                        else None
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
                    train_window_loss_sum = 0.0
                    train_window_loss_count = 0
                    if val_ppl < best_perplexity:
                        best_perplexity = val_ppl
                        best_path = os.path.join(
                            TRAIN_CONFIG["weight_save_dir"],
                            (
                                "best_lora.safetensors"
                                if TRAIN_CONFIG.get("save_lora_only", False)
                                else "best_model.safetensors"
                            ),
                        )
                        raw_model = model.module if hasattr(model, "module") else model
                        if TRAIN_CONFIG.get("save_lora_only", False):
                            _save_state_dict_safetensors(
                                _get_lora_state_dict(raw_model), best_path
                            )
                        else:
                            _save_state_dict_safetensors(raw_model.state_dict(), best_path)

        avg_epoch_loss = epoch_loss / max(
            1,
            (len(epoch_train_loader) if hasattr(epoch_train_loader, "__len__") else 1),
        )
        val_loss, val_ppl = calculate_perplexity(
            model,
            val_loader,
            device,
            False if deepspeed_enabled else use_amp,
            amp_dtype,
        )
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

        raw_model = model.module if hasattr(model, "module") else model
        if TRAIN_CONFIG.get("save_lora_only", False):
            checkpoint_state = {
                "epoch": int(epoch + 1),
                "global_step": int(global_step),
                "lora_state_dict": move_to_cpu(_get_lora_state_dict(raw_model)),
                "lora": _get_lora_config(raw_model),
                "base_model_path": effective_base_model_path,
                "checkpoint_kind": "lora",
                "optimizer_state_dict": (
                    move_to_cpu(optimizer.state_dict())
                    if not deepspeed_enabled
                    else None
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
                "best_perplexity": float(best_perplexity),
                "config": TRAIN_CONFIG,
            }
            epoch_path = os.path.join(
                TRAIN_CONFIG["weight_save_dir"], f"lora_epoch_{epoch + 1}.safetensors"
            )
        else:
            checkpoint_state = {
                "epoch": int(epoch + 1),
                "global_step": int(global_step),
                "model_state_dict": move_to_cpu(raw_model.state_dict()),
                "checkpoint_kind": "full",
                "optimizer_state_dict": (
                    move_to_cpu(optimizer.state_dict())
                    if not deepspeed_enabled
                    else None
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
                "best_perplexity": float(best_perplexity),
                "config": TRAIN_CONFIG,
            }
            epoch_path = os.path.join(
                TRAIN_CONFIG["weight_save_dir"], f"model_epoch_{epoch + 1}.safetensors"
            )
        _save_checkpoint_safetensors(checkpoint_state, epoch_path)

    final_path = os.path.join(
        TRAIN_CONFIG["weight_save_dir"],
        (
            "final_lora.safetensors"
            if TRAIN_CONFIG.get("save_lora_only", False)
            else "final_model.safetensors"
        ),
    )
    raw_model = model.module if hasattr(model, "module") else model
    if TRAIN_CONFIG.get("save_lora_only", False):
        _save_state_dict_safetensors(_get_lora_state_dict(raw_model), final_path)
    else:
        _save_state_dict_safetensors(raw_model.state_dict(), final_path)
    print(f"最终模型已保存: {final_path}")


if __name__ == "__main__":
    main()
