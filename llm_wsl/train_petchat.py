import os
import sys
import json
import time
import io
import math
import glob
import argparse
import random
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

if os.environ.get("LLM_DEBUG_CUDA_LAUNCH_BLOCKING", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    if "CUDA_LAUNCH_BLOCKING" in os.environ:
        del os.environ["CUDA_LAUNCH_BLOCKING"]

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

os.environ.setdefault("LLM_WSL_EXTENSION_NAME", "llm_wsl_extension_petchat")

try:
    from core.model import LlamaLite
    from core.config import (
        LLAMA_LITE_150M_CONFIG,
        LLAMA_LITE_450M_CONFIG,
        LLAMA_LITE_1_5B_CONFIG,
    )
    from chinese_tokenizer import ChineseTokenizer
    from custom_ops import patch_model, set_active_extension
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在 llm_wsl 目录下运行脚本，或检查目录结构。")
    raise


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


def _seed_all(seed: int):
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _try_import_ext():
    name = os.environ.get("LLM_WSL_EXTENSION_NAME", "").strip()
    if not name:
        name = "llm_wsl_extension_petchat"
    try:
        return __import__(name)
    except Exception:
        return None


def _build_wsl_extension():
    setup_path = os.path.join(CURRENT_DIR, "setup_petchat.py")
    if not os.path.exists(setup_path):
        return False
    cmd = [sys.executable, setup_path, "build_ext", "--inplace"]
    try:
        p = subprocess.run(cmd, cwd=CURRENT_DIR, check=False)
        return p.returncode == 0
    except Exception:
        return False


def _maybe_load_extension(cfg: dict):
    if not bool(int(cfg.get("enable_custom_extension", 1))):
        return None, None
    ext = _try_import_ext()
    if ext is not None:
        try:
            set_active_extension(ext)
        except Exception:
            pass
        return ext, getattr(ext, "ThreadManager", None)
    if bool(int(cfg.get("build_extension_if_missing", 0))):
        ok = _build_wsl_extension()
        if ok:
            ext = _try_import_ext()
            if ext is not None:
                try:
                    set_active_extension(ext)
                except Exception:
                    pass
                return ext, getattr(ext, "ThreadManager", None)
    return None, None


_THREAD_MANAGER = None


def _init_thread_manager(cfg: dict, ext_module, cpu_threads: int):
    global _THREAD_MANAGER
    if not bool(int(cfg.get("enable_thread_manager", 1))):
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


def find_latest_checkpoint(checkpoint_dir: str):
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None, 0, None
    latest_epoch = 0
    latest_path = None
    latest_kind = None
    for kind, glob_pat in (
        ("lora", "lora_epoch_*.safetensors"),
        ("full", "model_epoch_*.safetensors"),
    ):
        for cp in glob.glob(os.path.join(checkpoint_dir, glob_pat)):
            base = os.path.basename(cp)
            m = None
            try:
                import re

                m = re.search(r"_(\d+)\.safetensors$", base)
            except Exception:
                m = None
            if not m:
                continue
            ep = int(m.group(1))
            if ep > latest_epoch:
                latest_epoch = ep
                latest_path = cp
                latest_kind = kind
    return latest_path, latest_epoch, latest_kind


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
        return None
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception:
            return None
        try:
            return load_file(path, device="cpu")
        except Exception:
            return None
    try:
        sd_obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        sd_obj = torch.load(path, map_location="cpu")
    return _load_state_dict_like(sd_obj)


def _move_to_cpu(obj: Any):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_move_to_cpu(x) for x in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj


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
        if not isinstance(k, str) or not isinstance(v, torch.Tensor):
            continue
        t = v.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        tensors[k] = t.contiguous()
    if not tensors:
        raise ValueError("state_dict 中没有可保存的 Tensor")
    save_file(tensors, path)


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
    kind = meta_state.get("checkpoint_kind")
    weights_key = "lora_state_dict" if kind == "lora" else "model_state_dict"
    weights = {k: v for k, v in tensors.items() if isinstance(k, str)}
    meta_state[weights_key] = weights
    return meta_state


@dataclass
class ChatSample:
    input_ids: list[int]
    labels: list[int]


class PetChatJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: ChineseTokenizer,
        max_seq_len: int,
        seed: int = 1234,
        max_samples: Optional[int] = None,
        allow_missing_assistant: bool = False,
    ):
        self.jsonl_path = str(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.seed = int(seed)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.allow_missing_assistant = bool(allow_missing_assistant)
        self._items: list[ChatSample] = []
        self._load()

    def _normalize_messages(self, obj: dict) -> Optional[list[dict]]:
        msgs = obj.get("messages")
        if isinstance(msgs, list) and msgs:
            keep = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    continue
                keep.append({"role": role, "content": content})
            return keep if keep else None

        if isinstance(obj.get("instruction"), str) and isinstance(
            obj.get("output"), str
        ):
            return [
                {"role": "user", "content": obj["instruction"]},
                {"role": "assistant", "content": obj["output"]},
            ]

        if isinstance(obj.get("prompt"), str) and isinstance(
            obj.get("completion"), str
        ):
            return [
                {"role": "user", "content": obj["prompt"]},
                {"role": "assistant", "content": obj["completion"]},
            ]

        if isinstance(obj.get("question"), str) and isinstance(obj.get("answer"), str):
            return [
                {"role": "user", "content": obj["question"]},
                {"role": "assistant", "content": obj["answer"]},
            ]
        return None

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool):
        tok = getattr(self.tokenizer, "tokenizer", None)
        if tok is None or not hasattr(tok, "apply_chat_template"):
            parts = []
            for m in messages:
                parts.append(f"{m['role']}: {m['content']}")
            if add_generation_prompt:
                parts.append("assistant: ")
            return "\n".join(parts)
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _encode(self, text: str) -> list[int]:
        tok = getattr(self.tokenizer, "tokenizer", None)
        if tok is not None and hasattr(tok, "encode"):
            return tok.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(text)

    def _build_sample(self, messages: list[dict]) -> Optional[ChatSample]:
        if not messages:
            return None
        if messages[-1].get("role") != "assistant":
            if not self.allow_missing_assistant:
                return None
            full_msgs = messages
            prefix_msgs = messages
        else:
            full_msgs = messages
            prefix_msgs = messages[:-1]

        prefix_text = self._apply_chat_template(prefix_msgs, add_generation_prompt=True)
        full_text = self._apply_chat_template(full_msgs, add_generation_prompt=False)
        prefix_ids = self._encode(prefix_text)
        full_ids = self._encode(full_text)

        if not full_ids:
            return None

        if len(prefix_ids) > len(full_ids):
            prefix_ids = prefix_ids[: len(full_ids)]
        prompt_len = len(prefix_ids)

        eos_id = int(getattr(self.tokenizer, "eos_token_id", -1))
        if eos_id >= 0 and full_ids[-1] != eos_id:
            full_ids = full_ids + [eos_id]

        if len(full_ids) > self.max_seq_len:
            overflow = len(full_ids) - self.max_seq_len
            full_ids = full_ids[overflow:]
            prompt_len = max(0, prompt_len - overflow)

        labels = list(full_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return ChatSample(input_ids=list(full_ids), labels=labels)

    def _load(self):
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(self.jsonl_path)
        items = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                messages = self._normalize_messages(obj)
                if messages is None:
                    continue
                sample = self._build_sample(messages)
                if sample is None:
                    continue
                items.append(sample)
                if self.max_samples is not None and len(items) >= self.max_samples:
                    break
        rng = random.Random(self.seed)
        rng.shuffle(items)
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx: int):
        s = self._items[int(idx)]
        return {
            "input_ids": torch.tensor(s.input_ids, dtype=torch.long),
            "labels": torch.tensor(s.labels, dtype=torch.long),
        }


def make_collate_fn(pad_id: int):
    pad_id = int(pad_id)

    def collate(batch: list[dict]):
        max_len = max(int(x["input_ids"].shape[0]) for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ex in enumerate(batch):
            ids = ex["input_ids"]
            lab = ex["labels"]
            n = int(ids.shape[0])
            input_ids[i, :n] = ids
            labels[i, :n] = lab
            attention_mask[i, :n] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate


def _get_model_config(model_size: str, tokenizer: ChineseTokenizer):
    ms = str(model_size or "1.5b").lower()
    if ms in ("150m", "0.15b", "150"):
        model_config = LLAMA_LITE_150M_CONFIG.copy()
    elif ms in ("450m", "0.45b", "450"):
        model_config = LLAMA_LITE_450M_CONFIG.copy()
    elif ms in ("1.5b", "1_5b", "1.5", "1500m", "1500"):
        model_config = LLAMA_LITE_1_5B_CONFIG.copy()
    else:
        model_config = LLAMA_LITE_150M_CONFIG.copy()
    if int(model_config.get("vocab_size")) != int(tokenizer.vocab_size):
        model_config["vocab_size"] = int(tokenizer.vocab_size)
    return model_config


def get_lr_scheduler(
    optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float
):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int):
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp: bool, amp_dtype: torch.dtype):
    model.eval()
    losses = []
    for batch in loader:
        x = batch["input_ids"].to(device=device, dtype=torch.long, non_blocking=True)
        labels = batch["labels"].to(device=device, dtype=torch.long, non_blocking=True)
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
        with autocast_ctx:
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().flatten(0, 1),
                labels[:, 1:].contiguous().flatten(0, 1),
                ignore_index=-100,
            )
        losses.append(float(loss.item()))
    model.train()
    if not losses:
        return None, None
    mean_loss = float(sum(losses) / len(losses))
    return mean_loss, float(math.exp(min(20.0, mean_loss)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=os.environ.get("LLM_PETCHAT_TRAIN")
        or os.path.join(CURRENT_DIR, "data", "petchat", "train.jsonl"),
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=os.environ.get("LLM_PETCHAT_VAL")
        or os.path.join(CURRENT_DIR, "data", "petchat", "val.jsonl"),
    )
    parser.add_argument(
        "--base_model_path", type=str, default=os.environ.get("LLM_BASE_MODEL_PATH")
    )
    parser.add_argument(
        "--weight_save_dir",
        type=str,
        default=os.environ.get("LLM_PETCHAT_OUT")
        or os.path.join(CURRENT_DIR, "checkpoints", "petchat"),
    )
    parser.add_argument(
        "--model_size", type=str, default=os.environ.get("LLM_MODEL_SIZE") or "1.5b"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=int(os.environ.get("LLM_MAX_SEQ_LEN") or 2048),
    )
    parser.add_argument(
        "--epochs", type=int, default=int(os.environ.get("LLM_EPOCHS") or 3)
    )
    parser.add_argument(
        "--batch_size", type=int, default=int(os.environ.get("LLM_BATCH_SIZE") or 2)
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=int(os.environ.get("LLM_GAS") or 16),
    )
    parser.add_argument(
        "--learning_rate", type=float, default=float(os.environ.get("LLM_LR") or 8e-5)
    )
    parser.add_argument(
        "--weight_decay", type=float, default=float(os.environ.get("LLM_WD") or 0.0)
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument(
        "--grad_clip", type=float, default=float(os.environ.get("LLM_GRAD_CLIP") or 1.0)
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=int(os.environ.get("LLM_WARMUP") or 200)
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=float(os.environ.get("LLM_MIN_LR_RATIO") or 0.1),
    )
    parser.add_argument(
        "--num_workers", type=int, default=int(os.environ.get("LLM_NUM_WORKERS") or 2)
    )
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("LLM_SEED") or 1234)
    )
    parser.add_argument(
        "--mixed_precision",
        type=int,
        default=int(os.environ.get("LLM_MIXED_PRECISION") or 1),
    )
    parser.add_argument(
        "--precision_dtype",
        type=str,
        default=os.environ.get("LLM_PRECISION_DTYPE") or "bf16",
    )
    parser.add_argument(
        "--enable_torch_compile",
        type=int,
        default=int(os.environ.get("LLM_TORCH_COMPILE") or 1),
    )
    parser.add_argument(
        "--torch_compile_backend",
        type=str,
        default=os.environ.get("LLM_TORCH_COMPILE_BACKEND") or "inductor",
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default=os.environ.get("LLM_TORCH_COMPILE_MODE") or "default",
    )
    parser.add_argument(
        "--torch_compile_fullgraph",
        type=int,
        default=int(os.environ.get("LLM_TORCH_COMPILE_FULLGRAPH") or 0),
    )
    parser.add_argument(
        "--torch_compile_dynamic",
        type=int,
        default=int(os.environ.get("LLM_TORCH_COMPILE_DYNAMIC") or 0),
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=int(
            os.environ.get("LLM_CPU_THREADS") or max(1, min((os.cpu_count() or 1), 8))
        ),
    )
    parser.add_argument(
        "--cpu_interop_threads",
        type=int,
        default=int(os.environ.get("LLM_CPU_INTEROP_THREADS") or 1),
    )
    parser.add_argument(
        "--lora_rank", type=int, default=int(os.environ.get("LLM_LORA_RANK") or 8)
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=int(os.environ.get("LLM_LORA_ALPHA") or 16)
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=float(os.environ.get("LLM_LORA_DROPOUT") or 0.05),
    )
    parser.add_argument(
        "--lora_name", type=str, default=os.environ.get("LLM_LORA_NAME") or "petchat"
    )
    parser.add_argument(
        "--save_lora_only",
        type=int,
        default=int(os.environ.get("LLM_SAVE_LORA_ONLY") or 1),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=int(os.environ.get("LLM_MAX_TRAIN_SAMPLES") or 0),
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=int(os.environ.get("LLM_MAX_VAL_SAMPLES") or 0),
    )
    parser.add_argument(
        "--enable_custom_extension",
        type=int,
        default=int(os.environ.get("LLM_WSL_ENABLE_EXTENSION") or 1),
    )
    parser.add_argument(
        "--build_extension_if_missing",
        type=int,
        default=int(os.environ.get("LLM_WSL_BUILD_EXTENSION") or 0),
    )
    parser.add_argument(
        "--enable_thread_manager",
        type=int,
        default=int(os.environ.get("LLM_WSL_ENABLE_THREAD_MANAGER") or 1),
    )
    parser.add_argument(
        "--thread_manager_workers",
        type=int,
        default=int(
            os.environ.get("LLM_WSL_THREAD_WORKERS")
            or max(1, min(((os.cpu_count() or 1) // 2), 8))
        ),
    )
    args = parser.parse_args()

    cfg = vars(args)
    cpu_threads = _configure_cpu_threads(cfg)
    _seed_all(cfg["seed"])
    os.makedirs(cfg["weight_save_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChineseTokenizer(max_length=cfg["max_seq_len"])

    ext, ThreadManager = _maybe_load_extension(cfg)
    if ext is not None:
        print("成功加载自定义 WSL 扩展 (CUDA/C++ Thread)!")
        _init_thread_manager(cfg, ext, cpu_threads)
    else:
        print("未检测到自定义 WSL 扩展，将使用原生 PyTorch 实现。")

    train_ds = PetChatJsonlDataset(
        cfg["train_data_path"],
        tokenizer=tokenizer,
        max_seq_len=cfg["max_seq_len"],
        seed=cfg["seed"],
        max_samples=(cfg["max_train_samples"] or None),
    )
    val_ds = PetChatJsonlDataset(
        cfg["val_data_path"],
        tokenizer=tokenizer,
        max_seq_len=cfg["max_seq_len"],
        seed=cfg["seed"] + 1,
        max_samples=(cfg["max_val_samples"] or None),
    )

    collate_fn = make_collate_fn(int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model_config = _get_model_config(cfg["model_size"], tokenizer)
    if int(model_config.get("context_length") or cfg["max_seq_len"]) < int(
        cfg["max_seq_len"]
    ):
        model_config["context_length"] = int(cfg["max_seq_len"])
    model = LlamaLite(model_config).to(device)

    if ext is not None:
        try:
            model = patch_model(model, layer=2, ext=ext)
        except Exception:
            pass

    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)

    base_sd = _load_weights_from_path(cfg.get("base_model_path"))
    if isinstance(base_sd, dict) and base_sd:
        model.load_state_dict(base_sd, strict=False)
        print(f"已加载 base 权重: {cfg.get('base_model_path')}")

    model.enable_lora(
        rank=int(cfg["lora_rank"]),
        alpha=int(cfg["lora_alpha"]),
        dropout=float(cfg["lora_dropout"]),
        name=str(cfg["lora_name"]),
    )
    model.set_active_adapters([str(cfg["lora_name"])])
    model.freeze_base_model()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(cfg["learning_rate"]),
        betas=(float(cfg["beta1"]), float(cfg["beta2"])),
        weight_decay=float(cfg["weight_decay"]),
    )

    total_steps = max(
        1,
        (len(train_loader) * int(cfg["epochs"]))
        // max(1, int(cfg["gradient_accumulation_steps"])),
    )
    lr_scheduler = get_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(cfg["warmup_steps"]),
        min_lr_ratio=float(cfg["min_lr_ratio"]),
    )

    use_amp = bool(int(cfg["mixed_precision"])) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if str(cfg["precision_dtype"]).lower() == "bf16"
        else torch.float16
    )
    scaler_enabled = use_amp and (amp_dtype == torch.float16)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    if (
        bool(int(cfg["enable_torch_compile"]))
        and hasattr(torch, "compile")
        and device.type == "cuda"
    ):
        try:
            model = torch.compile(
                model,
                backend=str(cfg["torch_compile_backend"]),
                mode=str(cfg["torch_compile_mode"]),
                fullgraph=bool(int(cfg["torch_compile_fullgraph"])),
                dynamic=bool(int(cfg["torch_compile_dynamic"])),
            )
            print("已启用 torch.compile")
        except Exception as e:
            print(f"torch.compile 启用失败，将回退到 eager: {e}")

    resume_checkpoint = {}
    latest_path, latest_epoch, latest_kind = find_latest_checkpoint(
        cfg["weight_save_dir"]
    )
    if latest_path:
        try:
            resume_checkpoint = _load_checkpoint_safetensors(latest_path)
        except Exception:
            resume_checkpoint = {}
            latest_path = None
            latest_epoch = 0
            latest_kind = None

    start_epoch = 0
    global_step = 0
    best_val_ppl = float("inf")
    if latest_path and isinstance(resume_checkpoint, dict) and resume_checkpoint:
        ckpt_kind = resume_checkpoint.get("checkpoint_kind") or latest_kind
        if ckpt_kind == "lora":
            lora_sd = resume_checkpoint.get("lora_state_dict")
            if not isinstance(lora_sd, dict):
                lora_sd = {
                    k: v
                    for k, v in resume_checkpoint.items()
                    if ".lora_adapters." in str(k)
                }
            if isinstance(lora_sd, dict) and lora_sd:
                model_to_load = model.module if hasattr(model, "module") else model
                model_to_load.load_state_dict(lora_sd, strict=False)
            try:
                optimizer.load_state_dict(
                    resume_checkpoint.get("optimizer_state_dict") or {}
                )
            except Exception:
                pass
            try:
                lr_scheduler.load_state_dict(
                    resume_checkpoint.get("scheduler_state_dict") or {}
                )
            except Exception:
                pass
            try:
                if scaler_enabled and resume_checkpoint.get("scaler_state_dict"):
                    scaler.load_state_dict(resume_checkpoint.get("scaler_state_dict"))
            except Exception:
                pass
            start_epoch = int(resume_checkpoint.get("epoch", latest_epoch) or 0)
            global_step = int(resume_checkpoint.get("global_step", 0) or 0)
            best_val_ppl = float(
                resume_checkpoint.get("best_perplexity", best_val_ppl) or best_val_ppl
            )
            print(
                f"已恢复 LoRA 检查点: {latest_path} (epoch={start_epoch}, step={global_step})"
            )

    model.train()
    if start_epoch >= int(cfg["epochs"]):
        print("所有 Epoch 已完成，无需训练。")
        return

    log_interval = 10
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for epoch in range(start_epoch, int(cfg["epochs"])):
        print(f"\n===== Epoch {epoch + 1}/{int(cfg['epochs'])} =====")
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        progress = tqdm(train_loader, desc=f"Training {epoch + 1}")
        for step_idx, batch in enumerate(progress):
            x = batch["input_ids"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            labels = batch["labels"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=amp_dtype
                )
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(
                    logits[:, :-1, :].contiguous().flatten(0, 1),
                    labels[:, 1:].contiguous().flatten(0, 1),
                    ignore_index=-100,
                )
                loss = loss / max(1, int(cfg["gradient_accumulation_steps"]))

            cur_loss = float(loss.item()) * max(
                1, int(cfg["gradient_accumulation_steps"])
            )
            epoch_loss_sum += cur_loss
            epoch_loss_count += 1

            scaler.scale(loss).backward()
            if (step_idx + 1) % max(1, int(cfg["gradient_accumulation_steps"])) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["grad_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1
                if global_step % log_interval == 0:
                    lr = float(lr_scheduler.get_last_lr()[0])
                    avg = epoch_loss_sum / max(1, epoch_loss_count)
                    progress.set_postfix(
                        {"loss": f"{avg:.4f}", "lr": f"{lr:.6f}", "step": global_step}
                    )

        train_loss = epoch_loss_sum / max(1, epoch_loss_count)
        val_loss, val_ppl = evaluate_loss(model, val_loader, device, use_amp, amp_dtype)
        if val_ppl is None:
            val_ppl = float("inf")
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss if val_loss is not None else None}, val_ppl={val_ppl:.2f}, time={(time.time() - t0):.1f}s"
        )

        raw_model = model.module if hasattr(model, "module") else model
        if bool(int(cfg["save_lora_only"])):
            checkpoint_state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "lora_state_dict": _move_to_cpu(raw_model.get_lora_state_dict()),
                "lora": (
                    raw_model.get_lora_config()
                    if hasattr(raw_model, "get_lora_config")
                    else None
                ),
                "base_model_path": cfg.get("base_model_path"),
                "checkpoint_kind": "lora",
                "optimizer_state_dict": _move_to_cpu(optimizer.state_dict()),
                "scheduler_state_dict": _move_to_cpu(lr_scheduler.state_dict()),
                "scaler_state_dict": (
                    _move_to_cpu(scaler.state_dict()) if scaler_enabled else None
                ),
                "best_perplexity": best_val_ppl,
                "config": cfg,
            }
            epoch_path = os.path.join(
                cfg["weight_save_dir"], f"lora_epoch_{epoch + 1}.safetensors"
            )
        else:
            checkpoint_state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": _move_to_cpu(raw_model.state_dict()),
                "checkpoint_kind": "full",
                "optimizer_state_dict": _move_to_cpu(optimizer.state_dict()),
                "scheduler_state_dict": _move_to_cpu(lr_scheduler.state_dict()),
                "scaler_state_dict": (
                    _move_to_cpu(scaler.state_dict()) if scaler_enabled else None
                ),
                "best_perplexity": best_val_ppl,
                "config": cfg,
            }
            epoch_path = os.path.join(
                cfg["weight_save_dir"], f"model_epoch_{epoch + 1}.safetensors"
            )
        _save_checkpoint_safetensors(checkpoint_state, epoch_path)
        print(f"已保存检查点(safetensors): {epoch_path}")

        if isinstance(val_ppl, (int, float)) and val_ppl < best_val_ppl:
            best_val_ppl = float(val_ppl)
            best_path = os.path.join(
                cfg["weight_save_dir"],
                (
                    "best_lora.safetensors"
                    if bool(int(cfg["save_lora_only"]))
                    else "best_model.safetensors"
                ),
            )
            if bool(int(cfg["save_lora_only"])):
                _save_state_dict_safetensors(raw_model.get_lora_state_dict(), best_path)
            else:
                _save_state_dict_safetensors(raw_model.state_dict(), best_path)
            print(f"已保存 best: {best_path}")

    raw_model = model.module if hasattr(model, "module") else model
    final_path = os.path.join(
        cfg["weight_save_dir"],
        (
            "final_lora.safetensors"
            if bool(int(cfg["save_lora_only"]))
            else "final_model.safetensors"
        ),
    )
    if bool(int(cfg["save_lora_only"])):
        _save_state_dict_safetensors(raw_model.get_lora_state_dict(), final_path)
    else:
        _save_state_dict_safetensors(raw_model.state_dict(), final_path)
    print(f"训练完成，已保存: {final_path}")


if __name__ == "__main__":
    main()
