import torch
import torch.nn as nn
from torch.autograd import Function
import os


def _try_import_wsl_extension():
    name = os.environ.get("LLM_WSL_EXTENSION_NAME", "").strip()
    if not name:
        return None
    try:
        return __import__(name)
    except Exception:
        return None


_WSL_EXT = _try_import_wsl_extension()


class _SwiGLUExtFunction(Function):
    @staticmethod
    def forward(ctx, x_gate, x_val):
        use_ext = (
            _WSL_EXT is not None
            and hasattr(_WSL_EXT, "swiglu_forward")
            and hasattr(_WSL_EXT, "swiglu_backward")
            and x_gate.is_cuda
            and x_val.is_cuda
        )
        x_gate_c = x_gate.contiguous()
        x_val_c = x_val.contiguous()
        ctx.use_ext = bool(use_ext)
        ctx.save_for_backward(x_gate_c, x_val_c)
        if ctx.use_ext:
            return _WSL_EXT.swiglu_forward(x_gate_c, x_val_c)
        return torch.nn.functional.silu(x_gate_c) * x_val_c

    @staticmethod
    def backward(ctx, grad_output):
        x_gate, x_val = ctx.saved_tensors
        grad_output_c = grad_output.contiguous()
        if ctx.use_ext and _WSL_EXT is not None:
            grads = _WSL_EXT.swiglu_backward(grad_output_c, x_gate, x_val)
            if isinstance(grads, (tuple, list)) and len(grads) == 2:
                return grads[0], grads[1]
        a = x_gate
        b = x_val
        dy = grad_output_c
        s = torch.sigmoid(a)
        silu = a * s
        dsilu = s * (1.0 + a * (1.0 - s))
        grad_gate = dy * b * dsilu
        grad_val = dy * silu
        return grad_gate, grad_val


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if hasattr(torch.nn.functional, "rms_norm"):
            return torch.nn.functional.rms_norm(
                x, self.weight.shape, self.weight, self.eps
            )
        x_float = x.to(torch.float32)
        norm_x = self._norm(x_float)
        return norm_x.to(dtype=x.dtype) * self.weight


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_gate, x_val):
        return _SwiGLUExtFunction.apply(x_gate, x_val)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = int(cfg.get("ffn_hidden_dim") or 0)
        if hidden_dim <= 0:
            hidden_dim = 4 * cfg["emb_dim"]
            hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
        self.gate_proj = nn.Linear(cfg["emb_dim"], hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg["emb_dim"], hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, cfg["emb_dim"], bias=False)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        fused = self.swiglu(gate, up)
        output = self.down_proj(fused)
        output = self.dropout(output)
        return output
