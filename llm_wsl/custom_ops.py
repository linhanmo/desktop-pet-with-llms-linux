import os
import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import llm_wsl_extension
except ImportError:
    llm_wsl_extension = None

_ACTIVE_EXTENSION = llm_wsl_extension


def set_active_extension(ext):
    global _ACTIVE_EXTENSION
    _ACTIVE_EXTENSION = ext


def _try_import_extension(module_names):
    for name in module_names:
        if not name:
            continue
        try:
            return __import__(name)
        except Exception:
            continue
    return None


def get_extension(layer: int | None = None):
    env_name = os.environ.get("LLM_WSL_EXTENSION_NAME")
    candidates = []
    if isinstance(env_name, str) and env_name.strip():
        candidates.append(env_name.strip())
    if layer == 2:
        candidates.append("llm_wsl_extension_petchat")
        candidates.append("llm_wsl_extension_instruction")
        candidates.append("llm_wsl_extension_layer2")
    candidates.append("llm_wsl_extension_petchat")
    candidates.append("llm_wsl_extension")
    ext = _try_import_extension(candidates)
    return ext


class RMSNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, epsilon=1e-6):
        if _ACTIVE_EXTENSION is None:
            raise RuntimeError("WSL extension is not active")
        x = x.contiguous()
        weight = weight.contiguous()
        ctx.save_for_backward(x, weight)
        ctx.epsilon = epsilon
        output = _ACTIVE_EXTENSION.rms_norm(x, weight, epsilon)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        epsilon = ctx.epsilon
        grad_output = grad_output.contiguous()
        grad_input = _ACTIVE_EXTENSION.rms_norm_backward(
            grad_output, x, weight, epsilon
        )
        with torch.no_grad():
            variance = x.pow(2).mean(-1, keepdim=True)
            input_norm = x * torch.rsqrt(variance + epsilon)
            if grad_output.dim() == 3:
                grad_weight = (grad_output * input_norm).sum(dim=(0, 1))
            else:
                grad_weight = (grad_output * input_norm).sum(dim=0)
        return grad_input, grad_weight, None


class RoPEFunction(Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        if _ACTIVE_EXTENSION is None:
            raise RuntimeError("WSL extension is not active")
        x = x.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()
        ctx.save_for_backward(cos, sin)
        return _ACTIVE_EXTENSION.rope(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        if _ACTIVE_EXTENSION is None:
            raise RuntimeError("WSL extension is not active")
        cos, sin = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = _ACTIVE_EXTENSION.rope_backward(grad_output, cos, sin)
        return grad_input, None, None


def _rms_norm_cuda_call(x, weight, eps):
    return RMSNormFunction.apply(x, weight, eps)


try:
    import torch._dynamo as dynamo

    _rms_norm_cuda_call = dynamo.disable(_rms_norm_cuda_call)
except ImportError:
    pass


class FastRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if _ACTIVE_EXTENSION is not None and x.is_cuda:
            return _rms_norm_cuda_call(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


def patch_model(model, layer: int | None = None, ext=None):
    active = ext if ext is not None else get_extension(layer=layer)
    if active is None:
        print("[WSL Extension] Custom extension not found. Skipping patch.")
        return model
    set_active_extension(active)
    print("[WSL Extension] Patching model with FastRMSNorm (CUDA)...")
    from core.model import RMSNorm

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            fast_norm = FastRMSNorm(module.weight.shape[0], module.eps)
            with torch.no_grad():
                fast_norm.weight.copy_(module.weight)
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[1] if "." in name else name
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, fast_norm)
            else:
                pass
            count += 1
    print(f"[WSL Extension] Replaced {count} RMSNorm layers.")
    return model
