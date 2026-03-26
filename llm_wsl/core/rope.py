import torch

try:
    _ext_name = __import__("os").environ.get("LLM_WSL_EXTENSION_NAME", "").strip()
    if _ext_name:
        llm_wsl_extension = __import__(_ext_name)
    else:
        import llm_wsl_extension
except Exception:
    llm_wsl_extension = None


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def _rope_cuda_call(xq, xk, cos, sin):
    xq_out = llm_wsl_extension.rope(xq, cos, sin)
    xk_out = llm_wsl_extension.rope(xk, cos, sin)
    return xq_out, xk_out


try:
    import torch._dynamo as dynamo

    _rope_cuda_call = dynamo.disable(_rope_cuda_call)
except ImportError:
    pass


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    if llm_wsl_extension is not None and xq.is_cuda and xk.is_cuda:
        freqs_real = freqs_cis
        if torch.is_complex(freqs_real):
            cos = freqs_real.real.to(dtype=xq.dtype, device=xq.device)
            sin = freqs_real.imag.to(dtype=xq.dtype, device=xq.device)
            if cos.ndim > 2:
                pass
        else:
            cos = freqs_real.to(dtype=xq.dtype, device=xq.device)
            sin = torch.zeros_like(cos)
        if not xq.is_contiguous():
            xq = xq.contiguous()
        if not xk.is_contiguous():
            xk = xk.contiguous()
        if not cos.is_contiguous():
            cos = cos.contiguous()
        if not sin.is_contiguous():
            sin = sin.contiguous()
        return _rope_cuda_call(xq, xk, cos, sin)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
