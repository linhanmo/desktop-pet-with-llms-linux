import torch
import torch.nn as nn


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
        self.silu = nn.SiLU()

    def forward(self, x_gate, x_val):
        return self.silu(x_gate) * x_val


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

        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        gate = self.silu(self.gate_proj(x))
        up = self.up_proj(x)
        fused = gate * up
        output = self.down_proj(fused)
        output = self.dropout(output)
        return output
