import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_emb
import math


class _LoRAAdapter(nn.Module):
    def __init__(
        self, d_in: int, q_out: int, kv_out: int, rank: int, alpha: int, dropout: float
    ):
        super().__init__()
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.dropout = float(dropout)
        self.dropout_layer = nn.Dropout(p=self.dropout) if self.dropout > 0 else None

        self.q_A = nn.Parameter(torch.zeros(d_in, self.rank))
        self.q_B = nn.Parameter(torch.zeros(self.rank, q_out))
        self.k_A = nn.Parameter(torch.zeros(d_in, self.rank))
        self.k_B = nn.Parameter(torch.zeros(self.rank, kv_out))
        self.v_A = nn.Parameter(torch.zeros(d_in, self.rank))
        self.v_B = nn.Parameter(torch.zeros(self.rank, kv_out))

        nn.init.kaiming_uniform_(self.q_A, a=math.sqrt(5))
        nn.init.zeros_(self.q_B)
        nn.init.kaiming_uniform_(self.k_A, a=math.sqrt(5))
        nn.init.zeros_(self.k_B)
        nn.init.kaiming_uniform_(self.v_A, a=math.sqrt(5))
        nn.init.zeros_(self.v_B)

    def apply(self, x: torch.Tensor):
        scale = float(self.alpha) / float(self.rank)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        dq = (x @ self.q_A @ self.q_B) * scale
        dk = (x @ self.k_A @ self.k_B) * scale
        dv = (x @ self.v_A @ self.v_B) * scale
        return dq, dk, dv


class GQAWithLoRA(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        num_kv_heads,
        qkv_bias=False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_out // num_heads
        self.kv_dim = self.head_dim * num_kv_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, self.kv_dim, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, self.kv_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.lora_adapters = nn.ModuleDict()
        self.active_adapters = []

    def forward(self, x, freqs_cis=None):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        if self.active_adapters:
            for name in self.active_adapters:
                if name not in self.lora_adapters:
                    continue
                adapter = self.lora_adapters[name]
                dq, dk, dv = adapter.apply(x)
                queries = queries + dq
                keys = keys + dk
                values = values + dv

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_heads, self.head_dim)
        if freqs_cis is not None:
            queries, keys = apply_rotary_emb(queries, keys, freqs_cis)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.num_heads > self.num_kv_heads:
            num_rep = self.num_heads // self.num_kv_heads
            keys = (
                keys.unsqueeze(2)
                .expand(b, self.num_kv_heads, num_rep, num_tokens, self.head_dim)
                .reshape(b, self.num_heads, num_tokens, self.head_dim)
            )
            values = (
                values.unsqueeze(2)
                .expand(b, self.num_kv_heads, num_rep, num_tokens, self.head_dim)
                .reshape(b, self.num_heads, num_tokens, self.head_dim)
            )

        context_vec = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        context_vec = (
            context_vec.transpose(1, 2).contiguous().reshape(b, num_tokens, self.d_out)
        )

        context_vec = self.out_proj(context_vec)

        return context_vec

    def enable_lora(self, rank, alpha, dropout=0.0, name="default"):
        name = str(name)
        if name in self.lora_adapters:
            if name not in self.active_adapters:
                self.active_adapters.append(name)
            return
        adapter = _LoRAAdapter(
            d_in=self.W_query.in_features,
            q_out=self.W_query.out_features,
            kv_out=self.W_key.out_features,
            rank=int(rank),
            alpha=int(alpha),
            dropout=float(dropout),
        )
        self.lora_adapters[name] = adapter
        if name not in self.active_adapters:
            self.active_adapters.append(name)

    def set_active_adapters(self, names):
        if names is None:
            self.active_adapters = []
            return
        keep = []
        for n in names:
            n = str(n)
            if n in self.lora_adapters:
                keep.append(n)
        self.active_adapters = keep
