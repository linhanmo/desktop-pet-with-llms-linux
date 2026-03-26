import torch
import torch.nn as nn
from .layers import RMSNorm, FeedForward
from .attention import GQAWithLoRA


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.att = GQAWithLoRA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            num_kv_heads=cfg["n_kv_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)

    def forward(self, x, freqs_cis=None):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, freqs_cis=freqs_cis)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
