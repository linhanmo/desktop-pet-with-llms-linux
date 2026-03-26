LLAMA_LITE_450M_CONFIG = {
    "vocab_size": 151665,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_kv_heads": 4,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
LLAMA_LITE_150M_CONFIG = {
    "vocab_size": 151665,
    "context_length": 1024,
    "emb_dim": 512,
    "n_heads": 8,
    "n_kv_heads": 2,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

LLAMA_LITE_1_5B_CONFIG = {
    "vocab_size": 151665,
    "context_length": 4096,
    "emb_dim": 1536,
    "n_heads": 12,
    "n_kv_heads": 2,
    "n_layers": 28,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "ffn_hidden_dim": 8960,
}


class GPTConfig:
    def __init__(
        self,
        vocab_size,
        context_length,
        emb_dim,
        n_heads,
        n_layers,
        drop_rate,
        qkv_bias,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
