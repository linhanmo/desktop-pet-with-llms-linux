import torch
import torch.nn as nn
import torch.utils.checkpoint
from .layers import RMSNorm
from .transformer import TransformerBlock
from .rope import precompute_freqs_cis


class LlamaLite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                head_dim,
                cfg["context_length"],
                theta=float(cfg.get("rope_theta", 10000.0)),
            ),
        )

        self.gradient_checkpointing = False

        self.apply(self._init_weights)
        if bool(cfg.get("tie_word_embeddings")):
            self.out_head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self, enabled=True):
        self.gradient_checkpointing = enabled
        print(f"Gradient Checkpointing {'enabled' if enabled else 'disabled'}.")

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        x = self.drop_emb(tok_embeds)

        freqs_cis = self.freqs_cis[:seq_len]

        for block in self.trf_blocks:
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, freqs_cis, use_reentrant=False
                )
            else:
                x = block(x, freqs_cis=freqs_cis)

        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits

    def generate_text(
        self, idx, max_new_tokens, context_size=None, temperature=0.0, top_k=None
    ):
        if context_size is None:
            context_size = self.cfg["context_length"]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = self(idx_cond)

            logits = logits[:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def enable_lora(self, rank=8, alpha=16, dropout=0.0, name="default"):
        print(f"Enabling LoRA with rank={rank}, alpha={alpha}, dropout={dropout}...")
        for i, block in enumerate(self.trf_blocks):
            block.att.enable_lora(rank, alpha, dropout, name=name)

    def freeze_base_model(self):
        print("Freezing base model parameters...")
        for name, param in self.named_parameters():
            if ".lora_adapters." in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_base_model(self):
        print("Unfreezing base model parameters...")
        for param in self.parameters():
            param.requires_grad = True

    def load_lora_skill(self, skill_weights_path: str):
        import os

        if not os.path.exists(skill_weights_path):
            print(f"Error: Skill weights not found at {skill_weights_path}")
            return

        print(f"Loading LoRA skill from {skill_weights_path}...")
        try:
            device = next(self.parameters()).device
            payload = torch.load(skill_weights_path, map_location=device)

            if isinstance(payload, dict) and "lora_state_dict" in payload:
                state_dict = payload.get("lora_state_dict") or {}
                lora_cfg = (
                    payload.get("lora")
                    if isinstance(payload.get("lora"), dict)
                    else None
                )
            elif isinstance(payload, dict) and "model_state_dict" in payload:
                state_dict = payload.get("model_state_dict") or {}
                lora_cfg = (
                    payload.get("lora")
                    if isinstance(payload.get("lora"), dict)
                    else None
                )
            else:
                state_dict = payload if isinstance(payload, dict) else {}
                lora_cfg = None

            lora_keys = {k: v for k, v in state_dict.items() if ".lora_adapters." in k}
            if not lora_keys:
                print("Warning: No LoRA keys found in the provided weights.")
                return

            if lora_cfg is not None:
                adapters = lora_cfg.get("adapters")
                active = lora_cfg.get("active_adapters")
                if isinstance(adapters, dict):
                    for adapter_name, c in adapters.items():
                        if isinstance(c, dict):
                            self.enable_lora(
                                rank=int(c.get("rank", 8)),
                                alpha=int(c.get("alpha", 16)),
                                dropout=float(c.get("dropout", 0.0)),
                                name=str(adapter_name),
                            )
                if isinstance(active, list):
                    self.set_active_adapters(active)
            else:
                first_key = list(lora_keys.keys())[0]
                rank = None
                v = lora_keys.get(first_key)
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    rank = int(v.shape[1])
                if rank is None:
                    rank = 8
                print(f"Detected LoRA rank: {rank} from weights.")
                self.enable_lora(rank=rank, alpha=16, dropout=0.0, name="default")
                self.set_active_adapters(["default"])

            missing, unexpected = self.load_state_dict(lora_keys, strict=False)

            missing_lora = [k for k in missing if ".lora_adapters." in k]

            if missing_lora:
                print(f"Warning: Missing LoRA keys: {len(missing_lora)}")
            else:
                print("LoRA skill loaded successfully.")

        except Exception as e:
            print(f"Failed to load LoRA skill: {e}")
            import traceback

            traceback.print_exc()

    def unload_lora_skill(self):
        print("Unloading LoRA skill...")
        for block in self.trf_blocks:
            att = block.att
            att.set_active_adapters(None)
            try:
                att.lora_adapters.clear()
            except Exception:
                import torch.nn as nn

                att.lora_adapters = nn.ModuleDict()

        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("LoRA skill unloaded.")

    def set_active_adapters(self, names):
        for block in self.trf_blocks:
            block.att.set_active_adapters(names)

    def get_lora_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if ".lora_adapters." in k}

    def get_lora_config(self):
        adapters = {}
        active = None
        for block in self.trf_blocks:
            att = getattr(block, "att", None)
            if att is None:
                continue
            if active is None and isinstance(
                getattr(att, "active_adapters", None), list
            ):
                active = list(att.active_adapters)
            lora_adapters = getattr(att, "lora_adapters", None)
            if isinstance(lora_adapters, dict) or hasattr(lora_adapters, "items"):
                for name, module in lora_adapters.items():
                    if name in adapters:
                        continue
                    rank = getattr(module, "rank", None)
                    alpha = getattr(module, "alpha", None)
                    dropout = getattr(module, "dropout", None)
                    if rank is None or alpha is None or dropout is None:
                        continue
                    adapters[str(name)] = {
                        "rank": int(rank),
                        "alpha": int(alpha),
                        "dropout": float(dropout),
                    }
        return {"adapters": adapters, "active_adapters": active or []}

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
