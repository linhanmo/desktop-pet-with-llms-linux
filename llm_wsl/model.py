import torch
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from llm.core.config import LLAMA_LITE_150M_CONFIG
from llm.core.model import LlamaLite
from llm.chinese_tokenizer import ChineseTokenizer


def main():
    torch.manual_seed(123)
    print("--- Testing Modularized LlamaLite Model ---")
    model = LlamaLite(LLAMA_LITE_150M_CONFIG)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    vocab_size = LLAMA_LITE_150M_CONFIG["vocab_size"]
    print(f"Vocab size: {vocab_size}")
    start_context = "你好，我是"
    try:
        tokenizer = ChineseTokenizer()
        print("Using ChineseTokenizer (Local).")
        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
        print("\nInput text:", start_context)
        print("Encoded input text:", encoded)
        print("encoded_tensor.shape:", encoded_tensor.shape)
        out = model.generate_text(
            idx=encoded_tensor,
            max_new_tokens=10,
            context_size=LLAMA_LITE_150M_CONFIG["context_length"],
        )
        decoded_text = tokenizer.decode(out.squeeze(0).tolist())
        print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
        print("\nOutput:", out)
        print("Output length:", len(out[0]))
        print("Output text:", decoded_text)
    except Exception as e:
        print(f"Tokenizer error or not found: {e}")
        import traceback

        traceback.print_exc()
    print("\n--- Testing LoRA Integration ---")
    model.enable_lora(rank=8, alpha=16)
    model.freeze_base_model()
    model.print_trainable_parameters()


if __name__ == "__main__":
    main()
