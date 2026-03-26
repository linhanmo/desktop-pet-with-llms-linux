import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "data", "hf_cache")
import torch
from transformers import AutoTokenizer
from typing import List, Union, Optional


class ChineseTokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path=None,
        padding_side="left",
        max_length: Optional[int] = None,
    ):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "qwen_tokenizer"
            )
        print(f"Loading Qwen tokenizer from {pretrained_model_name_or_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            print(f"Error loading tokenizer from local path: {e}")
            raise e
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(
                f"Pad token was None, set to EOS token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})"
            )
        self.tokenizer.padding_side = padding_side
        self.default_max_length = max_length
        self.vocab_size = len(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode(self, text: str, allowed_special: Union[set, str] = None) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: str = "pt",
    ):
        if max_length is None and self.default_max_length is not None:
            max_length = self.default_max_length
        if padding:
            if max_length is not None:
                pad_strategy = "max_length"
            else:
                pad_strategy = "longest"
        else:
            pad_strategy = False
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=pad_strategy,
            return_tensors=return_tensors,
            add_special_tokens=False,
        )

    def batch_decode(
        self,
        sequences: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens
        )


def test_chinese_tokenizer():
    print("\n=== Testing ChineseTokenizer ===")
    try:
        tokenizer = ChineseTokenizer()
    except Exception as e:
        print(f"Failed to initialize tokenizer: {e}")
        return
    print(f"Vocab size: {tokenizer.vocab_size}")
    text = "你好，世界！这是一个测试句子。Hello World."
    print(f"\nOriginal text: {text}")
    ids = tokenizer.encode(text)
    print(f"Encoded IDs: {ids}")
    decoded_text = tokenizer.decode(ids)
    print(f"Decoded text: {decoded_text}")
    assert text == decoded_text, "Decode mismatch! Info lost."
    print("✅ Single encode/decode test passed.")
    texts = [
        "简短的句子",
        "这是一个稍微长一点的中文句子，用于测试 Padding 功能。",
        "Very short",
    ]
    print(f"\nBatch texts: {texts}")
    batch_out = tokenizer.batch_encode(texts, padding=True, truncation=False)
    input_ids = batch_out["input_ids"]
    attention_mask = batch_out["attention_mask"]
    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch attention_mask shape: {attention_mask.shape}")
    assert input_ids.shape[0] == 3
    assert input_ids.shape[1] == max(len(tokenizer.encode(t)) for t in texts)
    print("✅ Batch padding test passed.")
    decoded_batch = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    print(f"Batch decoded: {decoded_batch}")
    max_len = 5
    batch_out_trunc = tokenizer.batch_encode(
        texts, max_length=max_len, truncation=True, padding=True
    )
    print(
        f"\nTruncated input_ids shape (max_len={max_len}): {batch_out_trunc['input_ids'].shape}"
    )
    assert batch_out_trunc["input_ids"].shape[1] == max_len
    print("✅ Batch truncation test passed.")
    print("\n=== All ChineseTokenizer tests passed! ===")


if __name__ == "__main__":
    test_chinese_tokenizer()
