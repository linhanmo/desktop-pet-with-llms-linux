import json
import os
import random
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        if hasattr(tokenizer, "encode"):
            try:
                token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
            except TypeError:
                token_ids = tokenizer.encode(txt)
        else:
            raise ValueError("Tokenizer must implement encode method")
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class TensorPairListDataset(Dataset):
    def __init__(self, input_ids, target_ids):
        self.input_ids = input_ids
        self.target_ids = target_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def load_binary_dataset_meta(bin_path: str) -> dict:
    meta_path = f"{bin_path}.meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


class BinaryTokenIterableDataset(IterableDataset):
    def __init__(
        self,
        bin_path: str,
        seq_len: int,
        num_samples: int,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
        shuffle_buffer_size: int = 0,
        seed: int = 1234,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.bin_path = bin_path
        self.seq_len = int(seq_len)
        self.num_samples = int(num_samples)
        self.start_sample = int(start_sample)
        self.end_sample = (
            int(end_sample) if end_sample is not None else self.num_samples
        )
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        if dtype is None:
            try:
                meta = load_binary_dataset_meta(bin_path)
                dtype = meta.get("dtype")
            except Exception:
                dtype = None
        self.dtype = str(dtype) if dtype else "int32"
        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if self.num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if not (0 <= self.start_sample <= self.end_sample <= self.num_samples):
            raise ValueError("Invalid [start_sample, end_sample) range")

    def __len__(self):
        return self.end_sample - self.start_sample

    def _iter_indices(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        stride = world_size * num_workers
        offset = rank * num_workers + worker_id
        i = self.start_sample + offset
        while i < self.end_sample:
            yield i
            i += stride

    def __iter__(self):
        record_tokens = self.seq_len * 2
        np_dtype = np.int32
        if self.dtype in ("uint16", "u2"):
            np_dtype = np.uint16
        data = np.memmap(
            self.bin_path,
            dtype=np_dtype,
            mode="r",
            shape=(self.num_samples, record_tokens),
        )
        rng = random.Random(self.seed + self.start_sample + self.end_sample)
        buffer = []
        for idx in self._iter_indices():
            row = data[idx]
            x_np = np.array(row[: self.seq_len], dtype=np.int64, copy=True)
            y_np = np.array(row[self.seq_len :], dtype=np.int64, copy=True)
            x = torch.from_numpy(x_np)
            y = torch.from_numpy(y_np)
            sample = (x, y)
            if self.shuffle_buffer_size > 0:
                buffer.append(sample)
                if len(buffer) >= self.shuffle_buffer_size:
                    j = rng.randrange(len(buffer))
                    yield buffer.pop(j)
            else:
                yield sample
        if self.shuffle_buffer_size > 0 and buffer:
            rng.shuffle(buffer)
            for s in buffer:
                yield s


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    dataset = None
    sampler = None
    if isinstance(txt, str) and txt.endswith(".pt"):
        data = torch.load(txt, weights_only=False, map_location="cpu")
        dataset = TensorPairListDataset(data["input_ids"], data["target_ids"])
    elif isinstance(txt, str) and txt.endswith(".bin"):
        meta = load_binary_dataset_meta(txt)
        seq_len = int(meta["seq_len"])
        num_samples = int(meta["num_samples"])
        shuffle_buffer_size = int(meta.get("shuffle_buffer_size", 0))
        seed = int(meta.get("seed", 1234))
        dtype = meta.get("dtype")
        dataset = BinaryTokenIterableDataset(
            txt,
            seq_len=seq_len,
            num_samples=num_samples,
            start_sample=0,
            end_sample=num_samples,
            shuffle_buffer_size=shuffle_buffer_size if shuffle else 0,
            seed=seed,
            dtype=dtype,
        )
        shuffle = False
    elif isinstance(txt, (list, tuple)) and len(txt) == 2:
        dataset = TensorPairListDataset(txt[0], txt[1])
    else:
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    if (
        isinstance(dataset, Dataset)
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset)
        shuffle = False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )
    return dataloader
