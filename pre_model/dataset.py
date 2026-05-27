"""TinyStories + GPT-2 tokenizer streaming batches."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from .config_30m import GPT2_EOS_TOKEN_ID

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TRAIN = _PROJECT_ROOT / "data" / "tinystories" / "TinyStories-train.txt"
_DEFAULT_VALID = _PROJECT_ROOT / "data" / "tinystories" / "TinyStories-valid.txt"
_TOKENIZER_DIR = Path(__file__).resolve().parent / "gpt2_tokenizer"


def get_gpt2_tokenizer(*, cache_dir: Path | None = None):
    from transformers import GPT2Tokenizer

    root = cache_dir or _TOKENIZER_DIR
    root.mkdir(parents=True, exist_ok=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=str(root))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


class _TinyStoriesStream(IterableDataset):
    def __init__(
        self,
        text_path: Path,
        *,
        seq_len: int,
        tokenizer,
        max_stories: int | None = None,
        seed: int = 42,
    ) -> None:
        self.text_path = text_path
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.max_stories = max_stories
        self.seed = seed

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))

        buffer: list[int] = []
        stories = 0
        with self.text_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = self.tokenizer.encode(line, add_special_tokens=False)
                ids.append(GPT2_EOS_TOKEN_ID)
                buffer.extend(ids)
                stories += 1

                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len + 1 :]
                    yield torch.tensor(chunk, dtype=torch.long)

                if self.max_stories is not None and stories >= self.max_stories:
                    break

        if len(buffer) >= 2:
            padded = buffer + [GPT2_EOS_TOKEN_ID] * (self.seq_len + 1 - len(buffer))
            if len(padded) >= self.seq_len + 1:
                yield torch.tensor(padded[: self.seq_len + 1], dtype=torch.long)


def _collate(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack(batch, dim=0)
    return x[:, :-1], x[:, 1:]


class TinyStoriesDataLoader:
    """Thin wrapper around ``DataLoader`` for train/valid TinyStories streams."""

    def __init__(
        self,
        *,
        train_path: Path | None = None,
        valid_path: Path | None = None,
        seq_len: int = 256,
        batch_size: int = 8,
        max_train_stories: int | None = 2000,
        max_valid_stories: int | None = 500,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        from torch.utils.data import DataLoader

        self.tokenizer = get_gpt2_tokenizer()
        train_ds = _TinyStoriesStream(
            train_path or _DEFAULT_TRAIN,
            seq_len=seq_len,
            tokenizer=self.tokenizer,
            max_stories=max_train_stories,
            seed=seed,
        )
        valid_ds = _TinyStoriesStream(
            valid_path or _DEFAULT_VALID,
            seq_len=seq_len,
            tokenizer=self.tokenizer,
            max_stories=max_valid_stories,
            seed=seed + 1,
        )
        self.train = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=_collate,
            num_workers=num_workers,
        )
        self.valid = DataLoader(
            valid_ds,
            batch_size=batch_size,
            collate_fn=_collate,
            num_workers=num_workers,
        )
