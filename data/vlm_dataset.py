"""VLM image-text dataset: LLaVA-style conversation format."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class LLaVADataset(Dataset):
    """
    LLaVA conversation dataset.

    Expected JSON format (one entry per line or array):
    {
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\\nDescribe this image."},
            {"from": "gpt", "value": "This image shows..."}
        ]
    }

    Uses the tokenizer's apply_chat_template for proper formatting
    (e.g., Qwen3 uses <|im_start|>/<|im_end|> markers).
    """

    IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        json_path: str | Path,
        tokenizer,
        image_processor,
        *,
        image_base_dir: str | Path = "",
        max_seq_len: int = 2048,
        image_token_id: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_base_dir = Path(image_base_dir)
        self.max_seq_len = max_seq_len
        self.image_token_id = image_token_id

        data_path = Path(json_path)
        raw = data_path.read_text(encoding="utf-8")
        self.samples = json.loads(raw) if raw.strip().startswith("[") else [
            json.loads(line) for line in raw.strip().split("\n") if line.strip()
        ]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, sample: dict) -> Image.Image | None:
        img_path = sample.get("image", "")
        if not img_path:
            return None
        full_path = self.image_base_dir / img_path if self.image_base_dir else Path(img_path)
        if not full_path.exists():
            return None
        return Image.open(full_path).convert("RGB")

    def _normalize_role(self, role: str) -> str:
        """Normalize conversation roles to standard user/assistant."""
        role = role.lower()
        if role in ("human", "user"):
            return "user"
        if role in ("gpt", "assistant"):
            return "assistant"
        return role

    def _to_messages(self, conversations: list[dict]) -> list[dict]:
        """Convert raw conversations to messages format for apply_chat_template."""
        messages = []
        for turn in conversations:
            role = self._normalize_role(turn.get("from", turn.get("role", "")))
            content = turn.get("value", turn.get("content", ""))
            # Ensure <image> token is in first user turn
            if role == "user" and self.IMAGE_TOKEN not in content and not any(
                self.IMAGE_TOKEN in m.get("content", "") for m in messages
            ):
                content = f"{self.IMAGE_TOKEN}\n{content}"
            messages.append({"role": role, "content": content})
        return messages

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        conversations = sample.get("conversations", [])
        messages = self._to_messages(conversations)

        # Tokenize full conversation (prompt + response) using chat template
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,
        )

        # Tokenize prompt only (all but last assistant turn) to know where labels start
        assistant_turns = [m for m in messages if m["role"] == "assistant"]
        if assistant_turns:
            prompt_messages = messages.copy()
            # Find and remove last assistant turn
            for i in range(len(prompt_messages) - 1, -1, -1):
                if prompt_messages[i]["role"] == "assistant":
                    prompt_messages = prompt_messages[:i]
                    break
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,  # adds assistant header tokens
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors=None,
            )
        else:
            prompt_ids = full_ids

        prompt_len = len(prompt_ids)

        # Labels: mask prompt tokens, keep response tokens
        labels = [-100] * len(full_ids)
        for i in range(prompt_len, len(full_ids)):
            labels[i] = full_ids[i]

        # Load and preprocess image
        pil_image = self._load_image(sample)
        pixel_values = None
        if pil_image is not None:
            pixel_values = self.image_processor(
                images=pil_image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_vlm_batch(
    batch: list[dict],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences in a batch."""

    # Filter samples without images
    batch = [b for b in batch if b["pixel_values"] is not None]
    if not batch:
        return {}

    # Stack pixel values
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    # Pad text sequences
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b.get("attention_mask", torch.ones(L, dtype=torch.long))
        labels[i, :L] = b["labels"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def create_dataloader(
    json_path: str | Path,
    tokenizer,
    image_processor,
    *,
    image_base_dir: str | Path = "",
    batch_size: int = 8,
    max_seq_len: int = 2048,
    image_token_id: int | None = None,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    ds = LLaVADataset(
        json_path=json_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_base_dir=image_base_dir,
        max_seq_len=max_seq_len,
        image_token_id=image_token_id,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_vlm_batch(batch, pad_token_id=tokenizer.pad_token_id or 0),
        pin_memory=True,
    )
