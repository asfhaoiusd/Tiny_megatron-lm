"""VLM model: CLIP ViT → Projector → Qwen3. LLaVA-style forward with visual token injection."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import VLMConfig
from .projector import build_projector
from .vision_encoder import VisionEncoder


class VLMForConditionalGeneration(nn.Module):
    """
    CLIP (frozen) + Projector (trainable) + Qwen3 (LoRA / frozen).

    Input: pixel_values + input_ids + attention_mask + labels
    Output: logits, loss
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = VisionEncoder(config)
        self.projector = build_projector(config)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name,
            trust_remote_code=True,
        )

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            config.pad_token_id = self.tokenizer.eos_token_id

        self.image_token_id = self._get_image_token_id()

    def _get_image_token_id(self) -> int:
        # Try adding the image token if not in vocab
        tid = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        if tid == self.tokenizer.unk_token_id:
            self.tokenizer.add_tokens([self.IMAGE_TOKEN], special_tokens=True)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            tid = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        return tid

    def _merge_vision_text_embeds(
        self,
        vision_embeds: torch.Tensor,  # (B, V, llm_hidden)
        input_ids: torch.Tensor,       # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replace <image> token positions with vision embeddings.
        Returns: (inputs_embeds, attention_mask) where visual tokens are injected.
        """
        B = input_ids.shape[0]
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        device = text_embeds.device

        image_token_mask = (input_ids == self.image_token_id)
        num_images_per_sample = image_token_mask.sum(dim=1).tolist()
        img_idx = 0

        merged_embeds_list = []
        for b in range(B):
            sample_ids = input_ids[b]
            sample_embeds = text_embeds[b]
            img_mask = sample_ids == self.image_token_id
            n_img = img_mask.sum().item()

            # If model is not expecting images, just use text
            if n_img == 0:
                merged_embeds_list.append(sample_embeds)
                continue

            # Split text embeddings at image token positions, insert vision features
            img_positions = img_mask.nonzero(as_tuple=True)[0]
            segments = []

            start = 0
            for i, pos in enumerate(img_positions):
                # Text before this image token
                if start < pos:
                    segments.append(sample_embeds[start:pos])
                # Vision features for this image
                # Use each image chunk — split vision_embeds evenly or by position
                if i < vision_embeds.shape[1]:
                    # We need to handle one image -> many visual tokens
                    # Simplification: place all visual tokens at the first <image> position
                    if i == 0 and vision_embeds.shape[1] > 1:
                        segments.append(vision_embeds[b])  # All visual tokens at once
                    elif vision_embeds.shape[1] == 1 or i == 0:
                        pass  # Already added above
                start = pos + 1

            if start < len(sample_ids):
                segments.append(sample_embeds[start:])

            merged = torch.cat(segments, dim=0) if segments else sample_embeds
            merged_embeds_list.append(merged)

        # Pad to max length
        max_len = max(e.shape[0] for e in merged_embeds_list)
        padded_embeds = []
        attn_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        for b, embeds in enumerate(merged_embeds_list):
            L = embeds.shape[0]
            if L < max_len:
                pad = torch.zeros(max_len - L, embeds.shape[1], dtype=embeds.dtype, device=device)
                padded_embeds.append(torch.cat([embeds, pad]))
            else:
                padded_embeds.append(embeds[:max_len])
            attn_mask[b, : min(L, max_len)] = True

        return torch.stack(padded_embeds), attn_mask

    def _build_labels(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Build labels with visual tokens masked (-100) and proper padding."""
        B = labels.shape[0]
        padded = torch.full((B, max_len), -100, dtype=labels.dtype, device=labels.device)

        image_mask = (input_ids == self.image_token_id)

        for b in range(B):
            img_positions = image_mask[b].nonzero(as_tuple=True)[0]
            label_idx = 0

            # For each text segment after image tokens
            last_pos = -1
            for pos in img_positions:
                text_start = last_pos + 1
                text_len = pos - text_start
                if text_len > 0:
                    txt_end = label_idx + text_len
                    padded[b, label_idx:txt_end] = labels[b, text_start:pos]
                    label_idx = txt_end
                # Skip visual tokens in labels (they stay -100)
                label_idx += self.vision_encoder.num_patches
                last_pos = pos

            # Remaining text after last image token
            if last_pos + 1 < labels.shape[1]:
                remaining = labels.shape[1] - last_pos - 1
                padded[b, label_idx : label_idx + remaining] = labels[b, last_pos + 1 :]

        return padded

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        pixel_values: (B, C, H, W)
        input_ids: (B, T) — must contain <image> token(s)
        labels: (B, T) — same shape as input_ids, -100 for ignored tokens
        """
        # Encode image
        vision_features = self.vision_encoder(pixel_values)  # (B, 576, 1024)
        vision_embeds = self.projector(vision_features)       # (B, 576, llm_hidden)

        # Merge vision + text
        inputs_embeds, attn_mask = self._merge_vision_text_embeds(vision_embeds, input_ids)

        # Build labels with visual positions masked
        if labels is not None:
            merged_labels = self._build_labels(labels, input_ids, inputs_embeds.shape[1])
        else:
            merged_labels = None

        if merged_labels is not None:
            # Clamp to matching length
            min_len = min(inputs_embeds.shape[1], merged_labels.shape[1])
            inputs_embeds = inputs_embeds[:, :min_len, :]
            attn_mask = attn_mask[:, :min_len]
            merged_labels = merged_labels[:, :min_len]

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask.long() if attn_mask is not None else None,
            labels=merged_labels,
            use_cache=self.config.use_cache,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values,
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate text conditioned on image + prompt."""
        vision_features = self.vision_encoder(pixel_values)
        vision_embeds = self.projector(vision_features)
        inputs_embeds, attn_mask = self._merge_vision_text_embeds(vision_embeds, input_ids)

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask.long(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )
