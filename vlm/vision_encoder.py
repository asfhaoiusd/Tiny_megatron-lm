"""CLIP vision encoder wrapper — loads pretrained CLIP ViT, removes CLS token, outputs patch features."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

from .config import VLMConfig


class VisionEncoder(nn.Module):
    """Pretrained CLIP ViT, frozen. Outputs patch embeddings without CLS token."""

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(config.vision_model_name)
        self.config = config

        # Freeze all params
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, C, H, W) → (B, num_patches, vision_hidden)"""
        outputs = self.model(pixel_values, output_hidden_states=True)
        # CLIP Vision outputs: last_hidden_state = (B, 577, 1024) for ViT-L/14@336
        # The first token is CLS; we drop it to keep only patch features
        features = outputs.last_hidden_state[:, 1:, :]  # (B, 576, 1024)
        return features

    @property
    def num_patches(self) -> int:
        return self.config.vision_num_patches - 1  # 576


def get_image_processor(config: VLMConfig) -> CLIPImageProcessor:
    return CLIPImageProcessor.from_pretrained(config.vision_model_name)
