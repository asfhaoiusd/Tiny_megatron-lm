"""Modality projector: maps CLIP patch features into LLM embedding space."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import VLMConfig


class MLPProjector(nn.Module):
    """2-layer MLP with GELU: CLIP hidden → llm_hidden → llm_hidden."""

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.projector_hidden),
            nn.GELU(),
            nn.Linear(config.projector_hidden, config.llm_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, vision_hidden) → (B, N, llm_hidden)"""
        return self.proj(x)


def build_projector(config: VLMConfig) -> nn.Module:
    if config.projector_type == "mlp":
        return MLPProjector(config)
    raise ValueError(f"Unknown projector type: {config.projector_type}")
