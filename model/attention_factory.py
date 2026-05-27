"""Build self-attention from ``MoELLMConfig``."""

from __future__ import annotations

import torch.nn as nn

from .attention import CausalSelfAttention
from .config import MoELLMConfig
from .MLA import MLA, MLAConfig


def mla_config_from_moellm(config: MoELLMConfig) -> MLAConfig:
    return MLAConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )


def build_attention(config: MoELLMConfig) -> nn.Module:
    """Return an attention module with a unified forward API."""
    if config.attention_type == "mla":
        return MLA(mla_config_from_moellm(config))
    return CausalSelfAttention(config)
