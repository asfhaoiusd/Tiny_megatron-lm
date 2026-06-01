"""Hyperparameters for the MoE decoder-only LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

AttentionType = Literal["mha", "mqa", "mla"]


@dataclass
class MoELLMConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None  # GQA; None means equal to n_heads
    attention_type: AttentionType = "mha"  # mha/mqa → CausalSelfAttention; mla → MLA
    # MLA (DeepSeek-V2 style); used when attention_type == "mla"
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    v_head_dim: int = 64
    d_ff: int = 2048  # expert hidden dim (SwiGLU intermediate)
    max_seq_len: int = 2048
    dropout: float = 0.0
    # MoE
    n_experts: int = 8
    num_experts_per_tok: int = 2
    # Router / training
    router_aux_loss_coef: float = 0.01

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            object.__setattr__(self, "n_kv_heads", self.n_heads)
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide n_kv_heads"
        assert 1 <= self.num_experts_per_tok <= self.n_experts

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
