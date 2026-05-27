"""DeepSeek-V2 Multi-head Latent Attention (MLA)，结构对齐 HF ``DeepseekV2Attention``。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import make_rms_norm
from .rope import RotaryEmbedding, apply_rope


@dataclass
class MLAConfig:
    d_model: int = 512
    n_heads: int = 8
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    v_head_dim: int = 64
    max_seq_len: int = 2048
    dropout: float = 0.0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


class MLA(nn.Module):

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        h, d = config.n_heads, config.d_model
        qk = config.qk_head_dim
        d_nope, d_rope, d_v = config.qk_nope_head_dim, config.qk_rope_head_dim, config.v_head_dim

        self.n_heads = h
        self.qk_head_dim = qk
        self.qk_nope_head_dim = d_nope
        self.qk_rope_head_dim = d_rope
        self.v_head_dim = d_v
        self.kv_lora_rank = config.kv_lora_rank

        self.q_a_proj = nn.Linear(d, config.q_lora_rank, bias=False)
        self.q_a_norm = make_rms_norm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, h * qk, bias=False)

        self.kv_a_proj = nn.Linear(d, config.kv_lora_rank + d_rope, bias=False)
        self.kv_a_norm = make_rms_norm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, h * (d_nope + d_v), bias=False)

        self.o_proj = nn.Linear(h * d_v, d, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(d_rope, max_seq_len=config.max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape
        h = self.n_heads

        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.view(b, t, h, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj(x)
        c_latent, k_pe = torch.split(compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_norm(c_latent))
        kv = kv.view(b, t, h, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        #v不再是单独的一部分，而是与k_nope一起被压缩
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        #使用局部rope的原因是其运算与mla之间不能共存（根本原因是矩阵不满足交换律）
        k_pe = k_pe.view(b, 1, t, self.qk_rope_head_dim)
        cos, sin = self.rotary(x, t, position_offset=position_offset)
        q_pe, k_pe = apply_rope(q_pe, k_pe, cos, sin)
        k_pe = k_pe.expand(b, h, t, self.qk_rope_head_dim)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        present_kv = (k, v) if use_cache else None

        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        dropout_p = float(self.dropout.p) if self.training else 0.0
        total_len = k.shape[2]

        if attn_mask is None and past_kv is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        elif attn_mask is None and past_kv is not None and t == 1:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        else:
            bias = torch.zeros(1, 1, t, total_len, device=x.device, dtype=q.dtype)
            rows = torch.arange(t, device=x.device).view(1, 1, t, 1)
            cols = torch.arange(total_len, device=x.device).view(1, 1, 1, total_len)
            bias = bias.masked_fill(cols > (past_len + rows), float("-inf"))
            if attn_mask is not None:
                bias = bias + attn_mask.to(dtype=bias.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=dropout_p, is_causal=False
            )

        out = out.transpose(1, 2).contiguous().view(b, t, h * self.v_head_dim)
        return self.o_proj(out), present_kv


def _smoke_test() -> None:
    cfg = MLAConfig(d_model=256, n_heads=4, max_seq_len=128)
    mla = MLA(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    y, cache = mla(x, use_cache=True)
    assert y.shape == x.shape and cache is not None

    x2 = torch.randn(2, 1, cfg.d_model)
    y2, _ = mla(x2, past_kv=cache, position_offset=16, use_cache=True)
    assert y2.shape == x2.shape
    print("MLA smoke test OK", y.shape)


if __name__ == "__main__":
    _smoke_test()
