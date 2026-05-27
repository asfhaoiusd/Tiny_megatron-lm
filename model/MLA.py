"""DeepSeek-V2 Multi-head Latent Attention (MLA)，含 latent KV cache。"""

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
    """
    DeepSeek-V2 MLA。

    KV cache（``use_cache=True``）仅存 latent，不存完整 K/V::

        present_kv = (compressed_kv, k_pe_raw)
        compressed_kv: (B, T, kv_lora_rank)
        k_pe_raw:      (B, T, qk_rope_head_dim)  # RoPE 之前

    推理时由 ``kv_b_proj`` 从整条 ``compressed_kv`` 重建 ``k_nope`` / ``v``，
    再对整条 ``k_pe_raw`` 施加 RoPE 得到注意力用的 K。
    """

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

    def _latent_from_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """当前步 latent 与未旋转的 k_pe，形状均为 (B, T, *)."""
        mixed = self.kv_a_proj(x)
        c_latent, k_pe = torch.split(mixed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return c_latent, k_pe
    
    #重点就在这了，这里就是kvcache的生成过程，将latent序列生成k_nope和v，直接导致了kccache所占用内存的数量大大减少
    def _k_nope_v_from_latent(self, c_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """由整条 latent 序列重建 k_nope、v -> (B, H, T, *)."""
        b, t, _ = c_latent.shape
        h = self.n_heads
        kv = self.kv_b_proj(self.kv_a_norm(c_latent))
        kv = kv.view(b, t, h, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k_nope, v

    @staticmethod
    def _causal_bias(
        t_q: int, t_k: int, past_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        bias = torch.zeros(1, 1, t_q, t_k, device=device, dtype=dtype)
        rows = torch.arange(t_q, device=device).view(1, 1, t_q, 1)
        cols = torch.arange(t_k, device=device).view(1, 1, 1, t_k)
        return bias.masked_fill(cols > (past_len + rows), float("-inf"))

    @staticmethod
    def cache_seq_len(past_kv: tuple[torch.Tensor, torch.Tensor] | None) -> int:
        """已缓存 token 数（latent 在 dim=1）。"""
        if past_kv is None:
            return 0
        return past_kv[0].shape[1]

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

        c_new, k_pe_new = self._latent_from_x(x)

        if past_kv is not None:
            past_c, past_k_pe = past_kv
            c_latent = torch.cat([past_c, c_new], dim=1)
            k_pe_raw = torch.cat([past_k_pe, k_pe_new], dim=1)
        else:
            c_latent = c_new
            k_pe_raw = k_pe_new

        total_len = c_latent.shape[1]
        past_len = self.cache_seq_len(past_kv)

        k_nope, v = self._k_nope_v_from_latent(c_latent)

        cos_q, sin_q = self.rotary(x, t, position_offset=position_offset)
        q_pe, _ = apply_rope(q_pe, q_pe, cos_q, sin_q)

        k_pe = k_pe_raw.unsqueeze(1)
        cos_k, sin_k = self.rotary(x, total_len, position_offset=0)
        _, k_pe = apply_rope(k_pe, k_pe, cos_k, sin_k)
        k_pe = k_pe.expand(b, h, total_len, self.qk_rope_head_dim)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        present_kv: tuple[torch.Tensor, torch.Tensor] | None = (
            (c_latent, k_pe_raw) if use_cache else None
        )

        dropout_p = float(self.dropout.p) if self.training else 0.0

        if attn_mask is None and past_kv is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        elif attn_mask is None and past_kv is not None and t == 1:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        else:
            bias = self._causal_bias(t, total_len, past_len, x.device, q.dtype)
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
    c, k_pe = cache
    assert c.shape == (2, 16, cfg.kv_lora_rank)
    assert k_pe.shape == (2, 16, cfg.qk_rope_head_dim)

    full_k_elems = 16 * cfg.n_heads * cfg.qk_head_dim
    full_v_elems = 16 * cfg.n_heads * cfg.v_head_dim
    latent_elems = c.numel() // 2 + k_pe.numel() // 2
    full_elems = (full_k_elems + full_v_elems) // 2
    assert latent_elems < full_elems, "latent cache should be smaller than full KV"

    x2 = torch.randn(2, 1, cfg.d_model)
    y2, cache2 = mla(x2, past_kv=cache, position_offset=16, use_cache=True)
    assert y2.shape == x2.shape
    assert cache2 is not None
    assert cache2[0].shape[1] == 17

    print("MLA latent KV cache smoke test OK", y.shape)


if __name__ == "__main__":
    _smoke_test()
