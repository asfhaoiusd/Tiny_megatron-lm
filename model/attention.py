"""Multi-head attention: RoPE + PyTorch fused SDPA (GQA via ``enable_gqa`` when available)."""

from __future__ import annotations

import functools
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoELLMConfig
from .rope import apply_rope, RotaryEmbedding


@functools.lru_cache(maxsize=1)
def _sdpa_supports_enable_gqa() -> bool:
    try:
        return "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
    except (TypeError, ValueError):
        return False


def make_rms_norm(dim: int, eps: float = 1e-6) -> nn.Module:
    """Prefer ``torch.nn.RMSNorm``; fall back to a minimal RMSNorm."""
    if hasattr(nn, "RMSNorm"):
        try:
            return nn.RMSNorm((dim,), eps=eps)
        except TypeError:
            return nn.RMSNorm(dim, eps=eps)
    return _RMSNorm(dim, eps=eps)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MoELLMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=config.max_seq_len)
        self._use_native_gqa = _sdpa_supports_enable_gqa() and self.n_rep != 1

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        b, n_kv, t, d = x.shape
        x = x[:, :, None, :, :].expand(b, n_kv, self.n_rep, t, d).reshape(b, n_kv * self.n_rep, t, d)
        return x

    @staticmethod
    def _causal_attn_bias(t_q: int, t_k: int, past_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Additive mask broadcastable to (B, H, Tq, Tk)."""
        bias = torch.zeros(1, 1, t_q, t_k, device=device, dtype=dtype)
        rows = torch.arange(t_q, device=device).view(1, 1, t_q, 1)
        cols = torch.arange(t_k, device=device).view(1, 1, 1, t_k)
        bad = cols > (past_len + rows)
        return bias.masked_fill(bad, float("-inf"))

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        dropout_p: float,
        is_causal: bool,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        kwargs: dict[str, object] = {}
        if self._use_native_gqa:
            kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            **kwargs,  # type: ignore[arg-type]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x, t, position_offset=position_offset)
        q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        present_kv: tuple[torch.Tensor, torch.Tensor] | None = (k, v) if use_cache else None

        k_sdpa, v_sdpa = (k, v) if self._use_native_gqa else (self._repeat_kv(k), self._repeat_kv(v))
        total_len = k_sdpa.shape[2]
        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        dropout_p = float(self.dropout.p) if self.training else 0.0

        if attn_mask is None and past_kv is None:
            out = self._sdpa(q, k_sdpa, v_sdpa, dropout_p=dropout_p, is_causal=True, attn_mask=None)
        elif attn_mask is None and past_kv is not None and t == 1:
            out = self._sdpa(q, k_sdpa, v_sdpa, dropout_p=dropout_p, is_causal=False, attn_mask=None)
        else:
            bias = self._causal_attn_bias(t, total_len, past_len, x.device, q.dtype)
            if attn_mask is not None:
                bias = bias + attn_mask.to(dtype=bias.dtype)
            out = self._sdpa(q, k_sdpa, v_sdpa, dropout_p=dropout_p, is_causal=False, attn_mask=bias)

        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out), present_kv
