"""Decoder block: pre-norm attention + pre-norm MoE."""

from __future__ import annotations

import torch.nn as nn

from .attention import CausalSelfAttention, make_rms_norm
from .config import MoELLMConfig
from .moe import MoE


class DecoderLayer(nn.Module):
    def __init__(self, config: MoELLMConfig) -> None:
        super().__init__()
        self.input_layernorm = make_rms_norm(config.d_model)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = make_rms_norm(config.d_model)
        self.moe = MoE(config)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        h = self.input_layernorm(x)
        h, present_kv = self.self_attn(
            h,
            attn_mask=attn_mask,
            past_kv=past_kv,
            position_offset=position_offset,
            use_cache=use_cache,
        )
        x = x + h
        h = self.post_attention_layernorm(x)
        h, aux = self.moe(h)
        x = x + h
        return x, aux, present_kv


class MoELLM(nn.Module):
    """Decoder-only transformer with MoE FFN stacks."""

    def __init__(self, config: MoELLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.n_layers))
        self.norm = make_rms_norm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_offset: int | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        if past_key_values is not None:
            pos_off = past_key_values[0][0].shape[2] if position_offset is None else position_offset
        else:
            pos_off = 0 if position_offset is None else position_offset

        x = self.embed(input_ids)
        total_aux = input_ids.new_zeros(())
        next_past: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, aux, present = layer(
                x,
                attn_mask=attn_mask,
                past_kv=layer_past,
                position_offset=pos_off,
                use_cache=use_cache,
            )
            total_aux = total_aux + aux
            if use_cache and next_past is not None and present is not None:
                next_past.append(present)
        x = self.norm(x)
        logits = self.lm_head(x)
        past_out = next_past if use_cache else None
        return logits, total_aux, past_out
