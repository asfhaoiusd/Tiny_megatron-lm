"""MoE FFN: top-k routing + SwiGLU using stacked weights and ``F.linear`` (GEMM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoELLMConfig


class MoE(nn.Module):
    """
    Top-k token-choice routing. Returns ``(y, aux_loss)``.
    Experts share the same layout; forward uses ``F.linear`` (cuBLAS) per expert group.
    """

    def __init__(self, config: MoELLMConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.n_experts = config.n_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.aux_loss_coef = config.router_aux_loss_coef

        e, d_ff, d = config.n_experts, config.d_ff, config.d_model
        # Shapes match ``F.linear(in, weight)`` with weight ``(out, in)``.
        self.expert_w1 = nn.Parameter(torch.empty(e, d_ff, d))
        self.expert_w2 = nn.Parameter(torch.empty(e, d, d_ff))
        self.expert_w3 = nn.Parameter(torch.empty(e, d_ff, d))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.expert_w1, std=0.02)
        nn.init.trunc_normal_(self.expert_w2, std=0.02)
        nn.init.trunc_normal_(self.expert_w3, std=0.02)
        nn.init.trunc_normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        router_logits = self.router(x)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        y = torch.zeros_like(x)
        flat_x = x.view(-1, d)
        flat_sel = selected_experts.view(-1, self.top_k)
        flat_w = routing_weights.view(-1, self.top_k)
        flat_y = y.view(-1, d)

        for k in range(self.top_k):
            expert_idx = flat_sel[:, k]
            w = flat_w[:, k]
            for e in range(self.n_experts):
                mask = expert_idx == e
                if not mask.any():
                    continue
                h = flat_x[mask]
                w1, w2, w3 = self.expert_w1[e], self.expert_w2[e], self.expert_w3[e]
                hidden = F.silu(F.linear(h, w1)) * F.linear(h, w3)
                out_e = self.dropout(F.linear(hidden, w2))
                flat_y[mask] = flat_y[mask] + w[mask].unsqueeze(-1) * out_e

        if self.training and self.aux_loss_coef > 0:
            probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            density = probs.mean(dim=(0, 1))
            slots = float(b * t * self.top_k)
            counts = torch.bincount(selected_experts.view(-1), minlength=self.n_experts).to(probs.dtype)
            load = counts / slots
            aux = self.n_experts * (density * load).sum()
            aux_loss = self.aux_loss_coef * aux
        else:
            aux_loss = x.new_zeros(())

        return y, aux_loss
