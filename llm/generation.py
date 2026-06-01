"""Greedy decoding using KV cache (inference-only helpers)."""

import torch

from .blocks import MoELLM


@torch.inference_mode()
def greedy_decode(
    model: MoELLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Append up to `max_new_tokens` tokens by greedy argmax on the last position.
    Uses `use_cache` internally for efficiency.
    """
    if max_new_tokens <= 0:
        return input_ids

    logits, _, past = model(input_ids, use_cache=True)
    next_token = logits[:, -1:, :].argmax(dim=-1)
    generated = [input_ids, next_token]

    if eos_token_id is not None and (next_token == eos_token_id).all():
        return torch.cat(generated, dim=1)

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
        logits, _, past = model(next_token, past_key_values=past, use_cache=True)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token)

    return torch.cat(generated, dim=1)
