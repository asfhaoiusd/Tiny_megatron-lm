"""Quick-start examples for magetronLM LLM and VLM pipelines.

Run from project root:
    python quick_start.py llm     # LLM example
    python quick_start.py vlm     # VLM example (requires pretrained models)
"""

import sys
from pathlib import Path


def llm_example() -> None:
    """Create a ~30M MoELLM with MLA attention and run a forward pass."""
    import torch
    from llm import MoELLM
    from data.llm.config_30m import make_30m_config

    cfg = make_30m_config("mla")
    model = MoELLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 32))

    logits, aux_loss, _ = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        ids[:, 1:].reshape(-1),
    ) + aux_loss

    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"Forward loss: {loss.item():.4f}")
    print("LLM pipeline OK.")


def vlm_example() -> None:
    """Load VLM config and print architecture info. Requires internet for model download."""
    from vlm import VLMConfig, VLMForConditionalGeneration

    config = VLMConfig()
    print(f"Vision: {config.vision_model_name}")
    print(f"LLM:    {config.llm_model_name}")
    print(f"Hidden: vision={config.vision_hidden_size}, llm={config.llm_hidden_size}")
    print("VLM config OK. Full model loading requires GPU + pretrained weights.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "llm"
    {"llm": llm_example, "vlm": vlm_example}[mode]()
