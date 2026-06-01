"""LoRA setup for LLM using peft (Qwen3 / SmolLM2 compatible)."""

from __future__ import annotations

from peft import LoraConfig, get_peft_model, TaskType

from .config import VLMConfig


def apply_lora_to_llm(
    config: VLMConfig,
    llm,
) -> "PeftModel":
    """Apply LoRA to the LLM component. Returns peft-wrapped model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
    )
    llm_lora = get_peft_model(llm, lora_config)
    llm_lora.print_trainable_parameters()
    return llm_lora


def get_trainable_params(model) -> dict[str, int]:
    """Count trainable params by component."""
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            comp = name.split(".")[0]
            stats[comp] = stats.get(comp, 0) + param.numel()
    return stats


def freeze_component(module, name: str = "") -> None:
    """Freeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_component(module) -> None:
    """Unfreeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = True
