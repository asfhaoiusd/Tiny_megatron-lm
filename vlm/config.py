"""VLM configuration."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class VLMConfig:
    # Vision
    vision_model_name: str = "openai/clip-vit-large-patch14-336"
    vision_hidden_size: int = 1024
    vision_num_patches: int = 577  # 24*24 + 1 CLS

    # Projector
    projector_hidden: int = 2048  # intermediate dim, auto-set from LLM
    projector_type: Literal["mlp", "pixelshuffle"] = "mlp"

    # LLM
    llm_model_name: str = "Qwen/Qwen3-1.7B"
    llm_hidden_size: int = 2048  # Qwen3-1.7B: 28 layers, GQA 16Q/8KV, 32K ctx
    use_cache: bool = True

    # Image processing
    image_size: int = 336
    image_token: str = "<image>"

    # LoRA (stage 2)
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj", "o_proj")

    # Training
    max_seq_len: int = 2048  # total: image tokens + text tokens
    pad_token_id: int = 0  # will be auto-set from tokenizer in __post_init__

    def __post_init__(self):
        try:
            from transformers import AutoConfig
            llm_cfg = AutoConfig.from_pretrained(self.llm_model_name, trust_remote_code=True)
            self.llm_hidden_size = llm_cfg.hidden_size
            if hasattr(llm_cfg, "pad_token_id") and llm_cfg.pad_token_id is not None:
                self.pad_token_id = llm_cfg.pad_token_id
        except Exception:
            pass
        if self.projector_hidden == 2048 and self.llm_hidden_size != 2048:
            self.projector_hidden = self.llm_hidden_size
