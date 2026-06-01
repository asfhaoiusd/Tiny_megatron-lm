"""Convert trained VLM checkpoint to HuggingFace LLaVA format for vLLM serving.

vLLM natively supports LLaVA architecture models. This script:
1. Loads our trained components (CLIP, Projector, Qwen3+LoRA)
2. Saves in a format that vLLM's LLaVA implementation can load

Note: vLLM v0.11+ also supports Qwen3-VL natively. If LLaVA format
doesn't work, consider adapting to Qwen3-VL format instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _parse_args():
    p = argparse.ArgumentParser(description="Convert VLM to HuggingFace LLaVA format")
    p.add_argument("--model-path", type=Path, required=True, help="VLM checkpoint directory")
    p.add_argument("--output-path", type=Path, required=True, help="Output HF model directory")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = _parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    from vlm import VLMConfig, VLMForConditionalGeneration

    config = VLMConfig()
    model = VLMForConditionalGeneration(config)
    device = torch.device(args.device)
    model = model.to(device)

    # Load trained weights
    projector_path = args.model_path / "projector.pt"
    if projector_path.exists():
        model.projector.load_state_dict(torch.load(projector_path, map_location=device))
    else:
        projector_path = args.model_path / "vlm_full.pt"
        if projector_path.exists():
            model.load_state_dict(torch.load(projector_path, map_location=device), strict=False)

    # Load LoRA if present
    lora_path = args.model_path / "lora"
    if lora_path.exists():
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, str(lora_path))
        model.llm = model.llm.merge_and_unload()

    model.eval()

    print(f"Saving to HuggingFace format at {args.output_path}...")

    # Save LLM (Qwen3) + config in HF format
    model.llm.save_pretrained(str(args.output_path))
    model.tokenizer.save_pretrained(str(args.output_path))

    # Save vision tower (CLIP) separately
    vision_path = args.output_path / "vision_tower"
    vision_path.mkdir(exist_ok=True)
    model.vision_encoder.model.save_pretrained(str(vision_path))

    # Save projector
    torch.save(model.projector.state_dict(), args.output_path / "projector.pt")

    # Save VLM config for LLaVA format
    llava_config = {
        "architectures": ["LlavaForConditionalGeneration"],
        "model_type": "llava",
        "text_config": config.llm_model_name,
        "vision_config": config.vision_model_name,
        "image_token_index": model.image_token_id,
        "projector_hidden_act": "gelu",
        "vision_feature_layer": -1,
        "vision_feature_select_strategy": "default",
    }
    (args.output_path / "llava_config.json").write_text(json.dumps(llava_config, indent=2))

    print(f"""
Done! Model saved to {args.output_path}

To serve with vLLM:
    pip install vllm
    vllm serve {args.output_path} --port 8000

To test:
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{{"model": "{args.output_path.name}", "messages": [...]}}'

Or use Python API:
    from vllm import LLM
    llm = LLM(model="{args.output_path}")
""")

    tot = sum(p.numel() for p in model.parameters())
    print(f"Total model params: {tot / 1e9:.2f}B")


if __name__ == "__main__":
    main()
