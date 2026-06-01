"""VLM inference: generate text from image + prompt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image


def _parse_args():
    p = argparse.ArgumentParser(description="VLM generation")
    p.add_argument("--image", type=Path, required=True, help="Input image path")
    p.add_argument("--prompt", type=str, default="Describe this image in detail.")
    p.add_argument("--projector-ckpt", type=Path, help="Path to projector.pt")
    p.add_argument("--lora-path", type=Path, help="Path to LoRA adapter dir")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = _parse_args()

    print("Loading model...")
    from vlm import VLMConfig, VLMForConditionalGeneration, get_image_processor

    config = VLMConfig(use_cache=True)
    model = VLMForConditionalGeneration(config)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    if args.projector_ckpt:
        print(f"  Loading projector from {args.projector_ckpt}")
        model.projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))

    if args.lora_path:
        from peft import PeftModel
        print(f"  Loading LoRA from {args.lora_path}")
        model.llm = PeftModel.from_pretrained(model.llm, args.lora_path)
        model.llm = model.llm.merge_and_unload()

    image_processor = get_image_processor(config)

    # Load image
    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

    # Tokenize prompt
    full_prompt = f"{config.image_token}\n{args.prompt}"
    inputs = model.tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Generate
    torch.manual_seed(args.seed)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            top_p=0.9,
        )

    # Decode
    output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the new part (after prompt)
    prompt_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    response = output_text[len(prompt_text):].strip() if output_text.startswith(prompt_text) else output_text

    print("\n" + "=" * 50)
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    print(response)
    print("=" * 50)


if __name__ == "__main__":
    main()
