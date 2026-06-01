"""Stage 1: Projector Warmup — align CLIP features to LLM embedding space.

Freezes: CLIP ViT + LLM
Trains: Projector only
Data: LLaVA-Pretrain-558K (image-caption pairs)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _parse_args():
    p = argparse.ArgumentParser(description="VLM Stage 1: Projector alignment")
    p.add_argument("--data-json", type=Path, required=True, help="LLaVA-Pretrain JSON")
    p.add_argument("--image-dir", type=Path, default="", help="Base dir for images")
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/vlm_stage1"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 1: Projector Warmup (CLIP + LLM frozen)")
    print("=" * 60)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Setup model ---
    print("\n[1/4] Loading model components...")
    from vlm import VLMConfig, VLMForConditionalGeneration, get_image_processor

    config = VLMConfig()
    model = VLMForConditionalGeneration(config)
    model = model.to(device)
    image_processor = get_image_processor(config)

    # Freeze CLIP + LLM, only train projector
    from vlm.lora_utils import freeze_component, get_trainable_params

    freeze_component(model.vision_encoder, "vision_encoder")
    freeze_component(model.llm, "llm")
    # Projector params stay trainable (default)

    stats = get_trainable_params(model)
    total_trainable = sum(stats.values())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")
    print(f"  By component: {stats}")

    # --- Setup data ---
    print("\n[2/4] Loading data...")
    from data.vlm.dataset import create_dataloader

    loader = create_dataloader(
        json_path=args.data_json,
        tokenizer=model.tokenizer,
        image_processor=image_processor,
        image_base_dir=args.image_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        image_token_id=model.image_token_id,
        shuffle=True,
        num_workers=4,
    )
    print(f"  {len(loader.dataset)} samples, {len(loader)} batches")

    # --- Setup optimizer ---
    print("\n[3/4] Setting up optimizer...")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        warmup = int(args.max_steps * args.warmup_ratio)
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, args.max_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Train ---
    print("\n[4/4] Training...")
    model.train()
    t0 = time.time()
    running_loss = 0.0
    step = 0
    data_iter = iter(loader)

    while step < args.max_steps:
        step += 1

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        if not batch:
            continue

        pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        loss = outputs["loss"] / args.grad_accum
        loss.backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * args.grad_accum

        if step % args.log_every == 0:
            avg = running_loss / args.log_every
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step={step}/{args.max_steps} loss={avg:.4f} lr={lr_now:.2e} elapsed={elapsed:.0f}s")
            running_loss = 0.0

        if step % args.save_every == 0:
            ckpt_path = args.output_dir / f"step_{step}"
            ckpt_path.mkdir(exist_ok=True)
            torch.save(model.projector.state_dict(), ckpt_path / "projector.pt")
            print(f"  saved -> {ckpt_path}")

    # --- Final save ---
    final_path = args.output_dir / "final"
    final_path.mkdir(exist_ok=True)
    torch.save(model.projector.state_dict(), final_path / "projector.pt")
    print(f"\nDone! Projector saved to {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
