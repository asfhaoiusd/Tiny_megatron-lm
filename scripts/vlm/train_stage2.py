"""Stage 2: LoRA SFT — instruction-tune the VLM.

Trains: Projector + LoRA on LLM
Freezes: CLIP ViT
Data: LLaVA-Instruct + Chinese VQA data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset


def _parse_args():
    p = argparse.ArgumentParser(description="VLM Stage 2: LoRA instruction tuning")
    p.add_argument("--data-json", type=Path, action="append", default=[], help="Instruct JSON(s), can repeat")
    p.add_argument("--image-dir", type=Path, default="", help="Base dir for images")
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/vlm_stage2"))
    p.add_argument("--projector-ckpt", type=Path, required=True, help="Path to stage1 projector.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    if not args.data_json:
        print("Error: at least one --data-json required")
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 2: LoRA Instruction Tuning")
    print("=" * 60)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Setup model ---
    print("\n[1/5] Loading model...")
    from vlm import VLMConfig, VLMForConditionalGeneration, get_image_processor, apply_lora_to_llm
    from vlm.lora_utils import freeze_component, unfreeze_component, get_trainable_params

    config = VLMConfig()
    model = VLMForConditionalGeneration(config)
    model = model.to(device)
    image_processor = get_image_processor(config)

    # Load trained projector
    print(f"  Loading projector from {args.projector_ckpt}")
    model.projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))

    # Freeze CLIP
    freeze_component(model.vision_encoder, "vision_encoder")

    # Apply LoRA to LLM
    print("  Applying LoRA to LLM...")
    model.llm = apply_lora_to_llm(config, model.llm)
    model.llm.enable_input_require_grads()  # For gradient checkpointing

    # Unfreeze projector (re-verify)
    unfreeze_component(model.projector)

    stats = get_trainable_params(model)
    total_trainable = sum(stats.values())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")

    # --- Setup data ---
    print("\n[2/5] Loading data...")
    from data.vlm_dataset import LLaVADataset

    datasets = []
    for data_json in args.data_json:
        ds = LLaVADataset(
            json_path=data_json,
            tokenizer=model.tokenizer,
            image_processor=image_processor,
            image_base_dir=args.image_dir,
            max_seq_len=args.max_seq_len,
            image_token_id=model.image_token_id,
        )
        datasets.append(ds)
        print(f"  {data_json}: {len(ds)} samples")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    from data.vlm.dataset import collate_vlm_batch

    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: collate_vlm_batch(batch, pad_token_id=model.tokenizer.pad_token_id or 0),
        pin_memory=True,
    )
    print(f"  Total: {len(combined)} samples, {len(loader)} batches")

    # --- Setup optimizer ---
    print("\n[3/5] Setting up optimizer...")
    trainable = [p for p in model.parameters() if p.requires_grad]
    # Grouped LR: higher for projector, lower for LoRA
    projector_params = [p for n, p in model.named_parameters() if "projector" in n and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    other_params = [p for p in trainable if p not in projector_params and p not in lora_params]

    optimizer = torch.optim.AdamW([
        {"params": projector_params, "lr": args.lr * 10},
        {"params": lora_params, "lr": args.lr},
        {"params": other_params, "lr": args.lr},
    ], lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        warmup = int(args.max_steps * args.warmup_ratio)
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, args.max_steps - warmup)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Train ---
    print("\n[4/5] Training...")
    model.train()
    # Enable gradient checkpointing for memory
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()

    t0 = time.time()
    running_loss = 0.0
    step = 0
    data_iter = iter(loader)

    while step < args.max_steps:
        step += 1

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
            model.llm.save_pretrained(ckpt_path / "lora")
            print(f"  saved -> {ckpt_path}")

    # --- Final save ---
    print("\n[5/5] Saving final model...")
    final_path = args.output_dir / "final"
    final_path.mkdir(exist_ok=True)
    torch.save(model.projector.state_dict(), final_path / "projector.pt")
    model.llm.save_pretrained(final_path / "lora")
    # Save full model for inference
    torch.save(model.state_dict(), final_path / "vlm_full.pt")

    print(f"Done! Saved to {final_path}")
    print(f"Total time: {(time.time() - t0) / 3600:.1f} hours")


if __name__ == "__main__":
    main()
