"""
实验性 TinyStories 训练：GPT-2 分词器 + ~30M MoELLM。

在仓库根目录运行::

    python scripts/llm/train.py --max-steps 50

模型与 checkpoint 默认写入 ``checkpoints/moellm_30m/``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TinyStories + GPT-2 + ~30M MoELLM")
    p.add_argument("--output-dir", type=Path, default=None, help="默认 checkpoints/moellm_30m")
    p.add_argument("--train-file", type=Path, default=None)
    p.add_argument("--valid-file", type=Path, default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200, help="实验性短跑步数")
    p.add_argument("--max-train-stories", type=int, default=4000, help="限制读入故事数（加速试跑）")
    p.add_argument("--max-valid-stories", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--attention-type",
        default="mha",
        choices=("mha", "mqa", "mla"),
        help="MHA/MQA 用 CausalSelfAttention；MLA 用 DeepSeek-V2 风格 MLA",
    )
    return p.parse_args()



def main() -> None:
    root = _project_root()
    sys.path.insert(0, str(root))

    import torch
    import torch.nn.functional as F

    from llm import MoELLM
    from data.llm.config_30m import count_parameters, make_30m_config, preset_dir_for, save_config
    from data.llm.dataset import TinyStoriesDataLoader
    from device_util import pick_device

    args = _parse_args()
    out_dir = args.output_dir or preset_dir_for(args.attention_type)  # type: ignore[arg-type]
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    cfg = make_30m_config(args.attention_type)  # type: ignore[arg-type]
    save_config(cfg, out_dir)
    model = MoELLM(cfg).to(device)
    n_params = count_parameters(cfg)
    print(f"attention={args.attention_type} device={device} params={n_params:,} ({n_params / 1e6:.2f}M)")

    data = TinyStoriesDataLoader(
        train_path=args.train_file,
        valid_path=args.valid_file,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_train_stories=args.max_train_stories,
        max_valid_stories=args.max_valid_stories,
        seed=args.seed,
    )
    train_iter = iter(data.train)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    t0 = time.time()
    running = 0.0

    for step in range(1, args.max_steps + 1):
        try:
            inp, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(data.train)
            inp, tgt = next(train_iter)

        inp = inp.to(device)
        tgt = tgt.to(device)

        logits, aux, _ = model(inp)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            tgt.reshape(-1),
        ) + aux
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        running += loss.item()
        if step % args.log_every == 0:
            avg = running / args.log_every
            elapsed = time.time() - t0
            print(f"step={step} loss={avg:.4f} elapsed={elapsed:.1f}s")
            running = 0.0

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt = ckpt_dir / f"step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "config": cfg,
                    "loss": loss.item(),
                },
                ckpt,
            )
            print(f"saved {ckpt}")

    # quick valid pass
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for inp, tgt in data.valid:
            inp = inp.to(device)
            tgt = tgt.to(device)
            logits, aux, _ = model(inp)
            val_loss += (
                F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)) + aux
            ).item()
            val_batches += 1
            if val_batches >= 20:
                break
    val_avg = val_loss / max(val_batches, 1)
    metrics = {
        "attention_type": args.attention_type,
        "valid_loss": val_avg,
        "steps": args.max_steps,
        "params": n_params,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"valid_loss={val_avg:.4f} -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
