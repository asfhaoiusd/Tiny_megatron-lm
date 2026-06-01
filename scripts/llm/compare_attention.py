"""
对比 MHA / MQA / MLA：训练 loss 与每步耗时。

在仓库根目录::

    python scripts/llm/compare_attention.py --max-steps 50 --device cuda
    python scripts/llm/compare_attention.py --attention-types mha mla --max-steps 100
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
    p = argparse.ArgumentParser(description="Compare MHA / MQA / MLA on TinyStories ~30M")
    p.add_argument(
        "--attention-types",
        nargs="+",
        default=["mha", "mqa", "mla"],
        choices=("mha", "mqa", "mla"),
    )
    p.add_argument("--output-dir", type=Path, default=None, help="默认 checkpoints/attention_compare/")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-train-stories", type=int, default=4000)
    p.add_argument("--max-valid-stories", type=int, default=800)
    p.add_argument("--valid-batches", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=5, help="计时前预热步数")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _train_one(
    attention_type: str,
    args: argparse.Namespace,
    *,
    device,
    train_iter_factory,
    valid_loader,
    out_dir: Path,
) -> dict:
    import torch
    import torch.nn.functional as F

    from llm import MoELLM
    from data.llm.config_30m import count_parameters, make_30m_config, save_config

    cfg = make_30m_config(attention_type)  # type: ignore[arg-type]
    run_dir = out_dir / attention_type
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir)

    torch.manual_seed(args.seed)
    model = MoELLM(cfg).to(device)
    n_params = count_parameters(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_iter = train_iter_factory()

    def _next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = train_iter_factory()
            return next(train_iter)

    model.train()
    step_times: list[float] = []
    running_loss = 0.0

    for step in range(1, args.max_steps + 1):
        inp, tgt = _next_batch()
        inp = inp.to(device)
        tgt = tgt.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits, aux, _ = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)) + aux
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if step > args.warmup_steps:
            step_times.append(dt_ms)

        running_loss += loss.item()
        if step % max(args.max_steps // 5, 1) == 0:
            print(f"  [{attention_type}] step={step}/{args.max_steps} loss={loss.item():.4f}")

    train_loss = running_loss / args.max_steps
    ms_per_step = sum(step_times) / max(len(step_times), 1)

    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for inp, tgt in valid_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            logits, aux, _ = model(inp)
            val_loss += (
                F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1)) + aux
            ).item()
            val_batches += 1
            if val_batches >= args.valid_batches:
                break
    valid_avg = val_loss / max(val_batches, 1)

    result = {
        "attention_type": attention_type,
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "train_loss": round(train_loss, 4),
        "valid_loss": round(valid_avg, 4),
        "ms_per_step": round(ms_per_step, 2),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"  [{attention_type}] params={n_params/1e6:.2f}M "
        f"train_loss={train_loss:.4f} valid_loss={valid_avg:.4f} "
        f"ms/step={ms_per_step:.1f}"
    )
    return result


def main() -> None:
    root = _project_root()
    sys.path.insert(0, str(root))

    import torch

    from data.llm.dataset import TinyStoriesDataLoader
    from device_util import pick_device

    args = _parse_args()
    out_dir = args.output_dir or (root / "checkpoints" / "attention_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    data = TinyStoriesDataLoader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_train_stories=args.max_train_stories,
        max_valid_stories=args.max_valid_stories,
        seed=args.seed,
    )

    def train_iter_factory():
        return iter(data.train)

    results: list[dict] = []
    for attn in args.attention_types:
        print(f"\n=== attention={attn} ===")
        results.append(
            _train_one(
                attn,
                args,
                device=device,
                train_iter_factory=train_iter_factory,
                valid_loader=data.valid,
                out_dir=out_dir,
            )
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"{'type':<6} {'params(M)':>10} {'train_loss':>12} {'valid_loss':>12} {'ms/step':>10}")
    for r in results:
        print(
            f"{r['attention_type']:<6} {r['params_m']:>10.3f} "
            f"{r['train_loss']:>12.4f} {r['valid_loss']:>12.4f} {r['ms_per_step']:>10.1f}"
        )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
