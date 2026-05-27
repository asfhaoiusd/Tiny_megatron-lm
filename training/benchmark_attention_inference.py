"""
对比 MHA / MQA / MLA 的推理速度（prefill + 增量 decode）。

在仓库根目录::

    python training/benchmark_attention_inference.py --device cuda
    python training/benchmark_attention_inference.py --prefill-len 128 --decode-tokens 64

输出: ``pre_model/attention_inference/summary.json``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark MHA / MQA / MLA inference speed")
    p.add_argument(
        "--attention-types",
        nargs="+",
        default=["mha", "mqa", "mla"],
        choices=("mha", "mqa", "mla"),
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=1, help="推理 batch，默认 1")
    p.add_argument("--prefill-len", type=int, default=256, help="prefill 序列长度")
    p.add_argument("--decode-tokens", type=int, default=128, help="增量 decode 步数")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20, help="计时重复次数")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _cache_bytes_per_token(cfg) -> int:
    """每层每 token 的 KV cache 体积（bytes，bf16=2）。"""
    elem = 2
    h = cfg.n_heads
    hd = cfg.head_dim
    if cfg.attention_type == "mla":
        per_layer = elem * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
    elif cfg.attention_type == "mqa":
        per_layer = elem * 2 * cfg.n_kv_heads * hd
    else:  # mha
        per_layer = elem * 2 * h * hd
    return cfg.n_layers * per_layer


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench_prefill(model, input_ids, *, warmup: int, iters: int, device) -> float:
    import torch

    with torch.inference_mode():
        for _ in range(warmup):
            model(input_ids, use_cache=True)
        _sync(device)

        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                starts[i].record()
                model(input_ids, use_cache=True)
                ends[i].record()
            _sync(device)
            return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters

        times: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(input_ids, use_cache=True)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        return sum(times) / len(times)


def _bench_decode(
    model,
    input_ids,
    *,
    decode_tokens: int,
    warmup: int,
    iters: int,
    device,
) -> float:
    import torch

    with torch.inference_mode():
        _, _, past = model(input_ids, use_cache=True)
        probe = input_ids[:, -1:]

        for _ in range(warmup):
            model(probe, past_key_values=past, use_cache=True)
        _sync(device)

        if device.type == "cuda":
            total_ms = 0.0
            count = 0
            for _ in range(iters):
                _, _, past_run = model(input_ids, use_cache=True)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                tok = probe
                start.record()
                for _ in range(decode_tokens):
                    logits, _, past_run = model(tok, past_key_values=past_run, use_cache=True)
                    tok = logits[:, -1:, :].argmax(dim=-1)
                end.record()
                _sync(device)
                total_ms += start.elapsed_time(end)
                count += decode_tokens
            return total_ms / count

        times: list[float] = []
        for _ in range(iters):
            _, _, past_run = model(input_ids, use_cache=True)
            t0 = time.perf_counter()
            tok = probe
            for _ in range(decode_tokens):
                logits, _, past_run = model(tok, past_key_values=past_run, use_cache=True)
                tok = logits[:, -1:, :].argmax(dim=-1)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000.0 / decode_tokens)
        return sum(times) / len(times)


def _bench_one(attention_type: str, args: argparse.Namespace, device) -> dict:
    import torch

    from model import MoELLM
    from pre_model.config_30m import count_parameters, make_30m_config, save_config

    cfg = make_30m_config(attention_type)  # type: ignore[arg-type]
    run_dir = (args.output_dir or (_project_root() / "pre_model" / "attention_inference")) / attention_type
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir)

    torch.manual_seed(args.seed)
    model = MoELLM(cfg).to(device).eval()
    input_ids = torch.randint(
        0,
        min(cfg.vocab_size, 10000),
        (args.batch_size, args.prefill_len),
        device=device,
    )

    prefill_ms = _bench_prefill(
        model, input_ids, warmup=args.warmup, iters=args.iters, device=device
    )
    decode_ms = _bench_decode(
        model,
        input_ids,
        decode_tokens=args.decode_tokens,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    cache_b = _cache_bytes_per_token(cfg)
    n_params = count_parameters(cfg)
    result = {
        "attention_type": attention_type,
        "params_m": round(n_params / 1e6, 3),
        "prefill_len": args.prefill_len,
        "decode_tokens": args.decode_tokens,
        "batch_size": args.batch_size,
        "prefill_ms": round(prefill_ms, 3),
        "decode_ms_per_token": round(decode_ms, 3),
        "tokens_per_sec_decode": round(1000.0 / decode_ms, 1) if decode_ms > 0 else 0.0,
        "kv_cache_bytes_per_token": cache_b,
        "kv_cache_kb_per_token": round(cache_b / 1024, 2),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"  [{attention_type}] prefill={prefill_ms:.2f}ms "
        f"decode={decode_ms:.2f}ms/tok ({result['tokens_per_sec_decode']:.1f} tok/s) "
        f"cache={result['kv_cache_kb_per_token']:.2f}KB/tok"
    )
    return result


def main() -> None:
    root = _project_root()
    sys.path.insert(0, str(root))

    import torch

    from training.device_util import pick_device

    args = _parse_args()
    out_dir = args.output_dir or (root / "pre_model" / "attention_inference")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    print(
        f"device={device} batch={args.batch_size} prefill={args.prefill_len} "
        f"decode={args.decode_tokens} warmup={args.warmup} iters={args.iters}"
    )

    results: list[dict] = []
    for attn in args.attention_types:
        print(f"\n=== {attn} ===")
        results.append(_bench_one(attn, args, device))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Inference Summary ===")
    print(
        f"{'type':<6} {'prefill(ms)':>12} {'decode(ms/t)':>14} "
        f"{'tok/s':>8} {'KV(KB/t)':>10}"
    )
    for r in results:
        print(
            f"{r['attention_type']:<6} {r['prefill_ms']:>12.3f} {r['decode_ms_per_token']:>14.3f} "
            f"{r['tokens_per_sec_decode']:>8.1f} {r['kv_cache_kb_per_token']:>10.2f}"
        )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
