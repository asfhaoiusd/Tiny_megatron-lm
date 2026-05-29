"""train-watch helper: parse metrics.json / summary.json and print a comparison table with anomaly flags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


_METRIC_FILES = ["summary.json", "metrics.json"]

# Training columns
_TRAIN_COLS: list[tuple[str, str, int]] = [
    ("attention_type", "Type", 6),
    ("params_m", "Params(M)", 10),
    ("train_loss", "TrainLoss", 10),
    ("valid_loss", "ValidLoss", 10),
    ("ms_per_step", "ms/step", 9),
    ("steps", "Steps", 7),
    ("batch_size", "Batch", 6),
    ("seq_len", "SeqLen", 6),
]

# Inference columns
_INFER_COLS: list[tuple[str, str, int]] = [
    ("attention_type", "Type", 6),
    ("params_m", "Params(M)", 10),
    ("prefill_len", "Prefill", 8),
    ("decode_tokens", "DecTok", 7),
    ("prefill_ms", "PrefillMs", 10),
    ("decode_ms_per_token", "DecMsTok", 10),
    ("tokens_per_sec_decode", "Tok/s", 8),
    ("kv_cache_kb_per_token", "KVcacheKB", 10),
]


def _find_metrics_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in _METRIC_FILES:
        found.extend(root.rglob(pattern))
    return sorted(found)


def _experiment_name(path: Path) -> str:
    """Derive experiment name from path. E.g. pre_model/attention_compare/mha → 'attention_compare'."""
    parts = path.parts
    # Look for the directory immediately under pre_model/ or similar
    if "attention_compare" in parts:
        return "attention_compare"
    if "attention_inference" in parts:
        return "attention_inference"
    return parts[-2] if len(parts) >= 2 else path.parent.name


def _load_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    exp = _experiment_name(path)
    for e in items:
        if "attention_type" not in e:
            e["attention_type"] = path.parent.name
        if "params_m" not in e and "params" in e:
            e["params_m"] = round(e["params"] / 1e6, 3)
        e["_experiment"] = exp
    return items


def _is_training(rows: list[dict]) -> bool:
    """Detect if these are training metrics (have valid_loss/train_loss/ms_per_step)."""
    return any("valid_loss" in r or "train_loss" in r or "ms_per_step" in r for r in rows)


def _is_inference(rows: list[dict]) -> bool:
    return any("prefill_ms" in r or "decode_ms_per_token" in r for r in rows)


def _fmt(val, w: int) -> str:
    if val is None:
        return "-".ljust(w)
    if isinstance(val, float):
        return f"{val:.3f}".ljust(w)
    if isinstance(val, int):
        return str(val).ljust(w)
    return str(val)[:w].ljust(w)


def _print_section(title: str, rows: list[dict], cols: list[tuple[str, str, int]]) -> None:
    active = [(k, lbl, w) for k, lbl, w in cols if any(k in r for r in rows)]
    if not active or not rows:
        return

    print(f"\n{_bold(title)}")
    header_line = "  ".join(lbl.ljust(w) for _, lbl, w in active)
    print(header_line)
    print("-" * len(header_line))

    for r in rows:
        parts = [_fmt(r.get(key), w) for key, _, w in active]
        print("  ".join(parts))

    # Mark best — skip identity columns and params
    _SKIP_BEST = {"attention_type", "params_m", "prefill_len", "decode_tokens", "batch_size", "seq_len", "max_steps", "steps"}
    _HIGHER_BETTER = {"tokens_per_sec_decode"}
    if len(rows) > 1:
        for col_name in [k for k, _, _ in active if k not in _SKIP_BEST]:
            vals = [(i, r.get(col_name)) for i, r in enumerate(rows) if isinstance(r.get(col_name), (int, float))]
            if vals:
                if col_name in _HIGHER_BETTER:
                    best_idx = max(vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(vals, key=lambda x: x[1])[0]
                print(f"  * best {col_name}: {_green(rows[best_idx].get('attention_type', '?'))}")


def _check_anomalies(train_rows: list[dict], args) -> list[str]:
    warnings: list[str] = []
    for r in train_rows:
        name = r.get("attention_type", "?")
        vl = r.get("valid_loss")
        tl = r.get("train_loss")
        ms = r.get("ms_per_step")
        if vl is not None and args.alert_loss_above is not None and vl > args.alert_loss_above:
            warnings.append(f"{_red('[HIGH LOSS]')} {name}: valid_loss={vl:.3f} > {args.alert_loss_above}")
        if tl is not None and vl is not None and tl < vl * 0.3:
            warnings.append(f"{_yellow('[OVERFIT?]')} {name}: train_loss={tl:.3f} << valid_loss={vl:.3f}")
        if ms is not None and args.alert_time_above is not None and ms > args.alert_time_above:
            warnings.append(f"{_yellow('[SLOW]')} {name}: ms/step={ms:.1f} > {args.alert_time_above}")
    if len(train_rows) >= 2:
        param_counts = [r.get("params", 0) for r in train_rows]
        if max(param_counts) > 0:
            spread = (max(param_counts) - min(param_counts)) / max(param_counts)
            if spread > 0.05:
                warnings.append(
                    f"{_yellow('[UNFAIR]')} param spread={spread:.1%} > 5% — comparison may be invalid"
                )
    return warnings


def main() -> None:
    p = argparse.ArgumentParser(description="train-watch: monitor LLM training metrics")
    p.add_argument("--summary", type=Path, help="path to summary.json")
    p.add_argument("--metrics", type=Path, action="append", help="path to individual metrics.json")
    p.add_argument("--scan-dir", type=Path, default=None, help="dir to scan (default: pre_model)")
    p.add_argument("--alert-loss-above", type=float, default=10.0)
    p.add_argument("--alert-time-above", type=float, default=200.0)
    p.add_argument("--json", action="store_true", help="output as JSON")
    args = p.parse_args()

    entries: list[dict] = []

    if args.summary:
        entries.extend(_load_entries(args.summary))
    if args.metrics:
        for mp in args.metrics:
            entries.extend(_load_entries(mp))
    if not args.summary and not args.metrics:
        scan_root = args.scan_dir or (Path(__file__).resolve().parents[1] / "pre_model")
        files = _find_metrics_files(scan_root)
        if not files:
            print(f"No metrics.json or summary.json found under {scan_root}")
            sys.exit(1)
        print(f"Scanning {scan_root} — {len(files)} file(s) found\n")
        for fp in files:
            try:
                entries.extend(_load_entries(fp))
            except Exception:
                print(f"  skip: {fp} (parse error)")

    if not entries:
        print("No valid metrics found.")
        sys.exit(1)

    # Group by experiment, dedup within each group
    experiments: dict[str, dict[str, dict]] = {}
    for e in entries:
        exp = e.pop("_experiment", "?")
        key = e.get("attention_type", "?")
        if exp not in experiments:
            experiments[exp] = {}
        if key not in experiments[exp] or len(e) > len(experiments[exp][key]):
            experiments[exp][key] = e

    if args.json:
        all_rows = []
        for exp_entries in experiments.values():
            all_rows.extend(exp_entries.values())
        all_rows.sort(key=lambda r: r.get("attention_type", ""))
        json.dump(all_rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    all_warnings: list[str] = []
    for exp_name, exp_entries in experiments.items():
        rows = sorted(exp_entries.values(), key=lambda r: r.get("attention_type", ""))

        train_rows = [r for r in rows if _is_training([r])]
        infer_rows = [r for r in rows if _is_inference([r])]
        other = [r for r in rows if r not in train_rows and r not in infer_rows]

        label = exp_name.replace("_", " ").title()
        if train_rows:
            _print_section(f"[{label}] Training", train_rows, _TRAIN_COLS)
        if infer_rows:
            _print_section(f"[{label}] Inference", infer_rows, _INFER_COLS)
        if other:
            _print_section(f"[{label}] Other", other, _TRAIN_COLS)

        all_warnings.extend(_check_anomalies(train_rows, args))

    if all_warnings:
        print(f"\n{_bold('Warnings:')}")
        for w in all_warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
