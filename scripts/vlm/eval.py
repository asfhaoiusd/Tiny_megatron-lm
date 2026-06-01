"""VLM evaluation using lmms-eval framework.

Usage:
    python scripts/vlm/eval.py --model-path checkpoints/vlm_stage2/final --tasks mmbench,mmstar

Requires: pip install lmms-eval
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


BENCHMARK_TASKS = {
    "mmbench": "MMBench (通用多模态理解)",
    "mmstar": "MMStar (多模态推理)",
    "textvqa": "TextVQA (文字识别)",
    "mme": "MME (综合感知+认知)",
    "chartqa": "ChartQA (图表理解)",
    "scienceqa": "ScienceQA (科学推理)",
    "docvqa": "DocVQA (文档理解)",
    "pope": "POPE (幻觉检测)",
    "seedbench": "SEEDBench (多模态综合)",
}


def _parse_args():
    p = argparse.ArgumentParser(description="VLM evaluation with lmms-eval")
    p.add_argument("--model-path", type=Path, required=True, help="Path to trained VLM checkpoint dir")
    p.add_argument(
        "--tasks", type=str, default="mmbench,mmstar,textvqa,mme",
        help="Comma-separated benchmark names",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    p.add_argument("--list-tasks", action="store_true", help="Print available tasks and exit")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.list_tasks:
        print("Available benchmarks:")
        for name, desc in BENCHMARK_TASKS.items():
            print(f"  {name:<15} — {desc}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",")]

    print("=" * 50)
    print("VLM Evaluation")
    print(f"Model: {args.model_path}")
    print(f"Tasks: {tasks}")
    print("=" * 50)

    # lmms-eval uses llamafactory-compatible model names
    # For LLaVA-style models, we use the `llava` model type
    model_type = "llava"

    # Build lmms-eval command
    # lmms-eval needs a special model adapter for custom models
    # The simplest path: save model in HuggingFace LLaVA format, then eval

    import subprocess

    for task in tasks:
        print(f"\n--- Running {task} ---")
        cmd = [
            sys.executable, "-m", "lmms_eval",
            "--model", model_type,
            "--model_args", f"pretrained={args.model_path}",
            "--tasks", task,
            "--batch_size", str(args.batch_size),
            "--output_path", str(args.output_dir),
            "--log_samples",
        ]
        try:
            subprocess.run(cmd, check=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"})
        except subprocess.CalledProcessError:
            print(f"  Warning: {task} failed — model may not be in expected format.")
            print(f"  Try running with custom eval: python {__file__} --help")

    # Summarize
    result_files = list(args.output_dir.glob("**/results_*.json"))
    if result_files:
        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        summary = {}
        for rf in sorted(result_files):
            data = json.loads(rf.read_text())
            for task_name, result in data.get("results", {}).items():
                score = result.get("accuracy,none") or result.get("exact_match,none") or result.get("score", "?")
                if isinstance(score, float):
                    score = f"{score:.3f}"
                summary[task_name] = score
                print(f"  {task_name}: {score}")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    else:
        print("\nNo results found. Ensure lmms-eval ran successfully.")


if __name__ == "__main__":
    main()
