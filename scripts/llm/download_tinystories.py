"""
下载 TinyStories 文本数据到 ``data/llm/tinystories/``。

默认从 Hugging Face 镜像站拉取官方 ``TinyStories-train.txt`` / ``TinyStories-valid.txt``，
适合本仓库 MoELLM 语言建模实验（按行一条故事）。

用法（在仓库根目录）::

    python scripts/download_tinystories.py
    python scripts/download_tinystories.py --only valid
    python scripts/download_tinystories.py --mirror https://huggingface.co

环境变量 ``HF_ENDPOINT`` 可覆盖镜像根地址（例如 ``https://hf-mirror.com``）。
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 官方数据文件（论文与 roneneldan/TinyStories 数据集卡）
_FILES = {
    "train": "TinyStories-train.txt",
    "valid": "TinyStories-valid.txt",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_mirror() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def _download_url(mirror: str, filename: str) -> str:
    base = mirror.rstrip("/")
    if "huggingface.co" in base and "/datasets/" not in base:
        # 直连 HF：datasets/.../resolve/main/...
        return f"{base}/datasets/roneneldan/TinyStories/resolve/main/{filename}"
    # hf-mirror 等：同样路径结构
    return f"{base}/datasets/roneneldan/TinyStories/resolve/main/{filename}"


def _download_file(url: str, dest: Path, *, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    existing = tmp.stat().st_size if tmp.is_file() else 0
    headers: dict[str, str] = {"User-Agent": "magetronLM/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            total_i = int(total) + existing if total and existing else (int(total) if total else None)
            mode = "ab" if existing else "wb"
            with tmp.open(mode) as f:
                done = existing
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total_i:
                        pct = 100.0 * done / total_i
                        print(f"\r  {dest.name}: {done / 1e6:.1f} / {total_i / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r  {dest.name}: {done / 1e6:.1f} MB", end="", flush=True)
            print()
    except urllib.error.HTTPError as e:
        if e.code == 416 and tmp.is_file() and dest.is_file():
            return
        raise

    if dest.is_file():
        dest.unlink()
    tmp.replace(dest)


def main() -> None:
    p = argparse.ArgumentParser(description="下载 TinyStories 到 data/llm/tinystories/")
    p.add_argument(
        "--only",
        choices=("train", "valid", "all"),
        default="all",
        help="只下载指定划分（默认 train+valid）",
    )
    p.add_argument(
        "--mirror",
        default=_default_mirror(),
        help="HF 镜像或官方站根 URL（默认 hf-mirror.com 或 HF_ENDPOINT）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录，默认 <repo>/data/llm/tinystories",
    )
    args = p.parse_args()

    out_dir = args.out_dir or (_project_root() / "data" / "llm" / "tinystories")
    splits = list(_FILES.keys()) if args.only == "all" else [args.only]

    print(f"输出目录: {out_dir}")
    print(f"镜像: {args.mirror}")

    for split in splits:
        name = _FILES[split]
        dest = out_dir / name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[跳过] {name} 已存在 ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        url = _download_url(args.mirror, name)
        print(f"[下载] {url}")
        print(f"       -> {dest}")
        _download_file(url, dest)

    print("完成。训练集约 1.9 GB，验证集约 19 MB。")
    print("示例读取一行:")
    sample = (out_dir / _FILES["valid"])
    if sample.is_file():
        with sample.open(encoding="utf-8") as f:
            line = f.readline().strip()
        print(f"  {line[:120]}..." if len(line) > 120 else f"  {line}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断；可重新运行以断点续传（.part 文件）。", file=sys.stderr)
        sys.exit(130)
