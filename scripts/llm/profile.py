"""
用 PyTorch Profiler 分析 MoELLM 训练瓶颈（重点：GPU kernel 耗时）。

在仓库根目录::

    python scripts/llm/profile.py --device cuda
    python scripts/llm/profile.py --device cuda --warmup 3 --active 10

输出（``checkpoints/moellm_30m/profiler/``）::

  - ``trace.json`` — Chrome trace（chrome://tracing）
  - ``summary.txt`` — Kineto 算子表 + 模块表
  - ``cuda_timer.txt`` — CUDA Event 分段计时（Kineto 无 GPU 数据时仍可用）
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile MoELLM TinyStories training step")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3, help="profiler schedule: warmup steps")
    p.add_argument("--active", type=int, default=10, help="profiler schedule: active steps")
    p.add_argument("--repeat", type=int, default=1, help="profiler schedule: repeat cycles")
    p.add_argument("--timer-iters", type=int, default=20, help="CUDA Event 计时迭代次数")
    p.add_argument("--max-train-stories", type=int, default=500)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-stack", action="store_true", help="记录 Python 调用栈（trace 更大）")
    return p.parse_args()


def _train_step(model, optim, inp, tgt, vocab_size, device):
    import torch
    import torch.nn.functional as F

    inp = inp.to(device, non_blocking=True)
    tgt = tgt.to(device, non_blocking=True)

    logits, aux, _ = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1)) + aux
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    return loss


def _cuda_event_ms(start, end) -> float:
    return start.elapsed_time(end)


def _run_cuda_timer(model, optim, train_iter, vocab_size, device, *, iters: int) -> str:
    """用 CUDA Event 分段计时，不依赖 CUPTI。"""
    import torch
    import torch.nn.functional as F

    def _next_batch():
        try:
            return next(train_iter)
        except StopIteration:
            return None

    # 预热
    for _ in range(3):
        batch = _next_batch()
        if batch is None:
            break
        inp, tgt = batch
        _train_step(model, optim, inp, tgt, vocab_size, device)
    torch.cuda.synchronize()

    totals = {"h2d": 0.0, "forward": 0.0, "loss": 0.0, "backward": 0.0, "optim": 0.0, "total": 0.0}
    n = 0

    for _ in range(iters):
        batch = _next_batch()
        if batch is None:
            break
        inp, tgt = batch

        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)
        e4 = torch.cuda.Event(enable_timing=True)
        e5 = torch.cuda.Event(enable_timing=True)

        e0.record()
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        e1.record()

        logits, aux, _ = model(inp)
        e2.record()

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1)) + aux
        e3.record()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        e4.record()

        optim.step()
        e5.record()
        torch.cuda.synchronize()

        totals["h2d"] += _cuda_event_ms(e0, e1)
        totals["forward"] += _cuda_event_ms(e1, e2)
        totals["loss"] += _cuda_event_ms(e2, e3)
        totals["backward"] += _cuda_event_ms(e3, e4)
        totals["optim"] += _cuda_event_ms(e4, e5)
        totals["total"] += _cuda_event_ms(e0, e5)
        n += 1

    if n == 0:
        return "CUDA Event 计时：无有效 batch\n"

    lines = [
        "=== CUDA Event 分段计时（毫秒/步，不依赖 CUPTI） ===",
        f"iterations={n}",
        "",
    ]
    for key in ("h2d", "forward", "loss", "backward", "optim", "total"):
        ms = totals[key] / n
        pct = 100.0 * totals[key] / totals["total"] if totals["total"] else 0.0
        lines.append(f"  {key:10s}  {ms:7.2f} ms  ({pct:5.1f}%)")
    lines.append("")
    lines.append("解读: forward+backward 通常占大头；h2d 高则加 pin_memory；loss 高则考虑 fused CE。")
    return "\n".join(lines)


def _event_cuda_us(evt) -> int:
    """兼容不同 PyTorch 版本的 GPU 耗时字段（微秒）。"""
    for attr in ("device_time_total", "self_device_time_total", "device_time", "cuda_time"):
        val = getattr(evt, attr, None)
        if val:
            return int(val)
    return 0


def _kineto_cuda_total_ms(prof) -> float:
    return sum(_event_cuda_us(evt) for evt in prof.key_averages()) / 1000.0


def _cuda_sort_key(device_type: str) -> str:
    # 2.12 nightly 表格排序键为 device_time_total；旧版为 cuda_time_total
    return "device_time_total" if device_type == "cuda" else "cpu_time_total"


def _cuda_self_sort_key(device_type: str) -> str:
    return "self_device_time_total" if device_type == "cuda" else "self_cpu_time_total"


def _build_kineto_summary(prof, device_type: str) -> str:
    sort_key = _cuda_sort_key(device_type)
    self_key = _cuda_self_sort_key(device_type)
    time_label = "CUDA" if device_type == "cuda" else "CPU"
    chunks: list[str] = []

    cuda_ms = _kineto_cuda_total_ms(prof) if device_type == "cuda" else 0.0
    if device_type == "cuda":
        if cuda_ms > 0:
            chunks.append(f"Kineto 已采集到 GPU 时间，合计约 {cuda_ms:.2f} ms（active 步累计）")
        else:
            chunks.append(
                "Kineto 未采集到 GPU kernel 时间（本机 CUPTI 未生效，Windows 上常见）。"
                " **请以 cuda_timer.txt 为准**；trace.json 的 GPU 轨道也可能为空。"
            )
        chunks.append("")

    header = f"=== Top ops by {time_label} time (Kineto) ==="
    chunks.append(header)
    chunks.append(prof.key_averages().table(sort_by=sort_key, row_limit=30))
    chunks.append("")

    header = f"=== Top ops by self {time_label} time (Kineto) ==="
    chunks.append(header)
    chunks.append(prof.key_averages().table(sort_by=self_key, row_limit=25))
    chunks.append("")

    if device_type == "cuda" and cuda_ms > 0:
        header = "=== Top ops by CUDA time (grouped by input shape) ==="
        chunks.append(header)
        chunks.append(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="device_time_total", row_limit=20
            )
        )
        chunks.append("")

    header = "=== Top modules by CPU time ==="
    chunks.append(header)
    chunks.append(
        prof.key_averages(group_by_stack_n=0).table(sort_by="cpu_time_total", row_limit=15)
    )
    chunks.append("")

    if device_type == "cuda":
        # 仅列出有 CUDA 耗时的 op
        cuda_ops = sorted(
            [e for e in prof.key_averages() if _event_cuda_us(e) > 0],
            key=_event_cuda_us,
            reverse=True,
        )[:25]
        header = "=== GPU kernels with device_time > 0 (parsed) ==="
        chunks.append(header)
        if cuda_ops:
            chunks.append(f"{'Name':<55} {'GPU ms':>10} {'Calls':>8}")
            chunks.append("-" * 75)
            for e in cuda_ops:
                chunks.append(
                    f"{e.key[:55]:<55} {_event_cuda_us(e) / 1000:10.3f} {e.count:8d}"
                )
        else:
            chunks.append("(empty — 请以 cuda_timer.txt 为准)")
        chunks.append("")

    header = "=== By Python stack (depth=8) ==="
    chunks.append(header)
    chunks.append(
        prof.key_averages(group_by_stack_n=8).table(sort_by=sort_key, row_limit=15)
    )

    text = "\n".join(chunks)
    print(text)
    return text


def _export_chrome_trace(prof, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"moellm_profiler_{dest.name}"
    prof.export_chrome_trace(str(tmp))
    shutil.copy2(tmp, dest)
    tmp.unlink(missing_ok=True)
    return dest


def main() -> None:
    root = _project_root()
    sys.path.insert(0, str(root))

    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    from llm import MoELLM
    from data.llm.config_30m import MOELLM_30M_CONFIG, count_parameters
    from data.llm.dataset import TinyStoriesDataLoader
    from device_util import pick_device

    args = _parse_args()
    out_dir = args.output_dir or (root / "checkpoints" / "moellm_30m")
    prof_dir = out_dir / "profiler"
    prof_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    device_type = device.type
    torch.manual_seed(args.seed)

    cfg = MOELLM_30M_CONFIG
    model = MoELLM(cfg).to(device)
    print(f"device={device} params={count_parameters(cfg):,} ({count_parameters(cfg)/1e6:.2f}M)")

    data = TinyStoriesDataLoader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_train_stories=args.max_train_stories,
        max_valid_stories=100,
        seed=args.seed,
    )
    train_iter = iter(data.train)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    cuda_timer_text = ""
    if device_type == "cuda":
        print("\n[cuda timer] 运行 CUDA Event 分段计时 ...")
        timer_iter = iter(data.train)
        cuda_timer_text = _run_cuda_timer(
            model, optim, timer_iter, cfg.vocab_size, device, iters=args.timer_iters
        )
        print(cuda_timer_text)

    activities = [ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    total_steps = (args.warmup + args.active) * args.repeat
    prof_schedule = schedule(
        wait=0,
        warmup=args.warmup,
        active=args.active,
        repeat=args.repeat,
    )

    print(f"\n[kineto] warmup={args.warmup} active={args.active} repeat={args.repeat} ...")

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=args.with_stack,
        acc_events=True,  # 跨 cycle 累积事件，避免只剩最后一轮
    ) as prof:
        for step in range(1, total_steps + 1):
            try:
                inp, tgt = next(train_iter)
            except StopIteration:
                train_iter = iter(data.train)
                inp, tgt = next(train_iter)

            _train_step(model, optim, inp, tgt, cfg.vocab_size, device)
            prof.step()

            if device_type == "cuda":
                torch.cuda.synchronize()

    kineto_text = _build_kineto_summary(prof, device_type)

    trace_path = prof_dir / "trace.json"
    summary_path = prof_dir / "summary.txt"
    cuda_timer_path = prof_dir / "cuda_timer.txt"

    _export_chrome_trace(prof, trace_path)

    full_summary = kineto_text
    if cuda_timer_text:
        full_summary = cuda_timer_text + "\n\n" + kineto_text
        cuda_timer_path.write_text(cuda_timer_text, encoding="utf-8")

    summary_path.write_text(full_summary, encoding="utf-8")

    print("\n输出文件:")
    print(f"  {trace_path}  (chrome://tracing → 看 GPU 轨道)")
    print(f"  {summary_path}")
    if cuda_timer_text:
        print(f"  {cuda_timer_path}")


if __name__ == "__main__":
    main()
