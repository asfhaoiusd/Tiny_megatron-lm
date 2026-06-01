"""训练脚本共用的设备选择（含 RTX 50 系 / sm_120 兼容检测）。"""

from __future__ import annotations


def cuda_usable() -> bool:
    """``is_available()`` 为 True 时，再试跑一次最小 kernel。"""
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, device="cuda")
        _ = x + 1
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def pick_device(name: str = "auto"):
    """
    选择训练设备。

    - ``auto``：CUDA 可用且能执行 kernel 则用 GPU，否则回退 CPU 并打印说明
    - ``cuda``：强制 GPU；不可用则抛错
    - ``cpu``：强制 CPU
    """
    import torch

    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not cuda_usable():
            raise RuntimeError(
                "已指定 --device cuda，但当前 PyTorch 无法在此 GPU 上运行 kernel。\n"
                "常见原因：RTX 50 系 (sm_120) 需要支持 CUDA 12.8+ 的 PyTorch 构建。\n"
                "见 https://pytorch.org/get-started/locally/ ，或先用 --device cpu。"
            )
        return torch.device("cuda")

    if name == "auto":
        if cuda_usable():
            return torch.device("cuda")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(
                f"[device] 检测到 GPU ({props.name}, CC {props.major}.{props.minor})，"
                "但当前 PyTorch 无法在其上执行 CUDA kernel，已回退到 CPU。\n"
                "  若需使用 GPU，请安装支持 sm_120 的 PyTorch（CUDA 12.8/13.0）。"
            )
        return torch.device("cpu")

    raise ValueError(f"未知 device: {name!r}")
