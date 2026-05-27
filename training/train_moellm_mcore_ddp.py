"""
Train ``model.MoELLM`` with Megatron-Core parallel groups + DDP.

并行关系（Megatron 惯例，忽略 context parallel 时）::

    world_size = tensor_parallel × pipeline_parallel × expert_parallel × data_parallel

因此 **数据并行 (DP)** 与 **模型并行 (TP / PP)** 可以同时存在，前提是总卡数够分，例如：

- 4 卡：``TP=2, PP=1`` → ``DP = 4 / (2×1) = 2``（2 路张量并行 × 2 路数据并行）
- 8 卡：``TP=2, PP=2`` → ``DP = 8 / (2×2) = 2``

**仅 2 张卡时**：若 ``TP=2`` 或 ``PP=2``，则 ``DP=1``，无法在同一配置里再叠一层 DP（卡数不够）。

当前限制
~~~~~~~~
本仓库里的 ``MoELLM`` 使用普通 ``nn.Linear``，**没有**接入 ``megatron.core`` 的
``ColumnParallelLinear`` / ``RowParallelLinear`` / Pipeline 切分，因此 **仅支持
``--tensor-model-parallel-size 1`` 且 ``--pipeline-model-parallel-size 1``**（纯 DP）。
若你传入 ``TP>1`` 或 ``PP>1``，脚本会初始化进程组后立刻报错退出，并提示应改用
Megatron-LM 自带的 ``GPTModel`` 等实现。

运行示例（Linux + CUDA + NCCL，在 ``magetronLM`` 仓库根目录）::

    # 双卡纯数据并行
    torchrun --nproc_per_node=2 training/train_moellm_mcore_ddp.py --train-iters 100

    # 四卡：TP=2 + DP=2（需模型支持 TP；当前 MoELLM 会报错，仅作参数示例）
    torchrun --nproc_per_node=4 training/train_moellm_mcore_ddp.py \\
        --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1

Windows 原生上 NCCL 多卡通常不可用；请使用 **WSL2 / Linux**。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


def _ensure_paths() -> Path:
    """Megatron-LM 源码根目录与项目根目录加入 ``sys.path``。"""
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    megatron_lm_root = project_root / "Megatron-LM"
    if not megatron_lm_root.is_dir():
        raise RuntimeError(
            f"未找到 Megatron-LM 目录: {megatron_lm_root}\n"
            "请把 NVIDIA/Megatron-LM 克隆到仓库内的 Megatron-LM/，或修改本脚本中的路径。"
        )
    sys.path.insert(0, str(megatron_lm_root))
    sys.path.insert(0, str(project_root))
    return project_root


_ensure_paths()

import torch
import torch.distributed as dist
import torch.nn.functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.transformer_config import TransformerConfig

from model import MoELLM, MoELLMConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoELLM + Megatron-Core (DP via DDP; TP/PP 需 Megatron 原生模型)"
    )
    p.add_argument("--train-iters", type=int, default=50)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-distributed-optimizer", action="store_true", help="使用 reduce-scatter 的分布式优化器")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast 训练（需 GPU 支持）")
    p.add_argument(
        "--tensor-model-parallel-size",
        "--tp",
        type=int,
        default=1,
        dest="tensor_model_parallel_size",
        help="张量并行度（Megatron TP）。当前 MoELLM 仅支持 1。",
    )
    p.add_argument(
        "--pipeline-model-parallel-size",
        "--pp",
        type=int,
        default=1,
        dest="pipeline_model_parallel_size",
        help="流水线并行度（Megatron PP）。当前 MoELLM 仅支持 1。",
    )
    p.add_argument(
        "--expert-model-parallel-size",
        "--ep",
        type=int,
        default=1,
        dest="expert_model_parallel_size",
        help="专家并行度（Megatron EP）。当前 MoELLM 仅支持 1。",
    )
    return p.parse_args()


def _megatron_transformer_config(mcfg: MoELLMConfig, *, bf16: bool) -> TransformerConfig:
    """供 MCore ``DistributedDataParallel`` 使用的 ``TransformerConfig``（与并行/精度相关）。"""
    return TransformerConfig(
        num_layers=mcfg.n_layers,
        hidden_size=mcfg.d_model,
        num_attention_heads=mcfg.n_heads,
        num_query_groups=mcfg.n_kv_heads,
        ffn_hidden_size=mcfg.d_ff,
        bf16=bf16,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        expert_model_parallel_size=parallel_state.get_expert_model_parallel_world_size(),
    )


def _check_parallel_layout(world_size: int, tp: int, pp: int, ep: int) -> int:
    mp = tp * pp * ep
    if world_size % mp != 0:
        raise RuntimeError(
            f"world_size ({world_size}) 必须能被 TP×PP×EP ({tp}×{pp}×{ep}={mp}) 整除。"
        )
    dp = world_size // mp
    return dp


def main() -> None:
    args = _parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % max(torch.cuda.device_count(), 1))))

    tp, pp, ep = args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.expert_model_parallel_size
    data_parallel_size = _check_parallel_layout(world_size, tp, pp, ep)

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA。若在 Windows 上，请用 WSL2/Linux 或改为 CPU 实验脚本。")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl")
    try:
        if tp != 1 or pp != 1 or ep != 1:
            raise RuntimeError(
                "当前 ``model.MoELLM`` 未实现 Megatron 张量/流水线/专家并行（仍为整卡 ``nn.Module``）。\n"
                f"你设置了 TP={tp}, PP={pp}, EP={ep}，若模型已支持，对应数据并行组大小应为 DP={data_parallel_size} "
                f"(world_size={world_size})。\n"
                "若要在同一作业里同时使用 **模型并行 + 数据并行**，请：\n"
                "  1) 使用 Megatron-LM 的 ``megatron.core.models.gpt.GPTModel``（或 ``pretrain_gpt.py`` 管线），或\n"
                "  2) 把 MoELLM 的 Linear / Attention / MoE 改为 MCore 的并行算子并按 PP 切 stage。\n"
                "在此之前请使用 ``--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
                "--expert-model-parallel-size 1`` 做纯数据并行。"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
        )

        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        random.seed(args.seed + rank)

        mcfg = MoELLMConfig(
            vocab_size=4096,
            d_model=256,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            d_ff=1024,
            max_seq_len=args.seq_len,
            n_experts=8,
            num_experts_per_tok=2,
        )
        raw = MoELLM(mcfg).to(device=device, dtype=torch.bfloat16 if args.bf16 else torch.float32)

        ddp_cfg = DistributedDataParallelConfig(
            use_distributed_optimizer=args.use_distributed_optimizer,
            overlap_grad_reduce=False,
        )
        t_cfg = _megatron_transformer_config(mcfg, bf16=args.bf16)
        model = DistributedDataParallel(t_cfg, ddp_cfg, raw)

        optim_cfg = OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            bf16=args.bf16,
            use_distributed_optimizer=args.use_distributed_optimizer,
            use_precision_aware_optimizer=False,
            main_params_dtype=torch.float32,
            main_grads_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
        )
        optimizer = get_megatron_optimizer(optim_cfg, [model])

        if rank == 0:
            print(
                f"[rank0] world={world_size} TP={tp} PP={pp} EP={ep} → DP={data_parallel_size} | "
                f"device={device} bf16={args.bf16}"
            )
            n_params = sum(p.numel() for p in raw.parameters())
            print(f"[rank0] MoELLM params={n_params:,}")

        vocab = mcfg.vocab_size
        B, T = args.micro_batch_size, args.seq_len

        for it in range(1, args.train_iters + 1):
            model.zero_grad_buffer()
            # 各 rank 使用不同随机 batch（模拟分布式数据）
            x = torch.randint(0, vocab, (B, T), device=device)
            inp, tgt = x[:, :-1], x[:, 1:]

            if args.bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, aux, _ = model(inp)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, vocab),
                    tgt.reshape(-1),
                ) + aux.float()
            else:
                logits, aux, _ = model(inp)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab),
                    tgt.reshape(-1),
                ) + aux

            loss.backward()
            optimizer.step()

            if it % 10 == 0 and rank == 0:
                print(f"[rank0] iter={it} loss={loss.item():.4f}")

        if rank == 0:
            print("[rank0] done.")
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        print(
            "提示: WORLD_SIZE<2 时以单进程运行；双卡请使用:\n"
            "  torchrun --nproc_per_node=2 training/train_moellm_mcore_ddp.py",
            file=sys.stderr,
        )
    main()
