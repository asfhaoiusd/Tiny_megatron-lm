#!/usr/bin/env bash
set -euo pipefail
# 在 magetronLM 仓库根目录执行：双卡数据并行 + Megatron-Core DDP
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/Megatron-LM:${PYTHONPATH:-}"
exec torchrun --nproc_per_node=2 scripts/llm/train_ddp.py "$@"
