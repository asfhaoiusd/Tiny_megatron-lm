# magetronLM

基于 **PyTorch** 的 Decoder-only **大语言模型**参考实现，带 **Mixture-of-Experts (MoE)** 前馈层；可选接入本地 **Megatron-LM / Megatron-Core** 做多卡分布式训练（数据并行 + 进程组与 DDP 包装）。

> 名称中的 “magetron” 表示与 **Megatron** 生态相邻：自定义 `MoELLM` 与上游 `Megatron-LM` 可并列使用，二者许可证不同，上传 GitHub 时请分别遵守。

## 功能概览

| 模块 | 说明 |
|------|------|
| **模型** (`model/`) | Pre-LN、RoPE、GQA、`scaled_dot_product_attention`（含 `enable_gqa` 时原生 GQA）、Top‑k MoE（SwiGLU 专家 + router 负载均衡辅助损失）、Embedding 与 `lm_head` 权重共享、KV cache、`greedy_decode` |
| **训练** (`training/`) | `train_moellm_mcore_ddp.py`：Megatron-Core `parallel_state` + `DistributedDataParallel` + `get_megatron_optimizer`；当前 **仅 TP=PP=EP=1**（纯数据并行），因 `MoELLM` 尚未接入列/行并行线性层 |
| **脚本** (`scripts/`) | `run_train_moellm_2gpu.sh`：双卡 `torchrun` 示例（Linux + NCCL） |

## 环境要求

- **Python** 3.10+（建议）
- **PyTorch** 2.x，CUDA 构建（多卡训练需要 **NCCL**；Windows 原生多卡 NCCL 通常不可用，建议在 **Linux / WSL2** 下跑 `torchrun`）
- 若使用 Megatron 训练脚本：需将 **[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** 置于仓库内 **`Megatron-LM/`** 目录（或自行修改 `training/train_moellm_mcore_ddp.py` 中的路径逻辑），并按其文档安装依赖

## 安装

```bash
git clone <你的仓库 URL>
cd magetronLM

# PyTorch 请按官方指引选择 CUDA 版本，例如：
# pip install torch --index-url https://download.pytorch.org/whl/cu124

# （可选）Megatron-LM：克隆到子目录后安装
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
cd Megatron-LM && pip install -e . && cd ..
```

更完整的 Megatron 安装与数据流程见仓库内 **`Megatron-LM快速上手指南.md`**（若你随项目一并上传）。

## 目录结构

```
magetronLM/
├── model/                 # MoELLM 与配置、MoE、注意力、RoPE、贪心解码
├── training/              # Megatron-Core 集成训练入口
├── scripts/               # 启动脚本（bash）
├── Megatron-LM/           # 上游 Megatron-LM 源码（建议 git submodule 或单独说明）
├── Megatron-LM快速上手指南.md
└── README.md
```

## 快速使用（单进程 / 推理向）

在项目根目录执行，保证 `import model` 能找到包（当前根目录即包父目录）：

```python
import torch
from model import MoELLM, MoELLMConfig, greedy_decode

cfg = MoELLMConfig(vocab_size=8192, d_model=512, n_layers=8, n_heads=8)
model = MoELLM(cfg)
ids = torch.randint(0, cfg.vocab_size, (1, 32))

# 训练：logits + MoE aux loss；第三项为 KV cache（use_cache=True 时非 None）
logits, aux_loss, _ = model(ids)
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, cfg.vocab_size),
    ids[:, 1:].reshape(-1),
) + aux_loss

# 贪心续写（内部用 KV cache）
# out = greedy_decode(model, ids, max_new_tokens=16, eos_token_id=None)
```

## 多卡训练（Megatron-Core + 数据并行）

在 **Linux** 下，于仓库根目录：

```bash
export PYTHONPATH="${PWD}:${PWD}/Megatron-LM:${PYTHONPATH}"
torchrun --nproc_per_node=2 training/train_moellm_mcore_ddp.py --train-iters 100 --bf16
```

或使用脚本：

```bash
bash scripts/run_train_moellm_2gpu.sh --train-iters 100 --bf16
```

常用参数见 `python training/train_moellm_mcore_ddp.py --help`（如 `--tensor-model-parallel-size` / `--pipeline-model-parallel-size`；**当前 MoELLM 仅允许均为 1**）。

## 上传到 GitHub 的建议

1. **Megatron-LM 体积很大**：不建议把整个 `Megatron-LM/` 无压缩塞进同一仓库。可选做法：
   - 使用 **`git submodule`** 指向官方仓库固定 commit；或  
   - 在 **`.gitignore`** 中忽略 `Megatron-LM/`，在 README 中写明由用户自行 `git clone`。
2. **许可证**：本仓库中你自写的 `model/`、`training/`、`scripts/` 由你自行选择许可证（如 MIT）；**Megatron-LM** 遵循其自带的 **NVIDIA 许可证**，勿混为一谈。
3. **大文件 / 数据**：数据集、checkpoint 不要提交进 Git，用 Git LFS 或网盘链接说明即可。

## 参考链接

- [Megatron Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
- [Megatron-LM（上游）](https://github.com/NVIDIA/Megatron-LM)

## 免责声明

本项目为学习与研究用途的示例代码；生产环境训练需自行做稳定性、精度、合规与资源评估。
