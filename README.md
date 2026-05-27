# magetronLM

基于 **PyTorch** 的 Decoder-only **大语言模型**参考实现，带 **Mixture-of-Experts (MoE)** 前馈层；支持 **MHA / MQA / MLA** 三种注意力对比实验，以及可选接入本地 **Megatron-LM / Megatron-Core** 做多卡分布式训练。

> 名称中的 “magetron” 表示与 **Megatron** 生态相邻：自定义 `MoELLM` 与上游 `Megatron-LM` 可并列使用，二者许可证不同，上传 GitHub 时请分别遵守。

## 功能概览


| 模块                       | 说明                                                                                                                                                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **模型** (`model/`)        | Pre-LN、RoPE、**MHA / MQA / MLA**（`attention_type` 切换）、`scaled_dot_product_attention`（GQA/MHA 时可用 `enable_gqa`）、Top‑k MoE（SwiGLU + router aux loss）、Embedding 与 `lm_head` 权重共享、KV cache、`greedy_decode` |
| **预训练实验** (`pre_model/`) | ~30M 配置（GPT-2 词表 50257）、TinyStories 数据加载、checkpoint / metrics                                                                                                                                         |
| **训练** (`training/`)     | `train_tinystories_30m.py` 单卡实验训练；`compare_attention.py` MHA/MQA/MLA 对比；`profile_tinystories_30m.py` 性能分析；`train_moellm_mcore_ddp.py` Megatron-Core DDP                                               |
| **脚本** (`scripts/`)      | `download_tinystories.py` 下载数据；`run_train_moellm_2gpu.sh` 双卡 `torchrun` 示例                                                                                                                            |


### 注意力类型


| `attention_type` | 实现                                           | 说明                               |
| ---------------- | -------------------------------------------- | -------------------------------- |
| `mha`            | `CausalSelfAttention`，`n_kv_heads = n_heads` | 标准多头注意力                          |
| `mqa`            | `CausalSelfAttention`，`n_kv_heads = 1`       | 多查询注意力（与训练栈同路径，含 RoPE + 因果 SDPA） |
| `mla`            | `MLA`（DeepSeek-V2 风格）                        | 低秩 Q/KV + nope/rope 拆分           |


`model/MQA.py` 为早期独立草稿，**不参与训练**；对比实验请使用 `--attention-type mqa`。

## 环境要求

- **Python** 3.10+（建议）
- **PyTorch** 2.x（CUDA 构建用于 GPU 训练）
- **transformers**（GPT-2 分词器，TinyStories 实验）
- 多卡 Megatron 训练：**NCCL**；Windows 原生多卡 NCCL 通常不可用，建议在 **Linux / WSL2** 下跑 `torchrun`
- 若使用 Megatron 训练脚本：将 **[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** 置于仓库内 `**Megatron-LM/`** 目录

## 安装

```bash
git clone <你的仓库 URL>
cd magetronLM

python -m venv magetron
# Windows: magetron\Scripts\activate
# Linux:   source magetron/bin/activate

pip install torch transformers
# CUDA 版本请按官方指引选择，例如 cu124 / cu128 nightly（新显卡可能需要较新构建）

# （可选）Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
cd Megatron-LM && pip install -e . && cd ..
```

更完整的 Megatron 安装见 `**Megatron-LM快速上手指南.md**`（若随仓库提供）。

## 目录结构

```
magetronLM/
├── model/                      # MoELLM、MoE、注意力、MLA、RoPE、解码
│   ├── attention.py            # CausalSelfAttention（MHA/MQA/GQA）
│   ├── attention_factory.py    # 按 attention_type 构建注意力
│   ├── MLA.py                  # DeepSeek-V2 风格 MLA
│   └── ...
├── pre_model/                  # ~30M 实验配置与数据
│   ├── config_30m.py           # make_30m_config(mha|mqa|mla)
│   ├── dataset.py              # TinyStories + GPT-2 tokenizer
│   ├── moellm_30m_{mha,mqa,mla}/   # 单类型训练输出
│   └── attention_compare/      # compare_attention.py 汇总结果
├── training/
│   ├── train_tinystories_30m.py
│   ├── compare_attention.py
│   ├── profile_tinystories_30m.py
│   └── train_moellm_mcore_ddp.py
├── data/tinystories/           # 下载后的语料（不入库）
├── scripts/
│   ├── download_tinystories.py
│   └── run_train_moellm_2gpu.sh
├── Megatron-LM/                # 上游（建议 submodule 或用户自行 clone）
└── README.md
```

## TinyStories ~30M 实验（单卡）

### 1. 下载数据

```bash
python scripts/download_tinystories.py
# 默认写入 data/tinystories/（train ~1.8GB, valid ~19MB）
```

### 2. 训练某一种注意力

在仓库根目录执行（`import model` 需以根目录为工作目录）：

```bash
python training/train_tinystories_30m.py --attention-type mla --max-steps 500 --device cuda
python training/train_tinystories_30m.py --attention-type mha --device cuda
python training/train_tinystories_30m.py --attention-type mqa --device cuda
```


| 参数                 | 默认                             | 说明                      |
| ------------------ | ------------------------------ | ----------------------- |
| `--max-steps`      | `200`                          | **最大训练步数**              |
| `--attention-type` | `mha`                          | `mha` / `mqa` / `mla`   |
| `--batch-size`     | `8`                            | batch 大小                |
| `--seq-len`        | `256`                          | 序列长度                    |
| `--output-dir`     | `pre_model/moellm_30m_{type}/` | checkpoint 与 metrics    |
| `--device`         | `auto`                         | `auto` / `cpu` / `cuda` |


输出：`checkpoints/step_*.pt`、`metrics.json`（含 `valid_loss`）。

### 3. MHA / MQA / MLA 对比（loss + 耗时）

```bash
python training/compare_attention.py --max-steps 200 --device cuda
```


| 参数                  | 默认            | 说明                 |
| ------------------- | ------------- | ------------------ |
| `--max-steps`       | `100`         | 每种注意力类型的训练步数       |
| `--attention-types` | `mha mqa mla` | 要对比的类型列表           |
| `--warmup-steps`    | `5`           | 计时前预热（不计入 ms/step） |


结果：`pre_model/attention_compare/summary.json` 及各子目录 `mha/`、`mqa/`、`mla/` 下的 `metrics.json`。

### 4. 性能分析（Profiler）

```bash
python training/profile_tinystories_30m.py --device cuda --warmup 3 --active 10
```

输出目录默认 `pre_model/moellm_30m/profiler/`：`cuda_timer.txt`（分段计时）、`summary.txt`、`trace.json`。

## 快速使用（Python API）

```python
import torch
from model import MoELLM, MoELLMConfig, greedy_decode
from pre_model.config_30m import make_30m_config

cfg = make_30m_config("mla")  # 或 "mha" / "mqa"
model = MoELLM(cfg)
ids = torch.randint(0, cfg.vocab_size, (1, 32))

logits, aux_loss, _ = model(ids)
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, cfg.vocab_size),
    ids[:, 1:].reshape(-1),
) + aux_loss

# out = greedy_decode(model, ids, max_new_tokens=16, eos_token_id=None)
```

## 多卡训练（Megatron-Core + 数据并行）

在 **Linux** 下，于仓库根目录：

```bash
export PYTHONPATH="${PWD}:${PWD}/Megatron-LM:${PYTHONPATH}"
torchrun --nproc_per_node=2 training/train_moellm_mcore_ddp.py --train-iters 100 --bf16
```

或使用：

```bash
bash scripts/run_train_moellm_2gpu.sh --train-iters 100 --bf16
```

当前 `MoELLM` **仅支持 TP=PP=EP=1**（纯数据并行）。常用参数见 `python training/train_moellm_mcore_ddp.py --help`。

## 参考链接

- [DeepSeek-V2（MLA 论文）](https://arxiv.org/abs/2405.04434)
- [Megatron Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
- [Megatron-LM（上游）](https://github.com/NVIDIA/Megatron-LM)

## 免责声明

本项目为学习与研究用途的示例代码；生产环境训练需自行做稳定性、精度、合规与资源评估。