# magetronLM

基于 **PyTorch** 的 Decoder-only **大语言模型**参考实现，带 **Mixture-of-Experts (MoE)** 前馈层；支持 **MHA / MQA / MLA** 三种注意力对比实验（训练 loss、训练速度、推理 prefill/decode），以及可选接入 **Megatron-LM / Megatron-Core** 做多卡分布式训练。新增 **VLM 多模态管道**（CLIP ViT-L/14@336 + Qwen3-1.7B，LLaVA 架构），支持两阶段微调、lmms-eval 评测、vLLM 推理部署。

**仓库地址**：[https://github.com/asfhaoiusd/Tiny_megatron-lm](https://github.com/asfhaoiusd/Tiny_megatron-lm)

> 名称中的 “magetron” 表示与 **Megatron** 生态相邻：自定义 `MoELLM` 与上游 Megatron-LM 可并列使用，二者许可证不同，请勿混用。

## 功能概览

| 模块 | 说明 |
|------|------|
| **模型** (`model/`) | Pre-LN、RoPE、**MHA / MQA / MLA**（`attention_type` 切换）、SDPA、Top-k MoE（SwiGLU + router aux）、Embedding / `lm_head` 权重共享、`greedy_decode` |
| **VLM 多模态** (`vlm/`) | **CLIP ViT-L/14@336** (冻结) + 2层 MLP Projector + **Qwen3-1.7B** (LoRA)，LaVA 风格 visual token 注入 |
| **预训练实验** (`pre_model/`) | ~30M 配置（GPT-2 词表 50257）、TinyStories 数据、checkpoint / metrics |
| **训练** (`training/`) | 单卡训练、注意力对比、**推理测速**、Profiler、Megatron DDP → 含 VLM 两阶段训练 |
| **数据** (`data/`) | TinyStories 语料加载、VLM 图文数据集（LLaVA 对话格式） |
| **评测** (`eval/`) | lmms-eval 多模态评测（MMBench, MMStar, TextVQA 等） + vLLM 格式转换 |
| **脚本** (`scripts/`) | 数据下载、双卡 `torchrun` 示例；VLM 一键转换+部署 |
| **技能** (`.claude/skills/`) | `train-watch` 训练指标监控、`karpathy-guidelines` 编码规范 |

### 注意力类型

| `attention_type` | 实现 | KV cache（推理） |
|------------------|------|------------------|
| `mha` | `CausalSelfAttention`，`n_kv_heads = n_heads` | 完整 K、V |
| `mqa` | `CausalSelfAttention`，`n_kv_heads = 1` | 完整 K、V（K/V 头数少） |
| `mla` | `MLA`（[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 风格） | **latent**：`(compressed_kv, k_pe_raw)`，不存完整 K/V |

- `model/MQA.py` 为早期草稿，**不参与训练**；MQA 对比请用 `--attention-type mqa`（走 `CausalSelfAttention` + RoPE + 因果 SDPA）。
- MLA 推理时从 latent 经 `kv_b_proj` 展开 `k_nope` / `v`，再对 `k_pe_raw` 做 RoPE；**显存更省**，长序列 decode 时算力开销略高。

## 实验流程速览

```
下载数据 → 训练 / 对比训练 → 推理测速 → （可选）Profiler
   │            │                  │
download_    train_tinystories   benchmark_attention_
tinystories  compare_attention   inference
```

| 目标 | 命令 | 输出 |
|------|------|------|
| 训练 loss | `compare_attention.py` | `pre_model/attention_compare/summary.json` |
| 推理速度 | `benchmark_attention_inference.py` | `pre_model/attention_inference/summary.json` |
| 单类型训练 | `train_tinystories_30m.py --attention-type mla` | `pre_model/moellm_30m_mla/` |

## 环境要求

- **Python** 3.10+
- **PyTorch** 2.x（GPU 训练需 CUDA 构建）
- **transformers**（GPT-2 分词器 / CLIP / Qwen3）
- **peft**（LoRA）、**PIL / Pillow**（图像加载）
- **lmms-eval**（VLM 评测）、**vLLM**（VLM 推理部署）
- 多卡 Megatron：**NCCL**；Windows 原生多卡 NCCL 通常不可用，建议在 **Linux / WSL2** 使用 `torchrun`
- Megatron 脚本：将 [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 置于仓库内 `Megatron-LM/` 目录
- VLM 训练建议 **A100-40G**（Stage 1 ~28GB, Stage 2 ~35GB VRAM）

## 安装

```bash
git clone https://github.com/asfhaoiusd/Tiny_megatron-lm.git
cd Tiny_megatron-lm   # 或你的本地目录名 magetronLM

python -m venv magetron
# Windows: magetron\Scripts\activate
# Linux:   source magetron/bin/activate

pip install torch transformers peft pillow
# CUDA 请按官方选择版本；新显卡（如 RTX 50 系）可能需要较新 cu128 nightly

# VLM 评测与部署（可选）
pip install lmms-eval vllm

# （可选）Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
cd Megatron-LM && pip install -e . && cd ..
```

更完整的 Megatron 说明见 [Megatron-LM快速上手指南.md](Megatron-LM快速上手指南.md)（若随仓库提供）。

## 目录结构

```
magetronLM/
├── model/                         # LLM 核心模块（纯 NN）
│   ├── attention.py               # CausalSelfAttention（MHA/MQA）
│   ├── attention_factory.py       # 按 attention_type 构建
│   ├── MLA.py                     # MLA + latent KV cache
│   ├── blocks.py                  # MoELLM / DecoderLayer
│   └── generation.py              # greedy_decode
├── vlm/                           # VLM 多模态模块
│   ├── config.py                  # VLMConfig
│   ├── vision_encoder.py          # CLIP ViT 封装
│   ├── projector.py               # 2层 MLP Projector
│   ├── vlm_model.py               # VLMForConditionalGeneration
│   └── lora_utils.py              # LoRA / 冻结 / 参数统计
├── pre_model/
│   ├── config_30m.py              # make_30m_config(mha|mqa|mla)
│   ├── dataset.py
│   ├── attention_compare/         # 训练对比结果
│   └── attention_inference/       # 推理测速结果
├── data/                          # 数据集
│   ├── tinystories/               # TinyStories 语料
│   └── vlm_dataset.py             # LLaVA 对话数据集
├── training/                      # 训练 / 推理脚本
│   ├── train_tinystories_30m.py
│   ├── compare_attention.py
│   ├── benchmark_attention_inference.py
│   ├── profile_tinystories_30m.py
│   ├── train_moellm_mcore_ddp.py
│   ├── train_vlm_stage1.py        # VLM Stage 1: Projector 对齐
│   ├── train_vlm_stage2.py        # VLM Stage 2: LoRA SFT
│   └── generate_vlm.py            # VLM 单张推理
├── eval/                          # 评测
│   ├── run_lmms_eval.py           # lmms-eval 多模态评测
│   └── convert_to_llava.py        # → HF LLaVA 格式 (vLLM)
├── serve/
│   └── convert_and_serve.sh       # 一键转换 + vLLM 部署
├── scripts/                       # 辅助脚本
├── CLAUDE.md                      # Claude Code 项目指引
└── README.md
```

## VLM 多模态管道（CLIP + Qwen3-1.7B）

```
Image (336×336)
    │
CLIP ViT-L/14@336 (300M, 冻结)
    │  [576 patches × 1024-dim]
2层 MLP Projector (~30M, 可训)
    │  [576 × 2048-dim]
Qwen3-1.7B (1.7B, LoRA rank=64)
    │
    └──→ Text Output
```

| 组件 | 模型 | 参数量 | 状态 |
|------|------|--------|------|
| Vision | `openai/clip-vit-large-patch14-336` | 300M | 冻结 |
| Projector | 2层 MLP (1024→2048→2048) | ~30M | 可训 |
| LLM | `Qwen/Qwen3-1.7B` | 1.7B | LoRA (rank=64) |
| **总计** | | **~2B** | 实际可训 ~60M |

### 训练流程

```bash
# Stage 1: 模态对齐（只训 Projector）
python training/train_vlm_stage1.py \
    --data-json data/llava_pretrain.json \
    --image-dir data/images/ \
    --batch-size 64 \
    --max-steps 5000 \
    --lr 1e-3

# Stage 2: LoRA 指令微调（Projector + LoRA）
python training/train_vlm_stage2.py \
    --data-json data/llava_instruct.json \
    --image-dir data/images/ \
    --projector-ckpt pre_model/vlm_stage1/final/projector.pt \
    --batch-size 32 \
    --max-steps 5000 \
    --lr 2e-5
```

### 单张推理

```bash
python training/generate_vlm.py \
    --image test.jpg \
    --prompt "请详细描述这张图片" \
    --projector-ckpt pre_model/vlm_stage2/final/projector.pt \
    --lora-path pre_model/vlm_stage2/final/lora
```

### lmms-eval 评测

```bash
python eval/run_lmms_eval.py \
    --model-path pre_model/vlm_stage2/final \
    --tasks mmbench,mmstar,textvqa,mme
```

### vLLM 部署

```bash
# 一键：转 HF LLaVA 格式 + 启动 vLLM 服务
bash serve/convert_and_serve.sh \
    --model-path pre_model/vlm_stage2/final \
    --port 8000

# 测试 API
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "hf_llava", "messages": [{"role": "user", "content": "描述这张图片"}]}'
```

## TinyStories ~30M 实验（单卡）

### 1. 下载数据

```bash
python scripts/download_tinystories.py
# 写入 data/tinystories/（train ~1.8GB, valid ~19MB）
```

### 2. 训练某一种注意力

```bash
python training/train_tinystories_30m.py --attention-type mla --max-steps 500 --device cuda
python training/train_tinystories_30m.py --attention-type mha --device cuda
python training/train_tinystories_30m.py --attention-type mqa --device cuda
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max-steps` | `200` | **最大训练步数** |
| `--attention-type` | `mha` | `mha` / `mqa` / `mla` |
| `--batch-size` | `8` | batch 大小 |
| `--seq-len` | `256` | 序列长度 |
| `--output-dir` | `pre_model/moellm_30m_{type}/` | checkpoint、`metrics.json` |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |

### 3. 训练对比（loss + 每步耗时）

```bash
python training/compare_attention.py --max-steps 200 --device cuda
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max-steps` | `100` | 每种注意力训练步数 |
| `--attention-types` | `mha mqa mla` | 对比列表 |
| `--warmup-steps` | `5` | 计时预热步数 |

输出：`pre_model/attention_compare/summary.json`

### 4. 推理速度对比（prefill + decode）

```bash
python training/benchmark_attention_inference.py --device cuda
python training/benchmark_attention_inference.py --prefill-len 256 --decode-tokens 128 --iters 50
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--prefill-len` | `256` | 整段 prompt prefill 长度 |
| `--decode-tokens` | `128` | 增量 decode 计时 token 数 |
| `--batch-size` | `1` | 推理 batch |
| `--iters` | `20` | 计时重复次数 |

输出：`pre_model/attention_inference/summary.json`（`prefill_ms`、`decode_ms_per_token`、`tokens_per_sec_decode`、`kv_cache_kb_per_token`）。

**参考结果**（~30M、`prefill=128` / `decode=32`、RTX 5070 量级，仅供量级参考）：

| 类型 | prefill | decode | tok/s | KV cache |
|------|---------|--------|-------|----------|
| mha | ~14 ms | ~8.4 ms/tok | ~119 | ~7.9 KB/tok |
| mqa | ~13 ms | ~7.9 ms/tok | ~126 | ~1.3 KB/tok |
| mla | ~16 ms | ~9.5 ms/tok | ~105 | **~0.9 KB/tok** |

MLA **cache 最小**；decode 可能因每步 `kv_b_proj` 展开而略慢。长上下文、batch 推理时结论可能不同，请以本机实测为准。

### 5. Profiler（训练瓶颈）

```bash
python training/profile_tinystories_30m.py --device cuda --warmup 3 --active 10
```

输出：`pre_model/moellm_30m/profiler/`（`cuda_timer.txt`、`summary.txt`、`trace.json`）。

## 快速使用（Python API）

```python
import torch
from model import MoELLM, greedy_decode
from pre_model.config_30m import make_30m_config

cfg = make_30m_config("mla")  # 或 "mha" / "mqa"
model = MoELLM(cfg).eval()
ids = torch.randint(0, cfg.vocab_size, (1, 32))

# 训练
logits, aux_loss, _ = model(ids)
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, cfg.vocab_size),
    ids[:, 1:].reshape(-1),
) + aux_loss

# 推理（KV cache；MLA 为 latent cache）
out = greedy_decode(model, ids, max_new_tokens=16)
```

## 多卡训练（Megatron-Core + 数据并行）

在 **Linux** 下，于仓库根目录：

```bash
export PYTHONPATH="${PWD}:${PWD}/Megatron-LM:${PYTHONPATH}"
torchrun --nproc_per_node=2 training/train_moellm_mcore_ddp.py --train-iters 100 --bf16
```

或：

```bash
bash scripts/run_train_moellm_2gpu.sh --train-iters 100 --bf16
```

当前 `MoELLM` **仅支持 TP=PP=EP=1**（纯数据并行）。

## 上传到 GitHub 的建议

1. **Megatron-LM**：使用 git submodule 或在 `.gitignore` 中忽略，由用户自行 clone。
2. **许可证**：自写代码与 Megatron-LM（NVIDIA 许可证）分开说明。
3. **勿提交**：`data/tinystories/`、checkpoint、`pre_model/gpt2_tokenizer/`、虚拟环境 `magetron/`。

## 参考链接

- [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B) — 阿里开源 LLM，Apache 2.0
- [LLaVA](https://llava-vl.github.io/) — Large Language and Vision Assistant
- [DeepSeek-V2（MLA）](https://arxiv.org/abs/2405.04434)
- [nanoVLM](https://github.com/huggingface/nanoVLM) — VLM 代码参考
- [vLLM](https://github.com/vllm-project/vllm) — 高性能 LLM 推理引擎
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — 多模态评测框架
- [Megatron Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

## 免责声明

本项目为学习与研究用途的示例代码；生产环境训练需自行做稳定性、精度、合规与资源评估。
