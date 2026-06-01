# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

magetronLM — PyTorch Decoder-only LLM with MoE FFN + MHA/MQA/MLA attention comparison. Also includes a ~2B VLM pipeline (CLIP ViT-L/14@336 + Qwen3-1.7B, LLaVA-style). Companion Megatron-LM directory provides optional multi-GPU DDP via Megatron-Core.

## Virtual environment

```
magetron\Scripts\Activate.ps1    # Windows PowerShell
source magetron/bin/activate     # Linux / WSL2
```

Main dependencies: `torch` (2.x, CUDA), `transformers`, `peft`, `pillow`.

## Key commands

```bash
# ---- LLM (纯文本) ----
# Download TinyStories data (~1.8GB)
python scripts/llm/download_tinystories.py

# Train one attention type (~30M params, single GPU)
python scripts/llm/train.py --attention-type mla --max-steps 500 --device cuda
python scripts/llm/train.py --attention-type mha --device cuda
python scripts/llm/train.py --attention-type mqa --device cuda

# Compare all three attention types (loss + step time)
python scripts/llm/compare_attention.py --max-steps 200 --device cuda

# Inference speed benchmark (prefill + decode)
python scripts/llm/benchmark_inference.py --device cuda

# Profile training bottlenecks
python scripts/llm/profile.py --device cuda --warmup 3 --active 10

# Multi-GPU Megatron DDP (Linux only, requires Megatron-LM/ cloned)
bash scripts/llm/run_2gpu.sh --train-iters 100 --bf16

# ---- VLM (多模态) ----
# Stage 1: Projector alignment
python scripts/vlm/train_stage1.py --data-json data/llava_pretrain.json --image-dir data/images/

# Stage 2: LoRA instruction tuning
python scripts/vlm/train_stage2.py --data-json data/llava_instruct.json --projector-ckpt checkpoints/vlm_stage1/final/projector.pt

# Single-image inference
python scripts/vlm/generate.py --image test.jpg --projector-ckpt checkpoints/vlm_stage2/final/projector.pt

# lmms-eval benchmark
python scripts/vlm/eval.py --model-path checkpoints/vlm_stage2/final --tasks mmbench,mmstar

# vLLM serve
bash scripts/vlm/serve.sh --model-path checkpoints/vlm_stage2/final --port 8000
```

Smoke-test MLA independently: `python llm/MLA.py`

## Architecture

`llm/` — 纯文本 LLM 模块 (pure NN, no training logic):
- `config.py` — `MoELLMConfig` dataclass with all hyperparams
- `blocks.py` — `DecoderLayer` (Pre-LN attn + Pre-LN MoE), `MoELLM` (embed → layers → norm → lm_head, weight-tied)
- `attention.py` — `CausalSelfAttention` with RoPE + PyTorch SDPA, supports MHA/MQA via `n_kv_heads` (GQA)
- `MLA.py` — DeepSeek-V2 MLA with latent KV cache (`compressed_kv` + `k_pe_raw`, not full K/V)
- `attention_factory.py` — `build_attention(config)` routes by `attention_type`:
  - `"mla"` → `MLA` (separate `MLAConfig`)
  - `"mha"` / `"mqa"` → `CausalSelfAttention` (MQA = `n_kv_heads=1`)
- `moe.py` — top-k token-choice routing, stacked expert weights, SwiGLU, auxiliary load-balancing loss
- `rope.py` — `RotaryEmbedding` + `apply_rope`
- `generation.py` — `greedy_decode` with KV cache
- `MQA.py` — legacy standalone MQA module, **not used** in training; use `--attention-type mqa` instead

`data/llm/` — LLM 数据与配置:
- `config_30m.py` — `make_30m_config(type)` generates ~30M-param configs (GPT-2 vocab 50257)
- `dataset.py` — streaming TinyStories loader with GPT-2 tokenizer

`scripts/llm/` — LLM 脚本: `train.py`, `compare_attention.py`, `benchmark_inference.py`, `profile.py`, `train_ddp.py`, `download_tinystories.py`, `run_2gpu.sh`

## VLM pipeline (CLIP + Qwen3-1.7B, LLaVA-style)

`vlm/` — VLM core modules:
- `config.py` — `VLMConfig` dataclass (vision: CLIP ViT-L/14@336, LLM: Qwen3-1.7B)
- `vision_encoder.py` — CLIP wrapper, frozen, outputs (B, 576, 1024)
- `projector.py` — 2-layer MLP (1024→2048→2048, ~30M params)
- `vlm_model.py` — `VLMForConditionalGeneration` (CLIP+Projector+Qwen3, visual token injection)
- `lora_utils.py` — `apply_lora_to_llm()`, `freeze_component()`, `get_trainable_params()`

`data/vlm/` — VLM 数据: `dataset.py` (`LLaVADataset` with Qwen3 chat template)

`scripts/vlm/` — VLM 全流程: `train_stage1.py`, `train_stage2.py`, `generate.py`, `eval.py`, `convert_to_llava.py`, `serve.sh`

`checkpoints/` — 训练输出 (experiment results, model checkpoints)

`device_util.py` — `pick_device("auto"|"cuda"|"cpu")` with RTX 50-series sm_120 detection
`watch_metrics.py` — training metrics parser (scans `checkpoints/` for `metrics.json`)

**Training flow**: Stage 1 (projector alignment, LR=1e-3) → Stage 2 (LoRA SFT, rank=64, LR=2e-5)

**vLLM serving**: Convert to HF LLaVA format, then `vllm serve --trust-remote-code`

## Attention type comparison

| `attention_type` | Module | KV cache |
|---|---|---|
| `mha` | `CausalSelfAttention`, `n_kv_heads = n_heads` | Full K, V |
| `mqa` | `CausalSelfAttention`, `n_kv_heads = 1` | Full K, V (fewer heads) |
| `mla` | `MLA` (DeepSeek-V2 style) | Latent only: `(compressed_kv, k_pe_raw)` |

## MoELLM forward signature

```python
logits, aux_loss, past = model(input_ids, attn_mask=None, past_key_values=None, position_offset=0, use_cache=False)
```

`aux_loss` is a scalar tensor (zero when not training or `router_aux_loss_coef=0`).
`past` is `list[tuple[Tensor, Tensor]] | None` — for MLA each tuple is `(compressed_kv, k_pe_raw)`.

## Multi-GPU limitation

`scripts/llm/train_ddp.py` only supports TP=PP=EP=1 (pure data parallelism). NCCL required — Windows native multi-GPU NCCL is typically unavailable; use Linux/WSL2.
