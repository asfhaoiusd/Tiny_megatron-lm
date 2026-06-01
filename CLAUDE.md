# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

magetronLM — PyTorch Decoder-only LLM with MoE FFN + MHA/MQA/MLA attention comparison. Also includes a ~2B VLM pipeline (CLIP ViT-L/14@336 + Qwen3-1.7B, LLaVA-style). Companion Megatron-LM directory provides optional multi-GPU DDP via Megatron-Core.

## Virtual environment

```
magetron\Scripts\Activate.ps1    # Windows PowerShell
source magetron/bin/activate     # Linux / WSL2
```

Main dependencies: `torch` (2.x, CUDA), `transformers` (GPT-2 tokenizer).

## Key commands

```bash
# Download TinyStories data (~1.8GB)
python scripts/download_tinystories.py

# Train one attention type (~30M params, single GPU)
python training/train_tinystories_30m.py --attention-type mla --max-steps 500 --device cuda
python training/train_tinystories_30m.py --attention-type mha --device cuda
python training/train_tinystories_30m.py --attention-type mqa --device cuda

# Compare all three attention types (loss + step time)
python training/compare_attention.py --max-steps 200 --device cuda

# Inference speed benchmark (prefill + decode)
python training/benchmark_attention_inference.py --device cuda

# Profile training bottlenecks
python training/profile_tinystories_30m.py --device cuda --warmup 3 --active 10

# Multi-GPU Megatron DDP (Linux only, requires Megatron-LM/ cloned)
export PYTHONPATH="${PWD}:${PWD}/Megatron-LM:${PYTHONPATH}"
torchrun --nproc_per_node=2 training/train_moellm_mcore_ddp.py --train-iters 100 --bf16
```

Smoke-test MLA independently: `python model/MLA.py`

## Architecture

`model/` — pure NN modules, no training logic:
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

`pre_model/` — experiment presets and data:
- `config_30m.py` — `make_30m_config(type)` generates ~30M-param configs (GPT-2 vocab 50257); MLA settings hand-tuned to match parameter count
- `dataset.py` — streaming TinyStories loader with GPT-2 tokenizer

`training/` — entry-point scripts:
- `train_tinystories_30m.py` — single-GPU training loop
- `compare_attention.py` — train MHA/MQA/MLA sequentially, save `summary.json`
- `benchmark_attention_inference.py` — prefill + decode timing with KV cache
- `profile_tinystories_30m.py` — CUDA profiler wrapper
- `train_moellm_mcore_ddp.py` — Megatron-Core DDP wrapper (TP=PP=EP=1, data-parallel only)
- `device_util.py` — `pick_device("auto"|"cuda"|"cpu")` with RTX 50-series sm_120 detection

`scripts/` — `download_tinystories.py`, `run_train_moellm_2gpu.sh`

## VLM pipeline (CLIP + Qwen3-1.7B, LLaVA-style)

`vlm/` — VLM core modules:
- `config.py` — `VLMConfig` dataclass (vision: CLIP ViT-L/14@336, LLM: Qwen3-1.7B)
- `vision_encoder.py` — CLIP wrapper, frozen, outputs (B, 576, 1024)
- `projector.py` — 2-layer MLP (1024→2048→2048, ~30M params)
- `vlm_model.py` — `VLMForConditionalGeneration` (CLIP+Projector+Qwen3, visual token injection)
- `lora_utils.py` — `apply_lora_to_llm()`, `freeze_component()`, `get_trainable_params()`

`data/vlm_dataset.py` — `LLaVADataset` with Qwen3 chat template formatting

`training/` VLM scripts — `train_vlm_stage1.py` (projector only), `train_vlm_stage2.py` (LoRA SFT), `generate_vlm.py` (inference)

`eval/` VLM scripts — `run_lmms_eval.py` (lmms-eval benchmark runner), `convert_to_llava.py` (→ HF LLaVA format for vLLM)

`serve/convert_and_serve.sh` — one-step convert + vLLM serve on port 8000

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

`train_moellm_mcore_ddp.py` only supports TP=PP=EP=1 (pure data parallelism). NCCL required — Windows native multi-GPU NCCL is typically unavailable; use Linux/WSL2.
