"""~30M MoELLM preset with GPT-2 vocabulary (50257)."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

from model import MoELLM, MoELLMConfig
from model.config import AttentionType

# GPT-2 byte-level BPE
GPT2_VOCAB_SIZE = 50257
GPT2_EOS_TOKEN_ID = 50256

# Hand-tuned to land near 30M params with tied embeddings + 2-expert MoE (GQA baseline).
MOELLM_30M_CONFIG = MoELLMConfig(
    vocab_size=GPT2_VOCAB_SIZE,
    d_model=336,
    n_layers=6,
    n_heads=6,
    n_kv_heads=2,
    attention_type="mha",
    d_ff=960,
    max_seq_len=256,
    dropout=0.0,
    n_experts=2,
    num_experts_per_tok=1,
    router_aux_loss_coef=0.01,
)

PRESET_DIR = Path(__file__).resolve().parent / "moellm_30m"
ATTENTION_TYPES: tuple[AttentionType, ...] = ("mha", "mqa", "mla")


def _mla_dims_for_30m() -> dict[str, int]:
    head_dim = MOELLM_30M_CONFIG.d_model // MOELLM_30M_CONFIG.n_heads
    return {
        "q_lora_rank": 64,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": head_dim - 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": head_dim,
    }


def make_30m_config(attention_type: AttentionType = "mha") -> MoELLMConfig:
    """Return ~30M preset for MHA / MQA / MLA comparison."""
    base = replace(
        MOELLM_30M_CONFIG,
        attention_type=attention_type,
        n_kv_heads=MOELLM_30M_CONFIG.n_heads,
    )
    if attention_type == "mha":
        return base
    if attention_type == "mqa":
        return replace(base, n_kv_heads=1)
    if attention_type == "mla":
        return replace(base, **_mla_dims_for_30m())
    raise ValueError(f"unknown attention_type: {attention_type!r}")


def preset_dir_for(attention_type: AttentionType) -> Path:
    return Path(__file__).resolve().parent / f"moellm_30m_{attention_type}"


def count_parameters(config: MoELLMConfig | None = None) -> int:
    cfg = config or MOELLM_30M_CONFIG
    return sum(p.numel() for p in MoELLM(cfg).parameters())


def save_config(config: MoELLMConfig | None = None, out_dir: Path | None = None) -> Path:
    cfg = config or MOELLM_30M_CONFIG
    root = out_dir or PRESET_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    payload = asdict(cfg)
    payload["param_count"] = count_parameters(cfg)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_config(path: Path | None = None) -> MoELLMConfig:
    cfg_path = path or (PRESET_DIR / "config.json")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data.pop("param_count", None)
    return MoELLMConfig(**data)
