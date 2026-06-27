from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is on sys.path when imported standalone
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_MODELS_YAML = Path(__file__).parent / "models.yaml"


@dataclass
class ModelEntry:
    id: str
    label: str
    head_dim: int
    default_bits: int


@dataclass
class ChatStats:
    kv_memory_mb: float
    baseline_memory_mb: float
    tokens_per_sec: float
    compression_ratio: float


def load_models_yaml(path: Path | str | None = None) -> list[ModelEntry]:
    if path is None:
        path = _MODELS_YAML
    with open(path) as f:
        data = yaml.safe_load(f)
    return [ModelEntry(**m) for m in data["models"]]


def load_hf_model(
    model_id: str,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
    )
    model.eval()
    return model, tokenizer


def get_kv_shape(model) -> tuple[int, int]:
    cfg = model.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return n_kv_heads, head_dim


def get_stats(
    cache,
    n_new_tokens: int,
    elapsed: float,
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
) -> ChatStats:
    # baseline: K + V tensors, fp16 (2 bytes), all layers and heads
    baseline_bytes = n_layers * n_kv_heads * seq_len * head_dim * 2 * 2
    baseline_mb = baseline_bytes / 1e6

    if cache is not None:
        compressed = cache.compressed_bytes()
        kv_mb = compressed / 1e6
        ratio = baseline_bytes / max(compressed, 1)
    else:
        kv_mb = baseline_mb
        ratio = 1.0

    tps = n_new_tokens / max(elapsed, 1e-9)

    return ChatStats(
        kv_memory_mb=kv_mb,
        baseline_memory_mb=baseline_mb,
        tokens_per_sec=tps,
        compression_ratio=ratio,
    )
