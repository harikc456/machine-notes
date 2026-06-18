from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class MedusaConfig:
    # Model
    n_heads: int = 5
    d_model: int = 1536

    # Loss
    lambda_decay: float = 0.8

    # Teacher / data
    teacher_model_id: str = "google/gemma-4-e2b-it"
    dataset_id: str = "RyokoAI/ShareGPT52K"
    max_answer_len: int = 256
    max_seq_len: int = 768

    # Cache
    cache_dir: str = "medusa/cache"
    cache_shard_size: int = 5000
    val_split: float = 0.05

    # Optimiser — Muon for W1 (d_model×d_model), AdamW for W2 (vocab×d_model)
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    lr: float = 3e-4          # AdamW lr for W2 params
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 32
    n_epochs: int = 3
    warmup_steps: int = 200
    seed: int = 42


def load_config(path: str) -> MedusaConfig:
    data = yaml.safe_load(Path(path).read_text())
    return MedusaConfig(**data)
