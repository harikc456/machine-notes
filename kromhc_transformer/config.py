from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class KromHCConfig:
    """Configuration for KromHC Transformer training."""

    # Model dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # KromHC-specific
    model_type: str = "kromhc"      # "baseline" | "kromhc"
    use_kromhc: bool = True         # Enable head mixing
    qk_norm: bool = True            # Enable QK normalization

    # FFN
    ffn_hidden: int = 688

    # Sequence / vocab
    seq_len: int = 512
    vocab_size: int = 50257         # r50k_base tokenizer

    # Training
    seed: int = 42
    n_epochs: int = 10
    batch_size: int = 32
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    adamw_wd: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
    grad_accum_steps: int = 1


def load_config(path: str | Path) -> KromHCConfig:
    """Load KromHCConfig from YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return KromHCConfig()
    valid_fields = {f.name for f in KromHCConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return KromHCConfig(**raw)
