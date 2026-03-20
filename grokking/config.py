from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class GrokConfig:
    # Data
    p: int = 97
    operation: str = "add"       # add | sub | mul | div | x2_plus_xy_plus_y2
    train_fraction: float = 0.4
    seed: int = 42

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0

    # Training
    n_steps: int = 50_000
    batch_size: int = 512
    optimizer: str = "adamw"     # adamw | muon
    adamw_lr: float = 1e-3
    muon_lr: float = 0.02        # only used when optimizer=muon
    weight_decay: float = 1.0
    warmup_ratio: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 10


def load_config(path: str | Path) -> GrokConfig:
    """Load a GrokConfig from a YAML file.

    Unspecified fields take dataclass defaults. Unknown keys raise ValueError.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return GrokConfig()
    valid_fields = {f.name for f in GrokConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return GrokConfig(**raw)
