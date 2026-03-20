from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import yaml


@dataclass
class FlowConfig:
    # Data
    data_root: str = "data/"
    seed: int = 42

    # Model
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 12
    patch_size: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Flow matching
    p_uncond: float = 0.1

    # Training
    n_steps: int = 200_000
    batch_size: int = 128
    optimizer: str = "adamw"       # adamw | muon
    adamw_lr: float = 1e-4
    muon_lr: float = 0.02          # only used when optimizer=muon
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 100
    sample_every: int = 5_000
    save_every: int = 10_000

    # Sampling (also used for in-training grids)
    n_steps_euler: int = 100
    cfg_scale: float = 3.0

    def __post_init__(self) -> None:
        valid_opts = {"adamw", "muon"}
        if self.optimizer not in valid_opts:
            raise ValueError(f"optimizer must be one of {valid_opts}, got {self.optimizer!r}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 < self.p_uncond < 1.0):
            raise ValueError(f"p_uncond must be in (0, 1), got {self.p_uncond}")


def load_config(path: str | Path) -> FlowConfig:
    """Load a FlowConfig from a YAML file.

    Unspecified fields take dataclass defaults. Unknown keys raise ValueError.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return FlowConfig()
    valid_fields = {f.name for f in fields(FlowConfig)}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return FlowConfig(**raw)
