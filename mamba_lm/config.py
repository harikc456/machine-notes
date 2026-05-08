from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class MambaConfig:
    # Model
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16       # SSM state dimension (N in the paper)
    d_conv: int = 4         # depthwise conv kernel size
    expand: int = 2         # inner expansion factor; d_inner = d_model * expand
    dt_rank: int = -1       # Δ projection rank; -1 → ceil(d_model / 16)
    dt_min: float = 0.001   # initial Δ lower bound
    dt_max: float = 0.1     # initial Δ upper bound
    dt_init_floor: float = 1e-4

    # Vocab / sequence
    vocab_size: int = 50257
    seq_len: int = 512
    tie_embeddings: bool = True

    # Training
    seed: int = 42
    n_epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    def __post_init__(self) -> None:
        if self.dt_rank <= 0:
            self.dt_rank = math.ceil(self.d_model / 16)


def load_config(path: str | Path) -> MambaConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return MambaConfig()
    valid = {f for f in MambaConfig.__dataclass_fields__}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return MambaConfig(**raw)
