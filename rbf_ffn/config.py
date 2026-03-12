from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class RBFFFNConfig:
    # Model dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # RBF kernel
    K: int = 5
    centers: list[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    sigma_init: float = 0.5  # initial bandwidth; matches grid spacing
    sigma_variant: str = "global"  # one of: "global", "per_center", "per_dim"

    # Gate variant: one of "G0", "G1A", "G1B", "G2"
    gate_variant: str = "G0"
    sinkhorn_iters: int = 20  # only used by G2

    # Training
    num_classes: int = 10
    seq_len: int = 65
    vocab_size: int = 50257


def load_config(path: str | Path) -> RBFFFNConfig:
    """Load an RBFFFNConfig from a YAML file.

    The YAML file may specify any subset of RBFFFNConfig fields; unspecified
    fields take their dataclass defaults. Unknown keys raise ValueError.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return RBFFFNConfig()
    valid_fields = {f.name for f in RBFFFNConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return RBFFFNConfig(**raw)
