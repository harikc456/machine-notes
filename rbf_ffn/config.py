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
    sigma_init: float = 0.5
    sigma_variant: str = "global"  # "global" | "per_center" | "per_dim"

    # Gate variant
    gate_variant: str = "G0"       # "G0" | "G1A" | "G1B" | "G2"
    sinkhorn_iters: int = 20       # G2 only

    # Attention
    qk_norm: bool = False          # Enable QK normalization in attention

    # Sequence / vocab
    seq_len: int = 512
    vocab_size: int = 50257

    # Model type
    model_type: str = "rbf"        # "baseline" | "rbf" | "rational" | "rationalglu" | "pfd_rational" | "pfd_rationalglu" | "first_order_pfd_rational"
    ffn_hidden: int = 688          # FFN hidden dim (SwiGLU / RationalFFN); ignored by RBF model
    pfd_n: int = 4                 # Number of partial fraction terms for PFDRational* models

    # Training
    seed: int = 42
    n_epochs: int = 10
    batch_size: int = 32
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    adamw_wd: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
    grad_accum_steps: int = 1      # mini-batches per optimizer step; 1 = no accumulation


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
