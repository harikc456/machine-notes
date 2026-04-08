from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class SIGRegConfig:
    # ── Model dimensions ──────────────────────────────────────────────────────
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_hidden: int = 688
    dropout: float = 0.0          # typically 0 — no norms to pair with dropout

    # ── Attention ─────────────────────────────────────────────────────────────
    qk_norm: bool = True          # QK normalisation prevents attention entropy collapse
    seq_len: int = 512
    vocab_size: int = 65536

    # ── LM head ───────────────────────────────────────────────────────────────
    tie_embeddings: bool = True

    # ── Block architecture ────────────────────────────────────────────────────
    use_residual: bool = False      # add skip connections around attn and ffn
    norm_type: str = "none"         # "none" | "rmsnorm" | "layernorm"

    # ── SIGReg auxiliary loss ─────────────────────────────────────────────────
    # loss_type: which regulariser to apply at each collected layer
    #   "strong"  → ECF matching (all moments, forces Gaussian distribution)
    #   "weak"    → covariance matching (2nd moment only, forces spherical cloud)
    #   "both"    → strong + weak, weighted equally
    sigreg_loss_type: str = "strong"
    sigreg_weight: float = 0.1    # λ in: total_loss = ce_loss + λ * sigreg_loss
    sigreg_sketch_dim: int = 64   # projection dimension for the sketched ECF/cov estimator
    # Which layer outputs to regularise.
    #   "all"   → every block output
    #   "last"  → final block output only
    sigreg_layers: str = "all"

    # ── Training ──────────────────────────────────────────────────────────────
    seed: int = 42
    n_epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    def __post_init__(self) -> None:
        assert self.norm_type in ("none", "rmsnorm", "layernorm"), (
            f"norm_type must be 'none', 'rmsnorm', or 'layernorm', got '{self.norm_type}'"
        )
        assert self.sigreg_loss_type in ("strong", "weak", "both"), (
            f"sigreg_loss_type must be 'strong', 'weak', or 'both', got '{self.sigreg_loss_type}'"
        )
        assert self.sigreg_layers in ("all", "last"), (
            f"sigreg_layers must be 'all' or 'last', got '{self.sigreg_layers}'"
        )


def load_config(path: str | Path) -> SIGRegConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return SIGRegConfig()
    valid_fields = {f.name for f in SIGRegConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return SIGRegConfig(**raw)
