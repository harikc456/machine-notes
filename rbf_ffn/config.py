from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    # Model dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # Attention
    qk_norm: bool = False          # Enable QK normalization in attention
    qkv_silu: bool = False         # Apply SiLU after Q, K, V projections

    # Sequence / vocab
    seq_len: int = 512
    vocab_size: int = 50257

    # Model type
    model_type: str = "baseline"   # "baseline" | "rational" | "rationalglu" | "pfd_rational" | "pfd_rationalglu" | "first_order_pfd_rational" | "polar_mlp" | "polar_attn" | "polar_full"
    ffn_hidden: int = 688          # FFN hidden dim (SwiGLU / RationalFFN)
    pfd_n: int = 4                 # Number of partial fraction terms for PFDRational* models
    pre_lm_head_silu: bool = False # Apply SiLU activation before lm_head

    # Embedding / LM head
    tie_embeddings: bool = True        # If False, lm_head gets its own weight matrix (not shared with token_embedding)

    # Kronecker-factored LM head
    # Replaces lm_head dense weight W ∈ R^{V×H} with an implicit A⊗B.
    # Factor dimensions are derived automatically: find the most balanced factor
    # pair (p, m) of vocab_size (p ≈ m ≈ √V) and (q, n) of d_model (q ≈ n ≈ √H).
    # Forward: reshape h → H_mat ∈ R^{n×q}, compute Z_mat = B H_mat A^T, flatten.
    # W never materialises in memory; avoids rank suppression during back-prop.
    lm_head_kronecker: bool = False    # Replace lm_head with Kronecker-factored projection

    # Kronecker-factored MLP projections
    kronecker_mlp: bool = False        # Replace nn.Linear in MLP/FFN layers with KroneckerLinear
    kronecker_delta_mlp: bool = False      # Replace up_proj/down_proj with KroneckerDeltaLinear
    kronecker_delta_rank: int = 16         # Rank of the low-rank delta pathway

    # KromHC head mixing
    use_kromhc: bool = False           # wrap any block with KromHC head mixing
    kromhc_mixer_hidden: int = 32      # hidden dim of per-factor weight MLP

    # Weight normalization
    linear_weight_norm: bool = False   # Normalise each linear layer's weight rows after every optimizer step
    linear_weight_norm_value: float = 2.0  # Target L2 norm per output neuron
    linear_weight_norm_max_only: bool = False  # Scale down only; do not scale up if norm is below target

    # Activation coefficient normalization
    activation_norm: bool = False      # Normalise rational/PFD activation coefficients to L2 norm 2.0 after every optimizer step

    # Adaptive weight normalization (depth-based)
    adaptive_weight_norm: bool = False
    adaptive_norm_early: float = 2.5   # target norm at layer 0
    adaptive_norm_late: float = 1.2    # target norm at layer L-1 (must be >= 1.0)
    adaptive_norm_gamma: float = 0.3   # max phase correction magnitude
    adaptive_norm_beta: float = 5.0    # tanh sensitivity to gap derivative
    adaptive_norm_alpha: float = 0.9   # EMA smoothing factor for log-gap

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

    def __post_init__(self) -> None:
        if self.adaptive_weight_norm:
            if self.adaptive_norm_late < 1.0:
                raise ValueError(
                    f"adaptive_norm_late must be >= 1.0, got {self.adaptive_norm_late}"
                )
            if self.adaptive_norm_early <= self.adaptive_norm_late:
                raise ValueError(
                    f"adaptive_norm_early must be > adaptive_norm_late, "
                    f"got adaptive_norm_early={self.adaptive_norm_early} <= adaptive_norm_late={self.adaptive_norm_late}"
                )


def load_config(path: str | Path) -> ModelConfig:
    """Load an ModelConfig from a YAML file.

    The YAML file may specify any subset of ModelConfig fields; unspecified
    fields take their dataclass defaults. Unknown keys raise ValueError.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return ModelConfig()
    valid_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return ModelConfig(**raw)
