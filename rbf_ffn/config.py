from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


# Maps deprecated model_type → (attn_type, ffn_type)
_MODEL_TYPE_MAP: dict[str, tuple[str, str]] = {
    "baseline":                 ("standard", "swiglu"),
    "rational":                 ("standard", "rational"),
    "rationalglu":              ("standard", "rationalglu"),
    "pfd_rational":             ("standard", "pfd_rational"),
    "pfd_rationalglu":          ("standard", "pfd_rationalglu"),
    "first_order_pfd_rational": ("standard", "first_order_pfd_rational"),
    "polar_mlp":                ("standard", "polar"),
    "polar_attn":               ("polar",    "swiglu"),
    "polar_full":               ("polar",    "polar"),
    "xsa":                      ("xsa",      "swiglu"),
}


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

    # Per-head learnable gain: each head is scaled by (1 + g), g init=0.
    # Applied after positional embeddings (post-RoPE for Q/K, post-split for V).
    qkv_gain: bool = False
    qkv_gain_targets: list[str] = field(default_factory=lambda: ["q", "k", "v"])  # subset of {"q","k","v"}

    # Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    n_kv_heads: int = 0            # 0 = match n_heads (MHA). 1 = MQA. 1 < n < n_heads = GQA.

    # Sequence / vocab
    seq_len: int = 512
    vocab_size: int = 50257

    # Composable block type
    attn_type: str = "standard"    # "standard" | "polar" | "xsa"
    ffn_type: str = "swiglu"       # "swiglu" | "leaky_relu_sq" | "rational" | "rationalglu" | "pfd_rational" | "pfd_rationalglu" | "first_order_pfd_rational" | "polar"
    norm_type: str = "rmsnorm"     # "rmsnorm" | "dynamic_erf"
    orthogonal_ffn: bool = False        # Wrap FFN output to be orthogonal to input x (all layers)
    orthogonal_ffn_layers: list[int] = field(default_factory=list)  # If non-empty, wrap only these layer indices (overrides orthogonal_ffn)
    orthogonal_ffn_eps: float = 1e-8    # Epsilon for orthogonal projection stability
    gated_orthogonal_ffn: bool = False              # Wrap FFN with GatedOrthogonalMLPWrapper (orthogonal novelty + gated amplification)
    gated_orthogonal_ffn_gate_activation: str = "tanh"  # Gate activation: "tanh" | "softsign" | "identity"

    # Deprecated: use attn_type + ffn_type instead.
    # If set, translated to attn_type + ffn_type in __post_init__.
    model_type: str | None = None

    ffn_hidden: int = 688          # FFN hidden dim (SwiGLU / RationalFFN)
    pfd_n: int = 4                 # Number of partial fraction terms for PFDRational* models

    # Mixture of Experts
    moe_n_experts: int = 8        # Total number of experts (used when ffn_type="moe")
    moe_top_k: int = 2            # Experts activated per token
    moe_orthogonal: bool = False  # Apply Gram-Schmidt to active expert outputs (router-score order)
    pre_lm_head_silu: bool = False # Apply SiLU activation before lm_head

    # Embedding / LM head
    tie_embeddings: bool = True        # If False, lm_head gets its own weight matrix (not shared with token_embedding)

    # Kronecker-factored LM head
    lm_head_kronecker: bool = False    # Replace lm_head with Kronecker-factored projection

    # Low-rank adapter on top of tied-embedding LM head
    lm_head_lora_rank: int = 0         # 0 = disabled; >0 = LoRALMHead with this rank (tie_embeddings must be True)

    # Kronecker-factored MLP projections
    kronecker_mlp: bool = False        # Replace nn.Linear in MLP/FFN layers with KroneckerLinear
    kronecker_delta_mlp: bool = False      # Replace up_proj/down_proj with KroneckerDeltaLinear
    kronecker_delta_rank: int = 16         # Rank of the low-rank delta pathway

    # KromHC head mixing
    use_kromhc: bool = False           # wrap any block with KromHC head mixing
    kromhc_mixer_hidden: int = 32      # hidden dim of per-factor weight MLP

    # Attention Residuals (AttnRes) — replaces depth-wise accumulation with
    # learned softmax attention over all preceding layer outputs (arXiv:2603.15031)
    use_attn_res: bool = False

    # Looped transformer (weight-shared middle block)
    use_loop: bool = False             # repeat a single shared middle block N times
    loop_n_repeats: int = 4            # how many times to repeat the shared middle block
    loop_n_fixed: int = 2              # fixed layers at each end (head + tail)

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

    # Maximal Update Parameterization (muP)
    mup: bool = False
    mup_base_width: int = 256    # proxy model width at which muon_lr/adamw_lr were tuned
    mup_init_std: float = 0.02   # init std at base width; hidden matrices scaled by sqrt(base/d)

    def __post_init__(self) -> None:
        if self.model_type is not None:
            if self.model_type not in _MODEL_TYPE_MAP:
                raise ValueError(
                    f"Unknown model_type '{self.model_type}'. "
                    f"Valid values: {sorted(_MODEL_TYPE_MAP)}. "
                    f"Prefer attn_type + ffn_type directly."
                )
            self.attn_type, self.ffn_type = _MODEL_TYPE_MAP[self.model_type]

        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        if self.n_kv_heads < 1:
            raise ValueError(f"n_kv_heads must be >= 1 (got {self.n_kv_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        if self.qkv_gain:
            valid_targets = {"q", "k", "v"}
            bad = set(self.qkv_gain_targets) - valid_targets
            if bad:
                raise ValueError(f"qkv_gain_targets contains invalid entries: {bad}. Must be subset of {valid_targets}")
            if not self.qkv_gain_targets:
                raise ValueError("qkv_gain_targets must be non-empty when qkv_gain=True")

        if self.norm_type not in ("rmsnorm", "dynamic_erf"):
            raise ValueError(
                f"norm_type must be 'rmsnorm' or 'dynamic_erf', got '{self.norm_type}'"
            )

        if self.use_loop and self.use_kromhc:
            raise ValueError("use_loop and use_kromhc cannot be used together")
        if self.use_attn_res and self.use_kromhc:
            raise ValueError("use_attn_res and use_kromhc cannot be used together")
        if self.use_attn_res and self.use_loop:
            raise ValueError("use_attn_res and use_loop cannot be used together")
        if self.use_loop and self.loop_n_fixed < 1:
            raise ValueError("loop_n_fixed must be >= 1")

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

        if self.mup and self.mup_base_width <= 0:
            raise ValueError(
                f"mup_base_width must be > 0 when mup=True, got {self.mup_base_width}"
            )


def load_config(path: str | Path) -> ModelConfig:
    """Load a ModelConfig from a YAML file.

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
