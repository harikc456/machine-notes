from dataclasses import dataclass


@dataclass
class HyViTTinyConfig:
    """HyViT-Tiny: ~5M params. Fast iteration on CIFAR-10."""
    # Vision
    img_size: int        = 32
    patch_size: int      = 4
    in_channels: int     = 3
    num_classes: int     = 10

    # Model
    d_model: int         = 192    # Euclidean dim; Lorentz dim is d_model+1
    n_heads: int         = 3
    n_blocks: int        = 12
    mlp_ratio: int       = 4
    dropout: float       = 0.1
    embed_dropout: float = 0.1

    # Training
    batch_size: int      = 128
    lr: float            = 3e-4
    weight_decay: float  = 0.05
    epochs: int          = 100
    warmup_epochs: int   = 10
    grad_clip: float     = 1.0
    label_smoothing: float = 0.1

    # Paths
    data_root: str       = "data/cifar10"
    checkpoint_dir: str  = "checkpoints/hyvit_tiny"


@dataclass
class HyViTSmallConfig:
    """HyViT-Small: ~22M params."""
    img_size: int        = 32
    patch_size: int      = 4
    in_channels: int     = 3
    num_classes: int     = 10

    d_model: int         = 384
    n_heads: int         = 6
    n_blocks: int        = 12
    mlp_ratio: int       = 4
    dropout: float       = 0.1
    embed_dropout: float = 0.1

    batch_size: int      = 64
    lr: float            = 1e-4
    weight_decay: float  = 0.05
    epochs: int          = 200
    warmup_epochs: int   = 20
    grad_clip: float     = 1.0
    label_smoothing: float = 0.1

    data_root: str       = "data/cifar10"
    checkpoint_dir: str  = "checkpoints/hyvit_small"


@dataclass
class HypLMConfig:
    """HypLM: hyperbolic causal language model on WikiText-103."""
    # Vocabulary / sequence
    vocab_size: int      = 50304   # r50k_base (50257) padded to multiple of 64
    seq_len: int         = 256

    # Model (Euclidean dim; Lorentz dim is d_model+1)
    d_model: int         = 256
    n_heads: int         = 4
    n_blocks: int        = 6
    mlp_ratio: int       = 4
    dropout: float       = 0.1
    embed_dropout: float = 0.1

    # Training
    batch_size: int       = 16    # physical batch per step
    grad_accum_steps: int = 4     # effective batch = batch_size * grad_accum_steps = 64
    lr: float             = 3e-4
    weight_decay: float   = 0.1
    n_epochs: int         = 10
    warmup_ratio: float   = 0.05
    grad_clip: float      = 1.0
    seed: int             = 42

    # Paths
    checkpoint_dir: str  = "checkpoints/hyplm"


@dataclass
class EucLMConfig:
    """EucLM: Euclidean causal LM baseline, matched size to HypLMConfig."""
    vocab_size: int      = 50304
    seq_len: int         = 256

    d_model: int         = 256
    n_heads: int         = 4
    n_blocks: int        = 6
    mlp_ratio: int       = 4
    dropout: float       = 0.1
    embed_dropout: float = 0.1

    batch_size: int       = 16
    grad_accum_steps: int = 4
    lr: float             = 3e-4
    weight_decay: float   = 0.1
    n_epochs: int         = 10
    warmup_ratio: float   = 0.05
    grad_clip: float      = 1.0
    seed: int             = 42

    checkpoint_dir: str  = "checkpoints/euclm"
