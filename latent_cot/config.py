from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

VALID_CONDITIONS = {"floor", "z", "ceiling", "z_shuffled", "reconstruct"}


@dataclass
class ExperimentConfig:
    # Backbone
    backbone: str = "google/gemma-4-E2B-it"

    # Reasoning-encoding bottleneck
    n_slots: int = 16          # K: number of latent slots
    d_z: int = 32              # per-slot bottleneck dim
    encoder_heads: int = 8     # heads in the encoder's cross-attention
    diffusion_steps: int = 6   # T: refinement steps for DiffusionReasoningEncoder

    # Condition: floor | z | ceiling | z_shuffled | reconstruct
    condition: str = "z"
    strip_annotations: bool = True   # drop <<...>> calculator spans from traces

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    seed: int = 42
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0

    # Tokenization budgets
    max_trace_tokens: int = 256
    max_question_tokens: int = 128
    max_answer_tokens: int = 16

    # Data subsetting (0 = use all)
    max_train_samples: int = 0
    max_eval_samples: int = 200

    # Runtime
    output_dir: str = "latent_cot/runs"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(
                f"condition must be one of {sorted(VALID_CONDITIONS)}, got {self.condition!r}"
            )
        if self.n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {self.n_slots}")
        if self.d_z < 1:
            raise ValueError(f"d_z must be >= 1, got {self.d_z}")
        if self.encoder_heads < 1:
            raise ValueError(f"encoder_heads must be >= 1, got {self.encoder_heads}")
        if self.d_z % self.encoder_heads != 0:
            raise ValueError(
                f"d_z ({self.d_z}) must be divisible by encoder_heads ({self.encoder_heads})"
            )
        if self.diffusion_steps < 1:
            raise ValueError(f"diffusion_steps must be >= 1, got {self.diffusion_steps}")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from YAML. Unspecified fields take defaults;
    unknown keys raise ValueError."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return ExperimentConfig()
    valid = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return ExperimentConfig(**raw)
