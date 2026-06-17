from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class MTPConfig:
    # Draft model dimensions
    d_draft: int = 512
    n_blocks: int = 4
    ffn_hidden: int = 1366        # SwiGLU hidden; 0 = auto as int(8/3 * d_draft)
    n_heads: int = 8
    dropout: float = 0.0
    use_xsa: bool = False         # XSA orthogonalisation (no-op in cross-attn, for future)

    # Teacher
    teacher_model_id: str = "google/gemma-4-e2b-it"
    teacher_layers: list[int] = field(default_factory=lambda: [3, 8, 14, 17])
    d_teacher: int = 2048

    # Training
    max_draft: int = 8
    lambda_decay: float = 0.8
    lora_rank: int = 16
    max_prompt_len: int = 256

    # Inference
    tau: float = 2.0
    max_tree_nodes: int = 256

    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 16
    n_epochs: int = 3
    warmup_steps: int = 200
    seed: int = 42

    # Data / cache
    cache_dir: str = "mtp_draft/cache"
    cache_n_answer_positions: int = 8
    cache_shard_size: int = 5000

    def __post_init__(self) -> None:
        if self.ffn_hidden == 0:
            self.ffn_hidden = int(8 / 3 * self.d_draft)
        n = len(self.teacher_layers)
        assert (n & (n - 1)) == 0, (
            f"len(teacher_layers) must be a power of 2 for KromHC; got {n}"
        )


def load_config(path: str) -> MTPConfig:
    data = yaml.safe_load(Path(path).read_text())
    return MTPConfig(**data)
