# idlm/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class IDLMConfig:
    # Required: path to a trained rbf_ffn CausalLM checkpoint (.pt file)
    ar_checkpoint: str

    # LoRA
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])

    # Training
    seq_len: int = 512
    batch_size: int = 8
    max_epochs: int = 10
    lr: float = 3e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 42

    # Evaluation / ISD
    eval_every_epochs: int = 1
    stride: int = 4
    num_eval_examples: int = 200
    prompt_len: int = 64
    gen_len: int = 128

    # Must match the AR checkpoint's vocab_size
    vocab_size: int = 50257


def load_config(path: str | Path) -> IDLMConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raise ValueError("Empty config file")
    valid = {f for f in IDLMConfig.__dataclass_fields__}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return IDLMConfig(**raw)
