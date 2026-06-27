from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QuantConfig:
    method: Optional[Literal["turboquant", "spectralquant"]] = "turboquant"
    bits: int = 4               # key bits (TurboQuant/SpectralQuant only)
    value_bits: int = 2         # value bits for group quantization (TurboQuant only)
    value_group_size: int = 32  # group size for value quantization
    buffer_size: int = 128      # recent tokens kept in full precision (TurboQuant only)
    qjl_dim: int = 32           # QJL projection dim (SpectralQuant only)
    calibration_path: Optional[str] = None  # spectralquant: base path (no ext); triattention: stats .pt path
    signal_bit_boost: float = 2.0           # SpectralQuant only
    budget: int = 2048          # TriAttention: max KV tokens to retain after eviction
    divide_length: int = 128    # TriAttention: trigger eviction every N decode steps
    eviction: Optional[Literal["triattention"]] = None  # token eviction method (orthogonal to quantization)
