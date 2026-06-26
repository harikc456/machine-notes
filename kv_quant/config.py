from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QuantConfig:
    method: Literal["turboquant", "spectralquant"] = "turboquant"
    bits: int = 4               # key bits (TurboQuant: total bits for TurboQuantProd)
    value_bits: int = 2         # value bits for group quantization (TurboQuant only)
    value_group_size: int = 32  # group size for value quantization
    buffer_size: int = 128      # recent tokens kept in full precision (TurboQuant only)
    qjl_dim: int = 32           # QJL projection dim (SpectralQuant only)
    calibration_path: Optional[str] = None
    signal_bit_boost: float = 2.0
