from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QuantConfig:
    method: Literal["turboquant", "spectralquant"] = "turboquant"
    bits: int = 4
    qjl_dim: int = 32
    calibration_path: Optional[str] = None
    signal_bit_boost: float = 2.0
