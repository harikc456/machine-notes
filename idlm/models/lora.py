# idlm/models/lora.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """
    Frozen base linear + trainable low-rank adapter.

    Set `current_mask` (shape: ..., 1) before each forward to control which
    positions use LoRA. Positions with mask=0 use only the frozen base weights.
    current_mask=None → LoRA delta always added (use when position control isn't needed).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        d_out, d_in = base.weight.shape
        self.base = base
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.current_mask: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x)) * self.scale
        if self.current_mask is not None:
            delta = delta * self.current_mask
        return out + delta


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
) -> None:
    """Recursively replace named nn.Linear children in target_modules with LoRALinear."""
    for name, module in list(model.named_children()):
        if name in target_modules and isinstance(module, nn.Linear):
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)
            setattr(model, name, LoRALinear(module, rank, alpha))
        else:
            apply_lora(module, target_modules, rank, alpha)
