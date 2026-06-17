from __future__ import annotations
import math
import torch
import torch.nn as nn


class LoRALMHead(nn.Module):
    """
    Frozen teacher LM head weight with a trainable low-rank adapter.

    logits = x @ (W + B @ A).T
    where W is frozen, A and B are the LoRA matrices.

    B is initialised to zero so the adapter starts as a no-op.

    Input:  (B, S, d_teacher)
    Output: (B, S, vocab)
    """

    def __init__(self, frozen_weight: torch.Tensor, lora_rank: int = 16) -> None:
        super().__init__()
        vocab, d = frozen_weight.shape
        self.register_buffer("weight", frozen_weight.detach())
        self.lora_A = nn.Parameter(torch.empty(lora_rank, d))
        self.lora_B = nn.Parameter(torch.zeros(vocab, lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_teacher) → (B, S, vocab)"""
        W = self.weight + self.lora_B @ self.lora_A   # (vocab, d_teacher)
        return x @ W.T
