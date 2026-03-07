"""
Optimizer factory for HyViT.

Uses AdamW for all parameters, with two groups:
  - Group 1 (weight decay applies): weight matrices in Linear/LorentzLinear
  - Group 2 (no weight decay):      biases, norms (gamma/beta), embeddings, cls_token

The manifold structure is encoded in the forward pass (explicit projection ops),
not in the parameters themselves, so standard AdamW is correct here.
"""

import torch


# Parameter names that should NOT receive weight decay
NO_DECAY_NAMES = {"bias", "gamma", "beta", "pos_embed", "cls_token"}


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build AdamW with separate weight-decay parameter groups."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in NO_DECAY_NAMES) or param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
