"""
Modular arithmetic dataset for grokking experiments.

Usage:
    from grokking.data import build_dataset, split_dataset
    inputs, labels = build_dataset(cfg)          # all valid (a,b) pairs
    train_inp, train_lbl, val_inp, val_lbl = split_dataset(inputs, labels, cfg)
"""
from __future__ import annotations
import torch
from grokking.config import GrokConfig

_VALID_OPS = {"add", "sub", "mul", "div", "x2_plus_xy_plus_y2"}


def _compute_label(a: int, b: int, operation: str, p: int) -> int:
    if operation == "add":
        return (a + b) % p
    elif operation == "sub":
        return (a - b) % p
    elif operation == "mul":
        return (a * b) % p
    elif operation == "div":
        return (a * pow(b, -1, p)) % p
    elif operation == "x2_plus_xy_plus_y2":
        return (a ** 2 + a * b + b ** 2) % p
    else:
        raise ValueError(f"Unknown operation: {operation!r}. Valid: {_VALID_OPS}")


def build_dataset(cfg: GrokConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, labels) tensors for all valid (a, b) pairs.

    inputs: LongTensor (N, 4) — each row is [a, op_token, b, eq_token]
    labels: LongTensor (N,)   — result of operation mod p
    """
    op_tok = cfg.p       # token id for the operation symbol
    eq_tok = cfg.p + 1   # token id for '='

    sequences: list[list[int]] = []
    labels: list[int] = []

    for a in range(cfg.p):
        for b in range(cfg.p):
            if cfg.operation == "div" and b == 0:
                continue
            sequences.append([a, op_tok, b, eq_tok])
            labels.append(_compute_label(a, b, cfg.operation, cfg.p))

    inputs = torch.tensor(sequences, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return inputs, labels_t


def split_dataset(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    cfg: GrokConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproducible random train/val split.

    Returns (train_inputs, train_labels, val_inputs, val_labels).
    """
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    n = len(inputs)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * cfg.train_fraction)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]
