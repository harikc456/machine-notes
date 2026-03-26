"""
Tests for train.py. Uses a tiny synthetic model and DataLoader — no
WikiText-103 download required.
"""
from __future__ import annotations
import json
import math
import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch

import torch.nn as nn
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.model import CausalLM
from rbf_ffn.train import make_lr_lambda, train, apply_adaptive_weight_norm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_cfg(**kwargs) -> RBFFFNConfig:
    """Minimal config for fast CPU tests (baseline model avoids RBF complexity)."""
    defaults = dict(
        model_type="baseline",
        d_model=32,
        n_heads=2,
        n_layers=2,
        ffn_hidden=64,
        seq_len=8,
        batch_size=2,
        n_epochs=1,
        seed=0,
        vocab_size=50,
        warmup_ratio=0.0,
        grad_clip=1.0,
        grad_accum_steps=1,
    )
    defaults.update(kwargs)
    return RBFFFNConfig(**defaults)


def _fake_loaders(cfg: RBFFFNConfig, n_train: int = 8):
    """Synthetic DataLoaders that do not touch disk.

    Each sequence has length seq_len+1 because the training loop slices
    batch[:, :-1] (inputs) and batch[:, 1:] (targets).
    """
    seq  = cfg.seq_len + 1
    data = torch.randint(0, cfg.vocab_size, (n_train, seq))
    ds   = TensorDataset(data)

    def collate(batch):
        return torch.stack([b[0] for b in batch])

    loader = DataLoader(ds, batch_size=cfg.batch_size,
                        shuffle=False, collate_fn=collate)
    return loader, loader, loader   # train, val, test


class _MuonStub(AdamW):
    """AdamW stub for CPU testing. Accepts and ignores Muon-specific kwargs
    (momentum, etc.) — this stub tests training loop control flow only,
    not optimizer correctness."""
    def __init__(self, params, lr=1e-3, momentum=0.95, **kwargs):
        super().__init__(params, lr=lr)


# ── LR schedule tests (pure function, no model needed) ───────────────────────

def test_make_lr_lambda_warmup():
    """LR ramps linearly from 0 to 1 during warmup."""
    fn = make_lr_lambda(warmup_steps=10, total_steps=100)
    assert fn(0) == pytest.approx(0.0)
    assert fn(5) == pytest.approx(0.5)
    assert fn(10) == pytest.approx(1.0)


def test_make_lr_lambda_cosine_decay_floor():
    """After warmup, LR decays to the 0.1 floor at total_steps."""
    fn = make_lr_lambda(warmup_steps=0, total_steps=100)
    assert fn(0)   == pytest.approx(1.0)
    assert fn(100) == pytest.approx(0.1, abs=1e-4)


# ── Integration smoke tests ───────────────────────────────────────────────────

def _run_train(cfg: RBFFFNConfig, tmp_path, n_train: int = 8):
    """Helper: run train() with synthetic data and stubbed Muon."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("")   # train() copies config; content irrelevant
    loaders = _fake_loaders(cfg, n_train=n_train)
    with patch("rbf_ffn.train.get_dataloaders", return_value=loaders), \
         patch("rbf_ffn.train.Muon", _MuonStub):
        return train(cfg, config_path=config_path)


def test_training_completes_and_writes_metrics(tmp_path):
    """Baseline sanity: training finishes and produces metrics.jsonl."""
    cfg = _tiny_cfg()
    exp_dir = _run_train(cfg, tmp_path)
    assert (exp_dir / "metrics.jsonl").exists()
    assert (exp_dir / "checkpoint_final.pt").exists()


def test_effective_batch_size_in_metrics(tmp_path):
    """metrics.jsonl must contain effective_batch_size = batch_size * grad_accum_steps."""
    cfg = _tiny_cfg(grad_accum_steps=2, batch_size=2)
    exp_dir = _run_train(cfg, tmp_path, n_train=8)
    rows = [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert rows[-1]["effective_batch_size"] == 2 * 2   # batch_size=2, grad_accum_steps=2


def test_grad_accum_default_matches_no_accum(tmp_path):
    """With grad_accum_steps=1 (default), effective_batch_size equals batch_size."""
    cfg = _tiny_cfg(grad_accum_steps=1, batch_size=2)
    exp_dir = _run_train(cfg, tmp_path, n_train=8)
    row = json.loads(
        (exp_dir / "metrics.jsonl").read_text().strip().splitlines()[-1]
    )
    assert row["effective_batch_size"] == 2


def test_epoch_end_flush_completes_without_error(tmp_path):
    """Training completes when steps_per_epoch is not divisible by grad_accum_steps,
    exercising the epoch-end gradient flush path."""
    # n_train=6, batch_size=2 → 3 steps/epoch; 3 % grad_accum_steps=4 != 0 → flush fires
    cfg = _tiny_cfg(grad_accum_steps=4, batch_size=2)
    exp_dir = _run_train(cfg, tmp_path, n_train=6)
    assert (exp_dir / "metrics.jsonl").exists()
    row = json.loads(
        (exp_dir / "metrics.jsonl").read_text().strip().splitlines()[-1]
    )
    assert row["effective_batch_size"] == 2 * 4


# ── apply_adaptive_weight_norm ────────────────────────────────────────────────

def _adaptive_cfg(n_layers: int = 4, **kwargs) -> RBFFFNConfig:
    defaults = dict(
        model_type="baseline",
        d_model=32,
        n_heads=2,
        n_layers=n_layers,
        ffn_hidden=64,
        seq_len=8,
        vocab_size=50,
        seed=0,
        adaptive_weight_norm=True,
        adaptive_norm_early=2.5,
        adaptive_norm_late=1.2,
        adaptive_norm_gamma=0.3,
        adaptive_norm_beta=5.0,
        adaptive_norm_alpha=0.9,
    )
    defaults.update(kwargs)
    return RBFFFNConfig(**defaults)


def test_adaptive_weight_norm_zero_delta_matches_static_schedule():
    """With delta_log_gap=0.0 the correction term is zero, so row norms equal
    the static depth schedule exactly."""
    cfg = _adaptive_cfg(n_layers=4)
    model = CausalLM(cfg)
    apply_adaptive_weight_norm(model, cfg, delta_log_gap=0.0)

    L = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        expected = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - frac)
        for module in block.modules():
            if isinstance(module, nn.Linear):
                row_norms = module.weight.data.norm(dim=1)
                assert torch.allclose(
                    row_norms,
                    torch.full_like(row_norms, expected),
                    atol=1e-5,
                ), f"layer {layer_idx}: expected {expected:.4f}, got mean {row_norms.mean():.4f}"


def test_adaptive_weight_norm_floor_never_below_one():
    """Row norms never fall below 1.0 for any delta_log_gap, including
    extreme values that would push correction > static."""
    cfg = _adaptive_cfg(n_layers=4)
    model = CausalLM(cfg)

    for delta in [100.0, -100.0, 0.0, 0.5, -0.5]:
        apply_adaptive_weight_norm(model, cfg, delta_log_gap=delta)
        for block in model.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    row_norms = module.weight.data.norm(dim=1)
                    assert (row_norms >= 1.0 - 1e-5).all(), \
                        f"norm < 1.0 with delta={delta}: {row_norms.min():.4f}"


def test_adaptive_weight_norm_gamma_zero_disables_phase_correction():
    """gamma=0 makes the correction term identically zero regardless of delta,
    so any delta_log_gap produces the same result as delta=0."""
    cfg = _adaptive_cfg(n_layers=3, adaptive_norm_gamma=0.0)
    model = CausalLM(cfg)
    apply_adaptive_weight_norm(model, cfg, delta_log_gap=50.0)

    L = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        expected = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - frac)
        for module in block.modules():
            if isinstance(module, nn.Linear):
                row_norms = module.weight.data.norm(dim=1)
                assert torch.allclose(
                    row_norms,
                    torch.full_like(row_norms, expected),
                    atol=1e-5,
                )


def test_adaptive_weight_norm_early_greater_than_late():
    """Layer 0 has higher row norms than layer L-1 when delta=0."""
    cfg = _adaptive_cfg(n_layers=4)
    model = CausalLM(cfg)
    apply_adaptive_weight_norm(model, cfg, delta_log_gap=0.0)

    def mean_row_norm(block):
        norms = []
        for module in block.modules():
            if isinstance(module, nn.Linear):
                norms.append(module.weight.data.norm(dim=1).mean().item())
        return sum(norms) / len(norms)

    norm_first = mean_row_norm(model.blocks[0])
    norm_last  = mean_row_norm(model.blocks[-1])
    assert norm_first > norm_last, \
        f"expected layer 0 norm ({norm_first:.4f}) > layer L-1 norm ({norm_last:.4f})"


def test_adaptive_weight_norm_positive_delta_tightens_late_layers():
    """A positive delta_log_gap (gap growing = memorization) reduces the target
    norm below static for late layers. Specifically, with delta > 0 and gamma > 0,
    the last layer's row norms must be strictly less than its static target."""
    cfg = _adaptive_cfg(n_layers=4)
    model = CausalLM(cfg)

    # Compute expected static norm for the last layer (frac = 1.0)
    L = len(model.blocks)
    last_frac = (L - 1) / max(L - 1, 1)   # = 1.0
    static_last = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - last_frac)
    # static_last == adaptive_norm_late == 1.2

    # Apply with large positive delta → correction is positive → target < static_last
    apply_adaptive_weight_norm(model, cfg, delta_log_gap=10.0)

    for module in model.blocks[-1].modules():
        if isinstance(module, nn.Linear):
            row_norms = module.weight.data.norm(dim=1)
            # target should be max(1.0, 1.2 - correction) where correction > 0
            assert (row_norms < static_last - 1e-5).all(), \
                f"expected late-layer norm < static ({static_last:.4f}), got {row_norms.mean():.4f}"
