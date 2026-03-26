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

from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.train import make_lr_lambda, train


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

