"""
Smoke tests for the grokking training loop.
Uses tiny configs and patches Muon with an AdamW stub for CPU testing.
"""
from __future__ import annotations
import json
from unittest.mock import patch

import pytest
import torch
from torch.optim import AdamW

from grokking.config import GrokConfig
from grokking.train import make_lr_lambda, train


class _MuonStub(AdamW):
    """AdamW stub that accepts Muon's momentum kwarg — for CPU smoke tests only."""
    def __init__(self, params, lr=0.02, momentum=0.95, **kwargs):
        super().__init__(params, lr=lr)


def _tiny_cfg(**kwargs) -> GrokConfig:
    defaults = dict(
        p=7,
        operation="add",
        n_steps=30,
        log_every=10,
        batch_size=16,
        n_layers=1,
        d_model=32,
        n_heads=2,
        seed=0,
        warmup_ratio=0.0,
    )
    defaults.update(kwargs)
    return GrokConfig(**defaults)


# ── LR schedule ───────────────────────────────────────────────────────────────

def test_lr_lambda_warmup():
    fn = make_lr_lambda(warmup_steps=10, total_steps=100)
    assert fn(0)  == pytest.approx(0.0)
    assert fn(5)  == pytest.approx(0.5)
    assert fn(10) == pytest.approx(1.0)


def test_lr_lambda_cosine_floor():
    fn = make_lr_lambda(warmup_steps=0, total_steps=100)
    assert fn(0)   == pytest.approx(1.0)
    assert fn(100) == pytest.approx(0.1, abs=1e-4)


# ── Smoke tests ───────────────────────────────────────────────────────────────

def _run(cfg: GrokConfig, tmp_path, optimizer: str = "adamw"):
    cfg.optimizer = optimizer
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("")
    with patch("grokking.train.Muon", _MuonStub):
        return train(cfg, config_path=config_path)


def test_adamw_run_creates_metrics_jsonl(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path, optimizer="adamw")
    assert (exp_dir / "metrics.jsonl").exists()


def test_adamw_run_creates_plot_png(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path, optimizer="adamw")
    assert (exp_dir / "plot.png").exists()


def test_adamw_metrics_has_correct_fields(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path, optimizer="adamw")
    rows = [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 3   # 30 steps / log_every=10
    for row in rows:
        for key in ("step", "train_loss", "train_acc", "val_loss", "val_acc"):
            assert key in row


def test_muon_run_creates_metrics_jsonl(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path, optimizer="muon")
    assert (exp_dir / "metrics.jsonl").exists()


def test_n_steps_override(tmp_path):
    cfg = _tiny_cfg(n_steps=100)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("")
    with patch("grokking.train.Muon", _MuonStub):
        exp_dir = train(cfg, config_path=config_path, n_steps=20)
    rows = [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert rows[-1]["step"] == 20   # override respected


def test_config_yaml_copied_to_exp_dir(tmp_path):
    cfg = _tiny_cfg()
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("p: 7\n")
    with patch("grokking.train.Muon", _MuonStub):
        exp_dir = train(cfg, config_path=config_path)
    assert (exp_dir / "config.yaml").read_text() == "p: 7\n"


# ── All-operations smoke test (spec criterion 3) ──────────────────────────────

@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div", "x2_plus_xy_plus_y2"])
def test_all_operations_complete_without_error(operation, tmp_path):
    """Each supported operation must run through the training loop without error."""
    cfg = _tiny_cfg(operation=operation, n_steps=10, log_every=5)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("")
    with patch("grokking.train.Muon", _MuonStub):
        exp_dir = train(cfg, config_path=config_path)
    assert (exp_dir / "metrics.jsonl").exists()
