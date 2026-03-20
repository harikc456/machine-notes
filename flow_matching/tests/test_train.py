"""
Smoke tests for the flow matching training loop.
Uses tiny configs; patches CIFAR-100 and Muon to avoid downloads and GPU deps.
"""
from __future__ import annotations
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch.optim import AdamW

from flow_matching.config import FlowConfig


class _FakeCIFAR100:
    def __init__(self, *args, **kwargs):
        self.data    = np.random.randint(0, 256, (200, 32, 32, 3), dtype=np.uint8)
        self.targets = list(range(200))
        self.transform = kwargs.get("transform")
    def __len__(self):
        return 200
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx] % 100


class _MuonStub(AdamW):
    """AdamW stub that accepts Muon's momentum kwarg."""
    def __init__(self, params, lr=0.02, momentum=0.95, **kwargs):
        super().__init__(params, lr=lr)


def _tiny_cfg(**kwargs) -> FlowConfig:
    defaults = dict(
        d_model=64, n_layers=2, n_heads=2,
        batch_size=4, n_steps=6,
        log_every=2, sample_every=100, save_every=6,
        n_steps_euler=2, warmup_ratio=0.0,
    )
    defaults.update(kwargs)
    return FlowConfig(**defaults)


def _run(cfg: FlowConfig, tmp_path, optimizer: str = "adamw"):
    cfg.optimizer = optimizer
    cfg.data_root = str(tmp_path)
    config_path   = tmp_path / "cfg.yaml"
    config_path.write_text("")
    with (
        patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100),
        patch("flow_matching.train.Muon", _MuonStub),
    ):
        from flow_matching.train import train
        return train(cfg, config_path=config_path)


def test_train_creates_metrics_jsonl(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path)
    assert (exp_dir / "metrics.jsonl").exists()


def test_train_metrics_has_correct_fields(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path)
    rows = [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 3   # 6 steps / log_every=2
    for row in rows:
        assert "step"       in row
        assert "train_loss" in row
        assert "lr"         in row


def test_train_creates_plot_png(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path)
    assert (exp_dir / "plot.png").exists()


def test_train_creates_checkpoint(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path)
    ckpt_path = exp_dir / "ckpt.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model" in ckpt
    assert ckpt["step"] == 6


def test_train_n_steps_override(tmp_path):
    cfg = _tiny_cfg(n_steps=20)
    cfg.data_root  = str(tmp_path)
    config_path    = tmp_path / "cfg.yaml"
    config_path.write_text("")
    with (
        patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100),
        patch("flow_matching.train.Muon", _MuonStub),
    ):
        from flow_matching.train import train
        exp_dir = train(cfg, config_path=config_path, n_steps=6)
    rows = [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert rows[-1]["step"] == 6


def test_train_config_yaml_copied(tmp_path):
    cfg = _tiny_cfg()
    cfg.data_root = str(tmp_path)
    config_path   = tmp_path / "cfg.yaml"
    config_path.write_text("d_model: 64\n")
    with (
        patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100),
        patch("flow_matching.train.Muon", _MuonStub),
    ):
        from flow_matching.train import train
        exp_dir = train(cfg, config_path=config_path)
    assert (exp_dir / "config.yaml").read_text() == "d_model: 64\n"


def test_train_muon_mode(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path, optimizer="muon")
    assert (exp_dir / "metrics.jsonl").exists()
