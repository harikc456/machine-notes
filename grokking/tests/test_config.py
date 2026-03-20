from __future__ import annotations
import pytest
from grokking.config import GrokConfig, load_config


def test_defaults():
    cfg = GrokConfig()
    assert cfg.p == 97
    assert cfg.operation == "add"
    assert cfg.train_fraction == 0.4
    assert cfg.seed == 42
    assert cfg.d_model == 128
    assert cfg.n_heads == 4
    assert cfg.n_layers == 2
    assert cfg.dropout == 0.0
    assert cfg.n_steps == 50_000
    assert cfg.batch_size == 512
    assert cfg.optimizer == "adamw"
    assert cfg.adamw_lr == pytest.approx(1e-3)
    assert cfg.muon_lr == pytest.approx(0.02)
    assert cfg.weight_decay == pytest.approx(1.0)
    assert cfg.warmup_ratio == pytest.approx(0.01)
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.log_every == 10


def test_load_config_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("p: 13\noperation: mul\n")
    cfg = load_config(path)
    assert cfg.p == 13
    assert cfg.operation == "mul"
    assert cfg.n_steps == 50_000   # unspecified → default


def test_load_config_unknown_key_raises(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("unknown_key: 99\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_load_config_empty_yaml_returns_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("")
    cfg = load_config(path)
    assert cfg.p == 97
