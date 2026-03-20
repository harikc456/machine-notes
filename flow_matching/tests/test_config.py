from __future__ import annotations
import pytest
from flow_matching.config import FlowConfig, load_config


def test_defaults():
    cfg = FlowConfig()
    assert cfg.data_root == "data/"
    assert cfg.seed == 42
    assert cfg.d_model == 384
    assert cfg.n_heads == 6
    assert cfg.n_layers == 12
    assert cfg.patch_size == 4
    assert cfg.mlp_ratio == pytest.approx(4.0)
    assert cfg.dropout == pytest.approx(0.0)
    assert cfg.p_uncond == pytest.approx(0.1)
    assert cfg.n_steps == 200_000
    assert cfg.batch_size == 128
    assert cfg.optimizer == "adamw"
    assert cfg.adamw_lr == pytest.approx(1e-4)
    assert cfg.muon_lr == pytest.approx(0.02)
    assert cfg.weight_decay == pytest.approx(0.0)
    assert cfg.warmup_ratio == pytest.approx(0.05)
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.log_every == 100
    assert cfg.sample_every == 5_000
    assert cfg.save_every == 10_000
    assert cfg.n_steps_euler == 100
    assert cfg.cfg_scale == pytest.approx(3.0)


def test_load_config_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("d_model: 64\nn_layers: 2\n")
    cfg = load_config(path)
    assert cfg.d_model == 64
    assert cfg.n_layers == 2
    assert cfg.n_steps == 200_000   # unspecified → default


def test_load_config_unknown_key_raises(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("unknown_key: 99\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_load_config_empty_yaml_returns_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("")
    cfg = load_config(path)
    assert cfg.d_model == 384


def test_invalid_optimizer_raises():
    with pytest.raises(ValueError, match="optimizer"):
        FlowConfig(optimizer="sgd")


def test_invalid_dropout_raises():
    with pytest.raises(ValueError, match="dropout"):
        FlowConfig(dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        FlowConfig(dropout=-0.1)


def test_invalid_p_uncond_raises():
    with pytest.raises(ValueError, match="p_uncond"):
        FlowConfig(p_uncond=0.0)
    with pytest.raises(ValueError, match="p_uncond"):
        FlowConfig(p_uncond=1.0)
