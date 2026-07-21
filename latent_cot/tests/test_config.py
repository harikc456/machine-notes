import pytest
from pathlib import Path
from latent_cot.config import ExperimentConfig, load_config, VALID_CONDITIONS


def test_defaults():
    cfg = ExperimentConfig()
    assert cfg.backbone == "google/gemma-4-E2B-it"
    assert cfg.n_slots == 16 and cfg.d_z == 32
    assert cfg.condition == "z"
    assert cfg.n_slots * cfg.d_z < 4096  # bottleneck must stay small


def test_bad_condition_raises():
    with pytest.raises(ValueError):
        ExperimentConfig(condition="banana")


def test_bad_slots_raises():
    with pytest.raises(ValueError):
        ExperimentConfig(n_slots=0)


def test_yaml_load_and_unknown_key(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("condition: ceiling\nn_slots: 8\n")
    cfg = load_config(p)
    assert cfg.condition == "ceiling" and cfg.n_slots == 8

    bad = tmp_path / "bad.yaml"
    bad.write_text("nonsense_key: 1\n")
    with pytest.raises(ValueError):
        load_config(bad)
