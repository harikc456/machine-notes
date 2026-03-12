import pytest
from pathlib import Path
from rbf_ffn.config import RBFFFNConfig, load_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("filename,expected_variant", [
    ("g0_baseline.yaml",    "G0"),
    ("g1a_cross_kernel.yaml", "G1A"),
    ("g1b_input_driven.yaml", "G1B"),
    ("g2_sinkhorn.yaml",    "G2"),
])
def test_load_config_gate_variant(filename, expected_variant):
    cfg = load_config(CONFIGS_DIR / filename)
    assert cfg.gate_variant == expected_variant


def test_load_config_returns_rbfffnconfig():
    cfg = load_config(CONFIGS_DIR / "g0_baseline.yaml")
    assert isinstance(cfg, RBFFFNConfig)


def test_load_config_values_match_yaml():
    cfg = load_config(CONFIGS_DIR / "g0_baseline.yaml")
    assert cfg.d_model == 64
    assert cfg.K == 5
    assert cfg.sigma_init == 0.5
    assert cfg.n_layers == 2


def test_load_config_unknown_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("gate_variant: G0\nunknown_key: 99\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(bad)


def test_load_config_partial_yaml_uses_defaults(tmp_path):
    """A YAML with only gate_variant set uses dataclass defaults for the rest."""
    partial = tmp_path / "partial.yaml"
    partial.write_text("gate_variant: G1A\n")
    cfg = load_config(partial)
    assert cfg.gate_variant == "G1A"
    assert cfg.d_model == RBFFFNConfig().d_model   # default


def test_load_config_empty_yaml_uses_defaults(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    cfg = load_config(empty)
    assert cfg == RBFFFNConfig()
