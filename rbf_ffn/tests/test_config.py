import pytest
from pathlib import Path
from rbf_ffn.config import RBFFFNConfig, load_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_load_config_returns_rbfffnconfig():
    cfg = load_config(CONFIGS_DIR / "baseline.yaml")
    assert isinstance(cfg, RBFFFNConfig)


def test_load_config_values_match_yaml():
    cfg = load_config(CONFIGS_DIR / "baseline.yaml")
    assert cfg.d_model == 256
    assert cfg.n_layers == 6


def test_load_config_unknown_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_type: baseline\nunknown_key: 99\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(bad)


def test_load_config_partial_yaml_uses_defaults(tmp_path):
    """A YAML with only model_type set uses dataclass defaults for the rest."""
    partial = tmp_path / "partial.yaml"
    partial.write_text("model_type: rationalglu\n")
    cfg = load_config(partial)
    assert cfg.model_type == "rationalglu"
    assert cfg.d_model == RBFFFNConfig().d_model   # default


def test_load_config_empty_yaml_uses_defaults(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    cfg = load_config(empty)
    assert cfg == RBFFFNConfig()


# ── New training field defaults ───────────────────────────────────────────────

def test_new_fields_have_correct_defaults():
    cfg = RBFFFNConfig()
    assert cfg.model_type == "baseline"
    assert cfg.ffn_hidden == 688
    assert cfg.seed == 42
    assert cfg.n_epochs == 10
    assert cfg.batch_size == 32
    assert abs(cfg.muon_lr - 0.02) < 1e-9
    assert abs(cfg.adamw_lr - 3e-4) < 1e-9
    assert abs(cfg.adamw_wd - 0.1) < 1e-9
    assert abs(cfg.warmup_ratio - 0.02) < 1e-9
    assert abs(cfg.grad_clip - 1.0) < 1e-9


def test_seq_len_default_updated():
    """seq_len default changes from 65 → 512 for WikiText-103."""
    cfg = RBFFFNConfig()
    assert cfg.seq_len == 512


def test_load_config_accepts_model_type(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("model_type: baseline\n")
    cfg = load_config(p)
    assert cfg.model_type == "baseline"


def test_load_config_accepts_ffn_hidden(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("ffn_hidden: 1024\n")
    cfg = load_config(p)
    assert cfg.ffn_hidden == 1024


def test_grad_accum_steps_default():
    """Default must be 1 — identity, no behavioral change for existing configs."""
    cfg = RBFFFNConfig()
    assert cfg.grad_accum_steps == 1


def test_grad_accum_steps_yaml(tmp_path):
    """YAML can set grad_accum_steps; load_config accepts it."""
    p = tmp_path / "accum.yaml"
    p.write_text("grad_accum_steps: 4\n")
    cfg = load_config(p)
    assert cfg.grad_accum_steps == 4


def test_existing_yamls_load_without_grad_accum_steps():
    """All existing YAML configs remain valid (no grad_accum_steps key needed)."""
    all_yamls = [
        "baseline.yaml",
        "baseline_qk_norm.yaml",
        "baseline_weight_norm.yaml",
        "rational_ffn.yaml",
        "rationalglu_ffn.yaml",
        "pfd_rationalglu_ffn.yaml",
        "first_order_pfd_rational_ffn.yaml",
    ]
    for name in all_yamls:
        cfg = load_config(CONFIGS_DIR / name)
        assert cfg.grad_accum_steps == 1, f"{name} should default to 1"


# ── Adaptive weight norm fields ───────────────────────────────────────────────

def test_adaptive_weight_norm_defaults():
    cfg = RBFFFNConfig()
    assert cfg.adaptive_weight_norm is False
    assert cfg.adaptive_norm_early == pytest.approx(2.5)
    assert cfg.adaptive_norm_late  == pytest.approx(1.2)
    assert cfg.adaptive_norm_gamma == pytest.approx(0.3)
    assert cfg.adaptive_norm_beta  == pytest.approx(5.0)
    assert cfg.adaptive_norm_alpha == pytest.approx(0.9)


def test_adaptive_norm_late_below_one_raises():
    with pytest.raises(ValueError, match="adaptive_norm_late"):
        RBFFFNConfig(adaptive_weight_norm=True, adaptive_norm_late=0.9)


def test_adaptive_norm_early_not_greater_than_late_raises():
    with pytest.raises(ValueError, match="adaptive_norm_early"):
        RBFFFNConfig(adaptive_weight_norm=True, adaptive_norm_early=1.2, adaptive_norm_late=1.2)


def test_adaptive_norm_validation_only_when_enabled():
    """Validation is skipped when adaptive_weight_norm=False (default)."""
    cfg = RBFFFNConfig(adaptive_weight_norm=False, adaptive_norm_late=0.5)
    assert cfg.adaptive_norm_late == pytest.approx(0.5)


def test_adaptive_norm_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "adaptive_weight_norm: true\n"
        "adaptive_norm_early: 3.0\n"
        "adaptive_norm_late: 1.5\n"
    )
    cfg = load_config(p)
    assert cfg.adaptive_weight_norm is True
    assert cfg.adaptive_norm_early == pytest.approx(3.0)
    assert cfg.adaptive_norm_late  == pytest.approx(1.5)


def test_adaptive_norm_late_exactly_one_is_valid():
    """adaptive_norm_late=1.0 with adaptive_weight_norm=True is valid (floor is >= 1.0)."""
    cfg = RBFFFNConfig(adaptive_weight_norm=True, adaptive_norm_late=1.0, adaptive_norm_early=1.5)
    assert cfg.adaptive_norm_late == pytest.approx(1.0)


def test_baseline_adaptive_weight_norm_yaml_loads():
    cfg = load_config(CONFIGS_DIR / "baseline_adaptive_weight_norm.yaml")
    assert cfg.adaptive_weight_norm is True
    assert cfg.adaptive_norm_early == pytest.approx(2.5)
    assert cfg.adaptive_norm_late  == pytest.approx(1.2)
    assert cfg.adaptive_norm_gamma == pytest.approx(0.3)
    assert cfg.adaptive_norm_beta  == pytest.approx(5.0)
    assert cfg.adaptive_norm_alpha == pytest.approx(0.9)
    assert cfg.model_type == "baseline"
    assert cfg.n_layers == 6
