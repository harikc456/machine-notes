import json
import pytest
from pathlib import Path
from rbf_ffn._leaderboard_data import load_experiment, load_all_experiments, fmt, fmt_params


@pytest.fixture
def exp_dir(tmp_path):
    d = tmp_path / "20260101_120000_abc123_xsa_swiglu_qknorm_wnorm_d256"
    d.mkdir()
    (d / "metrics.jsonl").write_text(
        '{"epoch": 0, "train_loss": 5.0, "train_ppl": 148.4, "val_loss": 4.5, "val_ppl": 90.0, "epoch_time_s": 3600.0, "effective_batch_size": 16}\n'
        '{"epoch": 1, "train_loss": 4.0, "train_ppl": 54.6, "val_loss": 3.8, "val_ppl": 44.7, "epoch_time_s": 3600.0, "effective_batch_size": 16}\n'
    )
    (d / "config.yaml").write_text(
        "attn_type: xsa\nffn_type: swiglu\nqk_norm: true\nlinear_weight_norm: true\n"
        "d_model: 256\nn_layers: 6\nffn_hidden: 688\n"
    )
    (d / "params.json").write_text('{"n_params": 30478464}')
    return d


def test_load_experiment_returns_dict(exp_dir):
    result = load_experiment(exp_dir)
    assert isinstance(result, dict)


def test_load_experiment_best_ppl(exp_dir):
    result = load_experiment(exp_dir)
    assert result["best_val_ppl"] == pytest.approx(44.7)
    assert result["best_epoch"] == 1


def test_load_experiment_epochs_done(exp_dir):
    result = load_experiment(exp_dir)
    assert result["epochs_done"] == 2


def test_load_experiment_total_time(exp_dir):
    result = load_experiment(exp_dir)
    assert result["total_time_h"] == pytest.approx(2.0)


def test_load_experiment_params(exp_dir):
    result = load_experiment(exp_dir)
    assert result["n_params"] == 30478464


def test_load_experiment_config_text(exp_dir):
    result = load_experiment(exp_dir)
    assert "xsa" in result["config_text"]


def test_load_experiment_missing_metrics_returns_none(tmp_path):
    d = tmp_path / "empty_exp"
    d.mkdir()
    (d / "config.yaml").write_text("attn_type: xsa\n")
    assert load_experiment(d) is None


def test_load_experiment_infers_attn_from_dirname(tmp_path):
    d = tmp_path / "20260101_120000_abc123_xsa_swiglu_d256"
    d.mkdir()
    (d / "metrics.jsonl").write_text(
        '{"epoch": 0, "train_ppl": 100.0, "val_ppl": 80.0, "epoch_time_s": 100.0}\n'
    )
    (d / "config.yaml").write_text("d_model: 256\n")
    result = load_experiment(d)
    assert result["attn_type"] == "xsa"


def test_load_experiment_infers_standard_from_dirname(tmp_path):
    d = tmp_path / "20260101_120000_abc123_standard_swiglu_d256"
    d.mkdir()
    (d / "metrics.jsonl").write_text(
        '{"epoch": 0, "train_ppl": 100.0, "val_ppl": 80.0, "epoch_time_s": 100.0}\n'
    )
    (d / "config.yaml").write_text("d_model: 256\n")
    result = load_experiment(d)
    assert result["attn_type"] == "std"


def test_load_all_experiments_skips_non_dirs(tmp_path):
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "analysis.md").write_text("notes")
    results = load_all_experiments(tmp_path)
    assert results == []


def test_fmt_none():
    assert fmt(None) == "—"

def test_fmt_float():
    assert fmt(3.14159) == "3.14"

def test_fmt_bool_true():
    assert fmt(True) == "Y"

def test_fmt_bool_false():
    assert fmt(False) == "N"

def test_fmt_list():
    assert fmt([1, 3]) == "[1, 3]"

def test_fmt_params_millions():
    assert fmt_params(30_478_464) == "30.5M"

def test_fmt_params_billions():
    assert fmt_params(1_500_000_000) == "1.50B"

def test_fmt_params_none():
    assert fmt_params(None) == "—"
