import json
from pathlib import Path
import pytest
from idlm.chat_utils import discover_runs, RunInfo, load_model


def _make_run(root: Path, name: str, steps: list[dict]) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text(
        "ar_checkpoint: rbf_ffn/experiments/some/checkpoint_best.pt\n"
    )
    with open(run_dir / "metrics.jsonl", "w") as f:
        for row in steps:
            f.write(json.dumps(row) + "\n")
    return run_dir


def test_discover_runs_returns_sorted_newest_first(tmp_path):
    _make_run(tmp_path, "20260601_000000_aaa_idlm_r8_s4",
              [{"type": "step", "loss": 3.5}])
    _make_run(tmp_path, "20260607_000000_bbb_idlm_r8_s4",
              [{"type": "step", "loss": 2.1}])
    runs = discover_runs(tmp_path)
    assert len(runs) == 2
    assert runs[0].dir_name == "20260607_000000_bbb_idlm_r8_s4"
    assert runs[1].dir_name == "20260601_000000_aaa_idlm_r8_s4"


def test_discover_runs_extracts_final_loss(tmp_path):
    _make_run(tmp_path, "20260601_000000_aaa_idlm_r8_s4", [
        {"type": "step", "loss": 10.0},
        {"type": "step", "loss": 3.5},
    ])
    runs = discover_runs(tmp_path)
    assert runs[0].final_loss == pytest.approx(3.5)


def test_discover_runs_loss_none_when_no_steps(tmp_path):
    run_dir = tmp_path / "20260601_000000_aaa_idlm_r8_s4"
    run_dir.mkdir()
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text("ar_checkpoint: x.pt\n")
    # no metrics.jsonl
    runs = discover_runs(tmp_path)
    assert runs[0].final_loss is None


def test_discover_runs_skips_dirs_without_checkpoint(tmp_path):
    bad = tmp_path / "20260601_000000_incomplete_idlm"
    bad.mkdir()
    (bad / "config.yaml").write_text("ar_checkpoint: x.pt\n")
    # no checkpoint_best.pt
    runs = discover_runs(tmp_path)
    assert len(runs) == 0


def test_run_info_label_with_loss(tmp_path):
    _make_run(tmp_path, "20260607_113459_031611_idlm_r8_s4",
              [{"type": "step", "loss": 6.682}])
    runs = discover_runs(tmp_path)
    assert "20260607_113459" in runs[0].label
    assert "6.68" in runs[0].label


def test_run_info_label_no_loss(tmp_path):
    run_dir = tmp_path / "20260607_113459_031611_idlm_r8_s4"
    run_dir.mkdir()
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text("ar_checkpoint: x.pt\n")
    runs = discover_runs(tmp_path)
    assert "N/A" in runs[0].label


def test_load_model_raises_on_missing_ar_checkpoint(tmp_path):
    run_dir = tmp_path / "20260607_113459_031611_idlm_r8_s4"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "ar_checkpoint: /nonexistent/path/checkpoint_best.pt\n"
    )
    (run_dir / "checkpoint_best.pt").touch()  # exists but empty
    import torch
    with pytest.raises(FileNotFoundError):
        load_model(run_dir, repo_root=tmp_path, device=torch.device("cpu"))
