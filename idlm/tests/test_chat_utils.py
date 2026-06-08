import json
from pathlib import Path
import pytest
from idlm.chat_utils import discover_runs, RunInfo, load_model, discover_rbf_runs, ar_generate, load_rbf_model


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


def _make_rbf_run(root: Path, name: str, epochs: list[dict]) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text(
        "d_model: 64\nn_heads: 2\nn_layers: 2\nvocab_size: 50257\nseq_len: 64\n"
    )
    with open(run_dir / "metrics.jsonl", "w") as f:
        for row in epochs:
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


def test_discover_rbf_runs_sorted_newest_first(tmp_path):
    _make_rbf_run(tmp_path, "20260316_000000_swiglu_d256", [{"epoch": 0, "val_loss": 5.5}])
    _make_rbf_run(tmp_path, "20260404_000000_swiglu_qknorm_d256", [{"epoch": 0, "val_loss": 4.9}])
    runs = discover_rbf_runs(tmp_path)
    assert len(runs) == 2
    assert runs[0].dir_name == "20260404_000000_swiglu_qknorm_d256"


def test_discover_rbf_runs_reads_last_val_loss(tmp_path):
    _make_rbf_run(tmp_path, "20260404_000000_swiglu_d256", [
        {"epoch": 0, "val_loss": 5.5},
        {"epoch": 1, "val_loss": 4.9},
    ])
    runs = discover_rbf_runs(tmp_path)
    assert runs[0].final_loss == pytest.approx(4.9)


def test_discover_rbf_runs_model_type_is_rbf(tmp_path):
    _make_rbf_run(tmp_path, "20260404_000000_swiglu_d256", [{"epoch": 0, "val_loss": 4.9}])
    runs = discover_rbf_runs(tmp_path)
    assert runs[0].model_type == "rbf"


def test_discover_rbf_runs_skips_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "20260404_000000_swiglu_d256"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("d_model: 64\n")
    runs = discover_rbf_runs(tmp_path)
    assert len(runs) == 0


def test_discover_rbf_runs_none_loss_when_no_metrics(tmp_path):
    run_dir = tmp_path / "20260404_000000_swiglu_d256"
    run_dir.mkdir()
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text("d_model: 64\n")
    runs = discover_rbf_runs(tmp_path)
    assert runs[0].final_loss is None


def test_ar_generate_returns_correct_length():
    import torch
    import torch.nn as nn

    class _TinyLM(nn.Module):
        def forward(self, tokens):
            B, N = tokens.shape
            return torch.zeros(B, N, 50257), []

    model = _TinyLM()
    out = ar_generate(model, [1, 2, 3], gen_len=10, device=torch.device("cpu"))
    assert len(out) == 13


def test_ar_generate_caps_to_real_vocab():
    import torch
    import torch.nn as nn

    class _BigVocabLM(nn.Module):
        def forward(self, tokens):
            B, N = tokens.shape
            logits = torch.full((B, N, 65536), -1e9)
            logits[:, :, 50257:] = 10.0
            return logits, []

    model = _BigVocabLM()
    out = ar_generate(model, [1], gen_len=5, device=torch.device("cpu"), real_vocab_size=50257)
    assert all(0 <= t < 50257 for t in out[1:])


def test_load_rbf_model_raises_on_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "20260404_swiglu_d256"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "d_model: 64\nn_heads: 2\nn_layers: 2\nvocab_size: 50257\nseq_len: 64\n"
    )
    # no checkpoint_best.pt
    import torch
    with pytest.raises(FileNotFoundError):
        load_rbf_model(run_dir, device=torch.device("cpu"))
