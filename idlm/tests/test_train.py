# idlm/tests/test_train.py
"""Tests for training utilities: TPF/OH formula, eval logging, LoRA targets, mask checkpoint."""
import json
import pytest
import torch
from pathlib import Path
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.model import CausalLM
from idlm.models.idlm_model import IDLMCausalLM
from idlm.config import IDLMConfig
from idlm.generate import compute_tpf_oh

D, H, L_LAYERS, V, N = 32, 4, 2, 256, 12
DEVICE = torch.device("cpu")


def make_tiny_model(targets=None) -> IDLMCausalLM:
    if targets is None:
        targets = ["q_proj", "v_proj"]
    cfg = ModelConfig(d_model=D, n_heads=H, n_layers=L_LAYERS,
                      vocab_size=V, seq_len=N, ffn_hidden=86, dropout=0.0)
    return IDLMCausalLM(CausalLM(cfg), lora_rank=2, lora_alpha=4.0,
                        lora_target_modules=targets)


def make_eval_cfg(stride: int = 4) -> IDLMConfig:
    return IDLMConfig(
        ar_checkpoint="dummy.pt",
        stride=stride, prompt_len=4, gen_len=8,
        num_eval_examples=2, vocab_size=V,
    )


def tiny_loader():
    return [torch.randint(0, V - 1, (2, N))]


# ── Change 1: TPF/OH formula ────────────────────────────────────────────────

def test_run_isd_eval_tpf_oh_uses_correct_formula(monkeypatch):
    """tpf_oh_mean must equal compute_tpf_oh(alpha_mean, stride), not stride * alpha_mean.

    monkeypatch injects alpha=0.7 so the two formulas give clearly different values:
      correct:  4 / (1 + 0.3*4) = 4/2.2 ≈ 1.818
      buggy:    4 * 0.7         = 2.8
    """
    import idlm.generate as gen_module
    from idlm.train import run_isd_eval
    monkeypatch.setattr(gen_module, "isd_acceptance_rate", lambda *_: 0.7)

    model = make_tiny_model()
    cfg = make_eval_cfg(stride=4)
    metrics = run_isd_eval(model, tiny_loader(), cfg, DEVICE)

    expected_tpf = compute_tpf_oh(0.7, 4)   # ≈ 1.818
    buggy_tpf = 4 * 0.7                      # = 2.8
    assert abs(metrics["tpf_oh_mean"] - expected_tpf) < 1e-6, (
        f"got {metrics['tpf_oh_mean']:.4f}, expected {expected_tpf:.4f} (buggy={buggy_tpf:.4f})"
    )


def test_tpf_oh_formula_differs_from_stride_times_alpha():
    """Regression guard: the two formula variants give different values at alpha=0.7, stride=4."""
    correct = compute_tpf_oh(0.7, 4)
    buggy = 4 * 0.7
    assert abs(correct - buggy) > 0.5


# ── Change 2: eval metrics written to metrics.jsonl ─────────────────────────

def test_write_eval_row_appends_to_jsonl(tmp_path):
    """_write_eval_row must append a row with type='eval' and the given metrics."""
    from idlm.train import _write_eval_row
    metrics_path = tmp_path / "metrics.jsonl"
    _write_eval_row(metrics_path, epoch=3, alpha_mean=0.45, tpf_oh_mean=1.23)
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["type"] == "eval"
    assert rows[0]["epoch"] == 3
    assert abs(rows[0]["alpha_mean"] - 0.45) < 1e-6
    assert abs(rows[0]["tpf_oh_mean"] - 1.23) < 1e-6


def test_write_eval_row_appends_without_overwriting(tmp_path):
    """_write_eval_row must append, not overwrite existing step rows."""
    from idlm.train import _write_eval_row
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(json.dumps({"type": "step", "step": 1, "loss": 5.0}) + "\n")
    _write_eval_row(metrics_path, epoch=1, alpha_mean=0.3, tpf_oh_mean=0.9)
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["type"] == "step"
    assert rows[1]["type"] == "eval"


# ── Change 3: k_proj in default LoRA targets ────────────────────────────────

def test_default_lora_targets_include_k_proj():
    """IDLMConfig default lora_target_modules must include 'k_proj'."""
    cfg = IDLMConfig(ar_checkpoint="dummy.pt")
    assert "k_proj" in cfg.lora_target_modules


def test_k_proj_lora_layer_created():
    """With k_proj in targets, there are L_LAYERS*3 LoRA layers total."""
    from idlm.models.lora import LoRALinear
    model = make_tiny_model(targets=["q_proj", "k_proj", "v_proj"])
    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_layers) == L_LAYERS * 3


# ── Change 4: checkpoint on best L_mask ─────────────────────────────────────

def test_maybe_save_mask_checkpoint_saves_on_improvement(tmp_path):
    """_maybe_save_mask_checkpoint must save checkpoint_best_mask.pt when l_mask improves."""
    from idlm.train import _maybe_save_mask_checkpoint
    model = make_tiny_model()
    best = _maybe_save_mask_checkpoint(
        model, l_mask=4.0, best_l_mask=float("inf"), step=10, epoch=1, exp_dir=tmp_path
    )
    assert best == pytest.approx(4.0)
    assert (tmp_path / "checkpoint_best_mask.pt").exists()


def test_maybe_save_mask_checkpoint_skips_when_worse(tmp_path):
    """_maybe_save_mask_checkpoint must NOT overwrite when l_mask does not improve."""
    from idlm.train import _maybe_save_mask_checkpoint
    model = make_tiny_model()
    _maybe_save_mask_checkpoint(
        model, l_mask=3.0, best_l_mask=float("inf"), step=10, epoch=1, exp_dir=tmp_path
    )
    mtime = (tmp_path / "checkpoint_best_mask.pt").stat().st_mtime
    best = _maybe_save_mask_checkpoint(
        model, l_mask=5.0, best_l_mask=3.0, step=20, epoch=2, exp_dir=tmp_path
    )
    assert best == pytest.approx(3.0)
    assert (tmp_path / "checkpoint_best_mask.pt").stat().st_mtime == mtime


def test_maybe_save_mask_checkpoint_returns_new_best(tmp_path):
    """_maybe_save_mask_checkpoint must return the new best when l_mask improves."""
    from idlm.train import _maybe_save_mask_checkpoint
    model = make_tiny_model()
    best = _maybe_save_mask_checkpoint(
        model, l_mask=2.5, best_l_mask=3.0, step=30, epoch=3, exp_dir=tmp_path
    )
    assert best == pytest.approx(2.5)
    assert (tmp_path / "checkpoint_best_mask.pt").exists()
