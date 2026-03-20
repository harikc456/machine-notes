from __future__ import annotations
import torch
import pytest
from grokking.config import GrokConfig
from grokking.data import build_dataset, split_dataset, _compute_label


# ── Label correctness ─────────────────────────────────────────────────────────

def test_add_label():
    assert _compute_label(2, 3, "add", 5) == 0     # (2+3)%5 = 0

def test_sub_label():
    assert _compute_label(2, 3, "sub", 5) == 4     # (2-3)%5 = 4

def test_mul_label():
    assert _compute_label(2, 3, "mul", 5) == 1     # (2*3)%5 = 1

def test_div_label():
    # pow(2, -1, 5)=3, so (1*3)%5=3
    assert _compute_label(1, 2, "div", 5) == 3

def test_x2_plus_xy_plus_y2_label():
    # (4 + 6 + 9) % 5 = 19 % 5 = 4
    assert _compute_label(2, 3, "x2_plus_xy_plus_y2", 5) == 4

def test_unknown_op_raises():
    with pytest.raises(ValueError, match="Unknown operation"):
        _compute_label(1, 2, "bogus", 5)


# ── Dataset size ──────────────────────────────────────────────────────────────

def test_add_dataset_size():
    cfg = GrokConfig(p=5, operation="add")
    inputs, labels = build_dataset(cfg)
    assert len(inputs) == 25        # 5*5

def test_div_excludes_b_zero():
    cfg = GrokConfig(p=5, operation="div")
    inputs, labels = build_dataset(cfg)
    assert len(inputs) == 20        # 5*4 (b=0 excluded)


# ── Tensor shapes and types ───────────────────────────────────────────────────

def test_inputs_shape():
    cfg = GrokConfig(p=5, operation="add")
    inputs, labels = build_dataset(cfg)
    assert inputs.shape == (25, 4)
    assert inputs.dtype == torch.long

def test_labels_shape():
    cfg = GrokConfig(p=5, operation="add")
    inputs, labels = build_dataset(cfg)
    assert labels.shape == (25,)
    assert labels.dtype == torch.long


# ── Token format [a, op_token, b, eq_token] ───────────────────────────────────

def test_sequence_op_token_position():
    cfg = GrokConfig(p=5, operation="add")
    inputs, _ = build_dataset(cfg)
    # op_token = p = 5, eq_token = p+1 = 6
    assert (inputs[:, 1] == 5).all()    # col 1 is always op_token
    assert (inputs[:, 3] == 6).all()    # col 3 is always eq_token

def test_sequence_a_b_in_range():
    cfg = GrokConfig(p=5, operation="add")
    inputs, _ = build_dataset(cfg)
    assert inputs[:, 0].max() < 5      # a ∈ [0, p)
    assert inputs[:, 2].max() < 5      # b ∈ [0, p)


# ── Train/val split ───────────────────────────────────────────────────────────

def test_split_sizes_sum_to_total():
    cfg = GrokConfig(p=5, operation="add", train_fraction=0.4, seed=42)
    inputs, labels = build_dataset(cfg)
    t_inp, t_lbl, v_inp, v_lbl = split_dataset(inputs, labels, cfg)
    assert len(t_inp) + len(v_inp) == len(inputs)
    assert len(t_lbl) == len(t_inp)
    assert len(v_lbl) == len(v_inp)

def test_split_reproducible():
    cfg = GrokConfig(p=5, operation="add", seed=42)
    inputs, labels = build_dataset(cfg)
    t1, l1, _, _ = split_dataset(inputs, labels, cfg)
    t2, l2, _, _ = split_dataset(inputs, labels, cfg)
    assert torch.equal(t1, t2)
    assert torch.equal(l1, l2)

def test_split_no_overlap():
    cfg = GrokConfig(p=5, operation="add", seed=42)
    inputs, labels = build_dataset(cfg)
    t_inp, _, v_inp, _ = split_dataset(inputs, labels, cfg)
    train_set = {tuple(row.tolist()) for row in t_inp}
    val_set   = {tuple(row.tolist()) for row in v_inp}
    assert len(train_set & val_set) == 0
