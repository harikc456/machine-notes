import torch
import pytest
from mtp_draft.tree import build_tree

VOCAB, MAX_DRAFT = 100, 4


def _make_peaked_logits(top_token: int, secondary_token: int, gap: float = 5.0) -> torch.Tensor:
    """Logits with one clear top token and one secondary token gap nats below."""
    logits = torch.full((MAX_DRAFT, VOCAB), -100.0)
    logits[:, top_token] = 0.0
    logits[:, secondary_token] = -gap
    return logits


def test_single_path_when_tau_zero():
    """tau=0.0 → only the top token at each position → exactly 1 path."""
    logits = _make_peaked_logits(top_token=5, secondary_token=10)
    paths = build_tree(logits, tau=0.0)
    assert len(paths) == 1
    assert all(t == 5 for t in paths[0])


def test_two_candidates_per_position_when_tau_large():
    """With gap=2 and tau=3, both tokens qualify at every position."""
    logits = _make_peaked_logits(top_token=5, secondary_token=10, gap=2.0)
    paths = build_tree(logits, tau=3.0, max_tree_nodes=10000)
    # 2 candidates × 4 positions → 2^4 = 16 paths
    assert len(paths) == 16


def test_paths_sorted_by_score():
    """Paths must be returned highest cumulative log-prob first."""
    logits = _make_peaked_logits(top_token=5, secondary_token=10, gap=2.0)
    paths = build_tree(logits, tau=3.0, max_tree_nodes=10000)
    # First path: all top tokens
    assert paths[0] == [5] * MAX_DRAFT


def test_max_tree_nodes_respected():
    """Output must not exceed max_tree_nodes paths."""
    logits = torch.randn(MAX_DRAFT, VOCAB)
    paths = build_tree(logits, tau=100.0, max_tree_nodes=10)
    assert len(paths) <= 10


def test_empty_logits_returns_one_path():
    """Even with very negative logits, at least one path (the top tokens) is returned."""
    logits = torch.full((MAX_DRAFT, VOCAB), -1e9)
    logits[:, 0] = 0.0
    paths = build_tree(logits, tau=0.5)
    assert len(paths) >= 1
