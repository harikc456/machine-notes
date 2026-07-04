# rbf_ffn/tests/test_deepseek_sparse_attention.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.attention import DeepSeekSparseAttention

B, N, D, H = 2, 32, 64, 4   # N > sparse_topk so the pruning branch is exercised
HEAD_DIM = D // H
TOPK = 8


@pytest.fixture
def cfg():
    return ModelConfig(
        d_model=D, n_heads=H, dropout=0.0,
        attn_type="deepseek_sparse", sparse_topk=TOPK,
        sparse_index_dim=8, sparse_index_heads=2,
    )


def test_output_shape(cfg):
    attn = DeepSeekSparseAttention(cfg)
    x = torch.randn(B, N, D)
    assert attn(x).shape == (B, N, D)


def test_causal_mask(cfg):
    """Output at position i must not depend on positions j > i."""
    attn = DeepSeekSparseAttention(cfg)
    attn.eval()
    x = torch.randn(1, N, D)
    out_full = attn(x)

    x_corrupt = x.clone()
    x_corrupt[:, 1:, :] = torch.randn_like(x_corrupt[:, 1:, :])
    out_corrupt = attn(x_corrupt)

    assert torch.allclose(out_full[:, 0, :], out_corrupt[:, 0, :], atol=1e-5)


def test_early_positions_below_topk_still_causal(cfg):
    """A query at position i < topk has fewer than topk valid keys; it must
    still only attend within [0, i], not spill into padding/-inf selections."""
    attn = DeepSeekSparseAttention(cfg)
    attn.eval()
    x = torch.randn(1, N, D)
    out_full = attn(x)

    # Corrupt only position 5 (index 5); position 2's output must be unaffected
    # since 2 < 5 and topk=8 > 3 valid keys for position 2.
    x_corrupt = x.clone()
    x_corrupt[:, 5, :] = torch.randn_like(x_corrupt[:, 5, :])
    out_corrupt = attn(x_corrupt)

    assert torch.allclose(out_full[:, 2, :], out_corrupt[:, 2, :], atol=1e-5)


def test_gradient_flows(cfg):
    attn = DeepSeekSparseAttention(cfg)
    x = torch.randn(B, N, D, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attn, name).weight.grad is not None


def test_indexer_gets_no_gradient_from_main_loss(cfg):
    """Hard top-k selection blocks gradient to the indexer by construction;
    it must be trained separately (e.g. distillation), not via this loss."""
    attn = DeepSeekSparseAttention(cfg)
    x = torch.randn(B, N, D, requires_grad=True)
    attn(x).sum().backward()
    assert attn.index_q_proj.weight.grad is None
    assert attn.index_k_proj.weight.grad is None
    assert attn.index_weight.grad is None


def test_dense_fallback_when_seq_len_below_topk():
    """When N <= sparse_topk, the indexer path is skipped entirely."""
    cfg = ModelConfig(
        d_model=D, n_heads=H, dropout=0.0,
        attn_type="deepseek_sparse", sparse_topk=128,
    )
    attn = DeepSeekSparseAttention(cfg)
    x = torch.randn(B, N, D)
    assert attn(x).shape == (B, N, D)
    assert attn.index_q_proj.weight.grad is None  # never touched this forward


def test_gqa_shapes():
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_kv_heads=2, dropout=0.0,
        attn_type="deepseek_sparse", sparse_topk=TOPK,
    )
    attn = DeepSeekSparseAttention(cfg)
    x = torch.randn(B, N, D)
    assert attn(x).shape == (B, N, D)


def test_registry_entry():
    from rbf_ffn.models.attention import ATTN_REGISTRY
    assert ATTN_REGISTRY["deepseek_sparse"] is DeepSeekSparseAttention
