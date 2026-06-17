import torch
import pytest
from mtp_draft.config import MTPConfig
from mtp_draft.models.cross_attn_block import CrossAttnBlock

B, S, N = 2, 8, 64  # batch, draft positions, context length


@pytest.fixture
def cfg():
    return MTPConfig(d_draft=64, n_heads=4, ffn_hidden=128, dropout=0.0)


@pytest.fixture
def block(cfg):
    return CrossAttnBlock(cfg)


def test_output_shape(block, cfg):
    query = torch.randn(B, S, cfg.d_draft)
    context = torch.randn(B, N, cfg.d_draft)
    out = block(query, context)
    assert out.shape == (B, S, cfg.d_draft)


def test_query_positions_independent(block, cfg):
    """Changing one query position must not change another (no causal mask between draft positions)."""
    query = torch.randn(B, S, cfg.d_draft)
    context = torch.randn(B, N, cfg.d_draft)

    out1 = block(query, context)

    query2 = query.clone()
    query2[:, 0, :] += 1.0  # perturb only position 0
    out2 = block(query2, context)

    # position 0 changes
    assert not torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-4)
    # position 1 must NOT change (no causal dependency)
    assert torch.allclose(out1[:, 1, :], out2[:, 1, :], atol=1e-4)


def test_different_context_changes_output(block, cfg):
    query = torch.randn(B, S, cfg.d_draft)
    ctx1 = torch.randn(B, N, cfg.d_draft)
    ctx2 = torch.randn(B, N, cfg.d_draft)
    assert not torch.allclose(block(query, ctx1), block(query, ctx2), atol=1e-4)


def test_no_bias_on_projections(block):
    for name, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            assert m.bias is None, f"{name} has unexpected bias"


def test_gradients_flow(block, cfg):
    query = torch.randn(B, S, cfg.d_draft, requires_grad=True)
    context = torch.randn(B, N, cfg.d_draft, requires_grad=True)
    out = block(query, context)
    out.sum().backward()
    assert query.grad is not None
    assert context.grad is not None
