import pytest
import torch
from kromhc_transformer.models.transformer_block import LlamaBlock, KromHCBlock
from kromhc_transformer.config import KromHCConfig

def test_llama_block_shape():
    cfg = KromHCConfig(d_model=256, n_heads=8, ffn_hidden=688, dropout=0.0)
    block = LlamaBlock(cfg)
    x = torch.randn(2, 64, 256)
    out = block(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape

def test_kromhc_block_shape():
    cfg = KromHCConfig(d_model=256, n_heads=8, ffn_hidden=688, use_kromhc=True, dropout=0.0)
    block = KromHCBlock(cfg)
    x = torch.randn(2, 64, 256)
    out, H = block(x)
    assert out.shape == x.shape
    assert H.shape == (2 * 64, 8, 8)  # (B*N, n_heads, n_heads)

def test_kromhc_block_mixing_matrix_doubly_stochastic():
    cfg = KromHCConfig(d_model=256, n_heads=8, ffn_hidden=688, use_kromhc=True, dropout=0.0)
    block = KromHCBlock(cfg)
    x = torch.randn(2, 32, 256)
    _, H = block(x)
    row_sums = H.sum(dim=2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

def test_llama_block_gradient_flow():
    cfg = KromHCConfig(d_model=64, n_heads=4, ffn_hidden=128, dropout=0.0)
    block = LlamaBlock(cfg)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None

def test_kromhc_block_gradient_flow():
    cfg = KromHCConfig(d_model=64, n_heads=4, ffn_hidden=128, use_kromhc=True, dropout=0.0)
    block = KromHCBlock(cfg)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out, _ = block(x)
    out.sum().backward()
    assert x.grad is not None
