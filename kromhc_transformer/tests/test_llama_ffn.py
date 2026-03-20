import pytest
import torch
from kromhc_transformer.models.llama_ffn import SwiGLUFFN
from kromhc_transformer.config import KromHCConfig

def test_swiglu_ffn_shape():
    cfg = KromHCConfig(d_model=256, ffn_hidden=688)
    ffn = SwiGLUFFN(cfg)
    x = torch.randn(2, 512, 256)
    out = ffn(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_swiglu_ffn_no_bias():
    cfg = KromHCConfig(d_model=256, ffn_hidden=688)
    ffn = SwiGLUFFN(cfg)
    for name, param in ffn.named_parameters():
        if 'bias' in name:
            assert False, f"Found bias: {name}"

def test_swiglu_ffn_gradient_flow():
    cfg = KromHCConfig(d_model=64, ffn_hidden=128)
    ffn = SwiGLUFFN(cfg)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = ffn(x)
    out.sum().backward()
    assert x.grad is not None
