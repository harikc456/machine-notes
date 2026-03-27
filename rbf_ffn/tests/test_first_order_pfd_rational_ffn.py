# rbf_ffn/tests/test_first_order_pfd_rational_ffn.py
import torch
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.rational_ffn import FirstOrderPFDRationalFFN

B, N, D = 2, 16, 32


def make_cfg():
    return ModelConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="first_order_pfd_rational", pfd_n=4,
    )


def test_ffn_output_shape():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_ffn_no_bias():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_phi_receives_gradient():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    x = torch.randn(B, N, D)
    ffn(x).sum().backward()
    assert ffn.phi.grad is not None


def test_input_gradient_flows():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    x = torch.randn(B, N, D, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None


def test_pfd_act_receives_gradient():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    x = torch.randn(B, N, D)
    ffn(x).sum().backward()
    assert ffn.act.a.grad is not None
