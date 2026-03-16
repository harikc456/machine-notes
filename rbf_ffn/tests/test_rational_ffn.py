# rbf_ffn/tests/test_rational_ffn.py
import torch
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.rational_ffn import RationalActivation, RationalFFN, RationalGatedFFN

B, N, D = 2, 16, 32


def make_cfg():
    # model_type="rational" is a valid dataclass value (no runtime validation);
    # CausalLM dispatch for this value is wired in a later task.
    return RBFFFNConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="rational",
    )


def test_rational_activation_shape():
    act = RationalActivation()
    x = torch.randn(B, N, D)
    assert act(x).shape == (B, N, D)


def test_rational_activation_gradients():
    act = RationalActivation()
    x = torch.randn(B, N, D)
    act(x).sum().backward()
    assert act.a.grad is not None
    assert act.b.grad is not None


def test_rational_ffn_shape():
    ffn = RationalFFN(make_cfg())
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_rational_ffn_no_bias():
    ffn = RationalFFN(make_cfg())
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_rational_activation_input_gradient():
    act = RationalActivation()
    x = torch.randn(B, N, D, requires_grad=True)
    act(x).sum().backward()
    assert x.grad is not None


def make_gated_cfg():
    return RBFFFNConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="rationalglu",
    )


def test_rational_gated_ffn_shape():
    ffn = RationalGatedFFN(make_gated_cfg())
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_rational_gated_ffn_no_bias():
    ffn = RationalGatedFFN(make_gated_cfg())
    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_rational_gated_ffn_gate_gradient():
    ffn = RationalGatedFFN(make_gated_cfg())
    x = torch.randn(B, N, D)
    ffn(x).sum().backward()
    assert ffn.act.a.grad is not None
    assert ffn.act.b.grad is not None

