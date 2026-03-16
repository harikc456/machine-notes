import torch
import pytest
from rbf_ffn.models.gate_layer import G0Gate, G1AGate, G1BGate, G2SinkhornGate

D, K, B, N = 8, 5, 2, 10
DK = D * K


# ── G0 ──────────────────────────────────────────────────────────────────────

def test_g0_output_shape():
    gate = G0Gate(d_model=D, K=K)
    rbf_out = torch.rand(B, N, DK)
    out = gate(rbf_out)
    assert out.shape == (B, N, DK)


def test_g0_output_range():
    """G0 output in [0, 1] since RBF inputs in [0,1] and sigmoid in (0,1)."""
    gate = G0Gate(d_model=D, K=K)
    rbf_out = torch.rand(B, N, DK)
    out = gate(rbf_out)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6


def test_g0_has_weight_and_bias():
    gate = G0Gate(d_model=D, K=K)
    param_names = {n for n, _ in gate.named_parameters()}
    assert "w" in param_names
    assert "b" in param_names


def test_g0_init_is_nonnegative():
    """At init (w=1, b=0), output = sigmoid(rbf_out) * rbf_out >= 0."""
    gate = G0Gate(d_model=D, K=K)
    rbf_out = torch.rand(B, N, DK)
    out = gate(rbf_out)
    assert out.min() >= 0.0


def test_g0_gradient_flows():
    gate = G0Gate(d_model=D, K=K)
    rbf_out = torch.rand(B, N, DK, requires_grad=True)
    gate(rbf_out).sum().backward()
    assert rbf_out.grad is not None


# ── G1-A ────────────────────────────────────────────────────────────────────

def test_g1a_output_shape():
    gate = G1AGate(d_model=D, K=K)
    rbf_out = torch.rand(B, N, DK)
    out = gate(rbf_out)
    assert out.shape == (B, N, DK)


def test_g1a_has_linear():
    """G1-A must have a cross-kernel mixing linear layer."""
    gate = G1AGate(d_model=D, K=K)
    assert hasattr(gate, "mix")
    assert isinstance(gate.mix, torch.nn.Linear)
    assert gate.mix.in_features == DK
    assert gate.mix.out_features == DK


def test_g1a_gradient_flows():
    gate = G1AGate(d_model=D, K=K)
    rbf_out = torch.rand(B, N, DK, requires_grad=True)
    gate(rbf_out).sum().backward()
    assert rbf_out.grad is not None


# ── G1-B ────────────────────────────────────────────────────────────────────

def test_g1b_output_shape():
    gate = G1BGate(d_model=D, K=K)
    x = torch.randn(B, N, D)
    rbf_out = torch.rand(B, N, DK)
    out = gate(rbf_out, x)
    assert out.shape == (B, N, DK)


def test_g1b_has_linear():
    """G1-B must project from d_model → d_model*K."""
    gate = G1BGate(d_model=D, K=K)
    assert hasattr(gate, "proj")
    assert isinstance(gate.proj, torch.nn.Linear)
    assert gate.proj.in_features == D
    assert gate.proj.out_features == DK


def test_g1b_gate_independent_of_rbf():
    """Changing rbf_out must not change the gate signal (gate depends only on x)."""
    gate = G1BGate(d_model=D, K=K)
    x = torch.randn(B, N, D)
    rbf1 = torch.rand(B, N, DK)
    rbf2 = torch.rand(B, N, DK)  # different rbf_out
    with torch.no_grad():
        out1 = gate(rbf1, x)
        out2 = gate(rbf2, x)
    # Gate values are sigmoid(proj(x)), independent of rbf_out,
    # so out1/rbf1 == out2/rbf2 element-wise
    gate_vals1 = out1 / rbf1.clamp(min=1e-8)
    gate_vals2 = out2 / rbf2.clamp(min=1e-8)
    assert torch.allclose(gate_vals1, gate_vals2, atol=1e-5)


# ── G2 Sinkhorn ─────────────────────────────────────────────────────────────

def test_g2_output_shape():
    gate = G2SinkhornGate(d_model=D, K=K, n_iters=20)
    rbf_out = torch.rand(B, N, DK)
    out = gate(rbf_out)
    # Sinkhorn collapses K → weighted sum → d_model
    assert out.shape == (B, N, D)


def test_g2_sinkhorn_row_stochastic():
    """After Sinkhorn, rows (over K) should sum to ~1."""
    gate = G2SinkhornGate(d_model=D, K=K, n_iters=20)
    rbf_out = torch.rand(B, N, DK)
    A = rbf_out.reshape(B, N, D, K)
    W = gate._sinkhorn(A.clone())       # (B, N, D, K)
    row_sums = W.sum(dim=-1)            # (B, N, D)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_g2_sinkhorn_col_stochastic():
    """After Sinkhorn on a (D, K) non-square matrix, column sums converge to D/K."""
    gate = G2SinkhornGate(d_model=D, K=K, n_iters=20)
    rbf_out = torch.rand(B, N, DK)
    A = rbf_out.reshape(B, N, D, K)
    W = gate._sinkhorn(A.clone())       # (B, N, D, K)
    col_sums = W.sum(dim=-2)            # (B, N, K)
    expected = torch.full_like(col_sums, D / K)
    assert torch.allclose(col_sums, expected, atol=1e-3)


def test_g2_no_learnable_params():
    """G2 gate has no learnable parameters — aggregation only."""
    gate = G2SinkhornGate(d_model=D, K=K, n_iters=20)
    assert sum(p.numel() for p in gate.parameters()) == 0


def test_g2_gradient_flows():
    gate = G2SinkhornGate(d_model=D, K=K, n_iters=20)
    rbf_out = torch.rand(B, N, DK, requires_grad=True)
    gate(rbf_out).sum().backward()
    assert rbf_out.grad is not None


def test_g2_sinkhorn_builds_no_autograd_graph():
    """_sinkhorn must not build an autograd graph through its iterations.

    With n_iters=20 and 2 ops/iter, the loop creates 40 intermediate tensors.
    Each is shape (B, N, d_model, K) and is retained by the autograd graph for
    the entire backward pass — O(n_iters) activation memory that causes OOM.

    Fix: run the normalization loop under torch.no_grad() so no graph is built.
    Gradients still flow through the original rbf_out via (W * A).sum(-1) in
    forward(), since A (the reshape of rbf_out) is not touched by _sinkhorn.
    """
    gate = G2SinkhornGate(d_model=D, K=K, n_iters=20)
    rbf_out = torch.rand(B, N, DK, requires_grad=True)
    A = rbf_out.reshape(B, N, D, K)
    W = gate._sinkhorn(A.clone())
    assert W.grad_fn is None, (
        "_sinkhorn built an autograd graph through its iterations. "
        "Wrap the normalization loop in torch.no_grad() to fix OOM."
    )
