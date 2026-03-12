import torch
import torch.nn as nn
import torch.nn.functional as F


class G0Gate(nn.Module):
    """
    Baseline self-gate. Element-wise affine then sigmoid, applied to RBF output.

        gate = sigmoid(w ⊙ x + b)
        out  = gate ⊙ x

    w initialised to ones, b to zeros — near-linear pass-through at init
    (RBF outputs in [0,1]: gate output ranges from 0 to ~0.73, a soft squash not a linear pass-through)

    Input/output: (B, N, d_model * K)
    """

    def __init__(self, d_model: int, K: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d_model * K))
        self.b = nn.Parameter(torch.zeros(d_model * K))

    def forward(self, rbf_out: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.w * rbf_out + self.b)
        return gate * rbf_out


class G1AGate(nn.Module):
    """
    Cross-kernel mixing gate (self-gated variant).

    Flattened RBF output passes through a square linear layer to mix across
    all K kernel responses before the sigmoid gate is formed.

        gate = sigmoid(Linear(rbf_out))
        out  = gate ⊙ rbf_out

    Input/output: (B, N, d_model * K)
    """

    def __init__(self, d_model: int, K: int):
        super().__init__()
        self.mix = nn.Linear(d_model * K, d_model * K)

    def forward(self, rbf_out: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.mix(rbf_out))
        return gate * rbf_out


class G1BGate(nn.Module):
    """
    Input-driven cross-kernel gate.

    The gate signal is computed from the pre-RBF normalised input x,
    projected to d_model*K, keeping gate and gated signal on separate branches.

        gate = sigmoid(Linear(x))      # x: (B, N, d_model)
        out  = gate ⊙ rbf_out          # rbf_out: (B, N, d_model*K)

    Input: rbf_out (B, N, d_model*K), x (B, N, d_model)
    Output: (B, N, d_model*K)
    """

    def __init__(self, d_model: int, K: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model * K)

    def forward(self, rbf_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(x))
        return gate * rbf_out


class G2SinkhornGate(nn.Module):
    """
    Sinkhorn aggregation — replaces gating with a doubly-stochastic weighted sum.

    RBF output is reshaped to (B, N, d_model, K). Sinkhorn normalisation
    (n_iters iterations) produces a weight matrix W over (d_model, K).
    The weighted sum over K collapses to (B, N, d_model).

    No learnable parameters — pure aggregation.

    Input:  rbf_out (B, N, d_model * K)
    Output: (B, N, d_model)
    """

    def __init__(self, d_model: int, K: int, n_iters: int = 20):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.n_iters = n_iters

    def _sinkhorn(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: (B, N, d_model, K), all positive.
        Returns weight matrix W of same shape.
        Rows normalised over K (sum to 1.0); columns normalised over d_model.
        Column sums converge to d_model/K (not 1) for non-square matrices.
        """
        eps = 1e-8
        for _ in range(self.n_iters):
            A = A / (A.sum(dim=-2, keepdim=True) + eps)   # col-norm over d_model
            A = A / (A.sum(dim=-1, keepdim=True) + eps)   # row-norm over K
        return A

    def forward(self, rbf_out: torch.Tensor) -> torch.Tensor:
        B, N, DK = rbf_out.shape
        A = rbf_out.reshape(B, N, self.d_model, self.K)   # (B, N, d_model, K)
        W = self._sinkhorn(A.clone())                       # weights (B, N, d_model, K)
        # Weighted sum of original RBF activations over K → (B, N, d_model)
        return (W * A).sum(dim=-1)
