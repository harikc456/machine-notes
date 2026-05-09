import torch
import torch.nn as nn
import torch.nn.functional as F
from rbf_ffn.config import ModelConfig


class _Expert(nn.Module):
    """Single SwiGLU expert."""

    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up_proj   = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoEFFN(nn.Module):
    """
    Sparse top-k Mixture of Experts FFN.

    Each token is independently routed to `moe_top_k` out of `moe_n_experts`
    SwiGLU experts. The output is the softmax-weighted sum of those experts.

        router_logits = router(x)                        # (B, N, n_experts)
        top_k weights, indices = topk(softmax(logits))   # (B, N, top_k)
        out = sum_k( weight_k * expert_k(x) )

    With moe_orthogonal=True, Gram-Schmidt is applied to the expert outputs
    in descending router-score order before the weighted sum. Each expert
    must contribute a direction orthogonal to all higher-confidence experts,
    maximising the independent information written to the residual stream.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts   = cfg.moe_n_experts
        self.top_k       = cfg.moe_top_k
        self.orthogonal  = cfg.moe_orthogonal
        self.router      = nn.Linear(cfg.d_model, cfg.moe_n_experts, bias=False)
        self.experts     = nn.ModuleList(
            [_Expert(cfg.d_model, cfg.ffn_hidden) for _ in range(cfg.moe_n_experts)]
        )

    @staticmethod
    def _gram_schmidt(vecs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Orthogonalise rows of vecs (T, k, D) in-place via Gram-Schmidt.
        Row 0 is unchanged; each subsequent row has the projections onto all
        prior rows subtracted. Norms are preserved for row 0; later rows may
        shrink if they overlap with earlier ones.
        """
        basis = [vecs[:, 0]]
        for i in range(1, vecs.shape[1]):
            v = vecs[:, i]
            for b in basis:
                proj   = (v * b).sum(dim=-1, keepdim=True)
                norm_sq = (b * b).sum(dim=-1, keepdim=True).clamp(min=eps)
                v = v - (proj / norm_sq) * b
            basis.append(v)
        return torch.stack(basis, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.view(B * N, D)                              # (T, D)  T = B*N

        logits  = self.router(x_flat)                          # (T, n_experts)
        weights = F.softmax(logits, dim=-1)                    # (T, n_experts)
        top_w, top_idx = torch.topk(weights, self.top_k, dim=-1)  # (T, top_k)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)        # re-normalise

        if self.orthogonal:
            # collect expert outputs in router-score order: (T, top_k, D)
            T = x_flat.shape[0]
            expert_outs = torch.zeros(T, self.top_k, D, device=x.device, dtype=x.dtype)
            for k in range(self.top_k):
                idx = top_idx[:, k]
                for e in range(self.n_experts):
                    mask = (idx == e)
                    if not mask.any():
                        continue
                    expert_outs[mask, k] = self.experts[e](x_flat[mask]).to(expert_outs.dtype)
            expert_outs = self._gram_schmidt(expert_outs)       # orthogonalise
            out = (top_w.unsqueeze(-1) * expert_outs).sum(dim=1)  # (T, D)
        else:
            out = torch.zeros_like(x_flat)                     # (T, D)
            for k in range(self.top_k):
                idx = top_idx[:, k]                            # (T,)
                w   = top_w[:, k].unsqueeze(-1)               # (T, 1)
                for e in range(self.n_experts):
                    mask = (idx == e)
                    if not mask.any():
                        continue
                    out[mask] += w[mask] * self.experts[e](x_flat[mask])

        return out.view(B, N, D)
