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

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.moe_n_experts
        self.top_k     = cfg.moe_top_k
        self.router    = nn.Linear(cfg.d_model, cfg.moe_n_experts, bias=False)
        self.experts   = nn.ModuleList(
            [_Expert(cfg.d_model, cfg.ffn_hidden) for _ in range(cfg.moe_n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_flat = x.view(B * N, D)                              # (T, D)  T = B*N

        logits  = self.router(x_flat)                          # (T, n_experts)
        weights = F.softmax(logits, dim=-1)                    # (T, n_experts)
        top_w, top_idx = torch.topk(weights, self.top_k, dim=-1)  # (T, top_k)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)        # re-normalise

        out = torch.zeros_like(x_flat)                         # (T, D)
        for k in range(self.top_k):
            idx = top_idx[:, k]                                # (T,)
            w   = top_w[:, k].unsqueeze(-1)                   # (T, 1)
            # route each token to its assigned expert
            for e in range(self.n_experts):
                mask = (idx == e)                              # (T,)
                if not mask.any():
                    continue
                out[mask] += w[mask] * self.experts[e](x_flat[mask])

        return out.view(B, N, D)
