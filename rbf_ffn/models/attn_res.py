# rbf_ffn/models/attn_res.py
import torch
import torch.nn as nn


class AttnResLayer(nn.Module):
    """
    Full Attention Residual (AttnRes) mixer for one layer.

    Replaces depth-wise accumulation h_l = h_{l-1} + f(h_{l-1}) with
    learned softmax attention over all previous layer outputs:

        h_l = Σ_i softmax(w_l · RMSNorm(v_i)) * v_i

    w_l ∈ R^d is a learned per-layer pseudo-query (negligible parameter count).
    RMSNorm on keys prevents high-magnitude layers from dominating.

    Reference: Chen et al. arXiv:2603.15031 (Kimi AttnRes)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query    = nn.Parameter(torch.zeros(d_model))
        self.key_norm = nn.RMSNorm(d_model)

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """
        sources: list of n tensors, each (B, T, d_model)
        returns: (B, T, d_model) softmax-weighted combination
        """
        stacked = torch.stack(sources, dim=0)                           # (n, B, T, d)
        normed  = self.key_norm(stacked)                                # (n, B, T, d)
        scores  = torch.einsum('d,nbtd->nbt', self.query, normed)      # (n, B, T)
        weights = torch.softmax(scores, dim=0)                          # (n, B, T)
        return torch.einsum('nbt,nbtd->btd', weights, stacked)         # (B, T, d)
