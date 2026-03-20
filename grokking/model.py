"""
Transformer encoder for grokking experiments.

Input:  LongTensor (B, 4) — token sequence [a, op_token, b, eq_token]
Output: FloatTensor (B, p) — logits over p classes (at the = position)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from grokking.config import GrokConfig

SEQ_LEN = 4  # fixed: [a, op_token, b, eq_token]


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with full (non-causal) self-attention."""

    def __init__(self, cfg: GrokConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn  = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
            bias=True,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class GrokTransformer(nn.Module):
    """
    Small transformer for modular arithmetic classification.

        token_embedding + pos_embedding
        → N × TransformerBlock
        → LayerNorm
        → lm_head (at position SEQ_LEN-1, the = token)
        → logits (B, p)
    """

    def __init__(self, cfg: GrokConfig):
        super().__init__()
        vocab_size = cfg.p + 2  # p digit tokens + op_token + eq_token
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_embedding   = nn.Embedding(SEQ_LEN, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.p, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, 4) integer token ids
        returns: logits (B, p)
        """
        B = tokens.shape[0]
        positions = (
            torch.arange(SEQ_LEN, device=tokens.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        x = self.token_embedding(tokens) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x[:, -1, :])   # only the = position


def build_optimizer_groups(
    model: GrokTransformer,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split model parameters into Muon and AdamW groups.

    Rules (first match wins):
      1. Token or positional embedding weight → AdamW
      2. param.ndim == 2                      → Muon
      3. else                                 → AdamW

    Returns (muon_params, adamw_params).
    """
    embedding_ids = {
        id(model.token_embedding.weight),
        id(model.pos_embedding.weight),
    }
    seen: set[int] = set()
    muon:  list[torch.Tensor] = []
    adamw: list[torch.Tensor] = []

    for _, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if pid in embedding_ids:
            adamw.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)

    return muon, adamw
