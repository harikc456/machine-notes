# sigreg/models/model.py
"""
Causal language model for SIGReg experiments.

Architecture:
    token_embedding → N × TransformerBlock → lm_head

No residual connections. No RMSNorm. No pre-norm, no post-norm.
Each TransformerBlock fully transforms its input: x = ffn(attn(x)).

forward() returns (logits, hidden_states):
    logits:        (B, N, vocab_size)
    hidden_states: list of (B*N, d_model) tensors — one per block, or just
                   the last one when cfg.sigreg_layers == "last". These are
                   detached from the graph; the training loop recomputes the
                   SIGReg loss in the forward pass via collect_hidden=True.

The SIGReg loss is computed externally in train.py so that its gradient
always flows through the same graph as the CE loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from sigreg.config import SIGRegConfig
from sigreg.models.block import TransformerBlock


class SIGRegCausalLM(nn.Module):
    """
    Causal LM for SIGReg experiments.

    forward(tokens, collect_hidden=False):
        tokens:         (B, N) integer token ids
        collect_hidden: if True, return per-layer hidden states for loss computation.
                        Pass True during training, False during evaluation.

    returns: (logits, hidden_states)
        logits:        (B, N, vocab_size)
        hidden_states: list[Tensor(B*N, d_model)] or [] when collect_hidden=False
    """

    def __init__(self, cfg: SIGRegConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        # No final normalisation — consistent with no-norm philosophy.
        # lm_head projects features → logits.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        tokens: torch.Tensor,
        collect_hidden: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B, N = tokens.shape
        x = self.token_embedding(tokens)        # (B, N, d_model)

        hidden_states: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if collect_hidden:
                # Flatten (B, N, D) → (B*N, D) for SIGReg loss
                is_last = (i == len(self.blocks) - 1)
                if self.cfg.sigreg_layers == "all" or is_last:
                    hidden_states.append(x.reshape(B * N, -1))

        logits = self.lm_head(x)               # (B, N, vocab_size)
        return logits, hidden_states


def build_optimizer_groups(
    model: SIGRegCausalLM,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split parameters into Muon (2-D weight matrices) and AdamW (everything else).

    Rules (first match wins):
      1. param is token embedding weight → AdamW  (shared / 1-D effective)
      2. param.ndim == 2                 → Muon
      3. else                            → AdamW

    Returns (muon_params, adamw_params).
    """
    emb_id = id(model.token_embedding.weight)
    seen: set[int] = set()
    muon: list[torch.Tensor] = []
    adamw: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if pid == emb_id:
            adamw.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)

    return muon, adamw
