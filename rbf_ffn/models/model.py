# rbf_ffn/models/model.py
from __future__ import annotations
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import LlamaBlock, RationalBlock, RationalGLUBlock, PFDRationalBlock, PFDRationalGLUBlock, FirstOrderPFDRationalBlock, PolarMLPBlock


def build_optimizer_groups(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split model parameters into Muon and AdamW groups.

    Rules (first match wins, applied after deduplication by tensor id):
      1. "sigma_raw" in name           → AdamW
      2. param is token embedding weight → AdamW
      3. param.ndim == 2               → Muon
      4. else                          → AdamW

    Returns (muon_params, adamw_params).
    """
    emb_id = id(model.token_embedding.weight)   # type: ignore[attr-defined]
    seen: set[int] = set()
    muon: list[torch.Tensor] = []
    adamw: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if "sigma_raw" in name:
            adamw.append(param)
        elif pid == emb_id:
            adamw.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)

    return muon, adamw


class CausalLM(nn.Module):
    """
    Causal language model.

        token_embedding → N × Block → RMSNorm → lm_head (weight-tied)

    Block type is selected by cfg.model_type:
        "baseline"       → LlamaBlock          (SwiGLU FFN)
        "rational"       → RationalBlock       (RationalFFN)
        "rationalglu"    → RationalGLUBlock    (RationalGatedFFN)
        "pfd_rational"   → PFDRationalBlock    (PFDRationalFFN)
        "pfd_rationalglu"→ PFDRationalGLUBlock (PFDRationalGatedFFN)
        "first_order_pfd_rational" → FirstOrderPFDRationalBlock (FirstOrderPFDRationalFFN)
        "polar_mlp"      → PolarMLPBlock       (AdaptivePolarMLP)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        BlockClass = {
            "baseline":        LlamaBlock,
            "rational":        RationalBlock,
            "rationalglu":     RationalGLUBlock,
            "pfd_rational":    PFDRationalBlock,
            "pfd_rationalglu": PFDRationalGLUBlock,
            "first_order_pfd_rational": FirstOrderPFDRationalBlock,
            "polar_mlp":       PolarMLPBlock,
        }[cfg.model_type]
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([BlockClass(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: lm_head shares the embedding matrix
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N) integer token ids
        returns: logits (B, N, vocab_size)
        """
        x = self.token_embedding(tokens)   # (B, N, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)             # (B, N, vocab_size)
