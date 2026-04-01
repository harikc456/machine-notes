# rbf_ffn/models/model.py
from __future__ import annotations
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import (
    LlamaBlock, RationalBlock, RationalGLUBlock,
    PFDRationalBlock, PFDRationalGLUBlock, FirstOrderPFDRationalBlock,
    PolarMLPBlock, PolarAttnBlock, PolarFullBlock,
    KromHCWrapper,
)


def build_optimizer_groups(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split model parameters into Muon and AdamW groups.

    Rules (first match wins, applied after deduplication by tensor id):
      1. "sigma_raw" in name           → AdamW
      2. param is token embedding weight → AdamW
      3. "delta_" in name              → AdamW  (KroneckerDeltaLinear delta_C/delta_D)
      4. param.ndim == 2               → Muon
      5. else                          → AdamW

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
        elif "delta_" in name:
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
        "polar_attn"     → PolarAttnBlock      (PolarAttention + SwiGLU)
        "polar_full"     → PolarFullBlock      (PolarAttention + AdaptivePolarMLP)

    If cfg.use_kromhc=True, each block is wrapped in KromHCWrapper.

    forward() always returns (logits, hs):
        logits: (B, N, vocab_size)
        hs:     list of H tensors (B, N, n_heads, n_heads) per layer, or [] if not using KromHC
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
            "polar_attn":      PolarAttnBlock,
            "polar_full":      PolarFullBlock,
        }[cfg.model_type]

        def make_block():
            block = BlockClass(cfg)
            if cfg.use_kromhc:
                return KromHCWrapper(block, cfg)
            return block

        self.use_kromhc = cfg.use_kromhc
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([make_block() for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.pre_lm_head_silu = cfg.pre_lm_head_silu
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying: shares the embedding matrix

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        tokens: (B, N) integer token ids
        returns: (logits: (B, N, vocab_size), hs: list of H per layer or [])
        """
        x = self.token_embedding(tokens)
        hs: list[torch.Tensor] = []
        for block in self.blocks:
            result = block(x)
            if self.use_kromhc:
                x, H = result
                hs.append(H.detach())
            else:
                x = result
        x = self.norm(x)
        if self.pre_lm_head_silu:
            x = torch.nn.functional.silu(x)
        return self.lm_head(x), hs
