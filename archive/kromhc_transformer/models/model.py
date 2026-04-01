"""Causal Language Model with block dispatch."""
from __future__ import annotations
import torch
import torch.nn as nn
from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.models.transformer_block import LlamaBlock, KromHCBlock


def build_optimizer_groups(model: "CausalLM") -> tuple[list, list]:
    """Split parameters: Muon for 2D weight matrices, AdamW for everything else."""
    emb_id = id(model.token_embedding.weight)
    seen: set[int] = set()
    muon_params: list[torch.Tensor] = []
    adamw_params: list[torch.Tensor] = []

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        if pid == emb_id:
            adamw_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


class CausalLM(nn.Module):
    """
    Causal language model: token_embedding → N × Block → RMSNorm → lm_head.

    Block dispatch:
        "baseline" → LlamaBlock
        "kromhc"   → KromHCBlock
    lm_head weight is tied to token_embedding.
    """

    _BLOCKS = {
        "baseline": LlamaBlock,
        "kromhc": KromHCBlock,
    }

    def __init__(self, cfg: KromHCConfig):
        super().__init__()
        BlockClass = self._BLOCKS[cfg.model_type]
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([BlockClass(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N) → logits: (B, N, vocab_size)"""
        x = self.token_embedding(tokens)
        for block in self.blocks:
            output = block(x)
            x = output[0] if isinstance(output, tuple) else output
        x = self.norm(x)
        return self.lm_head(x)
