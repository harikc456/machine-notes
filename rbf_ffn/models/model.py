# rbf_ffn/models/model.py
from __future__ import annotations
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import TransformerBlock, KromHCWrapper, LoopBlock
from rbf_ffn.models.kronecker_linear import KroneckerLMHead, LoRALMHead
from rbf_ffn.models.attn_res import AttnResLayer


def build_optimizer_groups(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split model parameters into Muon and AdamW groups.

    Rules (first match wins, applied after deduplication by tensor id):
      1. "sigma_raw" in name           → AdamW
      2. param is token embedding weight → AdamW
      3. "delta_" in name              → AdamW  (KroneckerDeltaLinear delta_C/delta_D)
      4. "weight_gens" in name         → AdamW  (KromHC gating MLPs — tiny, not suited for Muon)
      5. "mixer_proj" in name          → AdamW  (KromHC output projection)
      6. param.ndim == 2               → Muon
      7. else                          → AdamW

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
        elif "weight_gens" in name:
            adamw.append(param)
        elif "mixer_proj" in name:
            adamw.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)

    return muon, adamw


class CausalLM(nn.Module):
    """
    Causal language model.

        token_embedding → N × TransformerBlock → RMSNorm → lm_head

    Block type is controlled by cfg.attn_type and cfg.ffn_type (see ATTN_REGISTRY
    and FFN_REGISTRY). If cfg.use_kromhc=True, each block is wrapped in KromHCWrapper
    (incompatible with use_attn_res; config validation rejects that combination).

    lm_head variants (selected by cfg):
        default (tie_embeddings=True)  → nn.Linear, weight tied to token_embedding
        tie_embeddings=False           → nn.Linear, independent weight (Muon-trained)
        lm_head_kronecker=True         → KroneckerLMHead; tie_embeddings is ignored
        lm_head_lora_rank>0            → LoRALMHead; tied base + low-rank adapter (A,B → Muon)

    forward() always returns (logits, hs):
        logits: (B, N, vocab_size)
        hs:     list of H tensors (B, N, n_heads, n_heads) per layer, or [] if not using KromHC
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        def make_block(layer_idx: int):
            block = TransformerBlock(cfg, layer_idx=layer_idx)
            if cfg.use_kromhc:
                return KromHCWrapper(block, cfg)
            return block

        self.use_kromhc  = cfg.use_kromhc
        self.use_attn_res = cfg.use_attn_res
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if cfg.use_loop:
            # Layout: loop_n_fixed fixed blocks → 1 shared LoopBlock → loop_n_fixed fixed blocks
            # The shared LoopBlock uses layer_idx=loop_n_fixed (its start position); orthogonal_ffn_layers
            # does not apply meaningfully to a weight-shared block, so orthogonal_ffn bool governs it.
            head = [make_block(i) for i in range(cfg.loop_n_fixed)]
            shared = LoopBlock(TransformerBlock(cfg, layer_idx=cfg.loop_n_fixed), cfg.loop_n_repeats, cfg.d_model, start_layer=cfg.loop_n_fixed)
            tail_start = cfg.loop_n_fixed + cfg.loop_n_repeats
            tail = [make_block(tail_start + i) for i in range(cfg.loop_n_fixed)]
            self.blocks = nn.ModuleList(head + [shared] + tail)
        else:
            self.blocks = nn.ModuleList([make_block(i) for i in range(cfg.n_layers)])
        if cfg.use_attn_res:
            self.attn_res_layers = nn.ModuleList(
                [AttnResLayer(cfg.d_model) for _ in range(len(self.blocks))]
            )
        self.norm = nn.RMSNorm(cfg.d_model)
        self.pre_lm_head_silu = cfg.pre_lm_head_silu
        if cfg.lm_head_kronecker:
            self.lm_head = KroneckerLMHead(cfg.d_model, cfg.vocab_size)
        elif cfg.lm_head_lora_rank > 0:
            self.lm_head = LoRALMHead(cfg.d_model, cfg.vocab_size, cfg.lm_head_lora_rank)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        tokens: (B, N) integer token ids
        returns: (logits: (B, N, vocab_size), hs: list of H per layer or [])
        """
        x = self.token_embedding(tokens)
        hs: list[torch.Tensor] = []
        if self.use_attn_res:
            sources = [x]
            for i, block in enumerate(self.blocks):
                h = self.attn_res_layers[i](sources)
                z = block(h)
                sources.append(z)
            x = sources[-1]
        else:
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
