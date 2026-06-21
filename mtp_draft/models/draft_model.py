from __future__ import annotations
import torch
import torch.nn as nn
from mtp_draft.config import MTPConfig
from mtp_draft.models.fusion import TeacherFeatureFusion
from mtp_draft.models.step_embed import StepEmbedding
from mtp_draft.models.cross_attn_block import CrossAttnBlock
from mtp_draft.models.lora_lm_head import LoRALMHead


class MTPDraftModel(nn.Module):
    """
    Multi-Token Prediction draft model.

    Forward pass:
      1. Fuse teacher hidden states → Q_fused (B, d_draft) via KromHC
      2. Add per-position step embeddings → Q (B, max_draft, d_draft)
      3. Prepend anchor hidden (last cached layer → d_draft) to token embeddings → context
      4. N cross-attention blocks: query = Q, context = context
      5. Project output back to d_teacher, apply frozen LM head + LoRA

    teacher_embedding_weight and teacher_lm_head_weight are registered as
    buffers (no gradient). Only the components listed in trainable_parameters()
    carry gradients.
    """

    def __init__(
        self,
        cfg: MTPConfig,
        teacher_embedding_weight: torch.Tensor,
        teacher_lm_head_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Frozen teacher embedding
        vocab, d_emb = teacher_embedding_weight.shape
        self.token_embedding = nn.Embedding(vocab, d_emb)
        self.token_embedding.weight = nn.Parameter(
            teacher_embedding_weight.detach(), requires_grad=False
        )

        # Context projection: d_teacher → d_draft (trained)
        self.ctx_proj = nn.Linear(cfg.d_teacher, cfg.d_draft, bias=False)
        # Projects anchor position's last-layer hidden → d_draft global context token
        self.anchor_proj = nn.Linear(cfg.d_teacher, cfg.d_draft, bias=False)

        # KromHC multi-layer feature fusion
        self.fusion = TeacherFeatureFusion(
            n_teacher_layers=len(cfg.teacher_layers),
            d_teacher=cfg.d_teacher,
            d_draft=cfg.d_draft,
        )

        # Step conditioning
        self.step_embed = StepEmbedding(d_model=cfg.d_draft, max_steps=cfg.max_draft + 1)

        # Cross-attention blocks
        self.blocks = nn.ModuleList([CrossAttnBlock(cfg) for _ in range(cfg.n_blocks)])

        # Output projection: d_draft → d_teacher (trained)
        self.out_proj = nn.Linear(cfg.d_draft, cfg.d_teacher, bias=False)

        # Frozen LM head weight + trainable LoRA
        self.lm_head = LoRALMHead(teacher_lm_head_weight, cfg.lora_rank)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        teacher_hiddens: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        teacher_hiddens: (B, n_teacher_layers, d_teacher)
        context_ids:     (B, seq_len)
        returns:         (B, max_draft, vocab)
        """
        B = teacher_hiddens.shape[0]
        device = teacher_hiddens.device

        # 1. Fuse teacher features → (B, d_draft)
        q_fused = self.fusion(teacher_hiddens)

        # 2. Step embeddings for each draft position
        steps = torch.arange(1, self.cfg.max_draft + 1, device=device)
        steps = steps.unsqueeze(0).expand(B, -1)          # (B, max_draft)
        step_embs = self.step_embed(steps)                 # (B, max_draft, d_draft)
        query = q_fused.unsqueeze(1) + step_embs           # (B, max_draft, d_draft)

        # 3. Context: anchor hidden (last cached layer) + token embeddings
        anchor = self.anchor_proj(
            teacher_hiddens[:, -1, :].to(self.anchor_proj.weight.dtype)
        ).unsqueeze(1)                                     # (B, 1, d_draft)
        ctx_emb = self.token_embedding(context_ids)        # (B, seq_len, d_teacher)
        ctx_proj = self.ctx_proj(ctx_emb)                  # (B, seq_len, d_draft)
        context = torch.cat([anchor, ctx_proj], dim=1)     # (B, 1+seq_len, d_draft)

        # 4. Cross-attention blocks
        for block in self.blocks:
            query = block(query, context)                  # (B, max_draft, d_draft)

        # 5. Project to d_teacher, apply LM head + LoRA
        out = self.out_proj(query)                         # (B, max_draft, d_teacher)
        return self.lm_head(out)                           # (B, max_draft, vocab)
