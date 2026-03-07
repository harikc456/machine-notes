"""
Hyperbolic Vision Transformer (HyViT).

Architecture:
  patches → HyperbolicPatchEmbed → N × LorentzTransformerBlock
          → LorentzLayerNorm → log_map_origin (CLS token) → linear classifier
"""

import torch
import torch.nn as nn
from models.patch_embed import HyperbolicPatchEmbed
from models.lorentz_block import LorentzTransformerBlock
from models.lorentz_layers import LorentzLayerNorm
from geometry.lorentz import log_map_origin


class HyViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = HyperbolicPatchEmbed(
            img_size    = cfg.img_size,
            patch_size  = cfg.patch_size,
            in_channels = cfg.in_channels,
            d_model     = cfg.d_model,
            dropout     = cfg.embed_dropout,
        )

        self.blocks = nn.ModuleList([
            LorentzTransformerBlock(
                d_model   = cfg.d_model,
                n_heads   = cfg.n_heads,
                mlp_ratio = cfg.mlp_ratio,
                dropout   = cfg.dropout,
            )
            for _ in range(cfg.n_blocks)
        ])

        self.norm = LorentzLayerNorm(cfg.d_model)

        # Classifier: map CLS token from H^{d_model} to logits via log_map
        # log_map_origin gives (0, vₛ); we take vₛ ∈ R^{d_model}
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns: (B, num_classes) logits
        """
        x = self.patch_embed(x)        # (B, N+1, d_model+1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)               # (B, N+1, d_model+1)

        # Extract CLS token, map to Euclidean tangent space, classify
        cls = x[:, 0, :]               # (B, d_model+1)
        cls_euclid = log_map_origin(cls)[..., 1:]   # (B, d_model)

        return self.head(cls_euclid)
