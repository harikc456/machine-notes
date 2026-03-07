"""
Hyperbolic Patch Embedding for Vision Transformers.

Pipeline:
  Image (B, C, H, W)
    → extract non-overlapping patches
    → Euclidean linear projection to R^{d_model}
    → project_to_hyperboloid → H^{d_model}
    → prepend class token (on H^{d_model})
    → add positional embeddings (in tangent space, then reproject)
"""

import torch
import torch.nn as nn
from geometry.lorentz import project_to_hyperboloid


class HyperbolicPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int     = 32,
        patch_size: int   = 4,
        in_channels: int  = 3,
        d_model: int      = 192,
        dropout: float    = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.patch_size = patch_size
        self.d_model    = d_model
        n_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Euclidean projection: flattened patch → R^{d_model}
        self.proj = nn.Linear(patch_dim, d_model, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)

        # Class token: learnable vector in tangent space at origin (R^{d_model})
        # Stored as Euclidean; mapped to hyperboloid via project_to_hyperboloid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional embeddings in tangent space (R^{d_model}): +1 for cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns: (B, N+1, d_model+1) — N patches + class token, on H^{d_model}
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Extract non-overlapping patches: (B, n_patches, C*p*p)
        x = x.unfold(2, p, p).unfold(3, p, p)         # (B, C, H//p, W//p, p, p)
        x = x.contiguous().view(B, C, -1, p * p)       # (B, C, n_patches, p²)
        x = x.permute(0, 2, 1, 3).contiguous()         # (B, n_patches, C, p²)
        x = x.view(B, -1, C * p * p)                   # (B, n_patches, C·p²)

        # Project to Euclidean space then lift to hyperboloid
        x = self.proj(x)                                # (B, n_patches, d_model)
        x = project_to_hyperboloid(x)                   # (B, n_patches, d_model+1)

        # Class token: spatial part from learnable param, time part computed
        cls = project_to_hyperboloid(
            self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        )                                               # (B, 1, d_model+1)

        # Prepend class token
        x = torch.cat([cls, x], dim=1)                 # (B, N+1, d_model+1)

        # Add positional embeddings to spatial components, then reproject
        x_space = x[..., 1:] + self.pos_embed          # (B, N+1, d_model)
        x_space = self.dropout(x_space)
        x = project_to_hyperboloid(x_space)             # (B, N+1, d_model+1)

        return x
