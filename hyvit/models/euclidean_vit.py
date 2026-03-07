"""
Euclidean ViT baseline.

Same architectural skeleton as HyViT — same depth, width, patch size, and
classifier head — but with standard dot-product attention and Euclidean
operations throughout. d_model has no +1 time dimension.

Used to isolate the effect of hyperbolic geometry from other design choices.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=192, dropout=0.0):
        super().__init__()
        n_patches      = (img_size // patch_size) ** 2
        patch_dim      = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.proj       = nn.Linear(patch_dim, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.drop       = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)           # (B, C, H//p, W//p, p, p)
        x = x.contiguous().view(B, C, -1, p * p)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, C * p * p)
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        return self.drop(x + self.pos_embed)


class EuclideanMHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh   = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)
        attn = F.softmax((Q @ K.transpose(-2, -1)) / self.scale, dim=-1)
        attn = self.drop(attn)
        out  = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class EuclideanBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = EuclideanMHSA(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = d_model * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class EuclideanViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbed(
            cfg.img_size, cfg.patch_size, cfg.in_channels, cfg.d_model, cfg.embed_dropout
        )
        self.blocks = nn.ModuleList([
            EuclideanBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))
