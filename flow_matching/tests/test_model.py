from __future__ import annotations
import torch
import pytest

from flow_matching.config import FlowConfig


# ── Embedding helpers ─────────────────────────────────────────────────────────

def test_timestep_embedding_shape():
    from flow_matching.model import timestep_embedding
    B, dim = 8, 64
    t = torch.rand(B)
    out = timestep_embedding(t, dim)
    assert out.shape == (B, dim), f"Expected ({B},{dim}), got {out.shape}"


def test_timestep_embedding_dim_must_be_even():
    from flow_matching.model import timestep_embedding
    with pytest.raises(AssertionError):
        timestep_embedding(torch.rand(4), dim=3)


def test_make_2d_sincos_pos_embed_shape():
    from flow_matching.model import make_2d_sincos_pos_embed
    d_model, grid_size = 64, 8
    emb = make_2d_sincos_pos_embed(d_model, grid_size)
    assert emb.shape == (1, grid_size * grid_size, d_model), \
        f"Expected (1,{grid_size**2},{d_model}), got {emb.shape}"


def test_make_2d_sincos_pos_embed_d_model_divisible_by_4():
    from flow_matching.model import make_2d_sincos_pos_embed
    with pytest.raises(AssertionError):
        make_2d_sincos_pos_embed(d_model=6, grid_size=8)


# ── PatchEmbed ────────────────────────────────────────────────────────────────

def test_patch_embed_output_shape():
    from flow_matching.model import PatchEmbed
    B, C, H, W = 2, 3, 32, 32
    patch_size, d_model = 4, 64
    model = PatchEmbed(patch_size=patch_size, d_model=d_model)
    x = torch.randn(B, C, H, W)
    out = model(x)
    n_patches = (H // patch_size) * (W // patch_size)  # 64
    assert out.shape == (B, n_patches, d_model), \
        f"Expected ({B},{n_patches},{d_model}), got {out.shape}"


def test_patch_embed_output_dtype():
    from flow_matching.model import PatchEmbed
    model = PatchEmbed(patch_size=4, d_model=64)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.dtype == torch.float32
