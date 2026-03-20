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


# ── DiTBlock ──────────────────────────────────────────────────────────────────

def test_ditblock_output_shape():
    from flow_matching.model import DiTBlock
    B, N, d_model = 2, 64, 64
    block = DiTBlock(d_model=d_model, n_heads=2, mlp_ratio=4.0, dropout=0.0)
    x = torch.randn(B, N, d_model)
    c = torch.randn(B, d_model)
    out = block(x, c)
    assert out.shape == (B, N, d_model), f"Expected ({B},{N},{d_model}), got {out.shape}"


def test_ditblock_adaln_mlp_outputs_6d():
    from flow_matching.model import DiTBlock
    d_model = 64
    block = DiTBlock(d_model=d_model, n_heads=2, mlp_ratio=4.0, dropout=0.0)
    c = torch.randn(2, d_model)
    out = block.adaln_mlp(c)
    assert out.shape == (2, 6 * d_model), f"Expected (2,{6*d_model}), got {out.shape}"


def test_ditblock_adaln_zero_init():
    """At init, adaln_mlp final linear weight and bias are zero-initialized."""
    from flow_matching.model import DiTBlock
    d_model = 64
    block = DiTBlock(d_model=d_model, n_heads=2, mlp_ratio=4.0, dropout=0.0)
    # Final linear of adaln_mlp should be zero-initialized
    final_layer = block.adaln_mlp[-1]
    assert torch.allclose(final_layer.weight, torch.zeros_like(final_layer.weight))
    assert torch.allclose(final_layer.bias,   torch.zeros_like(final_layer.bias))


# ── DiT ───────────────────────────────────────────────────────────────────────

def _tiny_cfg() -> FlowConfig:
    return FlowConfig(d_model=64, n_heads=2, n_layers=2, patch_size=4)


def test_dit_forward_shape():
    from flow_matching.model import DiT
    cfg = _tiny_cfg()
    model = DiT(cfg)
    B = 2
    x = torch.randn(B, 3, 32, 32)
    t = torch.rand(B)
    y = torch.randint(0, 100, (B,))
    out = model(x, t, y)
    assert out.shape == (B, 3, 32, 32), f"Expected ({B},3,32,32), got {out.shape}"


def test_dit_accepts_null_class_token():
    from flow_matching.model import DiT
    cfg = _tiny_cfg()
    model = DiT(cfg)
    B = 2
    x = torch.randn(B, 3, 32, 32)
    t = torch.rand(B)
    y = torch.full((B,), 100)   # null token (index 100)
    out = model(x, t, y)
    assert out.shape == (B, 3, 32, 32)


# ── build_optimizer_groups ────────────────────────────────────────────────────

def test_optimizer_groups_no_embedding_in_muon():
    from flow_matching.model import DiT, build_optimizer_groups
    model = DiT(_tiny_cfg())
    muon_params, adamw_params = build_optimizer_groups(model)

    # Collect names of muon params using id matching
    muon_ids = {id(p) for p in muon_params}
    for name, param in model.named_parameters():
        if id(param) in muon_ids:
            assert not name.startswith("time_embed."), \
                f"time_embed param {name!r} should not be in Muon group"
            assert not name.startswith("class_embed."), \
                f"class_embed param {name!r} should not be in Muon group"
            assert "adaln_mlp." not in name, \
                f"adaln_mlp param {name!r} should not be in Muon group"


def test_optimizer_groups_2d_matrices_in_muon():
    from flow_matching.model import DiT, build_optimizer_groups
    model = DiT(_tiny_cfg())
    muon_params, _ = build_optimizer_groups(model)

    # All muon params must be 2D matrices
    for p in muon_params:
        assert p.ndim == 2, f"Non-2D param in Muon group: shape {p.shape}"


def test_optimizer_groups_covers_all_params():
    from flow_matching.model import DiT, build_optimizer_groups
    model = DiT(_tiny_cfg())
    muon_params, adamw_params = build_optimizer_groups(model)

    all_ids = {id(p) for p in model.parameters()}
    covered = {id(p) for p in muon_params} | {id(p) for p in adamw_params}
    assert all_ids == covered, "Some params not assigned to any optimizer group"
