import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from geometry.lorentz import lorentz_inner
from models.patch_embed import HyperbolicPatchEmbed

MANIFOLD_EPS = 1e-3


def on_hyperboloid(x):
    inner = lorentz_inner(x.reshape(-1, x.shape[-1]), x.reshape(-1, x.shape[-1]))
    return torch.allclose(inner, torch.full_like(inner, -1.0), atol=MANIFOLD_EPS)


def test_patch_embed_output_shape():
    embed = HyperbolicPatchEmbed(img_size=32, patch_size=4, in_channels=3, d_model=192)
    x = torch.randn(4, 3, 32, 32)
    out = embed(x)
    # 32/4 = 8 patches per dim → 64 patches + 1 cls token = 65; Lorentz dim = 193
    assert out.shape == (4, 65, 193), f"got {out.shape}"


def test_patch_embed_output_on_manifold():
    embed = HyperbolicPatchEmbed(img_size=32, patch_size=4, in_channels=3, d_model=192)
    x = torch.randn(4, 3, 32, 32)
    out = embed(x)
    assert on_hyperboloid(out)


def test_cls_token_on_manifold():
    embed = HyperbolicPatchEmbed(img_size=32, patch_size=4, in_channels=3, d_model=192)
    x = torch.randn(2, 3, 32, 32)
    out = embed(x)
    cls   = out[:, 0, :]            # class token
    inner = lorentz_inner(cls, cls)
    assert torch.allclose(inner, torch.full_like(inner, -1.0), atol=MANIFOLD_EPS)
