import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import HyViTTinyConfig
from models.euclidean_vit import EuclideanViT
from models.hyvit import HyViT


def test_euclidean_vit_output_shape():
    cfg   = HyViTTinyConfig()
    model = EuclideanViT(cfg)
    x     = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 10)


def test_euclidean_vit_no_nan():
    cfg   = HyViTTinyConfig()
    model = EuclideanViT(cfg)
    x     = torch.randn(4, 3, 32, 32)
    assert not torch.isnan(model(x)).any()


def test_similar_parameter_count():
    """HyViT has ~d+1 dims per layer; should be within 20% of EuclideanViT."""
    cfg = HyViTTinyConfig()
    hyp = sum(p.numel() for p in HyViT(cfg).parameters())
    euc = sum(p.numel() for p in EuclideanViT(cfg).parameters())
    ratio = hyp / euc
    assert 0.9 < ratio < 1.2, f"parameter ratio {ratio:.2f} unexpected"
    print(f"\n  HyViT: {hyp/1e6:.2f}M  EuclideanViT: {euc/1e6:.2f}M  ratio: {ratio:.3f}")
