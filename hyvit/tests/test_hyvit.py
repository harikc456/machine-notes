import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.hyvit import HyViT
from config import HyViTTinyConfig, HyViTSmallConfig


def test_hyvit_forward_pass():
    cfg    = HyViTTinyConfig()
    model  = HyViT(cfg)
    x      = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, cfg.num_classes), f"got {logits.shape}"


def test_hyvit_no_nan():
    cfg    = HyViTTinyConfig()
    model  = HyViT(cfg)
    x      = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert not torch.isnan(logits).any()


def test_hyvit_parameter_count():
    tiny  = HyViT(HyViTTinyConfig())
    small = HyViT(HyViTSmallConfig())
    n_tiny  = sum(p.numel() for p in tiny.parameters())
    n_small = sum(p.numel() for p in small.parameters())
    assert n_tiny < n_small, "Tiny should have fewer params than Small"
    print(f"\n  Tiny: {n_tiny/1e6:.2f}M  Small: {n_small/1e6:.2f}M")
