import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.hyvit import HyViT
from config import HyViTTinyConfig
from optim.riemannian import build_optimizer


def test_optimizer_groups():
    cfg = HyViTTinyConfig()
    model = HyViT(cfg)
    opt = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    assert len(opt.param_groups) == 2, f"expected 2 groups, got {len(opt.param_groups)}"


def test_optimizer_step_no_nan():
    cfg    = HyViTTinyConfig()
    model  = HyViT(cfg)
    opt    = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    x      = torch.randn(2, 3, 32, 32)
    loss   = model(x).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in parameter '{name}' after optimizer step"
