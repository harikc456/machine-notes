import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import HyViTTinyConfig
from models.hyvit import HyViT
from models.euclidean_vit import EuclideanViT
from optim.riemannian import build_optimizer


def _run_steps(model, device, cfg, n_steps=2):
    opt  = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    for step in range(n_steps):
        x    = torch.randn(4, 3, 32, 32, device=device)
        y    = torch.randint(0, 10, (4,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        assert not torch.isnan(loss), f"NaN loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    return loss.item()


def test_hyvit_smoke():
    cfg    = HyViTTinyConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HyViT(cfg).to(device)
    loss   = _run_steps(model, device, cfg)
    print(f"\n  HyViT smoke loss: {loss:.4f}")


def test_euclidean_vit_smoke():
    cfg    = HyViTTinyConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EuclideanViT(cfg).to(device)
    loss   = _run_steps(model, device, cfg)
    print(f"\n  EuclideanViT smoke loss: {loss:.4f}")
