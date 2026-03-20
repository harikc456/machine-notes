"""
Euler ODE sampler and sample grid utilities for Rectified Flow.

CLI usage:
    python -m flow_matching.sample \\
        --config flow_matching/configs/dit_cfg.yaml \\
        --checkpoint <exp_dir>/ckpt.pt \\
        --cfg_scale 3.0 \\
        --out samples.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torchvision

import matplotlib
matplotlib.use("Agg")

from flow_matching.config import FlowConfig, load_config
from flow_matching.data import CIFAR100_MEAN, CIFAR100_STD


def euler_sample(
    model:     "DiT",
    y:         torch.Tensor,
    cfg_scale: float,
    n_steps:   int,
    device:    torch.device,
) -> torch.Tensor:
    """Euler ODE integrator from t=0 (noise) to t=1 (data) with CFG.

    y:         (B,) class indices in [0, 99]; pass 100 for unconditional
    cfg_scale: guidance scale (0 = no guidance, 1 = conditional only)
    n_steps:   number of Euler steps
    Returns:   (B, 3, 32, 32) generated images (normalised, not pixel-space)
    """
    B        = y.shape[0]
    y_null   = torch.full_like(y, 100)  # null CFG token
    x        = torch.randn(B, 3, 32, 32, device=device)
    dt       = 1.0 / n_steps

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            t       = i / n_steps
            t_batch = torch.full((B,), t, dtype=torch.float32, device=device)

            v_cond   = model(x, t_batch, y.to(device))
            v_uncond = model(x, t_batch, y_null.to(device))
            v        = v_uncond + cfg_scale * (v_cond - v_uncond)

            x = x + v * dt

    return x


def save_sample_grid(
    images: torch.Tensor,
    path:   Path,
    mean:   tuple[float, ...] = CIFAR100_MEAN,
    std:    tuple[float, ...] = CIFAR100_STD,
    nrow:   int = 10,
) -> None:
    """Save a (N, 3, 32, 32) tensor as a PNG grid after denormalising."""
    import matplotlib.pyplot as plt

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)
    imgs   = images.cpu() * std_t + mean_t
    imgs   = imgs.clamp(0, 1)

    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2)  # (3, H, W)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a trained Rectified Flow model")
    p.add_argument("--config",       required=True,  help="Path to YAML config")
    p.add_argument("--checkpoint",   required=True,  help="Path to ckpt.pt")
    p.add_argument("--cfg_scale",    type=float, default=None, help="CFG scale override")
    p.add_argument("--n_steps_euler",type=int,   default=None, help="Euler steps override")
    p.add_argument("--out",          default="samples.png",    help="Output PNG path")
    return p.parse_args()


if __name__ == "__main__":
    from flow_matching.model import DiT

    args    = _parse_args()
    cfg     = load_config(args.config)
    if args.cfg_scale    is not None: cfg.cfg_scale    = args.cfg_scale
    if args.n_steps_euler is not None: cfg.n_steps_euler = args.n_steps_euler

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = DiT(cfg).to(device)
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y       = torch.arange(100, device=device)  # one sample per class
    samples = euler_sample(model, y, cfg.cfg_scale, cfg.n_steps_euler, device)
    save_sample_grid(samples, Path(args.out))
    print(f"Saved sample grid → {args.out}")
