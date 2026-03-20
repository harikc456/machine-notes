"""
Training entry point for flow matching on CIFAR-100.

Usage:
    python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml
    python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml --n_steps 50000
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.optim import Muon
except ImportError:
    Muon = None  # type: ignore[assignment,misc]

import matplotlib
matplotlib.use("Agg")

from flow_matching.config import FlowConfig, load_config
from flow_matching.data import build_loaders
from flow_matching.model import DiT, build_optimizer_groups
from flow_matching.sample import euler_sample, save_sample_grid


def _cycle(loader):
    while True:
        yield from loader


def get_experiment_dir(cfg: FlowConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name  = f"{stamp}_{cfg.optimizer}"
    path  = Path(__file__).parent / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _save_plot(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    steps  = [r["step"]       for r in rows]
    losses = [r["train_loss"] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(steps, losses, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss (MSE)")
    ax.set_title("Rectified Flow on CIFAR-100")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(
    cfg:         FlowConfig,
    config_path: Path,
    n_steps:     int | None = None,
) -> Path:
    if n_steps is not None:
        cfg.n_steps = n_steps

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir      = get_experiment_dir(cfg)
    shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, _ = build_loaders(cfg, num_workers=0)
    data_iter       = _cycle(train_loader)

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = DiT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    warmup_steps = int(cfg.warmup_ratio * cfg.n_steps)
    lr_fn        = make_lr_lambda(warmup_steps, cfg.n_steps)

    if cfg.optimizer == "adamw":
        opt      = AdamW(
            model.parameters(),
            lr=cfg.adamw_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
        )
        optimizers = [opt]
        schedulers = [LambdaLR(opt, lr_fn)]
        adamw_opt  = opt
    else:
        if Muon is None:
            raise ImportError(
                "Muon optimizer is not available. "
                "Install a PyTorch build that includes it, or use optimizer: adamw."
            )
        muon_params, adamw_params = build_optimizer_groups(model)
        muon_opt  = Muon(muon_params, lr=cfg.muon_lr, momentum=0.95)
        adamw_opt = AdamW(
            adamw_params,
            lr=cfg.adamw_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
        )
        optimizers = [muon_opt, adamw_opt]
        schedulers = [LambdaLR(muon_opt, lr_fn), LambdaLR(adamw_opt, lr_fn)]

    # ── Training loop ─────────────────────────────────────────────────────────
    rows: list[dict] = []

    for step in range(1, cfg.n_steps + 1):
        model.train()
        x1, y = next(data_iter)
        x1, y = x1.to(device), y.to(device)
        B     = x1.shape[0]

        x0     = torch.randn_like(x1)
        t      = torch.rand(B, device=device)
        t_view = t.view(B, 1, 1, 1)

        # CFG label dropout
        mask   = torch.rand(B, device=device) < cfg.p_uncond
        y_cond = y.clone()
        y_cond[mask] = 100  # null token

        # Rectified Flow interpolation
        xt       = (1 - t_view) * x0 + t_view * x1
        v_target = x1 - x0

        # Forward + loss
        v_pred = model(xt, t, y_cond)
        loss   = F.mse_loss(v_pred, v_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()

        # Logging
        if step % cfg.log_every == 0:
            current_lr = adamw_opt.param_groups[0]["lr"]
            row = {"step": step, "train_loss": loss.item(), "lr": current_lr}
            rows.append(row)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"step {step:>7}  "
                f"train_loss={row['train_loss']:.4f}  "
                f"lr={row['lr']:.2e}"
            )

        # In-training samples
        if step % cfg.sample_every == 0:
            model.eval()
            with torch.no_grad():
                y_sample = torch.arange(100, device=device)
                samples  = euler_sample(
                    model, y_sample, cfg.cfg_scale, cfg.n_steps_euler, device,
                )
            save_sample_grid(samples, exp_dir / f"samples_step_{step}.png")
            model.train()

        # Checkpoint
        if step % cfg.save_every == 0:
            torch.save(
                {
                    "model":      model.state_dict(),
                    "step":       step,
                    "optimizers": [opt.state_dict() for opt in optimizers],
                    "schedulers": [sched.state_dict() for sched in schedulers],
                },
                exp_dir / "ckpt.pt",
            )

    _save_plot(rows, exp_dir / "plot.png")
    print(f"Done. Metrics → {metrics_path}")
    return exp_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Rectified Flow on CIFAR-100")
    p.add_argument("--config",  required=True, help="Path to YAML config")
    p.add_argument("--n_steps", type=int, default=None, help="Override n_steps")
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    cfg_path = Path(args.config)
    cfg      = load_config(cfg_path)
    train(cfg, config_path=cfg_path, n_steps=args.n_steps)
