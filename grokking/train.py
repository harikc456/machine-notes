"""
Training entry point for grokking modular arithmetic experiments.

Usage:
    python -m grokking.train --config grokking/configs/adamw_add.yaml
    python -m grokking.train --config grokking/configs/muon_add.yaml --n_steps 100000
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
matplotlib.use("Agg")   # must be set before any pyplot import; safe to call at module level

from grokking.config import GrokConfig, load_config
from grokking.data import build_dataset, split_dataset
from grokking.model import GrokTransformer, build_optimizer_groups


def get_experiment_dir(cfg: GrokConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name  = f"{stamp}_{cfg.operation}_{cfg.optimizer}"
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


@torch.no_grad()
def _eval_step(
    model: GrokTransformer,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    step: int,
) -> dict:
    model.eval()

    def _metrics(inp: torch.Tensor, lbl: torch.Tensor) -> tuple[float, float]:
        logits = model(inp.to(device))
        loss   = F.cross_entropy(logits, lbl.to(device)).item()
        acc    = (logits.argmax(dim=-1) == lbl.to(device)).float().mean().item()
        return loss, acc

    t_loss, t_acc = _metrics(train_inputs, train_labels)
    v_loss, v_acc = _metrics(val_inputs, val_labels)
    return {
        "step":       step,
        "train_loss": t_loss,
        "train_acc":  t_acc,
        "val_loss":   v_loss,
        "val_acc":    v_acc,
    }


def _save_plot(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt   # matplotlib.use("Agg") already called at module level

    steps     = [r["step"]       for r in rows]
    t_loss    = [r["train_loss"] for r in rows]
    v_loss    = [r["val_loss"]   for r in rows]
    t_acc     = [r["train_acc"]  for r in rows]
    v_acc     = [r["val_acc"]    for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.semilogy(steps, t_loss, label="train", linewidth=1.5)
    ax1.semilogy(steps, v_loss, label="val",   linewidth=1.5)
    ax1.set_ylabel("Loss (log scale)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, t_acc, label="train", linewidth=1.5)
    ax2.plot(steps, v_acc, label="val",   linewidth=1.5)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Grokking: modular arithmetic", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(
    cfg: GrokConfig,
    config_path: Path,
    n_steps: int | None = None,
) -> Path:
    if n_steps is not None:
        cfg.n_steps = n_steps

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = get_experiment_dir(cfg)
    shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    inputs, labels = build_dataset(cfg)
    train_inputs, train_labels, val_inputs, val_labels = split_dataset(
        inputs, labels, cfg
    )
    print(f"Train: {len(train_inputs)}, Val: {len(val_inputs)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GrokTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    warmup_steps = int(cfg.warmup_ratio * cfg.n_steps)
    lr_fn        = make_lr_lambda(warmup_steps, cfg.n_steps)

    if cfg.optimizer == "adamw":
        opt = AdamW(
            model.parameters(),
            lr=cfg.adamw_lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.98),
        )
        optimizers = [opt]
        schedulers = [LambdaLR(opt, lr_fn)]
    else:
        if Muon is None:
            raise ImportError(
                "Muon optimizer is not available. Install a PyTorch build that includes it, "
                "or use optimizer: adamw in your config."
            )
        muon_params, adamw_params = build_optimizer_groups(model)
        muon  = Muon(muon_params, lr=cfg.muon_lr, momentum=0.95)
        adamw = AdamW(
            adamw_params,
            lr=cfg.adamw_lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.98),
        )
        optimizers = [muon, adamw]
        schedulers = [LambdaLR(muon, lr_fn), LambdaLR(adamw, lr_fn)]

    # ── Training loop ─────────────────────────────────────────────────────────
    rng  = torch.Generator()
    rng.manual_seed(cfg.seed)
    rows: list[dict] = []

    for step in range(1, cfg.n_steps + 1):
        model.train()
        idx          = torch.randint(0, len(train_inputs), (cfg.batch_size,), generator=rng)
        batch_inputs = train_inputs[idx].to(device)
        batch_labels = train_labels[idx].to(device)

        logits = model(batch_inputs)
        loss   = F.cross_entropy(logits, batch_labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()

        if step % cfg.log_every == 0:
            row = _eval_step(
                model, train_inputs, train_labels,
                val_inputs, val_labels, device, step,
            )
            rows.append(row)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"step {step:>6}  "
                f"train_loss={row['train_loss']:.4f}  train_acc={row['train_acc']:.3f}  "
                f"val_loss={row['val_loss']:.4f}  val_acc={row['val_acc']:.3f}"
            )

    _save_plot(rows, exp_dir / "plot.png")
    print(f"Done. Metrics -> {metrics_path}")
    return exp_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train grokking transformer")
    p.add_argument("--config",   required=True, help="Path to YAML config")
    p.add_argument("--n_steps",  type=int, default=None, help="Override n_steps")
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    cfg_path = Path(args.config)
    cfg      = load_config(cfg_path)
    train(cfg, config_path=cfg_path, n_steps=args.n_steps)
