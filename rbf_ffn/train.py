"""
Training entry point for RBF-FFN WikiText-103 ablation experiments.

Usage:
    python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml
    python -m rbf_ffn.train --config rbf_ffn/configs/g0_baseline.yaml --n_epochs 5
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR

from rbf_ffn.config import RBFFFNConfig, load_config
from rbf_ffn.data import get_dataloaders
from rbf_ffn.models.model import CausalLM, build_optimizer_groups


def get_experiment_dir(cfg: RBFFFNConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (
        f"{stamp}_{cfg.model_type}_{cfg.gate_variant}_{cfg.sigma_variant}"
        f"_d{cfg.d_model}_K{cfg.K}"
    )
    path = Path(__file__).parent / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def collect_sigma_stats(model: CausalLM) -> dict:
    """Collect mean/std of all sigma values (softplus of sigma_raw) across RBF layers.

    sigma_std=0.0 for the global variant (each sigma_raw is a scalar per layer).
    """
    all_sigma = []
    all_scalar = True
    for name, param in model.named_parameters():
        if "sigma_raw" in name:
            all_sigma.append(F.softplus(param).detach().flatten())
            if param.numel() > 1:
                all_scalar = False
    if not all_sigma:
        return {}
    sigma_cat = torch.cat(all_sigma)
    return {
        "sigma_mean": sigma_cat.mean().item(),
        "sigma_std":  0.0 if all_scalar else sigma_cat.std().item(),
    }


@torch.no_grad()
def evaluate(model: CausalLM, loader, device: torch.device) -> tuple[float, float]:
    """Returns (val_loss, val_ppl) in nats/token."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        n_tokens = inputs.numel()
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        loss_sum    += loss.item() * n_tokens
        token_count += n_tokens
    val_loss = loss_sum / token_count
    return val_loss, math.exp(val_loss)


def train(cfg: RBFFFNConfig, config_path: Path, n_epochs: int | None = None) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if n_epochs is not None:
        cfg.n_epochs = n_epochs

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = get_experiment_dir(cfg)
    shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps     = cfg.n_epochs * steps_per_epoch
    warmup_steps    = int(cfg.warmup_ratio * total_steps)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    muon_params, adamw_params = build_optimizer_groups(model)
    muon  = Muon( muon_params,  lr=cfg.muon_lr, momentum=0.95)
    adamw = AdamW(adamw_params, lr=cfg.adamw_lr,
                  weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))

    lr_fn = make_lr_lambda(warmup_steps, total_steps)
    sched_muon  = LambdaLR(muon,  lr_fn)
    sched_adamw = LambdaLR(adamw, lr_fn)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    val_loss, val_ppl = float("inf"), float("inf")   # initialised in case n_epochs=0

    def save_checkpoint(name: str, epoch: int, val_loss: float, val_ppl: float):
        torch.save({
            "model":           model.state_dict(),
            "optimizer_muon":  muon.state_dict(),
            "optimizer_adamw": adamw.state_dict(),
            "scheduler_muon":  sched_muon.state_dict(),
            "scheduler_adamw": sched_adamw.state_dict(),
            "epoch":    epoch,
            "val_loss": val_loss,
            "val_ppl":  val_ppl,
        }, exp_dir / name)

    for epoch in range(cfg.n_epochs):
        model.train()
        loss_sum, token_count = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            batch   = batch.to(device)
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]
            n_tokens = inputs.numel()

            logits = model(inputs)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="mean",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            muon.step();  adamw.step()
            sched_muon.step(); sched_adamw.step()
            muon.zero_grad(); adamw.zero_grad()

            loss_sum    += loss.item() * n_tokens
            token_count += n_tokens

        train_loss = loss_sum / token_count
        train_ppl  = math.exp(train_loss)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        row: dict = {
            "epoch":        epoch,
            "train_loss":   train_loss,
            "train_ppl":    train_ppl,
            "val_loss":     val_loss,
            "val_ppl":      val_ppl,
            "epoch_time_s": epoch_time,
        }
        if cfg.model_type == "rbf":
            row.update(collect_sigma_stats(model))

        print(row)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint("checkpoint_best.pt", epoch, val_loss, val_ppl)

    save_checkpoint("checkpoint_final.pt", cfg.n_epochs - 1, val_loss, val_ppl)
    print(f"Done. Metrics → {metrics_path}")
    return exp_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train RBF-FFN on WikiText-103")
    p.add_argument("--config",   required=True, help="Path to YAML config")
    p.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    path   = Path(args.config)
    cfg    = load_config(path)
    train(cfg, config_path=path, n_epochs=args.n_epochs)