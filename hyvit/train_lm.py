"""
WikiText-103 training script for hyperbolic and Euclidean causal LMs.

Usage:
    python train_lm.py --model hyplm
    python train_lm.py --model euclm
    python train_lm.py --model hyplm --n_epochs 5
    python train_lm.py --model hyplm --resume checkpoints/hyplm/latest.pt
"""

import argparse
import json
import math
import os
import time

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from config import HypLMConfig, EucLMConfig
from data.wikitext import get_dataloaders
from models.hyplm import HypLM
from models.euclm import EucLM
from optim.riemannian import build_optimizer


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """Returns (val_loss, val_ppl) in nats/token."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        batch   = batch.to(device)
        inputs  = batch[:, :-1]
        targets = batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(inputs)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="mean",
            )
        loss_sum    += loss.item() * inputs.numel()
        token_count += inputs.numel()
    val_loss = loss_sum / token_count
    return val_loss, math.exp(val_loss)


def train(model_name: str, n_epochs_override: int | None, resume: str | None):
    cfg = HypLMConfig() if model_name == "hyplm" else EucLMConfig()
    if n_epochs_override is not None:
        cfg.n_epochs = n_epochs_override

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_name}  Device: {device}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.checkpoint_dir, "metrics.jsonl")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)
    # Optimizer steps per epoch = micro-batches / accumulation steps
    opt_steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_steps         = cfg.n_epochs * opt_steps_per_epoch
    warmup_steps        = int(cfg.warmup_ratio * total_steps)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = (HypLM(cfg) if model_name == "hyplm" else EucLM(cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ── Resume ────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    start_epoch   = 0
    ckpt: dict | None = None
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resuming from epoch {start_epoch} / {cfg.n_epochs}")
        if start_epoch >= cfg.n_epochs:
            print("Already at target epoch count. Nothing to do.")
            return

    sched_total_steps  = (ckpt or {}).get("total_steps",  total_steps)
    sched_warmup_steps = (ckpt or {}).get("warmup_steps", warmup_steps)
    lr_fn     = make_lr_lambda(sched_warmup_steps, sched_total_steps)
    scheduler = LambdaLR(optimizer, lr_fn)
    if ckpt is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    remaining_steps = (cfg.n_epochs - start_epoch) * len(train_loader)
    pbar = tqdm(total=remaining_steps, desc="training", unit="step", dynamic_ncols=True)

    def save_checkpoint(name: str, epoch: int, val_loss: float, val_ppl: float):
        raw_model = getattr(model, "_orig_mod", model)
        torch.save({
            "model":        raw_model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "epoch":        epoch,
            "val_loss":     val_loss,
            "val_ppl":      val_ppl,
            "best_val_loss": best_val_loss,
            "total_steps":  sched_total_steps,
            "warmup_steps": sched_warmup_steps,
        }, os.path.join(cfg.checkpoint_dir, name))

    val_loss, val_ppl = float("inf"), float("inf")

    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        loss_sum, token_count = 0.0, 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch   = batch.to(device)
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(inputs)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean",
                )
            # Scale loss so gradients are averaged over accumulation steps
            raw_loss = loss.item()
            (loss / cfg.grad_accum_steps).backward()

            loss_sum    += raw_loss * inputs.numel()
            token_count += inputs.numel()

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.update(1)
            pbar.set_postfix(epoch=f"{epoch+1}/{cfg.n_epochs}", loss=f"{raw_loss:.4f}")

        # Flush any remaining accumulated gradients at end of epoch
        if (step + 1) % cfg.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = loss_sum / token_count
        train_ppl  = math.exp(train_loss)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        pbar.set_postfix(
            epoch=f"{epoch+1}/{cfg.n_epochs}",
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
        )

        row = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_ppl":  train_ppl,
            "val_loss":   val_loss,
            "val_ppl":    val_ppl,
            "epoch_time_s": epoch_time,
        }
        tqdm.write(json.dumps(row))
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint("best.pt", epoch, val_loss, val_ppl)

        save_checkpoint("latest.pt", epoch, val_loss, val_ppl)

    pbar.close()
    save_checkpoint("final.pt", cfg.n_epochs - 1, val_loss, val_ppl)
    tqdm.write(f"Done. Metrics → {metrics_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a causal LM on WikiText-103")
    p.add_argument("--model",    choices=["hyplm", "euclm"], default="hyplm")
    p.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs")
    p.add_argument("--resume",   type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.model, args.n_epochs, args.resume)
