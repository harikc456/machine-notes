"""
Training entry point for WikiText-103 Mamba experiments.

Usage:
    python -m mamba_lm.train --config mamba_lm/configs/baseline.yaml
    python -m mamba_lm.train --config mamba_lm/configs/baseline.yaml --n_epochs 5
    python -m mamba_lm.train --resume mamba_lm/experiments/<name> --resume_from latest
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from mamba_lm.config import MambaConfig, load_config
from mamba_lm.data import get_dataloaders
from mamba_lm.models.mamba import CausalMambaLM, build_optimizer_groups


def get_experiment_dir(cfg: MambaConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name  = f"{stamp}_mamba_d{cfg.d_model}_L{cfg.n_layers}_N{cfg.d_state}"
    path  = Path(__file__).parent / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


@torch.no_grad()
def evaluate(model: CausalMambaLM, loader, device: torch.device) -> tuple[float, float]:
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


def train(
    cfg: MambaConfig,
    config_path: Path,
    n_epochs: int | None = None,
    resume_checkpoint: Path | None = None,
) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if n_epochs is not None:
        cfg.n_epochs = n_epochs

    t_init = time.time()

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.time()-t_init:.1f}s] Device: {device}")

    if resume_checkpoint is not None:
        exp_dir = resume_checkpoint.parent
    else:
        exp_dir = get_experiment_dir(cfg)
        shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"[{time.time()-t_init:.1f}s] Experiment dir: {exp_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"[{time.time()-t_init:.1f}s] Building dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps     = cfg.n_epochs * steps_per_epoch
    optimizer_steps = total_steps // cfg.grad_accum_steps
    warmup_steps    = int(cfg.warmup_ratio * optimizer_steps)
    print(f"[{time.time()-t_init:.1f}s] Dataloaders ready — {steps_per_epoch} steps/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[{time.time()-t_init:.1f}s] Building model...")
    model = CausalMambaLM(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{time.time()-t_init:.1f}s] Parameters: {n_params:,} — moving to {device}...")
    model = model.to(device)
    print(f"[{time.time()-t_init:.1f}s] Model on device")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print(f"[{time.time()-t_init:.1f}s] Building optimizer...")
    wd_params, no_wd_params = build_optimizer_groups(model)
    optimizer = AdamW(
        [
            {"params": wd_params,    "weight_decay": cfg.weight_decay},
            {"params": no_wd_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    print(f"[{time.time()-t_init:.1f}s] Optimizer ready")

    # ── Resume ────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    val_loss, val_ppl = float("inf"), float("inf")
    start_epoch = 0
    ckpt: dict | None = None
    if resume_checkpoint is not None:
        print(f"[{time.time()-t_init:.1f}s] Loading checkpoint {resume_checkpoint}...")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[{time.time()-t_init:.1f}s] Resuming from epoch {start_epoch} / {cfg.n_epochs}")
        if start_epoch >= cfg.n_epochs:
            print("Already at target epoch count. Nothing to do.")
            return exp_dir

    sched_optimizer_steps = (ckpt or {}).get("optimizer_steps", optimizer_steps)
    sched_warmup_steps    = (ckpt or {}).get("warmup_steps",    warmup_steps)
    lr_fn     = make_lr_lambda(sched_warmup_steps, sched_optimizer_steps)
    scheduler = LambdaLR(optimizer, lr_fn)
    if ckpt is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    # ── Compile ───────────────────────────────────────────────────────────────
    # torch.compile fuses the selective_scan loop into efficient CUDA kernels.
    if device.type == "cuda":
        print(f"[{time.time()-t_init:.1f}s] Compiling model (torch.compile)...")
        model = torch.compile(model, dynamic=False)
        print(f"[{time.time()-t_init:.1f}s] Compile done (first forward pass will still trigger JIT)")

    print(f"[{time.time()-t_init:.1f}s] Setup complete — starting training loop")

    remaining_steps = (cfg.n_epochs - start_epoch) * steps_per_epoch
    pbar = tqdm(total=remaining_steps, desc="training", unit="step", dynamic_ncols=True)

    def save_checkpoint(name: str, epoch: int, val_loss: float, val_ppl: float):
        raw_model = getattr(model, "_orig_mod", model)
        torch.save({
            "model":           raw_model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "epoch":           epoch,
            "val_loss":        val_loss,
            "val_ppl":         val_ppl,
            "best_val_loss":   best_val_loss,
            "optimizer_steps": sched_optimizer_steps,
            "warmup_steps":    sched_warmup_steps,
        }, exp_dir / name)

    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        loss_sum, token_count = 0.0, 0
        t0 = time.time()
        global_step = 0

        for batch in train_loader:
            batch   = batch.to(device)
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]

            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(inputs)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean",
                )

            raw_loss = loss.item()
            (loss / cfg.grad_accum_steps).backward()

            global_step += 1
            loss_sum    += raw_loss * inputs.numel()
            token_count += inputs.numel()

            if global_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.update(1)
            pbar.set_postfix(epoch=f"{epoch+1}/{cfg.n_epochs}", loss=f"{raw_loss:.4f}")

        # Flush remaining accumulated gradients
        if global_step % cfg.grad_accum_steps != 0:
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

        row: dict = {
            "epoch":                epoch,
            "train_loss":           train_loss,
            "train_ppl":            train_ppl,
            "val_loss":             val_loss,
            "val_ppl":              val_ppl,
            "epoch_time_s":         epoch_time,
            "effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
        }
        print(row)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint("checkpoint_best.pt", epoch, val_loss, val_ppl)

        save_checkpoint("checkpoint_latest.pt", epoch, val_loss, val_ppl)

    pbar.close()
    save_checkpoint("checkpoint_final.pt", cfg.n_epochs - 1, val_loss, val_ppl)
    print(f"Done. Metrics → {metrics_path}")
    return exp_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train Mamba on WikiText-103")
    p.add_argument("--config",      default=None)
    p.add_argument("--n_epochs",    type=int, default=None)
    p.add_argument("--resume",      type=str, default=None,
                   help="Experiment directory to resume from")
    p.add_argument("--resume_from", choices=["best", "final", "latest"], default=None)
    args = p.parse_args()
    if args.resume is None and args.config is None:
        p.error("--config is required when not using --resume")
    if args.resume_from is not None and args.resume is None:
        p.error("--resume_from requires --resume")
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.resume is not None:
        resume_dir        = Path(args.resume)
        path              = resume_dir / "config.yaml"
        cfg               = load_config(path)
        resume_checkpoint = resume_dir / f"checkpoint_{args.resume_from or 'latest'}.pt"
    else:
        path = Path(args.config)
        cfg  = load_config(path)
        resume_checkpoint = None

    train(cfg, config_path=path, n_epochs=args.n_epochs,
          resume_checkpoint=resume_checkpoint)
