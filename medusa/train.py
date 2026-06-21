"""
Phase 2: train Medusa-1 heads on pre-cached ShareGPT52K features.

Usage:
    python -m medusa.train --config medusa/configs/default.yaml
"""
from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path
from typing import Callable

# Must be set before the CUDA allocator is first initialised.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim import Muon
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from medusa.config import MedusaConfig, load_config
from medusa.data import get_dataloaders
from medusa.models.medusa_model import MedusaModel


def make_lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda



class _BF16Linear(torch.autograd.Function):
    """F.linear with bf16-cast backward.

    Forward identical to F.linear (bf16 under autocast).
    Backward casts grad_output to bf16 before the lm_head matmul so it runs
    on bf16 tensor cores instead of fp32. Weight is a frozen buffer — no
    grad_weight needed.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(weight)
        return F.linear(input, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight,) = ctx.saved_tensors
        grad_input = grad_output.to(weight.dtype) @ weight
        return grad_input, None


def _weighted_ce_loss(
    logits: torch.Tensor,   # (B*K, V)
    targets: torch.Tensor,  # (B, K)
    lambda_decay: float,
) -> torch.Tensor:
    B, K = targets.shape
    loss_per = F.cross_entropy(
        logits,
        targets.reshape(B * K),
        ignore_index=-100,
        reduction="none",
    ).reshape(B, K)
    valid = (targets != -100).float()
    n_valid = valid.sum(0).clamp(min=1)
    weights = logits.new_tensor([lambda_decay ** k for k in range(K)])
    return ((loss_per * valid).sum(0) / n_valid * weights).sum()


def training_step(
    model: MedusaModel,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    cfg: MedusaConfig,
    optimizers: list[Optimizer],
) -> torch.Tensor:
    """
    hidden:     (B, d_model)
    targets:    (B, n_heads) — -100 for padding positions
    Returns:    scalar loss (detached)
    """
    model.train()
    B, D = hidden.shape
    K = cfg.n_heads
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
        h = hidden.to(torch.bfloat16) if hidden.is_cuda else hidden
        hidden_states = model(h)                          # (B, K, D) bf16
        hidden_flat = hidden_states.reshape(B * K, D)
        if hidden.is_cuda:
            logits = _BF16Linear.apply(hidden_flat, model.lm_head_weight)
        else:
            logits = F.linear(hidden_flat, model.lm_head_weight)

    loss = _weighted_ce_loss(logits, targets, cfg.lambda_decay)

    for opt in optimizers:
        opt.zero_grad()
    if loss.grad_fn is not None:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
    return loss.detach()


def _sync_time(profiling: bool) -> float:
    """Wall time, with CUDA sync only when actively profiling."""
    if profiling and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def train(cfg: MedusaConfig, resume: bool = False) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB)")
        print(f"Compute capability: {props.major}.{props.minor}")

    print("Loading teacher LM head weight...")
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    lm_head_w = teacher.get_output_embeddings().weight.detach().clone().bfloat16()
    del teacher
    print(f"LM head weight: {lm_head_w.shape}  dtype={lm_head_w.dtype}")

    model = MedusaModel(cfg, lm_head_w).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    muon_params = list(model.parameters())

    train_loader, val_loader = get_dataloaders(cfg)
    total_steps = cfg.n_epochs * len(train_loader)
    lr_lambda = make_lr_lambda(cfg.warmup_steps, total_steps)

    muon_opt = Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum)
    muon_sched = LambdaLR(muon_opt, lr_lambda)

    optimizers = [muon_opt]

    best_val_loss = float("inf")
    start_epoch = 0
    ckpt_dir = Path(cfg.cache_dir).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "best.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        muon_opt.load_state_dict(ckpt["muon_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # Advance the LR scheduler to match the resumed epoch
        for _ in range(start_epoch * len(train_loader)):
            muon_sched.step()
        print(f"Resumed from epoch {ckpt['epoch']} (best_val_loss={best_val_loss:.4f})")

    epoch_bar = tqdm(range(start_epoch, cfg.n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        profile_batches = 10  # print per-component timing for the first N batches

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.n_epochs}",
                         leave=False, unit="batch")
        t_data = time.perf_counter()
        for hidden, targets in train_bar:
            profiling = epoch == 0 and n_batches < profile_batches
            t0 = _sync_time(profiling)
            hidden = hidden.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            t1 = _sync_time(profiling)
            loss = training_step(model, hidden, targets, cfg, optimizers)
            t2 = _sync_time(profiling)
            muon_sched.step()
            t3 = _sync_time(profiling)

            total_train_loss += loss.item()
            n_batches += 1

            if profiling:
                print(
                    f"  [profile batch {n_batches}] "
                    f"data={t0-t_data:.3f}s  transfer={t1-t0:.3f}s  "
                    f"step={t2-t1:.3f}s  sched={t3-t2:.3f}s"
                )

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{muon_sched.get_last_lr()[0]:.2e}",
            )
            t_data = t3

        avg_train = total_train_loss / max(1, n_batches)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="  Val", leave=False, unit="batch")
            for hidden, targets in val_bar:
                hidden = hidden.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                B_v, D_v = hidden.shape
                K_v = cfg.n_heads
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda):
                    h = hidden.to(torch.bfloat16) if hidden.is_cuda else hidden
                    hidden_states = model(h)
                    logits_v = F.linear(hidden_states.reshape(B_v * K_v, D_v), model.lm_head_weight)
                batch_loss = _weighted_ce_loss(logits_v, targets, cfg.lambda_decay).item()
                val_loss_sum += batch_loss
                n_val += 1
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")

        avg_val = val_loss_sum / max(1, n_val)
        epoch_bar.set_postfix(train=f"{avg_train:.4f}", val=f"{avg_val:.4f}")
        print(f"Epoch {epoch + 1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "muon_state": muon_opt.state_dict(),
                "best_val_loss": avg_val,
                "cfg": cfg,
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Checkpoint saved (val_loss={best_val_loss:.4f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medusa/configs/default.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from best.pt checkpoint")
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)
