"""
Phase 2: train Medusa-1 heads on pre-cached ShareGPT52K features.

Usage:
    python -m medusa.train --config medusa/configs/default.yaml
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
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


def training_step(
    model: MedusaModel,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    cfg: MedusaConfig,
    optimizers: list[Optimizer],
) -> torch.Tensor:
    """
    Single training step. Computes distance-weighted cross-entropy loss,
    back-propagates, clips gradients, steps all optimizers.

    hidden:     (B, d_model)
    targets:    (B, n_heads) — -100 for padding positions
    optimizers: list of optimizers to zero/step (Muon + AdamW in practice)
    Returns:    scalar loss (detached)
    """
    model.train()
    logits = model(hidden)  # (B, n_heads, vocab)

    loss = torch.zeros((), device=logits.device)
    for k in range(logits.shape[1]):
        w = cfg.lambda_decay ** k
        head_loss = F.cross_entropy(logits[:, k, :], targets[:, k], ignore_index=-100)
        if not torch.isnan(head_loss):
            loss = loss + w * head_loss

    for opt in optimizers:
        opt.zero_grad()
    if loss.grad_fn is not None:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
    return loss.detach()


def train(cfg: MedusaConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading teacher LM head weight...")
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id, torch_dtype=torch.bfloat16
    )
    lm_head_w = teacher.get_output_embeddings().weight.detach().clone().float()
    del teacher
    print(f"LM head weight: {lm_head_w.shape}")

    model = MedusaModel(cfg, lm_head_w).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # W1 (d_model×d_model square matrices) → Muon
    # W2 (vocab×d_model large matrices) → AdamW
    muon_params = [p for name, p in model.named_parameters() if name.endswith("W1.weight")]
    adamw_params = [p for name, p in model.named_parameters() if not name.endswith("W1.weight")]

    train_loader, val_loader = get_dataloaders(cfg)
    total_steps = cfg.n_epochs * len(train_loader)
    lr_lambda = make_lr_lambda(cfg.warmup_steps, total_steps)

    muon_opt = Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum)
    adamw_opt = AdamW(adamw_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    muon_sched = LambdaLR(muon_opt, lr_lambda)
    adamw_sched = LambdaLR(adamw_opt, lr_lambda)

    optimizers = [muon_opt, adamw_opt]

    best_val_loss = float("inf")
    ckpt_dir = Path(cfg.cache_dir).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.n_epochs):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for hidden, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.n_epochs}"):
            hidden = hidden.to(device)
            targets = targets.to(device)
            loss = training_step(model, hidden, targets, cfg, optimizers)
            muon_sched.step()
            adamw_sched.step()
            total_train_loss += loss.item()
            n_batches += 1

        avg_train = total_train_loss / max(1, n_batches)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for hidden, targets in val_loader:
                hidden = hidden.to(device)
                targets = targets.to(device)
                logits = model(hidden)  # (B, n_heads, vocab)
                batch_loss = 0.0
                for k in range(logits.shape[1]):
                    w = cfg.lambda_decay ** k
                    head_loss = F.cross_entropy(logits[:, k, :], targets[:, k], ignore_index=-100)
                    if not torch.isnan(head_loss):
                        batch_loss += w * head_loss.item()
                val_loss_sum += batch_loss
                n_val += 1

        avg_val = val_loss_sum / max(1, n_val)
        print(f"Epoch {epoch + 1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "muon_state": muon_opt.state_dict(),
                "adamw_state": adamw_opt.state_dict(),
                "cfg": cfg,
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Checkpoint saved (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medusa/configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
