"""
Phase 2: train the MTP draft model on pre-cached HotpotQA features.

Usage:
    python -m mtp_draft.train --config mtp_draft/configs/default.yaml
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from mtp_draft.config import MTPConfig, load_config
from mtp_draft.data import get_dataloaders
from mtp_draft.models.draft_model import MTPDraftModel


def make_lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """Cosine decay with linear warmup. Returns lr multiplier in [0, 1]."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def training_step(
    model: MTPDraftModel,
    hiddens: torch.Tensor,
    context_ids: torch.Tensor,
    targets: torch.Tensor,
    cfg: MTPConfig,
    optimizer: AdamW,
) -> torch.Tensor:
    """
    Single training step. Computes distance-weighted cross-entropy loss,
    back-propagates, clips gradients, and steps the optimizer.

    hiddens:     (B, n_layers, d_teacher)
    context_ids: (B, max_prompt_len)
    targets:     (B, max_draft) — -100 for positions beyond the answer
    Returns scalar loss tensor (detached).
    """
    model.train()
    logits = model(hiddens, context_ids)  # (B, max_draft, vocab)

    weighted_sum = torch.zeros((), device=logits.device)
    for i in range(cfg.max_draft):
        weight = cfg.lambda_decay ** i
        loss_i = F.cross_entropy(logits[:, i, :], targets[:, i], ignore_index=-100)
        weighted_sum = weighted_sum + weight * loss_i
    loss = weighted_sum / cfg.max_draft

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return loss.detach()


def train(cfg: MTPConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen teacher weights (CPU, no grad), then offload teacher model
    print("Loading teacher weights for embedding and LM head...")
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id,
        torch_dtype=torch.bfloat16,
    )
    teacher_embed_w = teacher.model.embed_tokens.weight.detach().clone().float()
    teacher_lm_head_w = teacher.lm_head.weight.detach().clone().float()
    del teacher
    print("Teacher weights extracted, model offloaded.")

    model = MTPDraftModel(cfg, teacher_embed_w, teacher_lm_head_w).to(device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"Trainable parameters: {n_trainable:,}")

    train_loader, val_loader = get_dataloaders(cfg)
    total_steps = cfg.n_epochs * len(train_loader)

    optimizer = AdamW(
        model.trainable_parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = LambdaLR(optimizer, make_lr_lambda(cfg.warmup_steps, total_steps))

    best_val_loss = float("inf")
    ckpt_dir = Path(cfg.cache_dir).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.n_epochs):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for hiddens, ctx_ids, targets, _valid_len in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.n_epochs}"
        ):
            hiddens = hiddens.to(device)
            ctx_ids = ctx_ids.to(device)
            targets = targets.to(device)

            loss = training_step(model, hiddens, ctx_ids, targets, cfg, optimizer)
            scheduler.step()
            total_train_loss += loss.item()
            n_batches += 1

        avg_train = total_train_loss / max(1, n_batches)

        # Validation — no backward pass, accumulate weighted loss
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for hiddens, ctx_ids, targets, _valid_len in val_loader:
                hiddens = hiddens.to(device)
                ctx_ids = ctx_ids.to(device)
                targets = targets.to(device)
                logits = model(hiddens, ctx_ids)
                batch_loss = 0.0
                for i in range(cfg.max_draft):
                    w = cfg.lambda_decay ** i
                    batch_loss += w * F.cross_entropy(
                        logits[:, i, :], targets[:, i], ignore_index=-100
                    ).item()
                val_loss_sum += batch_loss / cfg.max_draft
                n_val += 1

        avg_val = val_loss_sum / max(1, n_val)
        print(f"Epoch {epoch + 1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt = {
                "epoch": epoch,
                "model_state": {
                    k: v
                    for k, v in model.state_dict().items()
                    if "token_embedding" not in k and "lm_head.weight" not in k
                },
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Checkpoint saved (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mtp_draft/configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
