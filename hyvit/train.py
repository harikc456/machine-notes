"""
HyViT training script.

Usage:
    python train.py --config tiny  --model hyvit
    python train.py --config tiny  --model euclidean
    python train.py --config small --model hyvit
    python train.py --config tiny  --model hyvit --resume checkpoints/hyvit_tiny/best.pt
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn

from config import HyViTTinyConfig, HyViTSmallConfig
from data.cifar import build_loaders
from models.hyvit import HyViT
from models.euclidean_vit import EuclideanViT
from optim.riemannian import build_optimizer


def get_lr_multiplier(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss   = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += images.size(0)

    return {"loss": total_loss / total, "acc": total_correct / total}


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += images.size(0)

    return {"loss": total_loss / total, "acc": total_correct / total}


def build_model(model_name: str, cfg):
    if model_name == "hyvit":
        return HyViT(cfg)
    elif model_name == "euclidean":
        return EuclideanViT(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'hyvit' or 'euclidean'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--model",  choices=["hyvit", "euclidean"], default="hyvit")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg    = HyViTTinyConfig() if args.config == "tiny" else HyViTSmallConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Route checkpoint dir by model type
    if args.model == "euclidean":
        cfg.checkpoint_dir = cfg.checkpoint_dir.replace("hyvit", "euclidean")

    print(f"Model: {args.model}  Config: {args.config}  Device: {device}")

    train_loader, val_loader = build_loaders(cfg, num_workers=4)
    model = build_model(args.model, cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params / 1e6:.2f}M")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    optimizer = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: get_lr_multiplier(e, cfg.epochs, cfg.warmup_epochs),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    start_epoch, best_val_acc = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resumed from epoch {ckpt['epoch'] + 1}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        t0      = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, device, cfg)
        val_m   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} | "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']*100:.1f}% | "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']*100:.1f}% | "
            f"lr={lr_now:.2e} | {elapsed:.0f}s"
        )

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_acc":  best_val_acc,
                "cfg":           cfg,
                "model_name":    args.model,
            }, os.path.join(cfg.checkpoint_dir, "best.pt"))

        if (epoch + 1) % 20 == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                os.path.join(cfg.checkpoint_dir, f"epoch_{epoch+1:03d}.pt"),
            )


if __name__ == "__main__":
    main()
