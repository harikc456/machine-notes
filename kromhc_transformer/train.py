"""Training loop for KromHC Transformer on WikiText-103."""
from __future__ import annotations
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from kromhc_transformer.config import KromHCConfig, load_config
from kromhc_transformer.data import get_dataloaders
from kromhc_transformer.models.model import CausalLM, build_optimizer_groups


def _get_muon_optimizer(params, lr):
    """Return Muon optimizer if available, else AdamW."""
    if hasattr(torch.optim, "Muon"):
        return torch.optim.Muon(params, lr=lr)
    return AdamW(params, lr=lr)


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def get_experiment_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(__file__).parent / "experiments" / "results" / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.no_grad()
def evaluate(model: CausalLM, loader, device: torch.device) -> tuple[float, float]:
    """Returns (avg_loss, perplexity) in nats/token."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        use_cuda = device.type == "cuda"
        with torch.autocast("cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=True):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss_sum += loss.item() * inputs.numel()
        token_count += inputs.numel()
    avg_loss = loss_sum / token_count
    return avg_loss, math.exp(avg_loss)


def train(cfg: KromHCConfig, device: torch.device = None) -> dict:
    """Run full training loop. Returns metrics dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = CausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params | device: {device}")

    muon_params, adamw_params = build_optimizer_groups(model)
    optimizers = []
    if muon_params:
        optimizers.append(_get_muon_optimizer(muon_params, cfg.muon_lr))
    if adamw_params:
        optimizers.append(AdamW(adamw_params, lr=cfg.adamw_lr, weight_decay=cfg.adamw_wd))

    batches_per_epoch = min(cfg.max_train_batches, len(train_loader)) if cfg.max_train_batches > 0 else len(train_loader)
    total_steps = batches_per_epoch * cfg.n_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    schedulers = [LambdaLR(opt, make_lr_lambda(warmup_steps, total_steps)) for opt in optimizers]

    exp_dir = get_experiment_dir()
    metrics: dict = {
        "train_losses": [], "val_losses": [], "val_ppls": [],
        "test_loss": None, "test_ppl": None,
        "n_params": n_params, "config": cfg.__dict__,
    }

    start = time.time()
    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        use_cuda = device.type == "cuda"

        max_batches = cfg.max_train_batches if cfg.max_train_batches > 0 else len(train_loader)
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}", total=max_batches)):
            if step >= max_batches:
                break
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            with torch.autocast("cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=True):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            (loss / cfg.grad_accum_steps).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()
                for sched in schedulers:
                    sched.step()

            epoch_loss += loss.item()
            n_batches += 1

        val_loss, val_ppl = evaluate(model, val_loader, device)
        metrics["train_losses"].append(epoch_loss / n_batches)
        metrics["val_losses"].append(val_loss)
        metrics["val_ppls"].append(val_ppl)
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

    test_loss, test_ppl = evaluate(model, test_loader, device)
    metrics["test_loss"] = test_loss
    metrics["test_ppl"] = test_ppl
    metrics["wall_clock_s"] = time.time() - start

    # Save JSON artifact
    json_path = exp_dir / f"{cfg.model_type}_{cfg.seed}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save checkpoint
    torch.save(model.state_dict(), exp_dir / f"model_{cfg.seed}.pt")
    print(f"Results -> {exp_dir}")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, torch.device(args.device) if args.device else None)
