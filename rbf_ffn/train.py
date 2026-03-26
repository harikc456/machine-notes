"""
Training entry point for WikiText-103 ablation experiments.

Usage:
    python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml
    python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_qk_norm_weight_norm.yaml --n_epochs 5
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
from tqdm import tqdm

from rbf_ffn.config import RBFFFNConfig, load_config
from rbf_ffn.data import get_dataloaders
from rbf_ffn.models.model import CausalLM, build_optimizer_groups
from rbf_ffn.models.rational_ffn import PFDRationalActivation, RationalActivation


def get_experiment_dir(cfg: RBFFFNConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{stamp}_{cfg.model_type}_d{cfg.d_model}"
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


@torch.no_grad()
def apply_linear_weight_norm(model: CausalLM, target_norm: float) -> None:
    """Normalise each output neuron (row) of every Linear weight to `target_norm`.

    Skips weight-tied layers (lm_head shares the token embedding matrix).
    """
    tied_id = id(model.token_embedding.weight)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if id(module.weight) == tied_id:
                continue
            norms = module.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            module.weight.data.mul_(target_norm / norms)


_ACTIVATION_COEFF_NORM = 2.0


@torch.no_grad()
def apply_activation_coeff_norm(model: CausalLM) -> None:
    """Normalise each coefficient vector in rational/PFD activations to L2 norm 2.0.

    For RationalActivation: normalises a and b independently.
    For PFDRationalActivation: normalises a, b, and c independently; skips gamma (scalar).
    """
    for module in model.modules():
        if isinstance(module, RationalActivation):
            for param in (module.a, module.b):
                norm = param.data.norm().clamp(min=1e-8)
                param.data.mul_(_ACTIVATION_COEFF_NORM / norm)
        elif isinstance(module, PFDRationalActivation):
            for param in (module.a, module.b, module.c):
                norm = param.data.norm().clamp(min=1e-8)
                param.data.mul_(_ACTIVATION_COEFF_NORM / norm)


@torch.no_grad()
def evaluate(model: CausalLM, loader, device: torch.device) -> tuple[float, float]:
    """Returns (val_loss, val_ppl) in nats/token."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        n_tokens = inputs.numel()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
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
    total_steps     = cfg.n_epochs * steps_per_epoch          # micro-batches; used for pbar
    optimizer_steps = total_steps // cfg.grad_accum_steps     # optimizer updates; used for LR
    warmup_steps    = int(cfg.warmup_ratio * optimizer_steps)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    muon_params, adamw_params = build_optimizer_groups(model)
    muon  = Muon( muon_params,  lr=cfg.muon_lr, momentum=0.95)
    adamw = AdamW(adamw_params, lr=cfg.adamw_lr,
                  weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))

    lr_fn = make_lr_lambda(warmup_steps, optimizer_steps)
    sched_muon  = LambdaLR(muon,  lr_fn)
    sched_adamw = LambdaLR(adamw, lr_fn)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    val_loss, val_ppl = float("inf"), float("inf")   # initialised in case n_epochs=0
    pbar = tqdm(total=total_steps, desc="training", unit="step", dynamic_ncols=True)

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

        global_step = 0
        for batch in train_loader:
            batch    = batch.to(device)
            inputs   = batch[:, :-1]
            targets  = batch[:, 1:]
            n_tokens = inputs.numel()

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits = model(inputs)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean",
                )

            raw_loss = loss.item()                    # capture before dividing
            loss     = loss / cfg.grad_accum_steps    # non-in-place rebind
            loss.backward()

            global_step += 1                          # increment before gate so step N triggers after N micro-batches

            if global_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                muon.step();  adamw.step()
                sched_muon.step(); sched_adamw.step()
                muon.zero_grad(set_to_none=True)
                adamw.zero_grad(set_to_none=True)
                if cfg.linear_weight_norm:
                    apply_linear_weight_norm(model, cfg.linear_weight_norm_value)
                if cfg.activation_norm:
                    apply_activation_coeff_norm(model)

            loss_sum    += raw_loss * n_tokens
            token_count += n_tokens

            pbar.update(1)
            pbar.set_postfix(
                epoch=f"{epoch+1}/{cfg.n_epochs}",
                loss=f"{raw_loss:.4f}",
            )

        # Flush remaining accumulated gradients at epoch end.
        # No scheduler step: LR stays at last scheduled value (no full window completed).
        if global_step % cfg.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            muon.step();  adamw.step()
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            if cfg.linear_weight_norm:
                apply_linear_weight_norm(model, cfg.linear_weight_norm_value)
            if cfg.activation_norm:
                apply_activation_coeff_norm(model)

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

    pbar.close()
    save_checkpoint("checkpoint_final.pt", cfg.n_epochs - 1, val_loss, val_ppl)
    print(f"Done. Metrics → {metrics_path}")
    return exp_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train on WikiText-103")
    p.add_argument("--config",   required=True, help="Path to YAML config")
    p.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    path   = Path(args.config)
    cfg    = load_config(path)
    train(cfg, config_path=path, n_epochs=args.n_epochs)