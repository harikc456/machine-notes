"""
Training entry point for WikiText-103 ablation experiments.

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
from tqdm import tqdm

from rbf_ffn.config import ModelConfig, load_config
from rbf_ffn.data import get_dataloaders
from rbf_ffn.models.model import CausalLM, build_optimizer_groups
from rbf_ffn.models.rational_ffn import PFDRationalActivation, RationalActivation


def get_experiment_dir(cfg: ModelConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    norm_tags = ""
    if cfg.qk_norm:
        norm_tags += "_qknorm"
    if cfg.linear_weight_norm:
        norm_tags += "_wnorm"
    if cfg.adaptive_weight_norm:
        norm_tags += "_adpwnorm"
    if cfg.activation_norm:
        norm_tags += "_actnorm"
    name = f"{stamp}_{cfg.model_type}{norm_tags}_d{cfg.d_model}"
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
    """Collect mean/std of all sigma values (softplus of sigma_raw) across layers.

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


@torch.no_grad()
def apply_adaptive_weight_norm(
    model: CausalLM,
    cfg: ModelConfig,
    delta_log_gap: float,
) -> None:
    """Apply per-layer adaptive weight norm.

    Target norm decreases linearly from cfg.adaptive_norm_early (layer 0)
    to cfg.adaptive_norm_late (layer L-1).  A phase-aware derivative correction
    proportional to tanh(beta * delta_log_gap) is applied most strongly to late
    layers (correction weight = l/(L-1)).  A hard floor of 1.0 is enforced on
    every target to prevent flat-curvature dead zones.

    Iterates model.blocks only — lm_head (weight-tied to token_embedding) and
    embeddings are excluded by design; no explicit tie-guard is needed because
    those modules live outside model.blocks.
    """
    L = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        static = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - frac)
        correction = cfg.adaptive_norm_gamma * frac * math.tanh(cfg.adaptive_norm_beta * delta_log_gap)
        target = max(1.0, static - correction)

        for module in block.modules():
            if isinstance(module, nn.Linear):
                norms = module.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
                module.weight.data.mul_(target / norms)


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


def train(
    cfg: ModelConfig,
    config_path: Path,
    n_epochs: int | None = None,
    resume_checkpoint: Path | None = None,
) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if n_epochs is not None:
        cfg.n_epochs = n_epochs

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if resume_checkpoint is not None:
        exp_dir = resume_checkpoint.parent
    else:
        exp_dir = get_experiment_dir(cfg)
        shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps     = cfg.n_epochs * steps_per_epoch          # micro-batches; used for optimizer_steps
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

    # ── Resume ────────────────────────────────────────────────────────────────
    # Load into the uncompiled CausalLM so checkpoint keys always match.
    # torch.compile wraps the model in OptimizedModule whose state_dict() yields
    # "_orig_mod.*" keys; loading before compile avoids that mismatch entirely.
    best_val_loss = float("inf")
    val_loss, val_ppl = float("inf"), float("inf")   # initialised in case n_epochs=0
    ema_log_gap: float = 0.0
    delta_log_gap: float = 0.0
    start_epoch = 0
    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        muon.load_state_dict(ckpt["optimizer_muon"])
        adamw.load_state_dict(ckpt["optimizer_adamw"])
        sched_muon.load_state_dict(ckpt["scheduler_muon"])
        sched_adamw.load_state_dict(ckpt["scheduler_adamw"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        ema_log_gap   = ckpt.get("ema_log_gap", 0.0)
        print(f"Resuming from epoch {start_epoch} / {cfg.n_epochs}")
        if start_epoch >= cfg.n_epochs:
            print("Already at target epoch count. Nothing to do.")
            return exp_dir

    # ── Compile ───────────────────────────────────────────────────────────────
    # Compile AFTER resume so the checkpoint load above always targets the plain
    # CausalLM — not the OptimizedModule wrapper with "_orig_mod.*" key names.
    if device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    remaining_steps = (cfg.n_epochs - start_epoch) * steps_per_epoch
    pbar = tqdm(total=remaining_steps, desc="training", unit="step", dynamic_ncols=True)

    def save_checkpoint(name: str, epoch: int, val_loss: float, val_ppl: float):
        # Unwrap OptimizedModule (if compiled) so saved keys never have the
        # "_orig_mod." prefix — checkpoints remain loadable without compile.
        raw_model = getattr(model, "_orig_mod", model)
        torch.save({
            "model":           raw_model.state_dict(),
            "optimizer_muon":  muon.state_dict(),
            "optimizer_adamw": adamw.state_dict(),
            "scheduler_muon":  sched_muon.state_dict(),
            "scheduler_adamw": sched_adamw.state_dict(),
            "epoch":         epoch,
            "val_loss":      val_loss,
            "val_ppl":       val_ppl,
            "best_val_loss": best_val_loss,
            "ema_log_gap":   ema_log_gap,
        }, exp_dir / name)

    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        loss_sum, token_count = 0.0, 0
        t0 = time.time()

        global_step = 0
        for batch in train_loader:
            batch    = batch.to(device)
            inputs   = batch[:, :-1]
            targets  = batch[:, 1:]
            n_tokens = inputs.numel()

            torch.compiler.cudagraph_mark_step_begin()
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
                if cfg.adaptive_weight_norm:
                    apply_adaptive_weight_norm(model, cfg, delta_log_gap)
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
            if cfg.adaptive_weight_norm:
                apply_adaptive_weight_norm(model, cfg, delta_log_gap)
            if cfg.activation_norm:
                apply_activation_coeff_norm(model)

        train_loss = loss_sum / token_count
        train_ppl  = math.exp(train_loss)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        if cfg.adaptive_weight_norm:
            log_gap = math.log(max(val_loss, 1e-8) / max(train_loss, 1e-8))
            new_ema = cfg.adaptive_norm_alpha * log_gap + (1.0 - cfg.adaptive_norm_alpha) * ema_log_gap
            delta_log_gap = new_ema - ema_log_gap
            ema_log_gap = new_ema

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
    p = argparse.ArgumentParser(description="Train on WikiText-103")
    p.add_argument("--config",      default=None,
                   help="Path to YAML config (not used when --resume is set)")
    p.add_argument("--n_epochs",    type=int, default=None, help="Override n_epochs")
    p.add_argument("--resume",      type=str, default=None,
                   help="Experiment directory to resume training from")
    p.add_argument("--resume_from", choices=["best", "final", "latest"], default=None,
                   help="Which checkpoint to load when resuming (default: latest)")
    args = p.parse_args()
    if args.resume is None and args.config is None:
        p.error("--config is required when not using --resume")
    if args.resume_from is not None and args.resume is None:
        p.error("--resume_from requires --resume")
    return args


if __name__ == "__main__":
    args = parse_args()

    resume_checkpoint = None
    if args.resume is not None:
        resume_dir        = Path(args.resume)
        path              = resume_dir / "config.yaml"
        cfg               = load_config(path)
        resume_checkpoint = resume_dir / f"checkpoint_{args.resume_from or 'latest'}.pt"
    else:
        path = Path(args.config)
        cfg  = load_config(path)

    train(cfg, config_path=path, n_epochs=args.n_epochs,
          resume_checkpoint=resume_checkpoint)
