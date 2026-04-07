"""
Training entry point for SIGReg transformer experiments.

The model is a plain transformer (no residual connections, no RMSNorm).
At each training step the auxiliary SIGReg loss is computed over the hidden
states collected from the configured layers and added to the CE loss:

    total_loss = ce_loss + cfg.sigreg_weight * sigreg_loss

Usage:
    python -m sigreg.train --config sigreg/configs/baseline.yaml
    python -m sigreg.train --config sigreg/configs/baseline.yaml --n_epochs 5
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
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from sigreg.config import SIGRegConfig, load_config
from sigreg.losses import sigreg_loss as compute_sigreg_loss
from sigreg.models.model import SIGRegCausalLM, build_optimizer_groups

# Reuse the WikiText-103 data pipeline from rbf_ffn.
# get_dataloaders only reads cfg.seq_len, cfg.batch_size, cfg.seed — SIGRegConfig
# has all three so it can be passed directly without any adapter.
from sigreg.data import get_dataloaders


def get_experiment_dir(cfg: SIGRegConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    loss_tag = f"sigreg_{cfg.sigreg_loss_type}_w{cfg.sigreg_weight}"
    name = f"{stamp}_plain_{loss_tag}_d{cfg.d_model}"
    path = Path(__file__).parent / "experiments" / name
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
def evaluate(
    model: SIGRegCausalLM,
    loader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (val_loss, val_ppl) in nats/token (CE only, no SIGReg)."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        n_tokens = inputs.numel()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits, _ = model(inputs, collect_hidden=False)
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
    cfg: SIGRegConfig,
    config_path: Path,
    n_epochs: int | None = None,
) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if n_epochs is not None:
        cfg.n_epochs = n_epochs

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
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
    optimizer_steps = total_steps // cfg.grad_accum_steps
    warmup_steps    = int(cfg.warmup_ratio * optimizer_steps)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SIGRegCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"SIGReg: type={cfg.sigreg_loss_type}, weight={cfg.sigreg_weight}, "
          f"sketch_dim={cfg.sigreg_sketch_dim}, layers={cfg.sigreg_layers}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    muon_params, adamw_params = build_optimizer_groups(model)
    muon  = Muon( muon_params,  lr=cfg.lr * 6.67, momentum=0.95)  # Muon LR ~6.67× AdamW
    adamw = AdamW(adamw_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    lr_fn = make_lr_lambda(warmup_steps, optimizer_steps)
    sched_muon  = LambdaLR(muon,  lr_fn)
    sched_adamw = LambdaLR(adamw, lr_fn)

    # ── Compile ───────────────────────────────────────────────────────────────
    if device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    best_val_loss = float("inf")
    val_loss, val_ppl = float("inf"), float("inf")
    pbar = tqdm(total=cfg.n_epochs * steps_per_epoch, desc="training", unit="step", dynamic_ncols=True)

    def save_checkpoint(name: str, epoch: int, val_loss: float, val_ppl: float) -> None:
        raw_model = getattr(model, "_orig_mod", model)
        torch.save({
            "model":           raw_model.state_dict(),
            "optimizer_muon":  muon.state_dict(),
            "optimizer_adamw": adamw.state_dict(),
            "scheduler_muon":  sched_muon.state_dict(),
            "scheduler_adamw": sched_adamw.state_dict(),
            "epoch":           epoch,
            "val_loss":        val_loss,
            "val_ppl":         val_ppl,
            "best_val_loss":   best_val_loss,
        }, exp_dir / name)

    for epoch in range(cfg.n_epochs):
        model.train()
        loss_sum, token_count = 0.0, 0
        sigreg_sum = 0.0
        t0 = time.time()

        global_step = 0
        for batch in train_loader:
            batch   = batch.to(device)
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]
            n_tokens = inputs.numel()

            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, hidden_states = model(inputs, collect_hidden=True)

                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean",
                )

                # Auxiliary SIGReg loss: mean over collected layers
                if hidden_states and cfg.sigreg_weight > 0.0:
                    aux = torch.stack([
                        compute_sigreg_loss(h, cfg.sigreg_loss_type, cfg.sigreg_sketch_dim)
                        for h in hidden_states
                    ]).mean()
                    total_loss = ce_loss + cfg.sigreg_weight * aux
                else:
                    aux = torch.tensor(0.0, device=device)
                    total_loss = ce_loss

            raw_ce  = ce_loss.item()
            raw_aux = aux.item()

            total_loss = total_loss / cfg.grad_accum_steps
            total_loss.backward()

            global_step += 1
            if global_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                muon.step();  adamw.step()
                sched_muon.step(); sched_adamw.step()
                muon.zero_grad(set_to_none=True)
                adamw.zero_grad(set_to_none=True)

            loss_sum    += raw_ce * n_tokens
            token_count += n_tokens
            sigreg_sum  += raw_aux

            pbar.update(1)
            pbar.set_postfix(
                epoch=f"{epoch+1}/{cfg.n_epochs}",
                ce=f"{raw_ce:.4f}",
                aux=f"{raw_aux:.4f}",
            )

        # Flush remaining accumulated gradients
        if global_step % cfg.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            muon.step();  adamw.step()
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)

        train_loss = loss_sum / token_count
        train_ppl  = math.exp(train_loss)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        model.train()

        epoch_time = time.time() - t0
        mean_aux   = sigreg_sum / steps_per_epoch

        pbar.set_postfix(
            epoch=f"{epoch+1}/{cfg.n_epochs}",
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
        )

        row: dict = {
            "epoch":             epoch,
            "train_ce_loss":     train_loss,
            "train_ppl":         train_ppl,
            "val_loss":          val_loss,
            "val_ppl":           val_ppl,
            "sigreg_loss_mean":  mean_aux,
            "epoch_time_s":      epoch_time,
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
    p = argparse.ArgumentParser(description="Train SIGReg transformer on WikiText-103")
    p.add_argument("--config",   required=True, help="Path to YAML config")
    p.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs")
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    path  = Path(args.config)
    cfg   = load_config(path)
    train(cfg, config_path=path, n_epochs=args.n_epochs)
