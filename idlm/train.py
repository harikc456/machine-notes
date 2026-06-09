# idlm/train.py
"""
Training entry point for I-DLM.

Usage:
    python -m idlm.train --config idlm/configs/baseline.yaml
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

from rbf_ffn.config import load_config as load_ar_config
from rbf_ffn.models.model import CausalLM
from idlm.config import IDLMConfig, load_config
from idlm.data import get_dataloaders
from idlm.models.idlm_model import IDLMCausalLM

MASK_TOKEN_ID = 50256


def get_experiment_dir(cfg: IDLMConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{stamp}_idlm_r{cfg.lora_rank}_s{cfg.stride}"
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


def _write_eval_row(metrics_path: Path, epoch: int, alpha_mean: float, tpf_oh_mean: float) -> None:
    row = {"type": "eval", "epoch": epoch, "alpha_mean": alpha_mean, "tpf_oh_mean": tpf_oh_mean}
    with open(metrics_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _maybe_save_mask_checkpoint(
    model: "IDLMCausalLM",
    l_mask: float,
    best_l_mask: float,
    step: int,
    epoch: int,
    exp_dir: Path,
) -> float:
    if l_mask < best_l_mask:
        raw = getattr(model, "_orig_mod", model)
        state = {k: v for k, v in raw.state_dict().items()
                 if "lora_A" in k or "lora_B" in k}
        torch.save({"lora_state": state, "step": step, "epoch": epoch, "l_mask": l_mask},
                   exp_dir / "checkpoint_best_mask.pt")
        return l_mask
    return best_l_mask


def compute_loss(
    model: IDLMCausalLM,
    x_0: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, float]:
    """Build [x_t | x_0] input, run forward, return (loss, l_mask, l_clean, lambda)."""
    B, L = x_0.shape
    x_t = torch.full_like(x_0, MASK_TOKEN_ID)
    # x_0 first so masked positions can attend to clean tokens via causal attention
    tokens = torch.cat([x_0, x_t], dim=1)                   # (B, 2L)

    use_lora_mask = torch.zeros(B, 2 * L, 1, device=device)
    use_lora_mask[:, L:, :] = 1.0                           # LoRA active on x_t half

    logits = model(tokens, use_lora_mask)                    # (B, 2L, vocab_size)

    # L_clean: standard AR shift on x_0 half (introspection pathway)
    l_clean = F.cross_entropy(
        logits[:, :L - 1].reshape(-1, logits.size(-1)),
        x_0[:, 1:].reshape(-1),
    )

    # L_mask: predict x_0[i] at each masked position i (decode pathway)
    l_mask = F.cross_entropy(
        logits[:, L:].reshape(-1, logits.size(-1)),
        x_0.reshape(-1),
    )

    # Auto-balanced coefficient: stop-gradient ratio
    lam = l_mask.detach() / (l_clean.detach() + 1e-8)
    loss = l_mask + lam * l_clean
    return loss, l_mask.item(), l_clean.item(), lam.item()


@torch.no_grad()
def run_isd_eval(
    model: IDLMCausalLM,
    test_loader,
    cfg: IDLMConfig,
    device: torch.device,
) -> dict:
    try:
        from idlm.generate import isd_acceptance_rate, compute_tpf_oh
    except ImportError:
        return {"alpha_mean": None, "tpf_oh_mean": None}

    model.eval()
    alphas = []
    count = 0
    for batch in test_loader:
        batch = batch.to(device)
        for i in range(batch.size(0)):
            if count >= cfg.num_eval_examples:
                break
            seq = batch[i].tolist()
            alpha = isd_acceptance_rate(model, seq, cfg, device)
            alphas.append(alpha)
            count += 1
        if count >= cfg.num_eval_examples:
            break
    alpha_mean = sum(alphas) / len(alphas) if alphas else 0.0
    tpf_oh = compute_tpf_oh(alpha_mean, cfg.stride)
    model.train()
    return {"alpha_mean": alpha_mean, "tpf_oh_mean": tpf_oh}


def _load_ar_model(cfg: IDLMConfig, device: torch.device) -> CausalLM:
    ckpt_path = Path(cfg.ar_checkpoint)
    config_yaml = ckpt_path.parent / "config.yaml"
    ar_cfg = load_ar_config(config_yaml)
    ar_model = CausalLM(ar_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ar_model.load_state_dict(ckpt["model"])
    return ar_model


def train(cfg: IDLMConfig, config_path: Path) -> Path:
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = get_experiment_dir(cfg)
    shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")

    train_loader, _, test_loader = get_dataloaders(cfg)

    ar_model = _load_ar_model(cfg, device)
    model = IDLMCausalLM(
        ar_model,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_target_modules=cfg.lora_target_modules,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")
    with open(exp_dir / "model_info.json", "w") as f:
        json.dump({"n_trainable_params": n_trainable}, f, indent=2)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    steps_per_epoch = len(train_loader)
    total_steps = cfg.max_epochs * steps_per_epoch

    optimizer = AdamW(lora_params, lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = LambdaLR(optimizer, make_lr_lambda(cfg.warmup_steps, total_steps))

    if device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    best_alpha = 0.0
    best_l_mask = float("inf")
    step = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.max_epochs}",
                    unit="step", dynamic_ncols=True, leave=True)

        epoch_l_mask_sum = 0.0
        epoch_steps = 0

        for batch in pbar:
            batch = batch.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, l_mask, l_clean, lam = compute_loss(model, batch, device)

            loss.backward()
            raw_model = getattr(model, "_orig_mod", model)
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            epoch_l_mask_sum += l_mask
            epoch_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", l_mask=f"{l_mask:.4f}", lam=f"{lam:.3f}")

            row: dict = {
                "type": "step",
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
                "l_mask": l_mask,
                "l_clean": l_clean,
                "lambda": lam,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")

        pbar.close()

        epoch_l_mask_avg = epoch_l_mask_sum / max(epoch_steps, 1)
        raw_model = getattr(model, "_orig_mod", model)
        best_l_mask = _maybe_save_mask_checkpoint(
            raw_model, epoch_l_mask_avg, best_l_mask, step, epoch, exp_dir
        )

        if epoch % cfg.eval_every_epochs == 0:
            eval_metrics = run_isd_eval(raw_model, test_loader, cfg, device)
            print(f"\n[epoch {epoch}] {eval_metrics}")
            _write_eval_row(metrics_path, epoch,
                            eval_metrics.get("alpha_mean") or 0.0,
                            eval_metrics.get("tpf_oh_mean") or 0.0)

            alpha = eval_metrics.get("alpha_mean") or 0.0
            if alpha > best_alpha:
                best_alpha = alpha
                lora_state = {
                    k: v for k, v in raw_model.state_dict().items()
                    if "lora_A" in k or "lora_B" in k
                }
                torch.save({"lora_state": lora_state, "step": step, "epoch": epoch,
                            "alpha": best_alpha},
                           exp_dir / "checkpoint_best.pt")
    raw_model = getattr(model, "_orig_mod", model)
    lora_state = {k: v for k, v in raw_model.state_dict().items()
                  if "lora_A" in k or "lora_B" in k}
    torch.save({"lora_state": lora_state, "step": step}, exp_dir / "checkpoint_final.pt")
    print(f"Done. Metrics -> {metrics_path}")
    return exp_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, Path(args.config))
