from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from idlm.config import IDLMConfig, load_config
from idlm.models.idlm_model import IDLMCausalLM
from rbf_ffn.config import load_config as load_ar_config, ModelConfig
from rbf_ffn.models.model import CausalLM


@dataclass
class RunInfo:
    dir_name: str
    run_dir: Path
    final_loss: float | None
    model_type: str = "idlm"  # "idlm" or "rbf"

    @property
    def label(self) -> str:
        short = self.dir_name[:15]  # "20260607_113459"
        loss_str = f"{self.final_loss:.4f}" if self.final_loss is not None else "N/A"
        return f"{short} | loss: {loss_str}"


def discover_runs(experiments_dir: Path) -> list[RunInfo]:
    """Scan experiments_dir for valid I-DLM runs, sorted newest-first."""
    runs: list[RunInfo] = []
    for run_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "checkpoint_best.pt").exists():
            continue
        if not (run_dir / "config.yaml").exists():
            continue
        final_loss = _read_final_loss(run_dir / "metrics.jsonl")
        runs.append(RunInfo(
            dir_name=run_dir.name,
            run_dir=run_dir,
            final_loss=final_loss,
            model_type="idlm",
        ))
    return runs


def _read_final_loss(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    last_step_loss = None
    try:
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("type") == "step" and "loss" in row:
                last_step_loss = float(row["loss"])
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return last_step_loss


def load_model(
    run_dir: Path,
    repo_root: Path,
    device: torch.device,
) -> tuple[IDLMCausalLM, IDLMConfig]:
    """Load AR base + LoRA checkpoint from a run directory."""
    cfg = load_config(run_dir / "config.yaml")

    # ar_checkpoint in config is relative to repo root
    ar_ckpt_path = repo_root / cfg.ar_checkpoint
    if not ar_ckpt_path.exists():
        raise FileNotFoundError(f"AR checkpoint not found: {ar_ckpt_path}")

    ar_config_yaml = ar_ckpt_path.parent / "config.yaml"
    if not ar_config_yaml.exists():
        raise FileNotFoundError(f"AR config not found: {ar_config_yaml}")
    ar_cfg = load_ar_config(ar_config_yaml)
    ar_model = CausalLM(ar_cfg).to(device)
    ar_ckpt = torch.load(ar_ckpt_path, map_location=device, weights_only=True)
    ar_model.load_state_dict(ar_ckpt["model"])

    model = IDLMCausalLM(ar_model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules)
    lora_ckpt_path = run_dir / "checkpoint_best.pt"
    lora_ckpt = torch.load(lora_ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(lora_ckpt["lora_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in LoRA checkpoint: {unexpected}")
    model.eval()
    return model, cfg


def discover_rbf_runs(experiments_dir: Path) -> list[RunInfo]:
    """Scan rbf_ffn experiments_dir for valid AR runs, sorted newest-first."""
    runs: list[RunInfo] = []
    for run_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "checkpoint_best.pt").exists():
            continue
        if not (run_dir / "config.yaml").exists():
            continue
        final_loss = _read_rbf_val_loss(run_dir / "metrics.jsonl")
        runs.append(RunInfo(
            dir_name=run_dir.name,
            run_dir=run_dir,
            final_loss=final_loss,
            model_type="rbf",
        ))
    return runs


def _read_rbf_val_loss(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    last_val_loss = None
    try:
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            if "val_loss" in row:
                last_val_loss = float(row["val_loss"])
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return last_val_loss


@torch.no_grad()
def ar_generate(
    model,
    prompt_ids: list[int],
    gen_len: int,
    device: torch.device,
    real_vocab_size: int = 50257,
    max_ctx: int = 512,
) -> list[int]:
    """Autoregressive generation, capping logits to real_vocab_size."""
    model.eval()
    ids = list(prompt_ids)
    for _ in range(gen_len):
        ctx = ids[-max_ctx:]
        tokens = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(tokens)
        next_logits = logits[0, -1, :real_vocab_size]
        probs = torch.softmax(next_logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
    return ids


def load_rbf_model(
    run_dir: Path,
    device: torch.device,
) -> tuple[CausalLM, ModelConfig]:
    """Load a plain rbf_ffn CausalLM checkpoint."""
    cfg = load_ar_config(run_dir / "config.yaml")
    ckpt_path = run_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = CausalLM(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
