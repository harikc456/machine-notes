from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from idlm.config import IDLMConfig, load_config
from idlm.models.idlm_model import IDLMCausalLM
from rbf_ffn.config import load_config as load_ar_config
from rbf_ffn.models.model import CausalLM


@dataclass
class RunInfo:
    dir_name: str
    run_dir: Path
    final_loss: float | None

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
    ar_cfg = load_ar_config(ar_config_yaml)
    ar_model = CausalLM(ar_cfg).to(device)
    ar_ckpt = torch.load(ar_ckpt_path, map_location=device, weights_only=True)
    ar_model.load_state_dict(ar_ckpt["model"])

    model = IDLMCausalLM(ar_model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules)
    lora_ckpt_path = run_dir / "checkpoint_best.pt"
    lora_ckpt = torch.load(lora_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(lora_ckpt["lora_state"], strict=False)
    model.eval()
    return model, cfg
