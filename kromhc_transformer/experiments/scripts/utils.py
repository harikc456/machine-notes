"""Shared utilities for KromHC experiments."""
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def log_experiment(
    exp_id: str,
    hypothesis: str,
    config: dict,
    metrics: dict,
    hardware: dict,
    seed: int,
    git_hash: str = "",
    status: str = "completed",
    error_msg: str = None,
    output_dir: Path = None,
) -> Path:
    """Save structured experiment artifact as JSON."""
    artifact = {
        "experiment_id": f"{exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "hypothesis": hypothesis,
        "config": config,
        "metrics": metrics,
        "hardware": hardware,
        "seed": seed,
        "git_hash": git_hash,
        "status": status,
        "error_msg": error_msg,
    }
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{exp_id}_{seed}.json"
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    return path
