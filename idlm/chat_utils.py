from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


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
