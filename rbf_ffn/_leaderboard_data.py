#!/usr/bin/env python3
"""Shared data loading for leaderboard scripts."""

import ast
import json
from pathlib import Path

import yaml

CONFIG_FIELDS = [
    "attn_type",
    "ffn_type",
    "qk_norm",
    "linear_weight_norm",
    "orthogonal_ffn_layers",
    "moe_n_experts",
    "moe_top_k",
    "n_layers",
    "d_model",
]

_SKIP_DIRS = {"archive"}


def load_experiment(exp_dir: Path) -> dict | None:
    metrics_path = exp_dir / "metrics.jsonl"
    config_path = exp_dir / "config.yaml"

    if not metrics_path.exists() or not config_path.exists():
        return None

    rows = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append(ast.literal_eval(line))
    if not rows:
        return None

    config_text = config_path.read_text()
    config = yaml.safe_load(config_text) or {}

    name_parts = set(exp_dir.name.split("_"))
    if not config.get("attn_type"):
        if "xsa" in name_parts:
            config["attn_type"] = "xsa"
        elif "standard" in name_parts:
            config["attn_type"] = "std"
    if not config.get("ffn_type"):
        for candidate in ("swiglu", "moe", "rational", "rationalglu", "polar"):
            if candidate in name_parts:
                config["ffn_type"] = candidate
                break

    best_row = min(rows, key=lambda r: r.get("val_ppl", float("inf")))
    final_row = rows[-1]
    total_time_h = sum(r.get("epoch_time_s", 0) for r in rows) / 3600

    n_params = None
    model_info_path = exp_dir / "model_info.json"
    if model_info_path.exists():
        n_params = json.loads(model_info_path.read_text()).get("n_params")

    return {
        "name": exp_dir.name,
        "config_text": config_text,
        "epochs_done": len(rows),
        "best_val_ppl": best_row.get("val_ppl"),
        "best_epoch": best_row.get("epoch"),
        "final_val_ppl": final_row.get("val_ppl"),
        "final_train_ppl": final_row.get("train_ppl"),
        "total_time_h": total_time_h,
        "n_params": n_params,
        **{f: config.get(f) for f in CONFIG_FIELDS},
    }


def load_all_experiments(experiments_dir: Path, min_epochs: int = 1) -> list[dict]:
    results = []
    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in _SKIP_DIRS:
            continue
        data = load_experiment(d)
        if data and data["epochs_done"] >= min_epochs:
            results.append(data)
    return results


def fmt(val, digits: int = 2) -> str:
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Y" if val else "N"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    if isinstance(val, list):
        return str(val)
    return str(val)


def fmt_params(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)
