#!/usr/bin/env python3
"""Model leaderboard from experiment artifacts in rbf_ffn/experiments/."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from rbf_ffn._leaderboard_data import load_all_experiments, fmt, fmt_params

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def main():
    parser = argparse.ArgumentParser(description="Leaderboard of rbf_ffn experiments")
    parser.add_argument("--sort", default="best_val_ppl", choices=["best_val_ppl", "final_val_ppl", "name"])
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--min-epochs", type=int, default=1)
    args = parser.parse_args()

    exps = load_all_experiments(EXPERIMENTS_DIR, min_epochs=args.min_epochs)

    if args.filter:
        exps = [e for e in exps if args.filter.lower() in e["name"].lower()]

    if args.sort in ("best_val_ppl", "final_val_ppl"):
        exps.sort(key=lambda e: e[args.sort] or float("inf"))
    else:
        exps.sort(key=lambda e: e[args.sort] or "")

    if args.top:
        exps = exps[: args.top]

    console = Console(width=200)
    table = Table(
        title=f"[bold]rbf_ffn Leaderboard[/bold]  ({len(exps)} experiments, sorted by {args.sort})",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        highlight=True,
    )

    table.add_column("#", style="dim", justify="right", width=3)
    table.add_column("Experiment", style="cyan", no_wrap=True, max_width=45)
    table.add_column("attn", justify="center", width=9)
    table.add_column("ffn", justify="center", width=12)
    table.add_column("qk", justify="center", width=3)
    table.add_column("wn", justify="center", width=3)
    table.add_column("orth_layers", justify="center", width=14)
    table.add_column("MoE", justify="center", width=6)
    table.add_column("params", justify="right", width=8)
    table.add_column("ep", justify="right", width=4)
    table.add_column("best_ppl", justify="right", style="green bold", width=9)
    table.add_column("@ep", justify="right", style="dim", width=4)
    table.add_column("final_ppl", justify="right", width=10)
    table.add_column("trn_ppl", justify="right", style="dim", width=9)
    table.add_column("hrs", justify="right", style="dim", width=5)

    for rank, e in enumerate(exps, 1):
        parts = e["name"].split("_")
        short = "_".join(parts[3:]) if len(parts) > 3 else e["name"]

        n_exp = e.get("moe_n_experts")
        top_k = e.get("moe_top_k")
        moe_str = f"{n_exp}/{top_k}" if n_exp is not None else "—"

        table.add_row(
            str(rank),
            short,
            fmt(e.get("attn_type")),
            fmt(e.get("ffn_type")),
            fmt(e.get("qk_norm")),
            fmt(e.get("linear_weight_norm")),
            fmt(e.get("orthogonal_ffn_layers")),
            moe_str,
            fmt_params(e.get("n_params")),
            fmt(e.get("epochs_done"), 0),
            fmt(e.get("best_val_ppl")),
            fmt(e.get("best_epoch"), 0),
            fmt(e.get("final_val_ppl")),
            fmt(e.get("final_train_ppl")),
            fmt(e.get("total_time_h")),
        )

    console.print(table)
    console.print(f"[dim]Experiments dir: {EXPERIMENTS_DIR}[/dim]")


if __name__ == "__main__":
    main()
