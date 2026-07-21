from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from latent_cot.config import ExperimentConfig, load_config
from latent_cot.train import train_and_eval

DEFAULT_CONDITIONS = ["floor", "ceiling", "z", "z_shuffled"]


def run_all(base_cfg: ExperimentConfig, conditions: list[str]) -> list[dict]:
    results = []
    for cond in conditions:
        cfg = copy.deepcopy(base_cfg)
        cfg.condition = cond
        cfg.__post_init__()
        print(f"\n=== Running condition: {cond} ===")
        results.append(train_and_eval(cfg))
    return results


def format_table(results: list[dict]) -> str:
    order = {c: i for i, c in enumerate(DEFAULT_CONDITIONS)}
    rows = sorted(results, key=lambda r: order.get(r["condition"], 99))
    lines = [
        f"{'condition':<12} {'eval_acc':>9} {'n_eval':>7} {'train_loss':>11}",
        "-" * 42,
    ]
    for r in rows:
        lines.append(
            f"{r['condition']:<12} {r['eval_accuracy']:>9.3f} "
            f"{r['n_eval']:>7d} {r['final_train_loss']:>11.4f}"
        )
    lines.append("")
    lines.append("Read: z >> floor and z ~ ceiling  => reasoning compresses into z.")
    lines.append("      z ~ z_shuffled              => z is NOT using reasoning (illusion).")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    base_cfg = load_config(args.config) if args.config else ExperimentConfig()
    results = run_all(base_cfg, DEFAULT_CONDITIONS)
    table = format_table(results)
    print("\n" + table)
    out = Path(base_cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "killtest_results.json").write_text(json.dumps(results, indent=2))
    (out / "killtest_table.txt").write_text(table)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
