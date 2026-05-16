#!/usr/bin/env python3
"""Count model parameters from checkpoints and cache results in each experiment dir."""

from pathlib import Path
import json
import torch

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
CACHE_FILE = "params.json"


def count_params(exp_dir: Path) -> int | None:
    for ckpt_name in ("checkpoint_best.pt", "checkpoint_final.pt", "checkpoint_latest.pt"):
        ckpt_path = exp_dir / ckpt_name
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                state = ckpt.get("model", ckpt)
                return sum(v.numel() for v in state.values())
            except Exception as e:
                print(f"  [warn] failed to load {ckpt_path.name}: {e}")
                return None
    return None


def main():
    dirs = sorted(d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "archive")

    updated = 0
    skipped = 0
    failed = 0

    for exp_dir in dirs:
        cache_path = exp_dir / CACHE_FILE
        if cache_path.exists():
            skipped += 1
            continue

        print(f"Counting {exp_dir.name} ...", end=" ", flush=True)
        n = count_params(exp_dir)
        if n is None:
            print("no checkpoint found")
            failed += 1
            continue

        cache_path.write_text(json.dumps({"n_params": n}))
        print(f"{n:,}")
        updated += 1

    print(f"\nDone: {updated} written, {skipped} already cached, {failed} failed.")


if __name__ == "__main__":
    main()
