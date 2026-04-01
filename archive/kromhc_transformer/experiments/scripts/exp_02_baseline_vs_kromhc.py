"""Phase 2: Baseline vs. KromHC comparison on WikiText-103."""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[3]))

from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.train import train
from kromhc_transformer.experiments.scripts.utils import set_seeds, log_experiment


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


BASE_CONFIG = dict(
    d_model=512,
    n_heads=16,
    n_layers=12,
    ffn_hidden=1376,
    batch_size=16,
    n_epochs=10,
    seq_len=512,
    qk_norm=True,
)

VARIANTS = {
    "baseline": dict(model_type="baseline", use_kromhc=False),
    "kromhc":   dict(model_type="kromhc",   use_kromhc=True),
}


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Baseline vs. KromHC")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(__file__).parents[1] / "results"
    hardware = {"device": str(device), "cuda": torch.cuda.is_available()}
    git_hash = get_git_hash()
    results = {}

    for variant_name, variant_kwargs in VARIANTS.items():
        set_seeds(args.seed)
        cfg = KromHCConfig(**BASE_CONFIG, **variant_kwargs, seed=args.seed)
        print(f"\n=== Training {variant_name} (seed={args.seed}) ===")

        try:
            metrics = train(cfg, device)
            results[variant_name] = metrics
            log_experiment(
                exp_id=f"phase2_{variant_name}",
                hypothesis=f"H1: {variant_name} trains correctly on WikiText-103.",
                config=cfg.__dict__,
                metrics={k: v for k, v in metrics.items() if k != "config"},
                hardware=hardware,
                seed=args.seed,
                git_hash=git_hash,
                status="completed",
                output_dir=output_dir,
            )
        except Exception as e:
            log_experiment(
                exp_id=f"phase2_{variant_name}",
                hypothesis=f"H1: {variant_name} trains correctly.",
                config=cfg.__dict__,
                metrics={},
                hardware=hardware,
                seed=args.seed,
                git_hash=git_hash,
                status="error",
                error_msg=str(e),
                output_dir=output_dir,
            )
            raise

    # Summary
    if "baseline" in results and "kromhc" in results:
        b_ppl = results["baseline"]["test_ppl"]
        k_ppl = results["kromhc"]["test_ppl"]
        margin = 0.01 * b_ppl
        diff = abs(k_ppl - b_ppl)
        verdict = "✓ PASS (H0 not rejected)" if diff <= margin else "~ INCONCLUSIVE"
        print(f"\n{'='*60}")
        print(f"PHASE 2 SUMMARY (seed={args.seed})")
        print(f"{'='*60}")
        print(f"Baseline test_ppl : {b_ppl:.2f}")
        print(f"KromHC   test_ppl : {k_ppl:.2f}")
        print(f"Diff={diff:.2f}  margin={margin:.2f}  {verdict}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
