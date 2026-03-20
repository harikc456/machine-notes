"""Phase 1: POC — end-to-end training validation."""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).parents[3]))

from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.train import train
from kromhc_transformer.experiments.scripts.utils import set_seeds, log_experiment


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Phase 1: POC validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = KromHCConfig(
        model_type="kromhc",
        use_kromhc=True,
        qk_norm=True,
        d_model=256,
        n_heads=8,
        n_layers=6,
        ffn_hidden=688,
        batch_size=32,
        n_epochs=1,
        seq_len=512,
        seed=args.seed,
    )

    output_dir = Path(__file__).parents[1] / "results"
    status = "completed"
    error_msg = None
    metrics = {}

    try:
        metrics = train(cfg, device)
        print(f"\n✓ POC PASSED — test_ppl={metrics['test_ppl']:.2f}")
    except Exception as e:
        status = "error"
        error_msg = str(e)
        print(f"\n✗ POC FAILED: {e}")
        raise

    log_experiment(
        exp_id="poc_kromhc",
        hypothesis="H1: KromHC trains end-to-end without errors; loss decreases over 1 epoch.",
        config=cfg.__dict__,
        metrics={k: v for k, v in metrics.items() if k != "config"},
        hardware={"device": str(device), "cuda": torch.cuda.is_available()},
        seed=args.seed,
        git_hash=get_git_hash(),
        status=status,
        error_msg=error_msg,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
