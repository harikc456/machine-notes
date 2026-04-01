"""Phase 3: Ablation study — isolate head mixer contribution."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Ablations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Phase 3 (seed={args.seed}): To be implemented after Phase 2 results.")
    # TODO: Variant A — KromHC with use_kromhc=True
    # TODO: Variant B — same code path, use_kromhc=False (ablate the mixer)
    # TODO: Collect mixing matrix entropy and head variance diagnostics


if __name__ == "__main__":
    main()
