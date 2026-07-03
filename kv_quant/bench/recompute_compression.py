from __future__ import annotations
# kv_quant/bench/recompute_compression.py
"""Recompute the kv_compression column of a bench CSV from its kv_mb column.

kv_compression is meaningless unless it's computed against the *actual*
measured baseline (method=baseline,bits=fp16) kv_mb. run_bench.py can end up
writing a row's kv_compression before a real baseline exists in the file
(e.g. --no-baseline, or baseline run in a separate invocation) — in that case
it falls back to a per-row theoretical estimate that isn't comparable to the
baseline once one is measured. This recomputes every non-baseline row's
kv_compression as baseline_kv_mb / row_kv_mb, using the real measured numbers.

Usage:
    python -m kv_quant.bench.recompute_compression results/gemma4_2b.csv
"""
import argparse
import csv
from pathlib import Path


def recompute_compression_column(path: str) -> int:
    """Rewrite kv_compression in-place using the file's own baseline kv_mb.

    Returns the number of rows updated. No-op (returns 0) if the file has no
    baseline row or no numeric baseline kv_mb.
    """
    csv_path = Path(path)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "kv_compression" not in fieldnames:
        return 0

    baseline_kv_mb = None
    for row in rows:
        if row.get("method") == "baseline":
            try:
                baseline_kv_mb = float(row["kv_mb"])
            except (TypeError, ValueError):
                pass
            break

    if baseline_kv_mb is None or baseline_kv_mb <= 0:
        return 0

    updated = 0
    for row in rows:
        if row.get("method") == "baseline":
            row["kv_compression"] = 1.0
            continue
        try:
            kv_mb = float(row["kv_mb"])
        except (TypeError, ValueError):
            continue
        if kv_mb <= 0:
            continue
        row["kv_compression"] = round(baseline_kv_mb / kv_mb, 2)
        updated += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute kv_compression in a bench CSV from its measured kv_mb column"
    )
    parser.add_argument("csv", help="Path to the bench CSV to fix in place")
    args = parser.parse_args()

    updated = recompute_compression_column(args.csv)
    if updated:
        print(f"[recompute] Updated kv_compression for {updated} row(s) in {args.csv}")
    else:
        print(f"[recompute] Nothing to update in {args.csv} (no baseline row found)")


if __name__ == "__main__":
    main()
