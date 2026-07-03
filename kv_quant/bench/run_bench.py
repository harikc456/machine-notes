from __future__ import annotations
# kv_quant/bench/run_bench.py
"""Benchmark CLI for KV cache quantization.

Usage:
    python -m kv_quant.bench.run_bench \
        --model Qwen/Qwen2.5-7B-Instruct \
        --method turboquant spectralquant \
        --bits 2 3 4 \
        --tasks mmlu arc_easy hellaswag gsm8k \
        --calibration spectralquant_qwen25_7b.pt \
        --output results/qwen25_7b.csv
"""
import argparse
import csv
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_quant import wrap, QuantConfig
from kv_quant.bench.perplexity import compute_perplexity
from kv_quant.bench.memory import measure_kv_memory
from kv_quant.bench.recompute_compression import recompute_compression_column


def _load_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _run_lmeval(model, tokenizer, tasks: list[str], batch_size: str | int = "auto:4", num_fewshot: int = 0, offline: bool = False) -> dict[str, float]:
    """Run lm-evaluation-harness tasks. Returns {task: accuracy_0_to_1}."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness is required for --tasks. "
            "Install it with: pip install lm-eval"
        ) from None

    import os
    if offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # apply_chat_template helps generation tasks (gsm8k) but hurts
    # loglikelihood MC tasks (mmlu, arc, hellaswag) on IT models at low fewshot.
    # Only enable if the task list is exclusively generation-based.
    _MC_TASKS = {"mmlu", "arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"}
    apply_chat = (
        tokenizer.chat_template is not None
        and not any(t in _MC_TASKS for t in tasks)
    )
    # Use the model's actual max position embeddings rather than the tokenizer's
    # legacy model_max_length (often stuck at 2048) to avoid spurious truncation.
    max_length = (
        getattr(model.config, "max_position_embeddings", None)
        or getattr(model.config, "n_positions", None)
        or getattr(model.config, "seq_length", None)
        or getattr(model.config, "max_sequence_length", None)
    )
    if max_length is None:
        tok_max = getattr(tokenizer, "model_max_length", None)
        max_length = tok_max if tok_max and tok_max > 4096 else 32768
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
               apply_chat_template=apply_chat, max_length=max_length)
    results = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=num_fewshot,
        cache_requests=True, verbosity="WARNING",
    )

    # Metric key priority: lm-eval varies by task version and release.
    # GSM8k → exact_match,flexible-extract; MC tasks → acc,none.
    _METRIC_KEYS = (
        "acc,none",
        "exact_match,flexible-extract",
        "exact_match,none",
        "exact_match,strict-match",
        "acc",
        "exact_match",
    )
    scores: dict[str, float] = {}
    for task in tasks:
        task_res = results["results"].get(task, {})
        for key in _METRIC_KEYS:
            if key in task_res:
                scores[task] = float(task_res[key])
                break
        else:
            # Last resort: first numeric value that isn't stderr
            fallback = next(
                (v for k, v in task_res.items()
                 if isinstance(v, float) and "stderr" not in k),
                None,
            )
            scores[task] = fallback if fallback is not None else 0.0
            if fallback is None:
                print(f"[warn] No metric found for task '{task}'. Keys: {list(task_res)}")
    return scores


def _print_row(row: dict, header: list[str], col_w: int = 14) -> None:
    print("".join(str(row.get(h, "-")).ljust(col_w) for h in header))


def _load_completed(output: str, header: list[str]) -> set[tuple]:
    """Return set of (method, bits) already present in the CSV output file."""
    path = Path(output)
    if not path.exists():
        return set()
    completed = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row.get("method", ""), row.get("bits", "")))
    return completed


def _load_baseline_peak_bytes(output: str) -> int | None:
    """Recover the fp16 baseline's peak_bytes from an existing CSV (for --resume runs).

    Needed so quantized rows compute kv_compression against the same reference
    used originally, rather than falling back to each row's own fp16_est_bytes.
    """
    path = Path(output)
    if not path.exists():
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == "baseline" and row.get("kv_mb"):
                return round(float(row["kv_mb"]) * 1e6)
    return None


def _append_row(output: str, row: dict, header: list[str]) -> None:
    path = Path(output)
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_benchmark(args) -> None:
    tasks: list[str] = args.tasks or []
    skip_ppl: bool = args.no_ppl
    ppl_tokens: int = args.ppl_tokens
    header = ["method", "bits"] + ([] if skip_ppl else ["ppl"]) + ["kv_mb", "kv_compression"] + tasks
    rows: list[dict] = []

    col_w = 14
    print("\n" + "".join(h.ljust(col_w) for h in header))
    print("-" * (col_w * len(header)))

    completed = _load_completed(args.output, header) if args.output else set()
    if completed:
        print(f"[resume] Skipping {len(completed)} already-completed config(s): {sorted(completed)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    baseline_peak_bytes: int | None = None

    # --- Baseline ---
    if not args.no_baseline:
        key = ("baseline", "fp16")
        if key in completed:
            print("[baseline] Already done — skipping.")
            baseline_peak_bytes = _load_baseline_peak_bytes(args.output)
        else:
            print(f"[baseline] Loading {args.model}…")
            model, _ = _load_model(args.model)

            ppl  = None if skip_ppl else compute_perplexity(model, tokenizer, n_tokens=ppl_tokens)
            mem  = measure_kv_memory(model, tokenizer)
            baseline_peak_bytes = mem["peak_bytes"]
            lm_scores = _run_lmeval(model, tokenizer, tasks, args.lmeval_batch_size, args.num_fewshot, args.offline) if tasks else {}

            row = {
                "method": "baseline", "bits": "fp16",
                **({} if skip_ppl else {"ppl": round(ppl, 3)}),
                "kv_mb": round(mem["peak_bytes"] / 1e6, 1),
                "kv_compression": 1.0,
                **{t: round(lm_scores.get(t, 0) * 100, 2) for t in tasks},
            }
            rows.append(row)
            _print_row(row, header, col_w)
            if args.output:
                _append_row(args.output, row, header)
                # A real baseline now exists — fix any quantized rows that were
                # written earlier (e.g. a prior --no-baseline run) against a
                # theoretical estimate instead of this measured reference.
                recompute_compression_column(args.output)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # --- Quantized runs ---
    if args.baseline_only:
        return

    for method in args.method:
        cal_path = args.calibration if method == "spectralquant" else None
        for bits in args.bits:
            key = (method, str(bits))
            if key in completed:
                print(f"[{method} @ {bits}b] Already done — skipping.")
                continue

            print(f"[{method} @ {bits}b] Loading {args.model}…")
            model, _ = _load_model(args.model)
            cfg = QuantConfig(method=method, bits=bits, calibration_path=cal_path)
            model = wrap(model, cfg)

            ppl  = None if skip_ppl else compute_perplexity(model, tokenizer, n_tokens=ppl_tokens)
            mem  = measure_kv_memory(model, tokenizer)
            lm_scores = _run_lmeval(model, tokenizer, tasks, args.lmeval_batch_size, args.num_fewshot, args.offline) if tasks else {}

            ref_bytes = baseline_peak_bytes if baseline_peak_bytes else mem["fp16_est_bytes"]
            compression = round(ref_bytes / mem["peak_bytes"], 2) if mem["peak_bytes"] > 1 else "-"

            row = {
                "method": method, "bits": bits,
                **({} if skip_ppl else {"ppl": round(ppl, 3)}),
                "kv_mb": round(mem["peak_bytes"] / 1e6, 1),
                "kv_compression": compression,
                **{t: round(lm_scores.get(t, 0) * 100, 2) for t in tasks},
            }
            rows.append(row)
            _print_row(row, header, col_w)
            if args.output:
                _append_row(args.output, row, header)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if args.output:
        recompute_compression_column(args.output)
        # Reload so the printed table reflects any kv_compression fix-ups above.
        with open(args.output, newline="") as f:
            rows = list(csv.DictReader(f))

    # --- Final table ---
    print("\n" + "".join(h.ljust(col_w) for h in header))
    print("-" * (col_w * len(header)))
    for row in rows:
        _print_row(row, header, col_w)


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache quantization benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--method", nargs="+", default=["turboquant"],
                        choices=["turboquant", "spectralquant"])
    parser.add_argument("--bits",   nargs="+", type=int, default=[4])
    parser.add_argument("--tasks",  nargs="*", default=[],
                        help="lm-eval task names (requires pip install lm-eval), e.g. mmlu arc_easy hellaswag gsm8k")
    parser.add_argument("--calibration", default=None,
                        help="Path to spectralquant .pt calibration file")
    parser.add_argument("--lmeval-batch-size", default="auto:4",
                        help="lm-eval batch size. 'auto:N' lets lm-eval find the max up to N "
                             "(default: auto:4). Set to 1 if you hit OOM.")
    parser.add_argument("--num-fewshot", type=int, default=3,
                        help="Number of few-shot examples for lm-eval tasks (default: 5)")
    parser.add_argument("--ppl-tokens",  type=int, default=4096,
                        help="Number of tokens for perplexity eval (default: 4096)")
    parser.add_argument("--no-ppl",      action="store_true", help="Skip perplexity computation")
    parser.add_argument("--no-baseline",    action="store_true", help="Skip the fp16 baseline run")
    parser.add_argument("--baseline-only", action="store_true", help="Run only the fp16 baseline, skip quantized runs")
    parser.add_argument("--output",  default=None, help="CSV output path")
    parser.add_argument("--offline", action="store_true",
                        help="Set HF_DATASETS_OFFLINE=1 to use cached datasets without Hub checks")
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
