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
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_quant import wrap, QuantConfig
from kv_quant.bench.perplexity import compute_perplexity
from kv_quant.bench.memory import measure_kv_memory


def _load_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _run_lmeval(model, tokenizer, tasks: list[str]) -> dict[str, float]:
    """Run lm-evaluation-harness tasks. Returns {task: accuracy_0_to_1}."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, verbosity="WARNING")

    scores: dict[str, float] = {}
    for task in tasks:
        task_res = results["results"].get(task, {})
        # lm-eval uses "acc,none" or "exact_match,none" keys
        for key in ("acc,none", "exact_match,none", "acc"):
            if key in task_res:
                scores[task] = float(task_res[key])
                break
        else:
            scores[task] = 0.0
    return scores


def run_benchmark(args) -> None:
    tasks: list[str] = args.tasks or []
    header = ["method", "bits", "ppl", "kv_mb"] + tasks
    rows: list[dict] = []

    # --- Baseline ---
    print(f"[baseline] Loading {args.model}…")
    model, tokenizer = _load_model(args.model)

    ppl  = compute_perplexity(model, tokenizer)
    mem  = measure_kv_memory(model, tokenizer)
    lm_scores = _run_lmeval(model, tokenizer, tasks) if tasks else {}

    rows.append({
        "method": "baseline", "bits": "fp16",
        "ppl": round(ppl, 3),
        "kv_mb": round(mem["peak_bytes"] / 1e6, 1),
        **{t: round(lm_scores.get(t, 0) * 100, 2) for t in tasks},
    })
    del model
    torch.cuda.empty_cache()

    # --- Quantized runs ---
    for method in args.method:
        cal_path = args.calibration if method == "spectralquant" else None
        for bits in args.bits:
            print(f"[{method} @ {bits}b] Loading {args.model}…")
            model, _ = _load_model(args.model)
            cfg = QuantConfig(method=method, bits=bits, calibration_path=cal_path)
            model = wrap(model, cfg)

            ppl  = compute_perplexity(model, tokenizer)
            mem  = measure_kv_memory(model, tokenizer)
            lm_scores = _run_lmeval(model, tokenizer, tasks) if tasks else {}

            rows.append({
                "method": method, "bits": bits,
                "ppl": round(ppl, 3),
                "kv_mb": round(mem["peak_bytes"] / 1e6, 1),
                **{t: round(lm_scores.get(t, 0) * 100, 2) for t in tasks},
            })
            del model
            torch.cuda.empty_cache()

    # --- Print table ---
    col_w = 14
    print("\n" + "".join(h.ljust(col_w) for h in header))
    print("-" * (col_w * len(header)))
    for row in rows:
        print("".join(str(row.get(h, "-")).ljust(col_w) for h in header))

    # --- Write CSV ---
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache quantization benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--method", nargs="+", default=["turboquant"],
                        choices=["turboquant", "spectralquant"])
    parser.add_argument("--bits",   nargs="+", type=int, default=[4])
    parser.add_argument("--tasks",  nargs="*", default=[],
                        help="lm-eval task names, e.g. mmlu arc_easy hellaswag gsm8k")
    parser.add_argument("--calibration", default=None,
                        help="Path to spectralquant .pt calibration file")
    parser.add_argument("--output",  default=None, help="CSV output path")
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
