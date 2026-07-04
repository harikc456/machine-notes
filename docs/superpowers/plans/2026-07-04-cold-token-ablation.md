# Cold-Token Ablation Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a script that, per WikiText-2 passage and per layer, removes that layer's bottom-10%-attention prompt tokens and checks whether greedy regeneration still produces the same 30-token continuation as the full-context baseline.

**Architecture:** A small pure function for pruning + comparison (unit tested without any model/GPU), plus a thin orchestration script that reuses `accumulate_attention_scores`, `select_cold_tokens`, `chunk_token_ids`, `load_wikitext_token_ids` from `kv_quant/bench/pos_attention_correlation.py` and adds the baseline/ablation generation loop, CSV output, and findings write-up. Per passage: 1 baseline `generate()` call (all layers' attention in one shot, as in the POS-attention experiment) + 1 batched `generate()` call covering all layers' pruned variants at once (since `select_cold_tokens` removes a fixed token count regardless of layer, every layer's pruned prompt for a passage has identical length — no padding needed to batch them). This cuts the per-passage ablation cost from `num_layers` sequential calls to 1.

**Tech Stack:** PyTorch, HuggingFace `transformers` + `datasets`, pytest.

## Global Constraints

- Model: `google/gemma-4-E2B-it`, loaded with `attn_implementation="eager"` for the baseline run (SDPA/flash backends return `None` attentions). Ablation regeneration runs don't need attention output, so `attn_implementation` doesn't matter for them, but reuse the same loaded model.
- 25 passages from WikiText-2 test split (`wikitext-103-raw-v1`), 200 tokens each — same `chunk_token_ids`/`load_wikitext_token_ids` as the POS-attention experiment.
- `max_new_tokens=30`, greedy (`do_sample=False`) for both baseline and ablation runs.
- Cold set per layer = bottom 10% of the **first 200 positions only** (the prompt) by accumulated received-attention, via the existing `select_cold_tokens`.
- Pruning = splice cold positions out of `input_ids` (shorten the sequence), not attention masking.
- Ablation regenerations are batched: all `num_layers` pruned prompts for a passage are stacked into one `generate()` call (they're guaranteed equal length since `select_cold_tokens` removes a fixed count per call), instead of one `generate()` call per layer.
- Comparison = exact match of all 30 generated tokens, plus first-divergence index (30 if exact match).
- Outputs: `kv_quant/bench/results_ablation/cold_ablation.csv`, `kv_quant/bench/results_ablation/cold_ablation_summary.csv`, `kv_quant/bench/findings_cold_ablation.md`.
- New file: `kv_quant/bench/cold_token_ablation.py`, standalone, not wired into `run_bench.py`.
- Tests requiring the real model/GPU are marked `@pytest.mark.slow` (repo convention, see `pyproject.toml` markers).

---

### Task 1: Prompt pruning and continuation comparison

**Files:**
- Create: `kv_quant/bench/cold_token_ablation.py`
- Test: `tests/test_cold_token_ablation.py`

**Interfaces:**
- Produces: `prune_prompt(prompt_ids: list[int], cold_indices: list[int]) -> list[int]` — returns `prompt_ids` with the positions in `cold_indices` removed, order preserved.
- Produces: `compare_continuations(baseline: list[int], pruned: list[int]) -> tuple[bool, int]` — returns `(exact_match, first_divergence_idx)`. `first_divergence_idx` is the index of the first differing element; if all elements up to `min(len(baseline), len(pruned))` match and both are the same length, it's `len(baseline)` and `exact_match=True`.

- [ ] **Step 1: Write failing tests for `prune_prompt`**

```python
# tests/test_cold_token_ablation.py
from __future__ import annotations

from kv_quant.bench.cold_token_ablation import prune_prompt


def test_prune_prompt_removes_given_indices():
    prompt_ids = [10, 11, 12, 13, 14]
    pruned = prune_prompt(prompt_ids, cold_indices=[1, 3])
    assert pruned == [10, 12, 14]


def test_prune_prompt_empty_cold_set():
    prompt_ids = [10, 11, 12]
    pruned = prune_prompt(prompt_ids, cold_indices=[])
    assert pruned == [10, 11, 12]


def test_prune_prompt_unordered_indices():
    prompt_ids = [10, 11, 12, 13, 14]
    pruned = prune_prompt(prompt_ids, cold_indices=[3, 0])
    assert pruned == [11, 12, 14]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cold_token_ablation.py -v -k prune_prompt`
Expected: FAIL with `ImportError: cannot import name 'prune_prompt'` (module doesn't exist yet).

- [ ] **Step 3: Implement `prune_prompt`**

```python
# kv_quant/bench/cold_token_ablation.py
from __future__ import annotations


def prune_prompt(prompt_ids: list[int], cold_indices: list[int]) -> list[int]:
    """Return prompt_ids with the positions in cold_indices removed."""
    cold_set = set(cold_indices)
    return [tok for i, tok in enumerate(prompt_ids) if i not in cold_set]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cold_token_ablation.py -v -k prune_prompt`
Expected: PASS (3 passed)

- [ ] **Step 5: Write failing tests for `compare_continuations`**

```python
from kv_quant.bench.cold_token_ablation import compare_continuations


def test_compare_continuations_exact_match():
    baseline = [1, 2, 3, 4]
    pruned = [1, 2, 3, 4]
    exact_match, first_div = compare_continuations(baseline, pruned)
    assert exact_match is True
    assert first_div == 4


def test_compare_continuations_diverges_partway():
    baseline = [1, 2, 3, 4]
    pruned = [1, 2, 9, 9]
    exact_match, first_div = compare_continuations(baseline, pruned)
    assert exact_match is False
    assert first_div == 2


def test_compare_continuations_diverges_immediately():
    baseline = [1, 2, 3]
    pruned = [9, 2, 3]
    exact_match, first_div = compare_continuations(baseline, pruned)
    assert exact_match is False
    assert first_div == 0
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_cold_token_ablation.py -v -k compare_continuations`
Expected: FAIL with `ImportError: cannot import name 'compare_continuations'`

- [ ] **Step 7: Implement `compare_continuations`**

```python
def compare_continuations(baseline: list[int], pruned: list[int]) -> tuple[bool, int]:
    """Compare two equal-length greedy continuations token-by-token."""
    first_div = len(baseline)
    for i, (b, p) in enumerate(zip(baseline, pruned)):
        if b != p:
            first_div = i
            break
    exact_match = first_div == len(baseline)
    return exact_match, first_div
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `pytest tests/test_cold_token_ablation.py -v`
Expected: PASS (6 passed)

- [ ] **Step 9: Commit**

```bash
git add kv_quant/bench/cold_token_ablation.py tests/test_cold_token_ablation.py
git commit -m "feat(kv_quant): add prompt pruning and continuation comparison"
```

---

### Task 2: Orchestration script (baseline + batched per-layer ablation, CSV output)

**Files:**
- Modify: `kv_quant/bench/cold_token_ablation.py`
- Test: `tests/test_cold_token_ablation.py` (marked `@pytest.mark.slow`)

**Interfaces:**
- Consumes: `prune_prompt`, `compare_continuations` (Task 1); `accumulate_attention_scores`, `select_cold_tokens`, `chunk_token_ids`, `load_wikitext_token_ids` (imported from `kv_quant.bench.pos_attention_correlation`).
- Produces: `run_ablation_experiment(model, tokenizer, passages: list[list[int]], max_new_tokens: int = 30, cold_frac: float = 0.1) -> list[dict]` — returns per-`(passage_id, layer)` records: `{"passage_id": int, "layer": int, "num_removed": int, "exact_match": bool, "first_divergence_idx": int}`.
- Produces: `write_outputs(records: list[dict], results_dir: str) -> None`.
- Produces: `main() -> None`.

- [ ] **Step 1: Write a failing slow test for `run_ablation_experiment` on a tiny real model**

```python
import pytest


@pytest.mark.slow
def test_run_ablation_experiment_end_to_end_tiny_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kv_quant.bench.cold_token_ablation import run_ablation_experiment

    model_id = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager"
    ).eval()

    text = "The quick brown fox jumps over the lazy dog and then runs away fast."
    prompt_ids = tokenizer(text)["input_ids"][:20]
    passages = [prompt_ids]

    records = run_ablation_experiment(model, tokenizer, passages, max_new_tokens=3)

    n_layers = model.config.num_hidden_layers
    assert len(records) == n_layers  # one row per (passage, layer), 1 passage here
    for r in records:
        assert set(r.keys()) == {
            "passage_id", "layer", "num_removed", "exact_match", "first_divergence_idx",
        }
        assert r["num_removed"] >= 1
        assert 0 <= r["first_divergence_idx"] <= 3
    assert {r["layer"] for r in records} == set(range(n_layers))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cold_token_ablation.py -v -k run_ablation_experiment --run-slow`
Expected: FAIL with `ImportError: cannot import name 'run_ablation_experiment'`

- [ ] **Step 3: Implement `run_ablation_experiment`**

```python
def run_ablation_experiment(
    model,
    tokenizer,
    passages: list[list[int]],
    max_new_tokens: int = 30,
    cold_frac: float = 0.1,
) -> list[dict]:
    """For each passage, run a full-context baseline, then batch all layers'
    pruned (cold-token-removed) prompts into one generate() call and compare
    each layer's regenerated continuation to the baseline."""
    import torch

    from kv_quant.bench.pos_attention_correlation import (
        accumulate_attention_scores,
        select_cold_tokens,
    )

    device = next(model.parameters()).device
    num_layers = model.config.get_text_config().num_hidden_layers
    records: list[dict] = []

    for passage_id, prompt_ids in enumerate(passages):
        prompt_len = len(prompt_ids)
        input_ids = torch.tensor([prompt_ids], device=device)
        with torch.inference_mode():
            baseline_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        if baseline_output.attentions is None or baseline_output.attentions[0][0] is None:
            raise RuntimeError(
                "model.generate() returned no attentions. Ensure the model "
                "was loaded with attn_implementation='eager'."
            )

        full_ids = baseline_output.sequences[0].tolist()
        total_len = len(full_ids)
        baseline_continuation = full_ids[prompt_len:]

        scores = accumulate_attention_scores(
            list(baseline_output.attentions), total_len=total_len, num_layers=num_layers
        )

        # select_cold_tokens removes floor(prompt_len * cold_frac) tokens
        # regardless of which layer's scores it's given (same prompt_len,
        # same cold_frac every time) -> every layer's pruned prompt for this
        # passage has the same length. That lets us stack all num_layers
        # pruned variants into one batched generate() call instead of
        # num_layers separate calls, with no padding needed.
        pruned_prompts: list[list[int]] = []
        num_removed_per_layer: list[int] = []
        for layer in range(num_layers):
            prompt_scores = scores[layer][:prompt_len]
            cold_indices = select_cold_tokens(prompt_scores, frac=cold_frac)
            pruned_prompts.append(prune_prompt(prompt_ids, cold_indices))
            num_removed_per_layer.append(len(cold_indices))

        pruned_lengths = {len(p) for p in pruned_prompts}
        if len(pruned_lengths) != 1:
            raise RuntimeError(
                f"expected every layer's pruned prompt to have the same "
                f"length for batching, got lengths {sorted(pruned_lengths)}"
            )
        pruned_len = pruned_lengths.pop()

        batch_input_ids = torch.tensor(pruned_prompts, device=device)  # [num_layers, pruned_len]
        with torch.inference_mode():
            batched_pruned_output = model.generate(
                input_ids=batch_input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        for layer in range(num_layers):
            pruned_continuation = batched_pruned_output[layer].tolist()[pruned_len:]
            exact_match, first_div = compare_continuations(
                baseline_continuation, pruned_continuation
            )
            records.append({
                "passage_id": passage_id,
                "layer": layer,
                "num_removed": num_removed_per_layer[layer],
                "exact_match": exact_match,
                "first_divergence_idx": first_div,
            })

    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cold_token_ablation.py -v -k run_ablation_experiment --run-slow`
Expected: PASS (1 passed)

- [ ] **Step 5: Implement `write_outputs` and `main`**

```python
def write_outputs(records: list[dict], results_dir: str) -> None:
    import csv
    import os
    from collections import defaultdict

    os.makedirs(results_dir, exist_ok=True)

    records_path = os.path.join(results_dir, "cold_ablation.csv")
    with open(records_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "passage_id", "layer", "num_removed", "exact_match", "first_divergence_idx",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    by_layer: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_layer[r["layer"]].append(r)

    summary_path = os.path.join(results_dir, "cold_ablation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "n_passages", "exact_match_rate", "mean_first_divergence_idx"])
        for layer in sorted(by_layer):
            recs = by_layer[layer]
            n = len(recs)
            exact_rate = sum(r["exact_match"] for r in recs) / n
            mean_div = sum(r["first_divergence_idx"] for r in recs) / n
            writer.writerow([layer, n, f"{exact_rate:.3f}", f"{mean_div:.2f}"])


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kv_quant.bench.pos_attention_correlation import (
        chunk_token_ids,
        load_wikitext_token_ids,
    )

    model_id = "google/gemma-4-E2B-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    ).eval()

    token_ids = load_wikitext_token_ids(tokenizer)
    passages = chunk_token_ids(token_ids, n_passages=25, max_tokens=200)

    records = run_ablation_experiment(model, tokenizer, passages, max_new_tokens=30)

    write_outputs(records, results_dir="kv_quant/bench/results_ablation")
    print(f"Wrote {len(records)} records across {len(passages)} passages.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full test suite for this file**

Run: `pytest tests/test_cold_token_ablation.py -v --run-slow`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add kv_quant/bench/cold_token_ablation.py tests/test_cold_token_ablation.py
git commit -m "feat(kv_quant): add orchestration script for cold-token ablation experiment"
```

---

### Task 3: Run the full experiment and write narrative findings

**Files:**
- Create: `kv_quant/bench/findings_cold_ablation.md`

**Interfaces:**
- Consumes: `main()` from Task 2.

- [ ] **Step 1: Run the experiment against the real model**

Run: `python -m kv_quant.bench.cold_token_ablation`
Expected: prints `Wrote <N> records across 25 passages.` (N = 25 × num_hidden_layers); creates `kv_quant/bench/results_ablation/cold_ablation.csv` and `kv_quant/bench/results_ablation/cold_ablation_summary.csv`.

- [ ] **Step 2: Inspect the summary for per-layer exact-match rate and divergence trends**

Run: `column -s, -t kv_quant/bench/results_ablation/cold_ablation_summary.csv | less -S`
Look for: overall exact-match rate, whether early vs. late layers differ, and whether `mean_first_divergence_idx` trends toward 0 (immediate divergence) or 30 (no effective difference) at any depth.

- [ ] **Step 3: Write `kv_quant/bench/findings_cold_ablation.md`**

Write a findings doc following the same structure as `kv_quant/bench/findings_pos_attention.md`: a pointer to the raw per-layer table (`results_ablation/cold_ablation_summary.csv`, regenerated by `python -m kv_quant.bench.cold_token_ablation`), then a narrative section covering:
- Overall exact-match rate across all (passage, layer) pairs.
- Whether removing the bottom-10% cold tokens is more consequential at some layers than others (e.g. early layers where local context matters, vs. late layers).
- Any relationship to the POS-attention findings (e.g. if NUM/DET/PART removal at layers where they're most enriched among cold tokens still doesn't change output, that strengthens the case those tokens are genuinely low-value; if it does change output, that's evidence the correlational finding doesn't imply causal dispensability).
- Caveat: only 25 passages, single model, single `cold_frac=0.1`, splice-based removal (not masking) so position ids shift — note this as a possible confound distinct from "token removed."

- [ ] **Step 4: Commit**

```bash
git add kv_quant/bench/findings_cold_ablation.md kv_quant/bench/results_ablation/cold_ablation.csv kv_quant/bench/results_ablation/cold_ablation_summary.csv
git commit -m "docs(kv_quant): add findings for cold-token ablation experiment"
```
