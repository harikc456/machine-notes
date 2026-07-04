from __future__ import annotations


def prune_prompt(prompt_ids: list[int], cold_indices: list[int]) -> list[int]:
    """Return prompt_ids with the positions in cold_indices removed."""
    cold_set = set(cold_indices)
    return [tok for i, tok in enumerate(prompt_ids) if i not in cold_set]


def compare_continuations(baseline: list[int], pruned: list[int]) -> tuple[bool, int]:
    """Compare two equal-length greedy continuations token-by-token."""
    first_div = len(baseline)
    for i, (b, p) in enumerate(zip(baseline, pruned)):
        if b != p:
            first_div = i
            break
    exact_match = first_div == len(baseline)
    return exact_match, first_div


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
