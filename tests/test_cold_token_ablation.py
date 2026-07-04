from __future__ import annotations

import pytest

from kv_quant.bench.cold_token_ablation import compare_continuations, prune_prompt


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
