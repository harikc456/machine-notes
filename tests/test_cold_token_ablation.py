from __future__ import annotations

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
