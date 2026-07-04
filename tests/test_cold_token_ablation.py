from __future__ import annotations

from unittest.mock import MagicMock, patch

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


@pytest.mark.slow
def test_run_ablation_experiment_calls_generate_exactly_twice_per_passage():
    """The whole point of batching all layers' pruned prompts into one
    generate() call is to avoid a per-layer loop. Guard against a regression
    back to num_layers + 1 calls."""
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

    generate_spy = MagicMock(wraps=model.generate)
    model.generate = generate_spy

    run_ablation_experiment(model, tokenizer, passages, max_new_tokens=3)

    assert generate_spy.call_count == 2


def test_run_ablation_experiment_raises_when_baseline_attentions_missing():
    """If model.generate() comes back without attentions (e.g. model not
    loaded with attn_implementation='eager'), fail loudly instead of
    silently producing garbage cold-token selections."""
    import torch

    from kv_quant.bench.cold_token_ablation import run_ablation_experiment

    num_layers = 2

    baseline_output = MagicMock()
    baseline_output.attentions = None

    fake_model = MagicMock()
    fake_model.parameters.return_value = iter([torch.zeros(1)])
    fake_model.config.get_text_config.return_value.num_hidden_layers = num_layers
    fake_model.generate.return_value = baseline_output

    passages = [[1, 2, 3, 4]]

    with pytest.raises(RuntimeError, match="no attentions"):
        run_ablation_experiment(fake_model, tokenizer=None, passages=passages, max_new_tokens=1)

    fake_model.generate.assert_called_once()


def test_run_ablation_experiment_raises_when_pruned_prompt_lengths_differ():
    """select_cold_tokens is deterministic given a fixed prompt_len and
    cold_frac, so every layer's pruned prompt should end up the same length
    (required for batching into a single generate() call). Force the
    length-mismatch branch by making select_cold_tokens return a different
    number of indices on successive calls within one passage, and confirm
    the real guard in run_ablation_experiment raises rather than silently
    truncating/padding."""
    import torch

    from kv_quant.bench.cold_token_ablation import run_ablation_experiment

    num_layers = 2
    prompt_len = 4

    # q_len=1, kv_len=prompt_len, batch=1, heads=1
    layer_attn = torch.ones(1, 1, 1, prompt_len)
    baseline_output = MagicMock()
    baseline_output.attentions = [(layer_attn, layer_attn)]
    baseline_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])  # prompt_len + 1 new token

    fake_model = MagicMock()
    fake_model.parameters.return_value = iter([torch.zeros(1)])
    fake_model.config.get_text_config.return_value.num_hidden_layers = num_layers
    fake_model.generate.return_value = baseline_output

    passages = [[1, 2, 3, 4]]

    # First layer prunes 1 token, second layer prunes 2 -> unequal pruned lengths.
    with patch(
        "kv_quant.bench.pos_attention_correlation.select_cold_tokens",
        side_effect=[[0], [0, 1]],
    ):
        with pytest.raises(RuntimeError, match="same length"):
            run_ablation_experiment(
                fake_model, tokenizer=None, passages=passages, max_new_tokens=1
            )

    # Only the baseline call happened; the batched call is never reached
    # because the guard raises first.
    fake_model.generate.assert_called_once()
