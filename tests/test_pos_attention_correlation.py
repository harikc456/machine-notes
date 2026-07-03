from __future__ import annotations
import torch

from kv_quant.bench.pos_attention_correlation import accumulate_attention_scores


def test_accumulate_single_step_uniform_attention():
    # 1 layer, batch=1, heads=2, q_len=3, kv_len=3, all attention weights = 1.0
    layer_tensor = torch.ones(1, 2, 3, 3)
    attentions_per_step = [(layer_tensor,)]
    scores = accumulate_attention_scores(attentions_per_step, total_len=3, num_layers=1)
    assert len(scores) == 1
    # mean over heads -> 1.0 per (q,k); sum over 3 queries -> 3.0 per key position
    assert torch.allclose(scores[0], torch.tensor([3.0, 3.0, 3.0]))


def test_accumulate_across_decode_steps():
    # 1 layer, prefill step: q_len=2, kv_len=2; decode step: q_len=1, kv_len=3 (cache grew by 1)
    prefill = torch.ones(1, 1, 2, 2)  # each key gets 2.0 (sum over 2 queries)
    decode = torch.zeros(1, 1, 1, 3)
    decode[0, 0, 0, :] = torch.tensor([0.5, 0.5, 1.0])  # single query row
    attentions_per_step = [(prefill,), (decode,)]
    scores = accumulate_attention_scores(attentions_per_step, total_len=3, num_layers=1)
    # position 0: 2.0 (prefill) + 0.5 (decode) = 2.5
    # position 1: 2.0 (prefill) + 0.5 (decode) = 2.5
    # position 2: 0.0 (didn't exist during prefill) + 1.0 (decode) = 1.0
    assert torch.allclose(scores[0], torch.tensor([2.5, 2.5, 1.0]))


def test_accumulate_multiple_layers_independent():
    layer0 = torch.ones(1, 1, 1, 2)
    layer1 = torch.zeros(1, 1, 1, 2)
    layer1[0, 0, 0, 0] = 5.0
    attentions_per_step = [(layer0, layer1)]
    scores = accumulate_attention_scores(attentions_per_step, total_len=2, num_layers=2)
    assert torch.allclose(scores[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(scores[1], torch.tensor([5.0, 0.0]))


def test_select_cold_tokens_basic():
    from kv_quant.bench.pos_attention_correlation import select_cold_tokens
    scores = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0])
    cold = select_cold_tokens(scores, frac=0.4)
    assert cold == [1, 3]  # indices of the two lowest values, ascending by score


def test_select_cold_tokens_minimum_one():
    from kv_quant.bench.pos_attention_correlation import select_cold_tokens
    scores = torch.tensor([5.0, 1.0, 3.0])
    cold = select_cold_tokens(scores, frac=0.1)
    assert cold == [1]  # 10% of 3 rounds to 0, floor to 1
