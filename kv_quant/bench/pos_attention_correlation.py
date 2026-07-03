from __future__ import annotations

import torch


def accumulate_attention_scores(
    attentions_per_step: list[tuple[torch.Tensor, ...]],
    total_len: int,
    num_layers: int,
) -> list[torch.Tensor]:
    """Accumulate received-attention scores per key position, per layer.

    attentions_per_step[i] is a tuple of length num_layers, one tensor per
    layer shaped [batch=1, heads, q_len, kv_len] (batch must be 1). Each
    tensor is reduced by averaging over heads and summing over queries,
    then added into a running per-key-position total for that layer.
    """
    scores = [torch.zeros(total_len) for _ in range(num_layers)]
    for step_attn in attentions_per_step:
        for layer_idx, layer_tensor in enumerate(step_attn):
            reduced = layer_tensor[0].mean(dim=0)  # [q_len, kv_len]
            per_key = reduced.sum(dim=0)  # [kv_len]
            kv_len = per_key.shape[0]
            scores[layer_idx][:kv_len] += per_key
    return scores


def select_cold_tokens(scores: torch.Tensor, frac: float = 0.1) -> list[int]:
    """Return indices of the lowest-scoring frac fraction of tokens (>= 1)."""
    n = scores.shape[0]
    k = max(1, int(n * frac))
    order = torch.argsort(scores)
    return order[:k].tolist()
