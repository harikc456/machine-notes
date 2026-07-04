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
