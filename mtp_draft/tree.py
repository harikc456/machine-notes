from __future__ import annotations
import torch


def build_tree(
    logits: torch.Tensor,
    tau: float,
    max_tree_nodes: int = 256,
) -> list[list[int]]:
    """
    Construct speculative decoding candidate tree from draft logits.

    For each draft position i, selects candidate tokens whose log-probability
    is within tau nats of the top token. The full candidate tree is the
    Cartesian product of per-position candidate sets, pruned to max_tree_nodes
    paths by cumulative log-probability (highest first).

    Args:
        logits:         (max_draft, vocab) raw logits
        tau:            relative threshold in log-prob space (nats)
        max_tree_nodes: maximum number of candidate paths to return

    Returns:
        List of candidate token sequences (list[int]), sorted highest
        cumulative log-prob first. Each sequence has length max_draft.
    """
    max_draft = logits.shape[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)   # (max_draft, vocab)

    # paths: list of (cumulative_log_prob, token_sequence)
    paths: list[tuple[float, list[int]]] = [(0.0, [])]

    for i in range(max_draft):
        lp_i = log_probs[i]
        top_lp = lp_i.max().item()
        threshold = top_lp - tau
        # Always include the top token even if tau=0
        mask = lp_i >= threshold
        candidates = mask.nonzero(as_tuple=True)[0].tolist()
        if not candidates:
            candidates = [int(lp_i.argmax().item())]

        new_paths: list[tuple[float, list[int]]] = []
        for cum_lp, tokens in paths:
            for tok in candidates:
                new_lp = cum_lp + lp_i[tok].item()
                new_paths.append((new_lp, tokens + [tok]))

        # Prune to max_tree_nodes by score
        new_paths.sort(key=lambda x: -x[0])
        paths = new_paths[:max_tree_nodes]

    return [tokens for _, tokens in paths]
