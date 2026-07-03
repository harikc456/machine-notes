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


def chunk_token_ids(
    token_ids: list[int], n_passages: int, max_tokens: int
) -> list[list[int]]:
    """Split token_ids into up to n_passages chunks of exactly max_tokens each."""
    chunks = []
    for start in range(0, len(token_ids), max_tokens):
        if len(chunks) >= n_passages:
            break
        chunk = token_ids[start : start + max_tokens]
        if len(chunk) < max_tokens:
            break
        chunks.append(chunk)
    return chunks


def load_wikitext_token_ids(tokenizer) -> list[int]:
    """Load and tokenize the WikiText-2 test split (mirrors perplexity.py)."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = " ".join(ex["text"] for ex in dataset if ex["text"].strip())
    return tokenizer(text, return_tensors=None)["input_ids"]


def align_offsets_to_pos(
    offset_mapping: list[tuple[int, int]],
    word_spans: list[tuple[int, int, str]],
) -> list[str]:
    """Map each token's char offsets to the POS tag of the word it overlaps."""
    tags = []
    for start, end in offset_mapping:
        if start == end:
            tags.append("SPECIAL")
            continue
        tag = "X"
        for word_start, word_end, pos in word_spans:
            if start < word_end and end > word_start:
                tag = pos
                break
        tags.append(tag)
    return tags


def tag_text_pos(text: str, nlp) -> list[tuple[int, int, str]]:
    """Run spaCy POS tagging, returning (start_char, end_char, pos_tag) per token."""
    doc = nlp(text)
    return [(tok.idx, tok.idx + len(tok.text), tok.pos_) for tok in doc]


def load_spacy_model():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Install it with: "
            "python -m spacy download en_core_web_sm"
        ) from e
