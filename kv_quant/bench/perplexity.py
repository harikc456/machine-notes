from __future__ import annotations
import math
import torch


def compute_perplexity(
    model,
    tokenizer,
    n_tokens: int = 10_240,
    chunk_size: int = 512,
) -> float:
    """WikiText-2 perplexity over the first n_tokens tokens, evaluated in
    non-overlapping chunks of chunk_size. Returns float PPL."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(ex["text"] for ex in dataset if ex["text"].strip())
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    enc = enc[: n_tokens + 1]

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(enc) - 1, chunk_size):
            chunk = enc[i : i + chunk_size + 1]
            if len(chunk) < 2:
                break
            input_ids = chunk[:-1].unsqueeze(0).to(device)
            labels    = chunk[1:].unsqueeze(0).to(device)
            loss = model(input_ids, labels=labels).loss
            n = input_ids.shape[1]
            total_nll    += loss.item() * n
            total_tokens += n

    return math.exp(total_nll / total_tokens)
