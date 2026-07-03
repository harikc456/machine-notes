from __future__ import annotations

import logging
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


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


def compute_enrichment_ratios(records: list[dict]) -> dict[int, dict[str, float]]:
    """Per layer, per POS tag: ratio of (fraction cold) to (fraction overall)."""
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_layer[r["layer"]].append(r)

    result: dict[int, dict[str, float]] = {}
    for layer, recs in by_layer.items():
        total = len(recs)
        cold = [r for r in recs if r["is_cold"]]
        n_cold = len(cold)

        # Layers with zero cold tokens get an empty dict.
        if n_cold == 0:
            result[layer] = {}
            continue

        overall_counts: dict[str, int] = defaultdict(int)
        cold_counts: dict[str, int] = defaultdict(int)
        for r in recs:
            overall_counts[r["pos_tag"]] += 1
        for r in cold:
            cold_counts[r["pos_tag"]] += 1

        ratios = {}
        for tag, count in overall_counts.items():
            overall_frac = count / total
            cold_frac = cold_counts.get(tag, 0) / n_cold
            ratios[tag] = cold_frac / overall_frac
        result[layer] = ratios
    return result


def run_experiment(
    model,
    tokenizer,
    nlp,
    passages: list[list[int]],
    max_new_tokens: int = 30,
    cold_frac: float = 0.1,
) -> list[dict]:
    """Run generation + attention capture over each passage, returning flat
    per-token records with layer, POS tag, attention score, and cold flag."""
    import torch

    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    records: list[dict] = []

    for passage_id, prompt_ids in enumerate(passages):
        input_ids = torch.tensor([prompt_ids], device=device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        if output.attentions is None or output.attentions[0][0] is None:
            raise RuntimeError(
                "model.generate() returned no attentions. Ensure the model "
                "was loaded with attn_implementation='eager'."
            )

        full_ids = output.sequences[0].tolist()
        total_len = len(full_ids)

        if passage_id == 0:
            first_step_attn = output.attentions[0][0][0]  # [heads, q_len, kv_len]
            row_sums = first_step_attn.sum(dim=-1)
            if not torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-2
            ):
                raise RuntimeError("Sanity check failed: attention rows do not sum to ~1.0")

        scores = accumulate_attention_scores(
            list(output.attentions), total_len=total_len, num_layers=num_layers
        )

        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
        encoding = tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        offset_mapping = encoding["offset_mapping"]
        if len(offset_mapping) != total_len:
            logger.warning(
                "passage %d: offset_mapping length %d != token count %d, skipping",
                passage_id,
                len(offset_mapping),
                total_len,
            )
            continue

        word_spans = tag_text_pos(full_text, nlp)
        pos_tags = align_offsets_to_pos(offset_mapping, word_spans)

        for layer in range(num_layers):
            cold_indices = set(select_cold_tokens(scores[layer], frac=cold_frac))
            for pos in range(total_len):
                records.append({
                    "passage_id": passage_id,
                    "layer": layer,
                    "token": tokenizer.convert_ids_to_tokens(full_ids[pos]),
                    "pos_tag": pos_tags[pos] if pos < len(pos_tags) else "X",
                    "attn_score": scores[layer][pos].item(),
                    "is_cold": pos in cold_indices,
                })

    return records


def write_outputs(
    records: list[dict],
    enrichment: dict[int, dict[str, float]],
    results_dir: str,
    findings_path: str,
) -> None:
    import csv
    import os

    os.makedirs(results_dir, exist_ok=True)

    records_path = os.path.join(results_dir, "pos_attention_correlation.csv")
    with open(records_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["passage_id", "layer", "token", "pos_tag", "attn_score", "is_cold"]
        )
        writer.writeheader()
        writer.writerows(records)

    summary_path = os.path.join(results_dir, "pos_attention_enrichment_summary.csv")
    all_tags = sorted({tag for tags in enrichment.values() for tag in tags})
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + all_tags)
        for layer in sorted(enrichment):
            row = [layer] + [enrichment[layer].get(tag, "") for tag in all_tags]
            writer.writerow(row)

    with open(findings_path, "w") as f:
        f.write("# POS-Attention Correlation: Raw Enrichment Data\n\n")
        f.write(
            "Enrichment ratio = (fraction of bottom-10%-attention tokens with "
            "this POS tag) / (fraction of all tokens with this tag). "
            "Ratio > 1 means the tag is over-represented among low-attention "
            "tokens for that layer; < 1 means under-represented.\n\n"
        )
        f.write("| Layer | " + " | ".join(all_tags) + " |\n")
        f.write("|---" * (len(all_tags) + 1) + "|\n")
        for layer in sorted(enrichment):
            row = [f"{enrichment[layer].get(tag, ''):.2f}" if tag in enrichment[layer] else "-" for tag in all_tags]
            f.write(f"| {layer} | " + " | ".join(row) + " |\n")


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "google/gemma-4-E2B-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    ).eval()
    nlp = load_spacy_model()

    token_ids = load_wikitext_token_ids(tokenizer)
    passages = chunk_token_ids(token_ids, n_passages=25, max_tokens=200)

    records = run_experiment(model, tokenizer, nlp, passages, max_new_tokens=30)
    enrichment = compute_enrichment_ratios(records)

    write_outputs(
        records,
        enrichment,
        results_dir="results",
        findings_path="kv_quant/bench/findings_pos_attention.md",
    )
    print(f"Wrote {len(records)} records across {len(passages)} passages.")


if __name__ == "__main__":
    main()
