# POS-Attention Correlation Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a script that runs 25 WikiText-2 passages through `google/gemma-4-E2B-it`, finds the bottom-10%-by-attention ("not activated") tokens per layer, and reports whether their POS tags are correlated per layer.

**Architecture:** A set of small pure functions (attention score accumulation, cold-token selection, offset→POS alignment, enrichment-ratio computation) that are unit tested without any model or GPU, plus a thin orchestration script in the same file that loads the real model, runs `generate(output_attentions=True)`, and wires the pure functions together to produce CSVs and a findings write-up.

**Tech Stack:** PyTorch, HuggingFace `transformers` (5.12.1) + `datasets`, spaCy (`en_core_web_sm`), pytest.

## Global Constraints

- Model: `google/gemma-4-E2B-it`, loaded with `attn_implementation="eager"` (SDPA/flash backends return `None` attentions).
- 25 passages from WikiText-2 test split (`wikitext-103-raw-v1`), each truncated to 200 tokens.
- `max_new_tokens=30` per passage during generation.
- "Not activated" = bottom 10% of tokens per layer by accumulated received-attention (mean over heads, summed over queries, accumulated across prefill + all decode steps).
- POS tagging via spaCy `en_core_web_sm`; subword tokens inherit their parent word's tag.
- Ambiguous POS/token alignment for a passage → log a warning and skip that passage, don't crash.
- Outputs: `results/pos_attention_correlation.csv`, `results/pos_attention_enrichment_summary.csv`, `kv_quant/bench/findings_pos_attention.md`.
- New file: `kv_quant/bench/pos_attention_correlation.py`. Not wired into `run_bench.py`'s CLI.
- Tests requiring the real model/GPU are marked `@pytest.mark.slow` (existing repo convention — see `pyproject.toml`'s `testpaths`/`markers`).

---

### Task 1: Attention score accumulation + cold-token selection

**Files:**
- Create: `kv_quant/bench/pos_attention_correlation.py`
- Test: `tests/test_pos_attention_correlation.py`

**Interfaces:**
- Produces: `accumulate_attention_scores(attentions_per_step: list[tuple[torch.Tensor, ...]], total_len: int, num_layers: int) -> list[torch.Tensor]` — returns one 1D tensor of length `total_len` per layer.
- Produces: `select_cold_tokens(scores: torch.Tensor, frac: float = 0.1) -> list[int]` — indices of the lowest-scoring `frac` fraction, ascending order, at least 1.

- [ ] **Step 1: Write failing tests for `accumulate_attention_scores`**

```python
# tests/test_pos_attention_correlation.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pos_attention_correlation.py -v`
Expected: FAIL with `ImportError: cannot import name 'accumulate_attention_scores'` (module doesn't exist yet).

- [ ] **Step 3: Implement `accumulate_attention_scores`**

```python
# kv_quant/bench/pos_attention_correlation.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pos_attention_correlation.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write failing tests for `select_cold_tokens`**

```python
from kv_quant.bench.pos_attention_correlation import select_cold_tokens


def test_select_cold_tokens_basic():
    scores = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0])
    cold = select_cold_tokens(scores, frac=0.4)
    assert cold == [1, 3]  # indices of the two lowest values, ascending by score


def test_select_cold_tokens_minimum_one():
    scores = torch.tensor([5.0, 1.0, 3.0])
    cold = select_cold_tokens(scores, frac=0.1)
    assert cold == [1]  # 10% of 3 rounds to 0, floor to 1
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_pos_attention_correlation.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_cold_tokens'`

- [ ] **Step 7: Implement `select_cold_tokens`**

```python
def select_cold_tokens(scores: torch.Tensor, frac: float = 0.1) -> list[int]:
    """Return indices of the lowest-scoring frac fraction of tokens (>= 1)."""
    n = scores.shape[0]
    k = max(1, int(n * frac))
    order = torch.argsort(scores)
    return order[:k].tolist()
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `pytest tests/test_pos_attention_correlation.py -v`
Expected: PASS (5 passed)

- [ ] **Step 9: Commit**

```bash
git add kv_quant/bench/pos_attention_correlation.py tests/test_pos_attention_correlation.py
git commit -m "feat(kv_quant): add attention score accumulation and cold-token selection"
```

---

### Task 2: WikiText passage chunking

**Files:**
- Modify: `kv_quant/bench/pos_attention_correlation.py`
- Test: `tests/test_pos_attention_correlation.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `chunk_token_ids(token_ids: list[int], n_passages: int, max_tokens: int) -> list[list[int]]` — splits a flat token-id list into up to `n_passages` non-overlapping chunks of at most `max_tokens` each, dropping any final chunk shorter than `max_tokens`.
- Produces: `load_wikitext_token_ids(tokenizer) -> list[int]` — thin wrapper around `datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="test")`, joins non-empty text fields with spaces, and tokenizes. Not unit tested (network + tokenizer download), mirrors the existing untested pattern in `kv_quant/bench/perplexity.py`.

- [ ] **Step 1: Write failing tests for `chunk_token_ids`**

```python
from kv_quant.bench.pos_attention_correlation import chunk_token_ids


def test_chunk_token_ids_basic():
    token_ids = list(range(10))
    chunks = chunk_token_ids(token_ids, n_passages=3, max_tokens=3)
    assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_chunk_token_ids_stops_at_n_passages():
    token_ids = list(range(100))
    chunks = chunk_token_ids(token_ids, n_passages=2, max_tokens=4)
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_chunk_token_ids_drops_short_final_chunk():
    token_ids = list(range(7))
    chunks = chunk_token_ids(token_ids, n_passages=5, max_tokens=3)
    # 7 tokens / 3 per chunk = 2 full chunks + 1 short chunk of 1, which is dropped
    assert chunks == [[0, 1, 2], [3, 4, 5]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pos_attention_correlation.py -v -k chunk_token_ids`
Expected: FAIL with `ImportError: cannot import name 'chunk_token_ids'`

- [ ] **Step 3: Implement `chunk_token_ids` and `load_wikitext_token_ids`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pos_attention_correlation.py -v -k chunk_token_ids`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add kv_quant/bench/pos_attention_correlation.py tests/test_pos_attention_correlation.py
git commit -m "feat(kv_quant): add WikiText passage chunking for POS-attention experiment"
```

---

### Task 3: POS alignment

**Files:**
- Modify: `kv_quant/bench/pos_attention_correlation.py`
- Modify: `pyproject.toml` (add `spacy` dependency)
- Test: `tests/test_pos_attention_correlation.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `align_offsets_to_pos(offset_mapping: list[tuple[int, int]], word_spans: list[tuple[int, int, str]]) -> list[str]` — pure function, maps each token's character offsets to the POS tag of the word span it overlaps. Tokens with `(0, 0)` offsets (special tokens) get `"SPECIAL"`. Tokens with no overlapping word span get `"X"`.
- Produces: `tag_text_pos(text: str, nlp) -> list[tuple[int, int, str]]` — thin spaCy wrapper returning `(start_char, end_char, pos_tag)` per spaCy token.
- Produces: `load_spacy_model() -> spacy.Language` — loads `en_core_web_sm`, raising a clear `RuntimeError` with the install command if the model isn't downloaded.

- [ ] **Step 1: Add spaCy dependency**

Edit `pyproject.toml`, in the `dependencies` list add `"spacy>=3.7"`:

```toml
dependencies = [
    "datasets>=5.0.0",
    "liger-kernel>=0.8.0",
    "pyyaml",
    "spacy>=3.7",
    "tokenizers>=0.23.1",
]
```

- [ ] **Step 2: Install spaCy and the small English model**

Run: `pip install "spacy>=3.7" && python -m spacy download en_core_web_sm`
Expected: both commands exit 0.

- [ ] **Step 3: Write failing tests for `align_offsets_to_pos`**

```python
from kv_quant.bench.pos_attention_correlation import align_offsets_to_pos


def test_align_offsets_basic():
    # text: "The dog runs." tokenized into subwords covering "The", "dog", "runs", "."
    offset_mapping = [(0, 3), (4, 7), (8, 12), (12, 13)]
    word_spans = [(0, 3, "DET"), (4, 7, "NOUN"), (8, 12, "VERB"), (12, 13, "PUNCT")]
    tags = align_offsets_to_pos(offset_mapping, word_spans)
    assert tags == ["DET", "NOUN", "VERB", "PUNCT"]


def test_align_offsets_special_token():
    offset_mapping = [(0, 0), (0, 3)]
    word_spans = [(0, 3, "DET")]
    tags = align_offsets_to_pos(offset_mapping, word_spans)
    assert tags == ["SPECIAL", "DET"]


def test_align_offsets_subword_inherits_word_tag():
    # "running" tokenized as "runn" + "ing", both inside word span (0, 7, "VERB")
    offset_mapping = [(0, 4), (4, 7)]
    word_spans = [(0, 7, "VERB")]
    tags = align_offsets_to_pos(offset_mapping, word_spans)
    assert tags == ["VERB", "VERB"]


def test_align_offsets_no_overlapping_span():
    offset_mapping = [(50, 55)]
    word_spans = [(0, 3, "DET")]
    tags = align_offsets_to_pos(offset_mapping, word_spans)
    assert tags == ["X"]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_pos_attention_correlation.py -v -k align_offsets`
Expected: FAIL with `ImportError: cannot import name 'align_offsets_to_pos'`

- [ ] **Step 5: Implement `align_offsets_to_pos`, `tag_text_pos`, `load_spacy_model`**

```python
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
```

- [ ] **Step 6: Run alignment tests to verify they pass**

Run: `pytest tests/test_pos_attention_correlation.py -v -k align_offsets`
Expected: PASS (4 passed)

- [ ] **Step 7: Write and run an integration test for `tag_text_pos`**

```python
def test_tag_text_pos_real_spacy():
    from kv_quant.bench.pos_attention_correlation import load_spacy_model

    nlp = load_spacy_model()
    spans = tag_text_pos("The dog runs.", nlp)
    tags = [pos for _, _, pos in spans]
    assert tags == ["DET", "NOUN", "VERB", "PUNCT"]
```

Run: `pytest tests/test_pos_attention_correlation.py -v -k tag_text_pos_real_spacy`
Expected: PASS (1 passed)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml kv_quant/bench/pos_attention_correlation.py tests/test_pos_attention_correlation.py
git commit -m "feat(kv_quant): add POS tagging and token-to-word alignment"
```

---

### Task 4: Enrichment ratio computation

**Files:**
- Modify: `kv_quant/bench/pos_attention_correlation.py`
- Test: `tests/test_pos_attention_correlation.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3 (operates on plain dict records with keys `"layer": int, "pos_tag": str, "is_cold": bool`).
- Produces: `compute_enrichment_ratios(records: list[dict]) -> dict[int, dict[str, float]]` — for each layer, for each POS tag that appears at least once overall in that layer, `ratio = (fraction of cold tokens with this tag) / (fraction of all tokens with this tag)`. Tags with zero cold occurrences get `ratio = 0.0`. Layers with zero cold tokens (shouldn't happen given `select_cold_tokens`'s minimum of 1, but guarded anyway) get an empty dict.

- [ ] **Step 1: Write failing tests for `compute_enrichment_ratios`**

```python
from kv_quant.bench.pos_attention_correlation import compute_enrichment_ratios


def test_compute_enrichment_ratios_basic():
    records = [
        {"layer": 0, "pos_tag": "PUNCT", "is_cold": True},
        {"layer": 0, "pos_tag": "PUNCT", "is_cold": True},
        {"layer": 0, "pos_tag": "NOUN", "is_cold": False},
        {"layer": 0, "pos_tag": "NOUN", "is_cold": False},
        {"layer": 0, "pos_tag": "NOUN", "is_cold": False},
        {"layer": 0, "pos_tag": "NOUN", "is_cold": False},
    ]
    # PUNCT: overall_frac = 2/6, cold_frac = 2/2 -> ratio = 1.0 / (2/6) = 3.0
    # NOUN: overall_frac = 4/6, cold_frac = 0/2 -> ratio = 0.0
    ratios = compute_enrichment_ratios(records)
    assert ratios[0]["PUNCT"] == 3.0
    assert ratios[0]["NOUN"] == 0.0


def test_compute_enrichment_ratios_separates_layers():
    records = [
        {"layer": 0, "pos_tag": "DET", "is_cold": True},
        {"layer": 0, "pos_tag": "DET", "is_cold": False},
        {"layer": 1, "pos_tag": "VERB", "is_cold": True},
        {"layer": 1, "pos_tag": "VERB", "is_cold": False},
    ]
    ratios = compute_enrichment_ratios(records)
    assert set(ratios.keys()) == {0, 1}
    assert ratios[0] == {"DET": 1.0}
    assert ratios[1] == {"VERB": 1.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pos_attention_correlation.py -v -k compute_enrichment_ratios`
Expected: FAIL with `ImportError: cannot import name 'compute_enrichment_ratios'`

- [ ] **Step 3: Implement `compute_enrichment_ratios`**

```python
def compute_enrichment_ratios(records: list[dict]) -> dict[int, dict[str, float]]:
    """Per layer, per POS tag: ratio of (fraction cold) to (fraction overall)."""
    from collections import defaultdict

    by_layer: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_layer[r["layer"]].append(r)

    result: dict[int, dict[str, float]] = {}
    for layer, recs in by_layer.items():
        total = len(recs)
        cold = [r for r in recs if r["is_cold"]]
        n_cold = len(cold)

        overall_counts: dict[str, int] = defaultdict(int)
        cold_counts: dict[str, int] = defaultdict(int)
        for r in recs:
            overall_counts[r["pos_tag"]] += 1
        for r in cold:
            cold_counts[r["pos_tag"]] += 1

        ratios = {}
        for tag, count in overall_counts.items():
            overall_frac = count / total
            cold_frac = cold_counts.get(tag, 0) / n_cold if n_cold else 0.0
            ratios[tag] = cold_frac / overall_frac if overall_frac > 0 else 0.0
        result[layer] = ratios
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pos_attention_correlation.py -v -k compute_enrichment_ratios`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add kv_quant/bench/pos_attention_correlation.py tests/test_pos_attention_correlation.py
git commit -m "feat(kv_quant): add POS enrichment ratio computation"
```

---

### Task 5: Orchestration script (model loading, generation loop, CSV + findings output)

**Files:**
- Modify: `kv_quant/bench/pos_attention_correlation.py`
- Test: `tests/test_pos_attention_correlation.py` (marked `@pytest.mark.slow`)

**Interfaces:**
- Consumes: `accumulate_attention_scores`, `select_cold_tokens` (Task 1); `chunk_token_ids`, `load_wikitext_token_ids` (Task 2); `align_offsets_to_pos`, `tag_text_pos`, `load_spacy_model` (Task 3); `compute_enrichment_ratios` (Task 4).
- Produces: `run_experiment(model, tokenizer, nlp, passages: list[list[int]], max_new_tokens: int = 30, cold_frac: float = 0.1) -> list[dict]` — returns the flat list of per-token records (`passage_id, layer, token, pos_tag, attn_score, is_cold`) used by `compute_enrichment_ratios`.
- Produces: `write_outputs(records: list[dict], enrichment: dict[int, dict[str, float]], results_dir: str, findings_path: str) -> None`.
- Produces: `main() -> None` — the `if __name__ == "__main__":` entry point that wires everything together, including the first-passage smoke-test assertion from the spec.

- [ ] **Step 1: Write a failing slow test for `run_experiment` on a tiny real model**

This test uses a tiny public GPT-2-family model instead of Gemma so it runs fast in CI while still exercising the real `generate(output_attentions=True)` path end-to-end.

```python
import pytest


@pytest.mark.slow
def test_run_experiment_end_to_end_tiny_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kv_quant.bench.pos_attention_correlation import (
        load_spacy_model,
        run_experiment,
    )

    model_id = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager"
    ).eval()
    nlp = load_spacy_model()

    text = "The quick brown fox jumps over the lazy dog."
    token_ids = tokenizer(text)["input_ids"]
    passages = [token_ids]

    records = run_experiment(model, tokenizer, nlp, passages, max_new_tokens=3)

    assert len(records) > 0
    for r in records:
        assert set(r.keys()) == {
            "passage_id", "layer", "token", "pos_tag", "attn_score", "is_cold",
        }
    n_layers = model.config.num_hidden_layers
    assert {r["layer"] for r in records} == set(range(n_layers))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pos_attention_correlation.py -v -k run_experiment --run-slow`
Expected: FAIL with `ImportError: cannot import name 'run_experiment'`

- [ ] **Step 3: Implement `run_experiment`**

```python
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
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-2
            ), "Sanity check failed: attention rows do not sum to ~1.0"

        scores = accumulate_attention_scores(
            list(output.attentions), total_len=total_len, num_layers=num_layers
        )

        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
        encoding = tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        offset_mapping = encoding["offset_mapping"]
        if len(offset_mapping) != total_len:
            print(
                f"[warn] passage {passage_id}: offset_mapping length "
                f"{len(offset_mapping)} != token count {total_len}, skipping"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pos_attention_correlation.py -v -k run_experiment --run-slow`
Expected: PASS (1 passed)

- [ ] **Step 5: Implement `write_outputs` and `main`**

```python
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
```

- [ ] **Step 6: Run the full test suite for this file**

Run: `pytest tests/test_pos_attention_correlation.py -v --run-slow`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add kv_quant/bench/pos_attention_correlation.py tests/test_pos_attention_correlation.py
git commit -m "feat(kv_quant): add orchestration script for POS-attention experiment"
```

---

### Task 6: Run the full experiment and write narrative findings

**Files:**
- Modify: `kv_quant/bench/findings_pos_attention.md` (append narrative section below the auto-generated table)

**Interfaces:**
- Consumes: `main()` from Task 5.

- [ ] **Step 1: Run the experiment against the real model**

Run: `python -m kv_quant.bench.pos_attention_correlation`
Expected: prints `Wrote <N> records across 25 passages.`; creates `results/pos_attention_correlation.csv`, `results/pos_attention_enrichment_summary.csv`, and `kv_quant/bench/findings_pos_attention.md`.

- [ ] **Step 2: Inspect the enrichment summary for consistent patterns**

Run: `column -s, -t results/pos_attention_enrichment_summary.csv | less -S`
Look for POS tags whose ratio is consistently >1.5 or <0.5 across most layers, and any layer-dependent trend (e.g. ratios shifting from early to late layers).

- [ ] **Step 3: Append a narrative findings section to `kv_quant/bench/findings_pos_attention.md`**

Write 2-4 short paragraphs directly into the file (after the auto-generated table) describing: which POS tags are most/least attended overall, whether the pattern is stable across layers or concentrated in specific layers (e.g. early vs. late), and any caveat about the 25-passage sample size limiting statistical confidence.

- [ ] **Step 4: Commit**

```bash
git add kv_quant/bench/findings_pos_attention.md results/pos_attention_correlation.csv results/pos_attention_enrichment_summary.csv
git commit -m "docs(kv_quant): add findings for POS-attention correlation experiment"
```
