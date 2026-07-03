# POS–Attention Correlation Experiment (Gemma 4 E2B)

## Goal

Run ~25 inference requests through `google/gemma-4-E2B-it` and, per transformer
layer, identify the bottom 10% of tokens by aggregate attention received
("not activated" tokens). Check whether the POS tags of these tokens are
correlated — i.e. do certain parts of speech (punctuation, determiners,
stopwords, etc.) consistently receive the least attention in specific layers.

## Non-goals

- Not building reusable eviction/importance infra (that already exists in
  `kv_quant`'s TriAttention path) — this is a standalone analysis script.
- Not covering multiple models or a sweep of hyperparameters; single model,
  single config, exploratory pass.

## Data source

25 passages drawn from the WikiText-2 test split (`wikitext-103-raw-v1`,
same HF dataset already used in `kv_quant/bench/perplexity.py`), each
truncated to 200 tokens. Fixed length keeps the 25 runs comparable.

## Model / attention capture

- Load with `attn_implementation="eager"` — SDPA/flash-attention backends
  return `None` for attention weights even when `output_attentions=True`,
  so eager is required for this experiment.
- For each passage, call:
  ```python
  model.generate(
      **inputs,
      output_attentions=True,
      return_dict_in_generate=True,
      max_new_tokens=30,
  )
  ```
  This yields attentions for both the prefill pass and every decode step.
- Sanity check before the full loop: assert attentions are not `None`;
  print per-layer attention row sums (should be ≈1.0 across heads/queries)
  for the first passage only, as a smoke test.

## Attention score reduction

For each layer `l`:
1. Each captured attention tensor has shape `[batch, heads, q_len, kv_len]`.
2. Mean over the `heads` dimension.
3. Sum over the `q_len` (query) dimension to get attention *received* per
   key position for that pass/step.
4. Accumulate per key-position scores across the prefill pass and all
   decode steps (decode steps re-attend the whole growing cache, so a
   given position keeps accumulating additional received-attention as
   later steps run).

Result: one accumulated received-attention score per token position per
layer, per passage.

## "Not activated" definition

Per layer, per passage: rank all tokens (prompt tokens + generated tokens)
by accumulated received-attention score, ascending. The bottom 10% are
"not activated" for that layer.

## POS tagging and alignment

- Tag each passage's original text with spaCy (`en_core_web_sm`).
- Use the fast tokenizer's `offset_mapping` to align each subword token to
  the spaCy word span it falls inside.
- Each subword token inherits its parent word's POS tag. Multi-token words
  therefore contribute multiple token-level entries with the same tag —
  acceptable since the analysis compares tag *distributions*, not word
  counts.
- If offset-based alignment is ambiguous for a passage (e.g. tokenizer/
  spaCy disagree on boundaries in a way that can't be resolved), skip that
  passage: log a warning and continue with the rest.

## Aggregation & output

Across all 25 passages, per layer, compute:
- POS tag distribution among "not activated" (bottom-10%) tokens
- POS tag distribution among all tokens (baseline)
- Enrichment ratio per tag: `(% of tag in bottom-10%) / (% of tag overall)`

Outputs:
- `results/pos_attention_correlation.csv` — raw per-token records:
  `passage_id, layer, token, pos_tag, attn_score, is_cold`
- `results/pos_attention_enrichment_summary.csv` — per-layer, per-POS-tag
  enrichment ratios
- `kv_quant/bench/findings_pos_attention.md` — narrative write-up of any
  consistent per-layer POS correlation found

## Implementation location

New script: `kv_quant/bench/pos_attention_correlation.py`, runnable
standalone (not wired into `run_bench.py`'s CLI, since this is an
exploratory analysis, not a benchmark to sweep).

## Error handling

- Fail fast with a clear message if attentions come back `None`
  (wrong attn_implementation).
- Skip (log + continue) passages with ambiguous POS/token alignment
  rather than crashing the whole run.
- No other defensive handling — single model, single config, local
  exploratory run.
