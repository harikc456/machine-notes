# Cold-Token Ablation Experiment (Gemma 4 E2B)

## Goal

Test whether the "not activated" (bottom-10%-attention) tokens identified in
`kv_quant/bench/pos_attention_correlation.py` are actually dispensable: if we
remove them from the prompt and regenerate, does the model produce the same
continuation? This is the causal follow-up to the correlational POS-attention
findings.

## Non-goals

- Not building reusable KV-eviction infra (TriAttention already covers that).
- Not re-deriving POS correlations — this experiment only uses attention
  scores to pick which tokens to remove, not their POS tags.
- Not testing every layer's cold set independently against every other
  layer's baseline — each layer's ablation is compared only to *that
  passage's own* full-context baseline.

## Data source

Same 25 passages as `findings_pos_attention.md`: WikiText-2 test split
(`wikitext-103-raw-v1`), 200-token chunks, same chunking function
(`chunk_token_ids`) for direct comparability with the existing findings.

## Procedure (per passage)

1. **Baseline run.** Greedy-decode (`do_sample=False`) 30 new tokens from the
   full 200-token prompt with `output_attentions=True`, same as the original
   script. Record:
   - `baseline_continuation`: the 30 generated token ids.
   - Per-layer accumulated received-attention scores over the full sequence
     (via the existing `accumulate_attention_scores`).
2. **Per-layer ablation.** For each of the 35 layers:
   - Compute that layer's cold set using `select_cold_tokens` restricted to
     the **first 200 positions only** (the original prompt) — generated
     positions are excluded since we are pruning context, not output.
     `frac=0.1` → ~20 tokens removed per layer.
   - Splice those token positions out of the original prompt's `input_ids`
     (shortens the sequence; no attention masking, no position-id padding —
     positions simply shift for the remaining tokens).
   - Greedy-decode 30 new tokens from the pruned prompt. No
     `output_attentions` needed here (cheaper — this run is only for
     comparison, not further analysis).
   - Compare the pruned continuation to `baseline_continuation`:
     - `exact_match`: all 30 tokens identical.
     - `first_divergence_idx`: index of the first differing token (30 if
       exact match).

## Output

- `kv_quant/bench/results_ablation/cold_ablation.csv` — one row per
  `(passage_id, layer)`: `passage_id, layer, num_removed, exact_match,
  first_divergence_idx`.
- `kv_quant/bench/results_ablation/cold_ablation_summary.csv` — per-layer
  aggregate across the 25 passages: exact-match rate, mean
  `first_divergence_idx`.
- `kv_quant/bench/findings_cold_ablation.md` — narrative write-up, same style
  as `findings_pos_attention.md`.

## Implementation location

New script: `kv_quant/bench/cold_token_ablation.py`, standalone (mirrors
`pos_attention_correlation.py`'s structure and reuses its
`accumulate_attention_scores`, `select_cold_tokens`, `chunk_token_ids`,
`load_wikitext_token_ids` helpers via import rather than duplicating them).

## Scale / cost

25 passages × (1 baseline + 35 pruned regenerations) = 900 `generate()` calls
total. Pruned runs skip attention capture, so they're cheaper per-call than
the baseline runs from the original experiment.

## Error handling

- Fail fast if attentions come back `None` on the baseline run (wrong
  `attn_implementation`).
- No other defensive handling — single model, single config, local
  exploratory run, consistent with the original experiment's error-handling
  posture.
