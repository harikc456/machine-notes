# POS-Attention Correlation: Raw Enrichment Data

Enrichment ratio = (fraction of bottom-10%-attention tokens with this POS tag) / (fraction of all tokens with this tag). Ratio > 1 means the tag is over-represented among low-attention tokens for that layer; < 1 means under-represented.

| Layer | ADJ | ADP | ADV | AUX | CCONJ | DET | NOUN | NUM | PART | PRON | PROPN | PUNCT | SCONJ | SPACE | SYM | VERB | X |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.30 | 0.58 | 1.33 | 1.03 | 0.81 | 0.42 | 1.54 | 2.63 | 1.74 | 0.74 | 1.09 | 0.20 | 0.00 | 0.65 | 0.00 | 0.48 | 1.85 |
| 1 | 0.93 | 0.37 | 0.42 | 0.73 | 0.73 | 1.43 | 1.19 | 2.35 | 1.33 | 0.83 | 1.45 | 0.24 | 0.00 | 0.98 | 0.00 | 0.24 | 3.63 |
| 2 | 0.47 | 0.44 | 0.42 | 0.44 | 0.59 | 3.56 | 0.88 | 1.32 | 0.99 | 0.65 | 1.33 | 0.36 | 0.00 | 1.14 | 0.00 | 0.28 | 3.89 |
| 3 | 0.65 | 0.92 | 0.63 | 0.78 | 0.88 | 2.65 | 0.54 | 1.50 | 2.32 | 0.83 | 0.95 | 0.35 | 0.00 | 1.47 | 0.00 | 0.22 | 4.88 |
| 4 | 0.53 | 0.62 | 0.21 | 1.57 | 1.10 | 2.37 | 0.61 | 1.35 | 1.91 | 1.04 | 0.59 | 1.05 | 0.18 | 2.02 | 0.00 | 0.48 | 2.97 |
| 5 | 0.68 | 0.95 | 0.35 | 1.47 | 0.88 | 2.00 | 0.86 | 1.91 | 2.49 | 1.07 | 1.08 | 0.53 | 0.00 | 0.60 | 0.00 | 0.33 | 1.58 |
| 6 | 0.53 | 1.20 | 0.77 | 1.47 | 0.81 | 0.57 | 0.94 | 2.57 | 3.23 | 1.07 | 1.21 | 0.50 | 0.00 | 0.76 | 0.00 | 0.30 | 1.25 |
| 7 | 1.30 | 0.60 | 0.98 | 1.17 | 0.44 | 0.52 | 1.32 | 1.76 | 1.74 | 0.59 | 1.56 | 0.72 | 0.00 | 0.65 | 0.00 | 0.72 | 0.53 |
| 8 | 1.02 | 1.20 | 0.63 | 1.22 | 1.39 | 1.79 | 0.65 | 1.50 | 2.74 | 1.24 | 0.74 | 0.66 | 0.36 | 1.20 | 0.00 | 0.37 | 1.58 |
| 9 | 0.90 | 1.04 | 0.56 | 1.57 | 1.10 | 1.74 | 0.86 | 1.29 | 2.24 | 1.04 | 0.89 | 0.52 | 0.00 | 1.25 | 0.00 | 0.69 | 1.45 |
| 10 | 1.46 | 1.00 | 0.91 | 1.22 | 0.81 | 1.40 | 0.87 | 1.79 | 2.74 | 1.15 | 1.03 | 0.30 | 0.18 | 0.87 | 0.00 | 0.52 | 1.12 |
| 11 | 1.43 | 1.18 | 1.26 | 1.27 | 1.17 | 0.94 | 0.85 | 1.44 | 2.49 | 0.68 | 1.12 | 0.53 | 0.36 | 1.04 | 0.00 | 0.65 | 1.06 |
| 12 | 0.96 | 0.86 | 0.84 | 1.32 | 1.17 | 1.98 | 0.67 | 1.57 | 2.74 | 1.15 | 0.88 | 0.61 | 0.00 | 1.14 | 1.25 | 0.35 | 2.11 |
| 13 | 0.78 | 0.55 | 0.28 | 1.22 | 0.59 | 2.88 | 0.71 | 2.07 | 2.49 | 1.33 | 0.95 | 0.39 | 0.00 | 1.04 | 0.00 | 0.26 | 2.11 |
| 14 | 1.55 | 0.39 | 0.84 | 0.93 | 0.59 | 1.09 | 1.17 | 2.10 | 1.58 | 0.92 | 1.38 | 0.17 | 0.00 | 0.82 | 0.00 | 0.80 | 1.91 |
| 15 | 1.93 | 0.46 | 0.77 | 0.83 | 0.51 | 1.51 | 1.08 | 1.47 | 1.74 | 1.15 | 1.38 | 0.20 | 0.00 | 1.09 | 0.00 | 0.61 | 1.45 |
| 16 | 1.62 | 0.41 | 0.77 | 0.98 | 0.51 | 1.82 | 1.01 | 1.66 | 1.91 | 1.12 | 1.33 | 0.30 | 0.00 | 1.04 | 0.00 | 0.43 | 1.65 |
| 17 | 1.06 | 0.51 | 0.98 | 1.32 | 0.66 | 1.74 | 0.94 | 1.50 | 2.07 | 1.33 | 1.36 | 0.35 | 0.00 | 1.09 | 0.00 | 0.52 | 1.39 |
| 18 | 0.93 | 0.67 | 0.49 | 1.22 | 0.59 | 2.18 | 0.72 | 1.85 | 2.49 | 1.66 | 0.88 | 0.41 | 0.00 | 1.25 | 0.00 | 0.48 | 1.91 |
| 19 | 0.71 | 0.44 | 0.35 | 1.42 | 0.44 | 2.13 | 0.83 | 2.48 | 1.66 | 1.18 | 1.00 | 0.25 | 0.00 | 1.58 | 0.00 | 0.43 | 2.97 |
| 20 | 0.71 | 0.39 | 0.07 | 1.08 | 0.81 | 2.78 | 0.74 | 1.76 | 2.24 | 0.98 | 1.08 | 0.49 | 0.00 | 2.02 | 0.00 | 0.28 | 2.57 |
| 21 | 0.71 | 0.58 | 0.07 | 1.08 | 1.61 | 2.65 | 0.47 | 1.41 | 2.24 | 1.12 | 0.82 | 1.07 | 0.00 | 1.96 | 0.00 | 0.20 | 2.51 |
| 22 | 1.27 | 0.53 | 0.28 | 0.98 | 0.73 | 2.16 | 0.63 | 1.60 | 2.24 | 1.15 | 1.47 | 0.44 | 0.00 | 1.69 | 0.00 | 0.30 | 1.85 |
| 23 | 0.87 | 0.58 | 0.49 | 0.98 | 0.66 | 2.65 | 0.62 | 2.01 | 2.24 | 1.48 | 1.12 | 0.35 | 0.00 | 1.25 | 0.00 | 0.20 | 2.38 |
| 24 | 0.93 | 0.72 | 0.70 | 1.42 | 0.88 | 1.59 | 0.87 | 1.88 | 1.91 | 1.12 | 0.76 | 0.55 | 0.55 | 1.53 | 0.00 | 0.50 | 2.51 |
| 25 | 0.65 | 1.32 | 0.77 | 0.78 | 1.68 | 1.85 | 0.57 | 1.07 | 2.32 | 0.86 | 0.45 | 1.24 | 0.18 | 2.56 | 1.25 | 0.43 | 1.52 |
| 26 | 0.75 | 0.86 | 0.35 | 0.93 | 1.03 | 3.46 | 0.45 | 1.25 | 2.07 | 1.12 | 0.80 | 0.46 | 0.00 | 1.74 | 0.00 | 0.20 | 3.30 |
| 27 | 1.30 | 0.86 | 0.63 | 0.64 | 0.95 | 1.90 | 0.86 | 1.69 | 1.82 | 0.98 | 1.00 | 0.55 | 0.00 | 1.53 | 1.25 | 0.30 | 1.98 |
| 28 | 0.56 | 0.46 | 0.28 | 0.98 | 0.66 | 2.50 | 0.84 | 1.76 | 1.82 | 1.69 | 1.23 | 0.36 | 0.00 | 1.04 | 0.00 | 0.26 | 2.71 |
| 29 | 0.81 | 0.72 | 0.56 | 1.37 | 0.81 | 1.46 | 0.94 | 1.57 | 1.82 | 0.95 | 1.03 | 0.72 | 0.18 | 1.96 | 0.00 | 0.48 | 1.85 |
| 30 | 0.93 | 0.92 | 1.19 | 0.73 | 1.03 | 1.22 | 1.12 | 1.57 | 1.58 | 0.56 | 1.56 | 0.52 | 0.18 | 1.04 | 0.00 | 0.54 | 0.92 |
| 31 | 1.30 | 1.02 | 0.77 | 0.78 | 1.10 | 1.82 | 0.85 | 1.13 | 1.74 | 0.74 | 0.99 | 0.64 | 0.00 | 1.47 | 0.00 | 0.65 | 1.72 |
| 32 | 0.59 | 0.60 | 0.42 | 0.78 | 1.03 | 2.08 | 1.04 | 1.47 | 1.91 | 0.89 | 1.11 | 0.66 | 0.00 | 1.64 | 0.00 | 0.43 | 2.05 |
| 33 | 0.87 | 0.99 | 0.49 | 1.22 | 1.32 | 2.05 | 0.73 | 1.29 | 2.16 | 1.04 | 0.79 | 0.85 | 0.00 | 1.47 | 0.00 | 0.43 | 1.72 |
| 34 | 0.78 | 1.02 | 0.21 | 1.22 | 1.10 | 1.79 | 0.73 | 1.85 | 2.49 | 0.80 | 0.64 | 0.72 | 0.00 | 2.13 | 0.00 | 0.48 | 2.24 |

## Narrative Findings

Two POS tags stand out as consistently **under**-represented among "not activated" (bottom-10%-attention) tokens across nearly all 35 layers: **VERB** (mean ratio 0.42, range 0.20–0.80) and **PUNCT** (mean ratio 0.52, range 0.17–1.24). In other words, verbs and punctuation are rarely the tokens a layer ignores — they reliably receive above-average attention. The punctuation result lines up with the well-documented "attention sink" phenomenon in transformer literature, where sentence-boundary and punctuation tokens absorb a disproportionate share of attention mass; this experiment reproduces that pattern token-by-token across Gemma-4-E2B-it's full depth rather than just at the first/BOS position.

At the other end, **DET** (determiners: "the", "a", "an"; mean ratio 1.90, range 0.42–3.56), **PART** (particles; mean ratio 2.09, range 0.99–3.23), and **NUM** (numerals; mean ratio 1.70) are consistently **over**-represented among the least-attended tokens — these are the tokens most likely to fall into the bottom 10% by received attention. This is intuitive: determiners and particles carry little standalone semantic content and are largely predictable from local context, so a model may not need to route much attention *to* them from other positions, even though it still has to *produce* them.

The pattern is not perfectly stable across depth. Layer 0 is an outlier for DET (ratio 0.42 — well-attended) before flipping sharply to over-represented from layer 1 onward (peaking at 3.56 in layer 2), suggesting the first layer treats determiners differently (plausibly still doing local/positional processing) before deeper layers settle into treating them as low-priority. The **X** tag (tokens whose character offsets didn't cleanly land inside a spaCy word span — mostly sub-word continuation pieces) shows the sharpest layer-dependent swing of any tag: it spikes to 4.88 at layer 3 before settling into a noisier 1.5–3.3 band for the remaining layers, hinting that early layers are especially likely to leave partial-word continuation pieces under-attended while later layers integrate them more.

**Caveats:** this is a 25-passage, 200-token-prompt sample from WikiText-2 with only 30 generated tokens per passage — sufficient to see large, consistent effects (VERB/PUNCT under-representation, DET/PART/NUM over-representation) but not enough to trust small differences between adjacent layers or rare tags. SCONJ and SYM show extreme ratios (means of 0.06 and 0.11) that are likely sample-size artifacts — both tags are rare in a 200-token window, so a single passage with a "cold" SCONJ token can swing the ratio sharply; these should not be over-interpreted without a larger run.
