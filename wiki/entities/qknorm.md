---
title: QKNorm (Query-Key Normalization)
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [normalization, attention, training]
sources: [raw/papers/2010.04245v1.pdf]
confidence: high
---

# QKNorm — Query-Key Normalization

**Query-Key Normalization for Transformers**
*Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, Yuxuan Chen, Cyndx Technologies, arXiv:2010.04245, Oct 2020*

## Core Idea

Replace the dot-product similarity `q·k` in attention with **cosine similarity** scaled by a learnable parameter:

```
q̂_i = q_i / ||q_i||        (ℓ₂ normalize along head dimension)
k̂_j = k_j / ||k_j||

attention_score(i,j) = (q̂_i · k̂_j) · scale   # scale is learnable, init = 1/√d
```

Applied **after** multihead attention splits its input into heads, along the head dimension. Q and K only — V is unchanged.

## Motivation: Softmax Saturation

The dot product `q·k` is unbounded. When absolute differences between logits are large (e.g., [760, 752, 750]), softmax collapses to near-argmax even when relative differences are small (only 8 apart from 750 to 760). This limits the diversity of attention patterns heads can learn.

Cosine similarity bounds the pre-softmax range to `[-scale, +scale]`, making the input to softmax *bounded* and easier to learn from. The learnable `scale` parameter allows the model to recover the original dynamic range if needed.

## Difference from ScaleNorm

ScaleNorm (Nguyen & Salazar 2019) applies ℓ₂ normalization along the **embedding dimension** and before the input is split into heads. QKNorm applies along the **head dimension**, after splitting, and only to Q and K (not V). QKNorm complements LayerNorm rather than replacing it.

## Results

+0.928 BLEU average over SOTA bilingual benchmarks on 5 low-resource translation pairs from TED Talks and IWSLT'15.

## Relationship to rbf_ffn Experiments

QK norm (`qk_norm: true`) is used throughout §6–§10 of `findings.md`. Key empirical findings:
- **Adds ~0.5–1 ppl improvement** consistently on top of weight norm (§6.1: 58.97 → 58.16 with SwiGLU+wnorm)
- **Low cost**: ~1307s vs ~1223s/epoch vs baseline — negligible overhead
- **Consistent but secondary**: weight norm dominates; QK norm is a reliable +0.5–1 ppl add-on
- **Run-to-run variance is low** for XSA+qknorm: spread of 0.82 ppl across 3 runs (§8.2)

## See Also

- [[weight-normalization]] — the dominant normalization technique in the same experimental setting
- [[xsa]] — XSA is typically paired with QKNorm in experiments
- [[weight-norm-training]] — synthesis of normalization techniques
- [[hyper-connections]] — mHC also addresses training stability via a different (residual matrix) mechanism
