---
title: Grokking
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [grokking, training, normalization]
sources: [raw/papers/2201.02177v1.pdf, raw/papers/cliptogrok.pdf]
confidence: high
---

# Grokking

## Definition

Grokking is the phenomenon where a neural network **generalizes long after memorizing** training data. The model first achieves near-zero training loss while validation accuracy remains near chance; then, after thousands more optimization steps, validation accuracy rapidly improves to near-perfect.

First documented by **Power, Burda, Edwards, Babuschkin (OpenAI) & Misra (Google), 2022** on small algorithmically-generated datasets. ^[raw/papers/2201.02177v1.pdf]

## Original Paper: Key Findings (Power et al. 2022)

**Setting**: Small transformer trained on binary operation tables `a ∘ b = c` (modular arithmetic, permutation groups, bivariate polynomials). Dataset = subset of all possible equations. Network sees abstract symbols — no decimal or permutation notation.

**Observations**: ^[raw/papers/2201.02177v1.pdf]
- Training accuracy reaches ~100% at ~10³ steps; validation accuracy stays near chance
- Validation accuracy eventually jumps to near-perfect, but only at ~10⁶ steps (1000× later)
- The delay is more pronounced with smaller datasets — optimization required for generalization scales inversely with data fraction
- **Weight decay** was found to be "particularly effective at improving generalization" — the earliest hint that norm regularization is key

**Data efficiency curves**: Smaller datasets require disproportionately more optimization steps to generalize, but the *ceiling* of generalization (once achieved) is the same.

**Embedding analysis**: Learned symbol embeddings sometimes recover recognizable mathematical structure (e.g., Fourier-like representations of modular arithmetic groups).

## Why It Happens: Weight Norm Dynamics

Subsequent work converged on a central mechanism: **grokking delay is controlled by weight norm dynamics**.

- **Liu et al. (2023) — Goldilocks zone**: Generalization concentrates in a narrow spherical shell in weight space. Constraining norms to this zone nearly eliminates grokking delay.
- **Lee et al. (2024)**: Amplifying slow-varying gradient components accelerates generalization >50×.
- **Prieto et al. (2025)**: Unconstrained weight growth → Softmax Collapse (logit explosion) → generalization stalls entirely.

All lines converge: **controlling weight magnitude is the key to fast generalization**.

## Clip to Grok Intervention

[[clip-to-grok]] shows that per-row weight norm clipping on decoder layers (with `max_norm ≈ 2.0`) achieves 39–249× grokking speedup. The mechanism:
1. Clipping keeps norms near the Goldilocks zone
2. Prevents Softmax Collapse (logit growth)
3. Works with sign-based optimizers (Lion, SignSGD) which naturally complement ℓ₂ clipping

## Connection to Language Model Training

The Goldilocks zone insight generalizes beyond grokking:
- [[weight-normalization]] applies a full bidirectional norm constraint and provides ~21 ppl improvement in autoregressive LM (rbf_ffn §6.1)
- The `max_only` (clipping-only) mode regresses catastrophically in LM — suggesting the Goldilocks zone for LM requires *both* shrinking and scaling, unlike grokking tasks where only preventing growth matters

## Open Questions

- Is there a grokking-like delayed generalization phase in large-scale LM training?
- Does the Goldilocks zone concept generalize to the attention similarity bias addressed by [[xsa]]?
- Can edge initialization (normalizing only boundary layers) improve LM training as it does grokking?

## See Also

- [[clip-to-grok]] — the key paper on weight clipping for grokking
- [[weight-normalization]] — full bidirectional norm reparameterization
- [[weight-norm-training]] — practical normalization synthesis for transformers
