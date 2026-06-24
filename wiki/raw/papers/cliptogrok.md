---
source_url: N/A (local PDF, no arXiv ID detected)
ingested: 2026-06-24
sha256: N/A (pdf-derived summary)
---

# Clip to Grok: Weight Norm Clipping for Accelerated Generalization

**Authors:** Vladimir Volchkov, Aviad Rivlin  
**Affiliation:** Independent Researchers  
**Venue:** arXiv preprint  

## Summary

Demonstrates that per-row weight norm clipping on decoder layers accelerates grokking by 39–249× across six algebraic tasks (four modular arithmetic operations, composition, S₅ permutation), while eliminating the need for weight decay. The method requires a few lines of code and no additional memory overhead.

## Method

After each optimizer step, project every weight row in the decoder layers onto the ℓ₂ ball of radius `c`:

    w_row ← w_row · min(1, c / ‖w_row‖₂)

where `c = max_norm` (default 2.0 for modular multiplication). Applied only to attention projections, MLP layers, and LayerNorm parameters — not token embeddings or output head. No weight decay used.

## Connection to Grokking Dynamics

Analysis connects weight norm clipping to four frameworks:
1. **Omnigrok's Goldilocks zone:** Clipping collapses norms to `w ≈ w_c`, the optimal generalization zone
2. **α^L depth scaling law:** Motivates edge initialization strategy for deep models (normalize only embeddings, final LayerNorm, output head)
3. **Sign-based optimizers:** Lion and SignSGD naturally complement norm clipping; Lion+Clip achieves fastest convergence
4. **Softmax Collapse prevention:** Clipping eliminates the post-memorization logit growth that stalls standard training

## Results

- **39–249× acceleration** in grokking across six algebraic tasks
- **66×** on modular multiplication (2-layer), **18×** on 8-layer architectures (1.6M params)
- Zero failures across 300 runs on 8-layer architectures
- Edge initialization reduces interquartile range by 61–72%
- Lion+Clip: fastest convergence; stabilizes Lion's typically narrow LR tolerance
- Observation: GrokFast's gradient EMA filter approximates sign-based optimizers via Adam's second-moment normalization

## Relevance to Wiki

Primary source for [[clip-to-grok]] entity page. See [[grokking]] (foundational phenomenon: Power et al. 2022), [[weight-normalization]] (weight norm tools), [[weight-norm-training]] (interaction with training dynamics).
