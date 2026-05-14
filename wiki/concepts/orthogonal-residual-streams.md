---
title: Orthogonal Residual Streams
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [architecture, attention, residual, training]
sources: [raw/papers/2603.09078v1.pdf, raw/papers/2512.24880v2.pdf, raw/papers/2601.05732v1.pdf, raw/papers/2601.21579v1.pdf]
confidence: high
---

# Orthogonal Residual Streams

## Core Insight

A recurring theme across recent architecture papers: **modules should write information that is orthogonal to what the residual stream already contains**. When a module writes *along* an existing direction, it wastes capacity by recycling information rather than adding new signal.

Three independent lines of work instantiate this idea at different levels:

| Module | What it constrains | Method | Paper |
|---|---|---|---|
| Attention | Output ⊥ own value vector | Gram-Schmidt subtraction | [[xsa]] |
| FFN | Output ⊥ pre-norm input | Gram-Schmidt projection | `OrthogonalMLPWrapper` (rbf_ffn) |
| Residual mixing | Residual matrices are doubly-stochastic | Birkhoff polytope constraint | [[mhc]] / [[mhc-lite]] / [[kromhc]] |

## The Attention Similarity Bias (XSA)

Standard SA tends to write attention outputs similar to the current token's own value vector — re-aggregating what the token already encodes. [[xsa]] removes this by subtracting the component along `v_i`:

```
z_i = y_i − proj_{v_i}(y_i)
```

Motivation: SA's role is contextual aggregation; point-wise transformation is the FFN's role. Writing along the own-value direction competes with this division of labor.

## FFN Residual Projection (OrthogonalMLPWrapper)

The FFN analogue of XSA, studied in `rbf_ffn/findings.md §8.5, §8.7, §10`:

```
y      = mlp(x)
out    = y − proj_x(y)   # project out component along pre-norm input
```

This forces the FFN to add information not already present in the input token representation. Key empirical findings:
- **+1.7 ppl** over XSA+qknorm without wnorm (§8.5)
- **+1.3 ppl** over XSA+qknorm with wnorm (§8.9)
- **No overhead** — single dot product per position, no new parameters
- **Selective application beats global** (§10.2): applying only to layers 1,3 (middle odd) achieves 54.97 vs 55.57 for all layers
- **Last layer should NOT be wrapped** (§10.3): blocks.5 has high alignment (0.36) but wrapping it is harmful — the last layer intentionally writes along the residual stream direction before the LM head

## Hyper-Connections: Doubly-Stochastic Residual Matrices

[[mhc]] and its successors constrain the *residual mixing matrix* `H^res` to the Birkhoff polytope (doubly-stochastic matrices). This is orthogonality at the level of the multi-stream residual topology:
- Doubly-stochastic matrices have spectral norm ≤ 1 → prevents gradient explosion
- The constraint restores the identity mapping property of standard residuals
- Not the same as Gram-Schmidt orthogonality, but shares the goal of preventing redundant writes through the depth of the network

## Why Orthogonality Helps

1. **Division of labor**: each module can specialize in a distinct subspace. Redundant writes between modules prevent specialization.
2. **Gradient flow**: writing orthogonally to the input creates stronger gradient signal (orthogonal updates have maximum distance from the identity-shortcut path).
3. **Effective depth**: if module outputs are correlated with inputs, the effective depth of computation is reduced — the network is re-doing earlier work.
4. **Inductive bias for composition**: compositional tasks benefit from representations that incrementally add new structure rather than reprocessing existing structure.

## Selective Orthogonality: Probe Results

The alignment probe (§10.3 of findings.md) on a trained 6-layer model:

| Layer | Wrapped | Mean cos sim | Interpretation |
|---|---|---|---|
| blocks.0 | no | 0.040 | First layer writes ⊥ naturally — wrapping useless |
| blocks.1 | yes | **0.411** | Highest middle-layer alignment — wrapping most beneficial |
| blocks.2 | no | 0.176 | Low alignment — wrapping would add little |
| blocks.3 | yes | 0.320 | Second most aligned wrapped layer |
| blocks.4 | no | 0.292 | Next candidate to wrap (§10.3 experiment pending) |
| blocks.5 | no | 0.363 | High alignment but *intentional* — last layer writes toward LM head by design |

**Key lesson**: orthogonal constraint has a sweet spot. Too few = missed inductive bias. Too many = over-constrains layers with legitimate reasons to write along the residual direction. The last layer is the clearest exception.

## Open Questions

- Does selective wrapping (layers 1,3,4) compound the gain, or does adding blocks.4 regress?
- Is the "intentional alignment" of the last layer a universal property, or task/architecture dependent?
- Can the alignment probe guide initialization (wrap only high-alignment layers from the start)?
- How does XSA's value-direction constraint interact with the input-direction FFN constraint — are these addressing the same or different redundancy?

## See Also

- [[xsa]] — attention orthogonality paper
- [[hyper-connections]] — residual matrix doubly-stochastic constraint
- [[mhc]] · [[mhc-lite]] · [[kromhc]] — specific implementations
- [[weight-norm-training]] — complementary normalization perspective
