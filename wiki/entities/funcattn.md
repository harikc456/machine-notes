---
title: Functional Attention (FuncAttn)
created: 2026-07-06
updated: 2026-07-06
type: entity
tags: [attention, architecture]
sources: [raw/papers/2605.31559v1.pdf]
confidence: medium
---

# Functional Attention (FuncAttn)

*Xiao, Gao, Weber, Yang, Cremers (TU Munich / Oxford / UT Austin), ICML 2026*

## Core Claim

Reinterprets attention as a **functional correspondence between adaptive bases**, rather than a
pointwise similarity between tokens — inspired by the functional-maps framework from geometry
processing. Built for operator learning (learning mappings between infinite-dimensional function
spaces: PDE solving, 3D segmentation, regression), not for language modeling.

## Method

Learns adaptive bases Φ, Ψ via feed-forward networks, projects Q/K/V into a k-dimensional learned
spectral space, and computes attention as **least-squares regression** in that space rather than
softmax similarity:

```
FuncAttn(Q,K,V) = Φ (Q̃K̃ᵀK̃K̃ᵀ + λI_k)⁻¹ Ṽ
```

A Tikhonov (ridge) term λ gives a provable local Lipschitz continuity bound — a stability guarantee
standard softmax attention lacks.

## Results

SOTA or near-SOTA on 6 PDE benchmarks (Elasticity, Airfoil, Darcy, Pipe, Navier-Stokes,
Plasticity) against FNO-family and Transolver baselines; up to 3 orders of magnitude lower MSE than
vanilla attention in few-shot sinusoidal regression; also validated on 3D point-cloud segmentation
and cross-resolution generalization.

## Relation to Existing Work

Distinguished from linear-attention approximations (Linformer, Performer, Nyströmformer) — those
approximate standard softmax attention; FuncAttn solves a different least-squares problem in a
learned spectral space instead. Extends the Galerkin Transformer's view of attention as a
functional operator by explicitly learning bases rather than using an implicit basis change, and
generalizes Transolver's "slice-and-attend" tokens with a broader spectral framework.

## Domain Note

Centered on scientific ML / operator learning rather than LLM inference or language modeling —
adjacent to this wiki's core focus via the attention-mechanism-variant angle, but not deeply
cross-linked into KV-cache or speculative-decoding concept pages. Filed here as a lightweight
reference for anyone tracking novel attention formulations broadly.

## See Also

- [[flash-attention]] — standard tiled softmax attention this reinterprets away from
- [[qknorm]] — another attention-formulation variant, though for LLM stability rather than
  operator learning
