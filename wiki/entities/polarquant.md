---
title: PolarQuant
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [kv-cache, quantization, inference]
sources: [raw/papers/2502.02617v1.pdf]
confidence: high
---

# PolarQuant

**PolarQuant: Quantizing KV Caches with Polar Transformation**
*Insu Han (KAIST), Praneeth Kacham, Vahab Mirrokni, Amir Zandieh (Google Research), Amin Karbasi (Yale), arXiv:2502.02617, Feb 2025*

## Overview

PolarQuant is a KV cache quantization method that transforms embeddings into **polar coordinates** before quantizing. The key insight: after random preconditioning, the angles in the polar representation exhibit a **tightly bounded, concentrated distribution** with a known analytic form — which **eliminates the need for per-block normalization constants**.

See [[kv-cache]] for background and [[quantization]] for general quantization context.

## Problem with Traditional KV Quantization

Standard KV quantization groups data into blocks (channel-wise or token-wise) and independently normalizes each block — requiring storage of zero-point and scale in full precision (FP32). This overhead can add **>1 additional bit per quantized number**, negating memory savings.

## Technical Approach

1. **Random preconditioning**: Apply a random Hadamard matrix to the KV embeddings
2. **Polar transformation**: Convert to polar coordinates via a recursive algorithm
3. **Quantize angles**: The preconditioned distribution is analytically known — no normalization needed
4. **Decode**: Reverse polar transform at attention computation time

### Why Angles Concentrate After Preconditioning
Random preconditioning (Hadamard matrix) makes each coordinate approximately i.i.d. Gaussian, whose polar angles exhibit a Beta-like distribution that is tightly bounded and analytically characterized. This eliminates the data-dependent normalization step.

## Results

- **>4.2× KV cache compression** while achieving best quality scores vs. state-of-the-art
- Outperforms QJL, Lexico, and other methods on long-context evaluations
- Competitive with [[turboquant]] (different design space)

## Relationship to Other Work

- [[h2o]]: eviction-based (orthogonal approach)
- [[turboquant]]: also uses random rotation; different compression target (vector quantization vs polar)
- [[kv-cache-compression-comparison]]: side-by-side comparison
- [[quantization]]: general quantization landscape
