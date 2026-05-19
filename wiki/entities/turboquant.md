---
title: TurboQuant
created: 2026-05-14
updated: 2026-05-19
type: entity
tags: [kv-cache, quantization, inference]
sources: [raw/papers/2504.19874v1.pdf]
confidence: high
---

# TurboQuant

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
*Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research), arXiv:2504.19874, Apr 2025*

## Overview

TurboQuant is an **online vector quantization** scheme targeting near-optimal distortion rate for both MSE and inner product preservation — two objectives that previous methods traded off against each other. It's data-oblivious (no tuning/calibration required) and accelerator-friendly.

Primary application: [[kv-cache]] quantization. Secondary: nearest neighbor search / vector database compression.

## Core Technical Idea

Two-stage approach:

1. **MSE-optimal quantizer**: Randomly rotate input vectors (inducing a concentrated Beta distribution on coordinates), then apply optimal scalar quantizers per coordinate. This leverages near-independence of high-dimensional rotated coordinates.

2. **1-bit QJL residual**: Apply a 1-bit Quantized Johnson-Lindenstrauss transform to the residual, creating an unbiased inner product estimator. MSE-optimal quantizers introduce bias in inner product estimation — QJL residual corrects this.

### Why Rotation Helps
Random rotation makes each coordinate approximately i.i.d. Beta-distributed in high dimensions. This concentration property allows per-coordinate scalar quantization to achieve near-optimal joint distortion without modeling cross-coordinate dependencies.

### Theoretical Guarantees
- **Formal proof** of information-theoretic lower bounds on distortion rate
- TurboQuant achieves these bounds within a constant factor of ≈2.7×
- Data-oblivious: no calibration, adapts online

## Results

- **KV cache**: quality neutrality at 3.5 bits/channel; marginal degradation at 2.5 bits/channel
- **Nearest neighbor search**: outperforms product quantization in recall while reducing indexing time to near-zero
- Accelerator-friendly (no custom kernels required)

## Relationship to Other Work

- [[polarquant]]: also uses random rotation; targets polar coordinate quantization specifically for KV cache; different formalism
- [[h2o]]: eviction-based (complementary rather than competing)
- [[kv-cache-compression-comparison]]: side-by-side comparison
- [[quantization]]: general quantization landscape
- [[spectralquant]]: data-aware successor (Apr 2026) that breaks TurboQuant's data-oblivious bound — calibrated eigenvector rotation + selective QJL on signal dims only achieves +1.7–2.8 pp cosine similarity at 18.6% better compression; 15s calibration cost

Note: TurboQuant and PolarQuant share an author (Zandieh, Mirrokni from Google Research) and both use random preconditioning — they likely represent parallel explorations of the same fundamental idea. TurboQuant's theoretical guarantee is tight within the data-oblivious class; SpectralQuant shows that spending 15s on calibration decisively breaks the bound.
