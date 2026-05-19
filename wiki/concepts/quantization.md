---
title: Quantization
created: 2026-05-14
updated: 2026-05-19
type: concept
tags: [quantization, inference, training]
sources: [raw/papers/A Visual Guide to Quantization.md, raw/papers/2502.02617v1.pdf, raw/papers/2504.19874v1.pdf]
confidence: high
---

# LLM Quantization

## Overview

Quantization reduces the numerical precision of model parameters (and/or activations) to decrease memory footprint and increase computational throughput. Core trade-off: **compression ratio vs. task accuracy**.

## Data Types

| Type | Bits | Use Case |
|---|---|---|
| FP32 | 32 | Full precision training |
| BF16 | 16 | Mixed-precision training/inference |
| FP16 | 16 | Standard inference |
| INT8 | 8 | Post-training quantization (PTQ) |
| FP8 | 8 | Training + inference (emerging) |
| INT4/NF4 | 4 | Aggressive weight compression |
| INT2/1-bit | 2/1 | Extreme compression (BitNet etc.) |

A floating-point number = sign bit + exponent bits + mantissa (fraction) bits.
**BF16** has the same exponent range as FP32 (8 exponent bits) but less precision — better for ML than FP16 which clips large values.

## Categories

### Weight Quantization
Quantize model weights; activations remain in higher precision:
- **Post-Training Quantization (PTQ)**: quantize after training, no retraining
- **Quantization-Aware Training (QAT)**: train with quantization in the loop
- Key challenge: outlier weights — a few large-magnitude weights cause disproportionate quantization error

### KV Cache Quantization
Quantize the K/V tensors stored during inference:
- [[polarquant]]: polar coordinate transform, no normalization needed
- [[turboquant]]: random rotation + MSE quantizer + QJL residual; data-oblivious, near-optimal within that class
- [[spectralquant]]: calibrated eigenvector rotation + selective QJL on signal dims only; exploits 97% spectral gap in KV keys; 15s calibration; strictly better than TurboQuant (+1.7–2.8 pp, 18.6% better compression)
- Challenge: per-block normalization parameters add overhead — addressed by all three via rotation preconditioning

### Activation Quantization
Quantize activations (inputs/outputs of layers) during inference — most challenging due to outliers.

## Key Concepts

### Calibration
Most PTQ methods require a small calibration dataset to determine quantization ranges. Data-oblivious methods ([[turboquant]], QJL) avoid this entirely. Data-aware methods ([[spectralquant]]: 15s on one GPU) trade minimal calibration for meaningful quality gains by exploiting structural properties invisible to oblivious methods.

### Symmetric vs Asymmetric
- Symmetric: quantization range is [-max, max] — zero point = 0, one parameter (scale)
- Asymmetric: range is [min, max] — two parameters (scale + zero point); the extra storage overhead is the "memory overhead" problem addressed by PolarQuant/TurboQuant

### Block Quantization
Group elements into blocks; normalize each block independently. The scale/zero-point stored in FP32 per block adds overhead. At 1 byte per quantized value, storing even 1 FP32 scale per 8 values adds 0.5 bytes — >50% overhead.

## The Random Preconditioning Insight

[[polarquant]], [[turboquant]], and [[spectralquant]] all use rotation preconditioning before quantization. PolarQuant and TurboQuant use random rotation (oblivious); SpectralQuant uses calibrated eigenvector rotation. The calibrated version is strictly better: it aligns the coordinate axes with the actual signal/noise structure, enabling selective error correction that random rotation cannot.

## The Spectral Gap Insight (SpectralQuant)

KV cache key vectors exhibit d_eff ≈ 3–4% of head dimension as effective dimensionality — universally across model families. This means 96–97% of dimensions carry only noise. Applying error correction (QJL) uniformly to all dims, as TurboQuant does, worsens MSE on noise dims (variance added exceeds bias reduced). Selectively applying correction only to signal dims strictly improves quality *and* compression simultaneously.

## See Also

- [[kv-cache]] — KV cache quantization specifically
- [[polarquant]] — polar quantization for KV
- [[turboquant]] — vector quantization for KV
- [[spectralquant]] — calibrated spectral quantization for KV
- [[h2o]] — complementary eviction approach
- [[kv-cache-compression-comparison]] — methods compared
