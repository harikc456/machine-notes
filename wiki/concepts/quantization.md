---
title: Quantization
created: 2026-05-14
updated: 2026-05-14
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
- [[turboquant]]: random rotation + MSE quantizer + QJL residual
- Challenge: per-block normalization parameters add overhead — addressed by both PolarQuant and TurboQuant via random preconditioning

### Activation Quantization
Quantize activations (inputs/outputs of layers) during inference — most challenging due to outliers.

## Key Concepts

### Calibration
Most PTQ methods require a small calibration dataset to determine quantization ranges. Data-oblivious methods ([[turboquant]], QJL) avoid this.

### Symmetric vs Asymmetric
- Symmetric: quantization range is [-max, max] — zero point = 0, one parameter (scale)
- Asymmetric: range is [min, max] — two parameters (scale + zero point); the extra storage overhead is the "memory overhead" problem addressed by PolarQuant/TurboQuant

### Block Quantization
Group elements into blocks; normalize each block independently. The scale/zero-point stored in FP32 per block adds overhead. At 1 byte per quantized value, storing even 1 FP32 scale per 8 values adds 0.5 bytes — >50% overhead.

## The Random Preconditioning Insight

Both [[polarquant]] and [[turboquant]] use a random (Hadamard) matrix to precondition vectors before quantization. After preconditioning, coordinates become approximately i.i.d. — their distribution is analytically characterized, eliminating the need for data-dependent normalization. This is the key insight that makes both methods normalization-free.

## See Also

- [[kv-cache]] — KV cache quantization specifically
- [[polarquant]] — polar quantization for KV
- [[turboquant]] — vector quantization for KV
- [[h2o]] — complementary eviction approach
- [[kv-cache-compression-comparison]] — methods compared
