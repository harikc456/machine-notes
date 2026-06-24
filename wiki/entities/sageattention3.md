---
title: SageAttention3
created: 2026-06-24
updated: 2026-06-24
type: entity
tags: [quantization, attention, inference, training]
sources: [raw/papers/2505.11594v3.md]
confidence: high
---

# SageAttention3

**SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training**  
*Jintao Zhang, Jia Wei, Haoxu Wang, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Kai Jiang, Jianfei Chen, Jun Zhu — Tsinghua University / Shengshu Tech, NeurIPS 2025*

## Overview

SageAttention3 makes two independent contributions: FP4 microscaling for inference (hardware-first, requires Blackwell GPUs) and 8-bit backward-pass attention for training.

## Contribution 1: FP4 Microscaling Inference

### Hardware Context

NVIDIA Blackwell GPUs (RTX5090, GB200) introduced FP4 Tensor Cores (SM100). SageAttention3 is the first work to apply FP4 to attention computation.

### Data Type: NVFP4

NVFP4 uses E2M1 encoding (2 exponent bits, 1 mantissa bit) with 1×16 quantization group size. Scale factors are stored in E4M3 format. The 15 representable non-zero values make FP4 significantly more sensitive to outliers than INT4.

### Three Challenges and Solutions

**C1 — Per-tensor/per-token quantization inadequate for FP4**:  
Outliers make the dynamic range too wide for 15 values. Fix: use 1×16 micro-group quantization (scale per 16-element block), confining outlier effects.

**C2 — P̃ matrix quantization inaccuracy**:  
P̃ values fall in [0, 1]; the FP4 scale factor in E4M3 format ends up in [0, 0.167] — poorly utilizing the representable range.  
Fix: two-level quantization:
1. Per-token normalize: `s_P1 = rowmax(P̃) / (448 × 6)`, scale P̃ to `[0, 448×6]`
2. Apply FP4 microscaling to the scaled P̃
The output Matmul then corrects by `× s_P1`. This fully utilizes E4M3 range.

**C3 — Gradient accumulation in backward pass**:  
For training (SageBwd), the most accuracy-sensitive backward Matmul is kept in FP16.

### Results

- 1038 TOPS on RTX5090 — **5× FlashAttention2** on the same hardware
- Near-lossless end-to-end metrics on image/video generation

## Contribution 2: 8-bit Attention Training (SageBwd)

Extends 8-bit attention to the backward pass — the first trainable low-bit attention implementation.

Among the 5 backward Matmuls in attention, identifies the most accuracy-sensitive one (attention gradient w.r.t. attention map) and keeps it in FP16. Remaining 4 Matmuls use 8-bit.

**Results**:
- **Fine-tuning**: lossless convergence (instruction-following base models)
- **Pretraining**: slower convergence — not yet recommended for pretraining workloads

## Relationship to Other Work

- Third in the SageAttention series: [[sageattention]] (INT8) → [[sageattention2]] (INT4/FP8) → SageAttention3 (FP4)
- Inherits K-smoothing and Q+K-smoothing from SA1/SA2
- FP4 inference is Blackwell-only; [[sageattention2]] remains the practical choice for Ampere/Ada/Hopper

## See Also

- [[sageattention]] — first-generation INT8 quantized attention
- [[sageattention2]] — INT4/FP8; practical for current (non-Blackwell) hardware
- [[flash-attention]] — FlashAttention-3 and -4 use similar hardware-first design philosophy
- [[quantization]] — full quantization landscape including FP4 data type
