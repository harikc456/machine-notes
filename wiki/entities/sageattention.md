---
title: SageAttention
created: 2026-06-24
updated: 2026-06-24
type: entity
tags: [quantization, attention, inference]
sources: [raw/papers/2410.02367v9.md]
confidence: high
---

# SageAttention

**SageAttention: Accurate 8-bit Attention for Plug-and-Play Inference Acceleration**  
*Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen — Tsinghua University, ICLR 2025*

## Overview

SageAttention is a drop-in quantized attention kernel that replaces the standard FP16 attention computation with an 8-bit variant. The key observation is that existing quantization work targets linear layers; attention — which dominates compute at long sequences due to its O(N²) complexity — was left in full precision. SageAttention is the first to make this practical at negligible accuracy cost across diverse model types.

## Core Approach

Built on FlashAttention-2's tiling structure. Quantizes the two attention Matmuls:
- **QKᵀ Matmul**: Q and K quantized to **INT8** (per-block, FlashAttention tile-aligned scale)
- **P̃V Matmul**: P (softmax output) and V kept in **FP16** with **FP16 accumulator**

The FP16 Matmul (mma.f16.f16.f16) is 2× faster than FP16 with FP32 accumulator on RTX4090/3090 GPUs — this is a hardware property that SageAttention explicitly exploits.

### Challenge C1: K Channel-Wise Outliers

K matrices exhibit significant channel-wise outliers (a small number of channels have much larger magnitude than others). Naively quantizing INT8 per-block gives poor accuracy.

**Fix**: Subtract the per-channel mean of K before quantization (K-smoothing). The mean is a 1×D vector; subtracting it doesn't change the softmax output (adding a constant to each row of S doesn't change softmax). The smoothed K has much smaller outlier ratio.

### Challenge C2: P̃V Accuracy

Simply quantizing P,V to INT8 causes accuracy collapse. SageAttention avoids this by keeping P,V in FP16 with a FP16-capable fast accumulator — the V tensor is never quantized.

## Performance

| Metric | SageAttention | FlashAttention2 | xformers |
|---|---|---|---|
| Kernel OPS (TOPS, RTX4090 headdim=64) | 340 | 164 | 123 |
| Kernel speedup over FA2 | 2.1× | 1× | — |
| End-to-end latency loss | ~0% | baseline | — |

**Models validated**: LLaMA, Mistral (LM); Stable Diffusion, Unidiffuser (image); CogVideoX (video). No observable end-to-end quality degradation on any.

## Relationship to Other Work

- Builds on [[flash-attention]] tiling; adds quantization on top
- Succeeded by [[sageattention2]] (INT4 Q/K + FP8 P) and [[sageattention3]] (FP4 microscaling)
- Distinct from [[kv-cache]] quantization methods ([[polarquant]], [[turboquant]], [[spectralquant]]) — those compress *stored* KV tensors; SageAttention quantizes *compute* within the attention forward pass

## See Also

- [[sageattention2]] — INT4/FP8 extension; 3× FA2, 4.5× xformers
- [[sageattention3]] — FP4 microscaling; 5× FlashAttention on RTX5090
- [[flash-attention]] — IO-aware tiled attention that SageAttention builds on
- [[quantization]] — general quantization landscape
- [[kv-cache]] — KV storage compression (orthogonal to compute quantization)
