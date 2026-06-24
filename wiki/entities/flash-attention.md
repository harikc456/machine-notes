---
title: Flash Attention
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [inference, attention, architecture, training]
sources: [raw/papers/2205.14135v2.pdf]
confidence: high
---

# Flash Attention

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
*Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, Christopher Ré — Stanford, arXiv:2205.14135, 2022*

*FlashAttention-2 (arXiv:2307.08691, 2023), FlashAttention-3 (arXiv:2407.08608, 2024), FlashAttention-4 (2025, Blackwell-first)*

## The Core Insight: IO is the Bottleneck

Standard attention has HBM IO complexity O(N²d) — for every attention step, the full N×N attention matrix is written to and read from GPU HBM (high-bandwidth memory). This makes attention **IO-bound**, not compute-bound, especially at long sequences.

Flash Attention reorders computation to reduce this IO cost to O(Nd + N²/B) where B is the SRAM block size.

## Memory Hierarchy

Modern GPUs have two memory tiers with very different bandwidths:
- **HBM** (e.g., A100: 40–80 GB): large, slow (~2 TB/s)
- **SRAM** (on-chip): tiny (~20 MB total), fast (~19 TB/s)

Standard attention materializes Q, K, V, S=QKᵀ, and O to HBM repeatedly. Flash Attention keeps tiles in SRAM.

## Tiling Algorithm

```
For each block of Q (outer loop):
  For each block of K, V (inner loop):
    Load Q_block, K_block, V_block into SRAM
    Compute local S = Q_block × K_blockᵀ
    Update running softmax statistics (max, sum) — online softmax
    Accumulate O_block += softmax_weight × V_block
  Write O_block to HBM
```

**Key trick — online softmax**: the algorithm tracks running `(max, sum)` statistics to compute softmax incrementally across K/V blocks without ever materializing the full N×N matrix. Correction factors are applied when the running max is updated.

## Performance

- **FlashAttention v1 (2022)**: 7.6× speedup on GPT-2, 1.6× on T5 vs. standard PyTorch on A100
- **FlashAttention v2 (2023)**: 2× faster than v1; work partitioning across thread blocks; reduces non-matmul FLOPs
- **FlashAttention v3 (2024)**: H100-specific; exploits warp specialization, FP8 TMA pipeline
- **FlashAttention v4 (2025)**: Blackwell-first (SM100 / GB200); 5-stage warp-specialized pipeline; increased on-chip reuse

## Memory Savings

Standard attention: O(N²) memory for attention matrix (stored for backprop).
Flash Attention: O(N) memory — only stores the output O and softmax statistics (log-sum-exp). Recomputes attention during backward pass via recomputation ("rematerialization").

This enables training on much longer sequences within the same GPU memory budget.

## Flash Decoding

A variant for decoding (inference) at long contexts: parallelizes across the sequence length dimension rather than batch/heads, enabling efficient long-context generation even with small batch sizes.

## Quantized Attention Kernels (SageAttention family)

The SageAttention series builds on FlashAttention-2's tiling structure and adds quantization to further accelerate the attention Matmuls:

| Kernel | Quantization | TOPS (RTX4090) | Speedup vs FA2 |
|---|---|---|---|
| [[sageattention]] | INT8 Q/K, FP16 P/V | 340 | 2.1× |
| [[sageattention2]] | INT4 Q/K (per-thread), FP8 P/V (2-level) | 481 | 3× |
| [[sageattention3]] | FP4 NVFP4 microscaling (Blackwell only) | 1038 | 5× |

All three are plug-and-play replacements for FlashAttention-2 with near-zero end-to-end accuracy loss.

## Relationship to Other Work

- [[paged-attention]] manages where KV tensors are stored; Flash Attention manages how attention is computed over them — they compose
- Flash Attention is now the default attention backend in PyTorch, JAX, HuggingFace Transformers, vLLM, TensorRT-LLM
- [[kv-cache]] benefits from Flash Attention during both prefill and decode

## See Also

- [[paged-attention]] — memory management for KV cache; orthogonal to attention compute
- [[kv-cache]] — KV cache fundamentals
- [[continuous-batching]] — serving-level optimization that Flash Attention enables
- [[sageattention]] — INT8 quantized attention built on FA2 tiling; 2.1× speedup
- [[sageattention2]] — INT4/FP8 quantized attention; 3× speedup
- [[sageattention3]] — FP4 quantized attention for Blackwell; 5× speedup
