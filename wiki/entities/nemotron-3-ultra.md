---
title: Nemotron 3 Ultra
created: 2026-07-06
updated: 2026-07-06
type: entity
tags: [model, architecture, inference, kv-cache, quantization, sparsity, speculative, open-source]
sources: [raw/papers/2606.15007v1.pdf]
confidence: high
---

# Nemotron 3 Ultra

**Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning**
*NVIDIA, June 2026*

## Overview

550B total / 55B active parameter hybrid Mamba-Attention MoE model, the largest in the Nemotron 3
family (extends Nemotron 3 Super). 20T pretraining tokens, context extended to 1M, open-weights
(base, post-trained, NVFP4-quantized checkpoints + data + recipes on HuggingFace).

## Architecture

- 108 layers, model dim 8192, repeating pattern of Mamba-2 blocks + Attention layers + [[mixture-of-experts|LatentMoE]] layers for sparse scaling.
- Extreme GQA: 64 query heads, only 2 KV heads — Mamba layers replace most attention, directly
  cutting KV cache footprint (see [[kv-cache]]).
- LatentMoE (Elango et al. 2026): 512 experts/layer, top-22 activated, latent size 2048 — better
  accuracy per active parameter than standard granular MoE (cf. [[mixture-of-experts]]).
- **MTP** (Multi-Token Prediction): 2 shared-weight heads, each a single attention + single MoE
  layer, used as a built-in speculative-decoding draft mechanism (see [[speculative-decoding]]).

## NVFP4 Pretraining

Same recipe as Nemotron 3 Super: E2M1 datatype, 2D block quantization, Random Hadamard Transforms
on wgrad inputs, stochastic rounding on gradients; final 15% of layers and sensitive projections
(Mamba output, latent, QKV/attention, MTP, embeddings) kept in higher precision. Reported as the
largest-scale stable NVFP4 pretraining demonstration to date, with <0.4% average train-loss gap
against BF16 ablation branches at 5T/10T/16T-token checkpoints.

## Post-training

SFT → unified RLVR across reasoning/agentic/code/safety/usability/chat environments → 10+
domain-specialist teacher models → **MOPD** (Multi-teacher On-Policy Distillation) consolidates
teachers into Ultra via dense token-level guidance on student rollouts. Includes an inference-time
reasoning-effort control knob.

## Results

Claims ~6× inference throughput vs. SOTA open LLMs (GLM-5.1-754B-A40B, Kimi-K2.6-1T-A32B,
Qwen-3.5-397B-17B) at 8K/64K input/output, on-par accuracy — driven by the Mamba-Attention hybrid
(low attention cost + small KV cache) plus MoE's parameter efficiency.

## Relation to Existing Work

A hybrid-backbone counterpart to [[deepseek-v4]]'s CSA/HCA hybrid-attention approach to the same
problem (long-context, high-throughput inference) — Nemotron replaces most attention layers
outright with Mamba-2 rather than redesigning attention itself. The MTP head design is a direct,
architecturally-integrated form of [[speculative-decoding]], comparable in spirit to LayerSkip's
self-speculative approach ([[layerskip]]) but trained jointly with the base model from pretraining.

## See Also

- [[deepseek-v4]] — comparable large-scale hybrid-attention frontier release
- [[kv-cache]] — KV cache cost that the Mamba-heavy architecture minimizes
- [[mixture-of-experts]] — LatentMoE's role in scaling capacity
- [[speculative-decoding]] — MTP heads as built-in draft mechanism
