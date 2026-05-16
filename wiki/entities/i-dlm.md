---
title: I-DLM (Introspective Diffusion Language Model)
created: 2026-05-16
updated: 2026-05-16
type: entity
tags: [architecture, inference, training]
sources: [raw/papers/2604.11035v1.pdf]
confidence: high
---

# I-DLM (Introspective Diffusion Language Model)

**Introspective Diffusion Language Models**
*Yu, Jian, Wang, Zhou et al. — Together AI / UIUC / UT Austin / Princeton / Stanford, arXiv:2604.11035, Apr 2026*

## Core Problem: The Introspective Consistency Gap

Existing diffusion language models (DLMs) fail to match AR quality not because of model capacity, but because they lack **introspective consistency**: AR models are trained to predict each token given the previous ones, so by construction the model "agrees with its own generations." DLMs generate tokens in parallel via bidirectional attention and multi-step denoising — they are not trained to endorse what they produce.

**Introspective acceptance rate** α: for each generated token x_k ~ q_k, compute the causal distribution p_k via a separate forward pass; α = (1/L) Σ_k min(1, p_k(x_k)/q_k(x_k)). AR achieves α=1 by construction. Measured on IFEval: SDAR (8B) achieves 0.699, LLaDA 2.0-flash (8B) achieves 0.568 — substantial divergence.

## Three Contributions

### 1. Introspective-Consistency Training

Converts a pretrained AR model into a DLM using **only 4.5B tokens**:

- **Causal attention throughout**: unlike SDAR/LLaDA which use block-causal or bidirectional attention, I-DLM uses strict causal masking — tokens attend only to previous positions in both clean and masked regions
- **Logit shift**: hidden state at position i predicts token i+1 (standard AR mapping) — preserves the AR model's inherent logits→token mapping, which standard masked DLMs break
- **All-masked objective**: replace all tokens with [MASK] to form x_t; train on concatenation [x_t | x_0]; every masked position gets a supervision signal (vs. standard masking which wastes 1-r compute on unsupervised positions)
- **Auto-balanced loss**: dynamically rescale L_clean to match L_mask magnitude, preventing the decode pathway from dominating the introspection pathway

### 2. Introspective Strided Decoding (ISD)

Single-pass algorithm that simultaneously generates N tokens and verifies prior ones:
- At [MASK] positions: model proposes new tokens (decode pathway, distribution q)
- At introspection positions: model revisits previously generated tokens against the causal anchor distribution p
- At p ≥ 0.83 acceptance rate, ISD achieves compute efficiency > 1 (TPF/OH > 1) — the only DLM method above the break-even line
- At empirically observed p ≥ 0.85: efficiency 1.08–2.29×

### 3. AR-Compatible Serving Stack

I-DLM uses causal attention, making it directly integrable into existing AR serving systems (SGLang, continuous batching, paged KV cache):
- Gated residual LoRA: adapters applied only at mask positions; verification uses base model weights
- TPS growth rate 549 (vs. SDAR: 84) — throughput scales proportionally with tokens per forward pass

## Results

On 8B models:

| Model | MATH-500 | AIME-24 | LiveCodeBench-v6 | Throughput |
|---|---|---|---|---|
| Qwen3-8B (AR thinking) | ~95% | ~72% | — | 1× |
| LLaDA-2.1-mini (16B) | ~83% | ~43% | ~30% | 0.25× |
| SDAR (8B) | — | ~10% | — | ~0.3× |
| **I-DLM-8B** | **matches Qwen3** | **69.6** | **45.7** | **3.1× over SDAR** |

- First DLM to match strong same-scale AR quality
- 3.1× higher throughput than SDAR at concurrency=32
- 4× higher throughput + 18.2 pts over LLaDA-2.1-mini (16B)

## Compute Efficiency Comparison

| Method | Efficiency (TPF/OH) at p=0.85 |
|---|---|
| SDAR | 0.64–0.72 (always < 1) |
| TiDAR | ~0.80 at p=1 |
| **I-DLM (ISD)** | **1.08–2.29 (> 1)** |

Only I-DLM crosses the break-even line where parallel generation actually saves compute.

## Relationship to Block Diffusion

[[block-diffusion]] (ICLR 2025) takes a structural approach: autoregressive over blocks, diffusion within. I-DLM takes a training approach: convert AR models into DLMs by enforcing introspective consistency. Both close the AR/DLM quality gap, but I-DLM achieves stronger results at the same scale and is more directly deployable on existing AR infrastructure.

## See Also

- [[block-diffusion]] — alternative approach to closing AR/DLM quality gap
- [[diffusion-language-models]] — concept page covering the DLM landscape
- [[speculative-decoding]] — ISD is conceptually related to self-speculative decoding; both verify and generate in a single forward pass
- [[layerskip]] — self-speculative decoding; ISD is analogous but for DLMs
- [[kv-cache]] — I-DLM re-enables standard AR KV caching via causal attention
