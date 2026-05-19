---
title: Attention Residuals (AttnRes)
created: 2026-05-19
updated: 2026-05-19
type: entity
tags: [architecture, residual, attention, training, inference]
sources: [raw/papers/2603.15031v1.pdf]
confidence: high
---

# Attention Residuals (AttnRes)

**Kimi Team (Guangyu Chen, Yu Zhang, Jianlin Su et al.) · 16 Mar 2026 · arXiv: 2603.15031**

## Overview

Standard residual connections (`h_l = h_{l-1} + f_{l-1}(h_{l-1})`) accumulate all layer outputs with fixed unit weights, causing hidden-state magnitudes to grow as O(L) with depth (PreNorm dilution). Deeper layers must compensate by producing ever-larger outputs, progressively burying early-layer information.

AttnRes draws a formal duality between depth and sequence: just as Transformers replaced RNNs for sequence modeling by substituting attention for fixed-weight recurrence, AttnRes replaces fixed-weight depth accumulation with learned **softmax attention over preceding layer outputs**:

```
h_l = Σ_{i=0}^{l-1}  α_{i→l} · v_i

α_{i→l} = φ(w_l, k_i) / Σ_j φ(w_l, k_j)
```

where `w_l ∈ R^d` is a learned per-layer pseudo-query, `k_i = v_i = f_i(h_i)` (keys = values = layer outputs), and `φ(q, k) = exp(q^T RMSNorm(k))`. RMSNorm on keys prevents layers with large output magnitudes from dominating attention weights.

Only **one d-dimensional vector per layer** is added, negligible parameter count.

## Full AttnRes

Each layer l attends over all l−1 preceding layer outputs plus the token embedding (b_0 = h_1). Memory: O(Ld) to store all layer outputs. This overlaps with activations already retained for backprop, so vanilla training overhead is zero; at scale, pipeline parallelism requires transmitting all L outputs across stages → O(Ld) communication.

## Block AttnRes

Partition L layers into N blocks (typically N≈8). Within each block, layer outputs accumulate to a block summary: `b_n = Σ_{j∈B_n} f_j(h_j)`. Across blocks, each layer attends over the N block summaries plus the token embedding, replacing L individual sources with N coarser ones.

- Memory: O(Nd) vs O(Ld) for Full AttnRes
- N=8 recovers ~90% of Full AttnRes gain
- Effective rank of depth-mixing matrix lies between N (standard residual) and N+S (Full AttnRes)

The block structure also bounds the KV cache for inter-layer attention during inference.

## Infrastructure

**Training (pipeline parallelism):** Block AttnRes requires transmitting N block representations across pipeline stages. Cross-stage caching exploits the fact that each physical stage processes multiple virtual stages: only the ~PN_p incremental blocks since the last virtual stage need to be retransmitted, reducing per-transition communication from O(C) to O(P). Training overhead measured at <4% end-to-end.

**Inference (two-phase computation):**
- Phase 1: batch all S layers in a block as a single matrix query against cached block representations (inter-block attention). Since pseudo-queries w_l are input-independent, they can be batched without waiting for sequential outputs.
- Phase 2: sequential intra-block attention using the partial sum b_n^i, merged with Phase 1 via online softmax.
- Total I/O per layer: (S+N)d vs 3d for standard residuals and 34d for mHC (m=4 streams)
- Inference latency overhead: <2% on typical workloads

## Unified View: Depth-Wise Linear vs Softmax Attention

All residual variants are instances of depth-wise linear attention (Table 5 in paper):

| Method | Update rule | Depth mixing matrix |
|---|---|---|
| Standard residual | fixed unit weights | lower-triangular all-ones |
| Highway | scalar gates g_l | lower-triangular carry products |
| (m)HC | m parallel streams, learned transition matrices A_l | m-semiseparable |
| DenseFormer | learned scalar coefficients (static) | learned lower-triangular |
| Full AttnRes | softmax over all previous | dense rank-L |
| Block AttnRes | softmax over N block summaries | rank between N and N+S |

AttnRes generalizes prior methods from depth-wise *linear* to depth-wise *softmax* attention, completing the same linear-to-softmax transition that Transformers performed over sequences.

## Results

### Scaling Laws (5 model sizes, 194M–528M activated params)

| Model | Baseline | Block AttnRes | Full AttnRes | mHC-lite |
|---|---|---|---|---|
| 194M | 1.931 | 1.909 | **1.899** | 1.906 |
| 241M | 1.895 | 1.875 | **1.874** | 1.869 |
| 296M | 1.829 | 1.809 | **1.804** | 1.807 |
| 436M | 1.766 | 1.746 | **1.737** | 1.747 |
| 528M | 1.719 | 1.693 | **1.692** | 1.694 |

Block AttnRes at 5.6 PFLOP/s-days matches the baseline at 1.25× more compute.

### 48B Model (Kimi Linear, 1.4T tokens)

Block AttnRes with N=9 blocks (27 transformer blocks × 2 layers + embedding). Gains over baseline:

| Benchmark | Baseline | AttnRes | Δ |
|---|---|---|---|
| GPQA-Diamond | 36.9 | **44.4** | +7.5 |
| BBH | 76.3 | **78.0** | +1.7 |
| Math | 53.5 | **57.1** | +3.6 |
| HumanEval | 59.1 | **62.2** | +3.1 |
| MMLU | 73.5 | **74.6** | +1.1 |
| C-Eval | 79.6 | **82.5** | +2.9 |

Especially strong on multi-step reasoning (GPQA, Math) — consistent with the hypothesis that improved depth-wise information flow benefits compositional tasks.

### Training Dynamics

- **Output magnitude**: Baseline grows monotonically with depth (PreNorm dilution). Block AttnRes resets accumulation at block boundaries → periodic bounded pattern.
- **Gradient magnitude**: Baseline has disproportionately large gradients in early layers. AttnRes softmax competition distributes gradient mass across sources uniformly.

## Key Ablation Findings

- **Input-dependent query** (project from hidden state): better loss (1.731 vs 1.737) but requires sequential memory access during decoding → not used; learned pseudo-query chosen.
- **softmax > sigmoid**: sigmoid replaces competitive normalization with independent gates → 1.741 vs 1.737.
- **RMSNorm on keys essential**: without it, large-magnitude layers dominate attention; critical for Block AttnRes where block summaries accumulate more.
- **Single query better than multihead** (H=16): 1.752 vs 1.746. Optimal depth mix is uniform across channels — a layer's relevance is global, not per-subspace.
- **Optimal architecture shifts**: AttnRes prefers deeper, narrower models (d_model/L_b ≈ 45 vs 60 for baseline under fixed compute).

## See Also

- [[hyper-connections]] — alternative residual generalization (multi-stream); mHC outperformed by AttnRes in ablations
- [[mhc]] · [[mhc-lite]] · [[kromhc]] — HC family methods
- [[orthogonal-residual-streams]] — complementary residual-stream perspective (writes ⊥ existing directions)
- [[weight-norm-training]] — related training dynamics concern (PreNorm dilution)
- [[flash-attention]] — inspiration: depth-wise attention follows the same IO-aware design philosophy
