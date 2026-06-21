---
title: QKV Projection Sharing
created: 2026-06-17
updated: 2026-06-17
type: entity
tags: [attention, inference, kv-cache, architecture]
sources: [raw/papers/2606.04032v2.md]
confidence: high
---

# QKV Projection Sharing

**Do Transformers Need Three Projections? Systematic Study of QKV Variants**
*Ali Kayyam, Anusha Madan Gopal, M Anthony Lewis — BrainChip Inc., arXiv:2606.04032, ICML 2026*

## Core Question

Standard multi-head attention maintains three independent learned projections W_Q, W_K, W_V. This paper asks: are all three necessary? It evaluates three progressively aggressive sharing constraints across synthetic tasks, vision, and language modeling (300M–1.2B params, 10B tokens).

## Three Variants

### Q-K=V — Shared Key-Value (the winner)

Separate Q projection; K and V both come from a single shared `kv_proj`. The same vector serves as both the scoring key and the aggregated value:

```
A = Softmax(α Q K^T) K      # K serves as both K and V
```

**Cache benefit**: 50% KV cache reduction — only the K tensor needs to be cached; V is reused from K. At 32k context: 2.62 GB (QKV) → 1.31 GB (Q-K=V).

**Quality cost**: +3.1% perplexity at 300M (improves to +2.48% at 1.2B — trend suggests larger models are more robust).

**Why it works**: K and V occupy similar representational space — cosine similarity 0.73 across layers, similar effective rank (687 vs 702 out of 1024 dims). K is rich enough to absorb V's role when trained under the K=V constraint.

### Q=K-V — Shared Query-Key (avoid in causal LM)

Unified Q and K (symmetric attention), separate V:

```
A = Softmax(α K K^T) V
```

**No cache benefit**: must still cache K and V separately — zero inference memory savings.

**Quality cost**: +4.9% PPL (worse than Q-K=V despite identical parameter count).

**Why it fails**: Forces a symmetric attention matrix QKᵀ. Symmetric attention breaks causal directionality — each position strongly attends to itself and nearby tokens rather than the relevant context. Works for non-causal tasks (vision, set processing) where symmetry is appropriate; `(Q=K-V)⁺` variant adds 2D positional encoding to inject asymmetry there.

### Q=K=V — Single Projection (avoid)

All three from one projection:

```
A = Softmax(α K K^T) K
```

+25.4% PPL — combines the symmetry pathology of Q=K-V with the representational bottleneck. Not recommended.

## Compounding with Head Sharing

Projection sharing (reducing projections per head) and head sharing (GQA/MQA, reducing number of K/V heads) are **orthogonal** — they can be combined multiplicatively:

| Variant | PPL Δ vs QKV | Cache Reduction |
|---|---|---|
| QKV (baseline) | — | — |
| Q-K=V | +3.1% | 50% |
| GQA-4 | +0.7% | 75% |
| MQA | +1.5% | 93.8% |
| **Q-GQA-4** | **+3.9%** | **87.5%** |
| **Q-MQA** | **+4.8%** | **96.9%** |

Q-MQA approaches the theoretical maximum KV cache reduction (96.9%) while keeping PPL within 5%.

## Scale Validation

Rankings stable from 300M to 1.2B parameters. Quality degradation of Q-K=V *improves* with scale (3.1% → 2.48%), consistent with projection sharing becoming more benign as models grow more over-parameterized.

## Deployment Guidance (from paper)

| Scenario | Recommended | Cache ↓ | PPL Δ |
|---|---|---|---|
| Cloud (quality) | GQA-4 | 75% | +0.7% |
| Edge (balanced) | Q-K=V | 50% | +3.1% |
| Edge (aggressive) | Q-GQA-4 | 87.5% | +3.9% |
| IoT / Mobile | Q-MQA | 96.9% | +4.8% |

## Relationship to rbf_ffn Experiments

This paper provides the theoretical grounding for two new attention classes added to `rbf_ffn/models/attention.py`:

- **`KVSharedAttention`**: the Q-K=V variant. Q has its own projection; K and V share `kv_proj`. RoPE applied to Q and to the rotated-K used for scoring; V remains un-rotated.
- **`KVSharedExclusiveSelfAttention`**: Q-K=V combined with [[xsa]] Gram-Schmidt orthogonalisation. After computing the attention output Y, removes the component along the normalized value direction: `Z = Y - (Y·V̂) V̂`. Addresses attention similarity bias while also halving the KV cache.

Configs `baseline_kv_shared.yaml` and `baseline_xsa_kv_shared.yaml` set up baseline runs for both classes. No leaderboard results yet.

## See Also

- [[kv-cache]] — projection sharing as a new architectural reduction axis, complementary to quantization and eviction
- [[xsa]] — XSA Gram-Schmidt step applied on top of Q-K=V in `KVSharedExclusiveSelfAttention`
- [[deepseek-v4]] — MLA takes a different compression path (low-rank latent compression of K and V); projection sharing is a simpler hard-equality constraint
- [[attention]] — broader attention mechanism context
