---
title: XSA (Exclusive Self-Attention)
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [attention, architecture, training]
sources: [raw/papers/2603.09078v1.pdf]
confidence: high
---

# XSA — Exclusive Self-Attention

**Exclusive Self Attention**
*Shuangfei Zhai, Apple, arXiv:2603.09078, Mar 2026*

## Core Idea

Standard self-attention (SA) has an **attention similarity bias**: the attention output `y_i` tends to have high cosine similarity with the token's own value vector `v_i` — meaning SA spends capacity re-aggregating information the current position already encodes, competing with the FFN's role.

XSA fixes this with a single Gram-Schmidt subtraction step:

```
y_i = Σ a_{i,j} v_j          # standard SA output
z_i = y_i − (y_iᵀ v_i / ||v_i||²) · v_i    # remove component along own value
out = o_proj(z_i)
```

The output `z_i` is guaranteed orthogonal to the token's own value vector — attention is forced to capture only *contextual* information, freeing point-wise feature transformation to the FFN.

## Motivation: Attention Similarity Bias

Empirical observation on a 1.3B parameter model (sequence length 2048, 100B tokens): the cosine similarity `<y_i, v_i>` increases with layer depth and is consistently positive. This means SA increasingly aggregates value vectors similar to the current token's own — effectively writing back what was already there.

**Why this is harmful**:
1. SA's job is context modeling; point-wise transformation is the FFN's job
2. Writing along the own-value direction creates competition between contextual and point-wise modeling
3. The residual path already carries `v_i` forward — SA's redundant write wastes capacity

## Properties

- **Two lines of code change** on top of standard SA
- **No new parameters**
- **Minimal compute overhead** (one dot-product per position)
- **Consistent gains**: outperforms SA across model sizes up to 2.7B parameters
- **Larger gains at longer sequences**: orthogonality helps more when more context is available
- **Robust to learning rate variation**

## Results (from XSA paper)

- Better training/validation loss across three model sizes
- Better downstream evaluation results
- Increasingly larger gains as sequence length grows
- Robust to attention sinks

## Relationship to rbf_ffn Experiments

XSA is the core attention variant in `findings.md` §8–§10. Key empirical findings:

| Variant | Val PPL (ep 2) | vs SwiGLU baseline |
|---|---|---|
| XSA + SwiGLU (no norm) | 72.41 | −4.3% |
| XSA + qk_norm | 71.05 | −6.1% |
| XSA + qk_norm + orthogonal_ffn | 69.87 | −7.7% |
| XSA + qk_norm + wnorm | 56.88 | −24.8% |
| XSA + qk_norm + wnorm + orthogonal_ffn | 55.57 | −26.5% |

**Key finding**: XSA's benefit (+1.28 ppl over SwiGLU+wnorm) persists under weight normalization — the gain is structural (inductive bias), not an optimization artifact. The attention orthogonality and FFN orthogonality (OrthogonalMLPWrapper) stack additively, consistent with addressing different redundancy sources.

**Alignment probe** (§10.3): blocks.1 has the highest FFN-to-input cosine alignment (0.41) among middle layers, suggesting some layers are more prone to the same "writing along redundant directions" problem that motivates XSA. The selective orthogonal FFN (layers 1,3) achieves 54.97 — beating full orthogonal_ffn (55.57).

## Relationship to OrthogonalMLPWrapper

XSA and `OrthogonalMLPWrapper` are the same inductive bias applied to different modules:
- **XSA**: attention output is orthogonal to the current token's own *value vector*
- **OrthogonalMLPWrapper**: FFN output is orthogonal to the *pre-norm input*

Both constrain residual stream writes to be *information-adding* rather than information-recycling. See [[orthogonal-residual-streams]] for the unified view.

## See Also

- [[orthogonal-residual-streams]] — unified view of XSA, OrthogonalMLPWrapper, and mHC
- [[qknorm]] — normalization technique typically paired with XSA in experiments
- [[weight-normalization]] — the dominant training technique in the same experimental context
- [[hyper-connections]] — a different approach to constraining residual stream writes
