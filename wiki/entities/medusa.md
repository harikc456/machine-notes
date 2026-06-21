---
title: Medusa
created: 2026-06-17
updated: 2026-06-17
type: entity
tags: [inference, speculative]
sources: [raw/papers/2401.10774v3.md]
confidence: high
---

# Medusa

**Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**
*Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao*
*Princeton / Together AI / UIUC / CMU / UConn, arXiv:2401.10774, ICML 2024*

## Core Idea

[[speculative-decoding]] requires a separate draft model — difficult to train, extra memory footprint, complex distributed serving. Medusa eliminates the draft model entirely by attaching **K extra single-layer decoding heads** directly to the backbone LLM's last hidden state.

Each head k predicts the token k+1 positions ahead in a **single forward pass**:

```
p_t^(k) = softmax(W2^(k) · SiLU(W1^(k) · h_t + h_t))
```

W2 is initialized from the original LM head; W1 is initialized to zero (so the heads start as copies of the LM head and learn residual corrections).

## Tree Attention

Medusa heads produce top-s_k candidates at each position. These are assembled into a **candidate tree** via Cartesian product: all combinations of predictions across K heads. A specialized attention mask makes each candidate token attend only to the tokens in its own branch (its "history" in the tree), not across branches.

This allows processing all candidates simultaneously in a single forward pass through the full backbone, then verifying which candidate sequence prefix the backbone would have produced. The **longest accepted candidate prefix** is taken as the output.

## Training Modes

### Medusa-1: Frozen Backbone

Only the K extra heads are trained; backbone weights are frozen. Loss per head k:

```
L_Medusa-1 = Σ_k -λ_k log p_t^(k)(y_{t+k+1}),  λ_k = 0.8^k
```

λ_k decays with distance — farther-ahead predictions are harder and weighted lower. Trainable in ~5 hours on a single A100 PCIE with 60k ShareGPT samples. Compatible with quantized (QLoRA-style) backbone.

**Result**: >2.18× lossless speedup on Vicuna-7B.

### Medusa-2: Joint Training

Backbone trained jointly with the heads using a combined loss:

```
L_Medusa-2 = L_LM + λ_0 · L_Medusa-1
```

Requires special recipe to avoid degrading backbone quality:
- **Differential learning rates**: higher LR for heads, lower for backbone
- **Heads warmup**: train heads alone in Stage 1; then joint in Stage 2

**Result**: 2.83× speedup on Vicuna-7B with no quality loss (MT-Bench: 6.18 vs 6.17 baseline).

## Speedup Breakdown

| Technique added | Speedup |
|---|---|
| Medusa heads (no tree) | ~1.5× |
| + naive tree attention | ~1.9× |
| + optimized sparse tree | ~2.2× |
| + Medusa-2 joint training | ~2.8× |

## Extensions

**Typical acceptance**: Alternative to rejection sampling for multi-candidate verification. A candidate token is accepted if its probability under the original model exceeds a per-token threshold that accounts for the model's entropy (high-entropy positions are more permissive). Avoids rejection sampling's degradation at high temperatures. Acceleration rate increases monotonically with temperature under typical acceptance.

**Self-distillation**: When no supervised training data is available (e.g. RLHF-tuned models like Zephyr-7B), generate training data from the model's own outputs using seed prompts. Allows Medusa-2 to be trained without access to the original fine-tuning dataset.

## Positioning in the Speculative Decoding Landscape

| System | Draft Mechanism | Speedup | Extra Infra |
|---|---|---|---|
| SD (classic) | Separate draft model | 2–3× | Yes |
| Medusa | Single-layer heads on backbone | 2.2–2.83× | No |
| [[eagle]] | Feature-level AR draft | 2.7–3.5× | No |
| [[eagle-2]] | Dynamic tree + EAGLE | 3.05–4.26× | No |
| [[eagle-3]] | Direct token + multi-layer feat | up to 6.5× | No |
| [[dflash]] | Block diffusion draft | 6×+ | No |

Medusa is simpler than EAGLE (single-layer heads vs. autoregressive feature predictor), requires no draft autoregression at inference time, and integrates seamlessly into frozen-backbone serving workflows. Its draft accuracy (~0.6) is lower than EAGLE's (~0.8), which explains the speedup gap.

## See Also

- [[speculative-decoding]] — algorithm, guarantees, and the broader landscape including EAGLE family
- [[eagle]] — feature-level AR prediction; ~0.8 draft accuracy vs Medusa's ~0.6; 2.7–3.5×
- [[eagle-3]] — current SOTA in the no-draft-model family; 6.5×
- [[dflash]] — replaces AR drafting with block diffusion; constant draft cost; 6×+
- [[layerskip]] — self-speculative via early exit; no extra heads; lower overhead but lower speedup
- [[kv-cache]] — Medusa preserves standard AR KV caching (causal attention, backbone unchanged)
