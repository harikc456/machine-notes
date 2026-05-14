---
title: Early Exit and Layer Skipping
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [inference, architecture, optimization]
sources: [raw/papers/2404.16710v2.pdf, raw/papers/2505.17420v1.pdf]
confidence: high
---

# Early Exit and Layer Skipping

## Overview

Not all tokens require equal computation. Simple tokens (common words, punctuation, easy continuations) are well-predicted by intermediate transformer layers. Complex tokens (rare words, long-range dependencies, reasoning steps) need the full depth.

**Early exit** and **layer skipping** exploit this by routing "easy" tokens through fewer layers, reducing average compute per token without changing the output distribution on "hard" tokens.

## Two Approaches

### Early Exit

The model has one exit point per layer (or every k layers). At each exit point, a lightweight classifier or the layer's hidden state directly predicts whether the current token's representation is "confident enough" to exit.

- **Hard early exit**: token exits at layer i; layers i+1…L are never run for this token
- **Soft early exit** (self-speculative): token exits early as a *draft* candidate; remaining layers verify

### Layer Skipping

Rather than exiting entirely, specific intermediate layers are skipped on a per-token basis. The residual stream bypasses those layers.

Three strategies:
1. **Early skipping**: skip a contiguous prefix of layers
2. **Periodic skipping**: skip every k-th layer
3. **Input-aware skipping** (DASH): per-token MDP policy selects which layers to skip

## LayerSkip (Meta, 2024)

[[layerskip]] — the canonical early exit + self-speculative decoding paper.

**Training**: layer dropout with decreasing rates by depth (early layers never dropped, deep layers dropped frequently). This conditions the model to produce useful representations at all depths.

**Inference**:
- **Early exit mode**: run all tokens to a fixed exit layer e < L; use that layer's LM head
- **Self-speculative mode**: early layers (0..e) act as draft model; full model (0..L) verifies in one parallel pass

The self-speculative variant is lossless (same distribution as full model) when the verification step accepts all drafts, and degrades gracefully otherwise.

## DASH (2025)

**DASH: Input-Aware Dynamic Layer Skipping via Markov Decision Policies**
*arXiv:2505.17420*

Frames per-token layer selection as a Markov Decision Process:
- **State**: hidden state after each layer
- **Action**: execute next layer or skip it
- **Policy**: learned lightweight router that decides per-token, per-layer

Unlike static skip patterns (same layers skipped for all inputs), DASH adapts to the input. Achieves consistent speedups over full-model inference while maintaining quality, outperforming prior fixed-skip methods.

## SWIFT (ICLR 2025)

**SWIFT: On-the-Fly Self-Speculative Decoding**

Plug-and-play: no retraining required. Adaptively selects which intermediate layers to skip per token at inference time.

- No auxiliary models, no layer dropout training
- 1.3–1.6× speedup while preserving the original output distribution
- Selects skip candidates by measuring layer contribution at runtime

## Relationship to Speculative Decoding

Self-speculative decoding (LayerSkip, SWIFT) is a variant of [[speculative-decoding]] where:
- **Draft model** = early layers of the target model
- **Target model** = full model

Eliminates the need for a separate draft model and its memory footprint. The trade-off: the "draft" quality is bounded by how well early layers approximate full-model distributions, which is generally lower than a purpose-trained smaller sibling model.

## Practical Considerations

- Gains are highest for "easy" generation tasks (summarization, code completion) and lower for complex reasoning
- LayerSkip requires retraining with layer dropout; SWIFT and DASH are post-training approaches
- Layer skipping is orthogonal to [[kv-cache]] compression, [[quantization]], and [[continuous-batching]] — all can be combined

## See Also

- [[layerskip]] — Meta's training + inference framework for early exit
- [[speculative-decoding]] — general draft-verify framework; self-speculative is a variant
- [[continuous-batching]] — serving scheduler; early exit affects how many layers each token runs, not batching
