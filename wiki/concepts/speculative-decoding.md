---
title: Speculative Decoding
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [inference, speculative]
sources: [raw/papers/2211.17192v2.pdf]
confidence: high
---

# Speculative Decoding

**Fast Inference from Transformers via Speculative Decoding**
*Yaniv Leviathan, Matan Kalman, Yossi Matias, Google Research, ICML 2023, arXiv:2211.17192*

## Core Idea

Autoregressive decoding from large Transformers is slow: generating K tokens requires K serial forward passes of the target model. Speculative decoding achieves **2–3× speedup without changing outputs** by exploiting hardware's ability to run forward passes in parallel.

**Key observations:**
1. Language modeling tasks often contain "easy" subtasks well-approximated by smaller models
2. Large models are often *memory bandwidth* bottlenecked, not compute bottlenecked — additional compute is "free" within the same memory access

## Algorithm

1. **Draft generation**: A small, fast "approximation model" M_q generates γ candidate tokens in γ serial passes
2. **Parallel verification**: The large target model M_p evaluates all γ drafts *in a single parallel pass*
3. **Speculative sampling**: Accept tokens that would be consistent with M_p's distribution; sample corrections for rejected ones
4. **Guarantee**: The joint distribution of accepted tokens is *identical* to M_p's autoregressive distribution

The number of serial target model passes is at most 1 per generation step (and often fewer). The *expected* number of new tokens per target pass is >1 whenever the draft model is a reasonable approximator.

## Speculative Sampling

Standard rejection sampling isn't applicable (can't afford rejection in the sampling loop). Instead:
- Accept draft token t if `q(t|context)/p(t|context) ≥ uniform sample` — accept with probability proportional to how well the draft model matches the target
- On rejection: sample the *residual* distribution `max(0, p-q) / normalization`
- This maintains exact distributional guarantees

## Results (T5-XXL demonstration)

- **2–3× wall-clock speedup** vs. robust T5X implementation
- Zero change to model outputs or probability distributions
- Works on any pre-trained model without fine-tuning or architecture changes

## Practical Considerations

- Draft model quality is the key variable — the better M_q approximates M_p, the higher the acceptance rate
- Common practice: use a smaller version of the same model family (e.g., Llama-3-8B to draft for Llama-3-70B)
- Works alongside [[kv-cache]] optimization and [[quantization]] — orthogonal techniques

## Self-Speculative Decoding

A variant that eliminates the separate draft model by using the target model's own early layers as the drafter:

- **[[layerskip]]** (Meta, 2024): trains with layer dropout so early exits are useful; early layers (0..e) draft, full model verifies; reuses draft KV states in verification pass; up to 2.16× speedup
- **SWIFT** (ICLR 2025): plug-and-play, no retraining; adaptively selects layers to skip per token at runtime; 1.3–1.6× speedup, exact distribution match
- **DASH** (2025): per-token MDP policy for layer skipping; input-aware, outperforms fixed-skip methods

See [[early-exit-inference]] for the broader early exit landscape.

**Trade-off vs. standard speculative decoding**: self-speculative needs no extra model memory but draft quality is bounded by how good early layers are. Standard spec decoding with a purpose-trained sibling model typically achieves higher acceptance rates.

## Limitations

- Requires a suitable draft model (additional memory footprint) — self-speculative eliminates this at some quality cost
- Gains diminish if draft model is poor (acceptance rate drops)
- Memory bandwidth savings only realized when compute is truly the free resource

## See Also

- [[kv-cache]] — orthogonal memory bottleneck technique
- [[layerskip]] — self-speculative decoding via early exit layers
- [[early-exit-inference]] — concept page for early exit and layer skipping
- [[continuous-batching]] — serving scheduler; composes with speculative decoding
- [[deepseek-v3-2]] — uses scalable RL for improved reasoning, orthogonal to inference-time optimizations
- [[deepseek-v4]] — architectural KV reduction via CSA/HCA, complementary to speculative decoding
