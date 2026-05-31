---
title: Speculative Decoding
created: 2026-05-14
updated: 2026-05-31
type: concept
tags: [inference, speculative]
sources: [raw/papers/2211.17192v2.pdf, raw/papers/2603.03251v3.pdf, raw/papers/2401.15077v3.pdf, raw/papers/2406.16858v2.pdf, raw/papers/2503.01840v3.pdf, raw/papers/2602.06036v2.pdf]
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

## EAGLE Family: Feature-Level Speculative Decoding

The EAGLE series dramatically improves acceptance rates by reconsidering *what* the draft model predicts:

### [[eagle]] (Jan 2024 / Mar 2025)
Performs autoregression at the **feature (second-to-top-layer) level** rather than token level. Features vary more smoothly than discrete tokens, making prediction easier. Resolves feature uncertainty by inputting the token sequence shifted one time step ahead. Trains only a lightweight single-layer decoder plug-in on top of the frozen target model. Achieves **2.7×–3.5× speedup** (lossless), with draft accuracy ~0.8 vs Medusa's ~0.6.

### [[eagle-2]] (Jun 2024)
EAGLE-1 uses a static draft tree (fixed candidates per position). EAGLE-2 observes acceptance rates are **context-dependent** and that EAGLE's draft model is well-calibrated. Uses confidence scores to build **dynamic draft trees** at runtime — expanding branches where confidence is high, pruning where low. No extra training. **3.05×–4.26× speedup**, 20–40% over EAGLE-1.

### [[eagle-3]] (Apr 2025)
Removes the feature prediction constraint entirely, switching to **direct token prediction + multi-layer feature fusion** (low/mid/high-level target features). Fixes the resulting distribution shift via **training-time test**: Step 2 of training uses the draft model's own imperfect Step 1 output as input, matching test conditions. Unlocks a **data scaling law** — acceptance rate grows proportionally with training data (EAGLE-1/2 showed near-flat scaling). **Up to 6.5× speedup**, ~1.4× over EAGLE-2.

### [[dflash]] (May 2026, ICML 2026)
Replaces autoregressive drafting with a **block diffusion adapter** conditioned on target model hidden features. All γ draft tokens are generated in a *single parallel forward pass* (constant drafting cost, independent of γ), breaking the linear scaling wall of AR drafters. **Over 6× lossless acceleration**, up to **2.5× over EAGLE-3** on math/code benchmarks (GSM8K 5.15×, Math500 6.08×, AIME25 5.62×).

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

## Speculative Speculative Decoding (SSD / Saguaro)

Standard SD retains a sequential dependency: the draft model waits for verification before speculating the next round. **[[saguaro]]** (Kumar, Dao, May — Stanford/Princeton/Together AI, 2026) eliminates this by running speculator and verifier on **separate hardware in parallel**:

1. Speculator predicts likely verification outcomes (which tokens were accepted, what bonus token was sampled)
2. Pre-speculates token sequences for each likely outcome, storing them in a "speculation cache"
3. When verification completes: cache hit → return pre-speculated tokens immediately (zero draft overhead); cache miss → fall back to synchronous speculation

Results on Llama-3.1-70B (TP=4 H100) with Llama-3.2-1B draft: **30% faster than strongest SD baselines, up to 5× faster than AR**. Lossless — produces identical distribution to the target model.

Key challenge: predicting the bonus token (sampled from residual distribution) with ~90% accuracy using draft logits. See [[saguaro]] for the three optimizations (outcome prediction, acceptance/speculation tradeoff, batch-adaptive fallback).

## Limitations

- Requires a suitable draft model (additional memory footprint) — self-speculative eliminates this at some quality cost
- Gains diminish if draft model is poor (acceptance rate drops)
- Memory bandwidth savings only realized when compute is truly the free resource
- SSD requires separate hardware for speculator and verifier

## See Also

- [[kv-cache]] — orthogonal memory bottleneck technique
- [[layerskip]] — self-speculative decoding via early exit layers
- [[early-exit-inference]] — concept page for early exit and layer skipping
- [[continuous-batching]] — serving scheduler; composes with speculative decoding
- [[deepseek-v3-2]] — uses scalable RL for improved reasoning, orthogonal to inference-time optimizations
- [[saguaro]] — SSD: parallelizes drafting and verification across separate hardware; 30% faster than SD baselines
- [[deepseek-v4]] — architectural KV reduction via CSA/HCA, complementary to speculative decoding
- [[eagle]] — feature-level AR drafting; 2.7×–3.5× speedup, lossless
- [[eagle-2]] — dynamic draft trees; 3.05×–4.26×, 20–40% over EAGLE-1
- [[eagle-3]] — direct token prediction + training-time test; up to 6.5×, data scaling law unlocked
- [[dflash]] — block diffusion drafting; constant draft cost; 6×+ lossless, 2.5× over EAGLE-3
- [[block-diffusion]] — DFlash's draft engine architecture
- [[diffusion-language-models]] — DFlash uses diffusion as an AR model accelerator
- [[inference-kv-speculative]] — deep-dive companion: full EAGLE family section, KV compression detail, SSD algorithm
