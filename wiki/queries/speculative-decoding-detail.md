---
title: Speculative Decoding — Detail
created: 2026-07-06
updated: 2026-07-06
type: query
tags: [inference, speculative]
sources: []
confidence: high
---

# Speculative Decoding — Detail

Split out of [[inference-kv-speculative]] on 2026-07-06 (page-size threshold). Companion to
[[kv-cache-compression-detail]]. For the high-level survey, see
[[inference-improvements-summary]]. For memory-impact framing, see
[[memory-inference-techniques]]. For research gaps, see [[memory-inference-research-gaps]].

---

## Speculative Decoding

[[speculative-decoding]] — Leviathan et al., ICML 2023

**Problem**: autoregressive LLMs are memory-bandwidth-bound, not compute-bound. The GPU can process many tokens in parallel but the algorithm forces sequential generation.

**Insight**: if the model is bandwidth-bound, extra compute is "free" — as long as we stay within the same memory access budget.

### Algorithm

1. A small **draft model** generates γ tokens autoregressively (fast, cheap)
2. The **target model** verifies all γ tokens in a single parallel forward pass
3. Accept tokens greedily using a rejection sampling scheme:
   - If draft token probability ≤ target probability at that position: **accept**
   - Otherwise: **reject** with probability `1 - p_target/p_draft`; resample from adjusted distribution; discard remaining draft tokens
4. Guaranteed: the output distribution exactly matches the target model (no approximation)

### Why It Works

The target model processes γ+1 tokens in one pass, costing roughly the same memory bandwidth as 1 token. If the draft model has high acceptance rate (α), the expected tokens per target forward pass is `(1 - α^{γ+1}) / (1 - α)` — approaching γ+1 when α is high.

### Results

- **2–3× speedup** on typical text generation benchmarks
- **Exact distributional match** to target model — not an approximation
- Works best when the draft model's distribution is close to the target model's

### The Draft Model

The draft model is the key variable:
- Smaller version of the same model family (e.g., Llama-3.1-8B drafts for Llama-3.1-70B)
- Specialized draft head trained on top of the target model's early layers
- Self-drafting (Medusa): multiple draft heads attached to the target model

### EAGLE Family: Feature-Level and Beyond

The EAGLE series reconsiders *what* the draft model predicts and how it is structured, achieving far higher acceptance rates than vanilla SD.

#### [[eagle]] — Feature-Level AR Drafting (Mar 2025)

**Key insight**: Feature sequences (second-to-top-layer hidden states) are smoother and easier to autoregressively predict than discrete token sequences. EAGLE trains a single lightweight transformer decoder plug-in that autoregressively predicts the next feature, then uses the frozen target LM head to convert it to a token distribution.

**Uncertainty resolution**: Since the next feature depends on which token was sampled (e.g., "am" vs "always" yield different continuations), EAGLE feeds the actual token sequence shifted one time step ahead as an additional input — resolving the sampling ambiguity.

**Draft accuracy**: ~0.8, vs ~0.6 for Medusa and lower for Lookahead. **2.7×–3.5× speedup** on LLaMA2-Chat 70B, lossless in both greedy and non-greedy settings.

#### [[eagle-2]] — Context-Dependent Dynamic Draft Trees (Jun 2024)

EAGLE uses a static draft tree (fixed number of candidates per position). EAGLE-2 observes acceptance rates are **context-dependent** — easy queries need fewer branches, hard ones need more. It leverages the fact that EAGLE's draft model is well-calibrated (its confidence scores ≈ true acceptance rates) to dynamically expand or prune the draft tree at runtime.

**No extra training needed.** Works directly on any EAGLE checkpoint. **3.05×–4.26× speedup**, 20–40% over EAGLE-1.

#### [[eagle-3]] — Training-Time Test (Apr 2025)

**Root cause of EAGLE-1/2 data scaling plateau**: the feature prediction loss (l_fea) constrains the draft model's expressiveness; scaling training data hits diminishing returns because the constraint, not data, is the bottleneck.

**Fix — direct token prediction + multi-layer feature fusion**: remove l_fea; predict tokens directly; fuse low-, mid-, and high-level target features as conditioning.

**Problem this creates**: distribution shift at step 2. Step 1 now produces an unconstrained vector â (not a true feature f̂), so step 2 sees out-of-distribution input at inference time.

**Training-time test**: during training, step 2 is fed â from step 1 (not the ground-truth feature). This exactly matches the inference distribution, closing the shift.

**Result**: **up to 6.5× speedup**, ~1.4× over EAGLE-2. Critically, acceptance rate now scales proportionally with training data — a data scaling law that was absent in EAGLE-1/2.

#### [[dflash]] — Block Diffusion for Parallel Drafting (ICML 2026)

**Root cause of all AR-based SD ceiling**: drafting is still sequential — T_draft = γ × t_step grows linearly with speculation length, capping practical speedups at ~2–3× even with high acceptance rates.

**Fix**: replace AR drafting with a **block diffusion adapter** conditioned on the target model's hidden features. All γ draft tokens are generated in a **single parallel forward pass** (T_draft = t_parallel, constant). "The target knows best" — large AR model hidden states implicitly encode multi-step future-token information; the diffusion adapter reads these features to generate high-quality parallel drafts without being large itself.

**Results on Qwen3-8B (SGLang)**:

| Benchmark | EAGLE-3 | DFlash |
|---|---|---|
| GSM8K | 2.23× | 5.15× |
| Math500 | 2.05× | 6.08× |
| AIME25 | 2.05× | 5.62× |
| HumanEval | 2.17× | 5.14× |
| MBPP | 1.93× | 4.65× |
| MT-Bench | 1.90× | 2.75× |

Over 6× lossless acceleration on math/code, **2.5× over EAGLE-3** across most tasks. MT-Bench lower (2.75×) — conversational tasks have less concentrated future-token signal in hidden states.

#### [[dspark]] — Semi-Autoregressive + Confidence Scheduling (Jun 2026)

**Root causes addressed**: (1) parallel drafters suffer *suffix decay* — positions after the first are predicted independently of one another, causing acceptance rates to fall sharply for later tokens. (2) Verifying a fixed-length draft regardless of confidence wastes verifier compute when drafts are unlikely to be accepted.

**Semi-Autoregressive Generation (SAG)**:
- Parallel backbone (DFlash): generates base logits `U_k` for all γ positions simultaneously — O(1) draft cost preserved
- Markov head (sequential refinement): computes transition bias conditioned on the previous sampled token:

  `y_k = U_k + B_k`  where  `B(x_{k-1}, x_k) = (W_1[x_{k-1}]) W_2ᵀ[x_k]`

  Low-rank lookup; ~1.5% latency overhead; eliminates suffix decay — acceptance rate stays flat across the entire draft block.

**Confidence-Scheduled Verification**:
- Confidence head: predicts `c_k` = P(token k accepted | tokens 1..k-1 accepted)
- Calibrated via Sequential Temperature Scaling (STS) — ensures predicted probabilities match empirical acceptance rates
- Prefix survival: `a_{r,j} = ∏_{i≤j} c_{r,i}`

**Hardware-Aware Prefix Scheduler**:
Maximizes `Θ = τ × SPS(B)` (expected accepted tokens × profiled GPU steps/sec at batch size B).

Greedy per-request: select draft token additions ranked by marginal gain in expected accepted tokens per unit batch-size cost. Load-adaptive: more verification tokens when lightly loaded (minimize latency); prune low-confidence tails under congestion (maximize throughput for all users).

**Results**:

| Setting | Result |
|---|---|
| Offline avg accepted length | +25–30% over AR baselines |
| Suffix decay | Eliminated (flat acceptance across positions) |
| DeepSeek-V4-Flash @ 80 TPS/user SLA | **+51% aggregate throughput** vs MTP-1 |
| >120 TPS/user tier | **Newly achievable** under load |

DSpark vs prior work in the SD landscape:

| System | Draft cost | Suffix decay | Confidence-aware |
|---|---|---|---|
| EAGLE-3 | O(γ) | No | No |
| DFlash | O(1) | Yes | No |
| **DSpark** | **O(1)** | **No** | **Yes** |
| Saguaro (SSD) | varies | — | No |

[[saguaro]] and DSpark are orthogonal: DSpark improves draft quality and dynamically adjusts verification length per request; Saguaro parallelizes the draft–verify loop across separate hardware.

### Self-Speculative Decoding

Variant that eliminates the separate draft model. See [[early-exit-inference]] for full coverage.

- **[[layerskip]]** (Meta, 2024): layer dropout training → early layers (0..e) draft, full model verifies; reuses draft KV states; up to 2.16× speedup
- **SWIFT** (2025): no retraining; adaptively selects skip layers per token at runtime; 1.3–1.6×
- **DASH** (2025): MDP policy for per-token layer selection; input-aware

Trade-off: no extra model memory, but draft quality bounded by early-exit representation quality.

### Speculative Speculative Decoding (SSD / Saguaro)

[[saguaro]] — Kumar, Dao, May (Stanford / Princeton / Together AI), May 2026

**The remaining bottleneck in standard SD**: drafting and verification are still sequential — the draft model must wait for verification to finish before generating the next speculation. This idle time is the limiting factor.

**SSD eliminates this by running speculator and verifier on separate hardware in parallel**:
1. Draft model sends speculated tokens to verifier
2. While verification runs, the draft model **predicts the most likely verification outcomes** (k tokens accepted + which bonus token sampled)
3. Pre-speculates for each predicted outcome — stores in a "speculation cache"
4. When verification result arrives: cache hit → return pre-speculated tokens immediately (zero drafting latency); cache miss → synchronous fallback

**Key challenge — predicting the bonus token**: The bonus token is sampled from the residual distribution max(p_target − p_draft, 0). Saguaro uses draft logits to predict the most likely bonus token with ~90% accuracy.

**Results** (Llama-3.1-70B target, Llama-3.2-1B draft, TP=4 H100):
- **30% faster than strongest SD baselines** (vLLM, SGLang)
- **Up to 5× faster than autoregressive decoding**
- Lossless — same output distribution as target model
- Improves Pareto frontier across all batch sizes

**Distinction from tree-based SD**: tree methods increase *verifier* compute; SSD scales *speculator* compute with no extra verification overhead. Orthogonal and combinable.

---

## See Also

- [[inference-kv-speculative]] — overview page linking this and [[kv-cache-compression-detail]]
- [[kv-cache-compression-detail]] — companion detail page for KV cache compression
- [[inference-improvements-summary]] — full inference survey overview
- [[memory-inference-techniques]] — memory-focused inference survey with quantitative memory impact per technique
- [[memory-inference-research-gaps]] — methodological gaps, Pareto frontier analysis, untested compositions
- [[speculative-decoding]] — speculative decoding concept page
- [[eagle]] — feature-level AR drafting; 2.7–3.5× lossless
- [[eagle-2]] — dynamic draft trees; 3.05–4.26×; no extra training
- [[eagle-3]] — training-time test; up to 6.5×; data scaling law
- [[dflash]] — block diffusion parallel drafting; 6×+; 2.5× over EAGLE-3
- [[dspark]] — semi-AR draft (DFlash + Markov head) + confidence scheduler; +51% throughput DeepSeek-V4-Flash
- [[saguaro]] — SSD: parallel drafting + verification on separate hardware
- [[layerskip]] — Meta's self-speculative decoding via layer dropout
- [[early-exit-inference]] — early exit and layer skipping (LayerSkip, SWIFT, DASH)
- [[block-diffusion]] — BD3-LM: DFlash's draft engine architecture
- [[diffusion-language-models]] — DLMs as AR model accelerators (DFlash)
- [[nemotron-3-ultra]] — shared-weight MTP heads trained jointly with the base model as a built-in draft mechanism
