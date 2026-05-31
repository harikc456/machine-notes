---
title: Research Gaps in Memory Reduction and Inference Optimization
created: 2026-05-19
updated: 2026-05-31
type: query
tags: [survey, inference, kv-cache, quantization, sparsity, attention, comparison]
sources: []
confidence: medium
---

# Research Gaps in Memory Reduction and Inference Optimization

A critical-thinking pass over the wiki's memory/inference coverage ([[memory-reduction-survey]], [[inference-improvements-summary]], [[kv-cache-compression-comparison]]) to surface **methodological weaknesses, untested compositions, narrow validation regimes, and unfalsified shared premises** in the current literature.

Most "gaps" below are not "this technique doesn't exist" — they are claims that have been *asserted but not rigorously tested*, or compositions everyone assumes will work.

**Confidence is `medium`**: coverage is bounded by the wiki's current 51 pages. Some gaps may already be addressed by published work not yet ingested.

---

## 1. Methodological Gaps (Evaluation Validity)

### 1a. Non-comparable benchmarking
Across [[h2o]], [[triattention]], [[polarquant]], [[turboquant]], [[spectralquant]], each paper reports *different* primary metrics: peak VRAM, compression ratio, perplexity delta, throughput at fixed batch, task accuracy. There is **no standardized memory-reduction benchmark**.

- *Validity threat:* construct + statistical-conclusion validity. "5.95× compression" (SpectralQuant) is not directly comparable to "10.7× reduction" (TriAttention) — the former measures bits/element on retained tokens, the latter measures tokens retained at a fixed bit-width.
- *Gap:* a CONSORT-style reporting standard for KV/weight compression with mandatory disclosure of (i) context length distribution, (ii) prefill vs decode peak, (iii) downstream task accuracy not just PPL, (iv) hardware specifics.

### 1b. Perplexity ≠ capability — "lossless" is overclaimed
Most KV quantization papers ([[polarquant]], [[turboquant]]) declare losslessness via perplexity matching. [[spectralquant]] is rare in using cosine similarity on attention outputs. **None** report long-horizon reasoning accuracy.

- *Bias:* outcome switching — perplexity is the easy outcome; the *interesting* one is whether 32K-token chain-of-thought still solves the problem.
- *Gap:* standardized reasoning-degradation curves (AIME, SWE-bench, GPQA) vs compression ratio.

### 1c. Single-task long-context evaluation
[[triattention]] establishes pre-RoPE stability on AIME25 — one task, one distribution.

- *External validity threat.* Does pre-RoPE stability hold for code generation, multi-document QA, agentic tool-use traces (very different cache distributions)?
- *Gap:* a multi-domain long-context KV-eviction stability suite.

---

## 2. Coverage Gaps (Phases & Regimes Underexplored)

### 2a. Prefill memory peak is the forgotten phase
The survey is decode-dominated. [[flash-attention]] addresses the O(N²) attention matrix during prefill, but the **prefill KV write itself** is not addressed by [[h2o]], [[triattention]], [[paged-attention]], or [[spectralquant]] — all decode-phase compressors. For 100K+ token prompts, prefill is the OOM point.

- *Gap:* streaming-prefill KV compression that compresses *as* keys/values are produced.

### 2b. KV compression × MLA/CSA — does it stack?
MLA ([[deepseek-v4]]) already projects K/V into a low-rank latent. [[spectralquant]], [[turboquant]], etc. assume standard MHA/GQA KV. **No work** in the wiki on quantizing an MLA latent further without breaking the low-rank assumption.

- *Gap:* structural compression × statistical compression composition. The natural next paper for any rotation-based quantizer.

### 2c. Eviction × quantization Pareto frontier
[[kv-cache-compression-comparison]] notes that eviction and quantization are complementary, but the **quantitative Pareto frontier** (memory vs accuracy under both knobs) is absent. Is "quantize the budget, evict the rest" dominated by either alone? Unknown.

- *Gap:* a joint search over (budget, bit-width) on a fixed reasoning task.

### 2d. Diffusion LM memory landscape
[[block-diffusion]] and [[i-dlm]] use block-level (not token-level) caches. **The entire KV optimization stack (eviction, quantization, MLA, paged attention) is targeted at AR transformers.** None has been ported to DLMs.

- *Gap:* memory optimization for diffusion-LM blockwise caches — see [[diffusion-language-models]]. Fertile, low-competition area.

### 2e. Quantization-aware training is absent from the wiki
[[quantization]], [[polarquant]], [[turboquant]], [[spectralquant]] are all post-training or online. **QAT for KV** (training with simulated quantized KV in the loop) is unrepresented.

- *Gap:* training-aware KV/weight compression — the wiki has no entry for this branch.

### 2f. Pretraining-as-fine-tuning memory (GaLore, ReLoRA)
[[memory-reduction-survey]] covers LoRA/QLoRA strictly for fine-tuning. GaLore (low-rank gradient projection for *pretraining*) and ReLoRA are missing — likely a wiki coverage gap rather than a literature gap.

---

## 3. Unfalsified Shared Premises (Where the Subfield Could Be Wrong Together)

### 3a. "Effective dimensionality d_eff ≈ 3-4% of head dim" ([[spectralquant]])
A *universal* premise across model families tested — but "tested" = a finite set of pretrained dense/MoE LLMs.

- *Hasty generalization risk.* Does it hold for: (i) fine-tuned alignment models with warped distributions, (ii) reasoning-RL'd models like [[deepseek-v3-2]] (RL changes key statistics), (iii) MoE expert keys (per-expert distributions)?
- *Gap:* empirical study of d_eff stability across post-training stages.

### 3b. "Compute-rich vs memory-bandwidth-bound" assumption
The "recompute over store" theme of [[memory-reduction-survey]] assumes modern accelerators are compute-rich. **On Tenstorrent / Graphcore / TPUv5 / mobile NPUs the FLOPs/bandwidth ratio differs by >10×.** All Flash Attention / gradient checkpointing benchmarks in the wiki are GPU-only.

- *External validity threat.* The optimal memory-reduction stack on a TPU or NPU is likely different.
- *Gap:* hardware-architecture co-design — entirely missing from the wiki's taxonomy.

### 3c. Post-RoPE importance is *always* unstable ([[triattention]]'s premise)
[[triattention]] argues post-RoPE estimation fails because only recent queries are oriented to current K rotations. This is a single mechanistic hypothesis. **No ablation** tests whether the instability is due to RoPE specifically or to *any* position encoding. ALiBi, NoPE, YaRN cache-importance estimation is unstudied.

- *Gap:* position-encoding × eviction-stability matrix.

### 3e. "Diffusion drafters are high-quality across all task types" ([[dflash]] premise)
[[dflash]] achieves 6×+ speedup on math/code (GSM8K, AIME25) but only 2.75× on MT-Bench (conversational). This gap is unexplained in the paper — it could be that target model hidden states encode fewer future-token signals in open-ended dialogue than in structured reasoning, making the diffusion adapter less accurate as a drafter.

- *Gap:* systematic analysis of when diffusion drafting acceptance rates drop. Needed to determine the boundary conditions for DFlash's 2.5× advantage over EAGLE-3.

### 3d. "Self-speculative is free" ([[layerskip]] premise)
[[layerskip]] adds zero parameters for the draft, but draft quality is bounded by early-layer representational power. Empirical results are on Llama-class dense models.

- *Gap:* does self-speculation work on MoE? On reasoning-RL'd models where late layers may diverge more from early? No evidence in the wiki — see [[speculative-decoding]], [[saguaro]].

---

## 4. Compositional Gaps (Stacks Asserted but Not Measured)

The "Stacking" theme in [[memory-reduction-survey]] is the most under-evidenced section:

> "Inference stacks: MoE + GQA/MLA + Flash Attention + INT8/INT4 weights + KV quantization + PagedAttention + continuous batching"

Asserted as additive. **No paper in the wiki runs the full stack and measures additive memory savings vs. interaction effects.** Likely interactions worth measuring:

| A × B | Suspected interaction |
|---|---|
| INT4 weights × KV quant | Compounding numerical error → accuracy collapse below thresholds |
| MLA × KV quant | Low-rank latent has different statistics than raw K/V → quant methods miscalibrated |
| Eviction × speculative decoding | Verifier's KV state may diverge from drafter's after eviction |
| EAGLE-3 data scaling × KV quantization | More training data improves acceptance rate — does applying KV quant at inference degrade the acceptance rate faster than standard SD? |
| DFlash × tensor parallelism | Diffusion adapter is conditioned on target hidden states — correctness of this conditioning under TP sharding is unverified |
| AttnRes × MLA | [[attnres]] reads preceding-layer features; latent attention may not preserve them |
| MoE × KV eviction | Experts attend to different patterns → per-expert eviction policies? |

- *Gap:* systematic factorial study of composition. High-value, low-novelty paper waiting to be written.

---

## 5. Distribution-Shift / Robustness Gaps

All KV eviction policies are tuned on benchmark distributions. **No adversarial / red-team evaluation** in the wiki. Plausible failure modes:

- Adversarial prompts targeting evicted tokens with later questions (memory exfiltration of the *forgotten* part of the context).
- OOD prompts (mixed-language, code+text, long system prompts) where SpectralQuant's eigenvector calibration is no longer valid.

- *Gap:* robustness benchmarks for compressed-cache inference.

---

## 6. Process / Reporting Gaps

- **No preregistration culture** in this subfield. Most quantization papers state hypotheses post-hoc; [[kv-cache-compression-comparison]] would benefit from a registered benchmark with frozen evaluation prompts.
- **Hidden class restrictions in optimality claims.** [[turboquant]] proves optimality *within the data-oblivious class* — easy to miss, and [[spectralquant]] exploits it. A pattern: many "optimal" claims in this space have unstated scope restrictions.

---

## Recommended High-Value Gap Targets

Ranked by tractability × impact:

1. **MLA-aware KV quantization** (§2b) — direct, one-paper-sized, large impact for any DeepSeek-style deployment.
2. **Streaming-prefill KV compression** (§2a) — clear problem, no incumbent.
3. **Factorial composition study** (§4) — low novelty, high citation value.
4. **Position-encoding × eviction-stability** (§3c) — would falsify or extend [[triattention]]'s central claim.
5. **DLM memory optimization** (§2d) — entire stack to be re-derived for [[block-diffusion]] / [[i-dlm]].
6. **DFlash acceptance rate degradation analysis** (§3e) — understanding why diffusion drafting underperforms on open-ended dialogue vs. structured reasoning; needed to scope the deployment envelope for [[dflash]] over [[eagle-3]].

---

## Caveats

- Coverage assessment is bounded by the wiki's current 51 pages; some "gaps" may exist as published work not yet ingested. Recommended: a focused literature search on each gap before claiming novelty.
- Critique is structural — not independently re-validated against source PDFs in this analysis pass.
- The "no standardized benchmark" claim (§1a) is synthesized from variance in metrics across wiki entries; community efforts (MLPerf-Inference variants, EfficientML benchmark proposals) may already exist and warrant a search.

---

## See Also

- [[memory-reduction-survey]] — the survey this critique is grounded in
- [[inference-improvements-summary]] — companion inference survey
- [[kv-cache-compression-comparison]] — the comparison page that motivates §2c
- [[kv-cache]] — KV cache mechanics
- [[quantization]] — weight and KV quantization overview
- [[triattention]] — pre-RoPE eviction, see §1c and §3c
- [[spectralquant]] — d_eff universality claim, see §3a
- [[deepseek-v4]] — MLA/CSA, see §2b
- [[diffusion-language-models]] — DLM landscape, see §2d
- [[dflash]] — diffusion drafting for SD; see §3e
- [[eagle-3]] — EAGLE-3 data scaling law; see §4 interaction rows
- [[inference-kv-speculative]] — deep-dive on EAGLE family + KV compression; the detail layer behind the gaps above
