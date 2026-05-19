---
title: SpectralQuant
created: 2026-05-19
updated: 2026-05-19
type: entity
tags: [kv-cache, quantization, inference]
sources: [papers/spectralquant.pdf]
confidence: high
---

# SpectralQuant

**3% Is All You Need: Breaking TurboQuant's Compression Limit via Spectral Structure**
*Ashwin Gopinath (Sentra / MIT), April 2026*

## Overview

SpectralQuant is a data-aware KV cache compression algorithm that exploits a universal structural property of transformer attention heads: KV cache key vectors concentrate ≈96–97% of their variance in just 3–4% of dimensions (d_eff ≈ 4 out of 128). It modifies [[turboquant]] in three principled ways to achieve strictly better quality *and* better compression at the same bit budget — at the cost of 15 seconds of one-time calibration.

## The Core Discovery: The 97% Spectral Gap

Across six models and four families (Qwen, Llama, Mistral, Gemma), with head dimensions of both 128 and 256, the effective dimensionality of KV cache key vectors clusters at d_eff/d ≈ 3.1–3.4%:

| Model | head_dim | d_eff^keys | d_eff/d |
|---|---|---|---|
| Qwen 2.5-1.5B | 128 | 3.95 | 3.1% |
| Qwen 2.5-7B | 128 | 4.30 | 3.4% |
| Qwen 2.5-14B | 128 | 4.17 | 3.3% |
| Llama 3.1-8B | 128 | 3.64 | 2.9% |
| Mistral 7B | 128 | 4.18 | 3.3% |
| Gemma 2-9B | 256 | 8.20 | 3.2% |

The participation ratio PR(Σ) = (Σλ_i)² / Σλ_i² formalizes effective dimensionality. The spectral gap ratio κ = λ_{⌈d_eff⌉} / λ_{⌈d_eff⌉+1} ≈ 1.2, creating a natural signal/noise cut-point. This structure is stable: CV across calibration splits = 3.9%.

**Asymmetry**: d_eff^keys << d_eff^values — key vectors are far more spectrally concentrated than value vectors, explaining why low-rank compression works on keys but not values.

## Algorithm

Five stages (calibration is one-time per model):

**1. Calibration** (~15 seconds on a single B200 GPU)
- Collect 100 calibration sequences from WikiText-2 validation set
- Compute empirical covariance: Σ̂ = (1/N) Σ h_t h_t^⊤
- Extract eigenvectors U via `torch.linalg.eigh(Σ̂)`
- Compute d_eff = PR(Σ̂); set d_s = ⌈d_eff⌉ ≈ 4
- Train Lloyd-Max codebooks C_signal (signal dims) and C_noise (noise dims)
- Sample JL matrix A ∈ ℝ^{m×d_s} (for signal dims only)

**2. Spectral rotation**
- h̃ = U^⊤ h  (project into eigenvector coordinates)
- First d_s components = signal; remaining d − d_s = noise

**3. Non-uniform quantization**
- Signal dims: quantize with C_signal (more bits)
- Noise dims: quantize with C_noise (fewer bits)

**4. Selective QJL**
- Apply Johnson-Lindenstrauss error correction *only* to signal dims
- Skip QJL on all noise dims (this is the primary quality + compression gain)

**5. Decompression**
- Reverse quantization, apply inverse rotation U, recover ĥ

## Why Removing QJL from Noise Dims Improves Quality

[[turboquant]]'s QJL correction is designed for high-signal dimensions where quantization bias is meaningful. For a noise dimension with true value x ≈ 0:
- Quantization error ε_q ≈ 0 (already tiny)
- QJL estimates this error from a noisy projection: ê_q = A^⊤ A ε_q
- 𝔼[ê_q] ≈ 0 (unbiased) but Var(ê_q) > 0

Removing QJL sets x̂ = 0 directly: same zero expectation, zero variance. By bias-variance decomposition, selective correction strictly reduces MSE on noise dimensions. Ablation confirms: removing QJL from a random-rotation baseline gains +3.0 pp cosine similarity.

The analogy: running spell-check on random characters produces garbage. TurboQuant's uniform QJL on noise dims is analogous — it "corrects" noise into structured noise.

## Bit Accounting (SQ_noQJL_v3 vs TQ 3-bit)

| Component | TurboQuant | SpectralQuant | Difference |
|---|---|---|---|
| Signal (4 dims) | 3b quant + QJL | 3b quant + QJL | Same |
| Noise (124 dims) | 3b quant + QJL | 3b quant only | SQ drops QJL |
| Avg bits/element | 3.19 | 2.69 | −0.50 bits |
| Compression ratio | 5.02× | 5.95× | +18.6% |

## Results

On Qwen 2.5-14B (primary experiment):
- CosSim: **0.9485** vs TurboQuant 0.9226 (**+2.59 pp**)
- Compression: **5.95×** vs 5.02× (+18.6% better)
- Memory at 8K context: 270.5 MB vs 320.9 MB (−50.4 MB)

All four models tested:

| Model | TQ CosSim | SQ CosSim | Gain (pp) |
|---|---|---|---|
| Qwen 2.5-1.5B | 0.8443 | 0.8615 | +1.72 |
| Qwen 2.5-7B | 0.8538 | 0.8815 | +2.77 |
| Qwen 2.5-14B | 0.9226 | 0.9485 | +2.59 |
| Llama 3.1-8B | 0.9454 | 0.9628 | +1.74 |

- **Perplexity**: 9.51, identical to uncompressed inference
- **Needle-in-haystack**: perfect retrieval up to 8,192 tokens
- **Latency**: 4.5× faster attention decoding at 512 tokens vs TurboQuant; 2× per-step at 512 tokens

## Relationship to Other Work

- [[turboquant]] — direct predecessor; SpectralQuant modifies TurboQuant's three core design choices (random→calibrated rotation, uniform→non-uniform quantization, uniform→selective QJL)
- [[polarquant]] — also uses rotation preconditioning; data-oblivious like TurboQuant
- [[h2o]] — eviction (complementary, not competing)
- [[kv-cache-compression-comparison]] — side-by-side with all methods
- [[quantization]] — general quantization landscape
- [[kv-cache]] — KV cache bottleneck background
