---
source_url: N/A (local PDF, no arXiv ID — author: Ashwin Gopinath, Sentra/MIT, April 2026)
ingested: 2026-06-24
sha256: N/A (pdf-derived summary)
---

# 3% Is All You Need: Breaking TurboQuant's Compression Limit via Spectral Structure

**Authors:** Ashwin Gopinath  
**Affiliation:** Sentra (235 2nd Street, San Francisco, CA), MIT Department of Mechanical Engineering  
**Venue:** Preprint, April 2026  

## Summary

SpectralQuant exploits a universal spectral property of KV cache representations: effective dimensionality concentrates in ≈3–4% of the head dimension across models (Qwen, Llama, Mistral, Gemma). By rotating into signal-aligned eigenvector coordinates and applying QJL error correction only to signal dimensions, SpectralQuant achieves better compression AND better quality than TurboQuant simultaneously.

## The Spectral Discovery

Across six models in four families, including Gemma 2-9B (256-dim heads):
- `d_eff/d ≈ 3–4%` universally (d_eff ≈ 4 on standard 128-dim heads; ≈8 on 256-dim heads)
- 96–97% of dimensions carry noise rather than signal
- The spectral gap is stable across data splits (CV = 3.9%)

**Key insight:** TurboQuant's QJL error correction, applied uniformly to all dimensions, injects variance into noise dimensions without reducing their bias — because there's no signal to recover. Removing QJL from noise dimensions yields +1.7–2.8pp better cosine similarity at 18.6% better compression.

## SpectralQuant Algorithm

Three-part modification of TurboQuant:
1. **Calibrated eigenvector rotation** (instead of random Hadamard): aligns coordinates with actual signal structure (15 seconds of calibration on a single GPU)
2. **Selective QJL:** Apply 1-bit QJL error correction only to signal dimensions (~3%); omit from noise dimensions
3. **Non-uniform bit allocation:** Assign more bits to signal dimensions

## Results (Qwen 2.5-14B)

- **Cosine similarity: 0.9485** vs TurboQuant's 0.9226 (+2.59pp)
- **Compression ratio: 5.95×** vs TurboQuant's 5.02× (better compression AND better quality)
- **2.2× per-step speedup** at 512 tokens vs TurboQuant
- Perplexity identical to uncompressed inference (9.51) for all methods
- Perfect needle-in-a-haystack retrieval at all context lengths up to 8,192 tokens
- Results replicate across Qwen 1.5B, 7B, 14B; Llama 3.1-8B; five independent random seeds

## Relevance to Wiki

Primary source for [[spectralquant]] entity page. See [[turboquant]] (predecessor it improves upon), [[polarquant]] (polar coordinate approach), [[kv-cache-compression-comparison]], [[kv-cache]] for fundamentals.
