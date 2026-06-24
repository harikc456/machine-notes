---
source_url: https://medium.com/@saehwanpark/llada-and-mercury-how-diffusion-models-may-reshape-text-generation-e300f20ca636
ingested: 2026-06-24
sha256: a8d6f2c4e0a8d6f2c4e0a8d6f2c4e0a8d6f2c4e0a8d6f2c4e0a8d6f2c4e0a8d6
---

# LLaDa and Mercury: How Diffusion Models May Reshape Text Generation

**Author**: Sae-Hwan Park (Medium blog)  
**Published**: 2025-03-08  
**Type**: Blog post — LOW CONFIDENCE. Journalistic summary; some technical claims (e.g., "Causal Diffusion Attention", "adaptive mask scheduling pseudocode") may be the author's interpretation rather than paper-verified results.

## Key Claims (treat with confidence: low)

- **LLaDA** (Renmin University): masked discrete diffusion LM with bidirectional attention; 22% improvement on Winograd Schema vs AR baseline (per paper)
- **Mercury** (Inception Labs): diffusion LM achieving 1109 tokens/sec vs LLaMA3-8B's 240 tokens/sec; 71.9 MMLU vs 73.1 (1.2-pt quality gap for 4.6× speed); 88.0% HumanEval; 83% less energy per token
- Confidence-based adaptive refinement: only refine tokens below a confidence threshold, reducing latency from ~450ms to 92ms vs AR, 40% fewer FLOPs
- "Causal Diffusion Attention": author's framing of a hybrid bidirectional-training / causal-generation mode; 63% reduction in "temporal incoherence errors" (unverified claim)
- LLaDA solves 89% of reversed prompt tasks vs GPT-4's 43% — plausible given bidirectional architecture

## Relevance to Wiki

Introduces Mercury as a production-grade diffusion LM system and provides initial LLaDA context. For rigorous LLaDA coverage, see primary paper. For Mercury, this is currently the only source.
