---
source_url: https://huggingface.co/blog/codelion/optimal-model-architecture
ingested: 2026-06-24
sha256: b9e7a5c3d1b9e7a5c3d1b9e7a5c3d1b9e7a5c3d1b9e7a5c3d1b9e7a5c3d1b9e7
---

# The Optimal Architecture for Small Language Models

**Author**: Asankhaya Sharma (Hugging Face blog)  
**Published**: 2025-12-26  
**Type**: Blog post / empirical study — MEDIUM CONFIDENCE. Well-documented experiments with reproducible setup. Main caveat: small parameter count (70M), single dataset recipe fixed from prior work, single GPU, limited token budget (1B).

## Experimental Setup

- ~70M parameters (62–77M range), 1B training tokens
- 19 model configurations × 12 architecture families
- Fixed dataset: 50% FinePDFs + 30% DCLM + 20% FineWeb-Edu
- Single NVIDIA A40, BF16, AdamW + cosine schedule
- Evaluated on HellaSwag, PIQA, WinoGrande, ARC-C, MMLU, TruthfulQA, GSM8K

## Key Findings

### Depth-Width
- **Two-tier split**: models score ~38% or ~32%, nothing in between. Tier gap = 6pp; within-tier variance = 0.5pp.
- **Hidden ≥ 512 threshold**: below threshold, depth must be exactly 32 or ≥64 to compensate
- **32-layer Goldilocks**: best overall at 38.50% (vs 38.15% for 12-layer baseline)

### Architecture Families (32-layer, ~70M)
- All 12 modern architectures within ~1% of each other in accuracy
- AR models: 32–33% avg; Diffusion models: 31–32% avg (lower accuracy, ~6% gap)
- Best AR: LLaMA3-Canon (33.22%), GPT-2 32L (33.18%)

### Diffusion Models
- dLLM (MDLM objective): 289 tok/s vs 50 tok/s for LLaMA3 (5.8×); dLLM-Canon: 31.81% avg, **49.27% TruthfulQA** (highest of all)
- Dhara-70M (LLaMA3-Canon → WSD conversion): 183 tok/s, 31.85% avg, 47.50% TruthfulQA
- Best factuality hypothesis: bidirectional context + iterative refinement + non-AR generation reduces hallucination compounding

### WSD AR→Diffusion Conversion
- 100M tokens (10×) fewer than from-scratch dLLM training; matches or exceeds scratch quality
- WSD: progressive block size increase (warmup) → full MDLM training (stable)

### Canon Layers
- Depthwise causal convolutions from "Physics of Language Models" paper
- +1–2% TruthfulQA at 0.13% parameter overhead (LLaMA3: +1.0pp; dLLM: +2.19pp)

## Result: Dhara-70M
- LLaMA3-Canon (1B tokens) → WSD conversion (100M tokens)
- 183 tok/s, 31.85% avg, 47.50% TruthfulQA
- Available: huggingface.co/codelion/dhara-70m
