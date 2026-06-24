---
source_url: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
ingested: 2026-06-24
sha256: N/A (markdown clipping, already in papers/)
---

# A Visual Guide to Quantization

**Author:** Maarten Grootendorst  
**Published:** 2024-02-19  
**Source:** Substack newsletter (maartengrootendorst.com)  
**Original file:** papers/A Visual Guide to Quantization.md

## Summary

A comprehensive visual explainer (50+ illustrations) covering quantization for LLMs. Covers:

- **Floating point representation:** IEEE-754; sign, exponent, fraction bits; FP32 vs FP16 vs BF16
- **Data types for ML:** FP32 (full precision), FP16, BF16 (brain float), INT8, INT4; trade-offs in precision vs memory
- **Weight quantization:**
  - Absmax quantization (scale by max absolute value)
  - Zero-point quantization (asymmetric; shifts range to cover asymmetric distributions)
  - Symmetric vs asymmetric quantization; calibration datasets
- **Post-training quantization (PTQ):** GPTQ, AWQ, GGUF; apply to pretrained weights
- **Quantization-aware training (QAT):** simulate low precision during training; e.g., QLoRA uses 4-bit NF4 quantization
- **KV cache quantization:** quantizing key/value tensors reduces memory at inference
- **Activation quantization:** more challenging due to outliers; SmoothQuant migrates outliers to weights
- **Data types:** NF4 (Normal Float 4 — optimal for normally distributed weights), INT8, INT4, 1-bit

## Key Takeaways

- Weight outliers are the main challenge: a few large values force coarse quantization grids for the majority
- Groupwise quantization (per-group scales) improves quality at the cost of extra metadata
- QLoRA makes 4-bit finetuning practical: quantize base model to NF4, finetune adapters in FP16
- KV cache quantization is orthogonal to weight quantization — addresses inference memory, not storage

## Relevance to Wiki

Supplementary source for [[quantization]] concept page. See [[kv-cache]] for KV cache specifics, [[polarquant]], [[turboquant]], [[spectralquant]] for modern KV quantization methods.
