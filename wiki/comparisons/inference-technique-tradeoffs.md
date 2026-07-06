---
title: Inference Technique Trade-offs — Cross-Cutting Comparison
created: 2026-07-06
updated: 2026-07-06
type: comparison
tags: [comparison, inference, survey]
sources: []
confidence: high
---

# Inference Technique Trade-offs — Cross-Cutting Comparison

Split out of [[inference-improvements-summary]] on 2026-07-06 (page-size threshold). One-line
trade-off summary for every technique covered across the inference-efficiency survey pages, for
quick side-by-side scanning.

| Technique | What it trades | Gain |
|---|---|---|
| SageAttention (INT8) | ~0% accuracy; plug-and-play | Attention compute 2.1× FA2; 340 TOPS RTX4090 |
| SageAttention2 (INT4/FP8) | ~0% accuracy; plug-and-play | Attention compute 3× FA2; 481 TOPS RTX4090 |
| SageAttention3 (FP4, Blackwell) | ~0% accuracy; Blackwell only | Attention compute 5× FA2; 1038 TOPS RTX5090 |
| DualPath (agentic KV loading) | System complexity (dual-path infra) | 1.87× offline throughput; 1.96× online (agentic) |
| GQE (query-head MoE) | Router training overhead | 1.7–1.8× prefill speedup at ≥32k (matches GQA quality) |
| AttnRes (Block) | O(Nd) depth-attention memory; architectural change at training time | 1.25× compute advantage; +7.5 GPQA-Diamond; mitigates PreNorm dilution |
| GQA/DSA/CSA+HCA | Model quality (marginal) | KV cache ↓ 10–90% |
| MoE | Memory (all experts must load) | FLOPs/token ↓ |
| INT4 weights | Quality (marginal at INT8, moderate at INT4) | Memory ↓ 2–4× |
| H₂O pruning | Retrieval quality | Throughput ↑ 29× |
| TriAttention | Offline calibration; still eviction | Throughput ↑ 2.5× or KV ↓ 10.7× at matched accuracy (reasoning) |
| PolarQuant / TurboQuant | Small quality loss | KV memory ↓ 3–5× |
| SpectralQuant | 15s calibration | KV memory ↓ 5.95×; +1.7–2.8 pp cosine sim vs TurboQuant; 4.5× decode speedup |
| Speculative decoding | Requires draft model | Latency ↓ 2–3× (lossless) |
| EAGLE (feature-level AR drafting) | Draft model training; frozen target | Latency ↓ 2.7–3.5× (lossless) |
| EAGLE-2 (dynamic draft trees) | Context-dependent tree expansion logic | Latency ↓ 3.05–4.26× (lossless, no extra training) |
| EAGLE-3 (training-time test) | Larger draft training dataset | Latency ↓ up to 6.5×; data scaling law unlocked |
| DFlash (block diffusion drafting) | Draft adapter training; constant draft overhead | Latency ↓ 6×+; 2.5× over EAGLE-3; lossless |
| DSpark (semi-AR + confidence scheduler) | Draft model training; confidence head calibration | +25–30% accepted length; +51% serving throughput (DeepSeek-V4-Flash @ 80 TPS SLA) |
| Saguaro (SSD) | Separate speculator hardware; prediction overhead | Latency ↓ 5× vs AR, 30% over SD (lossless) |
| SuperBPE (superword tokenization) | Tokenizer retraining (CPU/memory intensive); no inference changes | Tokens ↓ 33%; inference FLOPs ↓ 32%; downstream +4.0% avg (30 tasks) |
| Self-speculative (LayerSkip) | Draft quality vs separate model | Latency ↓ 1.3–2.2× (lossless, no extra memory) |
| Flash Attention | Recomputes during backward pass | Attention IO ↓ 7.6×; memory O(N) |
| PagedAttention | Block table indirection overhead | KV fragmentation ↓ ~0%; throughput ↑ 2–4× |
| RadixAttention | Tree lookup overhead | Cross-request prefix reuse; throughput ↑ 2–4× over vLLM |
| Continuous batching + chunked prefill | Scheduling complexity | Decode throughput ↑ 4–10× |
| Early exit / layer skipping | Quality on hard tokens | Latency ↓ 1.3–2× per token |
| BD3-LM (block diffusion) | Fixed block size hyperparameter | Parallel within-block generation + KV caching restored to DLMs |
| I-DLM (introspective DLM) | Training on 4.5B extra tokens | 3.1× over SDAR; matches AR quality; AR-serving-stack compatible |
| Nemotron 3 Ultra (Mamba-Attention hybrid) | Architectural change at pretraining time; Mamba-2 replaces most attention | ~6× inference throughput vs. SOTA open LLMs at on-par accuracy |

## See Also

- [[inference-improvements-summary]] — full inference survey overview this table was split from
- [[kv-cache-compression-comparison]] — narrower head-to-head: H₂O vs PolarQuant vs TurboQuant vs TriAttention vs SpectralQuant
- [[hyper-connections-variants]] — comparison of doubly-stochastic residual constructions
- [[kv-cache-compression-detail]] — detail behind the KV cache rows
- [[speculative-decoding-detail]] — detail behind the speculative decoding rows
