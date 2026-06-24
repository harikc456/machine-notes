---
title: DualPath
created: 2026-06-24
updated: 2026-06-24
type: entity
tags: [inference, kv-cache]
sources: [raw/papers/2602.21548v2.md]
confidence: high
---

# DualPath

**DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference**  
*Wu, Chen, Zhong, Huang, Tan, Zhang, Zhang, Zhou, Liu, Zhou, Zhang, Jin, Huang — Peking University / Tsinghua University / DeepSeek-AI, Feb 2026*

## Context: The Agentic Inference Bottleneck

Modern LLM deployments are shifting toward **agentic** (multi-turn, tool-calling) workloads. These create a fundamentally different performance profile from single-turn chat:

- KV-cache hit rates ≥95% — the model re-reads nearly all context from cache on every turn
- Append length only ~429 tokens per turn — very short new computation
- Cache-compute ratio: ~22 GB of KV cache to load per PFLOP of computation (for DeepSeek-V3.2)
- This makes agentic inference **I/O-bound**, not compute-bound

### The Physical Bottleneck

In **PD-disaggregated** (prefill-decode disaggregated) systems, all KV-cache data flows through a single path:

```
Storage → SNIC (storage NIC) → Prefill Engine → CNIC (compute NIC) → Decode Engine
```

The SNIC on prefill engines becomes fully saturated (100% utilization), while decode engines' SNICs sit idle. GPUs on prefill engines run at 40% utilization because they're waiting for data. Provisioning more SNIC bandwidth to prefill engines is expensive and rarely available in standard clusters.

## Solution: Dual-Path Loading

DualPath adds a **storage-to-decode path**:

```
Storage → Decode Engine SNIC → Decode Engine Memory → CNIC → Prefill Engine
```

The decode engine's storage NIC is now utilized. Data transfers from decode to prefill via high-speed RDMA over the compute network (CNIC bandwidth > SNIC bandwidth in standard NVIDIA DGX SuperPOD configs).

This is combined with:
1. **CNIC-centric traffic management**: isolates KV-cache RDMA traffic from model inference collective communications (all-reduce etc.) to prevent interference
2. **Dynamic global scheduler**: assigns each request's KV-cache load to whichever path is less congested; balances compute and network utilization across all engines simultaneously

## Key Technical Insight

The compute network (CNIC, 400 Gbps east-west) has *higher aggregate bandwidth* than the storage network (SNIC) but is used in short intermittent bursts for collective operations. DualPath repurposes the idle CNIC bandwidth during decode phases for KV-cache transport — aggregate storage bandwidth effectively doubles without any hardware changes.

## Results

| Metric | Baseline (PD-disagg) | DualPath | Gain |
|---|---|---|---|
| Offline inference throughput | 1× | 1.87× | +87% |
| Online serving throughput | 1× | 1.96× (avg) | +96% avg |
| SLO violation | baseline | none | maintained |

Evaluated on 3 production model sizes with realistic agentic workloads (coding assistant trajectory: mean 157 rounds, mean 32.7k context, 98.7% KV hit rate).

## Relationship to Other Work

- Addresses a serving-layer bottleneck orthogonal to [[kv-cache]] compression (eviction/quantization) — DualPath still loads *all* KV cache, just via a better transport path
- Composable with [[radix-attention]] (prefix sharing reduces total KV cache volume to load) and [[paged-attention]] (manages where KV lives on-device)
- Related to Mooncake (DRAM-pool KV caching) but targets the *loading bandwidth* problem, not the *storage capacity* problem

## See Also

- [[kv-cache]] — KV cache fundamentals; DualPath addresses loading bandwidth, not compression
- [[paged-attention]] — on-device KV management; orthogonal
- [[radix-attention]] — prefix sharing reduces KV volume; composable with DualPath
- [[continuous-batching]] — serving infrastructure context
