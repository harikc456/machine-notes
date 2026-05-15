---
title: Memory Reduction Techniques for LLM Training and Inference
created: 2026-05-15
updated: 2026-05-15
type: query
tags: [survey, training, inference, quantization, kv-cache, optimization, sparsity, attention]
sources: []
confidence: high
---

# Memory Reduction Techniques for LLM Training and Inference

A structured survey of techniques that reduce peak or steady-state memory in LLM workflows, organized by phase. Memory pressure manifests differently in each phase: training is dominated by activations, optimizer states, and gradients; inference is dominated by model weights and the KV cache.

---

## Part I — Training

### 1. Mixed-Precision Training

**Core idea:** Store weights in FP32 but compute forward/backward passes in FP16 or BF16. A master FP32 copy is maintained for numerically stable weight updates; intermediate results use the lower-precision format.

**Memory impact:** Reduces activation and gradient buffers by ~2× during the forward/backward computation. The FP32 master copy adds back some weight memory, but activations (the largest consumer) shrink.

**BF16 vs FP16:** BF16 matches FP32's dynamic range (8 exponent bits) and is preferred for LLMs — FP16 clips large gradient values, requiring loss scaling workarounds. BF16 is the standard for frontier model training.

**FP8 training:** An emerging further step — forward and backward passes in FP8, master weights in BF16/FP32. Cuts activation memory another ~2× vs BF16. [[deepseek-v4]] uses FP8 training at scale; requires careful per-tensor scaling.

**Wiki cross-reference:** [[quantization]] covers data type trade-offs in detail.

---

### 2. Gradient Checkpointing (Activation Recomputation)

**Core idea:** During the forward pass, discard most intermediate activations instead of holding them in memory. During backpropagation, recompute the discarded activations on-the-fly from the nearest retained checkpoint.

**Memory impact:** Peak activation memory drops from O(L) (proportional to number of layers) to O(√L) with uniform checkpointing — at the cost of one additional forward pass worth of compute (~33% training overhead).

**Variants:**
- *Full recomputation*: Retain only inputs; recompute all activations backward. Maximum memory savings, maximum compute cost.
- *Selective recomputation*: Retain activations that are cheap to store but expensive to recompute; recompute only the cheap-to-recompute ones. Flash Attention (see §6) already recomputes attention weights rather than storing the full N×N matrix — this is selective recomputation applied specifically to attention.

**Practical use:** Almost universal in large-scale training. Often combined with mixed precision to further reduce the memory of retained checkpoints.

---

### 3. ZeRO — Zero Redundancy Optimizer

**Core idea (DeepSpeed ZeRO, Rajbhandari et al. 2020):** In standard data-parallel training, every GPU holds a full replica of weights, gradients, and optimizer states. ZeRO partitions these across GPUs so each device holds only its shard, communicating as needed.

**Three stages:**

| Stage | What is partitioned | Memory per GPU | Communication overhead |
|---|---|---|---|
| ZeRO-1 | Optimizer states | ~4× reduction | Low (allreduce gradients as usual) |
| ZeRO-2 | Optimizer states + gradients | ~8× reduction | Moderate (reduce-scatter gradients) |
| ZeRO-3 | Optimizer states + gradients + parameters | Linear reduction in # GPUs | Higher (allgather params each forward) |

A 175B model in FP32 with Adam optimizer states requires ~2.8 TB total. With 64 GPUs and ZeRO-3, this shrinks to ~44 GB per GPU — within reach of 80GB A100s.

**ZeRO-Infinity:** Extends ZeRO-3 to offload partitions to CPU RAM or NVMe SSDs, enabling training of trillion-parameter models on commodity hardware clusters. The bottleneck shifts from GPU memory to PCIe/NVMe bandwidth.

**Trade-off:** ZeRO-3 doubles communication volume vs. standard data parallelism. At sufficient scale, intra-node NVLink bandwidth hides this overhead; across nodes it can become a bottleneck.

---

### 4. Gradient Accumulation

**Core idea:** Instead of one large batch per optimizer step, accumulate gradients over several smaller micro-batches before applying the update. The optimizer step sees the same effective batch size, but peak GPU memory sees only one micro-batch at a time.

**Memory impact:** Linear reduction in activation and intermediate-gradient memory proportional to the accumulation factor. No extra communication.

**Limitation:** Does not reduce weight or optimizer state memory — those are fixed per-device. Primarily useful when the bottleneck is activation memory from large batches, or to simulate large batch sizes across multiple steps.

**Interaction with ZeRO:** Gradient accumulation and ZeRO are complementary; combined they address both the per-step activation peak and the persistent optimizer state overhead.

---

### 5. Parameter-Efficient Fine-Tuning (PEFT) / LoRA

**Core idea:** Instead of fine-tuning all weights, freeze the pretrained model and train only a small number of additional parameters. LoRA (Hu et al. 2021) reparameterizes weight updates as low-rank matrices: ΔW = AB where A ∈ ℝ^{d×r}, B ∈ ℝ^{r×k}, r ≪ min(d, k).

**Memory impact:**
- Weight memory: full model weights still loaded (inference-identical), but only A and B are updated → 90–99% reduction in trainable-parameter gradient+optimizer-state memory
- For a typical 7B model, LoRA with rank 16 trains ~40M parameters vs. 7B → ~175× reduction in optimizer state memory
- Activations also shrink proportionally to the LoRA paths

**Variants:**
- *QLoRA*: Quantize the frozen base weights to NF4 (4-bit) + LoRA on top. Enables fine-tuning a 65B model on a single 48GB GPU.
- *DoRA*: Decomposes weight updates into magnitude and direction components for better expressivity vs. LoRA.
- *Prefix Tuning / Prompt Tuning*: Even fewer parameters, but lower expressivity.

**Trade-off:** LoRA adapters cannot represent arbitrary weight updates (rank constraint). For full fine-tuning quality, ZeRO is the right tool; for resource-constrained fine-tuning, LoRA/QLoRA dominate practice.

---

### 6. Flash Attention

**Core idea (Dao et al., 2022/2023):** Standard attention computes and materializes the full N×N attention score matrix (N = sequence length), consuming O(N²) memory. Flash Attention tiles the computation into blocks that fit in SRAM, computing attention without ever materializing the full matrix. It uses an online softmax formulation to produce correct outputs with only O(N) HBM memory.

**Memory impact:** Attention's HBM footprint drops from O(N²) to O(N). For N = 32K, this is a 32K× reduction in attention memory alone. This is what makes long-context training (64K–1M tokens) tractable.

**Compute trade-off:** Reads input tiles multiple times (redundant HBM reads) but eliminates the dominant O(N²) write. Net result is faster (7.6× on GPT-2 per [[flash-attention]]) because HBM bandwidth, not FLOP count, is the bottleneck.

**Recomputation role:** Flash Attention recomputes attention weights during the backward pass rather than storing them — this is selective activation recomputation (§2) applied to attention.

**Wiki cross-reference:** [[flash-attention]] entity page has full benchmarks.

---

### 7. Mixture of Experts (MoE) — Training Memory Perspective

**Core idea:** Rather than a dense FFN that activates all parameters per token, MoE routes each token to a small subset of expert FFNs. Per-token active parameter count is decoupled from total model capacity.

**Memory impact:** A model with total parameters P and k/N expert activation rate has the same peak forward-pass memory as a dense model of size P × (k/N). All expert weights must still be stored (or partitioned via expert parallelism), but the activation memory footprint per token is drastically smaller.

**[[deepseek-v4]]:** 1.6T total parameters, 49B active per token (MoE-Pro). Effective active parameter ratio ≈ 3%. Activation memory profile resembles a 49B dense model while benefiting from 1.6T capacity.

**Infrastructure cost:** Expert parallelism requires all-to-all routing communication. Load imbalance causes idle capacity. Both are active engineering problems in frontier training.

**Wiki cross-reference:** [[mixture-of-experts]] covers routing, load balancing, and infrastructure in detail.

---

### 8. Optimizer Memory Reduction

**Standard Adam:** For each parameter θ, Adam stores θ (weight), g (gradient), m (first moment), v (second moment) — 4 copies in FP32 → 16 bytes/param.

**8-bit Adam (Dettmers et al., 2022):** Stores optimizer states m and v in INT8 with dynamic quantization. Reduces optimizer state memory ~4× with negligible accuracy loss for most tasks. Compatible with ZeRO partitioning.

**Muon (Kosson et al.):** Orthogonalizes gradients before applying momentum. Removes the v (second moment) entirely — only the weight and one momentum buffer, vs. Adam's two momentum buffers. [[deepseek-v4]] adopts Muon for non-embedding parameters. Memory: roughly 2 copies per param (weight + one moment) vs. 4 for Adam.

**Adafactor:** Factorizes the second-moment matrix (v) into row/column factors for matrix-shaped parameters, reducing optimizer state from O(d²) to O(d) per matrix. No per-element second moment stored. Used in T5/PaLM training.

**Wiki cross-reference:** [[deepseek-v4]] uses Muon with weight normalization in combination.

---

## Part II — Inference

### 9. Weight Quantization (PTQ)

**Core idea:** After training, quantize model weights to lower-bit representations. The model is loaded in quantized form; weights are dequantized to FP16 for matrix multiplications (W8A16) or kept quantized for fully quantized matmuls (W8A8).

**Memory impact:** INT8 → 2× reduction vs FP16 weights. INT4 → 4× reduction. A 70B FP16 model fits in 140 GB; INT4 shrinks it to ~35 GB — from two A100s to one.

**Key methods:**
- *GPTQ* (Frantar et al., 2022): Layer-wise second-order quantization. Minimizes per-layer output error. Near-lossless to 4-bit for large models.
- *AWQ* (Lin et al., 2023): Identifies and protects weight channels with large activation magnitudes before quantization. Works without calibration data for most tasks.
- *SmoothQuant*: Migrates quantization difficulty from activations to weights by scaling channels; enables W8A8 (fully quantized) inference.

**Trade-offs:** Quantization error accumulates across layers. Very low bitwidth (2-bit) suffers perplexity degradation. Extreme methods like BitNet (1-bit weights) require training from scratch.

**Wiki cross-reference:** [[quantization]] covers data types, calibration, and the random preconditioning insight (shared with KV quantization).

---

### 10. KV Cache Compression

The KV cache is often the dominant inference memory consumer at long sequence lengths or large batches (can exceed weight memory — see [[kv-cache]] for the formula).

#### 10a. Eviction — H₂O

**Core idea ([[h2o]]):** Not all tokens are equally important for future attention. H₂O maintains a budget-bounded cache by evicting tokens with low accumulated attention scores while always keeping recent tokens (recency window).

**Memory impact:** Reduces KV cache to a fixed fraction (e.g., 5–20%) of full sequence length. Memory bounded by budget size, independent of actual sequence length.

**Limitation:** Eviction is irreversible — retrieving an evicted token is impossible. Needle-in-haystack retrieval tasks suffer.

#### 10b. Quantization — PolarQuant, TurboQuant

**Core idea:** Instead of evicting tokens, reduce the bit-width of all stored K and V tensors.

- **[[polarquant]]:** Converts K/V vectors to polar coordinates; angular component quantized more aggressively than radial (aligned with attention sensitivity). Eliminates per-block normalization overhead via Hadamard preconditioning. Achieves >4.2× compression.

- **[[turboquant]]:** Random rotation (Hadamard) + MSE quantizer + 1-bit QJL residual. Data-oblivious (no calibration dataset needed). Near-neutral perplexity at 3.5 bits/element.

**Shared insight:** Both use random Hadamard preconditioning to make vector coordinates approximately i.i.d., enabling parameter-free normalization. Traditional per-block normalization stores FP32 scale factors that dominate at low bitwidth.

#### 10c. Architectural KV Reduction

**Core idea:** Modify the attention mechanism to produce fewer K/V pairs structurally.

- **Multi-Query Attention (MQA):** One shared K/V head for all query heads. KV cache shrinks by factor n_heads. Used in Falcon, Mistral.
- **Grouped-Query Attention (GQA):** Groups of g query heads share one K/V head. KV shrinks by n_heads/g. Used in Llama-3 (default g=8), Gemma. Interpolates between MHA (g=n_heads) and MQA (g=1).
- **Multi-Head Latent Attention (MLA):** Projects K/V into a low-rank latent vector before caching. Only the latent is cached; K/V are reconstructed at decode time. Used in DeepSeek-V3/V4 — 5–13× KV reduction vs. standard MHA.
- **CSA/HCA ([[deepseek-v4]]):** Compressed Self-Attention and Hybrid Cache Attention; 3.7–9.8× KV reduction vs DeepSeek-V3.2 by combining latent compression with selective full-head caching.

**Wiki cross-reference:** [[deepseek-v4]] and [[kv-cache]] cover MLA and CSA/HCA in depth.

#### 10d. Eviction + Quantization — Combining Approaches

Quantization and eviction are **complementary**: quantization losslessly retains all tokens at reduced precision; eviction aggressively shrinks token count. A combined policy (e.g., quantize the budget, evict the rest) can achieve higher compression than either alone. See [[kv-cache-compression-comparison]] for a direct comparison.

---

### 11. PagedAttention

**Core idea ([[paged-attention]]):** The KV cache is allocated in fixed-size non-contiguous "pages" (analogous to OS virtual memory pages), managed by a block table that maps logical sequence positions to physical GPU memory blocks.

**Memory impact:** Eliminates internal and external fragmentation in KV allocation. In traditional serving, reserved contiguous buffers waste 20–80% of memory due to fragmentation. PagedAttention brings this to near-zero waste.

**Throughput effect:** More efficient memory → more concurrent requests → 2–4× higher throughput vs TGI (vLLM benchmarks). Peak memory per request is unchanged, but memory utilization efficiency is greatly improved.

**Wiki cross-reference:** [[radix-attention]] extends this with cross-request prefix sharing via a radix tree.

---

### 12. Flash Attention at Inference

Flash Attention (see §6) also reduces memory at inference time for the **prefill** phase. For long prompts, the N×N attention matrix at prefill is the peak memory moment. Flash Attention's O(N) memory profile makes 100K+ token prompts feasible on constrained hardware.

For the decode phase (one new token per step), attention memory is smaller and KV cache management (§10) dominates.

---

### 13. Speculative Decoding — Memory Perspective

**Core idea ([[speculative-decoding]]):** A small draft model generates candidate tokens; the large target model verifies a batch in parallel. Lossless 2–3× throughput improvement.

**Memory impact:** Standard speculative decoding *increases* total memory (draft model + target model loaded simultaneously). The benefit is throughput, not memory.

**Self-speculative decoding ([[layerskip]]):** Uses the target model's own early layers as the draft — no extra model weights. Memory overhead is zero; the trade-off is lower draft quality bounded by early-layer representational power.

**Memory-throughput trade-off:** Standard speculative decoding trades higher memory for higher throughput. At constrained memory budgets, self-speculative or no speculative decoding is preferred.

---

### 14. Early Exit / Layer Skipping

**Core idea ([[early-exit-inference]]):** Not all tokens require all layers. Adaptive policies skip later transformer layers for "easy" tokens.

**Memory impact:** Reduces *compute* and *activation* memory per token during decode; does not reduce weight memory (all layers still resident). At high skip rates, effective compute and intermediate buffer footprint shrink proportionally.

**Methods:**
- **LayerSkip (Meta):** Training-time layer dropout enables early exit; self-speculative verification uses skipped states.
- **SWIFT:** Plug-and-play, no retraining; adaptive layer selection per token.
- **DASH:** MDP policy for per-token skip decisions; input-aware.

---

### 15. Mixture of Experts — Inference Memory

**Core idea:** MoE activates only a subset of expert FFN weights per token. Total model memory must accommodate all expert weights, but per-token *activation* memory is small.

**Expert offloading:** For models where total expert weight exceeds GPU VRAM (e.g., a 1.6T MoE), experts can be offloaded to CPU RAM and prefetched on demand. The routing decision for the next batch can be predicted, overlapping prefetch with compute. [[engram]] demonstrates <3% overhead for a related 100B-parameter lookup table with prefetching.

**Continuous batching interaction ([[continuous-batching]]):** With many concurrent requests, different tokens activate different experts — statistical multiplexing reduces per-request effective expert memory pressure.

---

## Summary Table

| Technique | Phase | Primary Memory Target | Key Trade-off |
|---|---|---|---|
| Mixed precision (BF16/FP8) | Training | Activations, gradients | Training stability; FP8 needs scaling |
| Gradient checkpointing | Training | Activations | +33% compute for recomputation |
| ZeRO-1/2/3 | Training | Optimizer states, gradients, weights | Higher inter-GPU communication |
| Gradient accumulation | Training | Activations (per step) | No weight/optimizer savings |
| LoRA / QLoRA | Fine-tuning | Optimizer states, gradients | Rank constraint limits expressivity |
| Flash Attention | Training + Inference | Attention matrix (O(N²)→O(N)) | Redundant tile reads (still net faster) |
| MoE | Training + Inference | Active activations per token | Routing overhead, expert communication |
| 8-bit Adam / Muon / Adafactor | Training | Optimizer states | Minor accuracy impact (8-bit Adam) |
| Weight quantization (INT8/INT4) | Inference | Model weights | Accuracy degradation at very low bits |
| KV eviction (H₂O) | Inference | KV cache | Irreversible; risky for retrieval tasks |
| KV quantization (PolarQuant/TurboQuant) | Inference | KV cache | Transform compute overhead |
| MQA / GQA / MLA / CSA | Inference | KV cache | Potential attention quality loss (MQA) |
| PagedAttention | Inference | KV cache fragmentation | Minimal (OS paging is near-zero cost) |
| Speculative decoding | Inference | — (increases memory) | Throughput gain, not memory gain |
| Self-speculative decoding | Inference | Draft model weights (zero extra) | Lower draft quality |
| Early exit / layer skipping | Inference | Activation memory per token | Weight memory unchanged |
| Expert offloading | Inference | Expert FFN weights | PCIe bandwidth bottleneck |

---

## Cross-Cutting Themes

**Recomputation vs. storage trade-off:** Several techniques deliberately trade compute for memory: gradient checkpointing recomputes activations, Flash Attention recomputes attention weights, speculative decoding runs extra forward passes. The invariant is that modern hardware is compute-rich relative to memory bandwidth — recomputing is often cheaper than storing and loading.

**Architectural vs. post-hoc techniques:** Architectural changes (MQA, GQA, MLA, MoE, Flash Attention) must be baked in at training time. Post-hoc techniques (quantization, eviction, LoRA, speculative decoding) can be applied to existing checkpoints. For new model development, architectural choices dominate memory at scale.

**Stacking:** Most techniques are composable. A typical frontier deployment stacks: MoE + GQA/MLA + Flash Attention (architectural) + INT8/INT4 weights + KV quantization + PagedAttention + continuous batching (system). Training stacks: MoE + Flash Attention + gradient checkpointing + ZeRO-3 + BF16/FP8 + Muon.

**The memory-throughput frontier:** Memory reduction and throughput improvement are often coupled — smaller KV caches allow larger batches (PagedAttention, GQA), which amortize fixed memory overhead and improve hardware utilization.

---

## See Also

- [[kv-cache]] — KV cache mechanics and compression landscape
- [[quantization]] — weight and KV quantization in depth
- [[flash-attention]] — IO-aware tiled attention
- [[paged-attention]] — OS-style KV memory management
- [[radix-attention]] — cross-request prefix sharing
- [[mixture-of-experts]] — MoE architecture and infrastructure
- [[h2o]] — KV eviction via heavy-hitter oracle
- [[polarquant]] — polar KV quantization
- [[turboquant]] — vector KV quantization
- [[kv-cache-compression-comparison]] — H₂O vs PolarQuant vs TurboQuant
- [[speculative-decoding]] — draft-verify inference speedup
- [[layerskip]] — self-speculative decoding via early exit
- [[early-exit-inference]] — early exit and layer skipping landscape
- [[continuous-batching]] — serving scheduler that composes with memory reduction
- [[deepseek-v4]] — CSA/HCA, MLA, FP8, Muon at frontier scale
- [[inference-improvements-summary]] — prior inference-focused survey
