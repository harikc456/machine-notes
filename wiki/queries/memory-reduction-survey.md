---
title: Memory Reduction Techniques for LLM Training
created: 2026-05-15
updated: 2026-05-19
type: query
tags: [survey, training, quantization, optimization, attention, sparsity]
sources: []
confidence: high
---

# Memory Reduction Techniques for LLM Training

A structured survey of techniques that reduce peak or steady-state memory during LLM training and fine-tuning. Training memory pressure is dominated by activations, optimizer states, and gradients. For inference-time memory reduction (weight quantization, KV cache compression, speculative decoding), see [[memory-inference-techniques]]. For research gaps in the inference space, see [[memory-inference-research-gaps]].

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
| AttnRes (Block) | Training + Inference | Training activation norms (bounds O(L) growth); +O(Nd) depth-attn memory | Architectural; must be trained in; <4% training overhead |

---

## Cross-Cutting Themes

**Recomputation vs. storage trade-off:** Several techniques deliberately trade compute for memory: gradient checkpointing recomputes activations, Flash Attention recomputes attention weights. The invariant is that modern hardware is compute-rich relative to memory bandwidth — recomputing is often cheaper than storing and loading.

**Architectural vs. post-hoc techniques:** Architectural changes (MQA, GQA, MLA, MoE, Flash Attention, [[attnres]]) must be baked in at training time. Post-hoc techniques (LoRA, 8-bit Adam) can be applied to existing checkpoints. For new model development, architectural choices dominate memory at scale.

**Stacking:** Most techniques are composable. A typical frontier training stack: MoE + Flash Attention + gradient checkpointing + ZeRO-3 + BF16/FP8 + Muon.

---

## See Also

- [[memory-inference-techniques]] — inference memory techniques (KV cache compression, weight quantization, PagedAttention, speculative decoding)
- [[memory-inference-research-gaps]] — research gaps and untested compositions in the inference memory literature
- [[inference-improvements-summary]] — broader inference survey including architecture and serving
- [[kv-cache]] — KV cache mechanics and compression landscape
- [[quantization]] — weight and KV quantization in depth
- [[flash-attention]] — IO-aware tiled attention
- [[paged-attention]] — OS-style KV memory management
- [[radix-attention]] — cross-request prefix sharing
- [[mixture-of-experts]] — MoE architecture and infrastructure
- [[deepseek-v4]] — CSA/HCA, MLA, FP8, Muon at frontier scale
- [[attnres]] — Attention Residuals: bounds O(L) activation growth with O(Nd) overhead
