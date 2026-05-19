# Wiki Index

> Content catalog. Every wiki page listed under its type with a one-line summary.
> Read this first to find relevant pages for any query.
> Last updated: 2026-05-19 | Total pages: 44

## Entities

- [[attnres]] — Attention Residuals (Kimi Team, Mar 2026): depth-wise softmax attention over preceding layers replaces fixed residuals; Block AttnRes = 1.25× compute advantage, +7.5 GPQA-Diamond on 48B model
- [[block-diffusion]] — BD3-LM (ICLR 2025): block-level AR + within-block discrete diffusion; variable-length generation, KV caching, SOTA diffusion LM perplexity
- [[clip-to-grok]] — Clip to Grok: per-row weight norm clipping accelerates grokking 39–249× without weight decay
- [[flash-attention]] — Flash Attention (Dao et al.): IO-aware tiled attention; 7.6× speedup on GPT-2; O(N) memory via online softmax
- [[layerskip]] — LayerSkip (Meta): layer dropout training + early exit inference + self-speculative decoding; up to 2.16× speedup
- [[paged-attention]] — PagedAttention (vLLM): OS-style paged KV cache management; eliminates fragmentation; 2–4× throughput over TGI
- [[radix-attention]] — RadixAttention (SGLang): radix tree cross-request prefix caching with LRU eviction; 2–4× throughput over vLLM
- [[derf]] — Derf: `erf(αx+s)` point-wise replacement for normalization layers; surpasses LayerNorm across vision, speech, DNA
- [[lejepa]] — LeJEPA: provable JEPA with SIGReg enforcing isotropic Gaussian embeddings — eliminates training heuristics
- [[lewm]] — LeWorldModel: stable end-to-end JEPA world model from pixels using prediction loss + SIGReg; 15M params
- [[deepseek-v4]] — DeepSeek-V4 series (Pro 1.6T, Flash 284B): CSA/HCA hybrid attention, mHC, Muon optimizer, 1M-token context
- [[deepseek-v3-2]] — DeepSeek-V3.2: DSA attention, scalable RL, agentic task synthesis, gold-medal IMO/IOI performance
- [[engram]] — Conditional memory module using N-gram lookup; complement to MoE sparsity for static knowledge retrieval
- [[h2o]] — H₂O Heavy-Hitter Oracle: KV cache eviction policy retaining "heavy hitter" tokens via attention score accumulation
- [[i-dlm]] — I-DLM: introspective consistency training converts AR models to DLMs; ISD decoding; first DLM to match same-scale AR quality
- [[kromhc]] — KromHC: Manifold-constrained HC via Kronecker-product residual matrices — exact doubly-stochastic, parameter-efficient
- [[mhc]] — mHC (Manifold-Constrained Hyper-Connections): projects HC residual matrices onto Birkhoff polytope via Sinkhorn-Knopp
- [[mhc-lite]] — mHC-lite: replaces SK iterations with convex combination of permutation matrices for exact doubly-stochastic residuals
- [[polarquant]] — PolarQuant: KV cache quantization using polar coordinate transformation to eliminate normalization overhead
- [[qknorm]] — QKNorm: cosine-similarity attention (ℓ₂-normalize Q and K) prevents softmax saturation; +0.7 ppl in LM experiments
- [[saguaro]] — Saguaro (SSD): speculative speculative decoding — parallelizes drafting and verification on separate hardware; 30% faster than SD, up to 5× over AR
- [[triattention]] — TriAttention: KV cache compression via trigonometric series in pre-RoPE space; 2.5× throughput or 10.7× KV reduction at matched accuracy on AIME25
- [[turboquant]] — TurboQuant: near-optimal online vector quantization via random rotation + MSE quantizer + 1-bit QJL residual
- [[weight-normalization]] — Weight Normalization (Salimans & Kingma 2016): decouple weight direction from magnitude; ~21 ppl gain in LM
- [[xsa]] — XSA (Exclusive Self-Attention): subtracts own-value-direction from attention output, addressing attention similarity bias

## Concepts

- [[conditional-memory]] — Conditional memory as a sparsity axis: static lookup vs. neural computation in LLMs
- [[diffusion-language-models]] — Diffusion LM landscape: quality gap causes (gradient variance, introspective consistency), BD3-LM vs I-DLM
- [[continuous-batching]] — Iteration-level scheduling + chunked prefill (SARATHI): zero padding waste, 1.25–10× decode throughput
- [[early-exit-inference]] — Early exit and layer skipping: adaptive per-token compute via LayerSkip, SWIFT, DASH
- [[grokking]] — Delayed generalization: training memorizes early, generalizes much later; weight norm dynamics are key (Power et al. 2022)
- [[jepa]] — Joint Embedding Predictive Architecture: predict in latent space; SIGReg solves collapse without heuristics
- [[hyper-connections]] — Hyper-Connections (HC) family: extends residual connections with dynamic mixing matrices across multiple streams
- [[kv-cache]] — KV cache: what it is, why it's a bottleneck, and the landscape of compression/eviction approaches
- [[mixture-of-experts]] — Mixture-of-Experts (MoE): conditional computation that scales capacity without proportional FLOP increase
- [[normalization-free-transformers]] — Point-wise functions (Derf, DyT) as replacements for LayerNorm; properties and comparisons
- [[orthogonal-residual-streams]] — Unified view of XSA, OrthogonalMLPWrapper, and mHC: modules should write ⊥ to existing residual directions
- [[quantization]] — LLM quantization overview: weight quantization, KV cache quantization, data types, trade-offs
- [[speculative-decoding]] — Speculative decoding: draft-then-verify algorithm for lossless 2-3× inference speedup
- [[weight-norm-training]] — Practical synthesis: weight norm + QK norm for transformers; interaction with architecture choices

## Comparisons

- [[kv-cache-compression-comparison]] — H₂O vs PolarQuant vs TurboQuant: eviction vs. quantization approaches to KV cache compression
- [[hyper-connections-variants]] — mHC vs mHC-lite vs KromHC: trade-offs in doubly-stochastic residual matrix construction

## Queries

- [[inference-improvements-summary]] — LLM inference efficiency survey: architecture (GQA/MLA/MoE), weight quantization, KV cache pruning (H₂O) and compression (PolarQuant/TurboQuant), speculative decoding
- [[memory-reduction-survey]] — Comprehensive survey of memory reduction techniques for LLM training (ZeRO, gradient checkpointing, LoRA, Flash Attention, MoE) and inference (KV compression, weight quantization, MQA/GQA/MLA, PagedAttention)
