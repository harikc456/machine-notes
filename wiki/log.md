# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: `## [YYYY-MM-DD] action | subject`
> Actions: ingest, update, query, lint, create, archive, delete
> When this file exceeds 500 entries, rotate: rename to log-YYYY.md, start fresh.

## [2026-05-14] create | Wiki initialized
- Domain: LLM efficiency and architecture research
- Structure created: SCHEMA.md, index.md, log.md, raw/papers/, entities/, concepts/, comparisons/, queries/

## [2026-05-14] ingest | Batch ingest: 11 sources from papers/
- Sources ingested:
  - 2601.07372v1.pdf → Engram (Conditional Memory via Scalable Lookup)
  - DeepSeek_V4.pdf → DeepSeek-V4
  - 2512.02556v1.pdf → DeepSeek-V3.2
  - 2502.02617v1.pdf → PolarQuant
  - A Visual Guide to Quantization.md → quantization survey
  - 2601.05732v1.pdf → mHC-lite
  - 2211.17192v2.pdf → Speculative Decoding (Leviathan et al., ICML 2023)
  - 2306.14048v3.pdf → H₂O Heavy-Hitter Oracle (NeurIPS 2023)
  - 2601.21579v1.pdf → KromHC
  - 2512.24880v2.pdf → mHC (Manifold-Constrained Hyper-Connections)
  - 2504.19874v1.pdf → TurboQuant
- Pages created:
  - entities/deepseek-v4.md
  - entities/deepseek-v3-2.md
  - entities/engram.md
  - entities/h2o.md
  - entities/kromhc.md
  - entities/mhc.md
  - entities/mhc-lite.md
  - entities/polarquant.md
  - entities/turboquant.md
  - concepts/conditional-memory.md
  - concepts/hyper-connections.md
  - concepts/kv-cache.md
  - concepts/quantization.md
  - concepts/speculative-decoding.md
  - concepts/mixture-of-experts.md
  - comparisons/kv-cache-compression-comparison.md
  - comparisons/hyper-connections-variants.md

## [2026-05-14] ingest | Batch ingest: 4 new sources from papers/
- Sources ingested:
  - 1602.07868v3.pdf → Weight Normalization (Salimans & Kingma, OpenAI 2016)
  - cliptogrok.pdf → Clip to Grok: Weight Norm Clipping for Accelerated Generalization
  - 2010.04245v1.pdf → QKNorm: Query-Key Normalization for Transformers
  - 2603.09078v1.pdf → XSA: Exclusive Self-Attention (Zhai, Apple, Mar 2026)
- SCHEMA.md updated: added `normalization` and `grokking` tags to taxonomy
- Pages created:
  - entities/weight-normalization.md
  - entities/clip-to-grok.md
  - entities/qknorm.md
  - entities/xsa.md
  - concepts/grokking.md
  - concepts/orthogonal-residual-streams.md (synthesis page — no single source)
  - concepts/weight-norm-training.md (synthesis page)
- index.md updated: total pages 16 → 23

## [2026-05-14] ingest | Batch ingest: 4 new sources from papers/
- Sources ingested:
  - 2201.02177v1.pdf → Grokking: Generalization Beyond Overfitting (Power et al., OpenAI, Jan 2022)
  - 2603.19312v2.pdf → LeWorldModel: Stable End-to-End JEPA from Pixels (Maes, LeCun et al., Mar 2026)
  - 2511.08544v3.pdf → LeJEPA: Provable and Scalable SSL Without the Heuristics (Balestriero & LeCun, Nov 2025)
  - 2512.10938v2.pdf → Stronger Normalization-Free Transformers / Derf (Chen et al., Dec 2025)
- SCHEMA.md updated: added `normalization-free`, `ssl`, `world-model` tags to taxonomy
- Pages updated:
  - concepts/grokking.md — added 2201.02177v1.pdf as primary source; expanded with original paper findings
- Pages created:
  - entities/derf.md
  - entities/lejepa.md
  - entities/lewm.md
  - concepts/normalization-free-transformers.md
  - concepts/jepa.md
- index.md updated: total pages 23 → 30

## [2026-05-14] lint | 1 issue found
- Broken wikilink: [[ssl]] in concepts/jepa.md → fixed (removed brackets, replaced with plain text)
- No orphans, no frontmatter issues, no unknown tags, no long pages, no index gaps
- 19 single-source pages (expected — each is a single-paper entity/concept)
- log.md: 76 lines, 5 entries — well within rotation threshold

## [2026-05-14] ingest | Batch ingest: 5 new sources from arxiv
- Sources ingested:
  - 2205.14135v2.pdf → Flash Attention (Dao et al., Stanford, 2022)
  - 2309.06180v1.pdf → PagedAttention / vLLM (Kwon et al., UC Berkeley, 2023)
  - 2308.16369v2.pdf → SARATHI: Chunked Prefill for Efficient LLM Inference
  - 2404.16710v2.pdf → LayerSkip: Early Exit + Self-Speculative Decoding (Meta, 2024)
  - 2505.17420v1.pdf → DASH: Dynamic Layer Skipping via MDP (2025)
- Pages created:
  - entities/flash-attention.md
  - entities/paged-attention.md
  - entities/layerskip.md
  - concepts/continuous-batching.md (includes SARATHI chunked prefill)
  - concepts/early-exit-inference.md (includes DASH, SWIFT)
- Pages updated:
  - concepts/speculative-decoding.md — added self-speculative decoding section
- index.md updated: total pages 31 → 36

## [2026-05-14] query | LLM inference improvements — structured survey
- Filed synthesis of inference optimization techniques across 4 categories
- Pages created:
  - queries/inference-improvements-summary.md
- index.md updated: total pages 30 → 31; Queries section populated

## [2026-05-14] lint | 1 issue found (low severity)
- Broken wikilinks: 0
- Orphans: 1 — queries/inference-improvements-summary.md (expected: query pages are terminal nodes; no fix needed)
- Index completeness: 36/36 pages accounted for
- Frontmatter: all valid (title, created, updated, type, tags, sources, confidence present on all pages)
- Long pages: 1 — queries/inference-improvements-summary.md at 287 lines (accepted: comprehensive survey; query pages exempt from 200-line split rule)
- Tags: all 21 taxonomy tags valid, no unknown tags
- log.md: 119 lines, 9 entries — well within 500-entry rotation threshold

## [2026-05-14] ingest | 2312.07104v1.pdf → RadixAttention / SGLang
- Pages created:
  - entities/radix-attention.md
- Pages updated:
  - entities/paged-attention.md — added [[radix-attention]] wikilink to prefix caching section
- index.md updated: total pages 36 → 37

## [2026-05-14] update | inference-improvements-summary.md — extended with new pages
- Added §5 Serving Infrastructure: Flash Attention, PagedAttention, Continuous Batching/Chunked Prefill
- Added §6 Early Exit / Layer Skipping
- Added self-speculative decoding subsection to §4
- Updated cross-cutting themes table with 5 new rows
- Updated See Also with 6 new wikilinks

## [2026-05-15] query | Memory reduction techniques survey (training + inference)
- Synthesized 15 techniques across training and inference phases
- Training: mixed precision, gradient checkpointing, ZeRO, gradient accumulation, LoRA/QLoRA, Flash Attention, MoE, 8-bit Adam/Muon/Adafactor
- Inference: weight quantization (PTQ), KV eviction (H₂O), KV quantization (PolarQuant/TurboQuant), MQA/GQA/MLA/CSA, PagedAttention, Flash Attention at inference, speculative decoding, early exit, expert offloading
- Pages created:
  - queries/memory-reduction-survey.md
- index.md updated: total pages 37 → 38

## [2026-05-16] ingest | Batch ingest: 4 new papers
- Sources ingested:
  - 2503.09573v3.pdf → Block Diffusion / BD3-LM (Arriola et al., Cornell Tech/Stanford/Cohere, ICLR 2025)
  - 2603.03251v3.pdf → Speculative Speculative Decoding / Saguaro (Kumar, Dao, May — Stanford/Princeton/Together AI, 2026)
  - 2604.04921v1.pdf → TriAttention: KV compression via trigonometric series in pre-RoPE space (Mao et al., MIT/NVIDIA/ZJU, Apr 2026)
  - 2604.11035v1.pdf → I-DLM: Introspective Diffusion Language Model (Yu, Jian et al., Together AI/UIUC/Princeton/Stanford, Apr 2026)
- Pages created:
  - entities/block-diffusion.md
  - entities/saguaro.md
  - entities/triattention.md
  - entities/i-dlm.md
  - concepts/diffusion-language-models.md
- Pages updated:
  - concepts/speculative-decoding.md — added SSD/Saguaro section + [[saguaro]] See Also link
  - concepts/kv-cache.md — added TriAttention to eviction section + [[triattention]] See Also link
  - comparisons/kv-cache-compression-comparison.md — added TriAttention column and section; updated Recommended Use
- index.md updated: total pages 38 → 43

## [2026-05-16] update | inference-improvements-summary.md + memory-reduction-survey.md
- inference-improvements-summary.md:
  - Added §3b TriAttention (pre-RoPE KV eviction for long-context reasoning)
  - Renumbered prior §3b KV quantization → §3c
  - Added SSD/Saguaro subsection to §4 (speculative decoding)
  - Added §7 Diffusion Language Models (BD3-LM, I-DLM) as new inference paradigm
  - Updated cross-cutting themes table: TriAttention, Saguaro, BD3-LM, I-DLM rows
  - Updated See Also: 6 new wikilinks
- memory-reduction-survey.md:
  - Extended §10a to cover TriAttention alongside H₂O; explains pre-RoPE stability advantage
  - Extended §13 (speculative decoding) to mention Saguaro and SSD memory model
  - Updated summary table: TriAttention and Saguaro rows added
  - Updated See Also: triattention, saguaro wikilinks added

## [2026-05-19] ingest | Attention Residuals (AttnRes) — Kimi Team, arXiv 2603.15031v1
- Source: papers/2603.15031v1.pdf
- Pages created:
  - entities/attnres.md
- Pages updated:
  - concepts/hyper-connections.md — added "Alternative Approach: Attention Residuals" section + [[attnres]] See Also link
- index.md updated: total pages 43 → 44

## [2026-05-19] ingest | SpectralQuant — Gopinath, Apr 2026 (papers/spectralquant.pdf)
- Pages created:
  - entities/spectralquant.md — full entity page: spectral gap discovery, 5-stage algorithm, bias-variance proof for selective QJL, results table
- Pages updated:
  - entities/turboquant.md — added SpectralQuant as data-aware successor in See Also
  - comparisons/kv-cache-compression-comparison.md — added SpectralQuant column + full §; updated Recommended Use; title updated to include SpectralQuant
  - concepts/quantization.md — added SpectralQuant to KV cache quantization §; updated Calibration §; replaced random preconditioning section with spectral gap insight; updated See Also
- index.md updated: total pages 44 → 45
- queries/inference-improvements-summary.md — added SpectralQuant to §3c; added row to cross-cutting themes table; added [[spectralquant]] to See Also
- queries/memory-reduction-survey.md — extended §10b (PolarQuant/TurboQuant → +SpectralQuant); added row to summary table; added [[spectralquant]] to See Also

## [2026-05-19] update | inference-improvements-summary.md + memory-reduction-survey.md
- inference-improvements-summary.md:
  - Added §1c Attention Residuals (AttnRes) under Architecture Improvements
  - Added AttnRes row to cross-cutting themes table
  - Added [[attnres]] to See Also
- memory-reduction-survey.md:
  - Added §15 Residual Architecture Improvements (AttnRes); renumbered prior §15 MoE → §16
  - Added AttnRes row to Summary Table
  - Updated Cross-Cutting Themes paragraph to include AttnRes
  - Added [[attnres]] to See Also

## [2026-05-19] create | memory-inference-research-gaps.md
- Pages created:
  - queries/memory-inference-research-gaps.md — critical-thinking analysis of gaps in memory reduction & inference optimization literature; 6 gap categories (methodological, coverage, shared-premise, compositional, robustness, process), ranked high-value targets
- index.md updated: total pages 45 → 46; added entry under Queries

## [2026-05-19] lint | 3 issues found
- Orphan (1): queries/memory-inference-research-gaps.md — no inbound links yet (newly created)
- Oversized (2): queries/inference-improvements-summary.md (426 lines), queries/memory-reduction-survey.md (341 lines) — candidates for splitting
- All frontmatter, tags, wikilinks, and index entries clean

## [2026-05-19] create | lint fixes — split oversized pages, resolve orphan
- Pages created:
  - queries/inference-kv-speculative.md — §3 KV cache (H₂O, TriAttention, PolarQuant, TurboQuant, SpectralQuant) + §4 Speculative Decoding (SD algorithm, Saguaro, LayerSkip) split from inference-improvements-summary
  - queries/memory-inference-techniques.md — Part II inference memory techniques (§9–16) split from memory-reduction-survey; includes [[memory-inference-research-gaps]] in See Also (orphan fix)
- Pages updated:
  - queries/inference-improvements-summary.md — §3/§4 replaced with brief summaries + links; trimmed to ~190 lines; links to new pages and [[memory-inference-research-gaps]]
  - queries/memory-reduction-survey.md — retitled to training-only; Part II inference content removed; training summary table + cross-cutting themes retained; links to new pages and [[memory-inference-research-gaps]]
- index.md updated: total pages 46 → 48; added entries for inference-kv-speculative and memory-inference-techniques

## [2026-05-31] ingest | Batch ingest: 4 new papers — EAGLE family + DFlash
- Sources ingested:
  - 2401.15077v3.pdf → EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (Li et al., Peking U/MSR/Waterloo, Mar 2025)
  - 2406.16858v2.pdf → EAGLE-2: Faster Inference with Dynamic Draft Trees (Li et al., Jun 2024)
  - 2503.01840v3.pdf → EAGLE-3: Scaling Inference Acceleration via Training-Time Test (Li et al., Apr 2025)
  - 2602.06036v2.pdf → DFlash: Block Diffusion for Flash Speculative Decoding (Chen/Liang/Liu, UC San Diego, ICML 2026)
- Pages created:
  - entities/eagle.md — feature-level AR drafting; uncertainty resolution via shifted token input; 2.7×–3.5× lossless
  - entities/eagle-2.md — dynamic draft trees via calibrated confidence scores; 3.05×–4.26×; no extra training
  - entities/eagle-3.md — direct token prediction + multi-layer features + training-time test; up to 6.5×; data scaling law
  - entities/dflash.md — block diffusion adapter for parallel drafting; constant draft cost; 6×+ lossless, 2.5× over EAGLE-3
- Pages updated:
  - concepts/speculative-decoding.md — added EAGLE Family section (EAGLE/EAGLE-2/EAGLE-3/DFlash); updated See Also; updated sources
  - concepts/diffusion-language-models.md — added DFlash entry under Key Systems; updated See Also; updated sources
  - entities/block-diffusion.md — added "Use in DFlash" section; updated See Also
- index.md updated: total pages 48 → 51; added eagle, eagle-2, eagle-3, dflash entries

## [2026-05-31] update | queries/ — extended with EAGLE family + DFlash
- queries/inference-improvements-summary.md — §4 speculative decoding: added EAGLE/EAGLE-2/EAGLE-3/DFlash with key numbers; updated cross-cutting themes table (4 new rows); updated See Also
- queries/inference-kv-speculative.md — §4 speculative decoding: added full EAGLE Family section (EAGLE/EAGLE-2/EAGLE-3/DFlash with technical detail, result tables, root-cause analysis); updated See Also
- queries/memory-inference-techniques.md — §13 speculative decoding: added EAGLE family + DFlash memory model paragraph; updated summary table (2 new rows); updated See Also
- queries/memory-inference-research-gaps.md — updated page count (45→51); added §3e (DFlash acceptance rate degradation gap); added 2 new composition rows to §4 table; added gap #6 to ranked targets; updated See Also

## [2026-05-31] lint | 4 issues found, all fixed
- Broken wikilinks: 0
- Orphans: 0
- Index completeness: 51/51 pages in index.md ✓
- Frontmatter: all valid ✓
- Tags: all 21 taxonomy tags valid, no unknown tags ✓
- Log size: 240 lines, well within 500-entry rotation threshold ✓
- Oversized (1): queries/inference-kv-speculative.md at 260 lines — accepted (query pages exempt from 200-line split rule, per prior precedent)
- Near-orphans fixed (3):
  - concepts/speculative-decoding.md → added [[inference-kv-speculative]] to See Also
  - queries/memory-inference-techniques.md → added [[inference-kv-speculative]] to See Also
  - queries/memory-inference-research-gaps.md → added [[inference-kv-speculative]] to See Also
  - concepts/weight-norm-training.md → added [[derf]] and [[normalization-free-transformers]] to See Also

## [2026-06-07] update | I-DLM small-scale reproduction results
- Pages updated:
  - entities/i-dlm.md — added "Small-Scale Reproduction" section: WikiText-103 setup (frozen rbf_ffn + per-position LoRA, rank=8, stride=4), key implementation note on [x_0|x_t] concatenation order (causal attention requires clean-first), baseline config table, first experiment results (run 20260606_193642_930058_idlm_r8_s4: α=0.34, PPL=155.7, TPF/OH=1.11 at d=256/3 epochs); bumped updated date to 2026-06-07
  - index.md — updated last-updated date and i-dlm summary with reproduction metrics

## [2026-06-17] ingest | Batch ingest: 2 new papers — QKV Projection Sharing + Medusa
- Sources ingested:
  - 2606.04032v2.pdf → "Do Transformers Need Three Projections? Systematic Study of QKV Variants" (Kayyam, Madan Gopal, Lewis — BrainChip Inc., ICML 2026)
  - 2401.10774v3.pdf → "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (Cai, Li, Geng, Peng, Lee, Chen, Dao — Princeton/Together AI, ICML 2024)
- Raw source files created:
  - wiki/raw/papers/2606.04032v2.md
  - wiki/raw/papers/2401.10774v3.md
- Pages created:
  - entities/qkv-projection-sharing.md — Q-K=V (K=V shared) wins with 50% cache at +3.1% PPL; Q=K-V fails (symmetric attention, zero cache benefit); Q=K=V catastrophic (+25.4% PPL); compound Q-MQA achieves 96.9% cache; validated at 1.2B/10B tokens; directly grounds the new KVSharedAttention and KVSharedExclusiveSelfAttention classes in rbf_ffn
  - entities/medusa.md — K extra single-layer decoding heads predict tokens in parallel; tree attention verifies all candidates in one backbone forward pass; Medusa-1 (frozen backbone) 2.2×, Medusa-2 (joint training) 2.83×; typical acceptance vs rejection sampling; draft accuracy ~0.6 vs EAGLE's ~0.8
- Pages updated:
  - concepts/speculative-decoding.md — added Medusa section (before EAGLE family); added [[medusa]] to See Also; bumped sources and updated date
  - concepts/kv-cache.md — added Q-K=V projection sharing to Architectural Reduction section; added [[qkv-projection-sharing]] to See Also; bumped updated date
  - entities/xsa.md — added "KV-Shared XSA Variant (In Progress)" section documenting KVSharedExclusiveSelfAttention; added [[qkv-projection-sharing]] to See Also; bumped updated date
  - index.md — total pages 51 → 53; added medusa and qkv-projection-sharing entries; bumped last-updated date

## [2026-06-24] ingest | Batch ingest: 7 new sources — SageAttention family, DualPath, GQE, DLM empirics
- Sources ingested:
  - 2410.02367v9.pdf → SageAttention (Zhang et al., Tsinghua, ICLR 2025)
  - 2411.10958v7.pdf → SageAttention2 (Zhang et al., Tsinghua, ICML 2025)
  - 2505.11594v3.pdf → SageAttention3 (Zhang et al., Tsinghua, NeurIPS 2025)
  - 2602.21548v2.pdf → DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference (Wu et al., Peking U / Tsinghua / DeepSeek-AI, Feb 2026)
  - 2606.20945v2.pdf → Grouped Query Experts: MoE on GQA Self-Attention (Tripathi & Kumar, FrontiersMind, Jun 2026)
  - LLaDa and Mercury...md → Blog post (Sae-Hwan Park, Medium, Mar 2025) — confidence: low
  - The Optimal Architecture for Small Language Models.md → Blog post (Sharma, HF, Dec 2025) — confidence: medium
- Raw source files created:
  - wiki/raw/papers/2410.02367v9.md
  - wiki/raw/papers/2411.10958v7.md
  - wiki/raw/papers/2505.11594v3.md
  - wiki/raw/papers/2602.21548v2.md
  - wiki/raw/papers/2606.20945v2.md
  - wiki/raw/papers/llada-mercury-blog.md
  - wiki/raw/papers/optimal-architecture-slm-blog.md
- Pages created:
  - entities/sageattention.md — INT8 Q/K attention quantization; K-smoothing; 2.1× FA2; plug-and-play
  - entities/sageattention2.md — INT4 Q/K (per-thread) + FP8 P/V (2-level acc); 3× FA2, 481 TOPS
  - entities/sageattention3.md — FP4 NVFP4 microscaling; 5× FA2 on RTX5090; also SageBwd (8-bit fine-tuning)
  - entities/dualpath.md — dual-path KV-Cache loading; agentic I/O bottleneck; 1.87× offline, 1.96× online
  - entities/gqe.md — MoE on GQA query heads; KV dense; 1.7–1.8× prefill speedup; matches GQA quality
- Pages updated:
  - concepts/quantization.md — added Attention Computation Quantization section (SageAttention family table + analysis); updated sources, tags, date
  - concepts/kv-cache.md — added §Serving-Layer Loading (DualPath) + §Sparse Query Computation (GQE); updated See Also, sources, date
  - concepts/mixture-of-experts.md — added MoE Beyond FFN Layers section (GQE); updated See Also, sources, date
  - concepts/diffusion-language-models.md — added Production Systems section (Mercury/LLaDA, confidence: low) + Small-Scale Architecture Empirics section (Sharma study); extended Open Questions; updated sources, date
  - entities/flash-attention.md — added Quantized Attention Kernels section (SageAttention table); extended See Also
  - queries/inference-improvements-summary.md — added Attention Computation Quantization §2 subsection; added DualPath to §5; added SA*/DualPath/GQE rows to cross-cutting table; extended See Also; updated date
  - queries/inference-kv-speculative.md — added §3d Attention Compute Quantization (full SA1/SA2/SA3 detail + comparison table); extended See Also; updated date, tags
- index.md updated: total pages 53 → 58; added entries for sageattention, sageattention2, sageattention3, dualpath, gqe; bumped date

## [2026-06-24] backfill | Raw source files for all 35 previously-uncovered papers
- All files written to wiki/raw/papers/ — one compact summary per paper with frontmatter
- Papers covered (in ingestion order):
  - 1602.07868v3.md — Weight Normalization (Salimans & Kingma, OpenAI, NIPS 2016)
  - 2010.04245v1.md — QKNorm: Query-Key Normalization for Transformers (Henry et al., 2020)
  - 2201.02177v1.md — Grokking: Generalization Beyond Overfitting (Power et al., OpenAI, 2022)
  - 2205.14135v2.md — FlashAttention v1 (Dao et al., Stanford, NeurIPS 2022)
  - 2211.17192v2.md — Fast Inference via Speculative Decoding (Leviathan et al., Google, ICML 2023)
  - 2306.14048v3.md — H₂O: Heavy-Hitter Oracle for KV Eviction (Zhang et al., NeurIPS 2023)
  - 2308.16369v2.md — SARATHI: Chunked Prefill + Decode-Maximal Batching (Agrawal et al., MSR India, 2023)
  - 2309.06180v1.md — PagedAttention / vLLM (Kwon et al., UC Berkeley, SOSP 2023)
  - 2312.07104v1.md — SGLang: Structured LM Programs + RadixAttention (Zheng et al., Stanford/UCB, 2024)
  - 2401.15077v3.md — EAGLE: Feature-Level Speculative Sampling (Li et al., Peking U/MSR, 2025)
  - 2404.16710v2.md — LayerSkip: Early Exit + Self-Speculative Decoding (Elhoushi et al., Meta FAIR, 2024)
  - 2406.16858v2.md — EAGLE-2: Dynamic Draft Trees via Calibrated Confidence (Li et al., 2024)
  - 2502.02617v1.md — PolarQuant: KV Cache via Polar Transformation (Han et al., KAIST/Google/Yale, 2025)
  - 2503.01840v3.md — EAGLE-3: Direct Token Prediction + Training-Time Test (Li et al., 2025)
  - 2503.09573v3.md — Block Diffusion / BD3-LM (Arriola et al., Cornell/Stanford, ICLR 2025)
  - 2504.19874v1.md — TurboQuant: Online VQ with Near-Optimal Distortion (Zandieh et al., Google/NYU, 2025)
  - 2505.17420v1.md — DASH: Dynamic Layer Skipping via MDP (Yang et al., SJTU, 2025)
  - 2511.08544v3.md — LeJEPA: Provable SSL Without the Heuristics (Balestriero & LeCun, Brown/NYU/Meta, 2025)
  - 2512.02556v1.md — Engram: Conditional Memory as N-gram Lookup (Cheng et al., Peking U/DeepSeek, 2026)
  - 2512.10938v2.md — DeepSeek-V3.2: DSA + Scalable RL + Agentic Synthesis (DeepSeek-AI, 2025)
  - 2512.24880v2.md — mHC: Manifold-Constrained Hyper-Connections (Xie et al., DeepSeek-AI, 2026)
  - 2601.05732v1.md — mHC-lite: Convex Combination of Permutations (Yang & Gao, Michigan/NTU, 2026)
  - 2601.07372v1.md — Derf: erf(αx+s) Normalization-Free Transformer (Chen et al., Princeton/NYU/CMU, 2026)
  - 2601.21579v1.md — KromHC: Kronecker-Product Residual Matrices (Zhou et al., Imperial College, 2026)
  - 2602.06036v2.md — DFlash: Block Diffusion for Flash Speculative Decoding (Chen et al., UCSD, ICML 2026)
  - 2603.03251v3.md — Speculative Speculative Decoding / Saguaro (Kumar et al., Stanford/Princeton, 2026)
  - 2603.09078v1.md — Exclusive Self Attention (XSA) (Zhai, Apple, 2026)
  - 2603.15031v1.md — Attention Residuals (AttnRes) (Kimi Team, Moonshot AI, 2026)
  - 2603.19312v2.md — LeWorldModel: Stable E2E JEPA from Pixels (Maes et al., Mila/NYU/Brown, 2026)
  - 2604.04921v1.md — TriAttention: Trigonometric KV Compression (Mao et al., MIT/NVIDIA/ZJU, 2026)
  - 2604.11035v1.md — I-DLM: Introspective Diffusion Language Model (Yu et al., Together AI/UIUC, 2026)
  - cliptogrok.md — Clip to Grok: Weight Norm Clipping for Grokking (Volchkov & Rivlin, 2026)
  - DeepSeek_V4.md — DeepSeek-V4: CSA/HCA + mHC + Muon + 1M context (DeepSeek-AI, 2026)
  - spectralquant.md — SpectralQuant: 3% Spectral Gap Beats TurboQuant (Gopinath, MIT/Sentra, 2026)
  - visual-guide-quantization.md — A Visual Guide to Quantization (Grootendorst, 2024)
