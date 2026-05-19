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
