	# Wiki Schema

## Domain
LLM efficiency and architecture research — covering inference optimization, KV cache management, quantization (weight and KV), sparse/conditional computation, architectural innovations (residual connections, attention variants), and frontier model releases.

## Conventions
- File names: lowercase, hyphens, no spaces (e.g., `kv-cache.md`)
- Every wiki page starts with YAML frontmatter (see below)
- Use `[[wikilinks]]` to link between pages (minimum 2 outbound links per page)
- When updating a page, always bump the `updated` date
- Every new page must be added to `index.md` under the correct section
- Every action must be appended to `log.md`
- **Provenance markers:** On pages synthesizing 3+ sources, append `^[raw/papers/source-file.pdf]` at the end of paragraphs whose claims come from a specific source.

## Frontmatter

```yaml
---
title: Page Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: entity | concept | comparison | query | summary
tags: [from taxonomy below]
sources: [raw/papers/source-name.pdf]
confidence: high | medium | low
contested: true   # optional, set when page has unresolved contradictions
contradictions: [other-page-slug]   # optional
---
```

### raw/ Frontmatter

```yaml
---
source_url: https://arxiv.org/abs/XXXX.XXXXX
ingested: YYYY-MM-DD
sha256: <hex digest of raw content below frontmatter>
---
```

## Tag Taxonomy

### Models & Systems
- `model` — a trained LLM or model family
- `architecture` — architectural component or design pattern
- `benchmark` — evaluation/benchmark result

### Techniques
- `inference` — inference optimization (speed, memory)
- `training` — training techniques and optimizations
- `quantization` — weight or activation quantization
- `kv-cache` — KV cache management, eviction, compression
- `sparsity` — sparse computation, MoE, conditional computation
- `attention` — attention mechanism variants
- `residual` — residual connection variants and improvements
- `speculative` — speculative decoding and related
- `optimization` — optimizer choices (Muon, Adam, etc.)
- `normalization` — weight normalization, QK normalization, layer norm variants
- `normalization-free` — architectures that replace normalization layers with point-wise functions
- `grokking` — generalization dynamics, delayed generalization, weight norm as regularization
- `ssl` — self-supervised learning, contrastive methods, JEPA
- `world-model` — learning environment dynamics for planning and control

### Meta
- `comparison` — side-by-side analysis
- `deepseek` — DeepSeek-AI publications
- `open-source` — open-weights model
- `survey` — overview/survey content

## Page Thresholds
- **Create a page** when an entity/concept appears in 2+ sources OR is central to one source
- **Add to existing page** when a source mentions something already covered
- **DON'T create a page** for passing mentions, minor details, or things outside the domain
- **Split a page** when it exceeds ~200 lines — break into sub-topics with cross-links
- **Archive a page** when content is fully superseded — move to `_archive/`, remove from index

## Entity Pages
One page per notable model, system, or algorithm. Include:
- Overview / what it is
- Key technical contributions
- Benchmark results where available
- Relationships to other entities ([[wikilinks]])
- Source references

## Concept Pages
One page per concept or technique. Include:
- Definition / explanation
- Current state of knowledge
- Key papers / implementations
- Open questions or debates
- Related concepts ([[wikilinks]])

## Comparison Pages
Side-by-side analyses. Include:
- What is being compared and why
- Dimensions of comparison (table format preferred)
- Verdict or synthesis
- Sources

## Update Policy
When new information conflicts with existing content:
1. Check the dates — newer sources generally supersede older ones
2. If genuinely contradictory, note both positions with dates and sources
3. Mark the contradiction in frontmatter: `contradictions: [page-name]`
4. Flag for user review in the lint report
