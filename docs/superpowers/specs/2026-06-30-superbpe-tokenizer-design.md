# SuperBPE Tokenizer Experiment for rbf_ffn

**Date:** 2026-06-30
**Status:** Design approved

---

## Goal

Add a SuperBPE tokenizer (two-stage BPE curriculum, vocab_size=48576) to the `rbf_ffn` project and pretrain a model with it on WikiText-103. SuperBPE lifts the whitespace boundary constraint in Stage 2 to learn multi-word "superword" tokens, reducing sequence length and shifting loss mass toward harder predictions.

---

## Scope

Four files are touched:

| File | Change |
|---|---|
| `rbf_ffn/superbpe_data.py` | New — two-stage tokenizer training + WikiText-103 data pipeline |
| `rbf_ffn/config.py` | Add `tokenizer: str = "r50k"` field + validation |
| `rbf_ffn/train.py` | Dispatch `get_dataloaders` by `cfg.tokenizer`; `_superbpe` tag in experiment naming |
| `rbf_ffn/configs/baseline_xsa_superbpe.yaml` | New experiment config |

No new packages are needed: `tokenizers` is already a dependency of `sigreg`.

---

## SuperBPE Tokenizer Training (two-stage curriculum)

**Parameters:**
- Total vocab size: `T = 48576`
- Stage 1 transition point: `t = 43718` (~90% of T, matching the paper's recommended ratio)
- Stage 2 superword slots: `T - t = 4858`

**Stage 1** — standard BPE with whitespace pretokenization:
- `Tokenizer(BPE())` with `pre_tokenizer = Sequence([Whitespace(), ByteLevel()])`
- `BpeTrainer(vocab_size=43718, special_tokens=[], min_frequency=2)`
- Trained on WikiText-103 train split texts
- Saved to `data_cache/superbpe48576_tokenizer/stage1/` (`vocab.json`, `merges.txt`)

**Stage 2** — superword BPE without whitespace boundary:
- Warm-start: load Stage 1 `vocab.json`/`merges.txt` into `BPE(vocab, merges)`
- Swap pre-tokenizer to `ByteLevel()` only (removes whitespace chunking)
- `BpeTrainer(vocab_size=48576, special_tokens=[], min_frequency=2)`
- Continue training on same WikiText-103 train texts
- Saved to `data_cache/superbpe48576_tokenizer/stage2/` (`vocab.json`, `merges.txt`)

**Fallback:** If the `tokenizers` library does not support warm-starting from existing merges (i.e., `BPE(vocab, merges)` ignores prior merges), Stage 2 trains from scratch with `ByteLevel()` pre-tokenizer and `initial_alphabet` seeded from Stage 1's vocab. This is a close approximation that still removes the whitespace constraint.

**Cache filenames:**
- Tokenizer: `data_cache/superbpe48576_tokenizer/stage{1,2}/`
- Token chunks: `data_cache/{split}_superbpe48576_{seq_len}.pt`

---

## Data Pipeline (`superbpe_data.py`)

Mirrors `sigreg/data.py` exactly:

```
_build_superbpe_tokenizer(cache_dir) → Tokenizer
_load_wikitext_split_texts(split) → list[str]
_load_split(split, seq_len, tokenizer) → Tensor (N, seq_len)
get_dataloaders(cfg) → (train_loader, val_loader, test_loader)
```

Reuses `rbf_ffn.data.chunk_tokens` and `rbf_ffn.data.TokenDataset`.
Shared cache dir: `rbf_ffn/data_cache/` (avoids re-downloading WikiText-103).
Encoding is batched (10k texts at a time) to avoid memory issues.

---

## Config Changes (`config.py`)

Add one field to `ModelConfig`:

```python
tokenizer: str = "r50k"   # "r50k" | "superbpe48576"
```

Validation in `__post_init__`:
- `tokenizer` must be in `{"r50k", "superbpe48576"}`
- If `tokenizer == "superbpe48576"`, assert `vocab_size == 48576`

---

## Training Loop Changes (`train.py`)

Dispatch at the top of `train()`:

```python
if cfg.tokenizer == "superbpe48576":
    from rbf_ffn.superbpe_data import get_dataloaders
else:
    from rbf_ffn.data import get_dataloaders
```

Experiment naming: add `_superbpe` tag in `get_experiment_dir` when `cfg.tokenizer == "superbpe48576"`.

---

## Experiment Config (`baseline_xsa_superbpe.yaml`)

```yaml
# XSA + SwiGLU baseline with SuperBPE tokenizer (vocab=48576, t=43718)
attn_type: xsa
ffn_type: swiglu
vocab_size: 48576
tokenizer: superbpe48576
d_model: 256
n_heads: 8
n_layers: 6
seq_len: 512
batch_size: 32
n_epochs: 3
qk_norm: true
linear_weight_norm: true
seed: 42
muon_lr: 0.02
adamw_lr: 3.0e-4
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

Matches the current best-performing baseline config family (XSA + qknorm + wnorm), swapping only the tokenizer and vocab size.

---

## Error Handling

- If WikiText-103 fails to download: propagates naturally (HuggingFace `datasets` raises).
- If Stage 2 warm-start fails (library limitation): fall back to independent Stage 2 training with ByteLevel-only pre-tokenizer; log a warning.
- Cached `.pt` files are not regenerated if they already exist (safe to re-run).

---

## Testing

Add `rbf_ffn/tests/test_superbpe_data.py`:
- `test_tokenizer_vocab_size`: checks `tokenizer.get_vocab_size() == 48576`
- `test_no_pad_token`: verifies no special tokens pollute the vocabulary
- `test_chunk_shape`: checks `_load_split` output shape `(N, seq_len)` with a tiny mock corpus
- `test_config_validation_tokenizer`: checks `ModelConfig(tokenizer="superbpe48576", vocab_size=50257)` raises `ValueError`
- `test_config_valid_superbpe`: checks `ModelConfig(tokenizer="superbpe48576", vocab_size=48576)` constructs without error
