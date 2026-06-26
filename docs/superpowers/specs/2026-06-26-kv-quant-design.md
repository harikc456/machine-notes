# KV Cache Quantization: TurboQuant + SpectralQuant for HuggingFace Models

**Date:** 2026-06-26
**Status:** Approved

## Overview

Implement TurboQuant and SpectralQuant as drop-in KV cache quantizers for HuggingFace causal LMs,
packaged as a `kv_quant` module. Deliverables: a `wrap(model, config)` API and a benchmark harness
measuring perplexity, memory, and downstream task scores.

Target architectures: Qwen (Qwen2.5, Qwen3) and Gemma (Gemma 2, Gemma 3).
Compression target: configurable bits-per-channel (sweep 2–4 bits).

---

## Module Structure

```
kv_quant/
├── __init__.py          # wrap(model, config) public API
├── config.py            # QuantConfig dataclass
├── ops/
│   ├── __init__.py
│   ├── rotation.py      # random orthogonal rotation, precomputed per head
│   ├── scalar_quant.py  # n-bit scalar quantizer, optimal levels for Beta distribution
│   └── qjl.py          # 1-bit QJL: sign(S @ x), inner-product residual correction
├── turboquant.py        # TurboQuantCache(DynamicCache)
├── spectralquant.py     # SpectralQuantCache(DynamicCache)
├── calibrate.py         # SpectralQuant calibration script (standalone)
└── bench/
    ├── __init__.py
    ├── perplexity.py    # WikiText-2 PPL at each bit budget
    ├── memory.py        # peak KV cache memory tracker
    └── run_bench.py     # CLI orchestrator
```

---

## Integration Pattern

Both cache classes subclass `transformers.DynamicCache` and override `update()`.

On each `update(key_states, value_states, layer_idx)` call:
1. Quantize incoming K/V → append to internal compressed buffers `_qk_cache`, `_qv_cache`
2. Dequantize full accumulated sequence
3. Return dequantized K, V — used directly by HF attention, which relies on the return value not on `key_cache`/`value_cache` attributes

`key_cache` and `value_cache` stay empty; `get_seq_length()` is overridden to read from `_qk_cache`.

`wrap(model, config)` monkey-patches `model.generate()` to inject `past_key_values=<cache_instance>` before each call. SpectralQuant additionally requires calibration to have been run first.

---

## Algorithm: TurboQuant

**No calibration required. Online, per-token.**

### Initialization (per head, head_dim = d)
- Sample random orthogonal `R ∈ ℝ^{d×d}` via QR decomp of random Gaussian — one per head, fixed buffer
- Sample sign matrix `S ∈ {±1}^{m×d}` for QJL, where `m = qjl_dim` (default 32)

### On update — quantize incoming `h ∈ ℝ^{batch × heads × seq × d}`
1. **Rotate**: `h̃ = h @ R.T` — coordinates become approximately Beta-distributed
2. **Scalar quantize**: clip to `[-scale, scale]` (scale = max abs value), apply n-bit uniform quantization per coordinate, store as int8 (4-bit packed for n≤4)
3. **QJL residual**: `residual = h - dequant(h̃) @ R`, store `sign(S @ residual.T)` packed as 1 bit per entry

### On dequantize
1. Dequantize stored ints → `h̃_dq`
2. Rotate back: `ĥ = h̃_dq @ R`
3. Return `ĥ` — inner-product correction from QJL is approximate in this prototype (no custom kernel; documented limitation)

### Memory layout per layer
- `_qk_cache`: `(seq, heads, d)` int8 + `(seq, heads)` float16 scale-per-token
- `_qk_qjl`: `(seq, heads, m)` packed bits (1 bit per entry)
- Effective bits: `n_bits + 16/d + m/d` bits per channel

---

## Algorithm: SpectralQuant

**15-second one-time calibration per model. Per-head calibrated eigenvectors + codebooks.**

### Calibration (`calibrate.py`)
1. Forward 100 WikiText-2 validation sequences through the frozen model; register forward hooks on each attention layer to capture key vectors before they enter the cache
2. Per head: compute empirical covariance `Σ̂ = (1/N) Σ h_t h_t^T`
3. `U, λ = torch.linalg.eigh(Σ̂)` — eigenvectors in ascending order; reverse for descending
4. Effective dimensionality: `d_eff = PR(Σ) = (Σλ_i)² / Σλ_i²`; set `d_s = ceil(d_eff)` (expect ~4 for d=128)
5. Project calibration data: `h̃ = h @ U`
6. Train Lloyd-Max codebooks via k-means:
   - `codebook_signal`: k=2^n_bits_signal centers on signal dims (top `d_s`)
   - `codebook_noise`: k=2^n_bits_noise centers on noise dims (`d - d_s`); fewer bits
7. Sample `S_signal ∈ {±1}^{m×d_s}` for selective QJL
8. Save `{layer_idx, head_idx, U, d_s, codebook_signal, codebook_noise, S_signal}` to `spectralquant_<model_id>.pt`

Bit split: signal dims get `min(8, ceil(bits * signal_bit_boost))` bits, noise dims get `max(1, bits - 1)` bits, where `signal_bit_boost` defaults to 2.0 and is configurable in `QuantConfig`. Total average bits across all dims is constrained to match the configured `bits` parameter.

### On update
1. **Project**: `h̃ = h @ U`
2. **Non-uniform quantize**: signal slice → nearest centroid in `codebook_signal`; noise slice → `codebook_noise`
3. **Selective QJL**: `sign(S_signal @ h̃[:, :, :d_s].T)` — signal dims only

### On dequantize
1. Nearest-codebook lookup → `h̃_dq`
2. Rotate back: `ĥ = h̃_dq @ U.T`

---

## `wrap()` API

```python
from kv_quant import wrap, QuantConfig

config = QuantConfig(
    method="turboquant",   # or "spectralquant"
    bits=4,
    qjl_dim=32,            # QJL projection dimension
    calibration_path=None, # required for spectralquant
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", ...)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

model = wrap(model, config)
output = model.generate(input_ids, max_new_tokens=200)  # uses quantized KV cache
```

SpectralQuant calibration:
```bash
python -m kv_quant.calibrate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output spectralquant_qwen25_7b.pt \
  --n-seqs 100
```

---

## Benchmark Harness

### CLI
```bash
python -m kv_quant.bench.run_bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --method turboquant spectralquant \
  --bits 2 3 4 \
  --tasks mmlu arc_easy hellaswag gsm8k \
  --calibration spectralquant_qwen25_7b.pt \
  --output results/qwen25_7b.csv
```

### Measurements per (method, bits) point

**Perplexity (`perplexity.py`)**: WikiText-2 test set, sliding window (stride=512, context=2048),
mean cross-entropy → PPL.

**Memory (`memory.py`)**: `torch.cuda.memory_allocated()` delta between baseline (no KV cache) and
peak during decode. Reports bytes and compression ratio vs fp16 baseline.

**LM-eval tasks**: calls `lm_eval.simple_evaluate(model=wrapped_model, tasks=[...])` programmatically.
MMLU, ARC-Easy, HellaSwag, GSM8K all have built-in lm-eval task definitions.

### Output format
```
method        bits  PPL    KV_MB   MMLU   ARC    HellaSwag  GSM8K
baseline      fp16  8.21   4096    68.2   79.4   81.3       62.1
turboquant    4     8.35   1024    67.9   79.1   81.0       61.8
turboquant    3     8.71    768    67.1   78.5   80.4       60.2
turboquant    2     9.80    512    65.4   77.2   79.1       57.3
spectralquant 4     8.28    512    68.1   79.3   81.2       61.9
spectralquant 3     8.41    384    67.8   79.0   81.0       61.5
spectralquant 2     8.95    256    67.0   78.2   80.3       60.1
```

CSV written to `--output`; table printed to stdout.

---

## Testing

### `tests/test_ops.py` (unit, no model download)
- `test_rotation_orthogonal`: `R @ R.T ≈ I` within 1e-5
- `test_scalar_quant_roundtrip`: round-trip error (SNR) within expected bound for each bit depth 2–8
- `test_qjl_inner_product`: inner product estimation on 1000 random vector pairs; mean bias < 0.01, variance within theoretical bound

### `tests/test_cache.py` (unit, synthetic model, no download)
- Construct `TurboQuantCache` and `SpectralQuantCache` with tiny synthetic params (d=16, 2 heads, 2 layers)
- Call `update()` 20 times with random K/V; assert returned shapes match input, sequence grows correctly
- Assert `_qk_cache` memory < fp16 equivalent at bits < 16
- Assert dequantized output has finite values (no NaN/Inf)

### `tests/test_integration.py` (slow, GPU + model download, gated behind `--run-slow`)
- Load `Qwen/Qwen2.5-0.5B` (smallest Qwen)
- Wrap with TurboQuant at 4 bits; generate 50 tokens from a fixed prompt
- Assert output is non-empty and has no repetition collapse
- Assert PPL on 10 WikiText-2 chunks is within 1.5× of unwrapped baseline

---

## Constraints and Known Limitations

- **Dequantize-on-return only**: the compressed buffers save memory, but each `update()` call dequantizes the full sequence to return to HF attention. This is correct for a research prototype; real throughput gains would require a custom attention kernel that operates on compressed K/V.
- **QJL inner-product correction is approximate**: without a fused kernel, the QJL bits aren't used to correct attention scores at runtime — they're available in the buffer but inner-product bias correction isn't applied in the attention path. Documented as a known gap.
- **Calibration hooks are architecture-specific**: `calibrate.py` registers hooks on `self_attn` submodules to capture pre-cache key vectors. Verified for Qwen2/Qwen3 and Gemma2/Gemma3 attention class names; other architectures need additions to the hook registration logic.
- **GQA**: both caches handle GQA naturally since `update()` receives the already-expanded (or grouped) K/V tensors from HF attention — no special-casing needed.

---

## Out of Scope

- TriAttention (pre-RoPE eviction) — separate future module
- Custom CUDA/Triton kernels for fused compressed attention
- Support for architectures beyond Qwen and Gemma
- Quantization of value vectors with a different bit budget than keys (same budget applied to both)
