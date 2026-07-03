# kv_quant

KV cache compression for HuggingFace causal LMs. Supports two quantization methods (TurboQuant, SpectralQuant) and one token eviction method (TriAttention), composable via a single `wrap()` call.

## Quick start

```python
from kv_quant import wrap, QuantConfig

model = AutoModelForCausalLM.from_pretrained(...).eval()
cfg = QuantConfig(method="turboquant", bits=4)
model = wrap(model, cfg)

# model.generate() now uses a compressed KV cache automatically
output = model.generate(**inputs, max_new_tokens=200)
```

## QuantConfig

```python
@dataclass
class QuantConfig:
    method: Optional[Literal["turboquant", "spectralquant"]] = "turboquant"
    bits: int = 4                  # key bits
    value_bits: int = 2            # value bits (TurboQuant only)
    value_group_size: int = 32     # group size for value quantization (TurboQuant only)
    buffer_size: int = 128         # recent tokens kept in fp16 (TurboQuant only)
    qjl_dim: int = 32             # QJL projection dim (SpectralQuant only)
    calibration_path: Optional[str] = None  # SpectralQuant: base path; TriAttention: stats .pt
    signal_bit_boost: float = 2.0  # SpectralQuant only
    budget: int = 2048             # TriAttention: max KV tokens to retain
    divide_length: int = 128       # TriAttention: eviction trigger interval (decode steps)
    eviction: Optional[Literal["triattention"]] = None
```

`calibration_path` serves two roles depending on context:
- **SpectralQuant**: base path (no extension) for files produced by `kv_quant.calibrate`
- **TriAttention**: path to a stats `.pt` file from `triattention/triattention/vllm/stats/`

## Methods

### TurboQuant (`method="turboquant"`)

No calibration required. Keys are compressed with Lloyd-Max codebook + QJL residual correction; values with asymmetric group quantization.

```python
cfg = QuantConfig(method="turboquant", bits=4, value_bits=2)
model = wrap(model, cfg)
```

### SpectralQuant (`method="spectralquant"`)

Data-aware per-head spectral rotation + non-uniform bit allocation. Requires calibration.

**Step 1 — calibrate:**
```bash
python -m kv_quant.calibrate \
    --model Qwen/Qwen3-1.7B \
    --output cal/qwen3_1b \
    --n-seqs 100 \
    --bits 4
# Writes: cal/qwen3_1b.pt, cal/qwen3_1b_meta.json, cal/qwen3_1b_quantizers.pt
```

**Step 2 — use:**
```python
cfg = QuantConfig(method="spectralquant", bits=4, calibration_path="cal/qwen3_1b")
model = wrap(model, cfg)
```

### TriAttention eviction (`eviction="triattention"`)

Orthogonal to quantization — can be combined with either method or used standalone (`method=None`). Evicts low-importance tokens every `divide_length` decode steps, keeping at most `budget` tokens.

```python
# Standalone eviction, no quantization
cfg = QuantConfig(method=None, eviction="triattention",
                  calibration_path="triattention/stats/qwen3_1b.pt")

# Combined with TurboQuant
cfg = QuantConfig(method="turboquant", bits=4, eviction="triattention",
                  calibration_path="triattention/stats/qwen3_1b.pt")

model = wrap(model, cfg)
```

## Benchmarking

```bash
python -m kv_quant.bench.run_bench \
    --model google/gemma-4-E2B-it \
    --method turboquant spectralquant \
    --bits 2 3 4 \
    --tasks mmlu arc_easy hellaswag gsm8k \
    --calibration cal/gemma4_2b \
    --output results/gemma4_2b.csv \
    --no-ppl
```

The CSV columns are: `method`, `bits`, `ppl`, `kv_mb`, and one column per task (accuracy ×100).
A `baseline` row (fp16, no quantization) is always included.

## Module layout

```
kv_quant/
├── config.py              # QuantConfig dataclass
├── __init__.py            # wrap() entry point
├── turboquant.py          # TurboQuantCache
├── spectralquant.py       # SpectralQuantCache
├── triattention_patch.py  # Combined quantization + eviction forward patch
├── calibrate.py           # SpectralQuant calibration CLI
├── ops/
│   ├── rotation.py        # Random rotation utilities
│   ├── scalar_quant.py    # Asymmetric group quantization
│   ├── qjl.py             # QJL projection
│   ├── turboquant_core.py # TurboQuantMSE (Lloyd-Max + rotation)
│   ├── codebook.py        # Codebook loader
│   └── codebooks/         # Precomputed Beta-distribution codebooks (d=256/512, bits=1–4)
└── bench/
    ├── run_bench.py        # Benchmark CLI
    ├── perplexity.py       # WikiText-2 perplexity
    └── memory.py           # Peak KV memory measurement
```

## Tests

```bash
pytest tests/test_cache.py tests/test_calibrate.py -v
```
