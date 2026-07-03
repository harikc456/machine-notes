# TriAttention KV Cache Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate TriAttention (token-eviction KV cache compression from `triattention/`) into the `kv_quant/` experiment framework as a third `method` option alongside `turboquant` and `spectralquant`.

**Architecture:** Use `apply_triattention_patch(model, ...)` from the local `triattention/` submodule — it patches `model.forward` to handle position tracking and token eviction automatically. `wrap()` then patches `model.generate` to inject a plain `DynamicCache` with a `compressed_bytes()` method. The patch and the generate wrapper compose independently.

**Tech Stack:** PyTorch, HuggingFace Transformers `DynamicCache`, local `triattention/triattention/` package (added to sys.path, not pip-installed).

## Global Constraints

- No new pip dependencies — `triattention/` is on sys.path via `os.path` manipulation identical to how `spectralquant/src` is added
- `triattention/` submodule root (containing the `triattention/` Python package) is at `<repo_root>/triattention/`
- `apply_triattention_patch` is imported from `triattention.methods.triattention`
- `calibration_path` in `QuantConfig` is reused as the stats `.pt` file path for TriAttention (no new field)
- `model_path` for `TriAttentionConfig` is read from `model.config._name_or_path` (set by `from_pretrained`)
- `compressed_bytes()` for TriAttention measures actual bfloat16 tensor sizes — full precision, not theoretical bits
- All existing tests must remain green

---

### Task 1: TriAttention integration + tests

**Files:**
- Modify: `kv_quant/config.py`
- Modify: `kv_quant/__init__.py`
- Modify: `tests/test_cache.py`

**Interfaces:**
- Produces: `_make_triattention_cache()` (public, used in tests and by `_make_cache`)
- Produces: `QuantConfig(method="triattention", budget=int, divide_length=int, calibration_path=str)`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cache.py`, after the last existing test (`test_wrap_spectralquant_raises_without_calibration`):

```python
# ---------------------------------------------------------------------------
# TriAttention tests
# ---------------------------------------------------------------------------

def test_wrap_triattention_requires_calibration_path():
    model = _mock_model()
    cfg = QuantConfig(method="triattention", budget=256, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)


def test_triattention_cache_compressed_bytes_single_layer():
    """compressed_bytes() sums bfloat16 bytes across KV tensors."""
    from kv_quant import _make_triattention_cache
    cache = _make_triattention_cache()
    B, H, S, D = 1, 2, 10, 64
    k = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    v = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    cache.key_cache.append(k)
    cache.value_cache.append(v)
    # bfloat16 = 2 bytes per element
    expected = (k.nelement() + v.nelement()) * 2
    assert cache.compressed_bytes() == expected


def test_triattention_cache_compressed_bytes_multiple_layers():
    """compressed_bytes() accumulates across multiple layers."""
    from kv_quant import _make_triattention_cache
    cache = _make_triattention_cache()
    B, H, S, D = 1, 2, 5, 64
    k = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    v = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    cache.key_cache.extend([k, k])   # 2 layers
    cache.value_cache.extend([v, v])
    expected = 2 * (k.nelement() + v.nelement()) * 2
    assert cache.compressed_bytes() == expected
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cache.py::test_wrap_triattention_requires_calibration_path \
       tests/test_cache.py::test_triattention_cache_compressed_bytes_single_layer \
       tests/test_cache.py::test_triattention_cache_compressed_bytes_multiple_layers -v
```

Expected: FAIL — `QuantConfig` does not accept `method="triattention"` and `_make_triattention_cache` does not exist.

- [ ] **Step 3: Update `kv_quant/config.py`**

Replace the file with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QuantConfig:
    method: Literal["turboquant", "spectralquant", "triattention"] = "turboquant"
    bits: int = 4               # key bits (TurboQuant/SpectralQuant only)
    value_bits: int = 2         # value bits for group quantization (TurboQuant only)
    value_group_size: int = 32  # group size for value quantization
    buffer_size: int = 128      # recent tokens kept in full precision (TurboQuant only)
    qjl_dim: int = 32           # QJL projection dim (SpectralQuant only)
    calibration_path: Optional[str] = None  # SpectralQuant: base path (no ext); TriAttention: stats .pt path
    signal_bit_boost: float = 2.0           # SpectralQuant only
    budget: int = 2048          # TriAttention: max KV tokens to retain
    divide_length: int = 128    # TriAttention: compress every N decode steps
```

- [ ] **Step 4: Update `kv_quant/__init__.py`**

Replace the file with:

```python
from __future__ import annotations
import os
import sys
import torch

from kv_quant.config import QuantConfig


def _get_kv_shape(model) -> tuple[int, int]:
    """Extract (n_kv_heads, head_dim) from a HF model config."""
    cfg = model.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
    )
    return n_kv_heads, head_dim


def _ensure_spectralquant_on_path() -> None:
    src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
    )
    if src not in sys.path:
        sys.path.insert(0, src)


def _ensure_triattention_on_path() -> None:
    src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "triattention")
    )
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_spectralquant_cal(base_path: str) -> tuple:
    """Load (EigenspectralCalibrator, quant_state_dict) from base_path."""
    _ensure_spectralquant_on_path()
    from spectralquant.calibration import EigenspectralCalibrator
    calibrator = EigenspectralCalibrator()
    calibrator.load(base_path)
    quant_state = torch.load(
        base_path + "_quantizers.pt", map_location="cpu", weights_only=True
    )
    return (calibrator, quant_state)


def _make_triattention_cache():
    """Return a DynamicCache with compressed_bytes() measuring actual KV storage."""
    from transformers import DynamicCache
    cache = DynamicCache()
    cache.compressed_bytes = lambda: sum(
        kc.nelement() * kc.element_size() + vc.nelement() * vc.element_size()
        for kc, vc in zip(cache.key_cache, cache.value_cache)
        if kc is not None
    )
    return cache


def _apply_triattention(model, config: QuantConfig) -> None:
    """Apply TriAttention compression patch to model.forward."""
    if not config.calibration_path:
        raise ValueError(
            "triattention requires config.calibration_path (path to a stats .pt file "
            "from triattention/triattention/vllm/stats/ or scripts/calibrate.py)"
        )
    model_path = getattr(model.config, "_name_or_path", None)
    if not model_path:
        raise ValueError(
            "TriAttention requires model.config._name_or_path to be set. "
            "Load the model via AutoModelForCausalLM.from_pretrained so that "
            "_name_or_path is populated."
        )
    _ensure_triattention_on_path()
    from triattention.methods.triattention import apply_triattention_patch
    from pathlib import Path

    apply_triattention_patch(
        model,
        stats_path=Path(config.calibration_path),
        model_path=Path(model_path),
        kv_budget=config.budget,
        divide_length=config.divide_length,
    )


def _make_cache(config: QuantConfig, n_kv_heads: int, head_dim: int, cal_data, device):
    if config.method == "turboquant":
        from kv_quant.turboquant import TurboQuantCache
        return TurboQuantCache(config, n_kv_heads, head_dim, device=device)
    if config.method == "spectralquant":
        from kv_quant.spectralquant import SpectralQuantCache
        return SpectralQuantCache(config, cal_data)
    if config.method == "triattention":
        return _make_triattention_cache()
    raise ValueError(f"Unknown method: {config.method!r}")


def wrap(model, config: QuantConfig):
    """Patch model.generate() to use a quantized KV cache.

    For spectralquant: config.calibration_path is a base path (no extension) to
    files produced by python -m kv_quant.calibrate.

    For triattention: config.calibration_path is the path to a stats .pt file
    (from triattention/triattention/vllm/stats/ or scripts/calibrate.py). The
    model must have been loaded via AutoModelForCausalLM.from_pretrained.
    """
    if config.method == "spectralquant":
        if not config.calibration_path:
            raise ValueError("spectralquant requires config.calibration_path")
        cal_data = _load_spectralquant_cal(config.calibration_path)
    elif config.method == "triattention":
        _apply_triattention(model, config)
        cal_data = None
    else:
        cal_data = None

    n_kv_heads, head_dim = _get_kv_shape(model)
    device = next(model.parameters()).device

    _orig_generate = model.generate

    def _wrapped_generate(*args, **kwargs):
        if "past_key_values" not in kwargs:
            kwargs["past_key_values"] = _make_cache(config, n_kv_heads, head_dim, cal_data, device)
        return _orig_generate(*args, **kwargs)

    model.generate = _wrapped_generate
    model._kv_quant_config = config
    model._make_kv_cache = lambda: _make_cache(config, n_kv_heads, head_dim, cal_data, device)
    return model
```

- [ ] **Step 5: Run all tests to verify they pass**

```bash
pytest tests/test_cache.py -v
```

Expected: All tests pass including the 3 new triattention tests. The full suite should be 15 tests (5 turboquant + 4 spectralquant + 4 wrap + 2 new triattention cache tests + 1 new triattention requires-calibration test).

Also run the calibrate tests:
```bash
pytest tests/test_calibrate.py -v
```

Expected: All 4 tests pass (no changes to calibrate.py).

- [ ] **Step 6: Commit**

```bash
git add kv_quant/config.py kv_quant/__init__.py tests/test_cache.py
git commit -m "feat(triattention): add TriAttention token-eviction as kv_quant method=triattention"
```
