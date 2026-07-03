# TriAttention + Quantization Combination Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TriAttention token eviction as an orthogonal axis that composes with any quantization method in `kv_quant/`, so users can run TurboQuant+TriAttention or SpectralQuant+TriAttention (or standalone TriAttention with no quantization).

**Architecture:** `QuantConfig` gains a separate `eviction` field independent of `method`. For standalone eviction (`method=None`), use the official `apply_triattention_patch` which operates on a plain `DynamicCache`. For combined mode, a custom forward patch in `kv_quant/triattention_patch.py` calls `cache.evict(keep_indices)` — a new method on both `TurboQuantCache` and `SpectralQuantCache` — while `TriAttention.compute_keep_indices()` scores using a proxy built from `cache.get_kv(layer_idx)` (new dequantization accessor on each cache class).

**Tech Stack:** PyTorch, HuggingFace Transformers `DynamicCache`, local `triattention/triattention/` package (sys.path injection, not pip-installed), existing `kv_quant/` classes.

## Global Constraints

- No new pip dependencies — all imports from `triattention/` go through sys.path injection like spectralquant
- `triattention/` submodule root (containing the `triattention/` Python package) is at `<repo_root>/triattention/`; `import triattention` works after inserting `<repo>/triattention/` into sys.path
- `calibration_path` is reused for both spectralquant (base path, no extension) and triattention stats (`.pt` path) — never add new QuantConfig fields without updating this note
- `model.config._name_or_path` (set by `from_pretrained`) is the model path used for TriAttentionConfig
- `get_kv(layer_idx)` always returns float32 tensors (dequantized from storage)
- `evict(keep_indices)` applies to ALL layers; `keep_indices` is a 1D int64 tensor of size `budget`
- All existing 42 tests (22 test_cache.py + 4 test_calibrate.py, minus 2 skipped integration) must remain green

---

## File Structure

```
kv_quant/
  config.py              ← Task 1: add eviction field, make method Optional
  __init__.py            ← Task 1 (foundation), Task 4 (wire combined patch)
  turboquant.py          ← Task 2: add get_kv(), evict()
  spectralquant.py       ← Task 3: add get_kv(), evict()
  triattention_patch.py  ← Task 4: apply_standalone_patch(), apply_combined_eviction_patch()
tests/
  test_cache.py          ← Tasks 1-4: add tests per task
```

---

### Task 1: QuantConfig + standalone TriAttention + wrap() foundation

**Files:**
- Modify: `kv_quant/config.py`
- Modify: `kv_quant/__init__.py`
- Modify: `tests/test_cache.py`

**Interfaces:**
- Produces: `QuantConfig(method=Optional["turboquant"|"spectralquant"], eviction=Optional["triattention"], budget=int, divide_length=int)`
- Produces: `_make_plain_cache() -> DynamicCache` with `.compressed_bytes()` method
- Produces: `_ensure_triattention_on_path() -> None` in `__init__.py`
- Produces: `_apply_triattention_standalone(model, config) -> None` in `__init__.py`
- Produces: updated `wrap()` with eviction guard rails and standalone branch

- [ ] **Step 1: Write failing tests**

Add these tests at the bottom of `tests/test_cache.py`, after the existing wrap tests:

```python
# ---------------------------------------------------------------------------
# TriAttention guard rail tests (no model/stats needed — tests ValueError only)
# ---------------------------------------------------------------------------

def test_wrap_standalone_triattention_requires_calibration_path():
    """method=None, eviction=triattention, no calibration_path → ValueError."""
    model = _mock_model()
    cfg = QuantConfig(method=None, eviction="triattention", budget=256, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)


def test_wrap_combined_triattention_requires_calibration_path():
    """method=turboquant, eviction=triattention, no calibration_path → ValueError."""
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", eviction="triattention", budget=256, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)


def test_wrap_triattention_requires_model_name_or_path():
    """eviction=triattention with _name_or_path=None → ValueError."""
    model = _mock_model()
    model.config._name_or_path = None
    cfg = QuantConfig(method=None, eviction="triattention", budget=256, calibration_path="/fake/stats.pt")
    with pytest.raises(ValueError, match="_name_or_path"):
        wrap(model, cfg)


def test_plain_cache_compressed_bytes_single_layer():
    """_make_plain_cache() compressed_bytes() sums bfloat16 KV bytes."""
    from kv_quant import _make_plain_cache
    cache = _make_plain_cache()
    B, H, S, D = 1, 2, 10, 64
    k = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    v = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    cache.key_cache.append(k)
    cache.value_cache.append(v)
    expected = (k.nelement() + v.nelement()) * 2  # bfloat16 = 2 bytes
    assert cache.compressed_bytes() == expected


def test_plain_cache_compressed_bytes_multiple_layers():
    """_make_plain_cache() accumulates across two layers."""
    from kv_quant import _make_plain_cache
    cache = _make_plain_cache()
    B, H, S, D = 1, 2, 5, 64
    k = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    v = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    cache.key_cache.extend([k, k])
    cache.value_cache.extend([v, v])
    expected = 2 * (k.nelement() + v.nelement()) * 2
    assert cache.compressed_bytes() == expected
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cache.py::test_wrap_standalone_triattention_requires_calibration_path \
       tests/test_cache.py::test_wrap_combined_triattention_requires_calibration_path \
       tests/test_cache.py::test_wrap_triattention_requires_model_name_or_path \
       tests/test_cache.py::test_plain_cache_compressed_bytes_single_layer \
       tests/test_cache.py::test_plain_cache_compressed_bytes_multiple_layers -v
```

Expected: FAIL — `QuantConfig` doesn't accept `method=None` or `eviction` field; `_make_plain_cache` doesn't exist.

- [ ] **Step 3: Update `kv_quant/config.py`**

Replace the file entirely:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QuantConfig:
    method: Optional[Literal["turboquant", "spectralquant"]] = "turboquant"
    bits: int = 4               # key bits (TurboQuant/SpectralQuant only)
    value_bits: int = 2         # value bits for group quantization (TurboQuant only)
    value_group_size: int = 32  # group size for value quantization
    buffer_size: int = 128      # recent tokens kept in full precision (TurboQuant only)
    qjl_dim: int = 32           # QJL projection dim (SpectralQuant only)
    calibration_path: Optional[str] = None  # spectralquant: base path (no ext); triattention: stats .pt path
    signal_bit_boost: float = 2.0           # SpectralQuant only
    budget: int = 2048          # TriAttention: max KV tokens to retain after eviction
    divide_length: int = 128    # TriAttention: trigger eviction every N decode steps
    eviction: Optional[Literal["triattention"]] = None  # token eviction method (orthogonal to quantization)
```

- [ ] **Step 4: Update `kv_quant/__init__.py`**

Replace the file entirely:

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


def _make_plain_cache():
    """Return a DynamicCache with compressed_bytes() measuring actual KV storage."""
    from transformers import DynamicCache
    cache = DynamicCache()
    cache.compressed_bytes = lambda: sum(
        kc.nelement() * kc.element_size() + vc.nelement() * vc.element_size()
        for kc, vc in zip(cache.key_cache, cache.value_cache)
        if kc is not None
    )
    return cache


def _apply_triattention_standalone(model, config: QuantConfig) -> None:
    """Apply the official TriAttention patch for standalone eviction (no quantization).

    Patches model.forward via apply_triattention_patch, which handles position
    tracking and eviction on the standard DynamicCache injected by wrap().
    """
    from pathlib import Path
    _ensure_triattention_on_path()
    from triattention.methods.triattention import apply_triattention_patch

    apply_triattention_patch(
        model,
        stats_path=Path(config.calibration_path),
        model_path=Path(model.config._name_or_path),
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
    if config.method is None:
        return _make_plain_cache()
    raise ValueError(f"Unknown method: {config.method!r}")


def wrap(model, config: QuantConfig):
    """Patch model.generate() to inject a compressed/evicting KV cache.

    method controls quantization (None = no quantization):
      "turboquant"    — TurboQuant key/value quantization
      "spectralquant" — SpectralQuant per-head Lloyd-Max (requires calibration_path)
      None            — plain DynamicCache (no quantization)

    eviction controls token eviction applied on top of quantization:
      "triattention"  — TriAttention eviction (requires calibration_path + _name_or_path)
      None            — no eviction

    config.calibration_path serves two roles:
      spectralquant: base path (no extension) for files from kv_quant.calibrate
      triattention:  path to a stats .pt file from triattention/triattention/vllm/stats/
    """
    if config.method == "spectralquant":
        if not config.calibration_path:
            raise ValueError("spectralquant requires config.calibration_path")
        cal_data = _load_spectralquant_cal(config.calibration_path)
    else:
        cal_data = None

    if config.eviction == "triattention":
        if not config.calibration_path:
            raise ValueError(
                "triattention eviction requires config.calibration_path "
                "(path to a stats .pt file from triattention/triattention/vllm/stats/)"
            )
        model_path = getattr(model.config, "_name_or_path", None)
        if not model_path:
            raise ValueError(
                "triattention requires model.config._name_or_path to be set. "
                "Load the model via AutoModelForCausalLM.from_pretrained."
            )
        if config.method is None:
            _apply_triattention_standalone(model, config)
        else:
            # Combined: Task 4 wires in apply_combined_eviction_patch here
            from kv_quant.triattention_patch import apply_combined_eviction_patch
            apply_combined_eviction_patch(model, config)

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

> Note: The `from kv_quant.triattention_patch import apply_combined_eviction_patch` line will fail with ImportError until Task 4 creates that file. Task 1 tests only exercise guard rails that fire BEFORE this import, so all Task 1 tests pass regardless. Task 4 creates `triattention_patch.py`.

- [ ] **Step 5: Run all tests**

```bash
pytest tests/test_cache.py tests/test_calibrate.py -v
```

Expected: All previously passing tests still pass. The 5 new tests also pass. Total ≈ 31 tests.

- [ ] **Step 6: Commit**

```bash
git add kv_quant/config.py kv_quant/__init__.py tests/test_cache.py
git commit -m "feat(kv-quant): add eviction field to QuantConfig; standalone TriAttention guard rails"
```

---

### Task 2: TurboQuantCache.get_kv() + TurboQuantCache.evict()

**Files:**
- Modify: `kv_quant/turboquant.py`
- Modify: `tests/test_cache.py`

**Interfaces:**
- Consumes: `TurboQuantCache` (existing), `ProdQuantized`, `ValueQuantized`, `dequantize_values`, `_flush_to_quantized()` (existing private method)
- Produces: `TurboQuantCache.get_kv(layer_idx: int) -> tuple[Tensor, Tensor]` — both tensors float32, shape `(B, H, S, D)`
- Produces: `TurboQuantCache.evict(keep_indices: Tensor) -> None` — keep_indices is 1D int64, applies to ALL layers

**Key data structures in `TurboQuantCache`:**
- `_qk[l]` = `ProdQuantized` (quantized keys for tokens older than buffer) or None
- `_qv[l]` = `ValueQuantized` (quantized values for tokens older than buffer) or None
- `_k_buf[l]` = Tensor `(B, H, buf_len, D)` — recent tokens, full precision
- `_v_buf[l]` = Tensor `(B, H, buf_len, D)` — recent values, full precision
- `_flush_to_quantized(layer_idx, keys, values)` — existing method; appends to `_qk`/`_qv`

- [ ] **Step 1: Write failing tests**

Add after the existing turboquant tests in `tests/test_cache.py`, before `_make_spectralquant_cal_data`:

```python
def test_turboquant_get_kv_shape_buffer_only():
    """get_kv() returns correct shape when all tokens are in buffer."""
    cfg = QuantConfig(bits=4, buffer_size=128)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k, v = _make_kv(seq=5, heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    k_out, v_out = cache.get_kv(0)
    assert k_out.shape == (1, 2, 5, 64)
    assert v_out.shape == (1, 2, 5, 64)
    assert k_out.dtype == torch.float32


def test_turboquant_get_kv_shape_with_quantized():
    """get_kv() reconstructs both quantized and buffer portions."""
    cfg = QuantConfig(bits=4, buffer_size=4)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k, v = _make_kv(seq=6, heads=2, d=64)  # 6 tokens → 2 flushed to quantized
    cache.update(k, v, layer_idx=0)
    k_out, v_out = cache.get_kv(0)
    assert k_out.shape == (1, 2, 6, 64)
    assert v_out.shape == (1, 2, 6, 64)
    assert k_out.dtype == torch.float32


def test_turboquant_evict_reduces_seq_length():
    """After evict(keep_indices), get_seq_length() == len(keep_indices)."""
    cfg = QuantConfig(bits=4, buffer_size=128)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k, v = _make_kv(seq=6, heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    keep = torch.tensor([0, 2, 4])
    cache.evict(keep)
    assert cache.get_seq_length(0) == 3


def test_turboquant_evict_multiple_layers():
    """evict() applies to all layers."""
    cfg = QuantConfig(bits=4, buffer_size=128)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k, v = _make_kv(seq=5, heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)
    keep = torch.tensor([0, 2, 4])
    cache.evict(keep)
    assert cache.get_seq_length(0) == 3
    assert cache.get_seq_length(1) == 3


def test_turboquant_evict_then_update_works():
    """After evict(), update() still appends correctly."""
    cfg = QuantConfig(bits=4, buffer_size=128)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k, v = _make_kv(seq=6, heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    keep = torch.tensor([0, 2, 4])
    cache.evict(keep)
    k_new, v_new = _make_kv(seq=1, heads=2, d=64)
    k_out, v_out = cache.update(k_new, v_new, layer_idx=0)
    assert k_out.shape[-2] == 4  # 3 kept + 1 new
    assert cache.get_seq_length(0) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cache.py::test_turboquant_get_kv_shape_buffer_only \
       tests/test_cache.py::test_turboquant_get_kv_shape_with_quantized \
       tests/test_cache.py::test_turboquant_evict_reduces_seq_length \
       tests/test_cache.py::test_turboquant_evict_multiple_layers \
       tests/test_cache.py::test_turboquant_evict_then_update_works -v
```

Expected: FAIL — `TurboQuantCache` has no `get_kv` or `evict` methods.

- [ ] **Step 3: Add `get_kv()` and `evict()` to `kv_quant/turboquant.py`**

Add the following two methods to the `TurboQuantCache` class, after the existing `get_seq_length` method:

```python
    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full dequantized (keys, values) for layer_idx as float32 tensors.

        Shape: (B, H, S, D) where S = get_seq_length(layer_idx).
        Raises IndexError if layer_idx has not been populated yet.
        """
        if layer_idx >= len(self._qk):
            raise IndexError(
                f"layer_idx {layer_idx} not in cache (cache has {len(self._qk)} layers)"
            )

        # Determine device from buffer or stored device hint
        if self._k_buf[layer_idx] is not None:
            device = self._k_buf[layer_idx].device
        elif self._device is not None:
            device = self._device
        else:
            device = torch.device("cpu")

        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []

        if self._qk[layer_idx] is not None:
            q = self._get_quantizer(layer_idx, device)
            parts_k.append(q.dequantize(self._qk[layer_idx]))
            parts_v.append(dequantize_values(self._qv[layer_idx], self.config.value_group_size))

        if self._k_buf[layer_idx] is not None:
            parts_k.append(self._k_buf[layer_idx].float())
            parts_v.append(self._v_buf[layer_idx].float())

        if not parts_k:
            raise ValueError(f"layer_idx {layer_idx} has no cached tokens")

        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    def evict(self, keep_indices: torch.Tensor) -> None:
        """Retain only the tokens at keep_indices, discarding all others.

        Applies to ALL layers. keep_indices is a 1D int64 tensor of positions
        to keep (size = budget, values in [0, get_seq_length()-1]).

        After eviction, the kept tokens are re-split: tokens beyond buffer_size
        go to quantized storage, the most recent buffer_size tokens stay in the
        buffer at their original dtype.
        """
        for layer_idx in range(len(self._qk)):
            if self._k_buf[layer_idx] is None and self._qk[layer_idx] is None:
                continue

            # Capture original dtype before clearing (need it for re-storage)
            orig_dtype = (
                self._k_buf[layer_idx].dtype
                if self._k_buf[layer_idx] is not None
                else torch.bfloat16
            )

            k_full, v_full = self.get_kv(layer_idx)          # (B, H, S, D) float32
            k_kept = k_full[..., keep_indices, :]             # (B, H, budget, D) float32
            v_kept = v_full[..., keep_indices, :]

            # Clear layer storage
            self._qk[layer_idx] = None
            self._qv[layer_idx] = None
            self._k_buf[layer_idx] = None
            self._v_buf[layer_idx] = None

            # Re-split: oldest tokens → quantized store; newest → buffer
            budget = k_kept.shape[-2]
            if budget <= self.config.buffer_size:
                self._k_buf[layer_idx] = k_kept.to(orig_dtype)
                self._v_buf[layer_idx] = v_kept.to(orig_dtype)
            else:
                n_quant = budget - self.config.buffer_size
                # _flush_to_quantized calls keys.float() internally, so any dtype is fine
                self._flush_to_quantized(
                    layer_idx,
                    k_kept[..., :n_quant, :].to(orig_dtype),
                    v_kept[..., :n_quant, :].to(orig_dtype),
                )
                self._k_buf[layer_idx] = k_kept[..., n_quant:, :].to(orig_dtype)
                self._v_buf[layer_idx] = v_kept[..., n_quant:, :].to(orig_dtype)
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_cache.py tests/test_calibrate.py -v
```

Expected: All tests pass. The 5 new turboquant tests pass; no regressions.

- [ ] **Step 5: Commit**

```bash
git add kv_quant/turboquant.py tests/test_cache.py
git commit -m "feat(turboquant): add get_kv() and evict() for TriAttention combination support"
```

---

### Task 3: SpectralQuantCache.get_kv() + SpectralQuantCache.evict()

**Files:**
- Modify: `kv_quant/spectralquant.py`
- Modify: `tests/test_cache.py`

**Interfaces:**
- Consumes: `SpectralQuantCache` (existing), `CompressedVector`, `_key_rot.unrotate()`, `_val_rot.unrotate()`, `_key_quants`, `_val_quants`, `_head_meta`, `_sk_sem/_sk_tail/_sv_sem/_sv_tail`
- Produces: `SpectralQuantCache.get_kv(layer_idx: int) -> tuple[Tensor, Tensor]` — both float32, shape `(B, H, S, D)`
- Produces: `SpectralQuantCache.evict(keep_indices: Tensor) -> None` — slices stored index tensors along the sequence dimension

**Key data structures in `SpectralQuantCache`:**
- `_sk_sem[l][h]` — semantic key indices, shape `(B, S, d_eff_int)`, dtype int32
- `_sk_tail[l][h]` — tail key indices, shape `(B, S, D - d_eff_int)`, dtype int32
- `_sv_sem[l][h]` — semantic value indices, same shape
- `_sv_tail[l][h]` — tail value indices, same shape
- `_head_meta[(l, h, "key")]` — dict with `d_eff_int, b_high, b_low, head_dim`
- `_key_quants[(l, h)]` — fitted `NonUniformQuantizer` for keys
- `_val_quants[(l, h)]` — fitted `NonUniformQuantizer` for values
- `_key_rot`, `_val_rot` — `SpectralRotation` instances

Eviction for SpectralQuantCache is simpler than TurboQuant: just slice the stored index tensors on dim=1 (the sequence dimension). No re-quantization needed.

- [ ] **Step 1: Write failing tests**

Add after the existing spectralquant tests in `tests/test_cache.py`, before the wrap() section:

```python
def test_spectralquant_get_kv_shape():
    """get_kv() returns (B, H, S, D) float32 tensors."""
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    k_out, v_out = cache.get_kv(0)
    assert k_out.shape == (1, 2, 5, 64)
    assert v_out.shape == (1, 2, 5, 64)
    assert k_out.dtype == torch.float32


def test_spectralquant_evict_reduces_seq_length():
    """After evict(keep_indices), get_seq_length() == len(keep_indices)."""
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(seq=5, heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    keep = torch.tensor([0, 2, 4])
    cache.evict(keep)
    assert cache.get_seq_length(0) == 3


def test_spectralquant_evict_multiple_layers():
    """evict() applies to all layers."""
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data(n_layers=2)
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(seq=5, heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)
    keep = torch.tensor([1, 3])
    cache.evict(keep)
    assert cache.get_seq_length(0) == 2
    assert cache.get_seq_length(1) == 2


def test_spectralquant_evict_then_update_works():
    """After evict(), update() still appends and returns correct shape."""
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(seq=5, heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    keep = torch.tensor([0, 2, 4])
    cache.evict(keep)
    k_new, v_new = _make_kv(seq=1, heads=2, d=64)
    k_out, v_out = cache.update(k_new, v_new, layer_idx=0)
    assert k_out.shape[-2] == 4  # 3 kept + 1 new
    assert cache.get_seq_length(0) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cache.py::test_spectralquant_get_kv_shape \
       tests/test_cache.py::test_spectralquant_evict_reduces_seq_length \
       tests/test_cache.py::test_spectralquant_evict_multiple_layers \
       tests/test_cache.py::test_spectralquant_evict_then_update_works -v
```

Expected: FAIL — `SpectralQuantCache` has no `get_kv` or `evict` methods.

- [ ] **Step 3: Add `get_kv()` and `evict()` to `kv_quant/spectralquant.py`**

Add the following two methods to the `SpectralQuantCache` class, after the existing `compressed_bytes` method:

```python
    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full dequantized (keys, values) for layer_idx as float32 tensors.

        Shape: (B, H, S, D). Decompresses stored index tensors using per-head
        NonUniformQuantizer and SpectralRotation. Raises IndexError if layer has
        no cached tokens.
        """
        from spectralquant.nonuniform_quantization import CompressedVector

        if layer_idx >= len(self._sk_sem) or self._sk_sem[layer_idx][0] is None:
            raise IndexError(f"layer_idx {layer_idx} has no cached tokens")

        k_hat_heads: list[torch.Tensor] = []
        v_hat_heads: list[torch.Tensor] = []

        for h in range(len(self._sk_sem[layer_idx])):
            k_meta = self._head_meta[(layer_idx, h, "key")]
            B = self._sk_sem[layer_idx][h].shape[0]
            S = self._sk_sem[layer_idx][h].shape[1]
            D = k_meta["head_dim"]

            k_cv = CompressedVector(
                semantic_indices=self._sk_sem[layer_idx][h],
                tail_indices=self._sk_tail[layer_idx][h],
                d_eff=k_meta["d_eff_int"],
                head_dim=D,
                b_high=k_meta["b_high"],
                b_low=k_meta["b_low"],
                original_shape=(B, S, D),
            )
            k_hat = self._key_rot.unrotate(
                self._key_quants[(layer_idx, h)].decompress(k_cv), layer_idx, h
            )  # (B, S, D)
            k_hat_heads.append(k_hat.float())

            v_meta = self._head_meta[(layer_idx, h, "value")]
            v_cv = CompressedVector(
                semantic_indices=self._sv_sem[layer_idx][h],
                tail_indices=self._sv_tail[layer_idx][h],
                d_eff=v_meta["d_eff_int"],
                head_dim=D,
                b_high=v_meta["b_high"],
                b_low=v_meta["b_low"],
                original_shape=(B, S, D),
            )
            v_hat = self._val_rot.unrotate(
                self._val_quants[(layer_idx, h)].decompress(v_cv), layer_idx, h
            )
            v_hat_heads.append(v_hat.float())

        k_full = torch.stack(k_hat_heads, dim=1)   # (B, H, S, D)
        v_full = torch.stack(v_hat_heads, dim=1)
        return k_full, v_full

    def evict(self, keep_indices: torch.Tensor) -> None:
        """Retain only the tokens at keep_indices, discarding all others.

        Applies to ALL layers. Slices the stored index tensors along dim=1
        (the sequence dimension) — no re-quantization needed.
        """
        for l in range(len(self._sk_sem)):
            for h in range(len(self._sk_sem[l])):
                if self._sk_sem[l][h] is None:
                    continue
                # Index tensors have shape (B, S, features) — slice on dim=1
                self._sk_sem[l][h] = self._sk_sem[l][h][:, keep_indices, :]
                self._sk_tail[l][h] = self._sk_tail[l][h][:, keep_indices, :]
                self._sv_sem[l][h] = self._sv_sem[l][h][:, keep_indices, :]
                self._sv_tail[l][h] = self._sv_tail[l][h][:, keep_indices, :]
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_cache.py tests/test_calibrate.py -v
```

Expected: All tests pass. The 4 new spectralquant tests pass; no regressions.

- [ ] **Step 5: Commit**

```bash
git add kv_quant/spectralquant.py tests/test_cache.py
git commit -m "feat(spectralquant): add get_kv() and evict() for TriAttention combination support"
```

---

### Task 4: Combined eviction patch + wire into wrap()

**Files:**
- Create: `kv_quant/triattention_patch.py`
- Modify: `kv_quant/__init__.py` (the `wrap()` in Task 1 already has the import; this task creates the file)
- Modify: `tests/test_cache.py`

**Interfaces:**
- Consumes: `TurboQuantCache.get_kv()` + `.evict()` (Task 2), `SpectralQuantCache.get_kv()` + `.evict()` (Task 3), `TriAttention`, `TriAttentionConfig` from `triattention.methods.triattention`
- Produces: `apply_combined_eviction_patch(model, config: QuantConfig) -> None` — patches `model.forward`; sets `model._triattention_compressor`

**How the combined patch works:**
1. Prefill (first call, `cache.get_seq_length() == 0`): runs forward normally, then initializes `comp.cache_positions`, `comp.absolute_position`, `comp.prefix_length` from the resulting cache length
2. Decode: overrides `position_ids` in kwargs to use `comp.absolute_position` (not `cache.get_seq_length()`, which equals `budget` after eviction). Runs forward. Updates position tracking. If `new_seq_len > budget AND absolute_position % divide_length == 0`, builds a proxy namespace with `key_cache`/`value_cache` from `cache.get_kv(l)` for each layer, calls `comp.compute_keep_indices(proxy, ...)`, then `cache.evict(keep_indices)`.

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_cache.py` at the bottom:

```python
# ---------------------------------------------------------------------------
# Combined mode guard rail tests (no real TriAttention needed)
# ---------------------------------------------------------------------------

def test_wrap_combined_turboquant_requires_calibration_path():
    """method=turboquant + eviction=triattention + no calibration_path → ValueError."""
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", eviction="triattention", budget=256, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)


def test_wrap_combined_spectralquant_requires_calibration_path():
    """method=spectralquant + eviction=triattention + no calibration_path → ValueError."""
    model = _mock_model()
    cfg = QuantConfig(method="spectralquant", eviction="triattention", budget=256, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)


def test_wrap_combined_requires_model_name_or_path():
    """method=turboquant + eviction=triattention + _name_or_path=None → ValueError."""
    model = _mock_model()
    model.config._name_or_path = None
    cfg = QuantConfig(method="turboquant", eviction="triattention", budget=256, calibration_path="/fake/stats.pt")
    with pytest.raises(ValueError, match="_name_or_path"):
        wrap(model, cfg)
```

- [ ] **Step 2: Run tests to verify the 3 new tests fail (ImportError from missing triattention_patch)**

```bash
pytest tests/test_cache.py::test_wrap_combined_turboquant_requires_calibration_path \
       tests/test_cache.py::test_wrap_combined_spectralquant_requires_calibration_path \
       tests/test_cache.py::test_wrap_combined_requires_model_name_or_path -v
```

Expected: FAIL — the guard rail tests pass their guards but hit `ImportError: No module named 'kv_quant.triattention_patch'`.

Actually, the `calibration_path` guard fires BEFORE the `triattention_patch` import, so `test_wrap_combined_turboquant_requires_calibration_path` and `test_wrap_combined_spectralquant_requires_calibration_path` should already PASS from Task 1. Only `test_wrap_combined_requires_model_name_or_path` will hit the ImportError. Re-check after running.

- [ ] **Step 3: Create `kv_quant/triattention_patch.py`**

Create this new file:

```python
"""Combined quantization + TriAttention eviction forward patch.

apply_combined_eviction_patch() patches model.forward to apply TriAttention
token eviction on top of any quantized KV cache that exposes get_kv(layer_idx)
and evict(keep_indices). Use this instead of the official apply_triattention_patch()
when a quantized cache (TurboQuantCache, SpectralQuantCache) is being injected
by wrap() — the official patch only works on plain DynamicCache.
"""
from __future__ import annotations
import os
import sys
import types
from pathlib import Path

import torch

from kv_quant.config import QuantConfig


def _ensure_triattention_on_path() -> None:
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "triattention"))
    if src not in sys.path:
        sys.path.insert(0, src)


def apply_combined_eviction_patch(model, config: QuantConfig) -> None:
    """Patch model.forward to evict tokens from a quantized KV cache.

    Preconditions (wrap() guarantees these before calling):
      - config.calibration_path is a valid path to a TriAttention stats .pt file
      - model.config._name_or_path is set to a HF model ID or local path
      - config.method is not None (a quantized cache with evict() will be injected)

    After patching:
      - model._triattention_compressor holds the TriAttention instance
      - model.forward intercepts decode steps to track positions and trigger eviction
    """
    _ensure_triattention_on_path()
    from triattention.methods.triattention import TriAttention, TriAttentionConfig

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_layers = model.config.num_hidden_layers

    comp = TriAttention(TriAttentionConfig(
        stats_path=Path(config.calibration_path),
        model_path=Path(model.config._name_or_path),
        device=device,
        dtype=dtype,
        budget=config.budget,
        divide_length=config.divide_length,
    ))
    model._triattention_compressor = comp

    _orig_forward = model.forward

    def _patched_forward(self_model, *args, **kwargs):
        # Resolve input_ids from positional or keyword args
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        past_kv = kwargs.get("past_key_values")

        # Pass through if not a cache with evict() (e.g., during non-generate calls)
        if past_kv is None or not hasattr(past_kv, "evict"):
            return _orig_forward(*args, **kwargs)

        seq_len = input_ids.shape[-1] if input_ids is not None else 1
        cached_len = past_kv.get_seq_length()

        # Prefill: cache is empty on the first call
        if cached_len == 0:
            output = _orig_forward(*args, **kwargs)
            filled = past_kv.get_seq_length()
            comp.cache_positions = list(range(filled))
            comp.absolute_position = filled
            comp.prefix_length = filled
            return output

        # Decode: override position_ids so the new token(s) have the correct
        # absolute position (after eviction, cached_len == budget but
        # comp.absolute_position > budget, so HF's default would be wrong)
        tok_device = input_ids.device if input_ids is not None else device
        kwargs["position_ids"] = torch.arange(
            comp.absolute_position,
            comp.absolute_position + seq_len,
            device=tok_device,
        ).unsqueeze(0)

        output = _orig_forward(*args, **kwargs)

        # Update position tracking after the forward pass
        comp.cache_positions.extend(
            range(comp.absolute_position, comp.absolute_position + seq_len)
        )
        comp.absolute_position += seq_len

        # Trigger eviction if above budget at the right interval
        new_seq_len = past_kv.get_seq_length()
        if (
            new_seq_len > config.budget
            and comp.absolute_position % config.divide_length == 0
        ):
            # Build a proxy with key_cache / value_cache lists for TriAttention scoring
            kv_pairs = [past_kv.get_kv(l) for l in range(n_layers)]
            proxy = types.SimpleNamespace(
                key_cache=[k for k, _v in kv_pairs],
                value_cache=[_v for _k, _v in kv_pairs],
            )
            keep_indices = comp.compute_keep_indices(
                proxy, prefix_length=getattr(comp, "prefix_length", 0)
            )
            past_kv.evict(keep_indices)
            comp.cache_positions = [comp.cache_positions[i] for i in keep_indices.tolist()]

        return output

    model.forward = types.MethodType(_patched_forward, model)
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_cache.py tests/test_calibrate.py -v
```

Expected: All tests pass. The 3 new combined guard rail tests pass. All previously passing tests unaffected. Total ≈ 42 tests (same as before + new ones).

- [ ] **Step 5: Commit**

```bash
git add kv_quant/triattention_patch.py tests/test_cache.py
git commit -m "feat(kv-quant): add combined TriAttention+quantization forward patch"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ `method=None` (no quantization + standalone eviction): Task 1
- ✅ `method="turboquant"|"spectralquant"` (quantization only): unchanged from existing code
- ✅ `method="turboquant"|"spectralquant"` + `eviction="triattention"` (combined): Tasks 2, 3, 4
- ✅ `get_kv()` interface on both cache classes: Tasks 2, 3
- ✅ `evict()` interface on both cache classes: Tasks 2, 3
- ✅ `_make_plain_cache()` public helper: Task 1
- ✅ Guard rails (ValueError) for all combinations: Tasks 1, 4
- ✅ Position tracking + position_ids override in combined patch: Task 4

**2. Placeholder scan:** None found. All steps contain complete code.

**3. Type consistency:**
- `get_kv(layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]` — same signature in Tasks 2, 3, and used identically in Task 4 ✅
- `evict(keep_indices: torch.Tensor) -> None` — same signature in Tasks 2, 3, called identically in Task 4 ✅
- `_make_plain_cache()` defined in Task 1 `__init__.py`, imported in Task 1 tests ✅
- `apply_combined_eviction_patch(model, config: QuantConfig) -> None` — imported in Task 1's `wrap()` (deferred import), defined in Task 4 ✅
