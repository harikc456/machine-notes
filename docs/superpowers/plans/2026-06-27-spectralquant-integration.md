# SpectralQuant Official Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the k-means VQ + QJL implementation in `kv_quant/spectralquant.py` and `kv_quant/calibrate.py` with the official `NonUniformQuantizer` + `SpectralRotation` from `spectralquant/src/spectralquant/`.

**Architecture:** The official SpectralQuant package lives at `spectralquant/src/spectralquant/` and is NOT pip-installed. Files that import from it must add `spectralquant/src` to `sys.path`. Calibration produces three files (`<base>.pt`, `<base>_meta.json`, `<base>_quantizers.pt`). `SpectralQuantCache` is a `DynamicCache` subclass (HF-compatible) that reconstructs quantizers from saved centroids and uses `SpectralRotation` + `NonUniformQuantizer.compress/decompress` on every `update()` call.

**Tech Stack:** PyTorch, transformers DynamicCache, spectralquant (local package), numpy

## Global Constraints

- `spectralquant` package is at `spectralquant/src/spectralquant/` — NOT pip-installed. Add this dir to `sys.path` in every file that imports from it; use `os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src"))` relative to `kv_quant/` files, and the same path adjusted for `tests/` files.
- Do NOT use `use_water_fill=True` in `NonUniformQuantizer` — always use v1 (single shared semantic codebook). This guarantees `_semantic_quantizer` is always set and `_per_dim_semantic_quantizers` is `None`.
- `calibration_path` in `QuantConfig` is now treated as a **base path without extension** for spectralquant (e.g., `"spectralquant_qwen3"`). The loader reads `<base>.pt`, `<base>_meta.json`, and `<base>_quantizers.pt`.
- Do NOT modify: `kv_quant/turboquant.py`, `kv_quant/ops/`, `chat/`, `tests/test_ops.py`, `tests/test_integration.py`, `tests/test_kv_quant_utils.py`.
- `weights_only=True` on all `torch.load` calls.
- Tests must not download any model or dataset — all test data constructed in-memory.
- After both tasks, `pytest tests/test_cache.py -v` must show all remaining tests passing.

---

### Task 1: Rewrite `kv_quant/calibrate.py` and remove stale tests

**Files:**
- Rewrite: `kv_quant/calibrate.py`
- Modify: `tests/test_cache.py` — remove `_compute_bit_split` import (line 7) and two tests (lines 69-84)

**Interfaces:**
- Produces: `calibrate(model_id, base_path, n_seqs, bits, device)` function (same CLI interface, `--output` is now a base path without extension)
- Produces: save format — `<base_path>.pt`, `<base_path>_meta.json` (EigenspectralCalibrator format), `<base_path>_quantizers.pt` (quant states dict, see below)
- quant_states dict format (keyed by `"L{l}_H{h}_key"` and `"L{l}_H{h}_value"`):
  ```python
  {
      "L0_H0_key": {
          "semantic_centroids": Tensor,  # float32, shape (2**b_high,), sorted
          "tail_centroids": Tensor,       # float32, shape (2**b_low,), sorted
          "d_eff_int": int,
          "b_high": int,
          "b_low": int,
          "head_dim": int,
      },
      "L0_H0_value": { ... },
      ...
  }
  ```
- Removes: `_compute_bit_split` function (replaced by `BitAllocator` inside `NonUniformQuantizer`)
- `_kmeans_codebook` and `make_sign_matrix` usage removed

- [ ] **Step 1: Write a failing test for the new calibrate.py save format**

Write in `tests/test_calibrate.py` (new file):

```python
"""Tests for kv_quant.calibrate — validates the new spectralquant save format."""
from __future__ import annotations
import os, sys, tempfile
import torch
import pytest

_SPECTRALQUANT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
)
if _SPECTRALQUANT_SRC not in sys.path:
    sys.path.insert(0, _SPECTRALQUANT_SRC)


def _make_fake_kv(n_layers=2, n_kv_heads=2, n_tokens=50, head_dim=64):
    """Synthetic K/V tensors simulating DynamicCache output."""
    torch.manual_seed(0)
    # all_keys[l] = list of (n_kv_heads, seq, head_dim) tensors
    all_keys = [[torch.randn(n_kv_heads, n_tokens, head_dim)] for _ in range(n_layers)]
    all_vals = [[torch.randn(n_kv_heads, n_tokens, head_dim)] for _ in range(n_layers)]
    return all_keys, all_vals


def test_save_format_files_exist():
    """calibrate() produces three files: .pt, _meta.json, _quantizers.pt."""
    from kv_quant.calibrate import _calibrate_from_kv
    all_keys, all_vals = _make_fake_kv()
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "cal")
        _calibrate_from_kv(all_keys, all_vals, head_dim=64, bits=4, base_path=base)
        assert os.path.exists(base + ".pt")
        assert os.path.exists(base + "_meta.json")
        assert os.path.exists(base + "_quantizers.pt")


def test_quant_state_keys():
    """quant_states dict has keys 'L{l}_H{h}_key' and 'L{l}_H{h}_value'."""
    from kv_quant.calibrate import _calibrate_from_kv
    all_keys, all_vals = _make_fake_kv(n_layers=1, n_kv_heads=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "cal")
        _calibrate_from_kv(all_keys, all_vals, head_dim=64, bits=4, base_path=base)
        qs = torch.load(base + "_quantizers.pt", map_location="cpu", weights_only=True)
        assert "L0_H0_key" in qs
        assert "L0_H1_key" in qs
        assert "L0_H0_value" in qs


def test_quant_state_fields():
    """Each quant_state entry has the required fields."""
    from kv_quant.calibrate import _calibrate_from_kv
    all_keys, all_vals = _make_fake_kv(n_layers=1, n_kv_heads=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "cal")
        _calibrate_from_kv(all_keys, all_vals, head_dim=64, bits=4, base_path=base)
        qs = torch.load(base + "_quantizers.pt", map_location="cpu", weights_only=True)
        state = qs["L0_H0_key"]
        assert "semantic_centroids" in state
        assert "tail_centroids" in state
        assert "d_eff_int" in state
        assert "b_high" in state
        assert "b_low" in state
        assert "head_dim" in state
        assert state["head_dim"] == 64
        assert 1 <= state["b_high"] <= 8
        assert 1 <= state["b_low"] <= 8


def test_calibrator_can_be_reloaded():
    """EigenspectralCalibrator.load() can reload the .pt + _meta.json."""
    from kv_quant.calibrate import _calibrate_from_kv
    from spectralquant.calibration import EigenspectralCalibrator
    all_keys, all_vals = _make_fake_kv(n_layers=1, n_kv_heads=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "cal")
        _calibrate_from_kv(all_keys, all_vals, head_dim=64, bits=4, base_path=base)
        cal = EigenspectralCalibrator()
        cal.load(base)
        assert cal._is_calibrated
        hcd = cal.get(0, 0, "key")
        assert hcd is not None
        assert hcd.eigenvectors.shape == (64, 64)
        assert hcd.eigenvalues.shape == (64,)
```

Run: `pytest tests/test_calibrate.py -v`
Expected: 4 FAILED (ImportError or AttributeError — `_calibrate_from_kv` doesn't exist yet)

- [ ] **Step 2: Write the new `kv_quant/calibrate.py`**

Replace the entire file with:

```python
"""SpectralQuant calibration: per-head eigenvectors and Lloyd-Max codebooks.

Usage:
    python -m kv_quant.calibrate \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --output spectralquant_qwen25_7b \\
        --n-seqs 100 --bits 4
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
from tqdm import tqdm

_SPECTRALQUANT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
)
if _SPECTRALQUANT_SRC not in sys.path:
    sys.path.insert(0, _SPECTRALQUANT_SRC)

from spectralquant.calibration import (
    EigenspectralCalibrator,
    HeadCalibrationData,
    _compute_covariance,
    _eigendecompose,
    _participation_ratio,
    _spectral_gap,
    _cumulative_variance_thresholds,
)
from spectralquant.nonuniform_quantization import NonUniformQuantizer


def _calibrate_from_kv(
    all_keys: list[list[torch.Tensor]],
    all_vals: list[list[torch.Tensor]],
    head_dim: int,
    bits: int,
    base_path: str,
) -> None:
    """Compute calibration from pre-collected K/V tensors and save results.

    Parameters
    ----------
    all_keys:
        all_keys[layer_idx] = list of (n_kv_heads, seq_len, head_dim) float tensors.
    all_vals:
        Same shape as all_keys but for values.
    head_dim:
        Head dimension.
    bits:
        Target average bits per coordinate (passed as avg_bits to NonUniformQuantizer).
    base_path:
        Output base path without extension. Writes <base>.pt, <base>_meta.json,
        <base>_quantizers.pt.
    """
    n_layers = len(all_keys)
    n_kv_heads = all_keys[0][0].shape[0] if all_keys and all_keys[0] else 0

    calibrator = EigenspectralCalibrator()
    quant_state: dict = {}

    for layer_idx in range(n_layers):
        if not all_keys[layer_idx]:
            continue
        # (n_kv_heads, total_tokens, head_dim)
        layer_keys = torch.cat(all_keys[layer_idx], dim=1).float()
        layer_vals = torch.cat(all_vals[layer_idx], dim=1).float()

        for head_idx in range(n_kv_heads):
            for kv_type, layer_data in (("key", layer_keys), ("value", layer_vals)):
                vectors = layer_data[head_idx]  # (total_tokens, head_dim)
                if vectors.shape[0] < 2:
                    continue

                cov = _compute_covariance(vectors)
                eigenvalues, eigenvectors = _eigendecompose(cov)
                d_eff = _participation_ratio(eigenvalues)
                gap = _spectral_gap(eigenvalues, d_eff)
                var_95, var_99 = _cumulative_variance_thresholds(eigenvalues)

                calibrator._calibration_data[(layer_idx, head_idx, kv_type)] = HeadCalibrationData(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    head_type=kv_type,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    d_eff=d_eff,
                    spectral_gap=gap,
                    var_95=var_95,
                    var_99=var_99,
                    n_samples=vectors.shape[0],
                    head_dim=head_dim,
                )

                # Forward rotation: x @ V (same as SpectralRotation.rotate)
                rotated = vectors @ eigenvectors  # (n_tokens, head_dim)
                quant = NonUniformQuantizer(eigenvalues=eigenvalues, avg_bits=float(bits))
                quant.fit(rotated, d_eff=d_eff)

                quant_state[f"L{layer_idx}_H{head_idx}_{kv_type}"] = {
                    "semantic_centroids": quant._semantic_quantizer._centroids.clone(),
                    "tail_centroids": quant._tail_quantizer._centroids.clone(),
                    "d_eff_int": quant._d_eff_int,
                    "b_high": quant._b_high,
                    "b_low": quant._b_low,
                    "head_dim": head_dim,
                }

    calibrator._is_calibrated = True
    calibrator.save(base_path)
    torch.save(quant_state, base_path + "_quantizers.pt")
    print(f"Saved: {base_path}.pt / {base_path}_meta.json / {base_path}_quantizers.pt")


def calibrate(
    model_id: str,
    base_path: str,
    n_seqs: int = 100,
    bits: int = 4,
    device: str = "cuda",
) -> None:
    """Calibrate a model and save results to disk.

    Parameters
    ----------
    model_id:
        HuggingFace model ID.
    base_path:
        Output base path without extension.
    n_seqs:
        Number of wikitext sequences to process.
    bits:
        Target average bits per coordinate.
    device:
        Torch device string.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [ex["text"] for ex in dataset if len(ex["text"].strip()) > 100][:n_seqs]

    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )

    all_keys: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    all_vals: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    for text in tqdm(texts, desc="Collecting KV vectors"):
        ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(device)
        cache = DynamicCache()
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True)
        for l in range(min(n_layers, len(cache.key_cache))):
            all_keys[l].append(cache.key_cache[l][0].float().cpu())
            all_vals[l].append(cache.value_cache[l][0].float().cpu())

    _calibrate_from_kv(all_keys, all_vals, head_dim=head_dim, bits=bits, base_path=base_path)
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="Base path without extension")
    parser.add_argument("--n-seqs", type=int, default=100)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    calibrate(args.model, args.output, args.n_seqs, args.bits, args.device)
```

- [ ] **Step 3: Remove `_compute_bit_split` import and two tests from `tests/test_cache.py`**

In `tests/test_cache.py`:
- Remove line 7: `from kv_quant.calibrate import _compute_bit_split`
- Remove lines 69-84 (both `test_compute_bit_split_budget` and `test_compute_bit_split_low_bits`):

```python
def test_compute_bit_split_budget():
    # For d=128, d_s=4, total_bits=4, boost=2.0:
    # bits_signal=8, bits_noise should make average ≈ 4
    bits_s, bits_n = _compute_bit_split(total_bits=4, d=128, d_s=4, signal_bit_boost=2.0)
    # Average: (4*bits_s + 124*bits_n) / 128 should be close to 4
    avg = (4 * bits_s + 124 * bits_n) / 128
    assert abs(avg - 4.0) < 1.0
    assert bits_s >= bits_n  # signal gets more bits
    assert 1 <= bits_n <= 8
    assert 1 <= bits_s <= 8


def test_compute_bit_split_low_bits():
    bits_s, bits_n = _compute_bit_split(total_bits=2, d=128, d_s=4, signal_bit_boost=2.0)
    assert bits_s >= bits_n
    assert bits_n >= 1
```

Remove the `from kv_quant.ops.qjl import make_sign_matrix` import (line 9) if it is no longer used anywhere else in `test_cache.py` after this edit. (Check: it's only used in `_make_synthetic_cal_data` which still exists at this point — leave it for now; Task 2 will remove it.)

- [ ] **Step 4: Run tests to verify calibrate tests pass and no regressions**

Run: `pytest tests/test_calibrate.py tests/test_cache.py -v`

Expected:
- `tests/test_calibrate.py`: 4 PASSED
- `tests/test_cache.py`: all remaining tests PASSED (turboquant tests + spectralquant tests still using old `_make_synthetic_cal_data` — these still pass because `spectralquant.py` hasn't changed yet)

If any test fails, debug and fix before committing.

- [ ] **Step 5: Commit**

```bash
git add kv_quant/calibrate.py tests/test_calibrate.py tests/test_cache.py
git commit -m "feat(spectralquant): rewrite calibrate with official NonUniformQuantizer + eigenspectral helpers"
```

---

### Task 2: Rewrite `kv_quant/spectralquant.py`, update `kv_quant/__init__.py`, update `tests/test_cache.py`

**Files:**
- Rewrite: `kv_quant/spectralquant.py`
- Modify: `kv_quant/__init__.py`
- Modify: `tests/test_cache.py`

**Interfaces:**
- Consumes from Task 1: `(EigenspectralCalibrator, quant_state_dict)` tuple as `cal_data`; quant_state_dict keyed by `"L{l}_H{h}_key"` and `"L{l}_H{h}_value"`
- `SpectralQuantCache(config: QuantConfig, cal_data: tuple)` — same constructor signature as before but `cal_data` is now a tuple, not a dict
- `update(key_states, value_states, layer_idx, cache_kwargs=None)` → `(k_full, v_full)` matching input dtype
- `get_seq_length(layer_idx=0)` → total accumulated sequence length for that layer
- `compressed_bytes()` → int, theoretical bit-packed size (bits used by indices at their declared bit-widths, not actual int32 tensor sizes)
- `_load_spectralquant_cal(base_path: str)` in `kv_quant/__init__.py` → `(EigenspectralCalibrator, quant_state_dict)`
- `wrap()` in `kv_quant/__init__.py`: passes new cal_data tuple to `SpectralQuantCache`

- [ ] **Step 1: Write failing tests for the new SpectralQuantCache**

Replace the `_make_synthetic_cal_data` function and the 4 SpectralQuantCache tests in `tests/test_cache.py`. Also remove the `from kv_quant.ops.qjl import make_sign_matrix` import (line 9) since it was only used in `_make_synthetic_cal_data`.

The new helper builds a real `EigenspectralCalibrator` (identity eigenvectors) and fits real `NonUniformQuantizer` instances. Uses `D=64` so the indices fit and accumulation works correctly.

Replace lines 9, 87-165 of `tests/test_cache.py` with:

```python
import os
import sys
_SPECTRALQUANT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
)
if _SPECTRALQUANT_SRC not in sys.path:
    sys.path.insert(0, _SPECTRALQUANT_SRC)


def _make_spectralquant_cal_data(
    n_layers: int = 2, n_kv_heads: int = 2, D: int = 64, avg_bits: int = 4
) -> tuple:
    """Synthetic (calibrator, quant_state) using identity eigenvectors — no model needed."""
    from spectralquant.calibration import EigenspectralCalibrator, HeadCalibrationData
    from spectralquant.nonuniform_quantization import NonUniformQuantizer

    calibrator = EigenspectralCalibrator()
    torch.manual_seed(0)
    quant_state: dict = {}

    for l in range(n_layers):
        for h in range(n_kv_heads):
            for kv_type in ("key", "value"):
                eigenvectors = torch.eye(D)
                eigenvalues = torch.ones(D)
                # With all eigenvalues=1, participation ratio = D, so d_eff_int = D-1.
                # We override d_eff via the fit() call to use D//2 for predictable allocation.
                d_eff_float = float(D // 2)

                calibrator._calibration_data[(l, h, kv_type)] = HeadCalibrationData(
                    layer_idx=l,
                    head_idx=h,
                    head_type=kv_type,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    d_eff=d_eff_float,
                    spectral_gap=None,
                    var_95=D // 2,
                    var_99=min(D * 3 // 4, D),
                    n_samples=200,
                    head_dim=D,
                )

                rotated = torch.randn(200, D)
                quant = NonUniformQuantizer(eigenvalues=eigenvalues, avg_bits=float(avg_bits))
                quant.fit(rotated, d_eff=d_eff_float)

                quant_state[f"L{l}_H{h}_{kv_type}"] = {
                    "semantic_centroids": quant._semantic_quantizer._centroids.clone(),
                    "tail_centroids": quant._tail_quantizer._centroids.clone(),
                    "d_eff_int": quant._d_eff_int,
                    "b_high": quant._b_high,
                    "b_low": quant._b_low,
                    "head_dim": D,
                }

    calibrator._is_calibrated = True
    return (calibrator, quant_state)


def test_spectralquant_update_returns_correct_shape():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=64)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape


def test_spectralquant_accumulates_sequence():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k1, v1 = _make_kv(seq=3, heads=2, d=64)
    k2, v2 = _make_kv(seq=1, heads=2, d=64)
    cache.update(k1, v1, layer_idx=0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    assert k_out.shape[-2] == 4
    assert cache.get_seq_length(0) == 4


def test_spectralquant_no_nan():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=64)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()


def test_spectralquant_compressed_smaller_than_fp16():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(batch=1, heads=2, seq=64, d=64)
    cache.update(k, v, layer_idx=0)
    fp16_k_bytes = k.nelement() * 2  # K only (1*2*64*64 * 2 bytes = 16384)
    # compressed_bytes() counts K+V at 4 bits/coord = 8192 bytes < 65536
    assert cache.compressed_bytes() < fp16_k_bytes * 4
```

Run: `pytest tests/test_cache.py::test_spectralquant_update_returns_correct_shape -v`
Expected: FAILED (old SpectralQuantCache doesn't accept a tuple)

- [ ] **Step 2: Rewrite `kv_quant/spectralquant.py`**

Replace the entire file with:

```python
from __future__ import annotations
import os
import sys
import torch
from transformers import DynamicCache

from kv_quant.config import QuantConfig

_SPECTRALQUANT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
)
if _SPECTRALQUANT_SRC not in sys.path:
    sys.path.insert(0, _SPECTRALQUANT_SRC)


def _restore_quant(state: dict):
    """Reconstruct a fitted NonUniformQuantizer from a saved quant_state entry."""
    from spectralquant.nonuniform_quantization import NonUniformQuantizer, LloydMaxQuantizer

    quant = NonUniformQuantizer(
        eigenvalues=torch.ones(state["head_dim"]),
        avg_bits=float(state["b_high"]),
    )
    quant._d_eff_int = state["d_eff_int"]
    quant._b_high = state["b_high"]
    quant._b_low = state["b_low"]
    quant._is_fitted = True

    sem_q = LloydMaxQuantizer(n_bits=state["b_high"])
    sem_q._centroids = state["semantic_centroids"].float()
    sem_q._is_fitted = True
    quant._semantic_quantizer = sem_q

    tail_q = LloydMaxQuantizer(n_bits=state["b_low"])
    tail_q._centroids = state["tail_centroids"].float()
    tail_q._is_fitted = True
    quant._tail_quantizer = tail_q

    return quant


class SpectralQuantCache(DynamicCache):
    """DynamicCache quantizing K/V with official NonUniformQuantizer + SpectralRotation.

    Per-head Lloyd-Max scalar quantization in the spectral basis:
      semantic regime (first d_eff coords): b_high bits
      tail regime (remaining coords): b_low bits

    cal_data: (EigenspectralCalibrator, quant_state_dict)
    """

    def __init__(self, config: QuantConfig, cal_data: tuple) -> None:
        super().__init__()
        self.config = config
        calibrator, quant_state = cal_data

        from spectralquant.spectral_rotation import SpectralRotation

        self._key_rot = SpectralRotation(calibrator, "key")
        self._val_rot = SpectralRotation(calibrator, "value")

        self._key_quants: dict = {}
        self._val_quants: dict = {}
        self._head_meta: dict = {}  # (l, h, kv_type) -> {d_eff_int, b_high, b_low, head_dim}

        for k, state in quant_state.items():
            # k format: "L{l}_H{h}_key" or "L{l}_H{h}_value"
            parts = k.split("_")  # ["L0", "H1", "key"] etc.
            l = int(parts[0][1:])
            h = int(parts[1][1:])
            kv_type = parts[2]
            meta = {
                "d_eff_int": state["d_eff_int"],
                "b_high": state["b_high"],
                "b_low": state["b_low"],
                "head_dim": state["head_dim"],
            }
            quant = _restore_quant(state)
            if kv_type == "key":
                self._key_quants[(l, h)] = quant
                self._head_meta[(l, h, "key")] = meta
            else:
                self._val_quants[(l, h)] = quant
                self._head_meta[(l, h, "value")] = meta

        # Per-layer per-head index storage: _sk_sem[l] = [Tensor or None] * n_heads
        self._sk_sem: list[list] = []
        self._sk_tail: list[list] = []
        self._sv_sem: list[list] = []
        self._sv_tail: list[list] = []

    def _ensure_layer(self, layer_idx: int, n_heads: int) -> None:
        while len(self._sk_sem) <= layer_idx:
            self._sk_sem.append([None] * n_heads)
            self._sk_tail.append([None] * n_heads)
            self._sv_sem.append([None] * n_heads)
            self._sv_tail.append([None] * n_heads)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress new K/V tokens and return full dequantized K, V for attention."""
        if layer_idx > len(self._sk_sem):
            raise IndexError(
                f"layer_idx {layer_idx} out of order; expected <= {len(self._sk_sem)}"
            )
        B, H, S, D = key_states.shape
        self._ensure_layer(layer_idx, H)

        from spectralquant.nonuniform_quantization import CompressedVector

        k_hat_heads = []
        v_hat_heads = []

        for h in range(H):
            k_h = key_states[:, h, :, :].float()   # (B, S, D)
            v_h = value_states[:, h, :, :].float()  # (B, S, D)

            k_rot = self._key_rot.rotate(k_h, layer_idx, h)  # (B, S, D)
            v_rot = self._val_rot.rotate(v_h, layer_idx, h)

            k_cv = self._key_quants[(layer_idx, h)].compress(k_rot)
            v_cv = self._val_quants[(layer_idx, h)].compress(v_rot)

            if self._sk_sem[layer_idx][h] is None:
                self._sk_sem[layer_idx][h] = k_cv.semantic_indices
                self._sk_tail[layer_idx][h] = k_cv.tail_indices
                self._sv_sem[layer_idx][h] = v_cv.semantic_indices
                self._sv_tail[layer_idx][h] = v_cv.tail_indices
            else:
                self._sk_sem[layer_idx][h] = torch.cat(
                    [self._sk_sem[layer_idx][h], k_cv.semantic_indices], dim=1
                )
                self._sk_tail[layer_idx][h] = torch.cat(
                    [self._sk_tail[layer_idx][h], k_cv.tail_indices], dim=1
                )
                self._sv_sem[layer_idx][h] = torch.cat(
                    [self._sv_sem[layer_idx][h], v_cv.semantic_indices], dim=1
                )
                self._sv_tail[layer_idx][h] = torch.cat(
                    [self._sv_tail[layer_idx][h], v_cv.tail_indices], dim=1
                )

            k_meta = self._head_meta[(layer_idx, h, "key")]
            S_full = self._sk_sem[layer_idx][h].shape[1]
            k_full_cv = CompressedVector(
                semantic_indices=self._sk_sem[layer_idx][h],
                tail_indices=self._sk_tail[layer_idx][h],
                d_eff=k_meta["d_eff_int"],
                head_dim=k_meta["head_dim"],
                b_high=k_meta["b_high"],
                b_low=k_meta["b_low"],
                original_shape=(B, S_full, D),
            )
            k_hat = self._key_rot.unrotate(
                self._key_quants[(layer_idx, h)].decompress(k_full_cv), layer_idx, h
            )  # (B, S_full, D)

            v_meta = self._head_meta[(layer_idx, h, "value")]
            v_full_cv = CompressedVector(
                semantic_indices=self._sv_sem[layer_idx][h],
                tail_indices=self._sv_tail[layer_idx][h],
                d_eff=v_meta["d_eff_int"],
                head_dim=v_meta["head_dim"],
                b_high=v_meta["b_high"],
                b_low=v_meta["b_low"],
                original_shape=(B, S_full, D),
            )
            v_hat = self._val_rot.unrotate(
                self._val_quants[(layer_idx, h)].decompress(v_full_cv), layer_idx, h
            )

            k_hat_heads.append(k_hat)
            v_hat_heads.append(v_hat)

        k_full = torch.stack(k_hat_heads, dim=1).to(key_states.dtype)
        v_full = torch.stack(v_hat_heads, dim=1).to(value_states.dtype)
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._sk_sem):
            return 0
        for slot in self._sk_sem[layer_idx]:
            if slot is not None:
                return slot.shape[1]
        return 0

    def compressed_bytes(self) -> int:
        """Theoretical compressed bytes using declared bit widths (not int32 tensor sizes)."""
        total_bits = 0
        for l_idx in range(len(self._sk_sem)):
            for h in range(len(self._sk_sem[l_idx])):
                sem = self._sk_sem[l_idx][h]
                if sem is None:
                    continue
                meta = self._head_meta[(l_idx, h, "key")]
                n_vecs = sem.shape[0] * sem.shape[1]  # B * S
                d_eff = meta["d_eff_int"]
                D = meta["head_dim"]
                b_high = meta["b_high"]
                b_low = meta["b_low"]
                # Key + Value: same bit structure
                total_bits += 2 * n_vecs * (d_eff * b_high + (D - d_eff) * b_low)
        return (total_bits + 7) // 8
```

- [ ] **Step 3: Update `kv_quant/__init__.py`**

Replace the entire file with:

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


def _make_cache(config: QuantConfig, n_kv_heads: int, head_dim: int, cal_data, device):
    if config.method == "turboquant":
        from kv_quant.turboquant import TurboQuantCache
        return TurboQuantCache(config, n_kv_heads, head_dim, device=device)
    if config.method == "spectralquant":
        from kv_quant.spectralquant import SpectralQuantCache
        return SpectralQuantCache(config, cal_data)
    raise ValueError(f"Unknown method: {config.method!r}")


def wrap(model, config: QuantConfig):
    """Patch model.generate() to use a quantized KV cache.

    For spectralquant, config.calibration_path must be a base path (no extension)
    pointing to files produced by `python -m kv_quant.calibrate`.
    """
    if config.method == "spectralquant":
        if not config.calibration_path:
            raise ValueError("spectralquant requires config.calibration_path")
        cal_data = _load_spectralquant_cal(config.calibration_path)
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

- [ ] **Step 4: Run the SpectralQuantCache tests to verify they pass**

Run: `pytest tests/test_cache.py -v`

Expected: all tests pass:
- 5 turboquant tests: PASSED
- 4 spectralquant tests: PASSED
- 3 wrap() tests: PASSED (turboquant ones)
- `test_wrap_spectralquant_raises_without_calibration`: PASSED

If any fail, debug. Common issues:
- `KeyError (l, h, "key")` in `__init__`: check key parsing in `__init__` — parts[2] should be `"key"` or `"value"`.
- Shape mismatch in `update()`: `semantic_indices` from `compress()` has shape `(B, S, d_eff_int)` when input `k_rot` has shape `(B, S, D)`. Verify cat is along `dim=1`.
- `_head_meta` lookup in `compressed_bytes()`: key is `(l_idx, h, "key")` — must match what was stored in `__init__`.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`

Expected: all tests pass. If `test_wrap_injects_turboquant_cache` or others fail, debug.

- [ ] **Step 6: Commit**

```bash
git add kv_quant/spectralquant.py kv_quant/__init__.py tests/test_cache.py
git commit -m "feat(spectralquant): replace VQ+QJL cache with official NonUniformQuantizer + SpectralRotation"
```
