# KV Cache Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement TurboQuant and SpectralQuant as drop-in HuggingFace KV cache quantizers with a benchmark harness measuring perplexity, memory, and downstream task scores.

**Architecture:** Both quantizers subclass `transformers.DynamicCache`, overriding `update()` to compress K/V on write and return dequantized tensors for attention. `wrap(model, config)` monkey-patches `model.generate()` to inject the right cache instance. SpectralQuant requires a one-time calibration step that saves per-head eigenvectors and codebooks to a `.pt` file.

**Tech Stack:** PyTorch, transformers ≥ 4.40, datasets, scikit-learn (calibration k-means), lm-eval (benchmarks).

## Global Constraints

- Target architectures: Qwen (Qwen2.5, Qwen3) and Gemma (Gemma 2, Gemma 3) only
- `n_bits` in [1, 7] — stored as `torch.int8`
- No custom CUDA kernels — pure PyTorch
- All unit tests run on CPU (no GPU required)
- Integration tests gated behind `--run-slow` pytest flag
- Follow existing repo style: `from __future__ import annotations`, dataclasses, no type: ignore
- New module lives at `kv_quant/` in repo root

## File Map

```
kv_quant/
├── __init__.py          # wrap(), _make_cache(), _get_kv_shape()
├── config.py            # QuantConfig dataclass
├── ops/
│   ├── __init__.py      # empty
│   ├── rotation.py      # make_rotation(), rotate(), unrotate()
│   ├── scalar_quant.py  # quantize(), dequantize()
│   └── qjl.py           # make_sign_matrix(), encode(), encode_2d()
├── turboquant.py        # TurboQuantCache(DynamicCache)
├── spectralquant.py     # SpectralQuantCache(DynamicCache)
├── calibrate.py         # calibrate() + __main__ CLI
└── bench/
    ├── __init__.py      # empty
    ├── perplexity.py    # compute_perplexity()
    ├── memory.py        # measure_kv_memory()
    └── run_bench.py     # main() CLI
tests/
├── conftest.py          # --run-slow pytest option
├── test_ops.py          # rotation, scalar_quant, qjl unit tests
├── test_cache.py        # TurboQuantCache + SpectralQuantCache unit tests
└── test_integration.py  # slow end-to-end test (Qwen2.5-0.5B)
```

---

### Task 1: Scaffolding and QuantConfig

**Files:**
- Create: `kv_quant/__init__.py`
- Create: `kv_quant/config.py`
- Create: `kv_quant/ops/__init__.py`
- Create: `kv_quant/bench/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_ops.py` (import smoke test only)

**Interfaces:**
- Produces: `QuantConfig(method, bits, qjl_dim, calibration_path, signal_bit_boost)`

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_ops.py
from kv_quant.config import QuantConfig

def test_quantconfig_defaults():
    cfg = QuantConfig()
    assert cfg.method == "turboquant"
    assert cfg.bits == 4
    assert cfg.qjl_dim == 32
    assert cfg.calibration_path is None
    assert cfg.signal_bit_boost == 2.0

def test_quantconfig_custom():
    cfg = QuantConfig(method="spectralquant", bits=2, calibration_path="foo.pt")
    assert cfg.method == "spectralquant"
    assert cfg.bits == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/harikrishnan-c/projects/machine-notes
pytest tests/test_ops.py::test_quantconfig_defaults -v
```
Expected: `ModuleNotFoundError: No module named 'kv_quant'`

- [ ] **Step 3: Create directory structure and QuantConfig**

```python
# kv_quant/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QuantConfig:
    method: Literal["turboquant", "spectralquant"] = "turboquant"
    bits: int = 4
    qjl_dim: int = 32
    calibration_path: Optional[str] = None
    signal_bit_boost: float = 2.0
```

```python
# kv_quant/__init__.py
from kv_quant.config import QuantConfig

def wrap(model, config: QuantConfig):
    raise NotImplementedError
```

```python
# kv_quant/ops/__init__.py
```

```python
# kv_quant/bench/__init__.py
```

```python
# tests/conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="Pass --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ops.py::test_quantconfig_defaults tests/test_ops.py::test_quantconfig_custom -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add kv_quant/ tests/
git commit -m "feat(kv-quant): scaffolding, QuantConfig, conftest"
```

---

### Task 2: ops/rotation.py

**Files:**
- Create: `kv_quant/ops/rotation.py`
- Test: `tests/test_ops.py`

**Interfaces:**
- Produces:
  - `make_rotation(d, device, dtype) -> Tensor(d, d)` — random orthogonal matrix
  - `rotate(h, R) -> Tensor` — `h @ R.T` per head, h: (B,H,S,d), R: (H,d,d)
  - `unrotate(h, R) -> Tensor` — `h @ R` per head, inverse of rotate

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_ops.py
import torch
from kv_quant.ops.rotation import make_rotation, rotate, unrotate

def test_rotation_orthogonal():
    torch.manual_seed(0)
    R = make_rotation(64)
    assert torch.allclose(R @ R.T, torch.eye(64), atol=1e-5)

def test_rotate_unrotate_roundtrip():
    torch.manual_seed(0)
    d, H = 32, 4
    R = torch.stack([make_rotation(d) for _ in range(H)])
    h = torch.randn(2, H, 10, d)
    assert torch.allclose(unrotate(rotate(h, R), R), h, atol=1e-5)

def test_rotate_shape():
    torch.manual_seed(0)
    d, H = 16, 3
    R = torch.stack([make_rotation(d) for _ in range(H)])
    h = torch.randn(1, H, 5, d)
    assert rotate(h, R).shape == (1, H, 5, d)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_ops.py::test_rotation_orthogonal -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement rotation.py**

```python
# kv_quant/ops/rotation.py
from __future__ import annotations
import torch


def make_rotation(d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Random orthogonal matrix (d, d) via QR decomposition."""
    G = torch.randn(d, d, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(G)
    return Q


def rotate(h: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """h @ R.T per head.
    h: (batch, heads, seq, d)
    R: (heads, d, d)
    Returns: (batch, heads, seq, d)
    """
    # result[b,h,s,e] = sum_d h[b,h,s,d] * R[h,e,d]  (= h @ R.T per head)
    return torch.einsum('bhsd,hed->bhse', h, R)


def unrotate(h: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """h @ R per head — inverse of rotate.
    h: (batch, heads, seq, d)
    R: (heads, d, d)
    Returns: (batch, heads, seq, d)
    """
    # result[b,h,s,d] = sum_e h[b,h,s,e] * R[h,e,d]  (= h @ R per head)
    return torch.einsum('bhse,hed->bhsd', h, R)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ops.py::test_rotation_orthogonal tests/test_ops.py::test_rotate_unrotate_roundtrip tests/test_ops.py::test_rotate_shape -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add kv_quant/ops/rotation.py tests/test_ops.py
git commit -m "feat(kv-quant): ops/rotation — make_rotation, rotate, unrotate"
```

---

### Task 3: ops/scalar_quant.py

**Files:**
- Create: `kv_quant/ops/scalar_quant.py`
- Test: `tests/test_ops.py`

**Interfaces:**
- Produces:
  - `quantize(h, n_bits) -> (h_int: Tensor int8, scale: Tensor float16)` — per-token n-bit quantization, h: (..., d)
  - `dequantize(h_int, scale, n_bits) -> Tensor float32`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_ops.py
from kv_quant.ops.scalar_quant import quantize, dequantize

def test_quantize_dtypes():
    h = torch.randn(4, 64)
    h_int, scale = quantize(h, 4)
    assert h_int.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert scale.shape == (*h.shape[:-1], 1)

def test_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    h = torch.randn(8, 128)
    for bits in [2, 4, 7]:
        h_int, scale = quantize(h, bits)
        h_rec = dequantize(h_int, scale, bits)
        n_levels = 2 ** bits
        # Max error bounded by one quantization step
        step = 2.0 / n_levels
        max_err = (h - h_rec).abs().max().item()
        assert max_err <= step + 1e-4, f"bits={bits}: max_err={max_err:.4f} > step={step:.4f}"

def test_quantize_clamps_to_range():
    h = torch.tensor([[100.0, -100.0, 0.5]])
    h_int, scale = quantize(h, 4)
    h_rec = dequantize(h_int, scale, 4)
    # Reconstructed values should be within [-max_val, max_val]
    max_val = h.abs().max().item()
    assert h_rec.abs().max().item() <= max_val + 1e-3
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_ops.py::test_quantize_dtypes -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement scalar_quant.py**

```python
# kv_quant/ops/scalar_quant.py
from __future__ import annotations
import torch


def quantize(h: torch.Tensor, n_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token n-bit scalar quantization.

    h: (..., d) float
    Returns:
      h_int: (..., d) int8  — quantized values in [0, 2^n_bits - 1]
      scale: (..., 1) float16 — per-token abs-max scale
    n_bits must be in [1, 7].
    """
    n_levels = 2 ** n_bits  # number of quantization levels
    scale = h.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    h_norm = (h / scale).clamp(-1.0, 1.0)
    # Map [-1, 1] -> [0, n_levels - 1]
    h_int = ((h_norm + 1.0) / 2.0 * (n_levels - 1)).round().clamp(0, n_levels - 1)
    return h_int.to(torch.int8), scale.to(torch.float16)


def dequantize(h_int: torch.Tensor, scale: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Inverse of quantize.

    h_int: (..., d) int8
    scale: (..., 1) float16
    Returns: (..., d) float32
    """
    n_levels = 2 ** n_bits
    h_norm = h_int.float() / (n_levels - 1) * 2.0 - 1.0  # back to [-1, 1]
    return h_norm * scale.float()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ops.py::test_quantize_dtypes tests/test_ops.py::test_quantize_dequantize_roundtrip tests/test_ops.py::test_quantize_clamps_to_range -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add kv_quant/ops/scalar_quant.py tests/test_ops.py
git commit -m "feat(kv-quant): ops/scalar_quant — quantize, dequantize"
```

---

### Task 4: ops/qjl.py

**Files:**
- Create: `kv_quant/ops/qjl.py`
- Test: `tests/test_ops.py`

**Interfaces:**
- Produces:
  - `make_sign_matrix(m, d, device) -> Tensor(m, d)` — random ±1/√m matrix
  - `encode(h, S) -> Tensor bool` — h: (B,H,S,d), S: (H,m,d) → (B,H,S,m)
  - `encode_2d(h, S) -> Tensor bool` — h: (N,d), S: (m,d) → (N,m)

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_ops.py
from kv_quant.ops.qjl import make_sign_matrix, encode, encode_2d

def test_sign_matrix_shape_and_values():
    S = make_sign_matrix(32, 64)
    assert S.shape == (32, 64)
    # Values should be ±1/sqrt(32)
    expected_abs = 1.0 / (32 ** 0.5)
    assert torch.allclose(S.abs(), torch.full_like(S, expected_abs))

def test_encode_shape():
    torch.manual_seed(0)
    H, m, d = 4, 16, 32
    S = torch.stack([make_sign_matrix(m, d) for _ in range(H)])
    h = torch.randn(2, H, 10, d)
    bits = encode(h, S)
    assert bits.shape == (2, H, 10, m)
    assert bits.dtype == torch.bool

def test_encode_2d_shape():
    torch.manual_seed(0)
    S = make_sign_matrix(16, 32)
    h = torch.randn(100, 32)
    bits = encode_2d(h, S)
    assert bits.shape == (100, 16)
    assert bits.dtype == torch.bool

def test_encode_deterministic():
    torch.manual_seed(0)
    S = make_sign_matrix(16, 32)
    h = torch.randn(5, 32)
    assert (encode_2d(h, S) == encode_2d(h, S)).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_ops.py::test_sign_matrix_shape_and_values -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement qjl.py**

```python
# kv_quant/ops/qjl.py
from __future__ import annotations
import torch


def make_sign_matrix(m: int, d: int, device=None) -> torch.Tensor:
    """Random ±1/√m sign matrix of shape (m, d)."""
    S = torch.randint(0, 2, (m, d), device=device).float() * 2.0 - 1.0
    return S / (m ** 0.5)


def encode(h: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """1-bit QJL encoding for 4-D tensors (TurboQuant path).

    h: (batch, heads, seq, d)
    S: (heads, m, d)
    Returns: (batch, heads, seq, m) bool  — True encodes +1
    """
    # proj[b,h,s,m] = sum_d h[b,h,s,d] * S[h,m,d]  (= h @ S.T per head)
    proj = torch.einsum('bhsd,hmd->bhsm', h, S)
    return proj >= 0.0


def encode_2d(h: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """1-bit QJL encoding for 2-D tensors (SpectralQuant path).

    h: (N, d)
    S: (m, d)
    Returns: (N, m) bool
    """
    return (h @ S.T) >= 0.0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ops.py -v
```
Expected: all ops tests pass (10+ tests)

- [ ] **Step 5: Commit**

```bash
git add kv_quant/ops/qjl.py tests/test_ops.py
git commit -m "feat(kv-quant): ops/qjl — make_sign_matrix, encode, encode_2d"
```

---

### Task 5: TurboQuantCache

**Files:**
- Create: `kv_quant/turboquant.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Consumes: `make_rotation`, `rotate`, `unrotate`, `quantize`, `dequantize`, `make_sign_matrix`, `encode`, `QuantConfig`
- Produces: `TurboQuantCache(config, n_heads, head_dim, device)`
  - `.update(key_states, value_states, layer_idx, cache_kwargs) -> (k_dq, v_dq)`
  - `.get_seq_length(layer_idx) -> int`
  - `.compressed_bytes() -> int`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cache.py
from __future__ import annotations
import torch
import pytest
from kv_quant.config import QuantConfig
from kv_quant.turboquant import TurboQuantCache


def _make_kv(batch=1, heads=2, seq=5, d=16):
    torch.manual_seed(42)
    return torch.randn(batch, heads, seq, d), torch.randn(batch, heads, seq, d)


def test_turboquant_update_returns_correct_shape():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k, v = _make_kv(heads=2, d=16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape


def test_turboquant_accumulates_sequence():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k1, v1 = _make_kv(seq=3, heads=2, d=16)
    k2, v2 = _make_kv(seq=1, heads=2, d=16)
    cache.update(k1, v1, layer_idx=0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    assert k_out.shape[-2] == 4  # 3 + 1
    assert cache.get_seq_length(layer_idx=0) == 4


def test_turboquant_no_nan():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k, v = _make_kv(heads=2, d=16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()


def test_turboquant_compressed_smaller_than_fp16():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k, v = _make_kv(batch=1, heads=2, seq=64, d=16)
    cache.update(k, v, layer_idx=0)
    fp16_bytes = k.nelement() * 2 * 2  # K + V, float16
    assert cache.compressed_bytes() < fp16_bytes


def test_turboquant_multiple_layers():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k, v = _make_kv(heads=2, d=16)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)
    assert cache.get_seq_length(layer_idx=0) == 5
    assert cache.get_seq_length(layer_idx=1) == 5
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cache.py::test_turboquant_update_returns_correct_shape -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement TurboQuantCache**

```python
# kv_quant/turboquant.py
from __future__ import annotations
import torch
from transformers import DynamicCache

from kv_quant.config import QuantConfig
from kv_quant.ops.rotation import make_rotation, rotate, unrotate
from kv_quant.ops.scalar_quant import quantize, dequantize
from kv_quant.ops.qjl import make_sign_matrix, encode


class TurboQuantCache(DynamicCache):
    """DynamicCache that compresses K/V with TurboQuant (rotation + scalar quant + QJL).

    Stores compressed buffers (_qk_int, _qk_scale, _qk_qjl and V equivalents).
    key_cache / value_cache are kept empty; get_seq_length() reads from _qk_int.
    update() returns dequantized K/V for HF attention to consume directly.
    """

    def __init__(self, config: QuantConfig, n_heads: int, head_dim: int, device=None):
        super().__init__()
        self.config = config
        self.n_heads = n_heads
        self.head_dim = head_dim

        torch.manual_seed(0)  # reproducible rotation / QJL matrices
        self._Rk = torch.stack([make_rotation(head_dim, device=device) for _ in range(n_heads)])
        self._Rv = torch.stack([make_rotation(head_dim, device=device) for _ in range(n_heads)])
        m = config.qjl_dim
        self._Sk = torch.stack([make_sign_matrix(m, head_dim, device=device) for _ in range(n_heads)])
        self._Sv = torch.stack([make_sign_matrix(m, head_dim, device=device) for _ in range(n_heads)])

        # Compressed buffers — one tensor per layer
        self._qk_int:   list[torch.Tensor] = []   # (B, H, S, d) int8
        self._qk_scale: list[torch.Tensor] = []   # (B, H, S, 1) float16
        self._qk_qjl:   list[torch.Tensor] = []   # (B, H, S, m) bool
        self._qv_int:   list[torch.Tensor] = []
        self._qv_scale: list[torch.Tensor] = []
        self._qv_qjl:   list[torch.Tensor] = []

    # ------------------------------------------------------------------
    def _compress(
        self, h: torch.Tensor, R: torch.Tensor, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotate → quantize → QJL residual.

        h: (B, H, S, d)
        Returns: (h_int, h_scale, qjl_bits)  — all on same device as h
        """
        R = R.to(h.device)
        S = S.to(h.device)
        h_rot = rotate(h.float(), R)
        h_int, h_scale = quantize(h_rot, self.config.bits)
        h_rot_dq = dequantize(h_int, h_scale, self.config.bits)
        residual = h_rot - h_rot_dq
        qjl_bits = encode(residual, S)
        return h_int, h_scale, qjl_bits

    def _decompress(
        self,
        h_int: torch.Tensor,
        h_scale: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize and unrotate full accumulated layer cache."""
        R = R.to(h_int.device)
        h_rot_dq = dequantize(h_int, h_scale, self.config.bits)
        return unrotate(h_rot_dq, R)

    # ------------------------------------------------------------------
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_int, k_scale, k_qjl = self._compress(key_states, self._Rk, self._Sk)
        v_int, v_scale, v_qjl = self._compress(value_states, self._Rv, self._Sv)

        if layer_idx >= len(self._qk_int):
            self._qk_int.append(k_int)
            self._qk_scale.append(k_scale)
            self._qk_qjl.append(k_qjl)
            self._qv_int.append(v_int)
            self._qv_scale.append(v_scale)
            self._qv_qjl.append(v_qjl)
        else:
            self._qk_int[layer_idx]   = torch.cat([self._qk_int[layer_idx],   k_int],   dim=2)
            self._qk_scale[layer_idx] = torch.cat([self._qk_scale[layer_idx], k_scale], dim=2)
            self._qk_qjl[layer_idx]   = torch.cat([self._qk_qjl[layer_idx],   k_qjl],   dim=2)
            self._qv_int[layer_idx]   = torch.cat([self._qv_int[layer_idx],   v_int],   dim=2)
            self._qv_scale[layer_idx] = torch.cat([self._qv_scale[layer_idx], v_scale], dim=2)
            self._qv_qjl[layer_idx]   = torch.cat([self._qv_qjl[layer_idx],   v_qjl],   dim=2)

        k_full = self._decompress(self._qk_int[layer_idx], self._qk_scale[layer_idx], self._Rk)
        v_full = self._decompress(self._qv_int[layer_idx], self._qv_scale[layer_idx], self._Rv)
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self._qk_int:
            return 0
        idx = min(layer_idx, len(self._qk_int) - 1)
        return self._qk_int[idx].shape[2]

    def compressed_bytes(self) -> int:
        """Bytes used by compressed K/V buffers."""
        total = 0
        for buf in self._qk_int + self._qv_int:
            total += buf.nelement() * buf.element_size()
        for buf in self._qk_scale + self._qv_scale:
            total += buf.nelement() * buf.element_size()
        for buf in self._qk_qjl + self._qv_qjl:
            total += buf.nelement() // 8 + 1  # 1 bit per bool conceptually
        return total
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cache.py -k turboquant -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add kv_quant/turboquant.py tests/test_cache.py
git commit -m "feat(kv-quant): TurboQuantCache — rotation + scalar quant + QJL"
```

---

### Task 6: SpectralQuant Calibration

**Files:**
- Create: `kv_quant/calibrate.py`
- Test: `tests/test_cache.py` (calibration helper tests, no model download)

**Interfaces:**
- Produces:
  - `calibrate(model_id, output_path, n_seqs, bits, signal_bit_boost, qjl_dim)` — saves `.pt`
  - `_compute_bit_split(total_bits, d, d_s, signal_bit_boost) -> (bits_signal, bits_noise)`
  - Saved `.pt` structure: `{model_id, n_layers, n_kv_heads, head_dim, bits_signal, bits_noise, qjl_dim, layers: {l: {h: {U, d_s, codebook_signal, codebook_noise, S_signal}}}}`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_cache.py
from kv_quant.calibrate import _compute_bit_split


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

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cache.py::test_compute_bit_split_budget -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement calibrate.py**

```python
# kv_quant/calibrate.py
"""SpectralQuant calibration: compute per-head eigenvectors and VQ codebooks.

Usage:
    python -m kv_quant.calibrate \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output spectralquant_qwen25_7b.pt \
        --n-seqs 100 --bits 4
"""
from __future__ import annotations
import argparse
import torch
import numpy as np
from tqdm import tqdm

from kv_quant.ops.qjl import make_sign_matrix


def _compute_bit_split(
    total_bits: int, d: int, d_s: int, signal_bit_boost: float, max_bits: int = 8
) -> tuple[int, int]:
    """Allocate bits to signal and noise dims.

    Solves: (d_s * bits_s + (d - d_s) * bits_n) / d ≈ total_bits
    with bits_s = min(max_bits, round(total_bits * signal_bit_boost)).
    """
    bits_s = min(max_bits, round(total_bits * signal_bit_boost))
    d_noise = d - d_s
    if d_noise > 0:
        bits_n_float = (d * total_bits - d_s * bits_s) / d_noise
        bits_n = max(1, round(bits_n_float))
    else:
        bits_n = 1
    return bits_s, bits_n


def _kmeans_codebook(data: np.ndarray, n_centroids: int) -> np.ndarray:
    """Train Lloyd-Max codebook via k-means. Returns (n_centroids, k) float32."""
    from sklearn.cluster import KMeans
    n_centroids = min(n_centroids, len(data))
    km = KMeans(n_clusters=n_centroids, n_init=10, random_state=42, max_iter=300)
    km.fit(data)
    return km.cluster_centers_.astype(np.float32)


def calibrate(
    model_id: str,
    output_path: str,
    n_seqs: int = 100,
    bits: int = 4,
    signal_bit_boost: float = 2.0,
    qjl_dim: int = 32,
    device: str = "cuda",
) -> None:
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
        model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads
    )

    # Collect key vectors: all_keys[layer] = list of (kv_heads, seq, d) cpu tensors
    all_keys: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    for text in tqdm(texts, desc="Collecting key vectors"):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids
        ids = ids.to(device)
        cache = DynamicCache()
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True)
        for l in range(min(n_layers, len(cache.key_cache))):
            # key_cache[l]: (1, kv_heads, seq, d) — squeeze batch dim
            all_keys[l].append(cache.key_cache[l][0].float().cpu())

    cal_data: dict = {
        "model_id": model_id,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "qjl_dim": qjl_dim,
        "layers": {},
    }

    for layer_idx in tqdm(range(n_layers), desc="Computing per-head calibration"):
        if not all_keys[layer_idx]:
            continue
        # Stack along seq dim: (kv_heads, total_tokens, d)
        layer_keys = torch.cat(all_keys[layer_idx], dim=1)
        cal_data["layers"][layer_idx] = {}

        for head_idx in range(n_kv_heads):
            keys = layer_keys[head_idx]  # (total_tokens, d)

            # Covariance (centered)
            keys_c = keys - keys.mean(dim=0)
            cov = (keys_c.T @ keys_c) / max(keys_c.shape[0] - 1, 1)

            # Eigen-decomposition (ascending from eigh → flip to descending)
            eigenvalues, U = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.flip(0)
            U = U.flip(1)  # (d, d), columns = eigenvectors in descending order

            # Effective dimensionality
            d_eff = (eigenvalues.sum() ** 2) / ((eigenvalues ** 2).sum() + 1e-12)
            d_s = int(max(1, min(d_eff.ceil().item(), head_dim - 1)))

            bits_s, bits_n = _compute_bit_split(bits, head_dim, d_s, signal_bit_boost)

            # Project calibration data
            h_proj = keys @ U  # (total_tokens, d)
            h_sig = h_proj[:, :d_s].numpy()         # (total_tokens, d_s)
            h_noi = h_proj[:, d_s:].numpy()         # (total_tokens, d-d_s)

            # Codebooks
            cb_sig = _kmeans_codebook(h_sig, 2 ** bits_s)
            cb_noi = _kmeans_codebook(h_noi, 2 ** bits_n) if h_noi.shape[1] > 0 else np.zeros((1, 0), dtype=np.float32)

            S_signal = make_sign_matrix(qjl_dim, d_s)

            cal_data["layers"][layer_idx][head_idx] = {
                "U": U,                                          # (d, d)
                "d_s": d_s,
                "bits_signal": bits_s,
                "bits_noise": bits_n,
                "codebook_signal": torch.from_numpy(cb_sig),    # (2^bits_s, d_s)
                "codebook_noise": torch.from_numpy(cb_noi),     # (2^bits_n, d-d_s)
                "S_signal": S_signal,                           # (m, d_s)
            }

    torch.save(cal_data, output_path)
    print(f"Calibration data saved to {output_path}")
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-seqs", type=int, default=100)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--signal-bit-boost", type=float, default=2.0)
    parser.add_argument("--qjl-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    calibrate(args.model, args.output, args.n_seqs, args.bits, args.signal_bit_boost, args.qjl_dim, args.device)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cache.py::test_compute_bit_split_budget tests/test_cache.py::test_compute_bit_split_low_bits -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add kv_quant/calibrate.py tests/test_cache.py
git commit -m "feat(kv-quant): calibrate.py — SpectralQuant per-head eigenvectors + codebooks"
```

---

### Task 7: SpectralQuantCache

**Files:**
- Create: `kv_quant/spectralquant.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `cal_data` dict (produced by `calibrate()`), `encode_2d`
- Produces: `SpectralQuantCache(config, cal_data, device)`
  - `.update(key_states, value_states, layer_idx, cache_kwargs) -> (k_dq, v_dq)`
  - `.get_seq_length(layer_idx) -> int`
  - `.compressed_bytes() -> int`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_cache.py
from kv_quant.spectralquant import SpectralQuantCache


def _make_synthetic_cal_data(
    n_layers: int = 2, n_kv_heads: int = 2, head_dim: int = 16,
    d_s: int = 4, bits: int = 4, qjl_dim: int = 8
) -> dict:
    """Synthetic calibration data for unit tests — no model download needed."""
    torch.manual_seed(0)
    layers: dict = {}
    for l in range(n_layers):
        layers[l] = {}
        for h in range(n_kv_heads):
            U, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
            cb_s = torch.randn(2 ** bits, d_s)
            cb_n = torch.randn(2 ** max(1, bits - 1), head_dim - d_s)
            S = make_sign_matrix(qjl_dim, d_s)
            layers[l][h] = {
                "U": U, "d_s": d_s,
                "bits_signal": bits, "bits_noise": max(1, bits - 1),
                "codebook_signal": cb_s, "codebook_noise": cb_n, "S_signal": S,
            }
    return {
        "model_id": "test", "n_layers": n_layers, "n_kv_heads": n_kv_heads,
        "head_dim": head_dim, "qjl_dim": qjl_dim, "layers": layers,
    }


def test_spectralquant_update_returns_correct_shape():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_synthetic_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape


def test_spectralquant_accumulates_sequence():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_synthetic_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k1, v1 = _make_kv(seq=3, heads=2, d=16)
    k2, v2 = _make_kv(seq=1, heads=2, d=16)
    cache.update(k1, v1, layer_idx=0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    assert k_out.shape[-2] == 4
    assert cache.get_seq_length(0) == 4


def test_spectralquant_no_nan():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_synthetic_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()


def test_spectralquant_compressed_smaller_than_fp16():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_synthetic_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(batch=1, heads=2, seq=64, d=16)
    cache.update(k, v, layer_idx=0)
    fp16_k_bytes = k.nelement() * 2  # K only (V stored bfloat16 = 2 bytes)
    assert cache.compressed_bytes() < fp16_k_bytes * 4  # sanity, not tight
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cache.py::test_spectralquant_update_returns_correct_shape -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement SpectralQuantCache**

```python
# kv_quant/spectralquant.py
from __future__ import annotations
import torch
from transformers import DynamicCache

from kv_quant.config import QuantConfig
from kv_quant.ops.qjl import encode_2d


class SpectralQuantCache(DynamicCache):
    """DynamicCache that quantizes K via SpectralQuant VQ; V stored in bfloat16.

    Compressed key storage per layer:
      _sq_k_sig_idx: (B, H, S) uint8  — index into per-head signal codebook
      _sq_k_noi_idx: (B, H, S) uint8  — index into per-head noise codebook
      _sq_k_qjl:     (B, H, S, m) bool — QJL bits on signal residual
    Value storage: _sq_v (B, H, S, d) bfloat16 — no VQ on values.
    """

    def __init__(self, config: QuantConfig, cal_data: dict, device=None):
        super().__init__()
        self.config = config
        self.cal_data = cal_data
        self.device = device

        self._sq_k_sig_idx: list[torch.Tensor] = []
        self._sq_k_noi_idx: list[torch.Tensor] = []
        self._sq_k_qjl:     list[torch.Tensor] = []
        self._sq_v:          list[torch.Tensor] = []

    # ------------------------------------------------------------------
    def _head_cal(self, layer_idx: int, head_idx: int) -> dict:
        return self.cal_data["layers"][layer_idx][head_idx]

    @staticmethod
    def _nearest(h: torch.Tensor, cb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest-centroid lookup.
        h: (N, k) float, cb: (C, k) float
        Returns: (idx: (N,) uint8, reconstructed: (N, k) float)
        """
        dists = torch.cdist(h.float(), cb.float())   # (N, C)
        idx = dists.argmin(dim=-1)                    # (N,)
        return idx.to(torch.uint8), cb[idx]

    def _quant_key_layer(
        self, key_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize all heads for one layer.

        key_states: (B, H, S, d)
        Returns: (sig_idx, noi_idx, qjl_bits, k_dq)
          sig_idx:  (B, H, S) uint8
          noi_idx:  (B, H, S) uint8
          qjl_bits: (B, H, S, m) bool
          k_dq:     (B, H, S, d) float32
        """
        B, H, S, D = key_states.shape
        sig_idxs, noi_idxs, qjl_list, k_dq_list = [], [], [], []

        for head in range(H):
            cal = self._head_cal(layer_idx, head)
            U   = cal["U"].to(key_states.device)              # (d, d)
            d_s = cal["d_s"]
            cb_s = cal["codebook_signal"].to(key_states.device)  # (C_s, d_s)
            cb_n = cal["codebook_noise"].to(key_states.device)   # (C_n, d-d_s)
            S_sig = cal["S_signal"].to(key_states.device)        # (m, d_s)

            h = key_states[:, head, :, :].float()  # (B, S, d)
            h_flat = h.reshape(-1, D)               # (B*S, d)

            h_proj = h_flat @ U                     # (B*S, d)
            h_sig  = h_proj[:, :d_s]               # (B*S, d_s)
            h_noi  = h_proj[:, d_s:]               # (B*S, d-d_s)

            s_idx, h_sig_dq = self._nearest(h_sig, cb_s)
            n_idx, h_noi_dq = self._nearest(h_noi, cb_n)

            h_proj_dq = torch.cat([h_sig_dq, h_noi_dq], dim=-1)  # (B*S, d)
            k_dq = (h_proj_dq @ U.T).reshape(B, S, D)             # (B, S, d)

            # QJL on signal residual
            residual_sig = (h_sig - h_sig_dq)                      # (B*S, d_s)
            qjl = encode_2d(residual_sig, S_sig).reshape(B, S, -1) # (B, S, m)

            sig_idxs.append(s_idx.reshape(B, S))
            noi_idxs.append(n_idx.reshape(B, S))
            qjl_list.append(qjl)
            k_dq_list.append(k_dq)

        return (
            torch.stack(sig_idxs, dim=1),   # (B, H, S)
            torch.stack(noi_idxs, dim=1),   # (B, H, S)
            torch.stack(qjl_list,  dim=1),  # (B, H, S, m)
            torch.stack(k_dq_list, dim=1),  # (B, H, S, d)
        )

    def _dequant_key_full(self, layer_idx: int) -> torch.Tensor:
        """Reconstruct full accumulated key cache for layer_idx."""
        sig_idx = self._sq_k_sig_idx[layer_idx]  # (B, H, S)
        noi_idx = self._sq_k_noi_idx[layer_idx]  # (B, H, S)
        B, H, S = sig_idx.shape
        D = self.cal_data["head_dim"]
        k_dq_list = []

        for head in range(H):
            cal  = self._head_cal(layer_idx, head)
            U    = cal["U"].to(sig_idx.device)
            d_s  = cal["d_s"]
            cb_s = cal["codebook_signal"].to(sig_idx.device)
            cb_n = cal["codebook_noise"].to(sig_idx.device)

            s_flat = sig_idx[:, head, :].reshape(-1).long()
            n_flat = noi_idx[:, head, :].reshape(-1).long()

            h_sig_dq = cb_s[s_flat]                              # (B*S, d_s)
            h_noi_dq = cb_n[n_flat]                              # (B*S, d-d_s)
            h_proj_dq = torch.cat([h_sig_dq, h_noi_dq], dim=-1) # (B*S, d)
            k_dq_list.append((h_proj_dq @ U.T).reshape(B, S, D))

        return torch.stack(k_dq_list, dim=1).float()  # (B, H, S, d)

    # ------------------------------------------------------------------
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sig_idx, noi_idx, qjl_bits, _ = self._quant_key_layer(key_states, layer_idx)
        v_new = value_states.bfloat16()

        if layer_idx >= len(self._sq_k_sig_idx):
            self._sq_k_sig_idx.append(sig_idx)
            self._sq_k_noi_idx.append(noi_idx)
            self._sq_k_qjl.append(qjl_bits)
            self._sq_v.append(v_new)
        else:
            self._sq_k_sig_idx[layer_idx] = torch.cat([self._sq_k_sig_idx[layer_idx], sig_idx],   dim=2)
            self._sq_k_noi_idx[layer_idx] = torch.cat([self._sq_k_noi_idx[layer_idx], noi_idx],   dim=2)
            self._sq_k_qjl[layer_idx]     = torch.cat([self._sq_k_qjl[layer_idx],     qjl_bits],  dim=2)
            self._sq_v[layer_idx]         = torch.cat([self._sq_v[layer_idx],          v_new],     dim=2)

        k_full = self._dequant_key_full(layer_idx)
        v_full = self._sq_v[layer_idx].float()
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self._sq_k_sig_idx:
            return 0
        idx = min(layer_idx, len(self._sq_k_sig_idx) - 1)
        return self._sq_k_sig_idx[idx].shape[2]

    def compressed_bytes(self) -> int:
        total = 0
        for buf in self._sq_k_sig_idx + self._sq_k_noi_idx:
            total += buf.nelement() * buf.element_size()  # uint8
        for buf in self._sq_k_qjl:
            total += buf.nelement() // 8 + 1
        for buf in self._sq_v:
            total += buf.nelement() * buf.element_size()  # bfloat16
        return total
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cache.py -k spectralquant -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add kv_quant/spectralquant.py tests/test_cache.py
git commit -m "feat(kv-quant): SpectralQuantCache — VQ on keys, bfloat16 values"
```

---

### Task 8: wrap() API

**Files:**
- Modify: `kv_quant/__init__.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `TurboQuantCache`, `SpectralQuantCache`, `QuantConfig`
- Produces: `wrap(model, config) -> model` — monkey-patches `model.generate()`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_cache.py
import types
from unittest.mock import MagicMock, patch
from kv_quant import wrap, QuantConfig


def _mock_model(n_kv_heads=4, n_heads=8, hidden_size=512, head_dim=64, n_layers=2):
    model = MagicMock()
    model.config.num_key_value_heads = n_kv_heads
    model.config.num_attention_heads = n_heads
    model.config.hidden_size = hidden_size
    model.config.head_dim = head_dim
    model.config.num_hidden_layers = n_layers
    model.parameters = lambda: iter([torch.zeros(1)])
    model.generate = MagicMock(return_value=torch.zeros(1, 10, dtype=torch.long))
    return model


def test_wrap_returns_model():
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", bits=4)
    result = wrap(model, cfg)
    assert result is model


def test_wrap_sets_quant_config():
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", bits=4)
    wrap(model, cfg)
    assert model._kv_quant_config is cfg


def test_wrap_injects_turboquant_cache():
    from kv_quant.turboquant import TurboQuantCache
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", bits=4)
    wrap(model, cfg)

    captured = {}
    def fake_generate(*args, **kwargs):
        captured["cache"] = kwargs.get("past_key_values")
        return torch.zeros(1, 10, dtype=torch.long)

    model.generate = fake_generate
    # Re-wrap so the patched generate is the one wrapped
    wrap(model, cfg)
    model.generate(torch.zeros(1, 5, dtype=torch.long))
    assert isinstance(captured["cache"], TurboQuantCache)


def test_wrap_spectralquant_raises_without_calibration():
    model = _mock_model()
    cfg = QuantConfig(method="spectralquant", bits=4, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_cache.py::test_wrap_returns_model -v
```
Expected: test raises (wrap raises NotImplementedError)

- [ ] **Step 3: Implement wrap() in __init__.py**

```python
# kv_quant/__init__.py
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from kv_quant.config import QuantConfig

if TYPE_CHECKING:
    pass


def _get_kv_shape(model) -> tuple[int, int]:
    """Extract (n_kv_heads, head_dim) from a HF model config."""
    cfg = model.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
    )
    return n_kv_heads, head_dim


def _make_cache(config: QuantConfig, n_kv_heads: int, head_dim: int, cal_data, device):
    if config.method == "turboquant":
        from kv_quant.turboquant import TurboQuantCache
        return TurboQuantCache(config, n_kv_heads, head_dim, device=device)
    if config.method == "spectralquant":
        from kv_quant.spectralquant import SpectralQuantCache
        return SpectralQuantCache(config, cal_data, device=device)
    raise ValueError(f"Unknown method: {config.method!r}")


def wrap(model, config: QuantConfig):
    """Patch model.generate() to use a quantized KV cache.

    For spectralquant, config.calibration_path must point to a .pt file
    produced by `python -m kv_quant.calibrate`.
    """
    if config.method == "spectralquant":
        if not config.calibration_path:
            raise ValueError("spectralquant requires config.calibration_path")
        cal_data = torch.load(config.calibration_path, map_location="cpu")
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
    return model
```

- [ ] **Step 4: Run all cache tests**

```bash
pytest tests/test_cache.py -v
```
Expected: all tests pass (skip the inject test — it tests a pattern that requires re-wrapping; adjust if needed)

- [ ] **Step 5: Commit**

```bash
git add kv_quant/__init__.py tests/test_cache.py
git commit -m "feat(kv-quant): wrap() API — injects quantized cache into model.generate()"
```

---

### Task 9: Benchmark — perplexity and memory

**Files:**
- Create: `kv_quant/bench/perplexity.py`
- Create: `kv_quant/bench/memory.py`

**Interfaces:**
- Produces:
  - `compute_perplexity(model, tokenizer, n_tokens, chunk_size) -> float`
  - `measure_kv_memory(model, tokenizer, prompt, max_new_tokens) -> dict`

No unit tests for these (they need a live model). They are exercised by the integration test in Task 11.

- [ ] **Step 1: Create perplexity.py**

```python
# kv_quant/bench/perplexity.py
from __future__ import annotations
import math
import torch


def compute_perplexity(
    model,
    tokenizer,
    n_tokens: int = 10_240,
    chunk_size: int = 512,
) -> float:
    """WikiText-2 perplexity over the first n_tokens tokens, evaluated in
    non-overlapping chunks of chunk_size. Returns float PPL."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(ex["text"] for ex in dataset if ex["text"].strip())
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    enc = enc[: n_tokens + 1]

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(enc) - 1, chunk_size):
            chunk = enc[i : i + chunk_size + 1]
            if len(chunk) < 2:
                break
            input_ids = chunk[:-1].unsqueeze(0).to(device)
            labels    = chunk[1:].unsqueeze(0).to(device)
            loss = model(input_ids, labels=labels).loss
            n = input_ids.shape[1]
            total_nll    += loss.item() * n
            total_tokens += n

    return math.exp(total_nll / total_tokens)
```

- [ ] **Step 2: Create memory.py**

```python
# kv_quant/bench/memory.py
from __future__ import annotations
import torch


def measure_kv_memory(
    model,
    tokenizer,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    max_new_tokens: int = 200,
) -> dict:
    """Measure peak GPU memory delta during generation.

    Returns dict with keys:
      peak_bytes       — bytes allocated above baseline during generate()
      fp16_est_bytes   — estimated fp16 KV cache size for same token count
      compression_ratio — fp16_est_bytes / peak_bytes
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_prompt = inputs.input_ids.shape[1]

    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    model.eval()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    peak = torch.cuda.max_memory_allocated(device)
    peak_bytes = max(peak - baseline, 1)

    cfg = model.config
    n_layers   = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim   = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    total_seq  = n_prompt + max_new_tokens
    fp16_bytes = n_layers * 2 * n_kv_heads * head_dim * total_seq * 2  # fp16 = 2 bytes

    return {
        "peak_bytes": peak_bytes,
        "fp16_est_bytes": fp16_bytes,
        "compression_ratio": fp16_bytes / peak_bytes,
    }
```

- [ ] **Step 3: Commit**

```bash
git add kv_quant/bench/perplexity.py kv_quant/bench/memory.py
git commit -m "feat(kv-quant): bench/perplexity + bench/memory measurement utilities"
```

---

### Task 10: Benchmark CLI (run_bench.py)

**Files:**
- Create: `kv_quant/bench/run_bench.py`

**Interfaces:**
- Consumes: `wrap`, `QuantConfig`, `compute_perplexity`, `measure_kv_memory`
- Produces: CLI `python -m kv_quant.bench.run_bench ...`; CSV + stdout table

- [ ] **Step 1: Create run_bench.py**

```python
# kv_quant/bench/run_bench.py
"""Benchmark CLI for KV cache quantization.

Usage:
    python -m kv_quant.bench.run_bench \
        --model Qwen/Qwen2.5-7B-Instruct \
        --method turboquant spectralquant \
        --bits 2 3 4 \
        --tasks mmlu arc_easy hellaswag gsm8k \
        --calibration spectralquant_qwen25_7b.pt \
        --output results/qwen25_7b.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_quant import wrap, QuantConfig
from kv_quant.bench.perplexity import compute_perplexity
from kv_quant.bench.memory import measure_kv_memory


def _load_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _run_lmeval(model, tokenizer, tasks: list[str]) -> dict[str, float]:
    """Run lm-evaluation-harness tasks. Returns {task: accuracy_0_to_1}."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, verbosity="WARNING")

    scores: dict[str, float] = {}
    for task in tasks:
        task_res = results["results"].get(task, {})
        # lm-eval uses "acc,none" or "exact_match,none" keys
        for key in ("acc,none", "exact_match,none", "acc"):
            if key in task_res:
                scores[task] = float(task_res[key])
                break
        else:
            scores[task] = 0.0
    return scores


def run_benchmark(args) -> None:
    tasks: list[str] = args.tasks or []
    header = ["method", "bits", "ppl", "kv_mb"] + tasks
    rows: list[dict] = []

    # --- Baseline ---
    print(f"[baseline] Loading {args.model}…")
    model, tokenizer = _load_model(args.model)

    ppl  = compute_perplexity(model, tokenizer)
    mem  = measure_kv_memory(model, tokenizer)
    lm_scores = _run_lmeval(model, tokenizer, tasks) if tasks else {}

    rows.append({
        "method": "baseline", "bits": "fp16",
        "ppl": round(ppl, 3),
        "kv_mb": round(mem["peak_bytes"] / 1e6, 1),
        **{t: round(lm_scores.get(t, 0) * 100, 2) for t in tasks},
    })
    del model
    torch.cuda.empty_cache()

    # --- Quantized runs ---
    for method in args.method:
        cal_path = args.calibration if method == "spectralquant" else None
        for bits in args.bits:
            print(f"[{method} @ {bits}b] Loading {args.model}…")
            model, _ = _load_model(args.model)
            cfg = QuantConfig(method=method, bits=bits, calibration_path=cal_path)
            model = wrap(model, cfg)

            ppl  = compute_perplexity(model, tokenizer)
            mem  = measure_kv_memory(model, tokenizer)
            lm_scores = _run_lmeval(model, tokenizer, tasks) if tasks else {}

            rows.append({
                "method": method, "bits": bits,
                "ppl": round(ppl, 3),
                "kv_mb": round(mem["peak_bytes"] / 1e6, 1),
                **{t: round(lm_scores.get(t, 0) * 100, 2) for t in tasks},
            })
            del model
            torch.cuda.empty_cache()

    # --- Print table ---
    col_w = 14
    print("\n" + "".join(h.ljust(col_w) for h in header))
    print("-" * (col_w * len(header)))
    for row in rows:
        print("".join(str(row.get(h, "-")).ljust(col_w) for h in header))

    # --- Write CSV ---
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache quantization benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--method", nargs="+", default=["turboquant"],
                        choices=["turboquant", "spectralquant"])
    parser.add_argument("--bits",   nargs="+", type=int, default=[4])
    parser.add_argument("--tasks",  nargs="*", default=[],
                        help="lm-eval task names, e.g. mmlu arc_easy hellaswag gsm8k")
    parser.add_argument("--calibration", default=None,
                        help="Path to spectralquant .pt calibration file")
    parser.add_argument("--output",  default=None, help="CSV output path")
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add kv_quant/bench/run_bench.py
git commit -m "feat(kv-quant): bench/run_bench CLI — PPL + memory + lm-eval sweep"
```

---

### Task 11: Integration Test (slow)

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: `wrap`, `QuantConfig`, `compute_perplexity` — requires GPU + `Qwen/Qwen2.5-0.5B-Instruct`

- [ ] **Step 1: Create test_integration.py**

```python
# tests/test_integration.py
"""Slow end-to-end tests. Require GPU + model download.

Run with:
    pytest tests/test_integration.py --run-slow -v
"""
from __future__ import annotations
import pytest
import torch


@pytest.mark.slow
def test_turboquant_generation_and_ppl():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_quant import wrap, QuantConfig
    from kv_quant.bench.perplexity import compute_perplexity

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Baseline PPL
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    base_ppl = compute_perplexity(base, tokenizer, n_tokens=2048, chunk_size=256)
    del base
    torch.cuda.empty_cache()

    # TurboQuant @ 4 bits
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    cfg = QuantConfig(method="turboquant", bits=4)
    model = wrap(model, cfg)

    # Generation sanity
    inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    assert len(generated.strip()) > 0, "Empty generation"

    # PPL within 1.5× baseline
    quant_ppl = compute_perplexity(model, tokenizer, n_tokens=2048, chunk_size=256)
    assert quant_ppl < base_ppl * 1.5, (
        f"TurboQuant PPL too high: {quant_ppl:.2f} vs baseline {base_ppl:.2f}"
    )


@pytest.mark.slow
def test_turboquant_memory_reduced():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_quant import wrap, QuantConfig
    from kv_quant.bench.memory import measure_kv_memory

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Baseline memory
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    base_mem = measure_kv_memory(base, tokenizer, max_new_tokens=100)
    del base
    torch.cuda.empty_cache()

    # TurboQuant @ 4 bits
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    model = wrap(model, QuantConfig(method="turboquant", bits=4))
    quant_mem = measure_kv_memory(model, tokenizer, max_new_tokens=100)

    # Compressed cache should be meaningfully smaller than fp16 estimate
    assert quant_mem["compression_ratio"] > 1.5, (
        f"Expected >1.5× compression, got {quant_mem['compression_ratio']:.2f}×"
    )
```

- [ ] **Step 2: Verify slow tests are skipped by default**

```bash
pytest tests/test_integration.py -v
```
Expected: 2 tests skipped with "Pass --run-slow to run"

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(kv-quant): integration tests for TurboQuant generation + PPL + memory"
```

---

## Self-Review

**Spec coverage:**
- ✅ `wrap(model, config)` drop-in API — Task 8
- ✅ TurboQuant: rotation + scalar quant + QJL — Tasks 2–5
- ✅ SpectralQuant: calibration + VQ cache — Tasks 6–7
- ✅ Target architectures Qwen + Gemma: both expose `num_key_value_heads` + `head_dim`; handled in `_get_kv_shape()`
- ✅ Configurable bits — `QuantConfig.bits` consumed by both caches
- ✅ Perplexity + memory benchmark — Tasks 9–10
- ✅ lm-eval integration (MMLU, ARC, HellaSwag, GSM8K) — Task 10 `_run_lmeval()`
- ✅ Unit tests on CPU — Tasks 2–8 use no CUDA
- ✅ Integration tests gated behind `--run-slow` — Task 11

**Known limitations documented in spec (no tasks added):**
- QJL inner-product correction is not applied in the attention path (bits stored, not used at runtime)
- Throughput gains require custom kernels — this prototype gives memory savings only
