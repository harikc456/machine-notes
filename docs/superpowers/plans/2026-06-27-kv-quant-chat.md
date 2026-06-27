# KV-Quant Chat UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit chat app that lets users try TurboQuant KV cache quantization on Qwen3/Gemma4 models with live memory, throughput, and compression stats.

**Architecture:** Two new files — `chat/kv_quant_utils.py` (data types + pure functions) and `chat/kv_quant_chat.py` (Streamlit app). Model list in `chat/models.yaml`. The app creates a `TurboQuantCache` instance before calling `model.generate()` and reads `cache.compressed_bytes()` afterward for stats. A background thread drives generation while `st.write_stream(TextIteratorStreamer(...))` streams tokens to the UI.

**Tech Stack:** Streamlit, transformers (AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer), PyTorch, PyYAML, `kv_quant` (local package at repo root)

## Global Constraints

- Do not modify any existing files (`chat/chat.py`, `chat/chat_utils.py`, `kv_quant/`, `tests/test_cache.py`, etc.)
- New test file goes in `tests/test_kv_quant_utils.py`
- `chat/models.yaml` lists exactly 6 models (4 Qwen3 + 2 Gemma4) with `head_dim: 128` for all
- `get_stats` baseline bytes formula: `n_layers * n_kv_heads * seq_len * head_dim * 2 * 2` (K+V × fp16 bytes)
- No `@st.cache_resource` for model loading — model stored in `st.session_state` after "Load Model" click
- `do_sample=False` for all generation calls (deterministic)
- `torch_dtype=torch.bfloat16` for all model loads
- Stats panel hidden until at least one generation has completed
- Quant sidebar controls disabled when "Baseline (full precision)" mode selected

---

### Task 1: models.yaml + kv_quant_utils.py

**Files:**
- Create: `chat/models.yaml`
- Create: `chat/kv_quant_utils.py`
- Create: `tests/test_kv_quant_utils.py`

**Interfaces:**
- Consumes: `kv_quant.turboquant.TurboQuantCache` (for `cache.compressed_bytes()` type hint only)
- Produces:
  - `ModelEntry` — dataclass with `.id: str`, `.label: str`, `.head_dim: int`, `.default_bits: int`
  - `ChatStats` — dataclass with `.kv_memory_mb: float`, `.baseline_memory_mb: float`, `.tokens_per_sec: float`, `.compression_ratio: float`
  - `load_models_yaml(path: Path | str | None = None) -> list[ModelEntry]`
  - `load_hf_model(model_id: str, device: torch.device) -> tuple[AutoModelForCausalLM, AutoTokenizer]`
  - `get_kv_shape(model) -> tuple[int, int]` — returns `(n_kv_heads, head_dim)`
  - `get_stats(cache, n_new_tokens: int, elapsed: float, n_layers: int, n_kv_heads: int, seq_len: int, head_dim: int) -> ChatStats`

- [ ] **Step 1: Write `chat/models.yaml`**

```yaml
models:
  - id: Qwen/Qwen3-0.6B-Instruct
    label: "Qwen3 0.6B"
    head_dim: 128
    default_bits: 4

  - id: Qwen/Qwen3-1.7B-Instruct
    label: "Qwen3 1.7B"
    head_dim: 128
    default_bits: 4

  - id: Qwen/Qwen3-4B-Instruct
    label: "Qwen3 4B"
    head_dim: 128
    default_bits: 4

  - id: Qwen/Qwen3-8B-Instruct
    label: "Qwen3 8B"
    head_dim: 128
    default_bits: 4

  - id: google/gemma-4-2b-it
    label: "Gemma4 2B"
    head_dim: 128
    default_bits: 4

  - id: google/gemma-4-9b-it
    label: "Gemma4 9B"
    head_dim: 128
    default_bits: 4
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_kv_quant_utils.py`:

```python
from __future__ import annotations
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chat.kv_quant_utils import ChatStats, ModelEntry, get_stats, load_models_yaml


def test_load_models_yaml_parses_entries(tmp_path):
    yaml_text = textwrap.dedent("""
    models:
      - id: Qwen/Qwen3-0.6B-Instruct
        label: "Qwen3 0.6B"
        head_dim: 128
        default_bits: 4
      - id: google/gemma-4-2b-it
        label: "Gemma4 2B"
        head_dim: 128
        default_bits: 4
    """)
    f = tmp_path / "models.yaml"
    f.write_text(yaml_text)
    entries = load_models_yaml(f)
    assert len(entries) == 2
    assert isinstance(entries[0], ModelEntry)
    assert entries[0].id == "Qwen/Qwen3-0.6B-Instruct"
    assert entries[0].label == "Qwen3 0.6B"
    assert entries[0].head_dim == 128
    assert entries[0].default_bits == 4
    assert entries[1].id == "google/gemma-4-2b-it"


def test_load_models_yaml_default_path():
    # Default path resolves to chat/models.yaml in the repo
    entries = load_models_yaml()
    assert len(entries) == 6
    ids = [e.id for e in entries]
    assert "Qwen/Qwen3-0.6B-Instruct" in ids
    assert "google/gemma-4-2b-it" in ids


def test_get_stats_baseline_mode():
    # baseline: n_layers=28, n_kv_heads=8, seq_len=512, head_dim=128
    # baseline_bytes = 28 * 8 * 512 * 128 * 2 * 2 = 58_720_256
    stats = get_stats(
        cache=None,
        n_new_tokens=50,
        elapsed=1.0,
        n_layers=28,
        n_kv_heads=8,
        seq_len=512,
        head_dim=128,
    )
    assert isinstance(stats, ChatStats)
    assert stats.tokens_per_sec == pytest.approx(50.0)
    assert stats.compression_ratio == pytest.approx(1.0)
    assert stats.kv_memory_mb == pytest.approx(stats.baseline_memory_mb)
    assert stats.baseline_memory_mb == pytest.approx(58_720_256 / 1e6, rel=1e-4)


def test_get_stats_quantized_mode():
    mock_cache = MagicMock()
    mock_cache.compressed_bytes.return_value = 5_000_000  # 5 MB
    # baseline_bytes = 28 * 8 * 512 * 128 * 2 * 2 = 58_720_256
    stats = get_stats(
        cache=mock_cache,
        n_new_tokens=100,
        elapsed=2.0,
        n_layers=28,
        n_kv_heads=8,
        seq_len=512,
        head_dim=128,
    )
    assert stats.tokens_per_sec == pytest.approx(50.0)
    assert stats.kv_memory_mb == pytest.approx(5.0)
    assert stats.baseline_memory_mb == pytest.approx(58_720_256 / 1e6, rel=1e-4)
    assert stats.compression_ratio == pytest.approx(58_720_256 / 5_000_000, rel=1e-3)


def test_get_stats_zero_elapsed_does_not_divide_by_zero():
    stats = get_stats(
        cache=None,
        n_new_tokens=10,
        elapsed=0.0,
        n_layers=2,
        n_kv_heads=4,
        seq_len=64,
        head_dim=64,
    )
    assert stats.tokens_per_sec >= 0
    assert not (stats.tokens_per_sec != stats.tokens_per_sec)  # not NaN
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /path/to/machine-notes
pytest tests/test_kv_quant_utils.py -v
```

Expected: ImportError — `chat.kv_quant_utils` does not exist yet.

- [ ] **Step 4: Write `chat/kv_quant_utils.py`**

```python
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is on sys.path when imported standalone
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_MODELS_YAML = Path(__file__).parent / "models.yaml"


@dataclass
class ModelEntry:
    id: str
    label: str
    head_dim: int
    default_bits: int


@dataclass
class ChatStats:
    kv_memory_mb: float
    baseline_memory_mb: float
    tokens_per_sec: float
    compression_ratio: float


def load_models_yaml(path: Path | str | None = None) -> list[ModelEntry]:
    if path is None:
        path = _MODELS_YAML
    with open(path) as f:
        data = yaml.safe_load(f)
    return [ModelEntry(**m) for m in data["models"]]


def load_hf_model(
    model_id: str,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
    )
    model.eval()
    return model, tokenizer


def get_kv_shape(model) -> tuple[int, int]:
    cfg = model.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return n_kv_heads, head_dim


def get_stats(
    cache,
    n_new_tokens: int,
    elapsed: float,
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
) -> ChatStats:
    # baseline: K + V tensors, fp16 (2 bytes), all layers and heads
    baseline_bytes = n_layers * n_kv_heads * seq_len * head_dim * 2 * 2
    baseline_mb = baseline_bytes / 1e6

    if cache is not None:
        compressed = cache.compressed_bytes()
        kv_mb = compressed / 1e6
        ratio = baseline_bytes / max(compressed, 1)
    else:
        kv_mb = baseline_mb
        ratio = 1.0

    tps = n_new_tokens / max(elapsed, 1e-9)

    return ChatStats(
        kv_memory_mb=kv_mb,
        baseline_memory_mb=baseline_mb,
        tokens_per_sec=tps,
        compression_ratio=ratio,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_kv_quant_utils.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add chat/models.yaml chat/kv_quant_utils.py tests/test_kv_quant_utils.py
git commit -m "feat(kv-quant-chat): add models.yaml, kv_quant_utils, and tests"
```

---

### Task 2: kv_quant_chat.py — Streamlit app

**Files:**
- Create: `chat/kv_quant_chat.py`

**Interfaces:**
- Consumes from Task 1:
  - `ModelEntry` — `.id`, `.label`, `.head_dim`, `.default_bits`
  - `ChatStats` — `.kv_memory_mb`, `.baseline_memory_mb`, `.tokens_per_sec`, `.compression_ratio`
  - `load_models_yaml(path) -> list[ModelEntry]`
  - `load_hf_model(model_id, device) -> tuple[AutoModelForCausalLM, AutoTokenizer]`
  - `get_kv_shape(model) -> tuple[int, int]`
  - `get_stats(cache, n_new_tokens, elapsed, n_layers, n_kv_heads, seq_len, head_dim) -> ChatStats`
- Consumes from kv_quant:
  - `QuantConfig(method="turboquant", bits=int, value_bits=int, buffer_size=int)`
  - `TurboQuantCache(config, n_kv_heads, head_dim, device=device)`

**Session state keys used:**

| Key | Type |
|-----|------|
| `model` | `AutoModelForCausalLM` |
| `tokenizer` | `AutoTokenizer` |
| `model_id` | `str` |
| `mode` | `"baseline"` or `"turboquant"` |
| `quant_config` | `QuantConfig` |
| `history` | `list[{"role": str, "content": str}]` |
| `last_stats` | `ChatStats \| None` |

- [ ] **Step 1: Write syntax check test**

Append to `tests/test_kv_quant_utils.py`:

```python
import py_compile, os

def test_kv_quant_chat_syntax():
    chat_path = os.path.join(os.path.dirname(__file__), "..", "chat", "kv_quant_chat.py")
    # Compiles to bytecode — raises SyntaxError if the file has syntax errors
    py_compile.compile(chat_path, doraise=True)
```

- [ ] **Step 2: Run the syntax test to verify it fails (file doesn't exist yet)**

```bash
pytest tests/test_kv_quant_utils.py::test_kv_quant_chat_syntax -v
```

Expected: FAIL — file not found / compile error.

- [ ] **Step 3: Write `chat/kv_quant_chat.py`**

```python
"""
KV-Quant Chat — try TurboQuant KV cache quantization on HuggingFace models.

Run from repo root:
    streamlit run chat/kv_quant_chat.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st
import torch
from transformers import TextIteratorStreamer

from chat.kv_quant_utils import (
    ChatStats,
    ModelEntry,
    get_kv_shape,
    get_stats,
    load_hf_model,
    load_models_yaml,
)
from kv_quant.config import QuantConfig
from kv_quant.turboquant import TurboQuantCache

_MODELS_YAML = Path(__file__).parent / "models.yaml"

st.set_page_config(page_title="KV-Quant Chat", layout="wide")
st.title("KV-Quant Chat")

# ── Load model list ───────────────────────────────────────────────────────────
model_entries: list[ModelEntry] = load_models_yaml(_MODELS_YAML)
model_labels = [e.label for e in model_entries]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")

    selected_idx = st.selectbox(
        "Model",
        range(len(model_entries)),
        format_func=lambda i: model_labels[i],
        key="model_selector",
    )
    entry = model_entries[selected_idx]

    st.subheader("Cache Mode")
    mode_label = st.radio(
        "Cache Mode",
        ["Baseline (full precision)", "TurboQuant"],
        key="cache_mode",
        label_visibility="collapsed",
    )
    use_quant = mode_label == "TurboQuant"
    quant_disabled = not use_quant

    key_bits = st.selectbox(
        "Key bits",
        [1, 2, 3, 4],
        index=[1, 2, 3, 4].index(entry.default_bits),
        key="key_bits",
        disabled=quant_disabled,
    )
    value_bits = st.selectbox(
        "Value bits",
        [1, 2],
        index=1,
        key="value_bits",
        disabled=quant_disabled,
    )
    buffer_size = st.selectbox(
        "Buffer size",
        [64, 128, 256],
        index=1,
        key="buffer_size",
        disabled=quant_disabled,
    )
    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=32,
        max_value=512,
        step=32,
        value=256,
        key="max_new_tokens",
    )

    st.divider()
    load_clicked = st.button("Load Model", use_container_width=True)

    if load_clicked:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with st.spinner(f"Loading {entry.label}…"):
            try:
                model, tokenizer = load_hf_model(entry.id, device)
                st.session_state["model"] = model
                st.session_state["tokenizer"] = tokenizer
                st.session_state["model_id"] = entry.id
                st.session_state["mode"] = "turboquant" if use_quant else "baseline"
                st.session_state["quant_config"] = QuantConfig(
                    method="turboquant",
                    bits=key_bits,
                    value_bits=value_bits,
                    buffer_size=buffer_size,
                )
                st.session_state["history"] = []
                st.session_state["last_stats"] = None
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if "model_id" in st.session_state:
        loaded_id: str = st.session_state["model_id"]
        loaded_mode: str = st.session_state.get("mode", "baseline")
        current_mode = "turboquant" if use_quant else "baseline"
        st.success(f"Loaded: {loaded_id.split('/')[-1]}")
        if loaded_id != entry.id or loaded_mode != current_mode:
            st.warning("Settings changed — click Load Model to apply.")

# ── Main area ─────────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.info("Select a model and click **Load Model** to start.")
    st.stop()

if st.button("Clear Chat"):
    st.session_state["history"] = []
    st.session_state["last_stats"] = None
    st.rerun()

for msg in st.session_state.get("history", []):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type a message…")

if user_input:
    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]
    current_mode = st.session_state.get("mode", "baseline")
    quant_config: QuantConfig = st.session_state.get("quant_config")
    _max_new: int = st.session_state.get("max_new_tokens", 256)

    history: list[dict] = st.session_state.get("history", [])
    history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    input_ids = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs: dict = dict(
        input_ids=input_ids,
        max_new_tokens=_max_new,
        streamer=streamer,
        do_sample=False,
    )

    cache: TurboQuantCache | None = None
    if current_mode == "turboquant" and quant_config is not None:
        n_kv_heads, head_dim = get_kv_shape(model)
        cache = TurboQuantCache(
            quant_config, n_kv_heads, head_dim, device=model.device
        )
        gen_kwargs["past_key_values"] = cache

    start = time.perf_counter()
    thread = threading.Thread(
        target=model.generate, kwargs=gen_kwargs, daemon=True
    )
    thread.start()

    with st.chat_message("assistant"):
        response: str = st.write_stream(streamer)

    thread.join()
    elapsed = time.perf_counter() - start

    history.append({"role": "assistant", "content": response})
    st.session_state["history"] = history

    n_new = len(tokenizer.encode(response, add_special_tokens=False))
    n_kv_heads_s, head_dim_s = get_kv_shape(model)
    seq_len = input_ids.shape[-1] + n_new

    st.session_state["last_stats"] = get_stats(
        cache=cache,
        n_new_tokens=n_new,
        elapsed=elapsed,
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads_s,
        seq_len=seq_len,
        head_dim=head_dim_s,
    )

# ── Stats panel ───────────────────────────────────────────────────────────────
stats: ChatStats | None = st.session_state.get("last_stats")
if stats is not None:
    st.divider()
    col1, col2, col3 = st.columns(3)
    delta = (
        None
        if stats.compression_ratio == 1.0
        else f"↓ from {stats.baseline_memory_mb:.1f} MB"
    )
    col1.metric("KV Memory", f"{stats.kv_memory_mb:.1f} MB", delta=delta)
    col2.metric("Tokens/sec", f"{stats.tokens_per_sec:.1f} tok/s")
    col3.metric("Compression", f"{stats.compression_ratio:.1f}×")
```

- [ ] **Step 4: Run the syntax test**

```bash
pytest tests/test_kv_quant_utils.py::test_kv_quant_chat_syntax -v
```

Expected: PASS — file compiles without errors.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
pytest tests/test_kv_quant_utils.py tests/test_ops.py tests/test_cache.py -v
```

Expected: All tests pass. (Integration tests under `--run-slow` are excluded.)

- [ ] **Step 6: Commit**

```bash
git add chat/kv_quant_chat.py tests/test_kv_quant_utils.py
git commit -m "feat(kv-quant-chat): Streamlit chat app with TurboQuant toggle and live stats"
```
