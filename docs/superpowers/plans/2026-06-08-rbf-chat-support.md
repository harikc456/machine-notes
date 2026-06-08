# rbf_ffn Chat Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `idlm/chat.py` to also support plain `rbf_ffn` AR checkpoint runs alongside existing I-DLM runs, using autoregressive generation instead of ISD.

**Architecture:** Add `model_type` discriminator to `RunInfo`, add `discover_rbf_runs()` / `load_rbf_model()` / `ar_generate()` helpers to `chat_utils.py`, then update `chat.py` with a model-type radio that switches run list, tokenizer, sliders, and generation path.

**Tech Stack:** PyTorch, tiktoken (`r50k_base` for rbf_ffn, `gpt2` for I-DLM), Streamlit 1.57, existing `rbf_ffn.models.model.CausalLM` and `rbf_ffn.config.load_config`.

---

## Key Facts

- **rbf_ffn experiments dir:** `rbf_ffn/experiments/` (136 runs), accessible as `Path(__file__).parent.parent / "rbf_ffn" / "experiments"` from `idlm/chat.py`
- **rbf_ffn checkpoint:** `checkpoint_best.pt` — loaded as `ckpt["model"]` into `CausalLM`
- **rbf_ffn metrics:** epoch-level JSONL with `val_loss` key (no `"type"` field), e.g. `{"epoch": 0, "val_loss": 5.57, ...}`
- **rbf_ffn tokenizer:** `r50k_base` (tiktoken), 50257 real tokens — same token IDs as `gpt2`
- **Kronecker vocab padding:** some rbf_ffn models have `vocab_size: 65536` in config for Kronecker LM head factoring. Cap logits at `real_vocab_size=50257` before sampling.
- **CausalLM forward:** `model(tokens)` → `(logits, hs)` where `logits: (B, N, vocab_size)`
- **No LoRA:** rbf_ffn models load directly, no LoRA wrapping
- **Generation:** autoregressive token-by-token sampling, no ISD stride

---

## File Map

| Action | Path | Change |
|---|---|---|
| Modify | `idlm/chat_utils.py` | Add `model_type` to `RunInfo`; add `discover_rbf_runs()`, `_read_rbf_val_loss()`, `load_rbf_model()`, `ar_generate()` |
| Modify | `idlm/tests/test_chat_utils.py` | Add tests for the four new functions |
| Modify | `idlm/chat.py` | Model-type radio, rbf run list, conditional sliders/tokenizer/generation |

---

## Task 1: `discover_rbf_runs()` + `ar_generate()` + `model_type` field

**Files:**
- Modify: `idlm/chat_utils.py`
- Modify: `idlm/tests/test_chat_utils.py`

- [ ] **Step 1: Write failing tests**

Add to `idlm/tests/test_chat_utils.py`:

```python
from idlm.chat_utils import discover_rbf_runs, ar_generate


def _make_rbf_run(root: Path, name: str, epochs: list[dict]) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text(
        "d_model: 64\nn_heads: 2\nn_layers: 2\nvocab_size: 50257\nseq_len: 64\n"
    )
    with open(run_dir / "metrics.jsonl", "w") as f:
        for row in epochs:
            f.write(json.dumps(row) + "\n")
    return run_dir


def test_discover_rbf_runs_sorted_newest_first(tmp_path):
    _make_rbf_run(tmp_path, "20260316_000000_swiglu_d256",
                  [{"epoch": 0, "val_loss": 5.5}])
    _make_rbf_run(tmp_path, "20260404_000000_swiglu_qknorm_d256",
                  [{"epoch": 0, "val_loss": 4.9}])
    runs = discover_rbf_runs(tmp_path)
    assert len(runs) == 2
    assert runs[0].dir_name == "20260404_000000_swiglu_qknorm_d256"


def test_discover_rbf_runs_reads_last_val_loss(tmp_path):
    _make_rbf_run(tmp_path, "20260404_000000_swiglu_d256", [
        {"epoch": 0, "val_loss": 5.5},
        {"epoch": 1, "val_loss": 4.9},
    ])
    runs = discover_rbf_runs(tmp_path)
    assert runs[0].final_loss == pytest.approx(4.9)


def test_discover_rbf_runs_model_type_is_rbf(tmp_path):
    _make_rbf_run(tmp_path, "20260404_000000_swiglu_d256",
                  [{"epoch": 0, "val_loss": 4.9}])
    runs = discover_rbf_runs(tmp_path)
    assert runs[0].model_type == "rbf"


def test_discover_rbf_runs_skips_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "20260404_000000_swiglu_d256"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("d_model: 64\n")
    # no checkpoint_best.pt
    runs = discover_rbf_runs(tmp_path)
    assert len(runs) == 0


def test_discover_rbf_runs_none_loss_when_no_metrics(tmp_path):
    run_dir = tmp_path / "20260404_000000_swiglu_d256"
    run_dir.mkdir()
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text("d_model: 64\n")
    runs = discover_rbf_runs(tmp_path)
    assert runs[0].final_loss is None


def test_ar_generate_returns_correct_length():
    import torch
    import torch.nn as nn

    class _TinyLM(nn.Module):
        def forward(self, tokens):
            B, N = tokens.shape
            return torch.zeros(B, N, 50257), []

    model = _TinyLM()
    prompt = [1, 2, 3]
    out = ar_generate(model, prompt, gen_len=10, device=torch.device("cpu"))
    assert len(out) == len(prompt) + 10


def test_ar_generate_caps_to_real_vocab():
    """Generated token IDs must be < real_vocab_size (50257)."""
    import torch
    import torch.nn as nn

    class _BigVocabLM(nn.Module):
        def forward(self, tokens):
            B, N = tokens.shape
            # logits over 65536, with mass only on tokens >= 50257
            logits = torch.full((B, N, 65536), -1e9)
            logits[:, :, 50257:] = 10.0
            return logits, []

    model = _BigVocabLM()
    out = ar_generate(model, [1], gen_len=5, device=torch.device("cpu"), real_vocab_size=50257)
    generated = out[1:]
    assert all(0 <= t < 50257 for t in generated)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/test_chat_utils.py::test_discover_rbf_runs_sorted_newest_first -v 2>&1 | tail -10
```
Expected: `ImportError` or `NameError` — `discover_rbf_runs` not defined.

- [ ] **Step 3: Add `model_type` field to `RunInfo` and update `discover_runs()`**

In `idlm/chat_utils.py`, update `RunInfo` (add field) and `discover_runs` (set it):

```python
@dataclass
class RunInfo:
    dir_name: str
    run_dir: Path
    final_loss: float | None
    model_type: str = "idlm"  # "idlm" or "rbf"

    @property
    def label(self) -> str:
        short = self.dir_name[:15]
        loss_str = f"{self.final_loss:.4f}" if self.final_loss is not None else "N/A"
        return f"{short} | loss: {loss_str}"
```

In `discover_runs()`, add `model_type="idlm"` to each `RunInfo(...)` constructor call:

```python
        runs.append(RunInfo(
            dir_name=run_dir.name,
            run_dir=run_dir,
            final_loss=final_loss,
            model_type="idlm",
        ))
```

- [ ] **Step 4: Add `discover_rbf_runs()`, `_read_rbf_val_loss()`, `ar_generate()` to `chat_utils.py`**

Append after `load_model()`:

```python
from rbf_ffn.config import ModelConfig


def discover_rbf_runs(experiments_dir: Path) -> list[RunInfo]:
    """Scan rbf_ffn experiments_dir for valid AR runs, sorted newest-first."""
    runs: list[RunInfo] = []
    for run_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "checkpoint_best.pt").exists():
            continue
        if not (run_dir / "config.yaml").exists():
            continue
        final_loss = _read_rbf_val_loss(run_dir / "metrics.jsonl")
        runs.append(RunInfo(
            dir_name=run_dir.name,
            run_dir=run_dir,
            final_loss=final_loss,
            model_type="rbf",
        ))
    return runs


def _read_rbf_val_loss(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    last_val_loss = None
    try:
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            if "val_loss" in row:
                last_val_loss = float(row["val_loss"])
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return last_val_loss


@torch.no_grad()
def ar_generate(
    model,
    prompt_ids: list[int],
    gen_len: int,
    device: torch.device,
    real_vocab_size: int = 50257,
) -> list[int]:
    """Autoregressive generation, capping logits to real_vocab_size."""
    model.eval()
    ids = list(prompt_ids)
    max_ctx = 512
    for _ in range(gen_len):
        ctx = ids[-max_ctx:]
        tokens = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(tokens)
        next_logits = logits[0, -1, :real_vocab_size]
        probs = torch.softmax(next_logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
    return ids
```

Also add `ModelConfig` to the import line already at the top:
```python
from rbf_ffn.config import load_config as load_ar_config, ModelConfig
```

- [ ] **Step 5: Run all tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/test_chat_utils.py -v
```
Expected: 15 tests PASS (7 existing + 8 new)

- [ ] **Step 6: Commit**

```bash
git add idlm/chat_utils.py idlm/tests/test_chat_utils.py
git commit -m "feat(chat): discover_rbf_runs, ar_generate, model_type field on RunInfo"
```

---

## Task 2: `load_rbf_model()` + test

**Files:**
- Modify: `idlm/chat_utils.py`
- Modify: `idlm/tests/test_chat_utils.py`

- [ ] **Step 1: Write failing test**

Add to `idlm/tests/test_chat_utils.py`:

```python
from idlm.chat_utils import load_rbf_model


def test_load_rbf_model_raises_on_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "20260404_swiglu_d256"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "d_model: 64\nn_heads: 2\nn_layers: 2\nvocab_size: 50257\nseq_len: 64\n"
    )
    # no checkpoint_best.pt
    import torch
    with pytest.raises(FileNotFoundError):
        load_rbf_model(run_dir, device=torch.device("cpu"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/test_chat_utils.py::test_load_rbf_model_raises_on_missing_checkpoint -v 2>&1 | tail -10
```
Expected: `ImportError` — `load_rbf_model` not defined.

- [ ] **Step 3: Add `load_rbf_model()` to `chat_utils.py`**

Append after `ar_generate()`:

```python
def load_rbf_model(
    run_dir: Path,
    device: torch.device,
) -> tuple[CausalLM, ModelConfig]:
    """Load a plain rbf_ffn CausalLM checkpoint."""
    cfg = load_ar_config(run_dir / "config.yaml")
    ckpt_path = run_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = CausalLM(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg
```

- [ ] **Step 4: Run all tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/test_chat_utils.py -v
```
Expected: 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add idlm/chat_utils.py idlm/tests/test_chat_utils.py
git commit -m "feat(chat): load_rbf_model helper"
```

---

## Task 3: Update `chat.py` to support both model types

**Files:**
- Modify: `idlm/chat.py`

This task rewrites `chat.py` in full. Read the current file first, then replace it entirely with the version below.

- [ ] **Step 1: Read the current `idlm/chat.py`** (to avoid blind overwrite)

```bash
wc -l /home/harikrishnan-c/projects/machine-notes/idlm/chat.py
```

- [ ] **Step 2: Write the updated `idlm/chat.py`**

```python
"""
I-DLM / rbf_ffn Chat UI — select a checkpoint and generate text interactively.

Run from repo root:
    PYTHONPATH=/path/to/machine-notes streamlit run idlm/chat.py
"""
from __future__ import annotations
import dataclasses
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st
import tiktoken
import torch

from idlm.chat_utils import (
    RunInfo,
    ar_generate,
    discover_rbf_runs,
    discover_runs,
    load_model,
    load_rbf_model,
)
from idlm.config import load_config as _load_idlm_config
from idlm.generate import isd_generate

IDLM_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
RBF_EXPERIMENTS_DIR = Path(__file__).parent.parent / "rbf_ffn" / "experiments"
REPO_ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="I-DLM / rbf_ffn Chat", layout="wide")
st.title("I-DLM / rbf_ffn Chat")


@st.cache_resource
def get_idlm_tokenizer():
    return tiktoken.get_encoding("gpt2")


@st.cache_resource
def get_rbf_tokenizer():
    return tiktoken.get_encoding("r50k_base")


@st.cache_resource
def get_idlm_runs() -> list[RunInfo]:
    return discover_runs(IDLM_EXPERIMENTS_DIR)


@st.cache_resource
def get_rbf_runs() -> list[RunInfo]:
    return discover_rbf_runs(RBF_EXPERIMENTS_DIR)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")

    model_family = st.radio(
        "Model family",
        ["I-DLM (ISD)", "rbf_ffn (AR)"],
        key="model_family",
    )
    is_idlm = model_family == "I-DLM (ISD)"

    runs = get_idlm_runs() if is_idlm else get_rbf_runs()
    if not runs:
        st.error(f"No valid runs found in {'idlm' if is_idlm else 'rbf_ffn'}/experiments/")
        st.stop()

    run_labels = [r.label for r in runs]

    def _reset_slider_defaults():
        idx = st.session_state["run_selector"]
        run = runs[idx]
        if is_idlm:
            try:
                cfg = _load_idlm_config(run.run_dir / "config.yaml")
                st.session_state["stride_slider"] = cfg.stride
                st.session_state["gen_len_slider"] = cfg.gen_len
            except Exception:
                st.session_state["stride_slider"] = 4
                st.session_state["gen_len_slider"] = 128
        else:
            st.session_state["gen_len_slider"] = 128

    selected_idx = st.selectbox(
        "Run",
        range(len(runs)),
        format_func=lambda i: run_labels[i],
        key="run_selector",
        on_change=_reset_slider_defaults,
    )
    selected_run = runs[selected_idx]

    if is_idlm:
        try:
            _default_cfg = _load_idlm_config(selected_run.run_dir / "config.yaml")
            default_stride = _default_cfg.stride
            default_gen_len = _default_cfg.gen_len
        except Exception:
            default_stride = 4
            default_gen_len = 128

        stride = st.slider("Stride", min_value=1, max_value=16,
                           value=default_stride, step=1, key="stride_slider")
        gen_len = st.slider("Gen len", min_value=32, max_value=512,
                            value=default_gen_len, step=32, key="gen_len_slider")
    else:
        gen_len = st.slider("Gen len", min_value=32, max_value=512,
                            value=128, step=32, key="gen_len_slider")

    st.divider()

    load_clicked = st.button("Load Model", use_container_width=True)

    if load_clicked:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with st.spinner(f"Loading {selected_run.dir_name[:15]}…"):
            try:
                if is_idlm:
                    model, cfg = load_model(selected_run.run_dir, REPO_ROOT, device)
                else:
                    model, cfg = load_rbf_model(selected_run.run_dir, device)
                st.session_state["model"] = model
                st.session_state["model_cfg"] = cfg
                st.session_state["model_key"] = selected_run.dir_name
                st.session_state["model_type"] = "idlm" if is_idlm else "rbf"
                st.session_state["device"] = device
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if "model_key" in st.session_state:
        st.success(f"Loaded: {st.session_state['model_key'][:15]}")
        if st.session_state["model_key"] != selected_run.dir_name:
            st.warning("Loaded model is from a different run — click Load Model to apply.")

# ── Main area ────────────────────────────────────────────────────────────────────
prompt_text = st.text_area(
    "Prompt",
    placeholder="Enter a prompt to continue…",
    height=120,
)

model_ready = "model" in st.session_state
generate_clicked = st.button(
    "Generate",
    disabled=not model_ready,
    help="Load a model first" if not model_ready else "",
)

if generate_clicked and prompt_text.strip():
    model = st.session_state["model"]
    base_cfg = st.session_state["model_cfg"]
    device = st.session_state["device"]
    loaded_type = st.session_state["model_type"]

    if loaded_type == "idlm":
        enc = get_idlm_tokenizer()
        cfg_override = dataclasses.replace(base_cfg, stride=stride, gen_len=gen_len)
        prompt_ids = enc.encode(prompt_text)
        with st.spinner("Generating (ISD)…"):
            output_ids = isd_generate(model, prompt_ids, cfg_override, device)
        _MASK_ID = 50256
        generated_ids = [t for t in output_ids[len(prompt_ids):] if 0 <= t < _MASK_ID]
    else:
        enc = get_rbf_tokenizer()
        prompt_ids = enc.encode(prompt_text)
        with st.spinner("Generating (AR)…"):
            output_ids = ar_generate(model, prompt_ids, gen_len=gen_len, device=device)
        generated_ids = output_ids[len(prompt_ids):]

    generated_text = enc.decode(generated_ids)

    st.subheader("Output")
    st.markdown(f"**Prompt:** {prompt_text}")
    st.markdown("**Continuation:**")
    st.code(generated_text, language=None)

elif generate_clicked and not prompt_text.strip():
    st.warning("Enter a prompt first.")
```

- [ ] **Step 3: Verify syntax**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run python -c "import ast; ast.parse(open('idlm/chat.py').read()); print('syntax ok')"
```
Expected: `syntax ok`

- [ ] **Step 4: Commit**

```bash
git add idlm/chat.py
git commit -m "feat(chat): add rbf_ffn model family with AR generation"
```

---

## Task 4: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/ -v --ignore=idlm/tests/test_train.py 2>&1 | tail -25
```
Expected: all tests PASS (16 in `test_chat_utils.py` + existing suite)

- [ ] **Step 2: Confirm git log**

```bash
git log --oneline -6
```
Expected: 3 new commits on top of the previous chat UI commits.

- [ ] **Step 3: Final commit if any loose changes**

```bash
git status
# commit anything unstaged if needed
```
