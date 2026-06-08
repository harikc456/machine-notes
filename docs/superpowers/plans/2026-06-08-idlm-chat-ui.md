# I-DLM Chat UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit UI (`idlm/chat.py`) for selecting a trained I-DLM checkpoint and generating text interactively via ISD.

**Architecture:** Pure helper functions in `idlm/chat_utils.py` (run discovery, model loading) tested independently; Streamlit UI in `idlm/chat.py` imports those helpers and owns all `st.*` calls. Model lives in `st.session_state` keyed by run directory name. Generation calls `isd_generate()` from the existing `idlm.generate` module with a config copy that reflects slider values.

**Tech Stack:** Streamlit 1.57, PyTorch, tiktoken (GPT-2), existing `idlm.*` and `rbf_ffn.*` modules.

---

## Important Notes

- `metrics.jsonl` has entries of type `"step"` and `"eval"`. There is **no val loss** entry — the closest proxy is the last `"step"` entry's `loss` (training loss). The UI labels this "final loss" (not "val loss") to be accurate.
- The `ar_checkpoint` path in each run's `config.yaml` is **relative to the repo root** (e.g., `rbf_ffn/experiments/.../checkpoint_best.pt`). Resolve it with `Path(repo_root) / cfg.ar_checkpoint`.
- Run with: `streamlit run idlm/chat.py` from the **repo root**.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `idlm/chat_utils.py` | `discover_runs()`, `load_model()` — pure, testable |
| Create | `idlm/chat.py` | Streamlit UI — sidebar, main area, session state |
| Create | `idlm/tests/test_chat_utils.py` | Unit tests for `chat_utils` functions |

---

## Task 1: `discover_runs()` helper + tests

**Files:**
- Create: `idlm/chat_utils.py`
- Create: `idlm/tests/test_chat_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
# idlm/tests/test_chat_utils.py
import json
from pathlib import Path
import pytest
from idlm.chat_utils import discover_runs, RunInfo


def _make_run(root: Path, name: str, steps: list[dict]) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text(
        "ar_checkpoint: rbf_ffn/experiments/some/checkpoint_best.pt\n"
    )
    with open(run_dir / "metrics.jsonl", "w") as f:
        for row in steps:
            f.write(json.dumps(row) + "\n")
    return run_dir


def test_discover_runs_returns_sorted_newest_first(tmp_path):
    _make_run(tmp_path, "20260601_000000_aaa_idlm_r8_s4",
              [{"type": "step", "loss": 3.5}])
    _make_run(tmp_path, "20260607_000000_bbb_idlm_r8_s4",
              [{"type": "step", "loss": 2.1}])
    runs = discover_runs(tmp_path)
    assert len(runs) == 2
    assert runs[0].dir_name == "20260607_000000_bbb_idlm_r8_s4"
    assert runs[1].dir_name == "20260601_000000_aaa_idlm_r8_s4"


def test_discover_runs_extracts_final_loss(tmp_path):
    _make_run(tmp_path, "20260601_000000_aaa_idlm_r8_s4", [
        {"type": "step", "loss": 10.0},
        {"type": "step", "loss": 3.5},
    ])
    runs = discover_runs(tmp_path)
    assert runs[0].final_loss == pytest.approx(3.5)


def test_discover_runs_loss_none_when_no_steps(tmp_path):
    run_dir = tmp_path / "20260601_000000_aaa_idlm_r8_s4"
    run_dir.mkdir()
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text("ar_checkpoint: x.pt\n")
    # no metrics.jsonl
    runs = discover_runs(tmp_path)
    assert runs[0].final_loss is None


def test_discover_runs_skips_dirs_without_checkpoint(tmp_path):
    bad = tmp_path / "20260601_000000_incomplete_idlm"
    bad.mkdir()
    (bad / "config.yaml").write_text("ar_checkpoint: x.pt\n")
    # no checkpoint_best.pt
    runs = discover_runs(tmp_path)
    assert len(runs) == 0


def test_run_info_label_with_loss(tmp_path):
    _make_run(tmp_path, "20260607_113459_031611_idlm_r8_s4",
              [{"type": "step", "loss": 6.682}])
    runs = discover_runs(tmp_path)
    assert "20260607_113459" in runs[0].label
    assert "6.68" in runs[0].label


def test_run_info_label_no_loss(tmp_path):
    run_dir = tmp_path / "20260607_113459_031611_idlm_r8_s4"
    run_dir.mkdir()
    (run_dir / "checkpoint_best.pt").touch()
    (run_dir / "config.yaml").write_text("ar_checkpoint: x.pt\n")
    runs = discover_runs(tmp_path)
    assert "N/A" in runs[0].label
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/test_chat_utils.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'idlm.chat_utils'`

- [ ] **Step 3: Write `idlm/chat_utils.py` with `RunInfo` and `discover_runs()`**

```python
# idlm/chat_utils.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunInfo:
    dir_name: str
    run_dir: Path
    final_loss: float | None

    @property
    def label(self) -> str:
        short = self.dir_name[:15]  # "20260607_113459"
        loss_str = f"{self.final_loss:.4f}" if self.final_loss is not None else "N/A"
        return f"{short} | loss: {loss_str}"


def discover_runs(experiments_dir: Path) -> list[RunInfo]:
    """Scan experiments_dir for valid I-DLM runs, sorted newest-first."""
    runs: list[RunInfo] = []
    for run_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "checkpoint_best.pt").exists():
            continue
        if not (run_dir / "config.yaml").exists():
            continue
        final_loss = _read_final_loss(run_dir / "metrics.jsonl")
        runs.append(RunInfo(
            dir_name=run_dir.name,
            run_dir=run_dir,
            final_loss=final_loss,
        ))
    return runs


def _read_final_loss(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    last_step_loss = None
    try:
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("type") == "step" and "loss" in row:
                last_step_loss = float(row["loss"])
    except (json.JSONDecodeError, OSError):
        return None
    return last_step_loss
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/test_chat_utils.py -v
```
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add idlm/chat_utils.py idlm/tests/test_chat_utils.py
git commit -m "feat(chat): run discovery helper with tests"
```

---

## Task 2: `load_model()` helper + tests

**Files:**
- Modify: `idlm/chat_utils.py`
- Modify: `idlm/tests/test_chat_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `idlm/tests/test_chat_utils.py`:

```python
from idlm.chat_utils import load_model


def test_load_model_raises_on_missing_ar_checkpoint(tmp_path):
    run_dir = tmp_path / "20260607_113459_031611_idlm_r8_s4"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "ar_checkpoint: /nonexistent/path/checkpoint_best.pt\n"
    )
    (run_dir / "checkpoint_best.pt").touch()  # exists but empty
    import torch
    with pytest.raises(Exception):
        load_model(run_dir, repo_root=tmp_path, device=torch.device("cpu"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/test_chat_utils.py::test_load_model_raises_on_missing_ar_checkpoint -v
```
Expected: `ImportError` or `NameError` — `load_model` not defined yet.

- [ ] **Step 3: Add `load_model()` to `idlm/chat_utils.py`**

Append to `idlm/chat_utils.py`:

```python
from __future__ import annotations  # already at top
# add these imports at the top of the file alongside existing ones:
# import torch  (add to top-level imports)
# from idlm.config import load_config
# from idlm.models.idlm_model import IDLMCausalLM
# from rbf_ffn.config import load_config as load_ar_config
# from rbf_ffn.models.model import CausalLM
```

Full updated import block for `chat_utils.py` (replace the existing imports section):

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from idlm.config import IDLMConfig, load_config
from idlm.models.idlm_model import IDLMCausalLM
from rbf_ffn.config import load_config as load_ar_config
from rbf_ffn.models.model import CausalLM
```

Then append this function after `_read_final_loss`:

```python
def load_model(
    run_dir: Path,
    repo_root: Path,
    device: torch.device,
) -> tuple[IDLMCausalLM, IDLMConfig]:
    """Load AR base + LoRA checkpoint from a run directory."""
    cfg = load_config(run_dir / "config.yaml")

    # ar_checkpoint in config is relative to repo root
    ar_ckpt_path = repo_root / cfg.ar_checkpoint
    if not ar_ckpt_path.exists():
        raise FileNotFoundError(f"AR checkpoint not found: {ar_ckpt_path}")

    ar_config_yaml = ar_ckpt_path.parent / "config.yaml"
    ar_cfg = load_ar_config(ar_config_yaml)
    ar_model = CausalLM(ar_cfg).to(device)
    ar_ckpt = torch.load(ar_ckpt_path, map_location=device, weights_only=True)
    ar_model.load_state_dict(ar_ckpt["model"])

    model = IDLMCausalLM(ar_model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules)
    lora_ckpt_path = run_dir / "checkpoint_best.pt"
    lora_ckpt = torch.load(lora_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(lora_ckpt["lora_state"], strict=False)
    model.eval()
    return model, cfg
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/test_chat_utils.py -v
```
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add idlm/chat_utils.py idlm/tests/test_chat_utils.py
git commit -m "feat(chat): model loading helper"
```

---

## Task 3: Streamlit UI — sidebar + model loading

**Files:**
- Create: `idlm/chat.py`

- [ ] **Step 1: Create `idlm/chat.py` with sidebar and model loading**

```python
# idlm/chat.py
"""
I-DLM Chat UI — select a checkpoint and generate text interactively.

Run from repo root:
    streamlit run idlm/chat.py
"""
from __future__ import annotations
import dataclasses
from pathlib import Path

import streamlit as st
import tiktoken
import torch

from idlm.chat_utils import RunInfo, discover_runs, load_model
from idlm.generate import isd_generate

EXPERIMENTS_DIR = Path("idlm/experiments")
REPO_ROOT = Path(".")

st.set_page_config(page_title="I-DLM Chat", layout="wide")
st.title("I-DLM Chat")

# ── Tokenizer (once per process) ──────────────────────────────────────────────
@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


# ── Run discovery (once per process) ──────────────────────────────────────────
@st.cache_resource
def get_runs() -> list[RunInfo]:
    return discover_runs(EXPERIMENTS_DIR)


runs = get_runs()
enc = get_tokenizer()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Checkpoint")

    if not runs:
        st.error("No valid runs found in idlm/experiments/")
        st.stop()

    run_labels = [r.label for r in runs]
    selected_idx = st.selectbox(
        "Run",
        range(len(runs)),
        format_func=lambda i: run_labels[i],
    )
    selected_run = runs[selected_idx]

    # Load defaults from this run's config (best-effort)
    try:
        from idlm.config import load_config as _lc
        _default_cfg = _lc(selected_run.run_dir / "config.yaml")
        default_stride = _default_cfg.stride
        default_gen_len = _default_cfg.gen_len
    except Exception:
        default_stride = 4
        default_gen_len = 128

    stride = st.slider("Stride", min_value=1, max_value=16,
                       value=default_stride, step=1)
    gen_len = st.slider("Gen len", min_value=32, max_value=512,
                        value=default_gen_len, step=32)

    st.divider()

    load_clicked = st.button("Load Model", use_container_width=True)

    if load_clicked:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with st.spinner(f"Loading {selected_run.dir_name[:15]}…"):
            try:
                model, cfg = load_model(selected_run.run_dir, REPO_ROOT, device)
                st.session_state["model"] = model
                st.session_state["model_cfg"] = cfg
                st.session_state["model_key"] = selected_run.dir_name
                st.session_state["device"] = device
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if "model_key" in st.session_state:
        st.success(f"Loaded: {st.session_state['model_key'][:15]}")

# ── Main area ──────────────────────────────────────────────────────────────────
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

    cfg_override = dataclasses.replace(base_cfg, stride=stride, gen_len=gen_len)
    prompt_ids = enc.encode(prompt_text)

    with st.spinner("Generating…"):
        output_ids = isd_generate(model, prompt_ids, cfg_override, device)

    generated_ids = output_ids[len(prompt_ids):]
    generated_text = enc.decode(generated_ids)

    st.subheader("Output")
    st.markdown(f"**Prompt:** {prompt_text}")
    st.markdown("**Continuation:**")
    st.code(generated_text, language=None)

elif generate_clicked and not prompt_text.strip():
    st.warning("Enter a prompt first.")
```

- [ ] **Step 2: Verify the file is importable (no syntax errors)**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run python -c "import ast; ast.parse(open('idlm/chat.py').read()); print('syntax ok')"
```
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add idlm/chat.py
git commit -m "feat(chat): Streamlit UI — sidebar and model loading"
```

---

## Task 4: Smoke test the app

- [ ] **Step 1: Run the full test suite to ensure nothing is broken**

```bash
cd /home/harikrishnan-c/projects/machine-notes
uv run pytest idlm/tests/ -v --ignore=idlm/tests/test_train.py 2>&1 | tail -20
```
Expected: all tests PASS (including the 7 new ones from Tasks 1–2)

- [ ] **Step 2: Launch the Streamlit app**

```bash
cd /home/harikrishnan-c/projects/machine-notes
streamlit run idlm/chat.py
```
Expected: Streamlit starts on `http://localhost:8501`

- [ ] **Step 3: Manual verification checklist**

Open `http://localhost:8501` in a browser and verify:
- [ ] Dropdown shows 12 runs sorted newest-first, each labeled `YYYYMMDD_HHMMSS | loss: X.XXXX`
- [ ] Stride and Gen len sliders are present with correct defaults from the selected run's config
- [ ] Selecting a different run updates the slider defaults
- [ ] Clicking "Load Model" shows a spinner then "Loaded: …" in the sidebar
- [ ] "Generate" button is disabled before a model is loaded
- [ ] After loading, typing a prompt and clicking "Generate" produces output
- [ ] The output section shows the prompt and the generated continuation separately
- [ ] Changing stride/gen_len sliders and regenerating works without reloading the model

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(idlm): chat UI for checkpoint selection and ISD generation"
```
