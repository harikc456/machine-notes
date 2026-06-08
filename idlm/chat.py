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

    def _reset_run_selector():
        st.session_state["run_selector"] = 0

    model_family = st.radio(
        "Model family",
        ["I-DLM (ISD)", "rbf_ffn (AR)"],
        key="model_family",
        on_change=_reset_run_selector,
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
        loaded_family = st.session_state.get("model_type", "idlm")
        current_family = "idlm" if is_idlm else "rbf"
        if st.session_state["model_key"] != selected_run.dir_name or loaded_family != current_family:
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
    loaded_type = st.session_state.get("model_type", "idlm")
    _stride = st.session_state.get("stride_slider", 4)
    _gen_len = st.session_state.get("gen_len_slider", 128)

    if loaded_type == "idlm":
        enc = get_idlm_tokenizer()
        cfg_override = dataclasses.replace(base_cfg, stride=_stride, gen_len=_gen_len)
        prompt_ids = enc.encode(prompt_text)
        with st.spinner("Generating (ISD)…"):
            output_ids = isd_generate(model, prompt_ids, cfg_override, device)
        _MASK_ID = 50256
        generated_ids = [t for t in output_ids[len(prompt_ids):] if 0 <= t < _MASK_ID]
    else:
        enc = get_rbf_tokenizer()
        prompt_ids = enc.encode(prompt_text)
        with st.spinner("Generating (AR)…"):
            output_ids = ar_generate(model, prompt_ids, gen_len=_gen_len, device=device)
        generated_ids = output_ids[len(prompt_ids):]

    generated_text = enc.decode(generated_ids)

    st.subheader("Output")
    st.markdown(f"**Prompt:** {prompt_text}")
    st.markdown("**Continuation:**")
    st.code(generated_text, language=None)

elif generate_clicked and not prompt_text.strip():
    st.warning("Enter a prompt first.")
