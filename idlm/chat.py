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
