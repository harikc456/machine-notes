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

    _key_bits_opts = [1, 2, 3, 4]
    key_bits = st.selectbox(
        "Key bits",
        _key_bits_opts,
        index=_key_bits_opts.index(entry.default_bits) if entry.default_bits in _key_bits_opts else 3,
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
        current_quant = QuantConfig(
            method="turboquant",
            bits=key_bits,
            value_bits=value_bits,
            buffer_size=buffer_size,
        )
        if (
            loaded_id != entry.id
            or loaded_mode != current_mode
            or st.session_state.get("quant_config") != current_quant
        ):
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

    n_kv_heads, head_dim = get_kv_shape(model)
    cache: TurboQuantCache | None = None
    if current_mode == "turboquant" and quant_config is not None:
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
    seq_len = input_ids.shape[-1] + n_new

    buf_size = quant_config.buffer_size if cache is not None else 0
    st.session_state["last_stats"] = get_stats(
        cache=cache,
        n_new_tokens=n_new,
        elapsed=elapsed,
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        buffer_size=buf_size,
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
    col2.metric("Tokens/sec (e2e)", f"{stats.tokens_per_sec:.1f} tok/s")
    col3.metric("Compression", f"{stats.compression_ratio:.1f}×")
