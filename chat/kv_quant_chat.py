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
from transformers.generation.streamers import TextIteratorStreamer

from chat.kv_quant_utils import (
    ChatStats,
    ModelEntry,
    get_kv_shape,
    get_stats,
    load_hf_model,
    load_models_yaml,
)
from kv_quant import _apply_triattention_standalone, _load_spectralquant_cal
from kv_quant.config import QuantConfig
from kv_quant.spectralquant import SpectralQuantCache
from kv_quant.triattention_patch import apply_combined_eviction_patch
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
        ["Baseline (full precision)", "TurboQuant", "SpectralQuant"],
        key="cache_mode",
        label_visibility="collapsed",
    )
    use_turboquant = mode_label == "TurboQuant"
    use_spectralquant = mode_label == "SpectralQuant"
    use_quant = use_turboquant or use_spectralquant
    turboquant_disabled = not use_turboquant

    _key_bits_opts = [1, 2, 3, 4]
    key_bits = st.selectbox(
        "Key bits",
        _key_bits_opts,
        index=_key_bits_opts.index(entry.default_bits) if entry.default_bits in _key_bits_opts else 3,
        key="key_bits",
        disabled=turboquant_disabled,
    )
    value_bits = st.selectbox(
        "Value bits",
        [1, 2],
        index=1,
        key="value_bits",
        disabled=turboquant_disabled,
    )
    buffer_size = st.selectbox(
        "Buffer size",
        [16, 32, 64, 128, 256],
        index=1,
        key="buffer_size",
        disabled=turboquant_disabled,
    )
    cal_path = st.text_input(
        "Calibration path (no extension)",
        key="cal_path",
        disabled=not use_spectralquant,
        placeholder="e.g. cal/llama3-8b",
    )

    st.subheader("Token Eviction")
    use_triattention = st.checkbox("TriAttention eviction", key="use_triattention")
    ta_stats_path = st.text_input(
        "TriAttention stats path (.pt file)",
        key="ta_stats_path",
        disabled=not use_triattention,
        placeholder="e.g. triattention/stats/llama3-8b.pt",
    )
    ta_budget = st.number_input(
        "Budget (max cached tokens)",
        min_value=64,
        max_value=8192,
        value=2048,
        step=64,
        key="ta_budget",
        disabled=not use_triattention,
    )
    ta_divide_length = st.number_input(
        "Eviction interval (decode steps)",
        min_value=1,
        max_value=512,
        value=128,
        step=1,
        key="ta_divide_length",
        disabled=not use_triattention,
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
                if use_spectralquant and not cal_path:
                    st.error("SpectralQuant requires a calibration path.")
                    st.stop()
                model, tokenizer = load_hf_model(entry.id, device)
                st.session_state["model"] = model
                st.session_state["tokenizer"] = tokenizer
                st.session_state["model_id"] = entry.id
                if use_turboquant:
                    mode = "turboquant"
                    quant_cfg = QuantConfig(
                        method="turboquant",
                        bits=key_bits,
                        value_bits=value_bits,
                        buffer_size=buffer_size,
                    )
                    cal_data = None
                elif use_spectralquant:
                    mode = "spectralquant"
                    quant_cfg = QuantConfig(
                        method="spectralquant",
                        bits=key_bits,
                        calibration_path=cal_path,
                    )
                    cal_data = _load_spectralquant_cal(cal_path)
                else:
                    mode = "baseline"
                    quant_cfg = None
                    cal_data = None
                # Apply TriAttention eviction patch if requested
                ta_cfg = None
                if use_triattention:
                    if not ta_stats_path:
                        st.error("TriAttention eviction requires a stats .pt file path.")
                        st.stop()
                    kw = dict(
                        eviction="triattention",
                        calibration_path=ta_stats_path,
                        budget=int(ta_budget),
                        divide_length=int(ta_divide_length),
                    )
                    if quant_cfg is not None:
                        ta_cfg = QuantConfig(method=quant_cfg.method, **kw,
                                             bits=quant_cfg.bits,
                                             value_bits=quant_cfg.value_bits,
                                             buffer_size=quant_cfg.buffer_size)
                        apply_combined_eviction_patch(model, ta_cfg)
                    else:
                        ta_cfg = QuantConfig(method=None, **kw)
                        _apply_triattention_standalone(model, ta_cfg)
                st.session_state["mode"] = mode
                st.session_state["quant_config"] = quant_cfg
                st.session_state["cal_data"] = cal_data
                st.session_state["ta_cfg"] = ta_cfg
                st.session_state["history"] = []
                st.session_state["last_stats"] = None
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if "model_id" in st.session_state:
        loaded_id: str = st.session_state["model_id"]
        loaded_mode: str = st.session_state.get("mode", "baseline")
        if use_turboquant:
            current_mode = "turboquant"
        elif use_spectralquant:
            current_mode = "spectralquant"
        else:
            current_mode = "baseline"
        st.success(f"Loaded: {loaded_id.split('/')[-1]}")
        if use_turboquant:
            current_quant = QuantConfig(
                method="turboquant",
                bits=key_bits,
                value_bits=value_bits,
                buffer_size=buffer_size,
            )
        elif use_spectralquant:
            current_quant = QuantConfig(
                method="spectralquant",
                bits=key_bits,
                calibration_path=cal_path,
            )
        else:
            current_quant = None
        loaded_ta: QuantConfig | None = st.session_state.get("ta_cfg")
        ta_changed = bool(use_triattention) != (loaded_ta is not None) or (
            use_triattention and loaded_ta is not None and (
                ta_stats_path != loaded_ta.calibration_path
                or int(ta_budget) != loaded_ta.budget
                or int(ta_divide_length) != loaded_ta.divide_length
            )
        )
        spectralquant_changed = (
            current_mode == "spectralquant"
            and st.session_state.get("quant_config") != current_quant
        )
        if loaded_id != entry.id or spectralquant_changed or ta_changed:
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
    _mode_label = st.session_state.get("cache_mode", "Baseline (full precision)")
    quant_config: QuantConfig
    if _mode_label == "TurboQuant":
        current_mode = "turboquant"
        quant_config = QuantConfig(
            method="turboquant",
            bits=st.session_state.get("key_bits", 4),
            value_bits=st.session_state.get("value_bits", 2),
            buffer_size=st.session_state.get("buffer_size", 32),
        )
    elif _mode_label == "SpectralQuant":
        current_mode = "spectralquant"
        quant_config = st.session_state.get("quant_config")
    else:
        current_mode = "baseline"
        quant_config = None
    _max_new: int = st.session_state.get("max_new_tokens", 256)

    history: list[dict] = st.session_state.get("history", [])
    history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    _chat_enc = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return a BatchEncoding or a raw tensor depending
    # on the transformers version; extract the tensor in either case.
    input_ids = (_chat_enc["input_ids"] if hasattr(_chat_enc, "keys") else _chat_enc).to(model.device)

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
    cache: TurboQuantCache | SpectralQuantCache | None = None
    if current_mode == "turboquant" and quant_config is not None:
        cache = TurboQuantCache(
            quant_config, n_kv_heads, head_dim, device=model.device
        )
        gen_kwargs["past_key_values"] = cache
    elif current_mode == "spectralquant" and quant_config is not None:
        cal_data = st.session_state.get("cal_data")
        cache = SpectralQuantCache(quant_config, cal_data)
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

    buf_size = (
        quant_config.buffer_size
        if isinstance(cache, TurboQuantCache)
        else 0
    )
    st.session_state["last_stats"] = get_stats(
        cache=cache,
        n_new_tokens=n_new,
        elapsed=elapsed,
        n_layers=getattr(model.config, "num_hidden_layers", None) or model.config.text_config.num_hidden_layers,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        buffer_size=buf_size,
    )

# ── Stats panel ───────────────────────────────────────────────────────────────
stats: ChatStats | None = st.session_state.get("last_stats")
if stats is not None:
    st.divider()
    _ta_cfg: QuantConfig | None = st.session_state.get("ta_cfg")
    col1, col2, col3, col4 = st.columns(4)
    delta = (
        None
        if stats.compression_ratio == 1.0
        else f"↓ from {stats.baseline_memory_mb:.1f} MB"
    )
    col1.metric("KV Memory", f"{stats.kv_memory_mb:.1f} MB", delta=delta)
    col2.metric("Tokens/sec (e2e)", f"{stats.tokens_per_sec:.1f} tok/s")
    col3.metric("Compression", f"{stats.compression_ratio:.1f}×")
    if _ta_cfg is not None:
        col4.metric("Eviction budget", f"{_ta_cfg.budget} tok")
    if stats.compression_ratio == 1.0 and st.session_state.get("mode", "baseline") != "baseline":
        _buf = st.session_state.get("quant_config")
        _buf_hint = f" (buffer size: {_buf.buffer_size} tokens)" if _buf is not None and hasattr(_buf, "buffer_size") else ""
        st.caption(f"No compression yet — sequence is shorter than the recency buffer{_buf_hint}. "
                   "Generate a longer response or reduce the buffer size to see compression.")
