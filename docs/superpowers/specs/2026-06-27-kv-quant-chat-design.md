# KV-Quant Chat UI Design

**Goal:** A standalone Streamlit app to interactively try TurboQuant KV cache quantization on HuggingFace models with live compression stats.

**Architecture:** Separate Streamlit app (`chat/kv_quant_chat.py`) using `kv_quant.wrap()` to patch `model.generate()`. Model list configured via `chat/models.yaml`. Stats computed from `cache.compressed_bytes()` and wall-clock timing.

**Tech Stack:** Streamlit, transformers, PyTorch, PyYAML, `kv_quant` (local package)

---

## Files

- Create: `chat/kv_quant_chat.py` — Streamlit app entry point
- Create: `chat/kv_quant_utils.py` — `load_hf_model()`, `get_stats()`, stats dataclass
- Create: `chat/models.yaml` — configurable model list

Run with: `streamlit run chat/kv_quant_chat.py`

---

## models.yaml Format

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

Fields:
- `id`: HuggingFace model ID (used for `AutoModelForCausalLM.from_pretrained`)
- `label`: display name in sidebar selector
- `head_dim`: KV head dimension; must be 64 or 128 to match precomputed Lloyd-Max codebooks
- `default_bits`: pre-fills the key bits selector when this model is chosen

Note: Gemma4 uses `head_dim: 128` (not 256) because Lloyd-Max codebooks are precomputed only for d∈{64, 128}. If head_dim is 256, the app falls back to 128 or shows a warning.

---

## Sidebar Controls

```
┌─ Sidebar ────────────────────────┐
│ Model                            │
│   [Qwen3-8B-Instruct       ▾]   │
│                                  │
│ Cache Mode                       │
│   ○ Baseline (full precision)    │
│   ● TurboQuant                   │
│                                  │
│ Key bits       [4  ▾]            │
│ Value bits     [2  ▾]            │
│ Buffer size    [128 ▾]           │
│                                  │
│ Max new tokens [256 ────────]    │
│                                  │
│ [Load Model]                     │
└──────────────────────────────────┘
```

- **Model selector**: populated from `models.yaml`; changing model resets `default_bits`
- **Cache Mode**: radio between "Baseline" and "TurboQuant"; quant controls disabled in baseline mode
- **Key bits**: selectbox `[1, 2, 3, 4]`, default from `models.yaml`
- **Value bits**: selectbox `[1, 2]`, default 2
- **Buffer size**: selectbox `[64, 128, 256]`, default 128
- **Max new tokens**: slider 32–512, step 32, default 256
- **Load Model**: triggers `@st.cache_resource` model load; switching mode or model requires reload; clears chat history

---

## Chat Interaction Flow

Conversation stored in `st.session_state["history"]` as `list[{"role": str, "content": str}]`.

Per turn:
1. User submits via `st.chat_input`
2. Full history rendered with `st.chat_message`
3. Prompt built using `tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True)`
4. Generation via `model.generate()` — TurboQuant wraps this transparently via `kv_quant.wrap()`
5. Response streamed token-by-token with `TextIteratorStreamer` + `st.write_stream`
6. Stats updated after generation completes

**Clear Chat** button: resets `st.session_state["history"]`, creates fresh KV cache.

Switching model or mode also clears history (new model loaded on next "Load Model" click).

---

## Stats Panel

Three `st.metric` columns displayed below the chat, updated after each generation:

```
┌──────────────────┬──────────────────┬──────────────────┐
│  KV Memory       │  Tokens/sec      │  Compression     │
│  142 MB          │  38.2 tok/s      │  6.1×            │
│  ↓ from 891 MB   │                  │  (baseline: 1×)  │
└──────────────────┴──────────────────┴──────────────────┘
```

Calculations:
- **KV Memory (quantized)**: `cache.compressed_bytes() / 1e6` MB
- **KV Memory (baseline)**: `n_layers × n_kv_heads × seq_len × head_dim × 2 × sizeof(fp16)` bytes
- **Tokens/sec**: `n_new_tokens / elapsed_wall_time` (wall clock around `model.generate()`)
- **Compression**: `baseline_bytes / compressed_bytes`; shows `1×` in baseline mode

Stats hidden until first generation. In baseline mode, KV Memory shows the full-precision estimate and Compression shows `1×`.

---

## kv_quant_utils.py Interface

```python
@dataclass
class ChatStats:
    kv_memory_mb: float
    baseline_memory_mb: float
    tokens_per_sec: float
    compression_ratio: float

def load_hf_model(
    model_id: str,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model + tokenizer with bfloat16, device_map=device."""

def get_stats(
    cache,          # TurboQuantCache or None for baseline
    n_new_tokens: int,
    elapsed: float,
    n_layers: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
) -> ChatStats:
    """Compute stats from cache state and timing."""
```

---

## Session State Keys

| Key | Type | Description |
|-----|------|-------------|
| `model` | `AutoModelForCausalLM` | loaded model |
| `tokenizer` | `AutoTokenizer` | loaded tokenizer |
| `model_id` | `str` | currently loaded model ID |
| `mode` | `str` | `"baseline"` or `"turboquant"` |
| `history` | `list[dict]` | chat turns |
| `last_stats` | `ChatStats \| None` | stats from last generation |
| `quant_config` | `QuantConfig` | active quant settings |
