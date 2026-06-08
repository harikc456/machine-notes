# I-DLM Chat UI — Design Spec

**Date:** 2026-06-08  
**Status:** Approved

## Overview

A Streamlit-based interactive UI (`idlm/chat.py`) for selecting a trained I-DLM checkpoint and generating text from a user-supplied prompt using Introspective Strided Decoding (ISD).

## Goals

- Browse all trained runs in `idlm/experiments/` with their final val loss
- Load any run's `checkpoint_best.pt` on demand
- Override `stride` and `gen_len` at runtime via sliders
- Type a prompt and see the generated continuation

## Out of Scope

- Multi-turn chat / conversation history
- Batch evaluation
- Comparing multiple runs side-by-side
- `checkpoint_best_mask.pt` (always use `checkpoint_best.pt`)

---

## Architecture

Single file: `idlm/chat.py`  
Run with: `streamlit run idlm/chat.py` from the repo root.

Imports directly from existing modules — no new modules created:
- `idlm.config.load_config` — parse each run's `config.yaml`
- `idlm.generate.isd_generate` — generation function
- `idlm.models.idlm_model.IDLMCausalLM` — LoRA-wrapped model
- `rbf_ffn.config.load_config` / `rbf_ffn.models.model.CausalLM` — AR base

---

## Run Discovery

On app startup, scan `idlm/experiments/*/`:
- Include a directory only if both `checkpoint_best.pt` and `config.yaml` exist
- Read `metrics.jsonl`; extract the last entry where `"type": "epoch"` for the final `loss`
- Format each entry as: `20260607_113459 | loss: 2.341` (first 15 chars of dir name + final epoch loss)
- Sort newest-first by directory name (timestamp prefix)

---

## Sidebar

| Control | Type | Range / Default |
|---|---|---|
| Run selector | Selectbox | All discovered runs, newest first |
| Stride | Slider (int) | 1–16, default from run's `config.yaml` |
| Gen len | Slider (int) | 32–512, step 32, default from run's `config.yaml` |
| Load Model | Button | Loads AR base + LoRA into session state |

"Load Model" shows a spinner while loading. Displays "Model loaded ✓" once done.

---

## Main Area

- **Prompt text area** — multiline input, placeholder: `"Enter a prompt to continue…"`
- **Generate button** — disabled until a model is loaded
- **Output** — generated text displayed below the button; prompt shown in normal weight, generated continuation shown in a styled block (e.g., `st.code` or a highlighted `st.markdown`)

---

## State Management

Model stored in `st.session_state["model"]` and `st.session_state["model_key"]`.  
`model_key = (run_dir_name, "checkpoint_best.pt")`.

- Clicking "Load Model" always reloads regardless of current key
- Stride and gen_len changes take effect immediately at generation time — no reload needed
- Tokenizer (tiktoken GPT-2) instantiated once at module load, not per generation

---

## Generation Flow

1. Encode prompt with tiktoken `gpt2` encoding → `prompt_ids: list[int]`
2. Call `isd_generate(model, prompt_ids, cfg_override, device)` where `cfg_override` has `stride` and `gen_len` from the sliders (other fields from loaded config)
3. Decode full output token list → string
4. Display: prompt text + generated continuation

---

## Error Handling

- If `metrics.jsonl` is missing or has no epoch entries, show `loss: N/A` in the dropdown
- If the AR base checkpoint path in `config.yaml` is relative, resolve it relative to the repo root
- Show `st.error(...)` if model loading fails (bad checkpoint, missing AR base, etc.)
- Disable Generate button with tooltip "Load a model first" if no model in session state
