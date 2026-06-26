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
