from __future__ import annotations
import torch


def _cache_bytes(cache) -> int:
    """Return actual bytes held by a KV cache object, or 0 if unrecognised."""
    if cache is None:
        return 0

    from kv_quant.turboquant import TurboQuantCache
    from kv_quant.spectralquant import SpectralQuantCache

    if isinstance(cache, TurboQuantCache):
        # compressed_bytes() accounts for actual quantized tensor sizes (uint8 / float32)
        # but excludes the fp16 recency buffer — add that separately.
        total = cache.compressed_bytes()
        for buf in (cache._k_buf, cache._v_buf):
            for t in buf:
                if t is not None:
                    total += t.nelement() * t.element_size()
        return total

    if isinstance(cache, SpectralQuantCache):
        # compressed_bytes() gives theoretical packed bits; sum actual index tensors instead.
        total = 0
        for store in (cache._sk_sem, cache._sk_tail, cache._sv_sem, cache._sv_tail):
            for heads in store:
                for t in heads:
                    if t is not None:
                        total += t.nelement() * t.element_size()
        return total

    # PlainDynamicCache (wrap() with method=None) and anything else with compressed_bytes()
    if hasattr(cache, "compressed_bytes"):
        return cache.compressed_bytes()

    # transformers>=4.56 Cache refactor: per-layer objects with .keys/.values tensors
    # (key_cache/value_cache lists were removed entirely, not just deprecated).
    if hasattr(cache, "layers"):
        total = sum(
            t.nelement() * t.element_size()
            for layer in cache.layers
            for t in (getattr(layer, "keys", None), getattr(layer, "values", None))
            if isinstance(t, torch.Tensor) and t.nelement() > 0
        )
        if total > 0:
            return total

    # Standard HF DynamicCache / HybridCache (fp16 baseline) — pre-4.56 API.
    if hasattr(cache, "key_cache"):
        total = sum(
            t.nelement() * t.element_size()
            for tensors in (cache.key_cache, cache.value_cache)
            for t in tensors
            if isinstance(t, torch.Tensor) and t.nelement() > 0
        )
        # HybridCache (Gemma 3/4) stores sliding-window layers in a nested cache.
        for attr in ("_sliding_window_cache", "sliding_window_cache"):
            swc = getattr(cache, attr, None)
            if swc is not None and hasattr(swc, "key_cache"):
                total += sum(
                    t.nelement() * t.element_size()
                    for tensors in (swc.key_cache, swc.value_cache)
                    for t in tensors
                    if isinstance(t, torch.Tensor) and t.nelement() > 0
                )
        if total > 0:
            return total

    # Legacy tuple-based cache
    if isinstance(cache, (list, tuple)):
        return sum(
            t.nelement() * t.element_size()
            for layer in cache
            for t in layer
            if isinstance(t, torch.Tensor)
        )

    # Generic fallback: walk all tensor-valued attributes (catches HybridCache variants,
    # StaticCache, and any future cache types that don't fit the patterns above).
    total = 0
    for val in vars(cache).values():
        if isinstance(val, torch.Tensor) and val.nelement() > 0:
            total += val.nelement() * val.element_size()
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, torch.Tensor) and item.nelement() > 0:
                    total += item.nelement() * item.element_size()
    return total


_LONG_PROMPT = (
    "The history of artificial intelligence dates back to the 1950s when Alan Turing proposed "
    "the Turing Test as a measure of machine intelligence. Since then, the field has evolved "
    "dramatically through several waves of progress and setbacks known as AI winters. "
    "Modern deep learning has transformed the field, enabling breakthroughs in computer vision, "
    "natural language processing, and reinforcement learning. Large language models have emerged "
    "as particularly powerful tools, capable of generating coherent text, answering questions, "
    "writing code, and engaging in complex reasoning tasks. The key innovation behind these models "
    "is the transformer architecture with its attention mechanism, which allows the model to "
    "relate distant tokens in a sequence. Training these models requires vast amounts of data and "
    "compute, leading to significant investment from technology companies worldwide. "
)


def measure_kv_memory(
    model,
    tokenizer,
    prompt: str = _LONG_PROMPT,
    max_new_tokens: int = 256,
) -> dict:
    """Measure KV cache memory by inspecting the cache object after generation.

    Returns dict with keys:
      peak_bytes        — bytes held in the KV cache after generate()
      fp16_est_bytes    — estimated fp16 KV cache size for same token count
      compression_ratio — fp16_est_bytes / peak_bytes
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    cache = out.past_key_values

    # Always count bytes actually held in the cache object — the CUDA allocation
    # delta (_mem_after - _mem_before) captures activations and temporary buffers
    # too, making compression_ratio meaningless (consistently ~0.15 for both fp16
    # and quantized runs).
    peak_bytes = max(_cache_bytes(cache), 1)

    # Use actual sequence length from cache so fp16_est_bytes reflects real tokens generated.
    if cache is not None and hasattr(cache, "get_seq_length"):
        total_seq = cache.get_seq_length()
    else:
        total_seq = out.sequences.shape[1]  # fallback: output token count

    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim   = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_layers   = cfg.num_hidden_layers

    # Only count layers that actually store KV cache entries.
    # layer_types excludes non-attention layers; num_kv_shared_layers counts layers that
    # reuse KV from a prior layer and write nothing themselves (Gemma4).
    _ATTN_TYPES = {"sliding_attention", "full_attention", "attention", "self_attention"}
    if hasattr(cfg, "layer_types"):
        n_kv_layers = sum(1 for lt in cfg.layer_types if lt in _ATTN_TYPES)
    else:
        n_kv_layers = n_layers
    n_kv_layers -= getattr(cfg, "num_kv_shared_layers", 0)
    n_kv_layers  = max(n_kv_layers, 1)

    fp16_bytes = n_kv_layers * 2 * n_kv_heads * head_dim * total_seq * 2  # fp16 = 2 bytes

    return {
        "peak_bytes": peak_bytes,
        "fp16_est_bytes": fp16_bytes,
        "compression_ratio": fp16_bytes / peak_bytes,
    }
