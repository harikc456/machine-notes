"""Combined quantization + TriAttention eviction forward patch.

apply_combined_eviction_patch() patches model.forward to apply TriAttention
token eviction on top of any quantized KV cache that exposes get_kv(layer_idx)
and evict(keep_indices). Use this instead of the official apply_triattention_patch()
when a quantized cache (TurboQuantCache, SpectralQuantCache) is being injected
by wrap() — the official patch only works on plain DynamicCache.
"""
from __future__ import annotations
import os
import sys
import types
from pathlib import Path

import torch

from kv_quant.config import QuantConfig


def _ensure_triattention_on_path() -> None:
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "triattention"))
    if src not in sys.path:
        sys.path.insert(0, src)


def apply_combined_eviction_patch(model, config: QuantConfig) -> None:
    """Patch model.forward to evict tokens from a quantized KV cache.

    Preconditions (wrap() guarantees these before calling):
      - config.calibration_path is a valid path to a TriAttention stats .pt file
      - model.config._name_or_path is set to a HF model ID or local path
      - config.method is not None (a quantized cache with evict() will be injected)

    After patching:
      - model._triattention_compressor holds the TriAttention instance
      - model.forward intercepts decode steps to track positions and trigger eviction
    """
    _ensure_triattention_on_path()
    from triattention.methods.triattention import TriAttention, TriAttentionConfig

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_layers = model.config.num_hidden_layers

    comp = TriAttention(TriAttentionConfig(
        stats_path=Path(config.calibration_path),
        model_path=Path(model.config._name_or_path),
        device=device,
        dtype=dtype,
        budget=config.budget,
        divide_length=config.divide_length,
    ))
    model._triattention_compressor = comp

    _orig_forward = model.forward

    def _patched_forward(self_model, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        past_kv = kwargs.get("past_key_values")

        if past_kv is None or not hasattr(past_kv, "evict"):
            return _orig_forward(*args, **kwargs)

        seq_len = input_ids.shape[-1] if input_ids is not None else 1
        cached_len = past_kv.get_seq_length()

        # Prefill: cache is empty on the first call
        if cached_len == 0:
            output = _orig_forward(*args, **kwargs)
            filled = past_kv.get_seq_length()
            comp.cache_positions = list(range(filled))
            comp.absolute_position = filled
            comp.prefix_length = filled
            return output

        # Decode: override position_ids so the new token(s) have the correct
        # absolute position (after eviction, cached_len == budget but
        # comp.absolute_position > budget, so HF's default would be wrong)
        tok_device = input_ids.device if input_ids is not None else device
        kwargs["position_ids"] = torch.arange(
            comp.absolute_position,
            comp.absolute_position + seq_len,
            device=tok_device,
        ).unsqueeze(0)

        output = _orig_forward(*args, **kwargs)

        comp.cache_positions.extend(
            range(comp.absolute_position, comp.absolute_position + seq_len)
        )
        comp.absolute_position += seq_len

        # Trigger eviction if above budget at the right interval
        new_seq_len = past_kv.get_seq_length()
        if (
            new_seq_len > config.budget
            and comp.absolute_position % config.divide_length == 0
        ):
            kv_pairs = [past_kv.get_kv(l) for l in range(n_layers)]
            proxy = types.SimpleNamespace(
                key_cache=[k for k, _v in kv_pairs],
                value_cache=[_v for _k, _v in kv_pairs],
            )
            keep_indices = comp.compute_keep_indices(
                proxy, prefix_length=getattr(comp, "prefix_length", 0)
            )
            past_kv.evict(keep_indices)
            comp.cache_positions = [comp.cache_positions[i] for i in keep_indices.tolist()]

        return output

    model.forward = types.MethodType(_patched_forward, model)
