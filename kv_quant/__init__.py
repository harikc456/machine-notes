from __future__ import annotations
import os
import sys
import torch

from kv_quant.config import QuantConfig


def _get_kv_shape(model) -> tuple[int, int]:
    """Extract (n_kv_heads, head_dim) from a HF model config."""
    cfg = model.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
    )
    return n_kv_heads, head_dim


def _ensure_spectralquant_on_path() -> None:
    src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
    )
    if src not in sys.path:
        sys.path.insert(0, src)


def _ensure_triattention_on_path() -> None:
    src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "triattention")
    )
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_spectralquant_cal(base_path: str) -> tuple:
    """Load (EigenspectralCalibrator, quant_state_dict) from base_path."""
    _ensure_spectralquant_on_path()
    from spectralquant.calibration import EigenspectralCalibrator
    calibrator = EigenspectralCalibrator()
    calibrator.load(base_path)
    quant_state = torch.load(
        base_path + "_quantizers.pt", map_location="cpu", weights_only=True
    )
    return (calibrator, quant_state)


def _make_plain_cache():
    """Return a cache-like object with key_cache, value_cache, and compressed_bytes() method."""
    class PlainDynamicCache:
        def __init__(self):
            self.key_cache = []
            self.value_cache = []

        def compressed_bytes(self):
            return sum(
                kc.nelement() * kc.element_size() + vc.nelement() * vc.element_size()
                for kc, vc in zip(self.key_cache, self.value_cache)
                if kc is not None
            )

    return PlainDynamicCache()


def _apply_triattention_standalone(model, config: QuantConfig) -> None:
    """Apply the official TriAttention patch for standalone eviction (no quantization).

    Patches model.forward via apply_triattention_patch, which handles position
    tracking and eviction on the standard DynamicCache injected by wrap().
    """
    from pathlib import Path
    _ensure_triattention_on_path()
    from triattention.methods.triattention import apply_triattention_patch

    apply_triattention_patch(
        model,
        stats_path=Path(config.calibration_path),
        model_path=Path(model.config._name_or_path),
        kv_budget=config.budget,
        divide_length=config.divide_length,
    )


def _make_cache(config: QuantConfig, n_kv_heads: int, head_dim: int, cal_data, device):
    if config.method == "turboquant":
        from kv_quant.turboquant import TurboQuantCache
        return TurboQuantCache(config, n_kv_heads, head_dim, device=device)
    if config.method == "spectralquant":
        from kv_quant.spectralquant import SpectralQuantCache
        return SpectralQuantCache(config, cal_data)
    if config.method is None:
        return _make_plain_cache()
    raise ValueError(f"Unknown method: {config.method!r}")


def wrap(model, config: QuantConfig):
    """Patch model.generate() to inject a compressed/evicting KV cache.

    method controls quantization (None = no quantization):
      "turboquant"    — TurboQuant key/value quantization
      "spectralquant" — SpectralQuant per-head Lloyd-Max (requires calibration_path)
      None            — plain DynamicCache (no quantization)

    eviction controls token eviction applied on top of quantization:
      "triattention"  — TriAttention eviction (requires calibration_path + _name_or_path)
      None            — no eviction

    config.calibration_path serves two roles:
      spectralquant: base path (no extension) for files from kv_quant.calibrate
      triattention:  path to a stats .pt file from triattention/triattention/vllm/stats/
    """
    if config.method == "spectralquant":
        if not config.calibration_path:
            raise ValueError("spectralquant requires config.calibration_path")
        cal_data = _load_spectralquant_cal(config.calibration_path)
    else:
        cal_data = None

    if config.eviction == "triattention":
        if not config.calibration_path:
            raise ValueError(
                "triattention eviction requires config.calibration_path "
                "(path to a stats .pt file from triattention/triattention/vllm/stats/)"
            )
        model_path = getattr(model.config, "_name_or_path", None)
        if not model_path:
            raise ValueError(
                "triattention requires model.config._name_or_path to be set. "
                "Load the model via AutoModelForCausalLM.from_pretrained."
            )
        if config.method is None:
            _apply_triattention_standalone(model, config)
        else:
            # Combined: Task 4 wires in apply_combined_eviction_patch here
            from kv_quant.triattention_patch import apply_combined_eviction_patch
            apply_combined_eviction_patch(model, config)

    n_kv_heads, head_dim = _get_kv_shape(model)
    device = next(model.parameters()).device

    _orig_generate = model.generate

    def _wrapped_generate(*args, **kwargs):
        if "past_key_values" not in kwargs:
            kwargs["past_key_values"] = _make_cache(config, n_kv_heads, head_dim, cal_data, device)
        return _orig_generate(*args, **kwargs)

    model.generate = _wrapped_generate
    model._kv_quant_config = config
    model._make_kv_cache = lambda: _make_cache(config, n_kv_heads, head_dim, cal_data, device)
    return model
