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


def _make_cache(config: QuantConfig, n_kv_heads: int, head_dim: int, cal_data, device):
    if config.method == "turboquant":
        from kv_quant.turboquant import TurboQuantCache
        return TurboQuantCache(config, n_kv_heads, head_dim, device=device)
    if config.method == "spectralquant":
        from kv_quant.spectralquant import SpectralQuantCache
        return SpectralQuantCache(config, cal_data)
    raise ValueError(f"Unknown method: {config.method!r}")


def wrap(model, config: QuantConfig):
    """Patch model.generate() to use a quantized KV cache.

    For spectralquant, config.calibration_path must be a base path (no extension)
    pointing to files produced by `python -m kv_quant.calibrate`.
    """
    if config.method == "spectralquant":
        if not config.calibration_path:
            raise ValueError("spectralquant requires config.calibration_path")
        cal_data = _load_spectralquant_cal(config.calibration_path)
    else:
        cal_data = None

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
