from __future__ import annotations
import os
import json
import torch

_CODEBOOK_CACHE: dict[tuple[int, int], dict] = {}
_CODEBOOK_DIR = os.path.join(os.path.dirname(__file__), "codebooks")


def get_codebook(d: int, bits: int) -> dict:
    """Load precomputed Lloyd-Max codebook for the Beta distribution on [-1,1].

    Codebooks are precomputed for d in {64, 128} and bits in {1, 2, 3, 4}.
    """
    key = (d, bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]
    path = os.path.join(_CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No precomputed codebook for d={d}, bits={bits}. "
            f"Available: d in {{64, 128}}, bits in {{1, 2, 3, 4}}."
        )
    with open(path) as f:
        cb = json.load(f)
    _CODEBOOK_CACHE[key] = cb
    return cb


def get_codebook_tensors(
    d: int, bits: int, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids, decision_boundaries) as tensors ready for quantization.

    centroids: (2^bits,) — optimal reconstruction values
    decision_boundaries: (2^bits - 1,) — thresholds for searchsorted
    """
    cb = get_codebook(d, bits)
    centroids = torch.tensor(cb["centroids"], device=device, dtype=dtype)
    boundaries = torch.tensor(cb["boundaries"], device=device, dtype=dtype)
    decision_boundaries = boundaries[1:-1].contiguous()
    return centroids, decision_boundaries
