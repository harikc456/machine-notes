from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import NamedTuple

from kv_quant.ops.codebook import get_codebook_tensors
from kv_quant.ops.rotation import make_rotation


# ---------------------------------------------------------------------------
# Named tuple outputs
# ---------------------------------------------------------------------------

class MSEQuantized(NamedTuple):
    indices: torch.Tensor    # (..., packed_len) uint8 bit-packed
    norms: torch.Tensor      # (...,) original L2 norms
    bits: int


class ProdQuantized(NamedTuple):
    mse_indices: torch.Tensor    # (..., packed_len) uint8
    qjl_signs: torch.Tensor     # (..., packed_len) uint8 packed sign bits
    residual_norms: torch.Tensor # (...,) L2 norms of residual
    norms: torch.Tensor          # (...,) original L2 norms
    mse_bits: int


class ValueQuantized(NamedTuple):
    data: torch.Tensor     # (..., packed_d) bit-packed uint8
    scales: torch.Tensor   # (..., n_groups) scale per group
    zeros: torch.Tensor    # (..., n_groups) zero point per group
    bits: int = 2


# ---------------------------------------------------------------------------
# Bit packing helpers (verbatim from official)
# ---------------------------------------------------------------------------

def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return indices.to(torch.uint8)
    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)
    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * bits
    return (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)


def _unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    batch_shape = packed.shape[:-1]
    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()
    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    return unpacked.reshape(*batch_shape, -1)[..., :d].long()


# ---------------------------------------------------------------------------
# TurboQuantMSE
# ---------------------------------------------------------------------------

class TurboQuantMSE(torch.nn.Module):
    """MSE-optimal scalar quantization via Lloyd-Max codebook.

    Normalizes input to unit sphere, applies random rotation (det=+1),
    then searchsorted into precomputed Beta-distribution codebook.
    """

    def __init__(self, dim: int, bits: int = 3, device=None, dtype=torch.float32, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.n_clusters = 2 ** bits

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        Pi = make_rotation(dim, device=device, dtype=dtype, generator=g)
        self.register_buffer("Pi", Pi)

        centroids, decision_boundaries = get_codebook_tensors(dim, bits, device=device, dtype=dtype)
        self.register_buffer("centroids", centroids)
        self.register_buffer("decision_boundaries", decision_boundaries)

    def quantize(self, x: torch.Tensor) -> MSEQuantized:
        """x: (..., d) → MSEQuantized"""
        norms = x.norm(dim=-1)
        x_unit = x / (norms.unsqueeze(-1) + 1e-10)
        y = torch.matmul(x_unit.float(), self.Pi.T)
        indices = torch.searchsorted(self.decision_boundaries.contiguous(), y.contiguous())
        packed = _pack_indices(indices, self.bits)
        return MSEQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        indices = _unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        x_hat = torch.matmul(y_hat, self.Pi)
        return x_hat * q.norms.unsqueeze(-1)


# ---------------------------------------------------------------------------
# TurboQuantProd
# ---------------------------------------------------------------------------

class TurboQuantProd(torch.nn.Module):
    """Inner-product optimal TurboQuant (Algorithm 2).

    Stage 1: TurboQuantMSE at (bits-1) bits
    Stage 2: QJL on residual with d×d Gaussian S; stores sign(S @ r) packed as bits

    Dequant: x̃_mse + ||r|| * sqrt(π/2)/d * signs @ S
    (unbiased estimator of <y, x>)
    """

    def __init__(self, dim: int, bits: int = 3, device=None, dtype=torch.float32, seed: int = 42):
        super().__init__()
        assert bits >= 2, "TurboQuantProd requires at least 2 bits"
        self.dim = dim
        self.bits = bits
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

        self.mse = TurboQuantMSE(dim=dim, bits=bits - 1, device=device, dtype=dtype, seed=seed)

        g = torch.Generator(device="cpu")
        g.manual_seed(seed + 1000)
        S = torch.randn(dim, dim, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        self.register_buffer("S", S)

    def _pack_qjl(self, projected: torch.Tensor) -> torch.Tensor:
        signs = (projected > 0).to(torch.uint8)
        d = signs.shape[-1]
        if d % 8 != 0:
            signs = F.pad(signs, (0, 8 - d % 8), value=0)
        signs_r = signs.reshape(*signs.shape[:-1], -1, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
        return (signs_r * powers).sum(dim=-1, dtype=torch.uint8)

    def _unpack_qjl(self, packed: torch.Tensor) -> torch.Tensor:
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
        unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
        signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :self.dim]
        return 2.0 * signs - 1.0

    def quantize(self, x: torch.Tensor) -> ProdQuantized:
        """x: (..., d) → ProdQuantized"""
        mse_q = self.mse.quantize(x)
        x_hat = self.mse.dequantize(mse_q)
        residual = x - x_hat
        residual_norms = residual.norm(dim=-1)
        projected = torch.matmul(residual.float(), self.S.T)
        packed_signs = self._pack_qjl(projected)
        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=packed_signs,
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        mse_q = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
        x_mse = self.mse.dequantize(mse_q)
        signs = self._unpack_qjl(q.qjl_signs)
        x_qjl = torch.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * q.residual_norms.unsqueeze(-1))
        return x_mse + x_qjl


# ---------------------------------------------------------------------------
# Value group quantization (asymmetric min-max per group)
# ---------------------------------------------------------------------------

def quantize_values(v: torch.Tensor, bits: int = 2, group_size: int = 32) -> ValueQuantized:
    """Asymmetric group quantization for value vectors.

    v: (..., seq_len, d). d must be divisible by group_size.
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"
    n_groups = d // group_size

    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values
    n_levels = 2 ** bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min

    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    if bits == 2:
        assert d % 4 == 0
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
        v_q_flat = packed
    elif bits == 4:
        assert d % 2 == 0
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
        v_q_flat = packed

    return ValueQuantized(data=v_q_flat, scales=scale.squeeze(-1), zeros=zero.squeeze(-1), bits=bits)


def dequantize_values(vq: ValueQuantized, group_size: int = 32) -> torch.Tensor:
    """Dequantize value vectors from ValueQuantized."""
    bits = vq.bits
    packed = vq.data
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        data = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4).float()
    elif bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        data = torch.stack([v0, v1], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2).float()
    else:
        data = packed.float()

    d = data.shape[-1]
    n_groups = d // group_size
    data = data.reshape(*data.shape[:-1], n_groups, group_size)
    v = data * vq.scales.unsqueeze(-1) + vq.zeros.unsqueeze(-1)
    return v.reshape(*data.shape[:-2], d)
