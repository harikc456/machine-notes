from __future__ import annotations
import torch
from transformers import DynamicCache

from kv_quant.config import QuantConfig
from kv_quant.ops.rotation import make_rotation, rotate, unrotate
from kv_quant.ops.scalar_quant import quantize, dequantize
from kv_quant.ops.qjl import make_sign_matrix, encode


class TurboQuantCache(DynamicCache):
    """DynamicCache that compresses K/V with TurboQuant (rotation + scalar quant + QJL).

    Stores compressed buffers (_qk_int, _qk_scale, _qk_qjl and V equivalents).
    key_cache / value_cache are kept empty; get_seq_length() reads from _qk_int.
    update() returns dequantized K/V for HF attention to consume directly.
    """

    def __init__(self, config: QuantConfig, n_heads: int, head_dim: int, device=None):
        super().__init__()
        self.config = config
        self.n_heads = n_heads
        self.head_dim = head_dim

        torch.manual_seed(0)  # reproducible rotation / QJL matrices
        self._Rk = torch.stack([make_rotation(head_dim, device=device) for _ in range(n_heads)])
        self._Rv = torch.stack([make_rotation(head_dim, device=device) for _ in range(n_heads)])
        m = config.qjl_dim
        self._Sk = torch.stack([make_sign_matrix(m, head_dim, device=device) for _ in range(n_heads)])
        self._Sv = torch.stack([make_sign_matrix(m, head_dim, device=device) for _ in range(n_heads)])

        # Compressed buffers — one tensor per layer
        self._qk_int:   list[torch.Tensor] = []   # (B, H, S, d) int8
        self._qk_scale: list[torch.Tensor] = []   # (B, H, S, 1) float16
        self._qk_qjl:   list[torch.Tensor] = []   # (B, H, S, m) bool
        self._qv_int:   list[torch.Tensor] = []
        self._qv_scale: list[torch.Tensor] = []
        self._qv_qjl:   list[torch.Tensor] = []

    # ------------------------------------------------------------------
    def _compress(
        self, h: torch.Tensor, R: torch.Tensor, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotate → quantize → QJL residual.

        h: (B, H, S, d)
        Returns: (h_int, h_scale, qjl_bits)  — all on same device as h
        """
        R = R.to(h.device)
        S = S.to(h.device)
        h_rot = rotate(h.float(), R)
        h_int, h_scale = quantize(h_rot, self.config.bits)
        h_rot_dq = dequantize(h_int, h_scale, self.config.bits)
        residual = h_rot - h_rot_dq
        qjl_bits = encode(residual, S)
        return h_int, h_scale, qjl_bits

    def _decompress(
        self,
        h_int: torch.Tensor,
        h_scale: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize and unrotate full accumulated layer cache."""
        R = R.to(h_int.device)
        h_rot_dq = dequantize(h_int, h_scale, self.config.bits)
        return unrotate(h_rot_dq, R)

    # ------------------------------------------------------------------
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_int, k_scale, k_qjl = self._compress(key_states, self._Rk, self._Sk)
        v_int, v_scale, v_qjl = self._compress(value_states, self._Rv, self._Sv)

        if layer_idx >= len(self._qk_int):
            self._qk_int.append(k_int)
            self._qk_scale.append(k_scale)
            self._qk_qjl.append(k_qjl)
            self._qv_int.append(v_int)
            self._qv_scale.append(v_scale)
            self._qv_qjl.append(v_qjl)
        else:
            self._qk_int[layer_idx]   = torch.cat([self._qk_int[layer_idx],   k_int],   dim=2)
            self._qk_scale[layer_idx] = torch.cat([self._qk_scale[layer_idx], k_scale], dim=2)
            self._qk_qjl[layer_idx]   = torch.cat([self._qk_qjl[layer_idx],   k_qjl],   dim=2)
            self._qv_int[layer_idx]   = torch.cat([self._qv_int[layer_idx],   v_int],   dim=2)
            self._qv_scale[layer_idx] = torch.cat([self._qv_scale[layer_idx], v_scale], dim=2)
            self._qv_qjl[layer_idx]   = torch.cat([self._qv_qjl[layer_idx],   v_qjl],   dim=2)

        k_full = self._decompress(self._qk_int[layer_idx], self._qk_scale[layer_idx], self._Rk)
        v_full = self._decompress(self._qv_int[layer_idx], self._qv_scale[layer_idx], self._Rv)
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._qk_int):
            return 0
        return self._qk_int[layer_idx].shape[2]

    def compressed_bytes(self) -> int:
        """Bytes used by compressed K/V buffers."""
        total = 0
        for buf in self._qk_int + self._qv_int:
            total += buf.nelement() * buf.element_size()
        for buf in self._qk_scale + self._qv_scale:
            total += buf.nelement() * buf.element_size()
        for buf in self._qk_qjl + self._qv_qjl:
            total += (buf.nelement() + 7) // 8  # 1 bit per bool, ceiling division
        return total
