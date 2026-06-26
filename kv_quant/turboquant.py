from __future__ import annotations
import torch
from transformers import DynamicCache

from kv_quant.config import QuantConfig
from kv_quant.ops.turboquant_core import (
    TurboQuantProd, ProdQuantized,
    quantize_values, dequantize_values, ValueQuantized,
)


class TurboQuantCache(DynamicCache):
    """DynamicCache that quantizes K/V with the official TurboQuant algorithm.

    Keys: TurboQuantProd (Lloyd-Max codebook + QJL residual correction).
    Values: asymmetric group quantization (group_size=32 by default).
    Recent buffer: last `buffer_size` tokens kept in fp16 for quality.

    One TurboQuantProd instance per layer, lazily initialized on first update.
    """

    def __init__(self, config: QuantConfig, n_heads: int, head_dim: int, device=None):
        super().__init__()
        self.config = config
        self.n_heads = n_heads
        self.head_dim = head_dim
        self._device = device

        # Per-layer quantizers (lazily created)
        self._key_quantizers: list[TurboQuantProd] = []

        # Per-layer compressed storage
        self._qk: list[ProdQuantized | None] = []   # quantized keys (old tokens)
        self._qv: list[ValueQuantized | None] = []   # quantized values (old tokens)

        # Per-layer recent buffer (last buffer_size tokens, full precision)
        self._k_buf: list[torch.Tensor | None] = []
        self._v_buf: list[torch.Tensor | None] = []

    # ------------------------------------------------------------------
    def _get_quantizer(self, layer_idx: int, device) -> TurboQuantProd:
        while len(self._key_quantizers) <= layer_idx:
            i = len(self._key_quantizers)
            self._key_quantizers.append(
                TurboQuantProd(
                    dim=self.head_dim,
                    bits=self.config.bits,
                    device=device,
                    seed=42 + i * 7,
                )
            )
        return self._key_quantizers[layer_idx]

    def _ensure_layer(self, layer_idx: int):
        while len(self._qk) <= layer_idx:
            self._qk.append(None)
            self._qv.append(None)
            self._k_buf.append(None)
            self._v_buf.append(None)

    # ------------------------------------------------------------------
    def _flush_to_quantized(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        """Compress tokens and append to quantized storage."""
        q = self._get_quantizer(layer_idx, keys.device)

        new_kq = q.quantize(keys.float())
        new_vq = quantize_values(
            values.float(),
            bits=self.config.value_bits,
            group_size=self.config.value_group_size,
        )

        if self._qk[layer_idx] is None:
            self._qk[layer_idx] = new_kq
            self._qv[layer_idx] = new_vq
        else:
            old_kq = self._qk[layer_idx]
            self._qk[layer_idx] = ProdQuantized(
                mse_indices=torch.cat([old_kq.mse_indices, new_kq.mse_indices], dim=-2),
                qjl_signs=torch.cat([old_kq.qjl_signs, new_kq.qjl_signs], dim=-2),
                residual_norms=torch.cat([old_kq.residual_norms, new_kq.residual_norms], dim=-1),
                norms=torch.cat([old_kq.norms, new_kq.norms], dim=-1),
                mse_bits=new_kq.mse_bits,
            )
            old_vq = self._qv[layer_idx]
            self._qv[layer_idx] = ValueQuantized(
                data=torch.cat([old_vq.data, new_vq.data], dim=-2),
                scales=torch.cat([old_vq.scales, new_vq.scales], dim=-2),
                zeros=torch.cat([old_vq.zeros, new_vq.zeros], dim=-2),
                bits=self.config.value_bits,
            )

    # ------------------------------------------------------------------
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V tokens; return full dequantized K, V for attention."""
        if layer_idx > len(self._qk):
            raise IndexError(
                f"layer_idx {layer_idx} out of order; expected <= {len(self._qk)}"
            )
        self._ensure_layer(layer_idx)

        # Append to buffer
        if self._k_buf[layer_idx] is None:
            self._k_buf[layer_idx] = key_states
            self._v_buf[layer_idx] = value_states
        else:
            self._k_buf[layer_idx] = torch.cat([self._k_buf[layer_idx], key_states], dim=-2)
            self._v_buf[layer_idx] = torch.cat([self._v_buf[layer_idx], value_states], dim=-2)

        # Flush oldest tokens to quantized storage when buffer exceeds limit
        buf_len = self._k_buf[layer_idx].shape[-2]
        if buf_len > self.config.buffer_size:
            n_flush = buf_len - self.config.buffer_size
            self._flush_to_quantized(
                layer_idx,
                self._k_buf[layer_idx][..., :n_flush, :],
                self._v_buf[layer_idx][..., :n_flush, :],
            )
            self._k_buf[layer_idx] = self._k_buf[layer_idx][..., n_flush:, :]
            self._v_buf[layer_idx] = self._v_buf[layer_idx][..., n_flush:, :]

        # Dequantize and return full K, V for attention
        parts_k, parts_v = [], []
        if self._qk[layer_idx] is not None:
            q = self._get_quantizer(layer_idx, key_states.device)
            parts_k.append(q.dequantize(self._qk[layer_idx]).to(key_states.dtype))
            parts_v.append(
                dequantize_values(self._qv[layer_idx], self.config.value_group_size)
                .to(value_states.dtype)
            )
        parts_k.append(self._k_buf[layer_idx])
        parts_v.append(self._v_buf[layer_idx])

        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    # ------------------------------------------------------------------
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._qk):
            return 0
        qk_len = 0
        if self._qk[layer_idx] is not None:
            qk_len = self._qk[layer_idx].norms.shape[-1]
        buf_len = 0 if self._k_buf[layer_idx] is None else self._k_buf[layer_idx].shape[-2]
        return qk_len + buf_len

    def compressed_bytes(self) -> int:
        """Bytes used by compressed K/V storage (excludes fp16 buffer)."""
        total = 0
        for kq in self._qk:
            if kq is None:
                continue
            total += kq.mse_indices.nelement()
            total += (kq.qjl_signs.nelement() + 7) // 8
            total += kq.residual_norms.nelement() * 2
            total += kq.norms.nelement() * 2
        for vq in self._qv:
            if vq is None:
                continue
            total += vq.data.nelement()
            total += vq.scales.nelement() * 2
            total += vq.zeros.nelement() * 2
        return total
