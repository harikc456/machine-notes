from __future__ import annotations
import os
import sys
import torch
from transformers import DynamicCache

from kv_quant.config import QuantConfig

_SPECTRALQUANT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
)
if _SPECTRALQUANT_SRC not in sys.path:
    sys.path.insert(0, _SPECTRALQUANT_SRC)


def _restore_quant(state: dict):
    """Reconstruct a fitted NonUniformQuantizer from a saved quant_state entry."""
    from spectralquant.nonuniform_quantization import NonUniformQuantizer, LloydMaxQuantizer

    quant = NonUniformQuantizer(
        eigenvalues=torch.ones(state["head_dim"]),
        avg_bits=float(state["b_high"]),
    )
    quant._d_eff_int = state["d_eff_int"]
    quant._b_high = state["b_high"]
    quant._b_low = state["b_low"]
    quant._is_fitted = True

    sem_q = LloydMaxQuantizer(n_bits=state["b_high"])
    sem_q._centroids = state["semantic_centroids"].float()
    sem_q._is_fitted = True
    quant._semantic_quantizer = sem_q

    tail_q = LloydMaxQuantizer(n_bits=state["b_low"])
    tail_q._centroids = state["tail_centroids"].float()
    tail_q._is_fitted = True
    quant._tail_quantizer = tail_q

    return quant


class SpectralQuantCache(DynamicCache):
    """DynamicCache quantizing K/V with official NonUniformQuantizer + SpectralRotation.

    Per-head Lloyd-Max scalar quantization in the spectral basis:
      semantic regime (first d_eff coords): b_high bits
      tail regime (remaining coords): b_low bits

    cal_data: (EigenspectralCalibrator, quant_state_dict)
    """

    def __init__(self, config: QuantConfig, cal_data: tuple) -> None:
        super().__init__()
        self.config = config
        calibrator, quant_state = cal_data

        from spectralquant.spectral_rotation import SpectralRotation

        self._key_rot = SpectralRotation(calibrator, "key")
        self._val_rot = SpectralRotation(calibrator, "value")

        self._key_quants: dict = {}
        self._val_quants: dict = {}
        self._head_meta: dict = {}  # (l, h, kv_type) -> {d_eff_int, b_high, b_low, head_dim}

        for k, state in quant_state.items():
            # k format: "L{l}_H{h}_key" or "L{l}_H{h}_value"
            parts = k.split("_")  # ["L0", "H1", "key"] etc.
            l = int(parts[0][1:])
            h = int(parts[1][1:])
            kv_type = parts[2]
            meta = {
                "d_eff_int": state["d_eff_int"],
                "b_high": state["b_high"],
                "b_low": state["b_low"],
                "head_dim": state["head_dim"],
            }
            quant = _restore_quant(state)
            if kv_type == "key":
                self._key_quants[(l, h)] = quant
                self._head_meta[(l, h, "key")] = meta
            else:
                self._val_quants[(l, h)] = quant
                self._head_meta[(l, h, "value")] = meta

        # Per-layer per-head index storage: _sk_sem[l] = [Tensor or None] * n_heads
        self._sk_sem: list[list] = []
        self._sk_tail: list[list] = []
        self._sv_sem: list[list] = []
        self._sv_tail: list[list] = []

    def _ensure_layer(self, layer_idx: int, n_heads: int) -> None:
        while len(self._sk_sem) <= layer_idx:
            self._sk_sem.append([None] * n_heads)
            self._sk_tail.append([None] * n_heads)
            self._sv_sem.append([None] * n_heads)
            self._sv_tail.append([None] * n_heads)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress new K/V tokens and return full dequantized K, V for attention."""
        if layer_idx > len(self._sk_sem):
            raise IndexError(
                f"layer_idx {layer_idx} out of order; expected <= {len(self._sk_sem)}"
            )
        B, H, S, D = key_states.shape
        self._ensure_layer(layer_idx, H)

        from spectralquant.nonuniform_quantization import CompressedVector

        k_hat_heads = []
        v_hat_heads = []

        for h in range(H):
            k_h = key_states[:, h, :, :].float()   # (B, S, D)
            v_h = value_states[:, h, :, :].float()  # (B, S, D)

            k_rot = self._key_rot.rotate(k_h, layer_idx, h)  # (B, S, D)
            v_rot = self._val_rot.rotate(v_h, layer_idx, h)

            k_cv = self._key_quants[(layer_idx, h)].compress(k_rot)
            v_cv = self._val_quants[(layer_idx, h)].compress(v_rot)

            if self._sk_sem[layer_idx][h] is None:
                self._sk_sem[layer_idx][h] = k_cv.semantic_indices
                self._sk_tail[layer_idx][h] = k_cv.tail_indices
                self._sv_sem[layer_idx][h] = v_cv.semantic_indices
                self._sv_tail[layer_idx][h] = v_cv.tail_indices
            else:
                self._sk_sem[layer_idx][h] = torch.cat(
                    [self._sk_sem[layer_idx][h], k_cv.semantic_indices], dim=1
                )
                self._sk_tail[layer_idx][h] = torch.cat(
                    [self._sk_tail[layer_idx][h], k_cv.tail_indices], dim=1
                )
                self._sv_sem[layer_idx][h] = torch.cat(
                    [self._sv_sem[layer_idx][h], v_cv.semantic_indices], dim=1
                )
                self._sv_tail[layer_idx][h] = torch.cat(
                    [self._sv_tail[layer_idx][h], v_cv.tail_indices], dim=1
                )

            k_meta = self._head_meta[(layer_idx, h, "key")]
            S_full = self._sk_sem[layer_idx][h].shape[1]
            k_full_cv = CompressedVector(
                semantic_indices=self._sk_sem[layer_idx][h],
                tail_indices=self._sk_tail[layer_idx][h],
                d_eff=k_meta["d_eff_int"],
                head_dim=k_meta["head_dim"],
                b_high=k_meta["b_high"],
                b_low=k_meta["b_low"],
                original_shape=(B, S_full, D),
            )
            k_hat = self._key_rot.unrotate(
                self._key_quants[(layer_idx, h)].decompress(k_full_cv), layer_idx, h
            )  # (B, S_full, D)

            v_meta = self._head_meta[(layer_idx, h, "value")]
            v_full_cv = CompressedVector(
                semantic_indices=self._sv_sem[layer_idx][h],
                tail_indices=self._sv_tail[layer_idx][h],
                d_eff=v_meta["d_eff_int"],
                head_dim=v_meta["head_dim"],
                b_high=v_meta["b_high"],
                b_low=v_meta["b_low"],
                original_shape=(B, S_full, D),
            )
            v_hat = self._val_rot.unrotate(
                self._val_quants[(layer_idx, h)].decompress(v_full_cv), layer_idx, h
            )

            k_hat_heads.append(k_hat)
            v_hat_heads.append(v_hat)

        k_full = torch.stack(k_hat_heads, dim=1).to(key_states.dtype)
        v_full = torch.stack(v_hat_heads, dim=1).to(value_states.dtype)
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._sk_sem):
            return 0
        for slot in self._sk_sem[layer_idx]:
            if slot is not None:
                return slot.shape[1]
        return 0

    def compressed_bytes(self) -> int:
        """Theoretical compressed bytes using declared bit widths (not int32 tensor sizes)."""
        total_bits = 0
        for l_idx in range(len(self._sk_sem)):
            for h in range(len(self._sk_sem[l_idx])):
                sem = self._sk_sem[l_idx][h]
                if sem is None:
                    continue
                meta = self._head_meta[(l_idx, h, "key")]
                n_vecs = sem.shape[0] * sem.shape[1]  # B * S
                d_eff = meta["d_eff_int"]
                D = meta["head_dim"]
                b_high = meta["b_high"]
                b_low = meta["b_low"]
                # Key + Value: same bit structure
                total_bits += 2 * n_vecs * (d_eff * b_high + (D - d_eff) * b_low)
        return (total_bits + 7) // 8
