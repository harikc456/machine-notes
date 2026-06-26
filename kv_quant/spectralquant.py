from __future__ import annotations
import torch
from transformers import DynamicCache

from kv_quant.config import QuantConfig
from kv_quant.ops.qjl import encode_2d


class SpectralQuantCache(DynamicCache):
    """DynamicCache that quantizes K via SpectralQuant VQ; V stored in bfloat16.

    Compressed key storage per layer:
      _sq_k_sig_idx: (B, H, S) uint8  — index into per-head signal codebook
      _sq_k_noi_idx: (B, H, S) uint8  — index into per-head noise codebook
      _sq_k_qjl:     (B, H, S, m) bool — QJL bits on signal residual
    Value storage: _sq_v (B, H, S, d) bfloat16 — no VQ on values.
    """

    def __init__(self, config: QuantConfig, cal_data: dict, device=None):
        super().__init__()
        self.config = config
        self.cal_data = cal_data
        self.device = device

        self._sq_k_sig_idx: list[torch.Tensor] = []
        self._sq_k_noi_idx: list[torch.Tensor] = []
        self._sq_k_qjl:     list[torch.Tensor] = []
        self._sq_v:          list[torch.Tensor] = []

    # ------------------------------------------------------------------
    def _head_cal(self, layer_idx: int, head_idx: int) -> dict:
        return self.cal_data["layers"][layer_idx][head_idx]

    @staticmethod
    def _nearest(h: torch.Tensor, cb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest-centroid lookup.
        h: (N, k) float, cb: (C, k) float
        Returns: (idx: (N,) uint8, reconstructed: (N, k) float)
        """
        dists = torch.cdist(h.float(), cb.float())   # (N, C)
        idx = dists.argmin(dim=-1)                    # (N,)
        return idx.to(torch.uint8), cb[idx]

    def _quant_key_layer(
        self, key_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize all heads for one layer.

        key_states: (B, H, S, d)
        Returns: (sig_idx, noi_idx, qjl_bits, k_dq)
          sig_idx:  (B, H, S) uint8
          noi_idx:  (B, H, S) uint8
          qjl_bits: (B, H, S, m) bool
          k_dq:     (B, H, S, d) float32
        """
        B, H, S, D = key_states.shape
        sig_idxs, noi_idxs, qjl_list, k_dq_list = [], [], [], []

        for head in range(H):
            cal = self._head_cal(layer_idx, head)
            U   = cal["U"].to(key_states.device)              # (d, d)
            d_s = cal["d_s"]
            cb_s = cal["codebook_signal"].to(key_states.device)  # (C_s, d_s)
            cb_n = cal["codebook_noise"].to(key_states.device)   # (C_n, d-d_s)
            S_sig = cal["S_signal"].to(key_states.device)        # (m, d_s)

            h = key_states[:, head, :, :].float()  # (B, S, d)
            h_flat = h.reshape(-1, D)               # (B*S, d)

            h_proj = h_flat @ U                     # (B*S, d)
            h_sig  = h_proj[:, :d_s]               # (B*S, d_s)
            h_noi  = h_proj[:, d_s:]               # (B*S, d-d_s)

            s_idx, h_sig_dq = self._nearest(h_sig, cb_s)
            n_idx, h_noi_dq = self._nearest(h_noi, cb_n)

            h_proj_dq = torch.cat([h_sig_dq, h_noi_dq], dim=-1)  # (B*S, d)
            k_dq = (h_proj_dq @ U.T).reshape(B, S, D)             # (B, S, d)

            # QJL on signal residual
            residual_sig = (h_sig - h_sig_dq)                      # (B*S, d_s)
            qjl = encode_2d(residual_sig, S_sig).reshape(B, S, -1) # (B, S, m)

            sig_idxs.append(s_idx.reshape(B, S))
            noi_idxs.append(n_idx.reshape(B, S))
            qjl_list.append(qjl)
            k_dq_list.append(k_dq)

        return (
            torch.stack(sig_idxs, dim=1),   # (B, H, S)
            torch.stack(noi_idxs, dim=1),   # (B, H, S)
            torch.stack(qjl_list,  dim=1),  # (B, H, S, m)
            torch.stack(k_dq_list, dim=1),  # (B, H, S, d)
        )

    def _dequant_key_full(self, layer_idx: int) -> torch.Tensor:
        """Reconstruct full accumulated key cache for layer_idx."""
        sig_idx = self._sq_k_sig_idx[layer_idx]  # (B, H, S)
        noi_idx = self._sq_k_noi_idx[layer_idx]  # (B, H, S)
        B, H, S = sig_idx.shape
        D = self.cal_data["head_dim"]
        k_dq_list = []

        for head in range(H):
            cal  = self._head_cal(layer_idx, head)
            U    = cal["U"].to(sig_idx.device)
            d_s  = cal["d_s"]
            cb_s = cal["codebook_signal"].to(sig_idx.device)
            cb_n = cal["codebook_noise"].to(sig_idx.device)

            s_flat = sig_idx[:, head, :].reshape(-1).long()
            n_flat = noi_idx[:, head, :].reshape(-1).long()

            h_sig_dq = cb_s[s_flat]                              # (B*S, d_s)
            h_noi_dq = cb_n[n_flat]                              # (B*S, d-d_s)
            h_proj_dq = torch.cat([h_sig_dq, h_noi_dq], dim=-1) # (B*S, d)
            k_dq_list.append((h_proj_dq @ U.T).reshape(B, S, D))

        return torch.stack(k_dq_list, dim=1).float()  # (B, H, S, d)

    # ------------------------------------------------------------------
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sig_idx, noi_idx, qjl_bits, _ = self._quant_key_layer(key_states, layer_idx)
        v_new = value_states.bfloat16()

        if layer_idx >= len(self._sq_k_sig_idx):
            self._sq_k_sig_idx.append(sig_idx)
            self._sq_k_noi_idx.append(noi_idx)
            self._sq_k_qjl.append(qjl_bits)
            self._sq_v.append(v_new)
        else:
            self._sq_k_sig_idx[layer_idx] = torch.cat([self._sq_k_sig_idx[layer_idx], sig_idx],   dim=2)
            self._sq_k_noi_idx[layer_idx] = torch.cat([self._sq_k_noi_idx[layer_idx], noi_idx],   dim=2)
            self._sq_k_qjl[layer_idx]     = torch.cat([self._sq_k_qjl[layer_idx],     qjl_bits],  dim=2)
            self._sq_v[layer_idx]         = torch.cat([self._sq_v[layer_idx],          v_new],     dim=2)

        k_full = self._dequant_key_full(layer_idx)
        v_full = self._sq_v[layer_idx].float()
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self._sq_k_sig_idx:
            return 0
        idx = min(layer_idx, len(self._sq_k_sig_idx) - 1)
        return self._sq_k_sig_idx[idx].shape[2]

    def compressed_bytes(self) -> int:
        total = 0
        for buf in self._sq_k_sig_idx + self._sq_k_noi_idx:
            total += buf.nelement() * buf.element_size()  # uint8
        for buf in self._sq_k_qjl:
            total += buf.nelement() // 8 + 1
        for buf in self._sq_v:
            total += buf.nelement() * buf.element_size()  # bfloat16
        return total
