# idlm/models/idlm_model.py
from __future__ import annotations
import torch
import torch.nn as nn
from rbf_ffn.models.model import CausalLM
from idlm.models.lora import LoRALinear, apply_lora


class IDLMCausalLM(nn.Module):
    """
    I-DLM model: frozen rbf_ffn CausalLM base + LoRA adapters at mask positions.

    forward(tokens, use_lora_mask):
        tokens:        (B, L) int64  — token ids
        use_lora_mask: (B, L, 1) float — 1.0 at decode (x_t) positions,
                       0.0 at introspection (x_0) positions
        returns:       (B, L, vocab_size) logits
    """

    def __init__(
        self,
        ar_model: CausalLM,
        lora_rank: int,
        lora_alpha: float,
        lora_target_modules: list[str],
    ):
        super().__init__()
        self.model = ar_model
        for p in self.model.parameters():
            p.requires_grad_(False)
        apply_lora(self.model, lora_target_modules, lora_rank, lora_alpha)

    def _lora_layers(self) -> list[LoRALinear]:
        return [m for m in self.model.modules() if isinstance(m, LoRALinear)]

    def _set_mask(self, mask: torch.Tensor | None) -> None:
        for lora in self._lora_layers():
            lora.current_mask = mask

    def forward(self, tokens: torch.Tensor, use_lora_mask: torch.Tensor) -> torch.Tensor:
        # Determine device/dtype from use_lora_mask for the zero-reset tensor.
        self._set_mask(use_lora_mask)
        try:
            logits, _ = self.model(tokens)
        finally:
            # Reset to a broadcastable zero so that direct AR calls (current_mask=None
            # path) add no LoRA delta and output equals the frozen base.
            zero = torch.zeros(1, 1, 1, dtype=use_lora_mask.dtype, device=use_lora_mask.device)
            self._set_mask(zero)
        return logits

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        ar_config,
        lora_rank: int,
        lora_alpha: float,
        lora_target_modules: list[str],
        device: torch.device,
    ) -> "IDLMCausalLM":
        """Load a trained rbf_ffn CausalLM checkpoint and wrap it."""
        ar_model = CausalLM(ar_config).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        ar_model.load_state_dict(ckpt["model"])
        return cls(ar_model, lora_rank, lora_alpha, lora_target_modules)
