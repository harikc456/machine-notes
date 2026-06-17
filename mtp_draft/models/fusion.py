from __future__ import annotations
import torch
import torch.nn as nn
from rbf_ffn.models.head_mixer import KromHCHeadMixer


class TeacherFeatureFusion(nn.Module):
    """
    Fuses N teacher hidden states (one per extracted layer) into a single
    d_draft vector using per-layer linear projections followed by KromHC
    head mixing.

    n_teacher_layers must be a power of 2 (KromHC requirement).

    Input:  (B, n_teacher_layers, d_teacher)
    Output: (B, d_draft)
    """

    def __init__(
        self,
        n_teacher_layers: int,
        d_teacher: int,
        d_draft: int,
        mixer_hidden: int = 32,
    ) -> None:
        super().__init__()
        assert (n_teacher_layers & (n_teacher_layers - 1)) == 0, (
            f"n_teacher_layers must be a power of 2; got {n_teacher_layers}"
        )
        self.n_layers = n_teacher_layers
        self.layer_projs = nn.ModuleList([
            nn.Linear(d_teacher, d_draft, bias=False)
            for _ in range(n_teacher_layers)
        ])
        self.head_mixer = KromHCHeadMixer(
            n_heads=n_teacher_layers,
            head_dim=d_draft,
            mixer_hidden=mixer_hidden,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, n_teacher_layers, d_teacher) → (B, d_draft)"""
        projected = torch.stack(
            [self.layer_projs[i](hidden_states[:, i, :]) for i in range(self.n_layers)],
            dim=1,
        )  # (B, n_layers, d_draft)
        mixed, _ = self.head_mixer(projected)  # (B, n_layers, d_draft)
        return mixed.mean(dim=1)  # (B, d_draft)
