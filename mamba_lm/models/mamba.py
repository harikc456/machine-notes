"""
Pure-PyTorch Mamba (S6) implementation.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023).

No mamba-ssm package required; uses a sequential scan compiled by torch.compile.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_lm.config import MambaConfig


# ---------------------------------------------------------------------------
# Selective scan (S6 core)
# ---------------------------------------------------------------------------

def selective_scan(x: torch.Tensor, dA: torch.Tensor, dB_x: torch.Tensor,
                   C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Sequential selective scan (compiled by torch.compile for speed).

    Args:
        x:    (B, L, d_inner)
        dA:   (B, L, d_inner, d_state)   discretised A  (exp(dt * A))
        dB_x: (B, L, d_inner, d_state)   discretised B * x  (dt * B * x)
        C:    (B, L, d_state)
        D:    (d_inner,)                  skip-connection weight

    Returns:
        y: (B, L, d_inner)
    """
    B, L, d_inner, d_state = dA.shape
    h = dA.new_zeros(B, d_inner, d_state)
    ys: list[torch.Tensor] = []
    for t in range(L):
        h = dA[:, t] * h + dB_x[:, t]              # (B, d_inner, d_state)
        y_t = (h * C[:, t].unsqueeze(1)).sum(-1)    # (B, d_inner)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)                       # (B, L, d_inner)
    return y + x * D


# ---------------------------------------------------------------------------
# Mamba block
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    One Mamba residual block:

        residual = x
        x = norm(x)
        x, z = in_proj(x).split(d_inner)
        x = causal_conv1d(x)
        x = silu(x)
        y = ssm(x)           # selective state space scan
        y = y * silu(z)      # gating
        return out_proj(y) + residual
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        d_model  = cfg.d_model
        d_inner  = d_model * cfg.expand
        d_state  = cfg.d_state
        d_conv   = cfg.d_conv
        dt_rank  = cfg.dt_rank

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.norm     = nn.RMSNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,   # left+right pad; we crop right side after
            bias=True,
        )

        # SSM input projections (dt, B, C are input-dependent → "selective")
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # Δ (dt) up-projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A: log of the diagonal magnitudes, shape (d_inner, d_state)
        # A is kept negative (stable): A_diag = -exp(A_log)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True   # type: ignore[attr-defined]

        # D: skip connection (residual from x to y)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True       # type: ignore[attr-defined]

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._init_dt_proj(cfg)

    # ------------------------------------------------------------------
    def _init_dt_proj(self, cfg: MambaConfig) -> None:
        """Initialise dt_proj bias so initial Δ values ∈ [dt_min, dt_max]."""
        dt_rank = cfg.dt_rank
        d_inner = self.d_inner

        # dt_proj weight: initialise with 1/sqrt(dt_rank) scale
        nn.init.uniform_(self.dt_proj.weight, -1.0 / math.sqrt(dt_rank),
                         1.0 / math.sqrt(dt_rank))

        # Bias: inv_softplus of uniform samples from [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).clamp(min=cfg.dt_init_floor)
        # softplus inverse: log(exp(y) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    # ------------------------------------------------------------------
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_inner)
        returns y: (B, L, d_inner)
        """
        B, L, _ = x.shape
        A = -torch.exp(self.A_log)          # (d_inner, d_state) — negative diagonal

        # Compute selective (input-dependent) parameters
        xz  = self.x_proj(x)               # (B, L, dt_rank + 2*d_state)
        dt_raw, B_ssm, C = xz.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )                                   # B_ssm/C: (B, L, d_state)
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)

        # ZOH discretisation
        # dA = exp(dt ⊗ A)         shape (B, L, d_inner, d_state)
        # dB ≈ dt * B               (ZOH first-order approx, matches mamba-ssm)
        # dB_x = dB * x            incorporate x before the scan
        dA   = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
        dB_x = torch.einsum("bld,bln,bld->bldn", dt, B_ssm, x)

        return selective_scan(x, dA, dB_x, C, self.D)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)               # (B, L, 2 * d_inner)
        x_in, z = xz.split(self.d_inner, dim=-1)

        # Causal depthwise conv
        x_conv = x_in.transpose(1, 2)      # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :x_in.shape[1]]   # crop right padding
        x_conv = x_conv.transpose(1, 2)    # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        y = self._ssm(x_conv)
        y = y * F.silu(z)                  # gating

        return self.out_proj(y) + residual


# ---------------------------------------------------------------------------
# Full causal language model
# ---------------------------------------------------------------------------

class CausalMambaLM(nn.Module):
    """
    token_embedding → N × MambaBlock → RMSNorm → lm_head

    forward(tokens) → logits: (B, L, vocab_size)
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L)  →  logits: (B, L, vocab_size)"""
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Optimizer groups
# ---------------------------------------------------------------------------

def build_optimizer_groups(
    model: CausalMambaLM,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split parameters into AdamW and weight-decay-free groups.

    Parameters marked with ``_no_weight_decay = True`` (A_log, D) and
    all 1-D tensors (biases, norms) get weight_decay=0.
    2-D weights get the configured weight decay.

    Returns (wd_params, no_wd_params).
    """
    emb_id = id(model.token_embedding.weight)
    seen: set[int] = set()
    wd: list[torch.Tensor]    = []
    no_wd: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if getattr(param, "_no_weight_decay", False):
            no_wd.append(param)
        elif pid == emb_id:
            no_wd.append(param)
        elif param.ndim == 1:
            no_wd.append(param)
        else:
            wd.append(param)

    return wd, no_wd
