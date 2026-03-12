# RBF-FFN WikiText-103 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the toy training loop with a rigorous WikiText-103 causal-LM ablation study comparing a Llama baseline (RMSNorm + RoPE + SwiGLU) against RBF-FFN variants (G0/G1A/G1B/G2 × σ-A/B/C).

**Architecture:** Shared CausalSelfAttention (RMSNorm + RoPE + SDPA, `is_causal=True`) with swappable FFN: `SwiGLUFFN` for the baseline, `RBFFFN` for ablations. A single `CausalLM` model class selects the block type from `cfg.model_type`.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.10, tiktoken, HuggingFace datasets, pytest, PyYAML

---

## Chunk 1: Config extension and RBFFFN patch

### Task 1: Extend RBFFFNConfig with training and model-type fields

**Context:** `load_config` raises `ValueError` on unknown YAML keys. New fields must be added to the dataclass *before* any YAML file is updated. This task must be committed before Task 8 (YAML updates).

**Files:**
- Modify: `rbf_ffn/config.py`
- Modify: `rbf_ffn/tests/test_config.py`

- [ ] **Step 1: Add new-field tests to test_config.py**

Append to `rbf_ffn/tests/test_config.py`:

```python
# ── New training field defaults ───────────────────────────────────────────────

def test_new_fields_have_correct_defaults():
    cfg = RBFFFNConfig()
    assert cfg.model_type == "rbf"
    assert cfg.ffn_hidden == 688
    assert cfg.seed == 42
    assert cfg.n_epochs == 10
    assert cfg.batch_size == 32
    assert abs(cfg.muon_lr - 0.02) < 1e-9
    assert abs(cfg.adamw_lr - 3e-4) < 1e-9
    assert abs(cfg.adamw_wd - 0.1) < 1e-9
    assert abs(cfg.warmup_ratio - 0.02) < 1e-9
    assert abs(cfg.grad_clip - 1.0) < 1e-9


def test_seq_len_default_updated():
    """seq_len default changes from 65 → 512 for WikiText-103."""
    cfg = RBFFFNConfig()
    assert cfg.seq_len == 512


def test_load_config_accepts_model_type(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("model_type: baseline\n")
    cfg = load_config(p)
    assert cfg.model_type == "baseline"


def test_load_config_accepts_ffn_hidden(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("ffn_hidden: 1024\n")
    cfg = load_config(p)
    assert cfg.ffn_hidden == 1024
```

- [ ] **Step 2: Run new tests — verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
pytest rbf_ffn/tests/test_config.py::test_new_fields_have_correct_defaults rbf_ffn/tests/test_config.py::test_load_config_accepts_model_type -v
```

Expected: `FAILED` — `RBFFFNConfig has no attribute 'model_type'`

- [ ] **Step 3: Update RBFFFNConfig in config.py**

Replace the full `RBFFFNConfig` class (keep `load_config` unchanged):

```python
@dataclass
class RBFFFNConfig:
    # Model dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # RBF kernel
    K: int = 5
    centers: list[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    sigma_init: float = 0.5
    sigma_variant: str = "global"  # "global" | "per_center" | "per_dim"

    # Gate variant
    gate_variant: str = "G0"       # "G0" | "G1A" | "G1B" | "G2"
    sinkhorn_iters: int = 20       # G2 only

    # Sequence / vocab
    seq_len: int = 512
    vocab_size: int = 50257

    # Model type
    model_type: str = "rbf"        # "baseline" | "rbf"
    ffn_hidden: int = 688          # SwiGLU hidden dim; ignored by RBF model

    # Training
    seed: int = 42
    n_epochs: int = 10
    batch_size: int = 32
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    adamw_wd: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
```

Note: `num_classes` field is removed (was only used by the toy training loop). If existing tests import it, they will need updating in Task 5.

- [ ] **Step 4: Strip `num_classes` and update `seq_len` in all existing YAML configs**

The existing six YAML files contain `num_classes: 10` (now an unknown key after removing it from the dataclass) and `seq_len: 65` (needs updating to 512). Do a minimal strip now so `load_config` does not raise on them. The full config content (with new training fields) will be written in Task 8.

Edit each of these six files to remove the `num_classes:` line and change `seq_len: 65` → `seq_len: 512`:
- `rbf_ffn/configs/g0_baseline.yaml`
- `rbf_ffn/configs/g1a_cross_kernel.yaml`
- `rbf_ffn/configs/g1b_input_driven.yaml`
- `rbf_ffn/configs/g2_sinkhorn.yaml`
- `rbf_ffn/configs/sigma_b_per_center.yaml`
- `rbf_ffn/configs/sigma_c_per_dim.yaml`

After the edit each file should look like (example for g0):
```yaml
gate_variant: G0
d_model: 64
n_heads: 4
n_layers: 2
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sigma_variant: global
sinkhorn_iters: 20
dropout: 0.1
seq_len: 512
```

- [ ] **Step 5: Run full config test suite**

```bash
pytest rbf_ffn/tests/test_config.py -v
```

Expected: all tests pass. `test_load_config_values_match_yaml` expects `d_model == 64`, `K == 5`, `sigma_init == 0.5`, `n_layers == 2` from `g0_baseline.yaml` — these are still correct after the minimal strip.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/config.py rbf_ffn/tests/test_config.py rbf_ffn/configs/
git commit -m "feat: extend RBFFFNConfig with model_type, ffn_hidden, and training hyperparameters; strip num_classes from YAMLs"
```

---

### Task 2: Patch RBFFFN — RMSNorm and bias=False on down_proj

**Context:** Two one-line changes to `rbf_ffn/models/rbf_ffn.py`:
1. `self.norm = nn.LayerNorm(D)` → `self.norm = nn.RMSNorm(D)` — matches Llama convention, keeps the intentional double-norm (see spec).
2. `nn.Linear(down_in, D)` → `nn.Linear(down_in, D, bias=False)` — eliminates a confound between baseline and RBF variants.

**Files:**
- Modify: `rbf_ffn/models/rbf_ffn.py`
- Modify: `rbf_ffn/tests/test_rbf_ffn.py`

- [ ] **Step 1: Write failing tests for the two changes**

Append to `rbf_ffn/tests/test_rbf_ffn.py`:

```python
def test_internal_norm_is_rmsnorm():
    """RBFFFN must use nn.RMSNorm internally, not LayerNorm."""
    ffn = make_ffn("G0")
    assert isinstance(ffn.norm, nn.RMSNorm), f"Expected RMSNorm, got {type(ffn.norm)}"


def test_down_proj_has_no_bias():
    """down_proj must have bias=False to match Llama no-bias convention."""
    for variant in ["G0", "G1A", "G1B", "G2"]:
        ffn = make_ffn(variant)
        assert ffn.down_proj.bias is None, f"{variant}: down_proj still has bias"
```

- [ ] **Step 2: Run new tests — verify they fail**

```bash
pytest rbf_ffn/tests/test_rbf_ffn.py::test_internal_norm_is_rmsnorm rbf_ffn/tests/test_rbf_ffn.py::test_down_proj_has_no_bias -v
```

Expected: `FAILED` — norm is LayerNorm, bias is not None.

- [ ] **Step 3: Patch rbf_ffn.py**

In `rbf_ffn/models/rbf_ffn.py`, make two edits:

Change line `self.norm = nn.LayerNorm(D)` to:
```python
self.norm = nn.RMSNorm(D)
```

Change the `down_proj` block (currently initialises kaiming + zeros bias) to:
```python
down_in = D * K if cfg.gate_variant != "G2" else D
self.down_proj = nn.Linear(down_in, D, bias=False)
nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
```

Remove the `nn.init.zeros_(self.down_proj.bias)` line entirely.

- [ ] **Step 4: Run new tests — verify they pass**

```bash
pytest rbf_ffn/tests/test_rbf_ffn.py::test_internal_norm_is_rmsnorm rbf_ffn/tests/test_rbf_ffn.py::test_down_proj_has_no_bias -v
```

Expected: `2 passed`

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest rbf_ffn/tests/ -v
```

Expected: all previously passing tests still pass, **with one known expected failure**: `test_transformer_block.py::test_residual_connection_present` calls `block.ffn.down_proj.bias.zero_()` which will raise `AttributeError` now that `bias=None`. This is intentional — the entire `test_transformer_block.py` is rewritten in Task 5 to remove that call. Confirm the failure is only that test, then proceed.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/models/rbf_ffn.py rbf_ffn/tests/test_rbf_ffn.py
git commit -m "feat: patch RBFFFN — RMSNorm internal norm, down_proj bias=False"
```

---

## Chunk 2: Shared attention stack

### Task 3: attention.py — RotaryEmbedding and CausalSelfAttention

**Files:**
- Create: `rbf_ffn/models/attention.py`
- Create: `rbf_ffn/tests/test_attention.py`

- [ ] **Step 1: Write failing tests**

```python
# rbf_ffn/tests/test_attention.py
import math
import torch
import torch.nn as nn
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.attention import RotaryEmbedding, CausalSelfAttention

B, N, D, H = 2, 16, 64, 4   # small dims for fast tests
HEAD_DIM = D // H            # 16


@pytest.fixture
def cfg():
    return RBFFFNConfig(d_model=D, n_heads=H, dropout=0.0)


# ── RotaryEmbedding ───────────────────────────────────────────────────────────

def test_rope_output_shape():
    rope = RotaryEmbedding(head_dim=HEAD_DIM)
    x = torch.randn(B, H, N, HEAD_DIM)
    out = rope(x)
    assert out.shape == (B, H, N, HEAD_DIM)


def test_rope_preserves_norm():
    """RoPE is a rotation; it must preserve the L2 norm of each head vector."""
    rope = RotaryEmbedding(head_dim=HEAD_DIM)
    x = torch.randn(B, H, N, HEAD_DIM)
    out = rope(x)
    norms_in  = x.norm(dim=-1)
    norms_out = out.norm(dim=-1)
    assert torch.allclose(norms_in, norms_out, atol=1e-5)


def test_rope_position_dependent():
    """Two tokens at different positions must get different rotations."""
    rope = RotaryEmbedding(head_dim=HEAD_DIM)
    x = torch.ones(1, 1, 2, HEAD_DIM)   # same vector at positions 0 and 1
    out = rope(x)
    assert not torch.allclose(out[:, :, 0, :], out[:, :, 1, :])


# ── CausalSelfAttention ───────────────────────────────────────────────────────

def test_attn_output_shape(cfg):
    attn = CausalSelfAttention(cfg)
    x = torch.randn(B, N, D)
    assert attn(x).shape == (B, N, D)


def test_attn_no_bias(cfg):
    """All projection weights must have bias=None."""
    attn = CausalSelfAttention(cfg)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attn, name).bias is None, f"{name} has unexpected bias"


def test_attn_causal_mask(cfg):
    """Output at position i must not depend on positions j > i."""
    attn = CausalSelfAttention(cfg)
    attn.eval()
    x = torch.randn(1, N, D)
    out_full = attn(x)

    # Corrupt all tokens after position 0 — position 0 output must be unchanged
    x_corrupt = x.clone()
    x_corrupt[:, 1:, :] = torch.randn_like(x_corrupt[:, 1:, :])
    out_corrupt = attn(x_corrupt)

    assert torch.allclose(out_full[:, 0, :], out_corrupt[:, 0, :], atol=1e-5)


def test_attn_gradient_flows(cfg):
    attn = CausalSelfAttention(cfg)
    x = torch.randn(B, N, D, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attn, name).weight.grad is not None


def test_attn_flash_flag_matches_hardware(cfg):
    """_use_flash must be True iff CUDA is present and flash_sdp is enabled."""
    attn = CausalSelfAttention(cfg)
    expected = torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled()
    assert attn._use_flash == expected
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
pytest rbf_ffn/tests/test_attention.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'rbf_ffn.models.attention'`

- [ ] **Step 3: Implement attention.py**

```python
# rbf_ffn/models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdp_kernel
from rbf_ffn.config import RBFFFNConfig

# Backend preference order: FlashAttention → MemEfficient → Math fallback.
# PyTorch tries each in order and picks the first that is supported for the
# given dtype/device/sequence-length at runtime.
_FLASH_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def _flash_available() -> bool:
    """Return True if the FlashAttention SDPA backend is globally enabled on CUDA."""
    return torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension: [x1, x2] → [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies position-dependent rotations to Q and K tensors.
    No learnable parameters; sin/cos cache is built lazily on first call.

    Input/output: (B, n_heads, N, head_dim)
    """

    def __init__(self, head_dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # (N, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (N, head_dim)
        self._cos = emb.cos()                          # (N, head_dim)
        self._sin = emb.sin()
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_heads, N, head_dim)"""
        seq_len = x.shape[2]
        if self._cos is None or seq_len > self._cached_len:
            self._build_cache(seq_len, x.device)
        cos = self._cos[:seq_len].unsqueeze(0).unsqueeze(0)   # (1, 1, N, head_dim)
        sin = self._sin[:seq_len].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (_rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    No bias on any projection. Causal mask via F.scaled_dot_product_attention
    with is_causal=True (no explicit mask tensor stored).

    On CUDA when FlashAttention is available, uses sdp_kernel to explicitly
    prefer the FlashAttention backend with graceful fallback to MemEfficient
    and Math backends. On CPU, delegates backend selection to PyTorch.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape
        def split_heads(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(split_heads(self.q_proj(x)))   # (B, H, N, head_dim)
        k = self.rope(split_heads(self.k_proj(x)))
        v = split_heads(self.v_proj(x))

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdp_kernel(_FLASH_BACKENDS):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
pytest rbf_ffn/tests/test_attention.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/models/attention.py rbf_ffn/tests/test_attention.py
git commit -m "feat: implement RotaryEmbedding and CausalSelfAttention (RoPE + SDPA, FlashAttention when available)"
```

---

## Chunk 3: SwiGLUFFN and updated transformer blocks

### Task 4: SwiGLUFFN (Llama baseline FFN)

**Files:**
- Create: `rbf_ffn/models/llama_ffn.py`
- Create: `rbf_ffn/tests/test_llama_ffn.py`

- [ ] **Step 1: Write failing tests**

```python
# rbf_ffn/tests/test_llama_ffn.py
import torch
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.llama_ffn import SwiGLUFFN

B, N, D = 2, 10, 64
FFN_HIDDEN = 172  # 8/3 * 64 ≈ 170, rounded to nearest even


@pytest.fixture
def ffn():
    cfg = RBFFFNConfig(d_model=D, ffn_hidden=FFN_HIDDEN)
    return SwiGLUFFN(cfg)


def test_output_shape(ffn):
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_no_bias_on_any_projection(ffn):
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert getattr(ffn, name).bias is None, f"{name} has unexpected bias"


def test_gradient_flows(ffn):
    x = torch.randn(B, N, D, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert getattr(ffn, name).weight.grad is not None


def test_projection_shapes(ffn):
    assert ffn.gate_proj.in_features  == D
    assert ffn.gate_proj.out_features == FFN_HIDDEN
    assert ffn.up_proj.in_features    == D
    assert ffn.up_proj.out_features   == FFN_HIDDEN
    assert ffn.down_proj.in_features  == FFN_HIDDEN
    assert ffn.down_proj.out_features == D
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest rbf_ffn/tests/test_llama_ffn.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'rbf_ffn.models.llama_ffn'`

- [ ] **Step 3: Implement llama_ffn.py**

```python
# rbf_ffn/models/llama_ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from rbf_ffn.config import RBFFFNConfig


class SwiGLUFFN(nn.Module):
    """
    Llama-style SwiGLU feed-forward network.

        gate = SiLU(gate_proj(x))
        up   = up_proj(x)
        out  = down_proj(gate * up)

    No bias on any projection (Llama convention).
    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.ffn_hidden
        self.gate_proj = nn.Linear(D, H, bias=False)
        self.up_proj   = nn.Linear(D, H, bias=False)
        self.down_proj = nn.Linear(H, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
pytest rbf_ffn/tests/test_llama_ffn.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/models/llama_ffn.py rbf_ffn/tests/test_llama_ffn.py
git commit -m "feat: implement SwiGLUFFN (Llama baseline FFN, no bias)"
```

---

### Task 5: Update transformer_block.py — LlamaBlock and RBFBlock

**Context:** The existing `RBFTransformerBlock` (using `nn.MultiheadAttention`) is replaced by `RBFBlock` (using `CausalSelfAttention`). `LlamaBlock` is added. The old class name is removed; all downstream references (only in tests) must be updated too.

**Files:**
- Modify: `rbf_ffn/models/transformer_block.py`
- Modify: `rbf_ffn/tests/test_transformer_block.py`

- [ ] **Step 1: Rewrite transformer_block.py**

```python
# rbf_ffn/models/transformer_block.py
import torch
import torch.nn as nn
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.attention import CausalSelfAttention
from rbf_ffn.models.llama_ffn import SwiGLUFFN
from rbf_ffn.models.rbf_ffn import RBFFFN


class LlamaBlock(nn.Module):
    """
    Llama-style transformer block with SwiGLU FFN.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RBFBlock(nn.Module):
    """
    Transformer block with RBF-FFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is RBFFFN

    Double normalisation: norm2 (outer, this block) + RBFFFN.norm (inner).
    Both are intentional — see spec. Do NOT remove either.
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = RBFFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

- [ ] **Step 2: Rewrite test_transformer_block.py**

```python
# rbf_ffn/tests/test_transformer_block.py
import torch
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock

D, H, B, N = 32, 4, 2, 16


def make_llama(variant: str = "G0") -> LlamaBlock:
    return LlamaBlock(RBFFFNConfig(d_model=D, n_heads=H, gate_variant=variant, dropout=0.0))


def make_rbf(variant: str = "G0") -> RBFBlock:
    return RBFBlock(RBFFFNConfig(d_model=D, n_heads=H, gate_variant=variant, dropout=0.0))


# ── LlamaBlock ────────────────────────────────────────────────────────────────

def test_llama_output_shape():
    assert make_llama()(torch.randn(B, N, D)).shape == (B, N, D)


def test_llama_gradient_flows():
    x = torch.randn(B, N, D, requires_grad=True)
    make_llama()(x).sum().backward()
    assert x.grad is not None


def test_llama_residual_connection():
    """Zero FFN and attn output projections → output equals input."""
    block = make_llama()
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)


# ── RBFBlock ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_rbf_output_shape(variant):
    assert make_rbf(variant)(torch.randn(B, N, D)).shape == (B, N, D)


@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_rbf_gradient_flows(variant):
    x = torch.randn(B, N, D, requires_grad=True)
    make_rbf(variant)(x).sum().backward()
    assert x.grad is not None


def test_rbf_residual_connection():
    """Zero FFN and attn output projections → output equals input."""
    block = make_rbf()
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)


def test_rbf_norm1_norm2_are_rmsnorm():
    block = make_rbf()
    assert isinstance(block.norm1, torch.nn.RMSNorm)
    assert isinstance(block.norm2, torch.nn.RMSNorm)
```

- [ ] **Step 3: Run new transformer block tests — verify they pass**

```bash
pytest rbf_ffn/tests/test_transformer_block.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run full test suite**

```bash
pytest rbf_ffn/tests/ -v
```

Expected: all tests pass. Note: tests in `test_rbf_ffn.py` that previously imported `RBFTransformerBlock` are now removed (the file was fully replaced).

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/models/transformer_block.py rbf_ffn/tests/test_transformer_block.py
git commit -m "feat: add LlamaBlock and RBFBlock with CausalSelfAttention and RMSNorm"
```

---

## Chunk 4: CausalLM model

### Task 6: model.py — full causal language model

**Files:**
- Create: `rbf_ffn/models/model.py`
- Create: `rbf_ffn/tests/test_model.py`

- [ ] **Step 1: Write failing tests**

```python
# rbf_ffn/tests/test_model.py
import torch
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.model import CausalLM

B, N = 2, 16
VOCAB = 256    # small for fast tests
D, H, L = 32, 4, 2


def make_model(model_type: str = "rbf", gate_variant: str = "G0") -> CausalLM:
    cfg = RBFFFNConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        model_type=model_type, gate_variant=gate_variant,
        ffn_hidden=86,   # 8/3 * 32 ≈ 85
        dropout=0.0,
    )
    return CausalLM(cfg)


def test_baseline_output_shape():
    model = make_model("baseline")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits = model(tokens)
    assert logits.shape == (B, N, VOCAB)


@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_rbf_output_shape(variant):
    tokens = torch.randint(0, VOCAB, (B, N))
    assert make_model("rbf", variant)(tokens).shape == (B, N, VOCAB)


def test_weight_tying():
    """LM head weight must be the same tensor object as the embedding weight."""
    model = make_model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_weight_tying_shared_memory():
    """A write to embedding weight must be reflected in lm_head weight."""
    model = make_model()
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = 999.0
    assert model.lm_head.weight[0, 0].item() == 999.0


def test_no_duplicate_params_in_optimizer_groups():
    """The tied embedding/lm_head weight must appear exactly once across groups."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids)), "Duplicate parameters in optimizer groups"


def test_embedding_in_adamw_not_muon():
    """Token embedding weight must be in AdamW group (not Muon)."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model()
    muon_params, _ = build_optimizer_groups(model)
    emb_id = id(model.token_embedding.weight)
    assert emb_id not in {id(p) for p in muon_params}


def test_all_2d_non_embedding_non_sigma_in_muon():
    """Every 2D param that is not the embedding and not sigma_raw must be in Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("rbf", "G0")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    emb_id = id(model.token_embedding.weight)
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if "sigma_raw" in name or id(param) == emb_id:
            assert id(param) in adamw_ids, f"{name} should be AdamW"
        elif param.ndim == 2:
            assert id(param) in muon_ids, f"{name} (2D) should be Muon"


def test_gradient_flows_baseline():
    model = make_model("baseline")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits = model(tokens)
    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None


def test_gradient_flows_rbf():
    model = make_model("rbf")
    tokens = torch.randint(0, VOCAB, (B, N))
    model(tokens).sum().backward()
    assert model.token_embedding.weight.grad is not None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest rbf_ffn/tests/test_model.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'rbf_ffn.models.model'`

- [ ] **Step 3: Implement model.py**

```python
# rbf_ffn/models/model.py
from __future__ import annotations
import torch
import torch.nn as nn
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock


def build_optimizer_groups(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split model parameters into Muon and AdamW groups.

    Rules (first match wins, applied after deduplication by tensor id):
      1. "sigma_raw" in name           → AdamW
      2. param is token embedding weight → AdamW
      3. param.ndim == 2               → Muon
      4. else                          → AdamW

    Returns (muon_params, adamw_params).
    """
    emb_id = id(model.token_embedding.weight)   # type: ignore[attr-defined]
    seen: set[int] = set()
    muon: list[torch.Tensor] = []
    adamw: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if "sigma_raw" in name:
            adamw.append(param)
        elif pid == emb_id:
            adamw.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)

    return muon, adamw


class CausalLM(nn.Module):
    """
    Causal language model.

        token_embedding → N × Block → RMSNorm → lm_head (weight-tied)

    Block type is selected by cfg.model_type:
        "baseline" → LlamaBlock (SwiGLU FFN)
        "rbf"      → RBFBlock   (RBF-FFN)
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        BlockClass = LlamaBlock if cfg.model_type == "baseline" else RBFBlock
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([BlockClass(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: lm_head shares the embedding matrix
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N) integer token ids
        returns: logits (B, N, vocab_size)
        """
        x = self.token_embedding(tokens)   # (B, N, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)             # (B, N, vocab_size)
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
pytest rbf_ffn/tests/test_model.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full test suite**

```bash
pytest rbf_ffn/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/models/model.py rbf_ffn/tests/test_model.py
git commit -m "feat: implement CausalLM with LlamaBlock/RBFBlock and weight-tied LM head"
```

---

## Chunk 5: Data pipeline

### Task 7: data.py — WikiText-103 dataset

**Files:**
- Create: `rbf_ffn/data.py`
- Create: `rbf_ffn/tests/test_data.py`
- Modify: `.gitignore` (add `rbf_ffn/data_cache/`)

- [ ] **Step 1: Add data_cache to .gitignore**

```bash
echo "rbf_ffn/data_cache/" >> .gitignore
```

- [ ] **Step 2: Write tests**

```python
# rbf_ffn/tests/test_data.py
"""
Tests for the data pipeline. These tests do NOT download WikiText-103.
They verify the chunking logic and Dataset contract using synthetic data.
"""
import torch
import pytest
from rbf_ffn.data import chunk_tokens, TokenDataset


def test_chunk_tokens_basic():
    tokens = list(range(20))
    chunks = chunk_tokens(tokens, seq_len=5)
    assert chunks.shape == (4, 5)
    assert chunks[0].tolist() == [0, 1, 2, 3, 4]
    assert chunks[3].tolist() == [15, 16, 17, 18, 19]


def test_chunk_tokens_discards_remainder():
    tokens = list(range(22))   # 22 tokens, seq_len=5 → 4 full chunks, 2 discarded
    chunks = chunk_tokens(tokens, seq_len=5)
    assert chunks.shape == (4, 5)


def test_chunk_tokens_exact_multiple():
    tokens = list(range(15))
    chunks = chunk_tokens(tokens, seq_len=5)
    assert chunks.shape == (3, 5)


def test_token_dataset_len():
    data = torch.arange(40).view(8, 5)   # 8 sequences of length 5
    ds = TokenDataset(data)
    assert len(ds) == 8


def test_token_dataset_getitem():
    data = torch.arange(40).view(8, 5)
    ds = TokenDataset(data)
    item = ds[0]
    assert item.shape == (5,)
    assert item.dtype == torch.long


def test_token_dataset_values():
    data = torch.arange(40, dtype=torch.long).view(8, 5)
    ds = TokenDataset(data)
    assert ds[2].tolist() == [10, 11, 12, 13, 14]
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
pytest rbf_ffn/tests/test_data.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'rbf_ffn.data'`

- [ ] **Step 4: Implement data.py**

```python
# rbf_ffn/data.py
"""
WikiText-103 data pipeline.

Usage:
    from rbf_ffn.data import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

On first call, downloads and tokenises WikiText-103 (~5 minutes).
Subsequent calls load from cache in rbf_ffn/data_cache/.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

_CACHE_DIR = Path(__file__).parent / "data_cache"


def chunk_tokens(tokens: list[int], seq_len: int) -> torch.Tensor:
    """
    Split a flat token list into non-overlapping chunks of seq_len.
    The trailing remainder (< seq_len tokens) is discarded.

    Returns: LongTensor of shape (n_chunks, seq_len)
    """
    t = torch.tensor(tokens, dtype=torch.long)
    n_chunks = len(t) // seq_len
    return t[: n_chunks * seq_len].view(n_chunks, seq_len)


class TokenDataset(Dataset):
    """Simple wrapper around a (N, seq_len) LongTensor."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _load_split(split: str, seq_len: int) -> torch.Tensor:
    """
    Load a tokenised split from cache, or build and cache it.

    split: "train" | "validation" | "test"
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}_r50k_{seq_len}.pt"

    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    # Lazy imports so the rest of the module works without these packages
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("r50k_base")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    texts = [row["text"] for row in dataset if row["text"].strip() != ""]
    full_text = "\n".join(texts)
    tokens = enc.encode(full_text)

    chunks = chunk_tokens(tokens, seq_len)
    torch.save(chunks, cache_file)
    print(f"Cached {len(chunks):,} sequences → {cache_file}")
    return chunks


def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for WikiText-103.

    cfg must have: seq_len, batch_size
    """
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def _make_loader(split: str, shuffle: bool, drop_last: bool) -> DataLoader:
        data = _load_split(split, cfg.seq_len)
        ds = TokenDataset(data)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
            generator=g if shuffle else None,
        )

    train_loader = _make_loader("train",      shuffle=True,  drop_last=True)
    val_loader   = _make_loader("validation", shuffle=False, drop_last=False)
    test_loader  = _make_loader("test",       shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
```

- [ ] **Step 5: Run tests — verify they all pass**

```bash
pytest rbf_ffn/tests/test_data.py -v
```

Expected: `6 passed` (no network access required — synthetic data only)

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/data.py rbf_ffn/tests/test_data.py .gitignore
git commit -m "feat: implement WikiText-103 data pipeline with chunking and disk cache"
```

---

## Chunk 6: Training loop, configs, and smoke tests

### Task 8: Update YAML configs

**Context:** All YAML files must be updated to include the new config fields. The `RBFFFNConfig` dataclass was updated in Task 1. Update all configs now.

**Files:**
- Modify: `rbf_ffn/configs/g0_baseline.yaml`
- Modify: `rbf_ffn/configs/g1a_cross_kernel.yaml`
- Modify: `rbf_ffn/configs/g1b_input_driven.yaml`
- Modify: `rbf_ffn/configs/g2_sinkhorn.yaml`
- Modify: `rbf_ffn/configs/sigma_b_per_center.yaml`
- Modify: `rbf_ffn/configs/sigma_c_per_dim.yaml`
- Create: `rbf_ffn/configs/baseline.yaml`

- [ ] **Step 1: Write baseline.yaml**

```yaml
# rbf_ffn/configs/baseline.yaml
# Llama SwiGLU baseline — gate_variant and sigma_variant are present for
# uniform directory naming but are ignored by the model (model_type: baseline).
model_type: baseline
gate_variant: G0
sigma_variant: global
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sinkhorn_iters: 20
vocab_size: 50257
seq_len: 512
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

- [ ] **Step 2: Rewrite the six existing ablation configs**

`rbf_ffn/configs/g0_baseline.yaml`:
```yaml
model_type: rbf
gate_variant: G0
sigma_variant: global
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sinkhorn_iters: 20
vocab_size: 50257
seq_len: 512
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

`rbf_ffn/configs/g1a_cross_kernel.yaml`:
```yaml
model_type: rbf
gate_variant: G1A
sigma_variant: global
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sinkhorn_iters: 20
vocab_size: 50257
seq_len: 512
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

`rbf_ffn/configs/g1b_input_driven.yaml`:
```yaml
model_type: rbf
gate_variant: G1B
sigma_variant: global
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sinkhorn_iters: 20
vocab_size: 50257
seq_len: 512
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

`rbf_ffn/configs/g2_sinkhorn.yaml`:
```yaml
model_type: rbf
gate_variant: G2
sigma_variant: global
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sinkhorn_iters: 20
vocab_size: 50257
seq_len: 512
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

`rbf_ffn/configs/sigma_b_per_center.yaml`:
```yaml
model_type: rbf
gate_variant: G0
sigma_variant: per_center
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sinkhorn_iters: 20
vocab_size: 50257
seq_len: 512
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

`rbf_ffn/configs/sigma_c_per_dim.yaml`:
```yaml
model_type: rbf
gate_variant: G0
sigma_variant: per_dim
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sinkhorn_iters: 20
vocab_size: 50257
seq_len: 512
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

- [ ] **Step 3: Update test_load_config_values_match_yaml in test_config.py**

After Step 2 rewrites `g0_baseline.yaml` to `d_model: 256, n_layers: 6`, the existing `test_load_config_values_match_yaml` test (which asserts `cfg.d_model == 64` and `cfg.n_layers == 2`) will fail. Update it to match the new values:

In `rbf_ffn/tests/test_config.py`, find the `test_load_config_values_match_yaml` function and update the assertions:

```python
def test_load_config_values_match_yaml():
    cfg = load_config(Path("rbf_ffn/configs/g0_baseline.yaml"))
    assert cfg.d_model == 256
    assert cfg.K == 5
    assert abs(cfg.sigma_init - 0.5) < 1e-9
    assert cfg.n_layers == 6
```

- [ ] **Step 4: Verify all configs load correctly**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -c "
from rbf_ffn.config import load_config
import glob
for f in sorted(glob.glob('rbf_ffn/configs/*.yaml')):
    cfg = load_config(f)
    print(f'{f}: model_type={cfg.model_type}, gate={cfg.gate_variant}, sigma={cfg.sigma_variant}')
"
```

Expected: 7 lines printed, one per config, no errors.

- [ ] **Step 5: Run config tests**

```bash
pytest rbf_ffn/tests/test_config.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/configs/ rbf_ffn/tests/test_config.py
git commit -m "feat: update all YAML configs with training hyperparameters and model_type"
```

---

### Task 9: Rewrite train.py

**Files:**
- Modify: `rbf_ffn/train.py`

- [ ] **Step 1: Rewrite train.py**

```python
# rbf_ffn/train.py
"""
Training entry point for RBF-FFN WikiText-103 ablation experiments.

Usage:
    python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml
    python -m rbf_ffn.train --config rbf_ffn/configs/g0_baseline.yaml --n_epochs 5
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR

from rbf_ffn.config import RBFFFNConfig, load_config
from rbf_ffn.data import get_dataloaders
from rbf_ffn.models.model import CausalLM, build_optimizer_groups


def get_experiment_dir(cfg: RBFFFNConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (
        f"{stamp}_{cfg.model_type}_{cfg.gate_variant}_{cfg.sigma_variant}"
        f"_d{cfg.d_model}_K{cfg.K}"
    )
    path = Path(__file__).parent / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def collect_sigma_stats(model: CausalLM) -> dict:
    """Collect mean/std of all sigma values (softplus of sigma_raw) across RBF layers.

    sigma_std=0.0 for the global variant (each sigma_raw is a scalar per layer).
    """
    all_sigma = []
    all_scalar = True
    for name, param in model.named_parameters():
        if "sigma_raw" in name:
            all_sigma.append(F.softplus(param).detach().flatten())
            if param.numel() > 1:
                all_scalar = False
    if not all_sigma:
        return {}
    sigma_cat = torch.cat(all_sigma)
    return {
        "sigma_mean": sigma_cat.mean().item(),
        "sigma_std":  0.0 if all_scalar else sigma_cat.std().item(),
    }


@torch.no_grad()
def evaluate(model: CausalLM, loader, device: torch.device) -> tuple[float, float]:
    """Returns (val_loss, val_ppl) in nats/token."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        n_tokens = inputs.numel()
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )
        loss_sum    += loss.item() * n_tokens
        token_count += n_tokens
    val_loss = loss_sum / token_count
    return val_loss, math.exp(val_loss)


def train(cfg: RBFFFNConfig, config_path: Path, n_epochs: int | None = None) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if n_epochs is not None:
        cfg.n_epochs = n_epochs

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = get_experiment_dir(cfg)
    shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps     = cfg.n_epochs * steps_per_epoch
    warmup_steps    = int(cfg.warmup_ratio * total_steps)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    muon_params, adamw_params = build_optimizer_groups(model)
    muon  = Muon( muon_params,  lr=cfg.muon_lr, momentum=0.95)
    adamw = AdamW(adamw_params, lr=cfg.adamw_lr,
                  weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))

    lr_fn = make_lr_lambda(warmup_steps, total_steps)
    sched_muon  = LambdaLR(muon,  lr_fn)
    sched_adamw = LambdaLR(adamw, lr_fn)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    val_loss, val_ppl = float("inf"), float("inf")   # initialised in case n_epochs=0

    def save_checkpoint(name: str, epoch: int, val_loss: float, val_ppl: float):
        torch.save({
            "model":           model.state_dict(),
            "optimizer_muon":  muon.state_dict(),
            "optimizer_adamw": adamw.state_dict(),
            "scheduler_muon":  sched_muon.state_dict(),
            "scheduler_adamw": sched_adamw.state_dict(),
            "epoch":    epoch,
            "val_loss": val_loss,
            "val_ppl":  val_ppl,
        }, exp_dir / name)

    for epoch in range(cfg.n_epochs):
        model.train()
        loss_sum, token_count = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            batch   = batch.to(device)
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]
            n_tokens = inputs.numel()

            logits = model(inputs)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="mean",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            muon.step();  adamw.step()
            sched_muon.step(); sched_adamw.step()
            muon.zero_grad(); adamw.zero_grad()

            loss_sum    += loss.item() * n_tokens
            token_count += n_tokens

        train_loss = loss_sum / token_count
        train_ppl  = math.exp(train_loss)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        row: dict = {
            "epoch":        epoch,
            "train_loss":   train_loss,
            "train_ppl":    train_ppl,
            "val_loss":     val_loss,
            "val_ppl":      val_ppl,
            "epoch_time_s": epoch_time,
        }
        if cfg.model_type == "rbf":
            row.update(collect_sigma_stats(model))

        print(row)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint("checkpoint_best.pt", epoch, val_loss, val_ppl)

    save_checkpoint("checkpoint_final.pt", cfg.n_epochs - 1, val_loss, val_ppl)
    print(f"Done. Metrics → {metrics_path}")
    return exp_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train RBF-FFN on WikiText-103")
    p.add_argument("--config",   required=True, help="Path to YAML config")
    p.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    path   = Path(args.config)
    cfg    = load_config(path)
    train(cfg, config_path=path, n_epochs=args.n_epochs)
```

- [ ] **Step 2: Verify train.py imports without error**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -c "from rbf_ffn.train import train; print('import ok')"
```

Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add rbf_ffn/train.py
git commit -m "feat: rewrite train.py for WikiText-103 causal LM with Muon + cosine schedule"
```

---

### Task 10: End-to-end smoke tests for all configs

**Goal:** Confirm that every config runs for 1 step without error and produces the expected artifact structure. Uses `--n_epochs 1` to avoid downloading WikiText-103 — this test requires network access; skip on CI if unavailable.

- [ ] **Step 1: Run baseline smoke test (downloads WikiText-103 on first run)**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml --n_epochs 1
```

Expected: prints device, parameter count, epoch metrics, creates experiment directory with `config.yaml`, `metrics.jsonl`, `checkpoint_best.pt`, `checkpoint_final.pt`.

- [ ] **Step 2: Run all RBF ablation configs**

```bash
python -m rbf_ffn.train --config rbf_ffn/configs/g0_baseline.yaml --n_epochs 1
python -m rbf_ffn.train --config rbf_ffn/configs/g1a_cross_kernel.yaml --n_epochs 1
python -m rbf_ffn.train --config rbf_ffn/configs/g1b_input_driven.yaml --n_epochs 1
python -m rbf_ffn.train --config rbf_ffn/configs/g2_sinkhorn.yaml --n_epochs 1
python -m rbf_ffn.train --config rbf_ffn/configs/sigma_b_per_center.yaml --n_epochs 1
python -m rbf_ffn.train --config rbf_ffn/configs/sigma_c_per_dim.yaml --n_epochs 1
```

Expected: each run completes without error. RBF runs print `sigma_mean` and `sigma_std` in their metrics.

- [ ] **Step 3: Verify artifact structure**

```bash
find rbf_ffn/experiments/ \( -name "*.pt" -o -name "metrics.jsonl" -o -name "config.yaml" \) | sort | head -40
```

Expected: 7 experiment directories (one per config), each containing `config.yaml`, `metrics.jsonl`, `checkpoint_best.pt`, `checkpoint_final.pt`.

- [ ] **Step 4: Verify metrics.jsonl content for one run**

```bash
head -1 $(ls -d rbf_ffn/experiments/*baseline*G0*global* | head -1)/metrics.jsonl
```

Expected: JSON line with keys `epoch`, `train_loss`, `train_ppl`, `val_loss`, `val_ppl`, `epoch_time_s`, and for RBF runs also `sigma_mean`, `sigma_std`.

- [ ] **Step 5: Run full unit test suite**

```bash
pytest rbf_ffn/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/experiments/.gitkeep
git commit -m "chore: confirm all 7 ablation configs smoke-test successfully"
```
