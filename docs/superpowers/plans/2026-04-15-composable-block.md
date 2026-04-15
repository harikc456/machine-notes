# Composable Block Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the proliferating per-combination block classes with a single `TransformerBlock` that assembles its attention and FFN from two independent registries, controlled by `attn_type` and `ffn_type` config fields.

**Architecture:** `ModelConfig` gains `attn_type` and `ffn_type` fields; `model_type` becomes a deprecated optional alias that is translated to the two new fields in `__post_init__`. `ATTN_REGISTRY` in `attention.py` and `FFN_REGISTRY` in `transformer_block.py` map string keys to classes. A single `TransformerBlock` replaces all ~10 named block classes. `model.py` drops the `BlockClass` dict entirely.

**Tech Stack:** Python 3.12, PyTorch, dataclasses, YAML (PyYAML)

---

## File Map

| File | Change |
|------|--------|
| `rbf_ffn/config.py` | Add `attn_type`, `ffn_type`; make `model_type` optional deprecated alias |
| `rbf_ffn/models/attention.py` | Add `ATTN_REGISTRY` dict at module bottom |
| `rbf_ffn/models/transformer_block.py` | Add `FFN_REGISTRY`; add `TransformerBlock`; delete the 10 named block classes |
| `rbf_ffn/models/model.py` | Drop `BlockClass` dict; import `TransformerBlock` instead of all named blocks |
| `rbf_ffn/train.py` | Update `get_experiment_dir` to use `attn_type`/`ffn_type` instead of `model_type` |
| `rbf_ffn/tests/test_config.py` | Update deprecated `model_type` test; add tests for new fields |
| `rbf_ffn/tests/test_transformer_block.py` | Rewrite fixtures to use `TransformerBlock` + `attn_type`/`ffn_type` |
| `rbf_ffn/tests/test_model.py` | Add `attn_type`/`ffn_type` smoke test; existing `model_type` tests keep working via compat layer |
| `rbf_ffn/configs/*.yaml` (27 files) | Replace `model_type` with `attn_type` + `ffn_type` |

---

## Task 1: Update `config.py` — add `attn_type`, `ffn_type`, deprecation alias

**Files:**
- Modify: `rbf_ffn/config.py`
- Modify: `rbf_ffn/tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Add to `rbf_ffn/tests/test_config.py`:

```python
def test_attn_type_default():
    cfg = ModelConfig()
    assert cfg.attn_type == "standard"


def test_ffn_type_default():
    cfg = ModelConfig()
    assert cfg.ffn_type == "swiglu"


def test_model_type_compat_baseline():
    """model_type='baseline' must translate to attn_type='standard', ffn_type='swiglu'."""
    cfg = ModelConfig(model_type="baseline")
    assert cfg.attn_type == "standard"
    assert cfg.ffn_type == "swiglu"


def test_model_type_compat_rationalglu():
    cfg = ModelConfig(model_type="rationalglu")
    assert cfg.attn_type == "standard"
    assert cfg.ffn_type == "rationalglu"


def test_model_type_compat_polar_attn():
    cfg = ModelConfig(model_type="polar_attn")
    assert cfg.attn_type == "polar"
    assert cfg.ffn_type == "swiglu"


def test_model_type_compat_polar_full():
    cfg = ModelConfig(model_type="polar_full")
    assert cfg.attn_type == "polar"
    assert cfg.ffn_type == "polar"


def test_model_type_compat_polar_mlp():
    cfg = ModelConfig(model_type="polar_mlp")
    assert cfg.attn_type == "standard"
    assert cfg.ffn_type == "polar"


def test_model_type_compat_xsa():
    cfg = ModelConfig(model_type="xsa")
    assert cfg.attn_type == "xsa"
    assert cfg.ffn_type == "swiglu"


def test_explicit_attn_ffn_type_no_model_type():
    """attn_type + ffn_type set directly, no model_type needed."""
    cfg = ModelConfig(attn_type="xsa", ffn_type="pfd_rationalglu")
    assert cfg.attn_type == "xsa"
    assert cfg.ffn_type == "pfd_rationalglu"
    assert cfg.model_type is None


def test_load_config_attn_ffn_type(tmp_path):
    """YAML with attn_type + ffn_type loads correctly."""
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("attn_type: xsa\nffn_type: pfd_rationalglu\n")
    cfg = load_config(yaml_path)
    assert cfg.attn_type == "xsa"
    assert cfg.ffn_type == "pfd_rationalglu"
```

Also update the existing `test_load_config_partial_yaml_uses_defaults` test — it previously asserted `cfg.model_type == "rationalglu"` which should still pass since `model_type` is preserved on the dataclass. No change needed there.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_config.py -v 2>&1 | tail -20
```

Expected: FAIL on all new tests with `AttributeError: 'ModelConfig' has no attribute 'attn_type'`

- [ ] **Step 3: Implement the changes in `config.py`**

Replace the full contents of `rbf_ffn/config.py` with:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


# Maps deprecated model_type → (attn_type, ffn_type)
_MODEL_TYPE_MAP: dict[str, tuple[str, str]] = {
    "baseline":                 ("standard", "swiglu"),
    "rational":                 ("standard", "rational"),
    "rationalglu":              ("standard", "rationalglu"),
    "pfd_rational":             ("standard", "pfd_rational"),
    "pfd_rationalglu":          ("standard", "pfd_rationalglu"),
    "first_order_pfd_rational": ("standard", "first_order_pfd_rational"),
    "polar_mlp":                ("standard", "polar"),
    "polar_attn":               ("polar",    "swiglu"),
    "polar_full":               ("polar",    "polar"),
    "xsa":                      ("xsa",      "swiglu"),
}


@dataclass
class ModelConfig:
    # Model dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # Attention
    qk_norm: bool = False          # Enable QK normalization in attention
    qkv_silu: bool = False         # Apply SiLU after Q, K, V projections

    # Sequence / vocab
    seq_len: int = 512
    vocab_size: int = 50257

    # Composable block type
    attn_type: str = "standard"    # "standard" | "polar" | "xsa"
    ffn_type: str = "swiglu"       # "swiglu" | "rational" | "rationalglu" | "pfd_rational" | "pfd_rationalglu" | "first_order_pfd_rational" | "polar"

    # Deprecated: use attn_type + ffn_type instead.
    # If set, translated to attn_type + ffn_type in __post_init__.
    model_type: str | None = None

    ffn_hidden: int = 688          # FFN hidden dim (SwiGLU / RationalFFN)
    pfd_n: int = 4                 # Number of partial fraction terms for PFDRational* models
    pre_lm_head_silu: bool = False # Apply SiLU activation before lm_head

    # Embedding / LM head
    tie_embeddings: bool = True        # If False, lm_head gets its own weight matrix (not shared with token_embedding)

    # Kronecker-factored LM head
    lm_head_kronecker: bool = False    # Replace lm_head with Kronecker-factored projection

    # Kronecker-factored MLP projections
    kronecker_mlp: bool = False        # Replace nn.Linear in MLP/FFN layers with KroneckerLinear
    kronecker_delta_mlp: bool = False      # Replace up_proj/down_proj with KroneckerDeltaLinear
    kronecker_delta_rank: int = 16         # Rank of the low-rank delta pathway

    # KromHC head mixing
    use_kromhc: bool = False           # wrap any block with KromHC head mixing
    kromhc_mixer_hidden: int = 32      # hidden dim of per-factor weight MLP

    # Weight normalization
    linear_weight_norm: bool = False   # Normalise each linear layer's weight rows after every optimizer step
    linear_weight_norm_value: float = 2.0  # Target L2 norm per output neuron
    linear_weight_norm_max_only: bool = False  # Scale down only; do not scale up if norm is below target

    # Activation coefficient normalization
    activation_norm: bool = False      # Normalise rational/PFD activation coefficients to L2 norm 2.0 after every optimizer step

    # Adaptive weight normalization (depth-based)
    adaptive_weight_norm: bool = False
    adaptive_norm_early: float = 2.5   # target norm at layer 0
    adaptive_norm_late: float = 1.2    # target norm at layer L-1 (must be >= 1.0)
    adaptive_norm_gamma: float = 0.3   # max phase correction magnitude
    adaptive_norm_beta: float = 5.0    # tanh sensitivity to gap derivative
    adaptive_norm_alpha: float = 0.9   # EMA smoothing factor for log-gap

    # Training
    seed: int = 42
    n_epochs: int = 10
    batch_size: int = 32
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    adamw_wd: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
    grad_accum_steps: int = 1      # mini-batches per optimizer step; 1 = no accumulation

    def __post_init__(self) -> None:
        if self.model_type is not None:
            if self.model_type not in _MODEL_TYPE_MAP:
                raise ValueError(
                    f"Unknown model_type '{self.model_type}'. "
                    f"Valid values: {sorted(_MODEL_TYPE_MAP)}. "
                    f"Prefer attn_type + ffn_type directly."
                )
            self.attn_type, self.ffn_type = _MODEL_TYPE_MAP[self.model_type]

        if self.adaptive_weight_norm:
            if self.adaptive_norm_late < 1.0:
                raise ValueError(
                    f"adaptive_norm_late must be >= 1.0, got {self.adaptive_norm_late}"
                )
            if self.adaptive_norm_early <= self.adaptive_norm_late:
                raise ValueError(
                    f"adaptive_norm_early must be > adaptive_norm_late, "
                    f"got adaptive_norm_early={self.adaptive_norm_early} <= adaptive_norm_late={self.adaptive_norm_late}"
                )


def load_config(path: str | Path) -> ModelConfig:
    """Load a ModelConfig from a YAML file.

    The YAML file may specify any subset of ModelConfig fields; unspecified
    fields take their dataclass defaults. Unknown keys raise ValueError.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return ModelConfig()
    valid_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return ModelConfig(**raw)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_config.py -v 2>&1 | tail -25
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/config.py rbf_ffn/tests/test_config.py
git commit -m "feat(rbf_ffn): add attn_type + ffn_type to ModelConfig with model_type compat"
```

---

## Task 2: Add `ATTN_REGISTRY` to `attention.py`

**Files:**
- Modify: `rbf_ffn/models/attention.py`
- Modify: `rbf_ffn/tests/test_attention.py`

- [ ] **Step 1: Write failing test**

Add to `rbf_ffn/tests/test_attention.py`:

```python
from rbf_ffn.models.attention import ATTN_REGISTRY


def test_attn_registry_keys():
    assert set(ATTN_REGISTRY.keys()) == {"standard", "polar", "xsa"}


def test_attn_registry_standard_is_causal_self_attention():
    from rbf_ffn.models.attention import CausalSelfAttention
    assert ATTN_REGISTRY["standard"] is CausalSelfAttention


def test_attn_registry_instantiates(cfg):
    for key, cls in ATTN_REGISTRY.items():
        attn = cls(cfg)
        x = torch.randn(B, N, D)
        assert attn(x).shape == (B, N, D), f"Registry key '{key}' produced wrong shape"
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_attention.py -v -k "registry" 2>&1 | tail -10
```

Expected: FAIL with `ImportError: cannot import name 'ATTN_REGISTRY'`

- [ ] **Step 3: Add `ATTN_REGISTRY` at the bottom of `attention.py`**

Append after the `CausalSelfAttention` class (at the end of the file):

```python
ATTN_REGISTRY: dict[str, type] = {
    "standard": CausalSelfAttention,
    "polar":    PolarAttention,
    "xsa":      ExclusiveSelfAttention,
}
```

- [ ] **Step 4: Run tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_attention.py -v 2>&1 | tail -15
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/models/attention.py rbf_ffn/tests/test_attention.py
git commit -m "feat(rbf_ffn): add ATTN_REGISTRY to attention.py"
```

---

## Task 3: Add `FFN_REGISTRY` and `TransformerBlock` to `transformer_block.py`

**Files:**
- Modify: `rbf_ffn/models/transformer_block.py`
- Modify: `rbf_ffn/tests/test_transformer_block.py`

- [ ] **Step 1: Write failing tests**

Replace the full contents of `rbf_ffn/tests/test_transformer_block.py` with:

```python
# rbf_ffn/tests/test_transformer_block.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import FFN_REGISTRY, TransformerBlock

D, H, B, N = 32, 4, 2, 16


def make_cfg(attn_type: str = "standard", ffn_type: str = "swiglu") -> ModelConfig:
    return ModelConfig(
        d_model=D, n_heads=H, dropout=0.0,
        attn_type=attn_type, ffn_type=ffn_type,
        ffn_hidden=86, pfd_n=4,
    )


# ── FFN_REGISTRY ──────────────────────────────────────────────────────────────

def test_ffn_registry_keys():
    expected = {
        "swiglu", "rational", "rationalglu",
        "pfd_rational", "pfd_rationalglu", "first_order_pfd_rational", "polar",
    }
    assert set(FFN_REGISTRY.keys()) == expected


def test_ffn_registry_swiglu_is_swiglu_ffn():
    from rbf_ffn.models.llama_ffn import SwiGLUFFN
    assert FFN_REGISTRY["swiglu"] is SwiGLUFFN


# ── TransformerBlock shape and basic behaviour ────────────────────────────────

@pytest.mark.parametrize("attn_type", ["standard", "polar", "xsa"])
@pytest.mark.parametrize("ffn_type", ["swiglu", "rational", "rationalglu", "pfd_rational", "pfd_rationalglu", "first_order_pfd_rational", "polar"])
def test_transformer_block_output_shape(attn_type, ffn_type):
    block = TransformerBlock(make_cfg(attn_type=attn_type, ffn_type=ffn_type))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_transformer_block_gradient_flows():
    block = TransformerBlock(make_cfg())
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None


def test_transformer_block_residual_connection():
    """Zero out o_proj and down_proj → output equals input."""
    block = TransformerBlock(make_cfg())
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)


def test_transformer_block_has_attn_and_ffn_attrs():
    block = TransformerBlock(make_cfg())
    assert hasattr(block, "attn")
    assert hasattr(block, "ffn")
    assert hasattr(block, "norm1")
    assert hasattr(block, "norm2")


def test_transformer_block_pfd_rational_gradient_flow():
    block = TransformerBlock(make_cfg(ffn_type="first_order_pfd_rational"))
    x = torch.randn(B, N, D)
    block(x).sum().backward()
    assert block.ffn.phi.grad is not None


def test_transformer_block_rational_residual():
    block = TransformerBlock(make_cfg(ffn_type="rational"))
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)
```

- [ ] **Step 2: Run to verify failures**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_transformer_block.py -v 2>&1 | tail -15
```

Expected: FAIL with `ImportError: cannot import name 'FFN_REGISTRY'` and `ImportError: cannot import name 'TransformerBlock'`

- [ ] **Step 3: Rewrite `transformer_block.py`**

Replace the full contents with:

```python
# rbf_ffn/models/transformer_block.py
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.attention import ATTN_REGISTRY
from rbf_ffn.models.llama_ffn import SwiGLUFFN
from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN, PFDRationalFFN, PFDRationalGatedFFN, FirstOrderPFDRationalFFN
from rbf_ffn.models.polar_ffn import AdaptivePolarMLP
from rbf_ffn.models.head_mixer import KromHCHeadMixer

FFN_REGISTRY: dict[str, type] = {
    "swiglu":                  SwiGLUFFN,
    "rational":                RationalFFN,
    "rationalglu":             RationalGatedFFN,
    "pfd_rational":            PFDRationalFFN,
    "pfd_rationalglu":         PFDRationalGatedFFN,
    "first_order_pfd_rational": FirstOrderPFDRationalFFN,
    "polar":                   AdaptivePolarMLP,
}


class TransformerBlock(nn.Module):
    """
    Composable causal transformer block.

    Builds attention and FFN from registries keyed by cfg.attn_type and
    cfg.ffn_type, so any attention variant can be paired with any FFN variant
    without additional block subclasses.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.attn_type not in ATTN_REGISTRY:
            raise ValueError(
                f"Unknown attn_type '{cfg.attn_type}'. Valid: {sorted(ATTN_REGISTRY)}"
            )
        if cfg.ffn_type not in FFN_REGISTRY:
            raise ValueError(
                f"Unknown ffn_type '{cfg.ffn_type}'. Valid: {sorted(FFN_REGISTRY)}"
            )
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = ATTN_REGISTRY[cfg.attn_type](cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = FFN_REGISTRY[cfg.ffn_type](cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class KromHCWrapper(nn.Module):
    """
    Wraps any transformer block with KromHC head mixing.

    Applies head mixing as an additive residual after the inner block:

        x_block = inner_block(x)
        heads   = x_block reshaped to (B*N, n_heads, head_dim)
        mixed   = KromHCHeadMixer(heads)
        out     = x_block + mixer_proj(mixed reshaped back)

    Returns (out, H) where H: (B, N, n_heads, n_heads).
    """

    def __init__(self, inner_block: nn.Module, cfg: ModelConfig):
        super().__init__()
        self.inner_block = inner_block
        self.n_heads   = cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, (
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        )
        self.head_dim  = cfg.d_model // cfg.n_heads
        self.head_mixer = KromHCHeadMixer(
            n_heads=cfg.n_heads,
            head_dim=self.head_dim,
            d_context=self.head_dim,
            mixer_hidden=cfg.kromhc_mixer_hidden,
        )
        self.mixer_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D)
        Returns: (out: (B, N, D), H: (B, N, n_heads, n_heads))
        """
        x_block = self.inner_block(x)                           # (B, N, D)
        B, N, D = x_block.shape
        heads = x_block.view(B * N, self.n_heads, self.head_dim)
        mixed, H = self.head_mixer(heads)                       # mixed: (B*N, n_heads, head_dim)
        correction = self.mixer_proj(mixed.view(B, N, D))
        H_4d = H.view(B, N, self.n_heads, self.n_heads)
        return x_block + correction, H_4d
```

- [ ] **Step 4: Run tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_transformer_block.py -v 2>&1 | tail -30
```

Expected: All PASS. (The parametrized shape test covers 3×7 = 21 combinations.)

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/models/transformer_block.py rbf_ffn/tests/test_transformer_block.py
git commit -m "feat(rbf_ffn): add FFN_REGISTRY + TransformerBlock, remove named block classes"
```

---

## Task 4: Update `model.py` — drop `BlockClass` dict, use `TransformerBlock`

**Files:**
- Modify: `rbf_ffn/models/model.py`

- [ ] **Step 1: Verify existing model tests still pass before touching model.py**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_model.py -v 2>&1 | tail -20
```

Some tests will already be broken due to the block class removal in Task 3. Note which ones fail.

- [ ] **Step 2: Update `model.py`**

Replace the import block and `CausalLM.__init__` block selection:

```python
# rbf_ffn/models/model.py
from __future__ import annotations
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import TransformerBlock, KromHCWrapper
from rbf_ffn.models.kronecker_linear import KroneckerLMHead


def build_optimizer_groups(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split model parameters into Muon and AdamW groups.

    Rules (first match wins, applied after deduplication by tensor id):
      1. "sigma_raw" in name           → AdamW
      2. param is token embedding weight → AdamW
      3. "delta_" in name              → AdamW  (KroneckerDeltaLinear delta_C/delta_D)
      4. "weight_gens" in name         → AdamW  (KromHC gating MLPs — tiny, not suited for Muon)
      5. "mixer_proj" in name          → AdamW  (KromHC output projection)
      6. param.ndim == 2               → Muon
      7. else                          → AdamW

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
        elif "delta_" in name:
            adamw.append(param)
        elif "weight_gens" in name:
            adamw.append(param)
        elif "mixer_proj" in name:
            adamw.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)

    return muon, adamw


class CausalLM(nn.Module):
    """
    Causal language model.

        token_embedding → N × TransformerBlock → RMSNorm → lm_head

    Block type is controlled by cfg.attn_type and cfg.ffn_type (see ATTN_REGISTRY
    and FFN_REGISTRY). If cfg.use_kromhc=True, each block is wrapped in KromHCWrapper.

    lm_head variants (selected by cfg):
        default (tie_embeddings=True)  → nn.Linear, weight tied to token_embedding
        tie_embeddings=False           → nn.Linear, independent weight (Muon-trained)
        lm_head_kronecker=True         → KroneckerLMHead; tie_embeddings is ignored

    forward() always returns (logits, hs):
        logits: (B, N, vocab_size)
        hs:     list of H tensors (B, N, n_heads, n_heads) per layer, or [] if not using KromHC
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        def make_block():
            block = TransformerBlock(cfg)
            if cfg.use_kromhc:
                return KromHCWrapper(block, cfg)
            return block

        self.use_kromhc = cfg.use_kromhc
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([make_block() for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.pre_lm_head_silu = cfg.pre_lm_head_silu
        if cfg.lm_head_kronecker:
            self.lm_head = KroneckerLMHead(cfg.d_model, cfg.vocab_size)
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        tokens: (B, N) integer token ids
        returns: (logits: (B, N, vocab_size), hs: list of H per layer or [])
        """
        x = self.token_embedding(tokens)
        hs: list[torch.Tensor] = []
        for block in self.blocks:
            result = block(x)
            if self.use_kromhc:
                x, H = result
                hs.append(H.detach())
            else:
                x = result
        x = self.norm(x)
        if self.pre_lm_head_silu:
            x = torch.nn.functional.silu(x)
        return self.lm_head(x), hs
```

- [ ] **Step 3: Run model tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_model.py -v 2>&1 | tail -30
```

Expected: All PASS. The `model_type` compat layer in `__post_init__` means `make_model("rationalglu")` etc. still work — `model_type` is translated to `attn_type`/`ffn_type` before `CausalLM` ever sees it.

- [ ] **Step 4: Commit**

```bash
git add rbf_ffn/models/model.py
git commit -m "refactor(rbf_ffn): model.py uses TransformerBlock, drops BlockClass dispatch dict"
```

---

## Task 5: Update `train.py` experiment naming

**Files:**
- Modify: `rbf_ffn/train.py`

- [ ] **Step 1: Update `get_experiment_dir`**

The current code does `f"{cfg.model_type}{norm_tags}_d{cfg.d_model}"`. Replace with `attn_type`/`ffn_type`:

Find this block in `train.py`:

```python
    name = f"{stamp}_{cfg.model_type}{norm_tags}_d{cfg.d_model}"
```

Replace with:

```python
    name = f"{stamp}_{cfg.attn_type}_{cfg.ffn_type}{norm_tags}_d{cfg.d_model}"
```

- [ ] **Step 2: Run train tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_train.py -v 2>&1 | tail -20
```

Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add rbf_ffn/train.py
git commit -m "fix(rbf_ffn): update experiment dir naming to use attn_type + ffn_type"
```

---

## Task 6: Run full test suite

- [ ] **Step 1: Run everything**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/ -v 2>&1 | tail -40
```

Expected: All tests PASS. If any fail, read the error message carefully:
- `AttributeError: 'TransformerBlock' has no attribute 'X'` → a test still references a removed named block class attribute. Update that test fixture to use `TransformerBlock(make_cfg(...))`.
- `KeyError` in registry → a test is passing a `model_type` not in `_MODEL_TYPE_MAP`. Check the value.

- [ ] **Step 2: Commit if clean, else fix and commit**

```bash
git add -p   # review any changes made during debugging
git commit -m "test(rbf_ffn): fix any residual test failures after composable block refactor"
```

---

## Task 7: Migrate all YAML configs

**Files:**
- Modify: all 27 files in `rbf_ffn/configs/`

The mapping is:

| `model_type` value | `attn_type` | `ffn_type` |
|---|---|---|
| `baseline` | `standard` | `swiglu` |
| `rational` | `standard` | `rational` |
| `rationalglu` | `standard` | `rationalglu` |
| `pfd_rational` | `standard` | `pfd_rational` |
| `pfd_rationalglu` | `standard` | `pfd_rationalglu` |
| `first_order_pfd_rational` | `standard` | `first_order_pfd_rational` |
| `polar_mlp` | `standard` | `polar` |
| `polar_attn` | `polar` | `swiglu` |
| `polar_full` | `polar` | `polar` |
| `xsa` | `xsa` | `swiglu` |

- [ ] **Step 1: Apply migration to every config file**

For each YAML in `rbf_ffn/configs/`, remove the `model_type: X` line and add two lines in its place:

```yaml
attn_type: <value from table above>
ffn_type: <value from table above>
```

Apply to each file:

**`baseline.yaml`**: `model_type: baseline` → `attn_type: standard` + `ffn_type: swiglu`
**`baseline_adaptive_weight_norm.yaml`**: same
**`baseline_kromhc.yaml`**: same
**`baseline_kronecker_delta.yaml`**: same
**`baseline_kronecker_lm_head.yaml`**: same
**`baseline_qk_norm_kromhc.yaml`**: same
**`baseline_qk_norm_weight_norm_kronecker.yaml`**: same
**`baseline_qk_norm_weight_norm_pre_silu.yaml`**: same
**`baseline_qk_norm.yaml`**: same
**`baseline_untied_embeddings.yaml`**: same
**`baseline_weight_norm.yaml`**: same
**`baseline_xsa.yaml`**: `model_type: xsa` → `attn_type: xsa` + `ffn_type: swiglu`
**`rational_ffn.yaml`**: `model_type: rational` → `attn_type: standard` + `ffn_type: rational`
**`rationalglu_ffn.yaml`**: `model_type: rationalglu` → `attn_type: standard` + `ffn_type: rationalglu`
**`rationalglu_qk_norm.yaml`**: same
**`pfd_rational_ffn.yaml`**: `model_type: pfd_rational` → `attn_type: standard` + `ffn_type: pfd_rational`
**`pfd_rationalglu_ffn.yaml`**: `model_type: pfd_rationalglu` → `attn_type: standard` + `ffn_type: pfd_rationalglu`
**`pfd_rationalglu_ffn_small.yaml`**: same
**`pfd_rationalglu_qk_norm.yaml`**: same
**`pfd_rationalglu_qk_norm_weight_norm.yaml`**: same
**`pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml`**: same
**`first_order_pfd_rational_ffn.yaml`**: `model_type: first_order_pfd_rational` → `attn_type: standard` + `ffn_type: first_order_pfd_rational`
**`first_order_pfd_rational_qk_norm_weight_norm.yaml`**: same
**`polar_mlp.yaml`**: `model_type: polar_mlp` → `attn_type: standard` + `ffn_type: polar`
**`polar_attn.yaml`**: `model_type: polar_attn` → `attn_type: polar` + `ffn_type: swiglu`
**`polar_full.yaml`**: `model_type: polar_full` → `attn_type: polar` + `ffn_type: polar`

- [ ] **Step 2: Verify configs load without errors**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -c "
from pathlib import Path
from rbf_ffn.config import load_config
configs = sorted(Path('rbf_ffn/configs').glob('*.yaml'))
for p in configs:
    cfg = load_config(p)
    print(f'OK  {p.name:55s} attn={cfg.attn_type} ffn={cfg.ffn_type}')
"
```

Expected: All 27 configs print `OK` with correct `attn_type` and `ffn_type` values.

- [ ] **Step 3: Run full test suite one final time**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/ -v 2>&1 | tail -20
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add rbf_ffn/configs/
git commit -m "chore(rbf_ffn): migrate all configs from model_type to attn_type + ffn_type"
```

---

## Self-Review

**Spec coverage:**
- ✅ `attn_type` + `ffn_type` replace `model_type` in config
- ✅ `ATTN_REGISTRY` covers all 3 attention variants
- ✅ `FFN_REGISTRY` covers all 7 FFN variants
- ✅ Single `TransformerBlock` replaces all 10 named block classes
- ✅ `model.py` drops `BlockClass` dict
- ✅ Deprecated `model_type` still works via `__post_init__` translation
- ✅ All 27 YAML configs migrated
- ✅ `train.py` experiment naming updated
- ✅ All existing tests updated or verified still passing

**Placeholder scan:** No TBDs or "fill in later" patterns found.

**Type consistency:** `ATTN_REGISTRY` and `FFN_REGISTRY` are `dict[str, type]` in both definition and tests. `ModelConfig.attn_type`/`ffn_type` are `str` throughout. `TransformerBlock` has `.attn` and `.ffn` attrs matching what old tests accessed on named block classes.
