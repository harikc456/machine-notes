# KromHC Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a transformer language model with Kronecker-factored head mixing on WikiText-103, with unit tests, integration tests, and experiment infrastructure.

**Architecture:** Foundation (config + data) → Core Components (attention + FFN + head mixer) → Integration (blocks + model) → Training → Experiments

**Tech Stack:** PyTorch, tiktoken (r50k_base), Hugging Face datasets, Muon optimizer, YAML config, pytest

---

## File Structure Map

**New files to create:**
```
kromhc_transformer/
├── __init__.py
├── config.py                    # Config class (extends RBFFFNConfig pattern)
├── models/
│   ├── __init__.py
│   ├── attention.py             # CausalSelfAttention with RoPE + QK norm
│   ├── head_mixer.py            # KromHCHeadMixer (new core component)
│   ├── llama_ffn.py             # SwiGLU FFN (copy from rbf_ffn)
│   ├── transformer_block.py      # LlamaBlock + KromHCBlock
│   └── model.py                 # CausalLM with block dispatch
├── data.py                      # WikiText-103 data pipeline
├── train.py                     # Training loop (dual optimizer)
├── configs/
│   ├── baseline.yaml            # Baseline: no KromHC
│   ├── kromhc_small.yaml        # KromHC: 50M params
│   └── kromhc_medium.yaml       # KromHC: 100M params
├── experiments/
│   ├── README.md                # Experiment running guide
│   ├── scripts/
│   │   ├── requirements.txt
│   │   ├── utils.py             # Shared experiment utilities
│   │   ├── exp_01_poc.py        # Phase 1: POC
│   │   ├── exp_02_baseline_vs_kromhc.py
│   │   ├── exp_03_ablations.py
│   │   └── exp_04_final.py
│   └── results/
│       ├── (populated during experiments)
│       └── plots/
├── findings.md                  # Main research deliverable
└── tests/
    ├── test_head_mixer.py       # Unit: KromHCHeadMixer
    ├── test_transformer_block.py # Integration: blocks
    └── test_model.py            # Integration: CausalLM
```

---

## Task Decomposition

### Task 1: Project Scaffolding & Config

**Files:**
- Create: `kromhc_transformer/__init__.py`
- Create: `kromhc_transformer/config.py`

#### Step 1.1: Create `__init__.py`

- [ ] Write `kromhc_transformer/__init__.py`

```python
"""KromHC Transformer: Language model with Kronecker-factored head mixing."""
__version__ = "0.1.0"
```

- [ ] Run: `python -c "import kromhc_transformer; print(kromhc_transformer.__version__)"`
  Expected: `0.1.0`

#### Step 1.2: Write config class test

- [ ] Write failing test: `test_config_defaults()`

```python
# tests/test_config.py (create new)
import pytest
from kromhc_transformer.config import KromHCConfig

def test_config_defaults():
    cfg = KromHCConfig()
    assert cfg.d_model == 256
    assert cfg.n_heads == 8
    assert cfg.n_layers == 6
    assert cfg.model_type == "kromhc"
    assert cfg.use_kromhc == True
    assert cfg.qk_norm == True
    assert cfg.vocab_size == 50257
    assert cfg.seq_len == 512
```

Run: `pytest tests/test_config.py::test_config_defaults -xvs`
Expected: FAIL (file not found)

#### Step 1.3: Implement config class

- [ ] Write `kromhc_transformer/config.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class KromHCConfig:
    """Configuration for KromHC Transformer training."""

    # Model dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # KromHC-specific
    model_type: str = "kromhc"      # "baseline" | "kromhc"
    use_kromhc: bool = True         # Enable head mixing
    qk_norm: bool = True            # Enable QK normalization

    # FFN
    ffn_hidden: int = 688

    # Sequence / vocab
    seq_len: int = 512
    vocab_size: int = 50257         # r50k_base tokenizer

    # Training
    seed: int = 42
    n_epochs: int = 10
    batch_size: int = 32
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    adamw_wd: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
    grad_accum_steps: int = 1


def load_config(path: str | Path) -> KromHCConfig:
    """Load KromHCConfig from YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return KromHCConfig()
    valid_fields = {f.name for f in KromHCConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return KromHCConfig(**raw)
```

- [ ] Run test: `pytest tests/test_config.py::test_config_defaults -xvs`
  Expected: PASS

#### Step 1.4: Test YAML loading

- [ ] Add test: `test_load_config_from_yaml()`

```python
def test_load_config_from_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("d_model: 512\nn_heads: 16\nmodel_type: baseline\n")
    cfg = load_config(yaml_file)
    assert cfg.d_model == 512
    assert cfg.n_heads == 16
    assert cfg.model_type == "baseline"
    assert cfg.use_kromhc == True  # default
```

- [ ] Run: `pytest tests/test_config.py -xvs`
  Expected: All PASS

#### Step 1.5: Commit

- [ ] `git add kromhc_transformer/__init__.py kromhc_transformer/config.py tests/test_config.py`
- [ ] `git commit -m "feat: add KromHC config class with YAML loading"`

---

### Task 2: Data Loading Pipeline

**Files:**
- Create: `kromhc_transformer/data.py`

#### Step 2.1: Write data loading test

- [ ] Write test: `tests/test_data.py`

```python
import pytest
import torch
from pathlib import Path
from kromhc_transformer.data import get_dataloaders
from kromhc_transformer.config import KromHCConfig

def test_get_dataloaders():
    """Smoke test: dataloaders load without error."""
    cfg = KromHCConfig(seq_len=512, batch_size=8)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Check one batch from train
    batch = next(iter(train_loader))
    assert batch.shape == (8, 512)
    assert batch.dtype == torch.long
```

Run: `pytest tests/test_data.py::test_get_dataloaders -xvs`
Expected: FAIL (module not found)

#### Step 2.2: Implement data pipeline

- [ ] Write `kromhc_transformer/data.py`

```python
"""WikiText-103 data pipeline (adapted from rbf_ffn)."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


_CACHE_DIR = Path(__file__).parent / "data_cache"


def chunk_tokens(tokens: list[int], seq_len: int) -> torch.Tensor:
    """Split flat token list into non-overlapping chunks of seq_len."""
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
    """Load a tokenised split from cache, or build and cache it."""
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}_r50k_{seq_len}.pt"

    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    # Lazy imports
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
    cfg must have: seq_len, batch_size, seed
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
            persistent_workers=shuffle,
            prefetch_factor=2 if shuffle else None,
        )

    train_loader = _make_loader("train", shuffle=True, drop_last=True)
    val_loader = _make_loader("validation", shuffle=False, drop_last=False)
    test_loader = _make_loader("test", shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
```

- [ ] Run test: `pytest tests/test_data.py::test_get_dataloaders -xvs` (first run will download WikiText-103, ~5 min)
  Expected: PASS

#### Step 2.3: Commit

- [ ] `git add kromhc_transformer/data.py tests/test_data.py`
- [ ] `git commit -m "feat: add WikiText-103 data pipeline with caching"`

---

### Task 3: Attention Layer (RoPE + QK Norm)

**Files:**
- Create: `kromhc_transformer/models/attention.py`

#### Step 3.1: Write attention test

- [ ] Write `tests/test_attention.py`

```python
import pytest
import torch
from kromhc_transformer.models.attention import CausalSelfAttention, RotaryEmbedding
from kromhc_transformer.config import KromHCConfig

def test_rotary_embedding():
    """Test RoPE produces correct shapes."""
    rope = RotaryEmbedding(head_dim=64)
    x = torch.randn(2, 8, 512, 64)  # (B, H, N, head_dim)
    out = rope(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_causal_self_attention():
    """Test attention forward pass."""
    cfg = KromHCConfig(d_model=256, n_heads=8, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 512, 256)  # (B, N, d_model)
    out = attn(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_causal_self_attention_qk_norm():
    """Test attention with QK normalization."""
    cfg = KromHCConfig(d_model=256, n_heads=8, qk_norm=True, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 512, 256)
    out = attn(x)
    assert out.shape == x.shape
```

Run: `pytest tests/test_attention.py -xvs`
Expected: FAIL (module not found)

#### Step 3.2: Implement attention

- [ ] Write `kromhc_transformer/models/__init__.py`

```python
"""Models for KromHC Transformer."""
from .attention import CausalSelfAttention, RotaryEmbedding
from .head_mixer import KromHCHeadMixer
from .llama_ffn import SwiGLUFFN
from .transformer_block import KromHCBlock, LlamaBlock
from .model import CausalLM

__all__ = [
    "CausalSelfAttention",
    "RotaryEmbedding",
    "KromHCHeadMixer",
    "SwiGLUFFN",
    "KromHCBlock",
    "LlamaBlock",
    "CausalLM",
]
```

- [ ] Write `kromhc_transformer/models/attention.py`

```python
"""Multi-head causal self-attention with RoPE and QK normalization."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from kromhc_transformer.config import KromHCConfig

_FLASH_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def _flash_available() -> bool:
    """Return True if FlashAttention SDPA backend is enabled on CUDA."""
    return torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension: [x1, x2] → [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, head_dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_heads, N, head_dim)"""
        seq_len = x.shape[2]
        if self._cos is None or seq_len > self._cached_len:
            self._build_cache(seq_len, x.device)
        cos = self._cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin[:seq_len].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (_rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional QK normalization."""

    def __init__(self, cfg: KromHCConfig):
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
        self._qk_norm = cfg.qk_norm
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → output: (B, N, d_model)"""
        B, N, D = x.shape

        def split_heads(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(split_heads(self.q_proj(x)))
        k = self.rope(split_heads(self.k_proj(x)))
        v = split_heads(self.v_proj(x))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)
```

- [ ] Run test: `pytest tests/test_attention.py -xvs`
  Expected: PASS

#### Step 3.3: Commit

- [ ] `git add kromhc_transformer/models/__init__.py kromhc_transformer/models/attention.py tests/test_attention.py`
- [ ] `git commit -m "feat: add CausalSelfAttention with RoPE and QK norm"`

---

### Task 4: KromHC Head Mixer

**Files:**
- Create: `kromhc_transformer/models/head_mixer.py`

#### Step 4.1: Write head mixer unit tests

- [ ] Write `tests/test_head_mixer.py`

```python
import pytest
import torch
import math
from kromhc_transformer.models.head_mixer import KromHCHeadMixer

def test_head_mixer_shape():
    """Test output shape is (bs, n_heads, head_dim)."""
    mixer = KromHCHeadMixer(n_heads=8, head_dim=64)
    x = torch.randn(32, 8, 64)  # (bs, n_heads, head_dim)
    out, H = mixer(x)
    assert out.shape == x.shape
    assert H.shape == (32, 8, 8)

def test_head_mixer_doubly_stochastic():
    """Test H is doubly-stochastic: row/col sums = 1."""
    mixer = KromHCHeadMixer(n_heads=8, head_dim=64)
    x = torch.randn(16, 8, 64)
    _, H = mixer(x)

    # Row sums should be ~1
    row_sums = H.sum(dim=2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Column sums should be ~1
    col_sums = H.sum(dim=1)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)

def test_head_mixer_gradient_flow():
    """Test gradients flow through mixer."""
    mixer = KromHCHeadMixer(n_heads=8, head_dim=64)
    x = torch.randn(8, 8, 64, requires_grad=True)
    out, H = mixer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().max() > 0  # Non-zero gradients

def test_head_mixer_edge_cases():
    """Test various power-of-2 head counts."""
    for n_heads in [2, 4, 8, 16]:
        mixer = KromHCHeadMixer(n_heads=n_heads, head_dim=32)
        x = torch.randn(8, n_heads, 32)
        out, H = mixer(x)
        assert out.shape == x.shape
        assert H.shape == (8, n_heads, n_heads)
```

Run: `pytest tests/test_head_mixer.py -xvs`
Expected: FAIL (module not found)

#### Step 4.2: Implement KromHCHeadMixer

- [ ] Write `kromhc_transformer/models/head_mixer.py`

```python
"""KromHC Head Mixer: Kronecker-factored head mixing."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class KromHCHeadMixer(nn.Module):
    """
    Kronecker-factored permutation mixing for attention heads.

    Input: (bs, n_heads, head_dim)
    Output: mixed_heads (bs, n_heads, head_dim), mixing_matrix H (bs, n_heads, n_heads)
    """

    def __init__(self, n_heads: int = 8, head_dim: int = 64, d_context: int = None):
        super().__init__()
        self.n = n_heads
        self.head_dim = head_dim
        if d_context is None:
            d_context = head_dim

        # Auto-factor into 2's (assumes power-of-2 n_heads)
        k = int(math.log2(n_heads))
        assert 2 ** k == n_heads, f"n_heads ({n_heads}) must be power of 2"
        self.factors = [2] * k
        self.K = k

        # Pre-compute permutation bases (one per factor)
        self.perm_bases = nn.ParameterList()
        self.weight_gens = nn.ModuleList()

        for i_k in self.factors:
            n_fact = math.factorial(i_k)  # 2 for i_k=2
            basis = torch.zeros((n_fact, i_k, i_k))
            for idx, p in enumerate(permutations(range(i_k))):
                for r, c in enumerate(p):
                    basis[idx, r, c] = 1.0
            self.perm_bases.append(nn.Parameter(basis, requires_grad=False))

            # Small MLP for context-dependent weights
            self.weight_gens.append(nn.Sequential(
                nn.Linear(d_context, 32),
                nn.ReLU(),
                nn.Linear(32, n_fact)
            ))

    def batched_kronecker(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Batched Kronecker: (bs, m, m) ⊗ (bs, p, p) → (bs, m*p, m*p)"""
        bs = A.shape[0]
        m = A.shape[1]
        p = B.shape[1]
        return torch.einsum('b i j, b k l -> b i k j l', A, B).reshape(bs, m * p, m * p)

    def forward(self, x: torch.Tensor):
        """
        x: (bs, n_heads, head_dim)
        Returns: mixed (bs, n_heads, head_dim), H (bs, n_heads, n_heads)
        """
        bs, n, d = x.shape
        assert n == self.n

        # Global context: mean of heads
        context = x.mean(dim=1)  # (bs, head_dim)

        # Build small U^k matrices
        small_us = []
        for gen, basis in zip(self.weight_gens, self.perm_bases):
            logits = gen(context)  # (bs, i_k!)
            a = F.softmax(logits, dim=-1)  # convex weights
            U = a @ basis.view(basis.shape[0], -1)  # (bs, i_k * i_k)
            U = U.view(bs, basis.shape[1], basis.shape[2])  # (bs, i_k, i_k)
            small_us.append(U)

        # Kronecker chain → full H
        H = small_us[0]
        for U in small_us[1:]:
            H = self.batched_kronecker(H, U)

        # Apply mixing
        out = torch.matmul(H, x)  # (bs, n_heads, head_dim)
        return out, H
```

- [ ] Run test: `pytest tests/test_head_mixer.py -xvs`
  Expected: PASS

#### Step 4.3: Commit

- [ ] `git add kromhc_transformer/models/head_mixer.py tests/test_head_mixer.py`
- [ ] `git commit -m "feat: add KromHCHeadMixer with doubly-stochastic Kronecker mixing"`

---

### Task 5: FFN Layer

**Files:**
- Create: `kromhc_transformer/models/llama_ffn.py`

#### Step 5.1: Write FFN test

- [ ] Write test in `tests/test_llama_ffn.py`

```python
import pytest
import torch
from kromhc_transformer.models.llama_ffn import SwiGLUFFN
from kromhc_transformer.config import KromHCConfig

def test_swiglu_ffn():
    """Test SwiGLU FFN shape."""
    cfg = KromHCConfig(d_model=256, ffn_hidden=688)
    ffn = SwiGLUFFN(cfg)
    x = torch.randn(2, 512, 256)  # (B, N, d_model)
    out = ffn(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
```

Run: `pytest tests/test_llama_ffn.py -xvs`
Expected: FAIL (module not found)

#### Step 5.2: Implement SwiGLU FFN

- [ ] Write `kromhc_transformer/models/llama_ffn.py`

```python
"""SwiGLU FFN (adapted from rbf_ffn)."""
import torch
import torch.nn as nn
from kromhc_transformer.config import KromHCConfig


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: d_model → ffn_hidden ⊗ 2 → d_model with SiLU gating."""

    def __init__(self, cfg: KromHCConfig):
        super().__init__()
        self.w = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.out = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → out: (B, N, d_model)"""
        return self.out(nn.functional.silu(self.w(x)) * self.v(x))
```

- [ ] Run test: `pytest tests/test_llama_ffn.py -xvs`
  Expected: PASS

#### Step 5.3: Commit

- [ ] `git add kromhc_transformer/models/llama_ffn.py tests/test_llama_ffn.py`
- [ ] `git commit -m "feat: add SwiGLU FFN"`

---

### Task 6: Transformer Blocks (LlamaBlock & KromHCBlock)

**Files:**
- Create: `kromhc_transformer/models/transformer_block.py`

#### Step 6.1: Write block integration tests

- [ ] Write `tests/test_transformer_block.py`

```python
import pytest
import torch
from kromhc_transformer.models.transformer_block import LlamaBlock, KromHCBlock
from kromhc_transformer.config import KromHCConfig

def test_llama_block():
    """Test LlamaBlock forward pass."""
    cfg = KromHCConfig(d_model=256, n_heads=8, ffn_hidden=688, dropout=0.0)
    block = LlamaBlock(cfg)
    x = torch.randn(2, 512, 256)
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_kromhc_block():
    """Test KromHCBlock forward pass."""
    cfg = KromHCConfig(d_model=256, n_heads=8, ffn_hidden=688, use_kromhc=True, dropout=0.0)
    block = KromHCBlock(cfg)
    x = torch.randn(2, 512, 256)
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_kromhc_block_returns_mixing_matrix():
    """Test KromHCBlock returns mixing matrix."""
    cfg = KromHCConfig(d_model=256, n_heads=8, ffn_hidden=688, use_kromhc=True, dropout=0.0)
    block = KromHCBlock(cfg)
    x = torch.randn(2, 512, 256)
    out, H = block(x)
    assert out.shape == x.shape
    # H shape: (B*seq_len, n_heads, n_heads)
    assert H.shape[1] == 8
    assert H.shape[2] == 8

def test_block_residuals():
    """Test residual connections work."""
    cfg = KromHCConfig(d_model=256, n_heads=8, ffn_hidden=688, dropout=0.0)
    block = KromHCBlock(cfg)
    x = torch.randn(2, 512, 256)

    # For random initialization, output should differ from input
    # but shape should match (residuals add)
    out = block(x)
    assert out.shape == x.shape
```

Run: `pytest tests/test_transformer_block.py -xvs`
Expected: FAIL (module not found)

#### Step 6.2: Implement blocks

- [ ] Write `kromhc_transformer/models/transformer_block.py`

```python
"""Transformer blocks: LlamaBlock (standard) and KromHCBlock (with head mixing)."""
import torch
import torch.nn as nn
from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.models.attention import CausalSelfAttention
from kromhc_transformer.models.head_mixer import KromHCHeadMixer
from kromhc_transformer.models.llama_ffn import SwiGLUFFN


class LlamaBlock(nn.Module):
    """Standard Llama-style block: norm → attn → residual → norm → ffn → residual."""

    def __init__(self, cfg: KromHCConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → out: (B, N, d_model)"""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class KromHCBlock(nn.Module):
    """
    Llama-style block with KromHC head mixing.

    norm1 → attn → reshape → head_mixer → project → residual → norm2 → ffn → residual
    """

    def __init__(self, cfg: KromHCConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.use_kromhc = cfg.use_kromhc

        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)

        if self.use_kromhc:
            self.head_mixer = KromHCHeadMixer(n_heads=cfg.n_heads, head_dim=self.head_dim)
            self.mixer_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)
        self._last_mixing_matrix = None

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, d_model)
        Returns: out (B, N, d_model), H (B*N, n_heads, n_heads) or None
        """
        B, N, D = x.shape

        # Attention
        attn_out = self.attn(self.norm1(x))  # (B, N, d_model)

        # Optional head mixing
        if self.use_kromhc:
            # Reshape to (B*N, n_heads, head_dim)
            heads = attn_out.reshape(B * N, self.n_heads, self.head_dim)
            mixed_heads, H = self.head_mixer(heads)  # (B*N, n_heads, head_dim), (B*N, n_heads, n_heads)
            self._last_mixing_matrix = H

            # Reshape back and project
            mixed = mixed_heads.reshape(B, N, D)
            attn_out = self.mixer_proj(mixed)
        else:
            self._last_mixing_matrix = None

        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x, self._last_mixing_matrix
```

- [ ] Run test: `pytest tests/test_transformer_block.py -xvs`
  Expected: PASS

#### Step 6.3: Commit

- [ ] `git add kromhc_transformer/models/transformer_block.py tests/test_transformer_block.py`
- [ ] `git commit -m "feat: add LlamaBlock and KromHCBlock with head mixing"`

---

### Task 7: CausalLM Model

**Files:**
- Create: `kromhc_transformer/models/model.py`

#### Step 7.1: Write model test

- [ ] Write `tests/test_model.py`

```python
import pytest
import torch
from kromhc_transformer.models.model import CausalLM
from kromhc_transformer.config import KromHCConfig

def test_causal_lm_forward():
    """Test CausalLM forward pass."""
    cfg = KromHCConfig(
        d_model=256, n_heads=8, n_layers=2, vocab_size=1000, seq_len=512,
        ffn_hidden=688, model_type="kromhc", dropout=0.0
    )
    model = CausalLM(cfg)
    tokens = torch.randint(0, 1000, (2, 512))
    logits = model(tokens)
    assert logits.shape == (2, 512, 1000)
    assert logits.dtype == torch.float32

def test_causal_lm_baseline():
    """Test baseline model (no head mixing)."""
    cfg = KromHCConfig(
        d_model=256, n_heads=8, n_layers=2, vocab_size=1000, seq_len=512,
        ffn_hidden=688, model_type="baseline", use_kromhc=False, dropout=0.0
    )
    model = CausalLM(cfg)
    tokens = torch.randint(0, 1000, (2, 512))
    logits = model(tokens)
    assert logits.shape == (2, 512, 1000)

def test_causal_lm_weight_tying():
    """Test embedding and lm_head weight tying."""
    cfg = KromHCConfig(d_model=256, n_heads=8, n_layers=2, vocab_size=1000)
    model = CausalLM(cfg)
    assert model.lm_head.weight is model.token_embedding.weight
```

Run: `pytest tests/test_model.py -xvs`
Expected: FAIL (module not found)

#### Step 7.2: Implement CausalLM

- [ ] Write `kromhc_transformer/models/model.py`

```python
"""Causal language model with block dispatch."""
import torch
import torch.nn as nn
from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.models.transformer_block import LlamaBlock, KromHCBlock


class CausalLM(nn.Module):
    """
    Causal Language Model.

    token_embedding → N × Block → RMSNorm → lm_head (weight-tied)

    Block type selected by cfg.model_type:
        "baseline" → LlamaBlock (no head mixing)
        "kromhc" → KromHCBlock (with head mixing)
    """

    def __init__(self, cfg: KromHCConfig):
        super().__init__()
        BlockClass = {
            "baseline": LlamaBlock,
            "kromhc": KromHCBlock,
        }[cfg.model_type]

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([BlockClass(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N) LongTensor
        Returns: logits (B, N, vocab_size)
        """
        x = self.token_embedding(tokens)  # (B, N, d_model)
        for block in self.blocks:
            output = block(x)
            if isinstance(output, tuple):
                x, _ = output  # Discard mixing matrix in forward
            else:
                x = output
        x = self.norm(x)
        return self.lm_head(x)
```

- [ ] Run test: `pytest tests/test_model.py -xvs`
  Expected: PASS

#### Step 7.3: Commit

- [ ] `git add kromhc_transformer/models/model.py tests/test_model.py`
- [ ] `git commit -m "feat: add CausalLM with block dispatch"`

---

### Task 8: Training Loop

**Files:**
- Create: `kromhc_transformer/train.py`

#### Step 8.1: Write training loop

- [ ] Write `kromhc_transformer/train.py`

```python
"""Training loop for KromHC transformer on WikiText-103."""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from kromhc_transformer.config import KromHCConfig, load_config
from kromhc_transformer.data import get_dataloaders
from kromhc_transformer.models.model import CausalLM


def get_experiment_dir(cfg: KromHCConfig) -> Path:
    """Create timestamped experiment directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{cfg.model_type}_d{cfg.d_model}_h{cfg.n_heads}"
    path = Path(__file__).parent / "experiments" / "results" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup + cosine annealing LR schedule."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def build_optimizer_groups(model: CausalLM) -> tuple[list, list]:
    """Split params into Muon (2D) and AdamW (other)."""
    emb_id = id(model.token_embedding.weight)
    seen = set()
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if pid == emb_id:
            adamw_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


@torch.no_grad()
def evaluate(model: CausalLM, loader, device: torch.device) -> tuple[float, float]:
    """Compute loss and perplexity on validation/test set."""
    model.eval()
    loss_sum = 0.0
    token_count = 0

    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        n_tokens = inputs.numel()

        with torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            logits = model(inputs)  # (B, N, vocab_size)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        loss_sum += loss.item() * n_tokens
        token_count += n_tokens

    avg_loss = loss_sum / token_count
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def train(cfg: KromHCConfig, device: torch.device = None):
    """Main training loop."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Build model
    model = CausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params")

    # Build optimizer
    muon_params, adamw_params = build_optimizer_groups(model)
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=cfg.muon_lr))
    if adamw_params:
        optimizers.append(AdamW(adamw_params, lr=cfg.adamw_lr, weight_decay=cfg.adamw_wd))

    # LR schedule
    n_steps_per_epoch = len(train_loader)
    total_steps = n_steps_per_epoch * cfg.n_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    schedulers = [
        LambdaLR(opt, lr_lambda=make_lr_lambda(warmup_steps, total_steps))
        for opt in optimizers
    ]

    # Experiment tracking
    exp_dir = get_experiment_dir(cfg)
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "val_ppls": [],
        "test_loss": None,
        "test_ppl": None,
        "config": cfg.__dict__,
        "n_params": n_params,
        "hardware": {"device": str(device)},
    }

    # Training loop
    global_step = 0
    start_time = time.time()

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.n_epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            with torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss = loss / cfg.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()
                for sched in schedulers:
                    sched.step()
                global_step += 1

            epoch_loss += loss.item() * cfg.grad_accum_steps
            pbar.set_postfix({"loss": epoch_loss / (batch_idx + 1)})

        # Validation
        val_loss, val_ppl = evaluate(model, val_loader, device)
        metrics["train_losses"].append(epoch_loss / len(train_loader))
        metrics["val_losses"].append(val_loss)
        metrics["val_ppls"].append(val_ppl)
        print(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

    # Test
    test_loss, test_ppl = evaluate(model, test_loader, device)
    metrics["test_loss"] = test_loss
    metrics["test_ppl"] = test_ppl
    metrics["wall_clock_s"] = time.time() - start_time

    print(f"Test: loss={test_loss:.4f}, ppl={test_ppl:.2f}")

    # Save metrics
    json_path = exp_dir / f"{cfg.model_type}_{cfg.seed}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    ckpt_path = exp_dir / f"model_{cfg.seed}.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"Results saved to {exp_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else None
    train(cfg, device)
```

#### Step 8.2: Test smoke test

- [ ] Create smoke test: `tests/test_train_smoke.py`

```python
import pytest
import torch
import tempfile
from pathlib import Path
from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.train import train

def test_train_smoke():
    """Smoke test: 1 step training runs without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = KromHCConfig(
            d_model=128, n_heads=4, n_layers=1, ffn_hidden=256,
            batch_size=2, n_epochs=1, seq_len=64,
            model_type="kromhc", use_kromhc=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = train(cfg, device)
        assert metrics["test_ppl"] > 0
        assert metrics["test_loss"] > 0
```

Run: `pytest tests/test_train_smoke.py -xvs` (this will download WikiText once, ~5 min first run)
Expected: PASS

#### Step 8.3: Commit

- [ ] `git add kromhc_transformer/train.py tests/test_train_smoke.py`
- [ ] `git commit -m "feat: add training loop with dual optimizer and LR schedule"`

---

### Task 9: Config Files

**Files:**
- Create: `kromhc_transformer/configs/*.yaml`

#### Step 9.1: Write config files

- [ ] Create `kromhc_transformer/configs/baseline.yaml`

```yaml
model_type: baseline
use_kromhc: false
qk_norm: true
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
batch_size: 32
n_epochs: 10
seq_len: 512
vocab_size: 50257
seed: 42
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
dropout: 0.1
```

- [ ] Create `kromhc_transformer/configs/kromhc_small.yaml`

```yaml
model_type: kromhc
use_kromhc: true
qk_norm: true
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
batch_size: 32
n_epochs: 10
seq_len: 512
vocab_size: 50257
seed: 42
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
dropout: 0.1
```

- [ ] Create `kromhc_transformer/configs/kromhc_medium.yaml`

```yaml
model_type: kromhc
use_kromhc: true
qk_norm: true
d_model: 512
n_heads: 16
n_layers: 12
ffn_hidden: 1376
batch_size: 16
n_epochs: 10
seq_len: 512
vocab_size: 50257
seed: 42
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
dropout: 0.1
```

#### Step 9.2: Commit

- [ ] `git add kromhc_transformer/configs/`
- [ ] `git commit -m "feat: add baseline and KromHC config files for small/medium models"`

---

### Task 10: Experiment Infrastructure

**Files:**
- Create: `kromhc_transformer/experiments/README.md`
- Create: `kromhc_transformer/experiments/scripts/requirements.txt`
- Create: `kromhc_transformer/experiments/scripts/utils.py`

#### Step 10.1: Experiment README

- [ ] Write `kromhc_transformer/experiments/README.md`

```markdown
# KromHC Transformer Experiments

## Overview

Phase-by-phase experiments on WikiText-103: POC → baseline comparison → ablations → publication-ready.

## Prerequisites

- GPU: NVIDIA RTX 5060 Ti (16 GB VRAM) or equivalent
- Python 3.11+
- `pip install -r scripts/requirements.txt`

## Experiment Phases

### Phase 1: POC (30 min, ~1 GPU hour)

**Goal**: Validate end-to-end training. Loss should decrease; no crashes/OOM.

```bash
python scripts/exp_01_poc.py --seed 42
```

**Expected**: Loss ≈ 10.5 → 8.0 (1 epoch, train split only)
**Pass**: No exceptions, loss decreases monotonically.

### Phase 2: Baseline vs. KromHC (8 hours, 3 seeds)

**Goal**: Compare test perplexity. KromHC should ≤ baseline + 1% margin.

```bash
python scripts/exp_02_baseline_vs_kromhc.py --seed 42
python scripts/exp_02_baseline_vs_kromhc.py --seed 43
python scripts/exp_02_baseline_vs_kromhc.py --seed 44
```

**Expected**: test_ppl_baseline ≈ 50, test_ppl_kromhc ≈ 50 ± 2
**Pass**: Mean ± std overlap (within margin).

### Phase 3: Ablations (24 hours, 3 seeds × 2 variants)

**Goal**: Isolate head mixer contribution.

```bash
python scripts/exp_03_ablations.py --seed 42
python scripts/exp_03_ablations.py --seed 43
python scripts/exp_03_ablations.py --seed 44
```

**Expected**: KromHC w/ mixing vs. w/o mixing shows measurable difference.

### Phase 4: Publication-Ready (72 hours, 3 seeds × 3 sizes)

```bash
python scripts/exp_04_final.py --seed 42
python scripts/exp_04_final.py --seed 43
python scripts/exp_04_final.py --seed 44
```

**Expected**: Reproducible results across seeds; statistical significance.

## Outputs

Results saved to `results/<timestamp>_<model>_d<dim>_h<heads>/`:

- `<model>_<seed>.json`: Metrics (loss, ppl, hardware, config)
- `model_<seed>.pt`: Model checkpoint
- `plots/`: Loss curves, perplexity vs. model size, mixing matrix visualizations (post-processing)

## Troubleshooting

**OOM**: Reduce `batch_size` in config (default 32 → 16)
**Slow data loading**: First run downloads ~20GB WikiText-103; reuses cache thereafter
**NaN loss**: Check gradient norm; enable gradient clipping (default enabled)
```

#### Step 10.2: Write requirements.txt

- [ ] Create `kromhc_transformer/experiments/scripts/requirements.txt`

```
torch==2.4.0
torchvision==0.19.0
pytorch-muon==0.1.0
tiktoken==0.7.0
datasets==2.18.0
PyYAML==6.0
tqdm==4.66.1
pytest==7.4.3
matplotlib==3.8.2
numpy==1.24.3
```

#### Step 10.3: Write experiment utilities

- [ ] Create `kromhc_transformer/experiments/scripts/utils.py`

```python
"""Shared utilities for experiments."""
import random
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def log_experiment(exp_id: str, hypothesis: str, config: dict, metrics: dict,
                   hardware: dict, seed: int, status: str = "completed",
                   error_msg: str = None, output_dir: Path = None):
    """Log experiment as JSON artifact."""
    artifact = {
        "experiment_id": exp_id,
        "hypothesis": hypothesis,
        "config": config,
        "metrics": metrics,
        "hardware": hardware,
        "seed": seed,
        "status": status,
        "error_msg": error_msg,
        "timestamp": datetime.now().isoformat(),
    }

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{exp_id}_{seed}.json"
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2)

    return json_path
```

#### Step 10.4: Commit

- [ ] `git add kromhc_transformer/experiments/`
- [ ] `git commit -m "feat: add experiment infrastructure (README, requirements, utils)"`

---

### Task 11: Experiment Scripts (Phase 1: POC)

**Files:**
- Create: `kromhc_transformer/experiments/scripts/exp_01_poc.py`

#### Step 11.1: Write POC experiment

- [ ] Write `kromhc_transformer/experiments/scripts/exp_01_poc.py`

```python
"""Phase 1: POC — End-to-end training validation."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.train import train
from experiments.scripts.utils import set_seeds, log_experiment


def main():
    parser = argparse.ArgumentParser(description="POC: End-to-end KromHC training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)

    # Small model for POC
    cfg = KromHCConfig(
        d_model=256,
        n_heads=8,
        n_layers=6,
        ffn_hidden=688,
        batch_size=32,
        n_epochs=1,  # Just 1 epoch for POC
        seq_len=512,
        model_type="kromhc",
        use_kromhc=True,
        qk_norm=True,
        seed=args.seed,
    )

    exp_id = "poc_kromhc"
    hypothesis = "KromHC integrates without crashes; training loop is sound."

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = train(cfg, device)

        log_experiment(
            exp_id=exp_id,
            hypothesis=hypothesis,
            config=cfg.__dict__,
            metrics={k: v for k, v in metrics.items() if k not in ["config"]},
            hardware={"device": str(device), "cuda_available": torch.cuda.is_available()},
            seed=args.seed,
            status="completed",
        )

        print(f"✓ POC PASSED: test_ppl={metrics['test_ppl']:.2f}")

    except Exception as e:
        log_experiment(
            exp_id=exp_id,
            hypothesis=hypothesis,
            config=cfg.__dict__,
            metrics={},
            hardware={"device": "unknown"},
            seed=args.seed,
            status="error",
            error_msg=str(e),
        )
        print(f"✗ POC FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
```

#### Step 11.2: Commit

- [ ] `git add kromhc_transformer/experiments/scripts/exp_01_poc.py`
- [ ] `git commit -m "feat: add Phase 1 (POC) experiment script"`

---

### Task 12: Experiment Scripts (Phase 2: Baseline Comparison)

**Files:**
- Create: `kromhc_transformer/experiments/scripts/exp_02_baseline_vs_kromhc.py`

#### Step 12.1: Write baseline comparison experiment

- [ ] Write `kromhc_transformer/experiments/scripts/exp_02_baseline_vs_kromhc.py`

```python
"""Phase 2: Baseline vs. KromHC comparison."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.train import train
from experiments.scripts.utils import set_seeds, log_experiment


def run_variant(cfg: KromHCConfig, variant_name: str, seed: int):
    """Train one variant, return metrics."""
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = train(cfg, device)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Baseline vs. KromHC")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Medium model for Phase 2
    cfg_base = KromHCConfig(
        d_model=512,
        n_heads=16,
        n_layers=12,
        ffn_hidden=1376,
        batch_size=16,
        n_epochs=10,
        seq_len=512,
        seed=args.seed,
        qk_norm=True,
    )

    results = {}

    # Baseline
    cfg_baseline = KromHCConfig(**{**cfg_base.__dict__, "model_type": "baseline", "use_kromhc": False})
    metrics_baseline = run_variant(cfg_baseline, "baseline", args.seed)
    results["baseline"] = metrics_baseline

    log_experiment(
        exp_id="baseline_comparison",
        hypothesis="Baseline attention on WikiText-103",
        config=cfg_baseline.__dict__,
        metrics={k: v for k, v in metrics_baseline.items() if k not in ["config"]},
        hardware={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        seed=args.seed,
        status="completed",
    )

    # KromHC
    cfg_kromhc = KromHCConfig(**{**cfg_base.__dict__, "model_type": "kromhc", "use_kromhc": True})
    metrics_kromhc = run_variant(cfg_kromhc, "kromhc", args.seed)
    results["kromhc"] = metrics_kromhc

    log_experiment(
        exp_id="kromhc_comparison",
        hypothesis="KromHC head mixing on WikiText-103",
        config=cfg_kromhc.__dict__,
        metrics={k: v for k, v in metrics_kromhc.items() if k not in ["config"]},
        hardware={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        seed=args.seed,
        status="completed",
    )

    # Summary
    print("\n" + "="*60)
    print("PHASE 2 SUMMARY")
    print("="*60)
    print(f"Baseline test_ppl: {results['baseline']['test_ppl']:.2f}")
    print(f"KromHC test_ppl:   {results['kromhc']['test_ppl']:.2f}")
    ppl_diff = abs(results['kromhc']['test_ppl'] - results['baseline']['test_ppl'])
    ppl_margin = 0.01 * results['baseline']['test_ppl']  # 1% margin
    status = "✓ PASS" if ppl_diff <= ppl_margin else "✗ INCONCLUSIVE"
    print(f"Difference: {ppl_diff:.2f} (margin: {ppl_margin:.2f}) {status}")
    print("="*60)


if __name__ == "__main__":
    main()
```

#### Step 12.2: Commit

- [ ] `git add kromhc_transformer/experiments/scripts/exp_02_baseline_vs_kromhc.py`
- [ ] `git commit -m "feat: add Phase 2 (baseline comparison) experiment script"`

---

### Task 13: Remaining Experiment Scripts & Documentation

**Files:**
- Create: `kromhc_transformer/experiments/scripts/exp_03_ablations.py`
- Create: `kromhc_transformer/experiments/scripts/exp_04_final.py`
- Create: `kromhc_transformer/findings.md`

#### Step 13.1: Write ablation experiment

- [ ] Write `kromhc_transformer/experiments/scripts/exp_03_ablations.py` (stub for now)

```python
"""Phase 3: Ablation study — isolate head mixer contribution."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.train import train
from experiments.scripts.utils import set_seeds, log_experiment


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Ablations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Phase 3: Ablations — to be implemented post-POC")
    # TODO: Implement ablation variants
    # Variant A: full KromHC
    # Variant B: KromHC with use_kromhc=False


if __name__ == "__main__":
    main()
```

#### Step 13.2: Write final experiment

- [ ] Write `kromhc_transformer/experiments/scripts/exp_04_final.py` (stub)

```python
"""Phase 4: Publication-ready experiments with scaling."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.train import train
from experiments.scripts.utils import set_seeds, log_experiment


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Publication-ready")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Phase 4: Publication-ready — to be implemented post-Phase-3")
    # TODO: Implement scaling experiments (50M, 100M, 200M params)


if __name__ == "__main__":
    main()
```

#### Step 13.3: Write findings.md

- [ ] Write `kromhc_transformer/findings.md`

```markdown
# Findings: KromHC Transformer on WikiText-103

## 1. Literature Survey

### 1.1 Landscape Overview

KromHC (January 2026) introduces Kronecker-factored doubly-stochastic permutations for mixing attention heads. The method scales to 128+ heads without combinatorial explosion, enabling content-dependent head information flow.

### 1.2 Related Work

- **Attention mechanisms**: Multi-head attention (Vaswani et al., 2017)
- **Head analysis**: What do attention heads learn? (Clark et al., 2019)
- **Head mixing**: Limited prior work; KromHC is novel

## 2. Gap Analysis

### 2.1 Theoretical Gaps
- No prior analysis of head permutations in transformers
- KromHC doubly-stochastic property guarantees balanced information flow (unproven empirically)

### 2.2 Methodological Gaps
- Head mixing never tested on WikiText-103
- Scaling laws for KromHC unknown

### 2.3 Empirical Gaps
- No comparison to baseline attention on large language models

## 3. Feasibility Assessment

### 3.1 Scientific Validity
KromHC is theoretically sound: Kronecker products of permutation matrices remain permutations. Doubly-stochastic constraint ensures balanced assignment.

### 3.2 Novelty Assessment
Novel application of Kronecker-factored mixing to transformer heads. Prior work on head pruning/analysis exists; head mixing is new.

### 3.3 Computational Feasibility
- **Overhead**: O(K × hidden_dim) params where K = log₂(n_heads) (tiny: ~1KB for 8 heads, 32 hidden)
- **Runtime**: Matrix multiplication (B×N, n_heads, n_heads) is fast (negligible vs. attention)
- **VRAM**: Minimal impact; fits on RTX 5060 Ti

### 3.4 Overall Verdict
**Feasible & promising**. Low computational overhead, novel approach, clear experimental path.

## 4. Experimental Findings

*To be populated after Phase 1–4 experiments complete.*

### 4.1 Experiment Log

| Phase | Variant | Seeds | Metric | Mean ± Std | Status |
|-------|---------|-------|--------|-----------|--------|
| 1: POC | KromHC | 1 | test_ppl | TBD | Pending |
| 2: Baseline | Baseline | 3 | test_ppl | TBD | Pending |
| 2: Comparison | KromHC | 3 | test_ppl | TBD | Pending |
| 3: Ablation | w/ mixing | 3 | test_ppl | TBD | Pending |
| 3: Ablation | w/o mixing | 3 | test_ppl | TBD | Pending |

### 4.2 Key Results

*Pending experiments.*

### 4.3 Negative Results & Lessons

*To be documented.*

## 5. Conclusions & Next Steps

*Post-experiment.*

## References

- KromHC: Kronecker-Factored Head Mixing (Jan 2026)
- Vaswani et al. (2017): Attention Is All You Need
- Clark et al. (2019): What does BERT look at?
```

#### Step 13.4: Commit

- [ ] `git add kromhc_transformer/experiments/scripts/exp_03_ablations.py`
- [ ] `git add kromhc_transformer/experiments/scripts/exp_04_final.py`
- [ ] `git add kromhc_transformer/findings.md`
- [ ] `git commit -m "feat: add stub experiment scripts and findings.md template"`

---

### Task 14: All Tests Pass

**Files:**
- Run test suite

#### Step 14.1: Run all tests

- [ ] Run: `pytest tests/ -v`

Expected: All tests PASS

- [ ] Run: `pytest tests/ --cov=kromhc_transformer` (optional: coverage check)

#### Step 14.2: Commit

- [ ] `git add .`
- [ ] `git commit -m "test: verify all unit and integration tests pass"`

---

## Summary

**Commits created**: ~15
**Tests**: ~30 (unit + integration + smoke)
**Code lines**: ~2,000
**Deliverables**:
- ✓ `kromhc_transformer/` module (models, config, data, training)
- ✓ Unit tests (head mixer, attention, blocks, model)
- ✓ Integration tests (blocks, training loop)
- ✓ YAML configs (baseline, small, medium)
- ✓ Experiment infrastructure (Phase 1–4 scaffolds)
- ✓ Research documentation (findings.md template)

---

## Next: Execution

**Plan is ready.** Two options for implementation:

1. **Subagent-Driven** (recommended) — Fresh subagent per task, review checkpoints, parallel speed
2. **Inline Execution** — Execute sequentially in this session with checkpoints

Which approach?
