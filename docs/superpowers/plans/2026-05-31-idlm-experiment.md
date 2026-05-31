# I-DLM Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a small-scale I-DLM (Introspective Diffusion Language Model) pipeline that fine-tunes a frozen `rbf_ffn` AR checkpoint with LoRA adapters using the introspective-consistency training objective, then evaluates with Introspective Strided Decoding (ISD) on WikiText-103.

**Architecture:** New top-level `idlm/` project. `IDLMCausalLM` wraps a frozen `rbf_ffn.CausalLM` checkpoint and replaces `q_proj`/`v_proj` in every attention layer with `LoRALinear` wrappers. Training concatenates `[x_t | x_0]` (all-masked + clean) and computes a masked decode loss (L_mask) and introspection loss (L_clean) with an auto-balanced coefficient λ. ISD evaluation runs generate-and-verify strides, logging acceptance rate α and compute efficiency TPF/OH.

**Tech Stack:** PyTorch, tiktoken, HuggingFace datasets, tqdm, PyYAML. No new dependencies beyond what `rbf_ffn` already uses.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `idlm/__init__.py` | Create | Package marker |
| `idlm/__main__.py` | Create | `python -m idlm` entry point |
| `idlm/config.py` | Create | `IDLMConfig` dataclass + `load_config()` |
| `idlm/data.py` | Create | WikiText-103 data loading (adapts `rbf_ffn/data.py`) |
| `idlm/models/__init__.py` | Create | Package marker |
| `idlm/models/lora.py` | Create | `LoRALinear`, `apply_lora()` |
| `idlm/models/idlm_model.py` | Create | `IDLMCausalLM` wrapping frozen AR base + LoRA |
| `idlm/train.py` | Create | Training loop with I-DLM loss |
| `idlm/generate.py` | Create | ISD evaluation entry point |
| `idlm/configs/baseline.yaml` | Create | First experiment config (points at weight_norm checkpoint) |
| `idlm/tests/__init__.py` | Create | Package marker |
| `idlm/tests/test_config.py` | Create | Config load/validation tests |
| `idlm/tests/test_lora.py` | Create | LoRALinear isolation and gradient tests |
| `idlm/tests/test_model.py` | Create | IDLMCausalLM shape, LoRA, loss tests |
| `idlm/tests/test_isd.py` | Create | ISD output shape and metrics bounds tests |
| `idlm/README.md` | Create | Project docs |
| `pyproject.toml` | Modify | Add `idlm*` to `packages.find.include` |
| `README.md` | Modify | Add `idlm/` row to Projects table |

---

## Shared Test Constants

All test files use the same tiny model constants — define them at the top of each test file:

```python
import math, torch, pytest
B, N, V, D, H, L_layers = 2, 16, 256, 32, 4, 2
MASK_ID = 50256          # EOS token reused as MASK (vocab_size=50257)
```

---

## Task 1: Package Scaffold

**Files:**
- Create: `idlm/__init__.py`
- Create: `idlm/__main__.py`
- Create: `idlm/models/__init__.py`
- Create: `idlm/tests/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create package init files**

```python
# idlm/__init__.py
"""I-DLM: Introspective Diffusion Language Model experiment."""
```

```python
# idlm/models/__init__.py
```

```python
# idlm/tests/__init__.py
```

```python
# idlm/__main__.py
"""Allow `python -m idlm.train` and `python -m idlm.generate`."""
```

- [ ] **Step 2: Add `idlm` to pyproject.toml packages**

In `pyproject.toml`, change the `include` line:
```toml
include = ["rbf_ffn*", "kromhc_transformer*", "grokking*", "flow_matching*", "sigreg*", "mamba_lm*", "idlm*"]
```

Also add `"idlm/tests"` to `testpaths`:
```toml
testpaths = ["rbf_ffn/tests", "kromhc_transformer/tests", "grokking/tests", "sigreg/tests", "idlm/tests"]
```

- [ ] **Step 3: Verify import works**

Run:
```bash
python -c "import idlm"
```
Expected: no output (no ImportError).

- [ ] **Step 4: Commit**

```bash
git add idlm/__init__.py idlm/__main__.py idlm/models/__init__.py idlm/tests/__init__.py pyproject.toml
git commit -m "feat(idlm): scaffold package structure"
```

---

## Task 2: Config

**Files:**
- Create: `idlm/config.py`
- Create: `idlm/tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# idlm/tests/test_config.py
import pytest
from pathlib import Path
from idlm.config import IDLMConfig, load_config


def test_defaults():
    cfg = IDLMConfig(ar_checkpoint="dummy.pt")
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16.0
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]
    assert cfg.seq_len == 512
    assert cfg.batch_size == 8
    assert cfg.max_steps == 10_000
    assert cfg.lr == 3e-4
    assert cfg.warmup_steps == 200
    assert cfg.grad_clip == 1.0
    assert cfg.seed == 42
    assert cfg.eval_every == 500
    assert cfg.stride == 4
    assert cfg.num_eval_examples == 200
    assert cfg.prompt_len == 64
    assert cfg.gen_len == 128
    assert cfg.vocab_size == 50257


def test_load_config_from_yaml(tmp_path):
    yaml_text = "ar_checkpoint: /some/path.pt\nlora_rank: 4\n"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.ar_checkpoint == "/some/path.pt"
    assert cfg.lora_rank == 4
    assert cfg.seq_len == 512  # default preserved


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("ar_checkpoint: x.pt\nunknown_key: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(p)


def test_ar_checkpoint_required():
    with pytest.raises(TypeError):
        IDLMConfig()
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest idlm/tests/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'idlm.config'`

- [ ] **Step 3: Implement config.py**

```python
# idlm/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class IDLMConfig:
    # Required: path to a trained rbf_ffn CausalLM checkpoint (.pt file)
    ar_checkpoint: str

    # LoRA
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    seq_len: int = 512
    batch_size: int = 8
    max_steps: int = 10_000
    lr: float = 3e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 42

    # Evaluation / ISD
    eval_every: int = 500
    stride: int = 4
    num_eval_examples: int = 200
    prompt_len: int = 64
    gen_len: int = 128

    # Must match the AR checkpoint's vocab_size
    vocab_size: int = 50257


def load_config(path: str | Path) -> IDLMConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raise ValueError("Empty config file")
    valid = {f for f in IDLMConfig.__dataclass_fields__}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return IDLMConfig(**raw)
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest idlm/tests/test_config.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add idlm/config.py idlm/tests/test_config.py
git commit -m "feat(idlm): IDLMConfig dataclass and load_config"
```

---

## Task 3: Data Loading

**Files:**
- Create: `idlm/data.py`

No separate tests — this adapts the working `rbf_ffn/data.py`. The cache is shared (same token cache files).

- [ ] **Step 1: Create data.py**

```python
# idlm/data.py
"""
WikiText-103 data pipeline for I-DLM.

Reuses rbf_ffn's token cache (same seq_len → same cache file).
Cache lives at rbf_ffn/data_cache/ by default.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# Point at rbf_ffn's existing cache to avoid re-downloading
_CACHE_DIR = Path(__file__).parent.parent / "rbf_ffn" / "data_cache"


def chunk_tokens(tokens: list[int], seq_len: int) -> torch.Tensor:
    t = torch.tensor(tokens, dtype=torch.long)
    n = len(t) // seq_len
    return t[: n * seq_len].view(n, seq_len)


class TokenDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _load_split(split: str, seq_len: int) -> torch.Tensor:
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}_r50k_{seq_len}.pt"
    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("r50k_base")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [row["text"] for row in dataset if row["text"].strip() != ""]
    tokens = enc.encode("\n".join(texts))
    chunks = chunk_tokens(tokens, seq_len)
    torch.save(chunks, cache_file)
    print(f"Cached {len(chunks):,} sequences → {cache_file}")
    return chunks


def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader). cfg needs seq_len, batch_size, seed."""
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def _make(split: str, shuffle: bool, drop_last: bool,
               persistent: bool = False, prefetch: int | None = None) -> DataLoader:
        ds = TokenDataset(_load_split(split, cfg.seq_len))
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=4, pin_memory=True,
            generator=g if shuffle else None,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )

    return (
        _make("train",      shuffle=True,  drop_last=True,  persistent=True, prefetch=2),
        _make("validation", shuffle=False, drop_last=False),
        _make("test",       shuffle=False, drop_last=False),
    )
```

- [ ] **Step 2: Smoke-test import**

```bash
python -c "from idlm.data import get_dataloaders; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add idlm/data.py
git commit -m "feat(idlm): WikiText-103 data loader"
```

---

## Task 4: LoRA Layer

**Files:**
- Create: `idlm/models/lora.py`
- Create: `idlm/tests/test_lora.py`

- [ ] **Step 1: Write failing tests**

```python
# idlm/tests/test_lora.py
import math
import torch
import torch.nn as nn
import pytest
from idlm.models.lora import LoRALinear, apply_lora

B, N, D_IN, D_OUT, RANK = 2, 16, 32, 32, 4


def make_base_linear() -> nn.Linear:
    lin = nn.Linear(D_IN, D_OUT, bias=False)
    lin.weight.requires_grad_(False)
    return lin


def test_output_shape():
    lora = LoRALinear(make_base_linear(), rank=RANK, alpha=8.0)
    x = torch.randn(B, N, D_IN)
    mask = torch.ones(B, N, 1)
    out = lora(x, mask)
    assert out.shape == (B, N, D_OUT)


def test_lora_zero_mask_matches_base():
    """With use_lora_mask=0, output must equal the frozen base linear exactly."""
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    x = torch.randn(B, N, D_IN)
    mask = torch.zeros(B, N, 1)
    with torch.no_grad():
        out_lora = lora(x, mask)
        out_base = base(x)
    assert torch.allclose(out_lora, out_base, atol=1e-6)


def test_lora_full_mask_differs_from_base_after_update():
    """After a gradient step, lora_B is non-zero and mask=1 output differs from base."""
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    x = torch.randn(B, N, D_IN)
    mask = torch.ones(B, N, 1)
    # Manually set lora_B to non-zero
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.01)
    out_lora = lora(x, mask)
    out_base = base(x)
    assert not torch.allclose(out_lora, out_base, atol=1e-6)


def test_lora_init_delta_zero():
    """lora_B is initialised to zero so initial delta is zero."""
    lora = LoRALinear(make_base_linear(), rank=RANK, alpha=8.0)
    assert torch.all(lora.lora_B.weight == 0)


def test_only_lora_params_have_grad():
    """Only lora_A and lora_B should accumulate gradients; base weight must not."""
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    x = torch.randn(B, N, D_IN)
    mask = torch.ones(B, N, 1)
    lora(x, mask).sum().backward()
    assert base.weight.grad is None
    assert lora.lora_A.weight.grad is not None
    assert lora.lora_B.weight.grad is not None


def test_apply_lora_replaces_target_modules():
    """apply_lora replaces named sub-modules with LoRALinear."""
    model = nn.Sequential(
        nn.Linear(D_IN, D_OUT, bias=False),   # index 0 — not targeted
    )
    model.q_proj = nn.Linear(D_IN, D_OUT, bias=False)
    model.v_proj = nn.Linear(D_IN, D_OUT, bias=False)
    apply_lora(model, ["q_proj", "v_proj"], rank=RANK, alpha=8.0)
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert isinstance(model[0], nn.Linear)  # untouched


def test_per_position_mask():
    """Mask can be a mix of 0 and 1 across positions."""
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.1)
    x = torch.randn(B, N, D_IN)
    mask = torch.zeros(B, N, 1)
    mask[:, :N // 2, :] = 1.0           # first half uses LoRA, second half uses base
    out = lora(x, mask)
    # Second half should equal base output
    base_out = base(x)
    assert torch.allclose(out[:, N // 2:, :], base_out[:, N // 2:, :], atol=1e-6)
    # First half should differ
    assert not torch.allclose(out[:, :N // 2, :], base_out[:, :N // 2, :], atol=1e-6)
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest idlm/tests/test_lora.py -v
```
Expected: `ModuleNotFoundError: No module named 'idlm.models.lora'`

- [ ] **Step 3: Implement lora.py**

```python
# idlm/models/lora.py
from __future__ import annotations
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Frozen base linear + trainable low-rank adapter.

    forward(x, use_lora_mask) applies the LoRA delta scaled by use_lora_mask,
    so the mask can be 0 (base only) or 1 (base + LoRA) per position.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        d_out, d_in = base.weight.shape
        self.base = base
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, use_lora_mask: torch.Tensor) -> torch.Tensor:
        """
        x:             (..., d_in)
        use_lora_mask: (..., 1)  — 1.0 at positions where LoRA is active, 0.0 elsewhere
        """
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x)) * self.scale
        return out + delta * use_lora_mask


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
) -> None:
    """
    Walk `model` and replace any direct attribute in `target_modules` that is
    an nn.Linear with a LoRALinear wrapper (in-place).

    Only replaces direct named children (not nested); call on each sub-module
    that contains the target linears (e.g., each attention block).
    For a full model, this function recurses into all sub-modules.
    """
    for name, module in list(model.named_children()):
        if name in target_modules and isinstance(module, nn.Linear):
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)
            setattr(model, name, LoRALinear(module, rank, alpha))
        else:
            apply_lora(module, target_modules, rank, alpha)
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest idlm/tests/test_lora.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add idlm/models/lora.py idlm/tests/test_lora.py
git commit -m "feat(idlm): LoRALinear with per-position mask + apply_lora"
```

---

## Task 5: IDLMCausalLM

**Files:**
- Create: `idlm/models/idlm_model.py`
- Create: `idlm/tests/test_model.py`

**Key design decisions:**
- `IDLMCausalLM` loads a `rbf_ffn.CausalLM` checkpoint, freezes all weights, then applies LoRA
- `forward(tokens, use_lora_mask)` sets the mask on all `LoRALinear` layers before the base model's forward
- The attention blocks call `q_proj(x)` and `v_proj(x)` — but those are now `LoRALinear` instances that expect `(x, mask)`. To avoid modifying `rbf_ffn`, we use a **hook** approach: we register a forward pre-hook on each `LoRALinear` that injects the current mask from a stored attribute on `IDLMCausalLM`

**Hook approach:** `IDLMCausalLM` stores `self._current_lora_mask` and registers a forward pre-hook on every `LoRALinear`. The hook supplies the mask automatically, so `LoRALinear.forward` is called with the stored mask.

Wait — the hook approach works if we intercept `LoRALinear.forward`. But `LoRALinear.forward` requires `use_lora_mask` as an argument. The base model calls `q_proj(x)` which goes to `LoRALinear.forward(x)` — missing the second arg.

**Revised approach:** Store the mask as an attribute on `LoRALinear` directly, and default to `None` → no LoRA applied when mask is None. Then `IDLMCausalLM` sets `.current_mask` on each `LoRALinear` before calling the base model forward. The `LoRALinear.forward(x)` (no mask argument) reads from `self.current_mask`.

This keeps the `LoRALinear` interface clean — `forward(x)` is the standard `nn.Linear` interface, compatible with the existing model forward.

**Revised `LoRALinear.forward` (no mask argument):**
```python
def forward(self, x):
    out = self.base(x)
    if self.current_mask is not None:
        out = out + self.lora_B(self.lora_A(x)) * self.scale * self.current_mask
    return out
```

This requires updating `lora.py` and its tests. The per-position mask is stored as `self.current_mask` (set externally).

- [ ] **Step 1: Update lora.py to use stored mask attribute**

Replace `idlm/models/lora.py` with:

```python
# idlm/models/lora.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """
    Frozen base linear + trainable low-rank adapter.

    Set `current_mask` (shape: ..., 1) before each forward to control which
    positions use LoRA. Positions with mask=0 use only the frozen base weights.
    mask=None → LoRA always active (use for inference without position control).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        d_out, d_in = base.weight.shape
        self.base = base
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.current_mask: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x)) * self.scale
        if self.current_mask is not None:
            delta = delta * self.current_mask
        return out + delta


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
) -> None:
    """Recursively replace named nn.Linear children in target_modules with LoRALinear."""
    for name, module in list(model.named_children()):
        if name in target_modules and isinstance(module, nn.Linear):
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)
            setattr(model, name, LoRALinear(module, rank, alpha))
        else:
            apply_lora(module, target_modules, rank, alpha)
```

- [ ] **Step 2: Update test_lora.py for new interface**

Replace `idlm/tests/test_lora.py` with:

```python
# idlm/tests/test_lora.py
import math
import torch
import torch.nn as nn
import pytest
from idlm.models.lora import LoRALinear, apply_lora

B, N, D_IN, D_OUT, RANK = 2, 16, 32, 32, 4


def make_base_linear() -> nn.Linear:
    lin = nn.Linear(D_IN, D_OUT, bias=False)
    lin.weight.requires_grad_(False)
    return lin


def test_output_shape():
    lora = LoRALinear(make_base_linear(), rank=RANK, alpha=8.0)
    x = torch.randn(B, N, D_IN)
    lora.current_mask = torch.ones(B, N, 1)
    out = lora(x)
    assert out.shape == (B, N, D_OUT)


def test_lora_zero_mask_matches_base():
    """With current_mask=0, output must equal the frozen base exactly."""
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    # Set lora_B non-zero so any delta would matter
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.1)
    x = torch.randn(B, N, D_IN)
    lora.current_mask = torch.zeros(B, N, 1)
    with torch.no_grad():
        out_lora = lora(x)
        out_base = base(x)
    assert torch.allclose(out_lora, out_base, atol=1e-6)


def test_lora_full_mask_differs_from_base():
    """With current_mask=1 and non-zero lora_B, output differs from base."""
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.01)
    x = torch.randn(B, N, D_IN)
    lora.current_mask = torch.ones(B, N, 1)
    out_lora = lora(x)
    out_base = base(x)
    assert not torch.allclose(out_lora, out_base, atol=1e-6)


def test_lora_init_delta_zero():
    lora = LoRALinear(make_base_linear(), rank=RANK, alpha=8.0)
    assert torch.all(lora.lora_B.weight == 0)


def test_only_lora_params_have_grad():
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    lora.current_mask = torch.ones(B, N, 1)
    x = torch.randn(B, N, D_IN)
    lora(x).sum().backward()
    assert base.weight.grad is None
    assert lora.lora_A.weight.grad is not None
    assert lora.lora_B.weight.grad is not None


def test_apply_lora_replaces_target_modules():
    model = nn.Module()
    model.q_proj = nn.Linear(D_IN, D_OUT, bias=False)
    model.v_proj = nn.Linear(D_IN, D_OUT, bias=False)
    model.other  = nn.Linear(D_IN, D_OUT, bias=False)
    apply_lora(model, ["q_proj", "v_proj"], rank=RANK, alpha=8.0)
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert isinstance(model.other,  nn.Linear)  # untouched


def test_per_position_mask():
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.1)
    x = torch.randn(B, N, D_IN)
    mask = torch.zeros(B, N, 1)
    mask[:, :N // 2, :] = 1.0
    lora.current_mask = mask
    out = lora(x)
    base_out = base(x)
    assert torch.allclose(out[:, N // 2:], base_out[:, N // 2:], atol=1e-6)
    assert not torch.allclose(out[:, :N // 2], base_out[:, :N // 2], atol=1e-6)
```

- [ ] **Step 3: Run lora tests — expect pass**

```bash
pytest idlm/tests/test_lora.py -v
```
Expected: 7 passed.

- [ ] **Step 4: Write failing model tests**

```python
# idlm/tests/test_model.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.model import CausalLM
from idlm.models.idlm_model import IDLMCausalLM

B, N, V, D, H, L_layers = 2, 16, 256, 32, 4, 2
MASK_ID = 50256


def make_ar_model() -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L_layers,
        vocab_size=V, seq_len=N,
        ffn_hidden=86, dropout=0.0,
    )
    return CausalLM(cfg)


def make_idlm(ar_model: CausalLM) -> IDLMCausalLM:
    return IDLMCausalLM(ar_model, lora_rank=4, lora_alpha=8.0,
                        lora_target_modules=["q_proj", "v_proj"])


def test_output_shape():
    """Forward over 2N tokens returns (B, 2N, V) logits."""
    model = make_idlm(make_ar_model())
    tokens = torch.randint(0, V, (B, 2 * N))
    mask = torch.zeros(B, 2 * N, 1)
    mask[:, :N, :] = 1.0
    logits = model(tokens, mask)
    assert logits.shape == (B, 2 * N, V)


def test_ar_weights_frozen():
    """All non-LoRA parameters must have requires_grad=False."""
    from idlm.models.lora import LoRALinear
    ar = make_ar_model()
    model = make_idlm(ar)
    for name, param in model.named_parameters():
        module_name = name.rsplit(".", 1)[0]
        # LoRA params are inside LoRALinear sub-modules named lora_A / lora_B
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_zero_mask_equals_base_ar():
    """With use_lora_mask=0 everywhere, output equals frozen AR model output."""
    ar = make_ar_model()
    model = make_idlm(ar)
    # Manually set lora_B non-zero so any delta would matter
    from idlm.models.lora import LoRALinear
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.weight.fill_(0.05)
    tokens = torch.randint(0, V, (B, 2 * N))
    mask = torch.zeros(B, 2 * N, 1)
    with torch.no_grad():
        idlm_out = model(tokens, mask)
        ar_out, _ = ar(tokens)
    assert torch.allclose(idlm_out, ar_out, atol=1e-5)


def test_only_lora_grads_flow():
    """Backward should only leave gradients on LoRA params."""
    ar = make_ar_model()
    model = make_idlm(ar)
    tokens = torch.randint(0, V, (B, 2 * N))
    mask = torch.ones(B, 2 * N, 1)
    logits = model(tokens, mask)
    logits.sum().backward()
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.grad is not None, f"{name} missing grad"
        else:
            assert param.grad is None, f"{name} should have no grad"


def test_lora_count():
    """Number of LoRA layers = n_layers * len(target_modules)."""
    from idlm.models.lora import LoRALinear
    ar = make_ar_model()
    model = make_idlm(ar)
    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    # 2 target modules (q_proj, v_proj) * L_layers layers = 4 for attn_type=standard
    # (standard attention has q_proj, k_proj, v_proj, o_proj)
    assert len(lora_layers) == L_layers * 2
```

- [ ] **Step 5: Run — expect failure**

```bash
pytest idlm/tests/test_model.py -v
```
Expected: `ModuleNotFoundError: No module named 'idlm.models.idlm_model'`

- [ ] **Step 6: Implement idlm_model.py**

```python
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
        tokens:        (B, 2L) int64  — concat of [x_t | x_0]
        use_lora_mask: (B, 2L, 1) float — 1.0 at x_t (decode) positions,
                       0.0 at x_0 (introspection) positions
        returns:       (B, 2L, vocab_size) logits
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
        """
        tokens:        (B, L) or (B, 2L) int64
        use_lora_mask: (B, L, 1) or (B, 2L, 1) float
        returns:       (B, L, vocab_size) logits
        """
        self._set_mask(use_lora_mask)
        logits, _ = self.model(tokens)
        self._set_mask(None)
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
```

- [ ] **Step 7: Run model tests — expect pass**

```bash
pytest idlm/tests/test_model.py -v
```
Expected: 5 passed.

- [ ] **Step 8: Commit**

```bash
git add idlm/models/lora.py idlm/tests/test_lora.py idlm/models/idlm_model.py idlm/tests/test_model.py
git commit -m "feat(idlm): IDLMCausalLM with frozen AR base and per-position LoRA"
```

---

## Task 6: Training Loss + train.py

**Files:**
- Create: `idlm/train.py`
- Extend: `idlm/tests/test_model.py` (add loss tests)

**Loss design:**
- Input tokens: `concat([x_t, x_0])` of length `2L` where `x_t = [MASK_ID]*L`
- `use_lora_mask`: shape `(B, 2L, 1)`, value 1.0 for positions `0..L-1`, 0.0 for `L..2L-1`
- `L_mask`: `CE(logits[:, :L], x_0)` — predict clean token at each masked position (0-shift)
- `L_clean`: `CE(logits[:, L:2L-1], x_0[:, 1:])` — standard AR shift in x_0 half (L-1 tokens)
- `λ = stop_gradient(L_mask / L_clean)` — auto-balanced coefficient
- `loss = L_mask + λ * L_clean`
- `MASK_ID = 50256` (GPT-2 EOS token reused as the diffusion MASK token)

- [ ] **Step 1: Add loss tests to test_model.py**

Append to `idlm/tests/test_model.py`:

```python
# --- Loss tests ---

def compute_idlm_loss(model, x_0, device):
    """Helper: build 2L input, run forward, return (loss, l_mask, l_clean, lam)."""
    import torch.nn.functional as F
    MASK_ID = 50256
    B, L = x_0.shape
    x_t = torch.full_like(x_0, MASK_ID)
    tokens = torch.cat([x_t, x_0], dim=1)          # (B, 2L)
    mask = torch.zeros(B, 2 * L, 1, device=device)
    mask[:, :L, :] = 1.0
    logits = model(tokens, mask)                    # (B, 2L, V)
    l_mask = F.cross_entropy(
        logits[:, :L].reshape(-1, logits.size(-1)),
        x_0.reshape(-1),
    )
    l_clean = F.cross_entropy(
        logits[:, L:2 * L - 1].reshape(-1, logits.size(-1)),
        x_0[:, 1:].reshape(-1),
    )
    with torch.no_grad():
        lam = (l_mask / (l_clean + 1e-8)).detach()
    loss = l_mask + lam * l_clean
    return loss, l_mask, l_clean, lam


def test_loss_terms_finite():
    model = make_idlm(make_ar_model())
    x_0 = torch.randint(0, V - 1, (B, N))  # avoid MASK_ID in targets
    loss, l_mask, l_clean, lam = compute_idlm_loss(model, x_0, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert torch.isfinite(l_mask)
    assert torch.isfinite(l_clean)
    assert lam > 0


def test_lambda_has_no_grad():
    """λ is stop-gradient — it must not appear in the computation graph."""
    model = make_idlm(make_ar_model())
    x_0 = torch.randint(0, V - 1, (B, N))
    loss, l_mask, l_clean, lam = compute_idlm_loss(model, x_0, torch.device("cpu"))
    assert not lam.requires_grad


def test_loss_backward_updates_lora_only():
    """After loss.backward(), only LoRA params should have gradients."""
    model = make_idlm(make_ar_model())
    x_0 = torch.randint(0, V - 1, (B, N))
    loss, *_ = compute_idlm_loss(model, x_0, torch.device("cpu"))
    loss.backward()
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.grad is not None, f"{name} missing grad after loss.backward()"
        else:
            assert param.grad is None, f"{name} should have no grad"
```

- [ ] **Step 2: Run new loss tests — expect pass**

```bash
pytest idlm/tests/test_model.py -v
```
Expected: 8 passed (5 original + 3 new).

- [ ] **Step 3: Implement train.py**

```python
# idlm/train.py
"""
Training entry point for I-DLM.

Usage:
    python -m idlm.train --config idlm/configs/baseline.yaml
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from rbf_ffn.config import load_config as load_ar_config
from rbf_ffn.models.model import CausalLM
from idlm.config import IDLMConfig, load_config
from idlm.data import get_dataloaders
from idlm.models.idlm_model import IDLMCausalLM

MASK_TOKEN_ID = 50256   # GPT-2 EOS token reused as diffusion MASK


def get_experiment_dir(cfg: IDLMConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{stamp}_idlm_r{cfg.lora_rank}_s{cfg.stride}"
    path = Path(__file__).parent / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def compute_loss(
    model: IDLMCausalLM,
    x_0: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, float]:
    """
    Build [x_t | x_0] input, run forward, return (loss, l_mask, l_clean, lambda).
    """
    B, L = x_0.shape
    x_t = torch.full_like(x_0, MASK_TOKEN_ID)
    tokens = torch.cat([x_t, x_0], dim=1)               # (B, 2L)

    use_lora_mask = torch.zeros(B, 2 * L, 1, device=device)
    use_lora_mask[:, :L, :] = 1.0

    logits = model(tokens, use_lora_mask)                # (B, 2L, vocab_size)

    # L_mask: predict x_0[i] at each masked position i (0-shift, decode pathway)
    l_mask = F.cross_entropy(
        logits[:, :L].reshape(-1, logits.size(-1)),
        x_0.reshape(-1),
    )

    # L_clean: standard AR shift in the x_0 half (introspection pathway)
    l_clean = F.cross_entropy(
        logits[:, L:2 * L - 1].reshape(-1, logits.size(-1)),
        x_0[:, 1:].reshape(-1),
    )

    # Auto-balanced coefficient: stop-gradient ratio
    lam = (l_mask.detach() / (l_clean.detach() + 1e-8))
    loss = l_mask + lam * l_clean

    return loss, l_mask.item(), l_clean.item(), lam.item()


@torch.no_grad()
def run_isd_eval(
    model: IDLMCausalLM,
    test_loader,
    cfg: IDLMConfig,
    device: torch.device,
) -> dict:
    """
    Run ISD evaluation on cfg.num_eval_examples test sequences.
    Returns dict with alpha_mean, tpf_oh_mean.
    Full ISD metrics are computed in generate.py; this is the training-loop
    lightweight version that tracks acceptance rate only.
    """
    from idlm.generate import isd_acceptance_rate
    model.eval()
    alphas = []
    count = 0
    for batch in test_loader:
        batch = batch.to(device)
        for i in range(batch.size(0)):
            if count >= cfg.num_eval_examples:
                break
            seq = batch[i].tolist()
            alpha = isd_acceptance_rate(model, seq, cfg, device)
            alphas.append(alpha)
            count += 1
        if count >= cfg.num_eval_examples:
            break
    alpha_mean = sum(alphas) / len(alphas) if alphas else 0.0
    # Estimate TPF/OH from stride and acceptance rate
    tpf_oh = cfg.stride * alpha_mean  # simplified: no rejection overhead modeled
    model.train()
    return {"alpha_mean": alpha_mean, "tpf_oh_mean": tpf_oh}


def _load_ar_model(cfg: IDLMConfig, device: torch.device) -> CausalLM:
    """Load the rbf_ffn CausalLM from the checkpoint specified in cfg."""
    ckpt_path = Path(cfg.ar_checkpoint)
    # The AR checkpoint stores a full model state dict; we need the ModelConfig.
    # The config.yaml lives alongside the checkpoint in the experiment dir.
    config_yaml = ckpt_path.parent / "config.yaml"
    ar_cfg = load_ar_config(config_yaml)
    ar_model = CausalLM(ar_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ar_model.load_state_dict(ckpt["model"])
    return ar_model


def train(cfg: IDLMConfig, config_path: Path) -> Path:
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = get_experiment_dir(cfg)
    shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, _, test_loader = get_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    ar_model = _load_ar_model(cfg, device)
    model = IDLMCausalLM(
        ar_model,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_target_modules=cfg.lora_target_modules,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")
    with open(exp_dir / "model_info.json", "w") as f:
        json.dump({"n_trainable_params": n_trainable}, f, indent=2)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = LambdaLR(optimizer, make_lr_lambda(cfg.warmup_steps, cfg.max_steps))

    # ── Compile ───────────────────────────────────────────────────────────────
    if device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    pbar = tqdm(total=cfg.max_steps, desc="training", unit="step", dynamic_ncols=True)
    step = 0
    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    best_alpha = 0.0

    while step < cfg.max_steps:
        model.train()
        batch = next_batch().to(device)
        x_0 = batch                                        # (B, L)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss, l_mask, l_clean, lam = compute_loss(model, x_0, device)

        loss.backward()
        raw_model = getattr(model, "_orig_mod", model)
        torch.nn.utils.clip_grad_norm_(
            [p for p in raw_model.parameters() if p.requires_grad],
            cfg.grad_clip,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", l_mask=f"{l_mask:.4f}", lam=f"{lam:.3f}")

        row: dict = {
            "step": step,
            "loss": loss.item(),
            "l_mask": l_mask,
            "l_clean": l_clean,
            "lambda": lam,
        }

        if step % cfg.eval_every == 0:
            raw_model = getattr(model, "_orig_mod", model)
            eval_metrics = run_isd_eval(raw_model, test_loader, cfg, device)
            row.update(eval_metrics)
            print(f"\n[step {step}] {row}")

            if eval_metrics["alpha_mean"] > best_alpha:
                best_alpha = eval_metrics["alpha_mean"]
                ckpt_path = exp_dir / "checkpoint_best.pt"
                # Save only LoRA weights
                lora_state = {
                    k: v for k, v in raw_model.state_dict().items()
                    if "lora_A" in k or "lora_B" in k
                }
                torch.save({"lora_state": lora_state, "step": step,
                            "alpha": best_alpha}, ckpt_path)

        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    pbar.close()
    raw_model = getattr(model, "_orig_mod", model)
    lora_state = {k: v for k, v in raw_model.state_dict().items()
                  if "lora_A" in k or "lora_B" in k}
    torch.save({"lora_state": lora_state, "step": step}, exp_dir / "checkpoint_final.pt")
    print(f"Done. Metrics → {metrics_path}")
    return exp_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, Path(args.config))
```

- [ ] **Step 4: Run all model tests — expect pass**

```bash
pytest idlm/tests/test_model.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add idlm/train.py idlm/tests/test_model.py
git commit -m "feat(idlm): training loop with I-DLM loss (L_mask + auto-balanced L_clean)"
```

---

## Task 7: ISD Evaluation (generate.py)

**Files:**
- Create: `idlm/generate.py`
- Create: `idlm/tests/test_isd.py`

**ISD algorithm (simplified for metrics):**
1. For each stride step, build `[prompt + accepted_so_far + MASK*S]`
2. Single forward: LoRA active at MASK positions (decode → q), inactive at accepted positions (verify → p)
3. Sample S tokens from q at MASK positions; compute p probs for accepted tokens
4. α for this step = mean min(1, p/q) across accepted tokens
5. Slide window; repeat for `gen_len // stride` steps
6. TPF/OH = stride / (1 + estimated_rejection_overhead), where overhead = (1 - α_mean) * stride

- [ ] **Step 1: Write failing ISD tests**

```python
# idlm/tests/test_isd.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.model import CausalLM
from idlm.models.idlm_model import IDLMCausalLM
from idlm.config import IDLMConfig
from idlm.generate import isd_acceptance_rate, isd_generate, compute_tpf_oh

B, N, V, D, H, L_layers = 2, 16, 256, 32, 4, 2
MASK_ID = 50256
DEVICE = torch.device("cpu")


def make_idlm() -> IDLMCausalLM:
    cfg = ModelConfig(d_model=D, n_heads=H, n_layers=L_layers,
                      vocab_size=V, seq_len=N, ffn_hidden=86, dropout=0.0)
    ar = CausalLM(cfg)
    return IDLMCausalLM(ar, lora_rank=4, lora_alpha=8.0,
                        lora_target_modules=["q_proj", "v_proj"])


def make_eval_cfg() -> IDLMConfig:
    return IDLMConfig(
        ar_checkpoint="dummy.pt",
        stride=2,
        prompt_len=4,
        gen_len=8,
        num_eval_examples=2,
        vocab_size=V,
    )


def test_isd_generate_output_length():
    """Generated sequence has prompt_len + gen_len tokens."""
    model = make_idlm()
    cfg = make_eval_cfg()
    prompt = list(range(cfg.prompt_len))
    tokens = isd_generate(model, prompt, cfg, DEVICE)
    assert len(tokens) == cfg.prompt_len + cfg.gen_len


def test_alpha_in_bounds():
    """Acceptance rate alpha must be in [0, 1]."""
    model = make_idlm()
    cfg = make_eval_cfg()
    seq = list(range(cfg.prompt_len + cfg.gen_len))
    alpha = isd_acceptance_rate(model, seq, cfg, DEVICE)
    assert 0.0 <= alpha <= 1.0


def test_tpf_oh_positive():
    alpha = 0.85
    stride = 4
    tpf = compute_tpf_oh(alpha, stride)
    assert tpf > 0


def test_tpf_oh_high_alpha():
    """At alpha=1.0 (perfect acceptance), TPF/OH = stride."""
    tpf = compute_tpf_oh(1.0, stride=4)
    assert abs(tpf - 4.0) < 1e-6


def test_tpf_oh_low_alpha():
    """At alpha=0.0 (all rejected), efficiency should be very low."""
    tpf = compute_tpf_oh(0.0, stride=4)
    assert tpf < 1.0
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest idlm/tests/test_isd.py -v
```
Expected: `ModuleNotFoundError: No module named 'idlm.generate'`

- [ ] **Step 3: Implement generate.py**

```python
# idlm/generate.py
"""
Introspective Strided Decoding (ISD) for I-DLM.

Usage:
    python -m idlm.generate --config idlm/configs/baseline.yaml \
        --checkpoint idlm/experiments/<run>/checkpoint_best.pt \
        --output results.json
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from idlm.config import IDLMConfig, load_config

MASK_TOKEN_ID = 50256


def compute_tpf_oh(alpha: float, stride: int) -> float:
    """
    Tokens Per Forward / Overhead ratio.

    Models efficiency as: each stride generates `stride` tokens.
    Rejected tokens must be resampled — overhead ≈ (1-alpha) * stride extra tokens.
    TPF/OH = stride / (1 + (1 - alpha) * stride)
    """
    overhead = (1.0 - alpha) * stride
    return stride / (1.0 + overhead)


@torch.no_grad()
def isd_generate(
    model,
    prompt: list[int],
    cfg: IDLMConfig,
    device: torch.device,
) -> list[int]:
    """
    Generate cfg.gen_len tokens after the prompt using ISD.

    Returns the full sequence (prompt + generated tokens).
    """
    model.eval()
    accepted = list(prompt)
    gen_len = cfg.gen_len
    stride = cfg.stride
    vocab_size = cfg.vocab_size

    while len(accepted) - len(prompt) < gen_len:
        remaining = gen_len - (len(accepted) - len(prompt))
        s = min(stride, remaining)
        current_len = len(accepted)

        tokens = torch.tensor(
            accepted + [MASK_TOKEN_ID] * s, dtype=torch.long, device=device
        ).unsqueeze(0)                                    # (1, current_len + s)

        total_len = tokens.size(1)
        use_lora_mask = torch.zeros(1, total_len, 1, device=device)
        use_lora_mask[:, current_len:, :] = 1.0          # MASK positions use LoRA

        logits = model(tokens, use_lora_mask)             # (1, total_len, V)
        q_logits = logits[0, current_len:, :]             # (s, V)
        q_probs = F.softmax(q_logits, dim=-1)

        proposed = torch.multinomial(q_probs, num_samples=1).squeeze(-1).tolist()
        accepted.extend(proposed)

    return accepted[:len(prompt) + gen_len]


@torch.no_grad()
def isd_acceptance_rate(
    model,
    sequence: list[int],
    cfg: IDLMConfig,
    device: torch.device,
) -> float:
    """
    Compute the introspective acceptance rate α on a complete sequence.

    For each stride window in the sequence (after the prompt), estimate:
        α_step = mean_k min(1, p(x_k) / q(x_k))

    where q is from the LoRA-active forward and p is from the base forward.
    Returns the mean α across all windows.
    """
    model.eval()
    prompt_len = cfg.prompt_len
    stride = cfg.stride
    gen_len = cfg.gen_len
    vocab_size = cfg.vocab_size

    gen_tokens = sequence[prompt_len:prompt_len + gen_len]
    if len(gen_tokens) < stride:
        return 1.0

    alphas = []
    for start in range(0, len(gen_tokens) - stride + 1, stride):
        window = gen_tokens[start:start + stride]
        prefix = sequence[:prompt_len + start]

        input_tokens = torch.tensor(
            prefix + [MASK_TOKEN_ID] * stride, dtype=torch.long, device=device
        ).unsqueeze(0)

        total_len = input_tokens.size(1)
        prefix_len = len(prefix)

        # q forward: LoRA active at MASK positions
        lora_mask = torch.zeros(1, total_len, 1, device=device)
        lora_mask[:, prefix_len:, :] = 1.0
        q_logits = model(input_tokens, lora_mask)[0, prefix_len:, :]  # (stride, V)
        q_probs = F.softmax(q_logits, dim=-1)

        # p forward: LoRA inactive everywhere (base AR weights)
        actual_tokens = torch.tensor(
            prefix + window, dtype=torch.long, device=device
        ).unsqueeze(0)
        p_mask = torch.zeros(1, total_len, 1, device=device)
        p_logits = model(actual_tokens, p_mask)[0, prefix_len:, :]    # (stride, V)
        p_probs = F.softmax(p_logits, dim=-1)

        # Acceptance rate for this window
        window_t = torch.tensor(window, dtype=torch.long, device=device)
        q_selected = q_probs[torch.arange(stride), window_t]          # (stride,)
        p_selected = p_probs[torch.arange(stride), window_t]          # (stride,)

        ratios = torch.minimum(
            torch.ones(stride, device=device),
            p_selected / (q_selected + 1e-10)
        )
        alphas.append(ratios.mean().item())

    return sum(alphas) / len(alphas) if alphas else 1.0


def evaluate_isd(
    model,
    test_loader,
    cfg: IDLMConfig,
    device: torch.device,
    output_path: Path,
) -> dict:
    """
    Run full ISD evaluation on cfg.num_eval_examples test sequences.
    Logs α, perplexity, and TPF/OH per example to output_path (JSONL).
    Returns summary dict.
    """
    from rbf_ffn.models.model import CausalLM
    results = []
    count = 0

    for batch in tqdm(test_loader, desc="ISD eval"):
        batch = batch.to(device)
        for i in range(batch.size(0)):
            if count >= cfg.num_eval_examples:
                break
            seq = batch[i].tolist()
            prompt = seq[:cfg.prompt_len]
            reference = seq[cfg.prompt_len:cfg.prompt_len + cfg.gen_len]

            # Generate
            generated = isd_generate(model, prompt, cfg, device)
            gen_tokens = generated[cfg.prompt_len:]

            # Acceptance rate
            alpha = isd_acceptance_rate(model, generated, cfg, device)

            # Perplexity of generated sequence vs reference under base AR
            # (use base model, lora_mask=0 everywhere)
            ref_tensor = torch.tensor(
                prompt + reference, dtype=torch.long, device=device
            ).unsqueeze(0)
            lora_off = torch.zeros(1, len(prompt) + len(reference), 1, device=device)
            with torch.no_grad():
                logits = model(ref_tensor, lora_off)
            ppl_loss = F.cross_entropy(
                logits[0, cfg.prompt_len - 1:-1],
                ref_tensor[0, cfg.prompt_len:],
            )
            ppl = math.exp(ppl_loss.item())

            tpf_oh = compute_tpf_oh(alpha, cfg.stride)

            row = {"alpha": alpha, "ppl": ppl, "tpf_oh": tpf_oh}
            results.append(row)
            count += 1

        if count >= cfg.num_eval_examples:
            break

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    alpha_mean = sum(r["alpha"] for r in results) / len(results)
    ppl_mean   = sum(r["ppl"]   for r in results) / len(results)
    tpf_mean   = sum(r["tpf_oh"] for r in results) / len(results)
    summary = {"alpha_mean": alpha_mean, "ppl_mean": ppl_mean, "tpf_oh_mean": tpf_mean,
               "n_examples": len(results)}
    print(f"ISD summary: {summary}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     default="idlm_isd_results.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load AR model + LoRA weights
    from rbf_ffn.config import load_config as load_ar_config
    from rbf_ffn.models.model import CausalLM
    from idlm.models.idlm_model import IDLMCausalLM

    ckpt_path = Path(args.checkpoint)
    config_yaml = ckpt_path.parent / "config.yaml"
    ar_cfg = load_ar_config(config_yaml)
    ar_model = CausalLM(ar_cfg).to(device)

    # Load the original AR weights from the idlm config's ar_checkpoint
    ar_ckpt = torch.load(cfg.ar_checkpoint, map_location=device, weights_only=True)
    ar_model.load_state_dict(ar_ckpt["model"])

    model = IDLMCausalLM(ar_model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules)

    # Overlay LoRA weights from the I-DLM checkpoint
    lora_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(lora_ckpt["lora_state"], strict=False)
    print(f"Loaded LoRA checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    from idlm.data import get_dataloaders
    _, _, test_loader = get_dataloaders(cfg)

    evaluate_isd(model, test_loader, cfg, device, Path(args.output))
```

- [ ] **Step 4: Run ISD tests — expect pass**

```bash
pytest idlm/tests/test_isd.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Run all idlm tests**

```bash
pytest idlm/tests/ -v
```
Expected: all tests pass (config + lora + model + isd).

- [ ] **Step 6: Commit**

```bash
git add idlm/generate.py idlm/tests/test_isd.py
git commit -m "feat(idlm): ISD evaluation — isd_generate, isd_acceptance_rate, compute_tpf_oh"
```

---

## Task 8: Config, README, Project Updates

**Files:**
- Create: `idlm/configs/baseline.yaml`
- Create: `idlm/README.md`
- Modify: `README.md`

- [ ] **Step 1: Create baseline.yaml**

Find the best weight_norm checkpoint path first:
```bash
ls rbf_ffn/experiments/ | grep "standard_swiglu_qknorm_wnorm" | sort -r | head -1
```

Then create `idlm/configs/baseline.yaml` (substitute the actual path):

```yaml
# I-DLM baseline: fine-tune the best rbf_ffn AR checkpoint (SwiGLU + QK-norm + weight-norm)
# AR base: 58.16 val PPL on WikiText-103
ar_checkpoint: rbf_ffn/experiments/20260404_105557_877306_standard_swiglu_qknorm_wnorm_d256/checkpoint_best.pt

# LoRA
lora_rank: 8
lora_alpha: 16.0
lora_target_modules: [q_proj, v_proj]

# Training
seq_len: 512
batch_size: 8
max_steps: 10000
lr: 0.0003
warmup_steps: 200
grad_clip: 1.0
seed: 42

# Evaluation / ISD
eval_every: 500
stride: 4
num_eval_examples: 200
prompt_len: 64
gen_len: 128

vocab_size: 50257
```

- [ ] **Step 2: Validate config loads cleanly**

```bash
python -c "
from idlm.config import load_config
cfg = load_config('idlm/configs/baseline.yaml')
print('ar_checkpoint:', cfg.ar_checkpoint)
print('lora_rank:', cfg.lora_rank)
print('max_steps:', cfg.max_steps)
"
```
Expected: prints the three values without error.

- [ ] **Step 3: Create idlm/README.md**

```markdown
# I-DLM: Introspective Diffusion Language Model

Small-scale reproduction of [I-DLM](https://arxiv.org/abs/2604.11035) on WikiText-103.

**Method:** Fine-tune a frozen `rbf_ffn` AR checkpoint using the I-DLM
introspective-consistency objective, then decode with Introspective Strided
Decoding (ISD).

**Best AR base:** `rbf_ffn` SwiGLU + QK-norm + weight-norm → 58.16 val PPL

## Training

```bash
python -m idlm.train --config idlm/configs/baseline.yaml
```

## ISD Evaluation

```bash
python -m idlm.generate \
    --config idlm/configs/baseline.yaml \
    --checkpoint idlm/experiments/<run>/checkpoint_best.pt \
    --output results.jsonl
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| α | Introspective acceptance rate — how often the base AR model endorses ISD proposals |
| PPL | Perplexity of generated continuations vs WikiText-103 reference |
| TPF/OH | Tokens per forward / overhead — efficiency ratio; >1 means ISD beats sequential AR |

## Config Reference

| Key | Default | Description |
|-----|---------|-------------|
| `ar_checkpoint` | — | Path to `rbf_ffn` checkpoint `.pt` file |
| `lora_rank` | 8 | LoRA adapter rank |
| `lora_alpha` | 16.0 | LoRA scaling factor |
| `lora_target_modules` | `[q_proj, v_proj]` | Attention modules to apply LoRA |
| `seq_len` | 512 | Sequence length (model sees 2×seq_len during training) |
| `batch_size` | 8 | Training batch size |
| `max_steps` | 10000 | Training steps |
| `lr` | 3e-4 | AdamW learning rate (LoRA params only) |
| `stride` | 4 | ISD stride — tokens proposed per forward pass |
| `num_eval_examples` | 200 | Number of test sequences for ISD eval |
| `prompt_len` | 64 | Prompt prefix length for ISD generation |
| `gen_len` | 128 | Generation length for ISD evaluation |

## Reference

*Introspective Diffusion Language Models*
Yu, Jian, Wang, Zhou et al. — Together AI / UIUC / UT Austin / Princeton / Stanford
arXiv:2604.11035, Apr 2026
```

- [ ] **Step 4: Update root README.md**

Add a row to the Projects table (after `flow_matching`):

```markdown
| [`idlm/`](#idlm) | Introspective Diffusion LM (I-DLM) — LoRA fine-tune AR checkpoint with masked diffusion objective + ISD decoding | WikiText-103 | Active |
```

And add an `## idlm` section at the bottom (before `## archive`):

```markdown
## idlm

Small-scale reproduction of [I-DLM](https://arxiv.org/abs/2604.11035): fine-tune a frozen `rbf_ffn` AR checkpoint with LoRA adapters using the introspective-consistency training objective, then evaluate with Introspective Strided Decoding.

Key metrics: introspective acceptance rate α, perplexity, and TPF/OH compute efficiency.

```bash
python -m idlm.train --config idlm/configs/baseline.yaml
```

See [`idlm/README.md`](idlm/README.md) for full config reference and ISD evaluation instructions.
```

- [ ] **Step 5: Run full test suite**

```bash
pytest idlm/tests/ rbf_ffn/tests/ -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add idlm/configs/baseline.yaml idlm/README.md README.md
git commit -m "feat(idlm): baseline config, README, and root README update"
```

---

## Task 9: Smoke Test End-to-End

- [ ] **Step 1: Verify the checkpoint path exists**

```bash
ls rbf_ffn/experiments/20260404_105557_877306_standard_swiglu_qknorm_wnorm_d256/checkpoint_best.pt
```
Expected: file exists. If not, find the correct wnorm experiment dir:
```bash
ls rbf_ffn/experiments/ | grep "standard_swiglu_qknorm_wnorm" | sort -r | head -3
```
Update `idlm/configs/baseline.yaml` accordingly.

- [ ] **Step 2: Run a 10-step smoke test**

Create a temporary config override and run 10 steps:

```bash
python -c "
from idlm.config import load_config
from idlm.train import train
from pathlib import Path
cfg = load_config('idlm/configs/baseline.yaml')
cfg.max_steps = 10
cfg.eval_every = 10
cfg.batch_size = 2
cfg.num_eval_examples = 2
cfg.gen_len = 16
cfg.prompt_len = 8
train(cfg, Path('idlm/configs/baseline.yaml'))
print('Smoke test passed.')
"
```
Expected: 10 steps complete, an eval runs, metrics printed, `Smoke test passed.`.

- [ ] **Step 3: Commit final state**

```bash
git add -u
git commit -m "test(idlm): verified 10-step smoke test passes end-to-end"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ New `idlm/` top-level project (Task 1)
- ✅ IDLMConfig with all fields from spec (Task 2)
- ✅ Fine-tunes frozen rbf_ffn checkpoint (Tasks 5, 6)
- ✅ LoRA on q_proj + v_proj, r=8, α=16 (Task 4)
- ✅ All-masked objective with 2L concatenation (Task 6)
- ✅ Auto-balanced loss λ = stop_gradient(L_mask / L_clean) (Task 6)
- ✅ ISD evaluation: α, PPL, TPF/OH on WikiText-103 continuations (Task 7)
- ✅ baseline.yaml pointing at weight_norm checkpoint (Task 8)
- ✅ Tests: shape, LoRA isolation, gradient isolation, loss terms, α bounds (Tasks 4, 5, 6, 7)

**Type consistency check:**
- `IDLMCausalLM.forward(tokens, use_lora_mask)` → used in `train.py:compute_loss` and `generate.py` ✅
- `LoRALinear.current_mask` → set in `IDLMCausalLM._set_mask` ✅
- `isd_acceptance_rate(model, sequence, cfg, device)` → called in `train.py:run_isd_eval` ✅
- `compute_tpf_oh(alpha, stride)` → called in `generate.py:evaluate_isd` ✅
- `load_config(path)` returns `IDLMConfig` → used in `train.py` and `generate.py` ✅
