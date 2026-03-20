# Flow Matching on CIFAR-100 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Rectified Flow with a DiT vector field network and classifier-free guidance on CIFAR-100.

**Architecture:** Straight-line flow paths (t=0=noise, t=1=data), MSE velocity loss, DiT backbone with adaLN-Zero conditioning on time + class label, CFG via label dropout during training and two-pass guided sampling at inference.

**Tech Stack:** Python 3.12+, PyTorch, torchvision (CIFAR-100), matplotlib, PyYAML. All files live in `flow_matching/`. Tests in `flow_matching/tests/`. Run tests with `pytest flow_matching/tests/`.

---

## File Map

| File | Responsibility |
|---|---|
| `flow_matching/__init__.py` | empty package marker |
| `flow_matching/config.py` | `FlowConfig` dataclass + `load_config` |
| `flow_matching/data.py` | `build_loaders` — CIFAR-100 DataLoaders |
| `flow_matching/model.py` | `timestep_embedding`, `make_2d_sincos_pos_embed`, `PatchEmbed`, `DiTBlock`, `DiT`, `build_optimizer_groups` |
| `flow_matching/train.py` | `train()` — step-based training loop, JSONL logging, plot, sampling, checkpointing |
| `flow_matching/sample.py` | `euler_sample()`, `save_sample_grid()`, CLI entry point |
| `flow_matching/configs/dit_cfg.yaml` | Default AdamW config |
| `flow_matching/configs/dit_muon.yaml` | Muon variant config |
| `flow_matching/configs/dit_small.yaml` | Small config for smoke tests |
| `flow_matching/tests/__init__.py` | empty |
| `flow_matching/tests/test_config.py` | Config dataclass + load_config tests |
| `flow_matching/tests/test_data.py` | DataLoader shape tests (mocked CIFAR-100) |
| `flow_matching/tests/test_model.py` | PatchEmbed, DiTBlock, DiT forward shape, optimizer groups |
| `flow_matching/tests/test_train.py` | Smoke training run (mocked CIFAR-100, stub Muon) |
| `flow_matching/tests/test_sample.py` | `euler_sample` shape, CFG two-pass |

---

## Task 1: Scaffold

**Files:**
- Create: `flow_matching/__init__.py`
- Create: `flow_matching/tests/__init__.py`
- Create: `flow_matching/configs/.gitkeep`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p flow_matching/tests flow_matching/configs flow_matching/experiments
touch flow_matching/__init__.py flow_matching/tests/__init__.py flow_matching/configs/.gitkeep
```

- [ ] **Step 2: Verify the module is importable**

```bash
python -c "import flow_matching; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add flow_matching/
git commit -m "feat(flow-matching): scaffold module structure"
```

---

## Task 2: Config

**Files:**
- Create: `flow_matching/tests/test_config.py`
- Create: `flow_matching/config.py`

- [ ] **Step 1: Write the failing tests**

Create `flow_matching/tests/test_config.py`:

```python
from __future__ import annotations
import pytest
from flow_matching.config import FlowConfig, load_config


def test_defaults():
    cfg = FlowConfig()
    assert cfg.data_root == "data/"
    assert cfg.seed == 42
    assert cfg.d_model == 384
    assert cfg.n_heads == 6
    assert cfg.n_layers == 12
    assert cfg.patch_size == 4
    assert cfg.mlp_ratio == pytest.approx(4.0)
    assert cfg.dropout == pytest.approx(0.0)
    assert cfg.p_uncond == pytest.approx(0.1)
    assert cfg.n_steps == 200_000
    assert cfg.batch_size == 128
    assert cfg.optimizer == "adamw"
    assert cfg.adamw_lr == pytest.approx(1e-4)
    assert cfg.muon_lr == pytest.approx(0.02)
    assert cfg.weight_decay == pytest.approx(0.0)
    assert cfg.warmup_ratio == pytest.approx(0.05)
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.log_every == 100
    assert cfg.sample_every == 5_000
    assert cfg.save_every == 10_000
    assert cfg.n_steps_euler == 100
    assert cfg.cfg_scale == pytest.approx(3.0)


def test_load_config_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("d_model: 64\nn_layers: 2\n")
    cfg = load_config(path)
    assert cfg.d_model == 64
    assert cfg.n_layers == 2
    assert cfg.n_steps == 200_000   # unspecified → default


def test_load_config_unknown_key_raises(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("unknown_key: 99\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)


def test_load_config_empty_yaml_returns_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("")
    cfg = load_config(path)
    assert cfg.d_model == 384


def test_invalid_optimizer_raises():
    with pytest.raises(ValueError, match="optimizer"):
        FlowConfig(optimizer="sgd")


def test_invalid_dropout_raises():
    with pytest.raises(ValueError, match="dropout"):
        FlowConfig(dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        FlowConfig(dropout=-0.1)


def test_invalid_p_uncond_raises():
    with pytest.raises(ValueError, match="p_uncond"):
        FlowConfig(p_uncond=0.0)
    with pytest.raises(ValueError, match="p_uncond"):
        FlowConfig(p_uncond=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest flow_matching/tests/test_config.py -v
```
Expected: `ModuleNotFoundError` — `flow_matching.config` doesn't exist yet.

- [ ] **Step 3: Implement `flow_matching/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import yaml


@dataclass
class FlowConfig:
    # Data
    data_root: str = "data/"
    seed: int = 42

    # Model
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 12
    patch_size: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Flow matching
    p_uncond: float = 0.1

    # Training
    n_steps: int = 200_000
    batch_size: int = 128
    optimizer: str = "adamw"       # adamw | muon
    adamw_lr: float = 1e-4
    muon_lr: float = 0.02          # only used when optimizer=muon
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 100
    sample_every: int = 5_000
    save_every: int = 10_000

    # Sampling (also used for in-training grids)
    n_steps_euler: int = 100
    cfg_scale: float = 3.0

    def __post_init__(self) -> None:
        valid_opts = {"adamw", "muon"}
        if self.optimizer not in valid_opts:
            raise ValueError(f"optimizer must be one of {valid_opts}, got {self.optimizer!r}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 < self.p_uncond < 1.0):
            raise ValueError(f"p_uncond must be in (0, 1), got {self.p_uncond}")


def load_config(path: str | Path) -> FlowConfig:
    """Load a FlowConfig from a YAML file.

    Unspecified fields take dataclass defaults. Unknown keys raise ValueError.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return FlowConfig()
    valid_fields = {f.name for f in fields(FlowConfig)}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return FlowConfig(**raw)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest flow_matching/tests/test_config.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add flow_matching/config.py flow_matching/tests/test_config.py
git commit -m "feat(flow-matching): add FlowConfig dataclass and load_config"
```

---

## Task 3: Data

**Files:**
- Create: `flow_matching/tests/test_data.py`
- Create: `flow_matching/data.py`

- [ ] **Step 1: Write the failing tests**

Create `flow_matching/tests/test_data.py`:

```python
from __future__ import annotations
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch

from flow_matching.config import FlowConfig


class _FakeCIFAR100:
    """Minimal CIFAR-100 stub — avoids network download in tests."""
    def __init__(self, *args, **kwargs):
        self.data = np.random.randint(0, 256, (50, 32, 32, 3), dtype=np.uint8)
        self.targets = list(range(50))
        self.transform = kwargs.get("transform")

    def __len__(self) -> int:
        return 50

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx] % 100


def test_build_loaders_batch_shape(tmp_path):
    from flow_matching.data import build_loaders

    cfg = FlowConfig(data_root=str(tmp_path), batch_size=4)
    with patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100):
        train_loader, val_loader = build_loaders(cfg, num_workers=0)

    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, 32, 32), f"Expected (4,3,32,32), got {x.shape}"
    assert x.dtype == torch.float32
    assert 0 <= int(y.min()) and int(y.max()) <= 99


def test_build_loaders_returns_two_loaders(tmp_path):
    from flow_matching.data import build_loaders
    from torch.utils.data import DataLoader

    cfg = FlowConfig(data_root=str(tmp_path), batch_size=4)
    with patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100):
        result = build_loaders(cfg, num_workers=0)

    assert len(result) == 2
    train_loader, val_loader = result
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest flow_matching/tests/test_data.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `flow_matching/data.py`**

```python
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from flow_matching.config import FlowConfig


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def build_loaders(cfg: FlowConfig, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """Build CIFAR-100 train and val DataLoaders.

    Train: RandomCrop + HorizontalFlip + Normalize
    Val:   Normalize only
    Returns (train_loader, val_loader).
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=True,  download=True, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest flow_matching/tests/test_data.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add flow_matching/data.py flow_matching/tests/test_data.py
git commit -m "feat(flow-matching): add CIFAR-100 data loaders"
```

---

## Task 4: Model — Embedding Helpers

**Files:**
- Create: `flow_matching/model.py` (initial, with helpers only)
- Create: `flow_matching/tests/test_model.py` (initial tests)

- [ ] **Step 1: Write the failing tests**

Create `flow_matching/tests/test_model.py`:

```python
from __future__ import annotations
import math
import torch
import pytest

from flow_matching.config import FlowConfig


# ── Embedding helpers ─────────────────────────────────────────────────────────

def test_timestep_embedding_shape():
    from flow_matching.model import timestep_embedding
    B, dim = 8, 64
    t = torch.rand(B)
    out = timestep_embedding(t, dim)
    assert out.shape == (B, dim), f"Expected ({B},{dim}), got {out.shape}"


def test_timestep_embedding_dim_must_be_even():
    from flow_matching.model import timestep_embedding
    with pytest.raises(AssertionError):
        timestep_embedding(torch.rand(4), dim=3)


def test_make_2d_sincos_pos_embed_shape():
    from flow_matching.model import make_2d_sincos_pos_embed
    d_model, grid_size = 64, 8
    emb = make_2d_sincos_pos_embed(d_model, grid_size)
    assert emb.shape == (1, grid_size * grid_size, d_model), \
        f"Expected (1,{grid_size**2},{d_model}), got {emb.shape}"


def test_make_2d_sincos_pos_embed_d_model_divisible_by_4():
    from flow_matching.model import make_2d_sincos_pos_embed
    with pytest.raises(AssertionError):
        make_2d_sincos_pos_embed(d_model=6, grid_size=8)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest flow_matching/tests/test_model.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `flow_matching/model.py` with embedding helpers**

```python
"""
DiT (Diffusion Transformer) vector field network for Rectified Flow.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from flow_matching.config import FlowConfig


# ── Embedding helpers ─────────────────────────────────────────────────────────

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding.

    t:   (B,) float in [0, 1]
    dim: embedding dimension (must be even)
    Returns: (B, dim)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t[:, None].float() * freqs[None]   # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


def make_2d_sincos_pos_embed(d_model: int, grid_size: int = 8) -> torch.Tensor:
    """Fixed 2D sinusoidal positional embeddings.

    d_model must be divisible by 4 (half per axis, each axis uses sin+cos).
    Returns: (1, grid_size^2, d_model) — intended to register as a buffer.
    """
    assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
    half = d_model // 2  # half the dims for each spatial axis

    omega = 1.0 / (
        10000 ** (torch.arange(half // 2).float() / max(half // 2 - 1, 1))
    )

    grid = torch.arange(grid_size).float()
    emb = torch.outer(grid, omega)                              # (grid_size, half//2)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (grid_size, half)

    # Expand for (h, w) pairs: each token gets [h_emb || w_emb]
    emb_h = emb.unsqueeze(1).expand(-1, grid_size, -1)  # (grid_size, grid_size, half)
    emb_w = emb.unsqueeze(0).expand(grid_size, -1, -1)  # (grid_size, grid_size, half)
    pos = torch.cat([emb_h, emb_w], dim=-1)             # (grid_size, grid_size, d_model)
    pos = pos.reshape(grid_size * grid_size, d_model)    # (N, d_model)
    return pos.unsqueeze(0)                              # (1, N, d_model)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest flow_matching/tests/test_model.py -v
```
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add flow_matching/model.py flow_matching/tests/test_model.py
git commit -m "feat(flow-matching): add sinusoidal embedding helpers"
```

---

## Task 5: Model — PatchEmbed

**Files:**
- Modify: `flow_matching/model.py` (add `PatchEmbed`)
- Modify: `flow_matching/tests/test_model.py` (add `PatchEmbed` tests)

- [ ] **Step 1: Add failing tests to `test_model.py`**

Append to `flow_matching/tests/test_model.py`:

```python
# ── PatchEmbed ────────────────────────────────────────────────────────────────

def test_patch_embed_output_shape():
    from flow_matching.model import PatchEmbed
    B, C, H, W = 2, 3, 32, 32
    patch_size, d_model = 4, 64
    model = PatchEmbed(patch_size=patch_size, d_model=d_model)
    x = torch.randn(B, C, H, W)
    out = model(x)
    n_patches = (H // patch_size) * (W // patch_size)  # 64
    assert out.shape == (B, n_patches, d_model), \
        f"Expected ({B},{n_patches},{d_model}), got {out.shape}"


def test_patch_embed_output_dtype():
    from flow_matching.model import PatchEmbed
    model = PatchEmbed(patch_size=4, d_model=64)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.dtype == torch.float32
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest flow_matching/tests/test_model.py::test_patch_embed_output_shape -v
```
Expected: `ImportError` — `PatchEmbed` not defined yet.

- [ ] **Step 3: Add `PatchEmbed` to `flow_matching/model.py`**

Append after the embedding helpers:

```python
# ── PatchEmbed ────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into patches and project to d_model.

    Input:  (B, 3, H, W)
    Output: (B, H//p * W//p, d_model)
    """

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)          # (B, d_model, H//p, W//p)
        B, C, H, W = x.shape
        x = x.flatten(2)          # (B, d_model, N)
        x = x.transpose(1, 2)     # (B, N, d_model)
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest flow_matching/tests/test_model.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add flow_matching/model.py flow_matching/tests/test_model.py
git commit -m "feat(flow-matching): add PatchEmbed"
```

---

## Task 6: Model — DiTBlock

**Files:**
- Modify: `flow_matching/model.py` (add `DiTBlock`)
- Modify: `flow_matching/tests/test_model.py` (add `DiTBlock` tests)

- [ ] **Step 1: Add failing tests to `test_model.py`**

Append to `flow_matching/tests/test_model.py`:

```python
# ── DiTBlock ──────────────────────────────────────────────────────────────────

def test_ditblock_output_shape():
    from flow_matching.model import DiTBlock
    B, N, d_model = 2, 64, 64
    block = DiTBlock(d_model=d_model, n_heads=2, mlp_ratio=4.0, dropout=0.0)
    x = torch.randn(B, N, d_model)
    c = torch.randn(B, d_model)
    out = block(x, c)
    assert out.shape == (B, N, d_model), f"Expected ({B},{N},{d_model}), got {out.shape}"


def test_ditblock_adaln_mlp_outputs_6d():
    from flow_matching.model import DiTBlock
    d_model = 64
    block = DiTBlock(d_model=d_model, n_heads=2, mlp_ratio=4.0, dropout=0.0)
    c = torch.randn(2, d_model)
    out = block.adaln_mlp(c)
    assert out.shape == (2, 6 * d_model), f"Expected (2,{6*d_model}), got {out.shape}"


def test_ditblock_adaln_zero_init():
    """At init, adaln_mlp final layer is zero → block acts as identity on random input."""
    from flow_matching.model import DiTBlock
    d_model = 64
    block = DiTBlock(d_model=d_model, n_heads=2, mlp_ratio=4.0, dropout=0.0)
    # Final linear of adaln_mlp should be zero-initialized
    final_layer = block.adaln_mlp[-1]
    assert torch.allclose(final_layer.weight, torch.zeros_like(final_layer.weight))
    assert torch.allclose(final_layer.bias,   torch.zeros_like(final_layer.bias))
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest flow_matching/tests/test_model.py::test_ditblock_output_shape -v
```
Expected: `ImportError` — `DiTBlock` not defined yet.

- [ ] **Step 3: Add `DiTBlock` to `flow_matching/model.py`**

Append after `PatchEmbed`:

```python
# ── DiTBlock ──────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning.

    Forward:
        x: (B, N, d_model) — patch tokens
        c: (B, d_model)    — conditioning signal (time + class)
    Returns: (B, N, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True, bias=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        ffn_hidden = int(mlp_ratio * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        # adaLN MLP: SiLU activation + single linear projecting c → 6 * d_model
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init the final projection (the "Zero" in adaLN-Zero)
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Compute 6 adaLN parameters; unsqueeze to broadcast over sequence dim
        s = self.adaln_mlp(c)  # (B, 6*d_model)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = s.chunk(6, dim=-1)
        shift_msa = shift_msa.unsqueeze(1)  # (B, 1, d_model)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa  = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp  = gate_mlp.unsqueeze(1)

        # Attention sublayer
        normed = (1 + scale_msa) * self.norm1(x) + shift_msa
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + gate_msa * attn_out

        # FFN sublayer
        normed = (1 + scale_mlp) * self.norm2(x) + shift_mlp
        x = x + gate_mlp * self.ffn(normed)

        return x
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest flow_matching/tests/test_model.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add flow_matching/model.py flow_matching/tests/test_model.py
git commit -m "feat(flow-matching): add DiTBlock with adaLN-Zero"
```

---

## Task 7: Model — DiT + build_optimizer_groups

**Files:**
- Modify: `flow_matching/model.py` (add `DiT`, `build_optimizer_groups`)
- Modify: `flow_matching/tests/test_model.py` (add `DiT` and optimizer group tests)

- [ ] **Step 1: Add failing tests to `test_model.py`**

Append to `flow_matching/tests/test_model.py`:

```python
# ── DiT ───────────────────────────────────────────────────────────────────────

def _tiny_cfg() -> FlowConfig:
    return FlowConfig(d_model=64, n_heads=2, n_layers=2, patch_size=4)


def test_dit_forward_shape():
    from flow_matching.model import DiT
    cfg = _tiny_cfg()
    model = DiT(cfg)
    B = 2
    x = torch.randn(B, 3, 32, 32)
    t = torch.rand(B)
    y = torch.randint(0, 100, (B,))
    out = model(x, t, y)
    assert out.shape == (B, 3, 32, 32), f"Expected ({B},3,32,32), got {out.shape}"


def test_dit_accepts_null_class_token():
    from flow_matching.model import DiT
    cfg = _tiny_cfg()
    model = DiT(cfg)
    B = 2
    x = torch.randn(B, 3, 32, 32)
    t = torch.rand(B)
    y = torch.full((B,), 100)   # null token (index 100)
    out = model(x, t, y)
    assert out.shape == (B, 3, 32, 32)


# ── build_optimizer_groups ────────────────────────────────────────────────────

def test_optimizer_groups_no_embedding_in_muon():
    from flow_matching.model import DiT, build_optimizer_groups
    model = DiT(_tiny_cfg())
    muon_params, adamw_params = build_optimizer_groups(model)

    # Collect names of muon params using id matching
    muon_ids = {id(p) for p in muon_params}
    for name, param in model.named_parameters():
        if id(param) in muon_ids:
            assert not name.startswith("time_embed."), \
                f"time_embed param {name!r} should not be in Muon group"
            assert not name.startswith("class_embed."), \
                f"class_embed param {name!r} should not be in Muon group"
            assert "adaln_mlp." not in name, \
                f"adaln_mlp param {name!r} should not be in Muon group"


def test_optimizer_groups_2d_matrices_in_muon():
    from flow_matching.model import DiT, build_optimizer_groups
    model = DiT(_tiny_cfg())
    muon_params, _ = build_optimizer_groups(model)

    # All muon params must be 2D matrices
    for p in muon_params:
        assert p.ndim == 2, f"Non-2D param in Muon group: shape {p.shape}"


def test_optimizer_groups_covers_all_params():
    from flow_matching.model import DiT, build_optimizer_groups
    model = DiT(_tiny_cfg())
    muon_params, adamw_params = build_optimizer_groups(model)

    all_ids = {id(p) for p in model.parameters()}
    covered = {id(p) for p in muon_params} | {id(p) for p in adamw_params}
    assert all_ids == covered, "Some params not assigned to any optimizer group"
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest flow_matching/tests/test_model.py::test_dit_forward_shape -v
```
Expected: `ImportError` — `DiT` not defined yet.

- [ ] **Step 3: Add `DiT` and `build_optimizer_groups` to `flow_matching/model.py`**

Append to `flow_matching/model.py`:

```python
# ── DiT ───────────────────────────────────────────────────────────────────────

class DiT(nn.Module):
    """Diffusion Transformer vector field network.

    Forward(x, t, y) -> v
        x: (B, 3, 32, 32) — noisy image at time t
        t: (B,)            — time values in [0, 1]
        y: (B,)            — class indices in [0, 100]; 100 = null CFG token
    Returns: (B, 3, 32, 32) — predicted velocity field
    """

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.d_model    = cfg.d_model
        self.patch_size = cfg.patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.d_model)

        # Fixed 2D sinusoidal positional embedding (buffer, not a parameter)
        grid_size = 32 // cfg.patch_size  # e.g. 8 for patch_size=4
        pos_embed = make_2d_sincos_pos_embed(cfg.d_model, grid_size)
        self.register_buffer("pos_embed", pos_embed)

        # Time & class conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.class_embed = nn.Embedding(101, cfg.d_model)  # 100 classes + null

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.patch_size ** 2 * 3)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Patchify + positional embedding
        x = self.patch_embed(x)       # (B, 64, d_model)
        x = x + self.pos_embed        # broadcast (1, 64, d_model) → (B, 64, d_model)

        # 2. Conditioning signal
        t_sin = timestep_embedding(t, self.d_model)  # (B, d_model)
        c = self.time_embed(t_sin) + self.class_embed(y)  # (B, d_model)

        # 3. Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # 4. Output projection
        x = self.norm(x)   # (B, 64, d_model)
        x = self.head(x)   # (B, 64, patch_size^2 * 3)

        # 5. Unpatchify: (B, 64, 48) → (B, 3, 32, 32)
        p  = self.patch_size
        h  = w = 32 // p   # 8
        x  = x.reshape(B, h, w, p, p, 3)
        x  = x.permute(0, 5, 1, 3, 2, 4).contiguous().reshape(B, 3, h * p, w * p)

        return x


# ── Optimizer groups ──────────────────────────────────────────────────────────

def build_optimizer_groups(
    model: DiT,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split model parameters into Muon and AdamW groups.

    Rules (first match wins):
      1. name starts with "time_embed." or "class_embed." or
         contains "adaln_mlp."                              → AdamW
      2. param.ndim == 2                                    → Muon
      3. else                                               → AdamW

    Returns (muon_params, adamw_params).
    """
    seen:  set[int]          = set()
    muon:  list[torch.Tensor] = []
    adamw: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if (
            name.startswith("time_embed.")
            or name.startswith("class_embed.")
            or "adaln_mlp." in name
        ):
            adamw.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)

    return muon, adamw
```

- [ ] **Step 4: Run all model tests**

```bash
pytest flow_matching/tests/test_model.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add flow_matching/model.py flow_matching/tests/test_model.py
git commit -m "feat(flow-matching): add DiT model and build_optimizer_groups"
```

---

## Task 8: Euler Sampler

**Files:**
- Create: `flow_matching/tests/test_sample.py`
- Create: `flow_matching/sample.py`

- [ ] **Step 1: Write the failing tests**

Create `flow_matching/tests/test_sample.py`:

```python
from __future__ import annotations
import torch
import pytest

from flow_matching.config import FlowConfig


def _tiny_model():
    from flow_matching.model import DiT
    cfg = FlowConfig(d_model=64, n_heads=2, n_layers=2, patch_size=4)
    return DiT(cfg).eval()


def test_euler_sample_output_shape():
    from flow_matching.sample import euler_sample
    model  = _tiny_model()
    B      = 4
    y      = torch.randint(0, 100, (B,))
    device = torch.device("cpu")

    with torch.no_grad():
        out = euler_sample(model, y, cfg_scale=1.0, n_steps=2, device=device)

    assert out.shape == (B, 3, 32, 32), f"Expected ({B},3,32,32), got {out.shape}"


def test_euler_sample_cfg_scale_zero_equals_uncond():
    """cfg_scale=0 → guided velocity = uncond velocity."""
    from flow_matching.sample import euler_sample
    model  = _tiny_model()
    B      = 2
    y      = torch.zeros(B, dtype=torch.long)
    device = torch.device("cpu")

    torch.manual_seed(0)
    with torch.no_grad():
        out_cfg0 = euler_sample(model, y, cfg_scale=0.0, n_steps=2, device=device)

    torch.manual_seed(0)
    with torch.no_grad():
        # With cfg_scale=0, guided = uncond; equivalent to passing null token
        y_null = torch.full_like(y, 100)
        out_uncond = euler_sample(model, y_null, cfg_scale=0.0, n_steps=2, device=device)

    assert torch.allclose(out_cfg0, out_uncond, atol=1e-5)


def test_euler_sample_null_class_accepted():
    from flow_matching.sample import euler_sample
    model  = _tiny_model()
    y      = torch.full((2,), 100)  # null tokens
    with torch.no_grad():
        out = euler_sample(model, y, cfg_scale=3.0, n_steps=2, device=torch.device("cpu"))
    assert out.shape == (2, 3, 32, 32)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest flow_matching/tests/test_sample.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `flow_matching/sample.py`**

```python
"""
Euler ODE sampler and sample grid utilities for Rectified Flow.

CLI usage:
    python -m flow_matching.sample \\
        --config flow_matching/configs/dit_cfg.yaml \\
        --checkpoint <exp_dir>/ckpt.pt \\
        --cfg_scale 3.0 \\
        --out samples.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torchvision

import matplotlib
matplotlib.use("Agg")

from flow_matching.config import FlowConfig, load_config
from flow_matching.data import CIFAR100_MEAN, CIFAR100_STD


def euler_sample(
    model:     "DiT",
    y:         torch.Tensor,
    cfg_scale: float,
    n_steps:   int,
    device:    torch.device,
) -> torch.Tensor:
    """Euler ODE integrator from t=0 (noise) to t=1 (data) with CFG.

    y:         (B,) class indices in [0, 99]; pass 100 for unconditional
    cfg_scale: guidance scale (0 = no guidance, 1 = conditional only)
    n_steps:   number of Euler steps
    Returns:   (B, 3, 32, 32) generated images (normalised, not pixel-space)
    """
    B        = y.shape[0]
    y_null   = torch.full_like(y, 100)  # null CFG token
    x        = torch.randn(B, 3, 32, 32, device=device)
    dt       = 1.0 / n_steps

    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            t       = i / n_steps
            t_batch = torch.full((B,), t, dtype=torch.float32, device=device)

            v_cond   = model(x, t_batch, y.to(device))
            v_uncond = model(x, t_batch, y_null.to(device))
            v        = v_uncond + cfg_scale * (v_cond - v_uncond)

            x = x + v * dt

    return x


def save_sample_grid(
    images: torch.Tensor,
    path:   Path,
    mean:   tuple[float, ...] = CIFAR100_MEAN,
    std:    tuple[float, ...] = CIFAR100_STD,
    nrow:   int = 10,
) -> None:
    """Save a (N, 3, 32, 32) tensor as a PNG grid after denormalising."""
    import matplotlib.pyplot as plt

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)
    imgs   = images.cpu() * std_t + mean_t
    imgs   = imgs.clamp(0, 1)

    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2)  # (3, H, W)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a trained Rectified Flow model")
    p.add_argument("--config",       required=True,  help="Path to YAML config")
    p.add_argument("--checkpoint",   required=True,  help="Path to ckpt.pt")
    p.add_argument("--cfg_scale",    type=float, default=None, help="CFG scale override")
    p.add_argument("--n_steps_euler",type=int,   default=None, help="Euler steps override")
    p.add_argument("--out",          default="samples.png",    help="Output PNG path")
    return p.parse_args()


if __name__ == "__main__":
    from flow_matching.model import DiT

    args    = _parse_args()
    cfg     = load_config(args.config)
    if args.cfg_scale    is not None: cfg.cfg_scale    = args.cfg_scale
    if args.n_steps_euler is not None: cfg.n_steps_euler = args.n_steps_euler

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = DiT(cfg).to(device)
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y       = torch.arange(100, device=device)  # one sample per class
    samples = euler_sample(model, y, cfg.cfg_scale, cfg.n_steps_euler, device)
    save_sample_grid(samples, Path(args.out))
    print(f"Saved sample grid → {args.out}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest flow_matching/tests/test_sample.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add flow_matching/sample.py flow_matching/tests/test_sample.py
git commit -m "feat(flow-matching): add Euler sampler with CFG"
```

---

## Task 9: Training Loop

**Files:**
- Create: `flow_matching/tests/test_train.py`
- Create: `flow_matching/train.py`

- [ ] **Step 1: Write the failing tests**

Create `flow_matching/tests/test_train.py`:

```python
"""
Smoke tests for the flow matching training loop.
Uses tiny configs; patches CIFAR-100 and Muon to avoid downloads and GPU deps.
"""
from __future__ import annotations
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch.optim import AdamW

from flow_matching.config import FlowConfig


class _FakeCIFAR100:
    def __init__(self, *args, **kwargs):
        self.data    = np.random.randint(0, 256, (200, 32, 32, 3), dtype=np.uint8)
        self.targets = list(range(200))
        self.transform = kwargs.get("transform")
    def __len__(self):
        return 200
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx] % 100


class _MuonStub(AdamW):
    """AdamW stub that accepts Muon's momentum kwarg."""
    def __init__(self, params, lr=0.02, momentum=0.95, **kwargs):
        super().__init__(params, lr=lr)


def _tiny_cfg(**kwargs) -> FlowConfig:
    defaults = dict(
        d_model=64, n_layers=2, n_heads=2,
        batch_size=4, n_steps=6,
        log_every=2, sample_every=100, save_every=6,
        n_steps_euler=2, warmup_ratio=0.0,
    )
    defaults.update(kwargs)
    return FlowConfig(**defaults)


def _run(cfg: FlowConfig, tmp_path, optimizer: str = "adamw"):
    cfg.optimizer = optimizer
    cfg.data_root = str(tmp_path)
    config_path   = tmp_path / "cfg.yaml"
    config_path.write_text("")
    with (
        patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100),
        patch("flow_matching.train.Muon", _MuonStub),
    ):
        from flow_matching.train import train
        return train(cfg, config_path=config_path)


def test_train_creates_metrics_jsonl(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path)
    assert (exp_dir / "metrics.jsonl").exists()


def test_train_metrics_has_correct_fields(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path)
    rows = [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 3   # 6 steps / log_every=2
    for row in rows:
        assert "step"       in row
        assert "train_loss" in row
        assert "lr"         in row


def test_train_creates_plot_png(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path)
    assert (exp_dir / "plot.png").exists()


def test_train_creates_checkpoint(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path)
    ckpt_path = exp_dir / "ckpt.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model" in ckpt
    assert ckpt["step"] == 6


def test_train_n_steps_override(tmp_path):
    cfg = _tiny_cfg(n_steps=20)
    cfg.data_root  = str(tmp_path)
    config_path    = tmp_path / "cfg.yaml"
    config_path.write_text("")
    with (
        patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100),
        patch("flow_matching.train.Muon", _MuonStub),
    ):
        from flow_matching.train import train
        exp_dir = train(cfg, config_path=config_path, n_steps=6)
    rows = [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert rows[-1]["step"] == 6


def test_train_config_yaml_copied(tmp_path):
    cfg = _tiny_cfg()
    cfg.data_root = str(tmp_path)
    config_path   = tmp_path / "cfg.yaml"
    config_path.write_text("d_model: 64\n")
    with (
        patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100),
        patch("flow_matching.train.Muon", _MuonStub),
    ):
        from flow_matching.train import train
        exp_dir = train(cfg, config_path=config_path)
    assert (exp_dir / "config.yaml").read_text() == "d_model: 64\n"


def test_train_muon_mode(tmp_path):
    exp_dir = _run(_tiny_cfg(), tmp_path, optimizer="muon")
    assert (exp_dir / "metrics.jsonl").exists()
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest flow_matching/tests/test_train.py -v
```
Expected: `ModuleNotFoundError` — `flow_matching.train` doesn't exist yet.

- [ ] **Step 3: Implement `flow_matching/train.py`**

```python
"""
Training entry point for flow matching on CIFAR-100.

Usage:
    python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml
    python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml --n_steps 50000
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.optim import Muon
except ImportError:
    Muon = None  # type: ignore[assignment,misc]

import matplotlib
matplotlib.use("Agg")

from flow_matching.config import FlowConfig, load_config
from flow_matching.data import build_loaders
from flow_matching.model import DiT, build_optimizer_groups
from flow_matching.sample import euler_sample, save_sample_grid


def _cycle(loader):
    while True:
        yield from loader


def get_experiment_dir(cfg: FlowConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name  = f"{stamp}_{cfg.optimizer}"
    path  = Path(__file__).parent / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _save_plot(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    steps  = [r["step"]       for r in rows]
    losses = [r["train_loss"] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(steps, losses, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss (MSE)")
    ax.set_title("Rectified Flow on CIFAR-100")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(
    cfg:         FlowConfig,
    config_path: Path,
    n_steps:     int | None = None,
) -> Path:
    if n_steps is not None:
        cfg.n_steps = n_steps

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir      = get_experiment_dir(cfg)
    shutil.copy(config_path, exp_dir / "config.yaml")
    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, _ = build_loaders(cfg, num_workers=0)
    data_iter       = _cycle(train_loader)

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = DiT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    warmup_steps = int(cfg.warmup_ratio * cfg.n_steps)
    lr_fn        = make_lr_lambda(warmup_steps, cfg.n_steps)

    if cfg.optimizer == "adamw":
        opt      = AdamW(
            model.parameters(),
            lr=cfg.adamw_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
        )
        optimizers = [opt]
        schedulers = [LambdaLR(opt, lr_fn)]
        adamw_opt  = opt
    else:
        if Muon is None:
            raise ImportError(
                "Muon optimizer is not available. "
                "Install a PyTorch build that includes it, or use optimizer: adamw."
            )
        muon_params, adamw_params = build_optimizer_groups(model)
        muon_opt  = Muon(muon_params, lr=cfg.muon_lr, momentum=0.95)
        adamw_opt = AdamW(
            adamw_params,
            lr=cfg.adamw_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
        )
        optimizers = [muon_opt, adamw_opt]
        schedulers = [LambdaLR(muon_opt, lr_fn), LambdaLR(adamw_opt, lr_fn)]

    # ── Training loop ─────────────────────────────────────────────────────────
    rows: list[dict] = []

    for step in range(1, cfg.n_steps + 1):
        model.train()
        x1, y = next(data_iter)
        x1, y = x1.to(device), y.to(device)
        B     = x1.shape[0]

        x0     = torch.randn_like(x1)
        t      = torch.rand(B, device=device)
        t_view = t.view(B, 1, 1, 1)

        # CFG label dropout
        mask   = torch.rand(B, device=device) < cfg.p_uncond
        y_cond = y.clone()
        y_cond[mask] = 100  # null token

        # Rectified Flow interpolation
        xt       = (1 - t_view) * x0 + t_view * x1
        v_target = x1 - x0

        # Forward + loss
        v_pred = model(xt, t, y_cond)
        loss   = F.mse_loss(v_pred, v_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()

        # Logging
        if step % cfg.log_every == 0:
            current_lr = adamw_opt.param_groups[0]["lr"]
            row = {"step": step, "train_loss": loss.item(), "lr": current_lr}
            rows.append(row)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"step {step:>7}  "
                f"train_loss={row['train_loss']:.4f}  "
                f"lr={row['lr']:.2e}"
            )

        # In-training samples
        if step % cfg.sample_every == 0:
            model.eval()
            with torch.no_grad():
                y_sample = torch.arange(100, device=device)
                samples  = euler_sample(
                    model, y_sample, cfg.cfg_scale, cfg.n_steps_euler, device,
                )
            save_sample_grid(samples, exp_dir / f"samples_step_{step}.png")
            model.train()

        # Checkpoint
        if step % cfg.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step},
                exp_dir / "ckpt.pt",
            )

    _save_plot(rows, exp_dir / "plot.png")
    print(f"Done. Metrics → {metrics_path}")
    return exp_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Rectified Flow on CIFAR-100")
    p.add_argument("--config",  required=True, help="Path to YAML config")
    p.add_argument("--n_steps", type=int, default=None, help="Override n_steps")
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    cfg_path = Path(args.config)
    cfg      = load_config(cfg_path)
    train(cfg, config_path=cfg_path, n_steps=args.n_steps)
```

- [ ] **Step 4: Run all tests**

```bash
pytest flow_matching/tests/test_train.py -v
```
Expected: all PASS

- [ ] **Step 5: Run the full test suite**

```bash
pytest flow_matching/tests/ -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add flow_matching/train.py flow_matching/tests/test_train.py
git commit -m "feat(flow-matching): add training loop with JSONL logging and checkpointing"
```

---

## Task 10: YAML Configs

**Files:**
- Create: `flow_matching/configs/dit_cfg.yaml`
- Create: `flow_matching/configs/dit_muon.yaml`
- Create: `flow_matching/configs/dit_small.yaml`

- [ ] **Step 1: Create `flow_matching/configs/dit_cfg.yaml`**

```yaml
# Default DiT config — AdamW optimizer, full model
data_root: data/
seed: 42

d_model: 384
n_heads: 6
n_layers: 12
patch_size: 4
mlp_ratio: 4.0
dropout: 0.0

p_uncond: 0.1

n_steps: 200000
batch_size: 128
optimizer: adamw
adamw_lr: 1.0e-4
weight_decay: 0.0
warmup_ratio: 0.05
grad_clip: 1.0
log_every: 100
sample_every: 5000
save_every: 10000

n_steps_euler: 100
cfg_scale: 3.0
```

- [ ] **Step 2: Create `flow_matching/configs/dit_muon.yaml`**

```yaml
# DiT Muon variant — Muon for 2D weight matrices, AdamW for rest
data_root: data/
seed: 42

d_model: 384
n_heads: 6
n_layers: 12
patch_size: 4
mlp_ratio: 4.0
dropout: 0.0

p_uncond: 0.1

n_steps: 200000
batch_size: 128
optimizer: muon
adamw_lr: 1.0e-4
muon_lr: 0.02
weight_decay: 0.0
warmup_ratio: 0.05
grad_clip: 1.0
log_every: 100
sample_every: 5000
save_every: 10000

n_steps_euler: 100
cfg_scale: 3.0
```

- [ ] **Step 3: Create `flow_matching/configs/dit_small.yaml`**

```yaml
# Small DiT config — reduced model for rapid iteration / smoke tests
data_root: data/
seed: 42

d_model: 192
n_heads: 3
n_layers: 6
patch_size: 4
mlp_ratio: 4.0
dropout: 0.0

p_uncond: 0.1

n_steps: 1000
batch_size: 32
optimizer: adamw
adamw_lr: 1.0e-4
weight_decay: 0.0
warmup_ratio: 0.05
grad_clip: 1.0
log_every: 10
sample_every: 500
save_every: 500

n_steps_euler: 10
cfg_scale: 3.0
```

- [ ] **Step 4: Verify configs load cleanly**

```bash
python -c "
from flow_matching.config import load_config
for p in ['flow_matching/configs/dit_cfg.yaml',
          'flow_matching/configs/dit_muon.yaml',
          'flow_matching/configs/dit_small.yaml']:
    cfg = load_config(p)
    print(f'{p}: optimizer={cfg.optimizer}, d_model={cfg.d_model}')
"
```
Expected output:
```
flow_matching/configs/dit_cfg.yaml: optimizer=adamw, d_model=384
flow_matching/configs/dit_muon.yaml: optimizer=muon, d_model=384
flow_matching/configs/dit_small.yaml: optimizer=adamw, d_model=192
```

- [ ] **Step 5: Commit**

```bash
git add flow_matching/configs/
git commit -m "feat(flow-matching): add YAML configs (default, muon, small)"
```

---

## Final Verification

- [ ] **Run the full test suite one last time**

```bash
pytest flow_matching/tests/ -v
```
Expected: all PASS, no warnings about missing modules.

- [ ] **Verify module is importable**

```bash
python -c "
from flow_matching.config import FlowConfig, load_config
from flow_matching.data import build_loaders
from flow_matching.model import DiT, build_optimizer_groups
from flow_matching.train import train
from flow_matching.sample import euler_sample, save_sample_grid
print('All imports OK')
"
```
Expected: `All imports OK`
