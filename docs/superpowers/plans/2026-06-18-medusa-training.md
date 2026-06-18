# Medusa-1 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `medusa/` module that trains K Medusa-1 decoding heads on Gemma 4 E2B's last hidden state using ShareGPT52K, with a two-phase cache-then-train pipeline.

**Architecture:** Phase 1 (`cache.py`) runs Gemma 4 E2B on ShareGPT52K conversations, extracts per-position last hidden states, int8-quantizes them per-vector, and writes sharded `.pt` files. Phase 2 (`train.py`) loads those shards and trains K single-layer residual MLP heads with distance-weighted cross-entropy loss (Medusa-1). No backbone fine-tuning.

**Tech Stack:** PyTorch, HuggingFace `transformers` + `datasets`, tqdm, PyYAML, pytest.

## Global Constraints

- `medusa/` is entirely standalone — no imports from `mtp_draft/`.
- Medusa-1 only: backbone weights are never updated.
- `MedusaModel.forward` signature: `(h: Tensor[B, d_model]) -> Tensor[B, n_heads, vocab]`.
- Shard format: list of dicts with keys `hidden_int8` `(n_pos, d_model) int8`, `scale` `(n_pos,) float32`, `targets` `(n_pos, n_heads) int64` (`-100` for padding).
- Per-vector int8 quantization: one scale scalar per hidden vector.
- `run_in_background` all teacher forward passes.

---

## File Map

| File | Responsibility |
|---|---|
| `medusa/config.py` | `MedusaConfig` dataclass + `load_config` |
| `medusa/__init__.py` | empty package marker |
| `medusa/models/__init__.py` | empty package marker |
| `medusa/models/medusa_model.py` | `MedusaHead`, `MedusaModel` |
| `medusa/data.py` | `MedusaDataset`, `get_dataloaders` |
| `medusa/cache.py` | Phase 1: extract + quantize last hidden states |
| `medusa/train.py` | Phase 2: Medusa-1 training loop |
| `medusa/configs/default.yaml` | Default hyperparameters |
| `medusa/tests/__init__.py` | empty package marker |
| `medusa/tests/test_medusa_model.py` | model unit tests |
| `medusa/tests/test_data.py` | dataset unit tests |
| `medusa/tests/test_cache.py` | cache quantization unit tests |
| `medusa/tests/test_train.py` | training smoke test |

---

## Task 1: Config + Scaffold

**Files:**
- Create: `medusa/__init__.py`
- Create: `medusa/models/__init__.py`
- Create: `medusa/tests/__init__.py`
- Create: `medusa/config.py`
- Create: `medusa/configs/default.yaml`

**Interfaces:**
- Produces: `MedusaConfig` dataclass; `load_config(path: str) -> MedusaConfig`

- [ ] **Step 1: Create package markers**

```bash
mkdir -p medusa/models medusa/configs medusa/tests medusa/checkpoints
touch medusa/__init__.py medusa/models/__init__.py medusa/tests/__init__.py
```

- [ ] **Step 2: Write `medusa/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class MedusaConfig:
    # Model
    n_heads: int = 5
    d_model: int = 1536

    # Loss
    lambda_decay: float = 0.8

    # Teacher / data
    teacher_model_id: str = "google/gemma-4-e2b-it"
    dataset_id: str = "RyokoAI/ShareGPT52K"
    max_answer_len: int = 256
    max_seq_len: int = 768

    # Cache
    cache_dir: str = "medusa/cache"
    cache_shard_size: int = 5000
    val_split: float = 0.05

    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 32
    n_epochs: int = 3
    warmup_steps: int = 200
    seed: int = 42


def load_config(path: str) -> MedusaConfig:
    data = yaml.safe_load(Path(path).read_text())
    return MedusaConfig(**data)
```

- [ ] **Step 3: Write `medusa/configs/default.yaml`**

```yaml
n_heads: 5
d_model: 1536
lambda_decay: 0.8
teacher_model_id: "google/gemma-4-e2b-it"
dataset_id: "RyokoAI/ShareGPT52K"
max_answer_len: 256
max_seq_len: 768
cache_dir: "medusa/cache"
cache_shard_size: 5000
val_split: 0.05
lr: 3.0e-4
weight_decay: 0.1
grad_clip: 1.0
batch_size: 32
n_epochs: 3
warmup_steps: 200
seed: 42
```

- [ ] **Step 4: Verify config round-trips**

```bash
python -c "
from medusa.config import load_config
cfg = load_config('medusa/configs/default.yaml')
assert cfg.n_heads == 5
assert cfg.d_model == 1536
print('OK', cfg)
"
```

Expected: prints `OK MedusaConfig(n_heads=5, ...)` with no errors.

- [ ] **Step 5: Commit**

```bash
git add medusa/ 
git commit -m "feat(medusa): scaffold — config, package structure, default yaml"
```

---

## Task 2: MedusaModel

**Files:**
- Create: `medusa/models/medusa_model.py`
- Create: `medusa/tests/test_medusa_model.py`

**Interfaces:**
- Consumes: `MedusaConfig` from Task 1
- Produces:
  - `MedusaHead(d_model: int, lm_head_weight: Tensor[vocab, d_model]) -> Module`
    - `forward(h: Tensor[B, d_model]) -> Tensor[B, vocab]`
  - `MedusaModel(cfg: MedusaConfig, lm_head_weight: Tensor[vocab, d_model]) -> Module`
    - `forward(h: Tensor[B, d_model]) -> Tensor[B, n_heads, vocab]`

- [ ] **Step 1: Write failing tests**

```python
# medusa/tests/test_medusa_model.py
from __future__ import annotations
import torch
import pytest
from medusa.config import MedusaConfig
from medusa.models.medusa_model import MedusaHead, MedusaModel


VOCAB = 64
D_MODEL = 32


@pytest.fixture
def lm_weight():
    return torch.randn(VOCAB, D_MODEL)


@pytest.fixture
def cfg():
    return MedusaConfig(n_heads=3, d_model=D_MODEL)


def test_head_output_shape(lm_weight):
    head = MedusaHead(D_MODEL, lm_weight)
    h = torch.randn(4, D_MODEL)
    out = head(h)
    assert out.shape == (4, VOCAB)


def test_head_w1_init_zero(lm_weight):
    head = MedusaHead(D_MODEL, lm_weight)
    assert head.W1.weight.abs().max().item() == 0.0


def test_head_w2_init_from_lm_head(lm_weight):
    head = MedusaHead(D_MODEL, lm_weight)
    assert torch.allclose(head.W2.weight, lm_weight.float())


def test_model_output_shape(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    h = torch.randn(4, D_MODEL)
    out = model(h)
    assert out.shape == (4, cfg.n_heads, VOCAB)


def test_model_head_count(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    assert len(model.heads) == cfg.n_heads


def test_model_trainable_params(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Each head: W1 (d_model*d_model) + W2 (vocab*d_model)
    expected = cfg.n_heads * (D_MODEL * D_MODEL + VOCAB * D_MODEL)
    assert n == expected
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest medusa/tests/test_medusa_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'medusa.models.medusa_model'`

- [ ] **Step 3: Write `medusa/models/medusa_model.py`**

```python
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from medusa.config import MedusaConfig


class MedusaHead(nn.Module):
    """Single Medusa decoding head.

    Implements: logits = W2 · SiLU(W1 · h + h)
    W1 init: zero  (head starts as LM head copy, learns residual correction)
    W2 init: clone of frozen teacher LM head weight
    """

    def __init__(self, d_model: int, lm_head_weight: torch.Tensor) -> None:
        super().__init__()
        vocab_size = lm_head_weight.shape[0]
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.W1.weight)
        self.W2 = nn.Linear(d_model, vocab_size, bias=False)
        self.W2.weight = nn.Parameter(lm_head_weight.clone().float())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d_model)
        return self.W2(F.silu(self.W1(h) + h))  # (B, vocab)


class MedusaModel(nn.Module):
    """K Medusa-1 decoding heads sharing a common input hidden state."""

    def __init__(self, cfg: MedusaConfig, lm_head_weight: torch.Tensor) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            MedusaHead(cfg.d_model, lm_head_weight)
            for _ in range(cfg.n_heads)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d_model)
        return torch.stack([head(h) for head in self.heads], dim=1)  # (B, n_heads, vocab)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest medusa/tests/test_medusa_model.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add medusa/models/medusa_model.py medusa/tests/test_medusa_model.py
git commit -m "feat(medusa): MedusaHead and MedusaModel — K residual MLP heads"
```

---

## Task 3: MedusaDataset

**Files:**
- Create: `medusa/data.py`
- Create: `medusa/tests/test_data.py`

**Interfaces:**
- Consumes: `MedusaConfig` from Task 1
- Produces:
  - `MedusaDataset(shard_paths: list[Path], cfg: MedusaConfig) -> Dataset`
    - `__getitem__` returns `(hidden: Tensor[d_model] float32, targets: Tensor[n_heads] int64)`
  - `get_dataloaders(cfg: MedusaConfig) -> tuple[DataLoader, DataLoader]`

- [ ] **Step 1: Write failing tests**

```python
# medusa/tests/test_data.py
from __future__ import annotations
import tempfile
from pathlib import Path
import torch
import pytest
from medusa.config import MedusaConfig
from medusa.data import MedusaDataset, get_dataloaders

D_MODEL = 32
N_HEADS = 3
N_POS = 5


@pytest.fixture
def cfg(tmp_path):
    return MedusaConfig(
        n_heads=N_HEADS,
        d_model=D_MODEL,
        batch_size=2,
        val_split=0.5,
        cache_dir=str(tmp_path),
        seed=42,
    )


def _make_shard(path: Path, n_entries: int = 4) -> None:
    shard = []
    for _ in range(n_entries):
        h = torch.randint(-127, 128, (N_POS, D_MODEL), dtype=torch.int8)
        scale = torch.rand(N_POS) + 0.01
        targets = torch.randint(0, 100, (N_POS, N_HEADS), dtype=torch.long)
        shard.append({"hidden_int8": h, "scale": scale, "targets": targets})
    torch.save(shard, path)


def test_dataset_len(cfg, tmp_path):
    _make_shard(tmp_path / "train_shard_0000.pt", n_entries=3)
    ds = MedusaDataset([tmp_path / "train_shard_0000.pt"], cfg)
    assert len(ds) == 3 * N_POS


def test_dataset_item_shapes(cfg, tmp_path):
    _make_shard(tmp_path / "train_shard_0000.pt")
    ds = MedusaDataset([tmp_path / "train_shard_0000.pt"], cfg)
    h, targets = ds[0]
    assert h.shape == (D_MODEL,)
    assert targets.shape == (N_HEADS,)


def test_dataset_dequantize(cfg, tmp_path):
    shard = [{
        "hidden_int8": torch.full((1, D_MODEL), 63, dtype=torch.int8),
        "scale": torch.tensor([2.0]),
        "targets": torch.zeros(1, N_HEADS, dtype=torch.long),
    }]
    path = tmp_path / "train_shard_0000.pt"
    torch.save(shard, path)
    ds = MedusaDataset([path], cfg)
    h, _ = ds[0]
    assert h.dtype == torch.float32
    assert torch.allclose(h, torch.full((D_MODEL,), 126.0))


def test_dataset_targets_dtype(cfg, tmp_path):
    _make_shard(tmp_path / "train_shard_0000.pt")
    ds = MedusaDataset([tmp_path / "train_shard_0000.pt"], cfg)
    _, targets = ds[0]
    assert targets.dtype == torch.long


def test_get_dataloaders(cfg, tmp_path):
    _make_shard(tmp_path / "train_shard_0000.pt", n_entries=4)
    _make_shard(tmp_path / "validation_shard_0000.pt", n_entries=2)
    train_dl, val_dl = get_dataloaders(cfg)
    h, t = next(iter(train_dl))
    assert h.shape == (2, D_MODEL)
    assert t.shape == (2, N_HEADS)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest medusa/tests/test_data.py -v
```

Expected: `ModuleNotFoundError: No module named 'medusa.data'`

- [ ] **Step 3: Write `medusa/data.py`**

```python
from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from medusa.config import MedusaConfig


class MedusaDataset(Dataset):
    """Dataset over pre-cached Medusa hidden-state shards.

    Shard format (list of dicts):
        hidden_int8 : Tensor(n_pos, d_model) int8
        scale       : Tensor(n_pos,) float32  — per-vector scale
        targets     : Tensor(n_pos, n_heads) int64  — -100 for padding

    Returns per item:
        hidden  : Tensor(d_model,) float32 — dequantized
        targets : Tensor(n_heads,) int64
    """

    def __init__(self, shard_paths: list[Path], cfg: MedusaConfig) -> None:
        self.items: list[tuple[dict, int]] = []
        for path in shard_paths:
            shard: list[dict] = torch.load(path, weights_only=False)
            for entry in shard:
                n_pos = entry["hidden_int8"].shape[0]
                for i in range(n_pos):
                    self.items.append((entry, i))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        entry, i = self.items[idx]
        hidden = entry["hidden_int8"][i].float() * entry["scale"][i]
        targets = entry["targets"][i]
        return hidden, targets


def get_dataloaders(cfg: MedusaConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from pre-cached shards.

    Expects shards named ``train_shard_*.pt`` and ``validation_shard_*.pt``
    inside ``cfg.cache_dir``.
    """
    cache = Path(cfg.cache_dir)
    train_shards = sorted(cache.glob("train_shard_*.pt"))
    val_shards = sorted(cache.glob("validation_shard_*.pt"))

    train_ds = MedusaDataset(train_shards, cfg)
    val_ds = MedusaDataset(val_shards, cfg)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest medusa/tests/test_data.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add medusa/data.py medusa/tests/test_data.py
git commit -m "feat(medusa): MedusaDataset and get_dataloaders over int8 shard format"
```

---

## Task 4: Cache — Phase 1

**Files:**
- Create: `medusa/cache.py`
- Create: `medusa/tests/test_cache.py`

**Interfaces:**
- Consumes: `MedusaConfig` from Task 1
- Produces: sharded `.pt` files in `cfg.cache_dir` matching shard format from Task 3

- [ ] **Step 1: Write failing tests for quantization helpers**

```python
# medusa/tests/test_cache.py
from __future__ import annotations
import torch
import pytest


def _import_helpers():
    from medusa.cache import _quantise_int8, _dequantise_int8
    return _quantise_int8, _dequantise_int8


def test_quantise_shape():
    q, s = _import_helpers()[0]
    t = torch.randn(4, 32)
    q_t, scales = q(t)
    assert q_t.shape == t.shape
    assert q_t.dtype == torch.int8
    assert scales.shape == (4,)


def test_quantise_range():
    q, _ = _import_helpers()
    t = torch.randn(8, 16)
    q_t, _ = q(t)
    assert q_t.abs().max().item() <= 127


def test_roundtrip_close():
    q, dq = _import_helpers()
    t = torch.randn(6, 64)
    q_t, scales = q(t)
    t_rec = dq(q_t, scales)
    assert t_rec.shape == t.shape
    assert (t - t_rec).abs().max().item() < 0.02


def test_zero_tensor():
    q, dq = _import_helpers()
    t = torch.zeros(2, 8)
    q_t, scales = q(t)
    t_rec = dq(q_t, scales)
    assert torch.allclose(t_rec, t)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest medusa/tests/test_cache.py -v
```

Expected: `ModuleNotFoundError: No module named 'medusa.cache'`

- [ ] **Step 3: Write `medusa/cache.py`**

```python
"""
Phase 1: extract Gemma 4 E2B last hidden states from ShareGPT52K and save int8 shards.

Usage:
    python -m medusa.cache --config medusa/configs/default.yaml --split train
    python -m medusa.cache --config medusa/configs/default.yaml --split validation
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from medusa.config import MedusaConfig, load_config


def _quantise_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vector int8 quantization for (n_pos, d_model) tensor.

    Returns:
        q_t:   (n_pos, d_model) int8
        scales: (n_pos,) float32 — one scale per row
    """
    scales = t.float().abs().amax(dim=-1).clamp(min=1e-8) / 127.0
    q_t = (t.float() / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q_t, scales.float()


def _dequantise_int8(q_t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of _quantise_int8. Returns float32 tensor."""
    return q_t.float() * scales.unsqueeze(-1)


def _extract_conversation(
    example: dict,
    tokenizer,
    cfg: MedusaConfig,
) -> tuple[list[int], list[int]]:
    """Extract the first human→gpt exchange from a ShareGPT52K conversation.

    Returns (prompt_ids, answer_ids). Both empty on failure.
    """
    convs = example.get("conversations", [])
    human_turn = next((c for c in convs if c.get("from") == "human"), None)
    gpt_turn = next((c for c in convs if c.get("from") == "gpt"), None)
    if not human_turn or not gpt_turn:
        return [], []

    human_text = human_turn.get("value", "").strip()
    gpt_text = gpt_turn.get("value", "").strip()
    if not human_text or not gpt_text:
        return [], []

    messages = [{"role": "user", "content": human_text}]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = human_text

    prompt_ids = tokenizer.encode(prompt_text)
    answer_ids = tokenizer.encode(gpt_text, add_special_tokens=False)

    # Truncate so total sequence fits within max_seq_len
    max_prompt = cfg.max_seq_len - min(cfg.max_answer_len, len(answer_ids))
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    return prompt_ids, answer_ids


def extract_and_cache(cfg: MedusaConfig, split: str = "train") -> None:
    """Run teacher on ShareGPT52K, extract last hidden states, write int8 shards."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(cfg.cache_dir, exist_ok=True)

    raw = load_dataset(cfg.dataset_id, split="train")
    splits = raw.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    dataset = splits["train"] if split == "train" else splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    shard_data: list[dict] = []
    shard_idx = 0
    cache_dir = Path(cfg.cache_dir)

    for example in tqdm(dataset, desc=f"Caching {split}"):
        prompt_ids, answer_ids = _extract_conversation(example, tokenizer, cfg)
        if not prompt_ids or not answer_ids:
            continue

        n_pos = min(cfg.max_answer_len, len(answer_ids))
        # Full sequence: prompt + first n_pos answer tokens
        full_ids = prompt_ids + list(answer_ids[:n_pos])
        input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=False)
            # last_hidden_state: (1, seq_len, d_model)
            last_hidden = model(input_ids).hidden_states[-1][0].float().cpu()

        # Re-run once requesting hidden states
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            last_hidden = out.hidden_states[-1][0].float().cpu()  # (seq_len, d_model)

        handoff = len(prompt_ids) - 1  # index of last prompt token

        # Extract positions: handoff, handoff+1, ..., handoff+n_pos-1
        # These are the positions just before each answer token is emitted
        positions = list(range(handoff, handoff + n_pos))
        hidden_vecs = last_hidden[positions]  # (n_pos, d_model)

        # Build targets: for position p, targets[k] = full_ids[p + k + 1]
        full_t = torch.tensor(full_ids, dtype=torch.long)
        targets = torch.full((n_pos, cfg.n_heads), -100, dtype=torch.long)
        for i, pos in enumerate(positions):
            for k in range(cfg.n_heads):
                tgt_idx = pos + k + 1
                if tgt_idx < len(full_t):
                    targets[i, k] = full_t[tgt_idx]

        q_hidden, scales = _quantise_int8(hidden_vecs)

        shard_data.append({
            "hidden_int8": q_hidden.cpu(),
            "scale": scales.cpu(),
            "targets": targets.cpu(),
        })

        if len(shard_data) >= cfg.cache_shard_size:
            path = cache_dir / f"{split}_shard_{shard_idx:04d}.pt"
            torch.save(shard_data, path)
            print(f"  Saved {path}")
            shard_data = []
            shard_idx += 1

    if shard_data:
        path = cache_dir / f"{split}_shard_{shard_idx:04d}.pt"
        torch.save(shard_data, path)
        print(f"  Saved {path}")

    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medusa/configs/default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    args = parser.parse_args()
    extract_and_cache(load_config(args.config), split=args.split)
```

- [ ] **Step 4: Run quantization tests — expect PASS**

```bash
pytest medusa/tests/test_cache.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Fix double-forward bug**

The implementation above runs the model twice (a bug). Fix `extract_and_cache` so it only runs the model once:

Replace the two `with torch.no_grad():` blocks with:

```python
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            last_hidden = out.hidden_states[-1][0].float().cpu()  # (seq_len, d_model)
```

- [ ] **Step 6: Commit**

```bash
git add medusa/cache.py medusa/tests/test_cache.py
git commit -m "feat(medusa): cache.py — extract last hidden states from ShareGPT52K, int8 shards"
```

---

## Task 5: Training Loop

**Files:**
- Create: `medusa/train.py`
- Create: `medusa/tests/test_train.py`

**Interfaces:**
- Consumes: `MedusaConfig` (Task 1), `MedusaModel` (Task 2), `get_dataloaders` (Task 3)
- Produces: `medusa/checkpoints/best.pt` with keys `model_state`, `optimizer_state`, `cfg`, `epoch`

- [ ] **Step 1: Write failing smoke test**

```python
# medusa/tests/test_train.py
from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import pytest

from medusa.config import MedusaConfig
from medusa.models.medusa_model import MedusaModel


VOCAB = 64
D_MODEL = 32
N_HEADS = 3
B = 4


@pytest.fixture
def cfg():
    return MedusaConfig(n_heads=N_HEADS, d_model=D_MODEL, lambda_decay=0.8, grad_clip=1.0)


@pytest.fixture
def model(cfg):
    lm_w = torch.randn(VOCAB, D_MODEL)
    return MedusaModel(cfg, lm_w)


def test_training_step_returns_scalar(cfg, model):
    from medusa.train import training_step
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-3)
    hidden = torch.randn(B, D_MODEL)
    targets = torch.randint(0, VOCAB, (B, N_HEADS))
    loss = training_step(model, hidden, targets, cfg, optimizer)
    assert loss.shape == ()
    assert loss.item() > 0


def test_training_step_updates_weights(cfg, model):
    from medusa.train import training_step
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-3)
    w_before = model.heads[0].W1.weight.clone()
    hidden = torch.randn(B, D_MODEL)
    targets = torch.randint(0, VOCAB, (B, N_HEADS))
    training_step(model, hidden, targets, cfg, optimizer)
    assert not torch.equal(model.heads[0].W1.weight, w_before)


def test_training_step_ignores_minus100(cfg, model):
    from medusa.train import training_step
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-3)
    hidden = torch.randn(B, D_MODEL)
    # All targets padded
    targets = torch.full((B, N_HEADS), -100, dtype=torch.long)
    loss = training_step(model, hidden, targets, cfg, optimizer)
    assert not torch.isnan(loss)


def test_make_lr_lambda():
    from medusa.train import make_lr_lambda
    fn = make_lr_lambda(warmup_steps=100, total_steps=1000)
    assert fn(0) == pytest.approx(0.0)
    assert fn(50) == pytest.approx(0.5)
    assert fn(100) == pytest.approx(1.0)
    assert 0.0 < fn(550) < 1.0
    assert fn(1000) == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest medusa/tests/test_train.py -v
```

Expected: `ModuleNotFoundError: No module named 'medusa.train'`

- [ ] **Step 3: Write `medusa/train.py`**

```python
"""
Phase 2: train Medusa-1 heads on pre-cached ShareGPT52K features.

Usage:
    python -m medusa.train --config medusa/configs/default.yaml
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from medusa.config import MedusaConfig, load_config
from medusa.data import get_dataloaders
from medusa.models.medusa_model import MedusaModel


def make_lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def training_step(
    model: MedusaModel,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    cfg: MedusaConfig,
    optimizer: AdamW,
) -> torch.Tensor:
    """
    Single training step. Computes distance-weighted cross-entropy loss,
    back-propagates, clips gradients, steps optimizer.

    hidden:  (B, d_model)
    targets: (B, n_heads) — -100 for padding positions
    Returns: scalar loss (detached)
    """
    model.train()
    logits = model(hidden)  # (B, n_heads, vocab)

    loss = torch.zeros((), device=logits.device)
    for k in range(logits.shape[1]):
        w = cfg.lambda_decay ** k
        loss = loss + w * F.cross_entropy(logits[:, k, :], targets[:, k], ignore_index=-100)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return loss.detach()


def train(cfg: MedusaConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading teacher LM head weight...")
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id, torch_dtype=torch.bfloat16
    )
    lm_head_w = teacher.get_output_embeddings().weight.detach().clone().float()
    del teacher
    print(f"LM head weight: {lm_head_w.shape}")

    model = MedusaModel(cfg, lm_head_w).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    train_loader, val_loader = get_dataloaders(cfg)
    total_steps = cfg.n_epochs * len(train_loader)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = LambdaLR(optimizer, make_lr_lambda(cfg.warmup_steps, total_steps))

    best_val_loss = float("inf")
    ckpt_dir = Path(cfg.cache_dir).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.n_epochs):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for hidden, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.n_epochs}"):
            hidden = hidden.to(device)
            targets = targets.to(device)
            loss = training_step(model, hidden, targets, cfg, optimizer)
            scheduler.step()
            total_train_loss += loss.item()
            n_batches += 1

        avg_train = total_train_loss / max(1, n_batches)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for hidden, targets in val_loader:
                hidden = hidden.to(device)
                targets = targets.to(device)
                logits = model(hidden)  # (B, n_heads, vocab)
                batch_loss = 0.0
                for k in range(logits.shape[1]):
                    w = cfg.lambda_decay ** k
                    batch_loss += w * F.cross_entropy(
                        logits[:, k, :], targets[:, k], ignore_index=-100
                    ).item()
                val_loss_sum += batch_loss
                n_val += 1

        avg_val = val_loss_sum / max(1, n_val)
        print(f"Epoch {epoch + 1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Checkpoint saved (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medusa/configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest medusa/tests/test_train.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest medusa/tests/ -v
```

Expected: all tests PASS (no teacher required — all tests use random tensors or synthetic shards).

- [ ] **Step 6: Commit**

```bash
git add medusa/train.py medusa/tests/test_train.py
git commit -m "feat(medusa): Medusa-1 training loop — distance-weighted MTP loss, cosine LR, checkpoint"
```

---

## Usage Summary

```bash
# Phase 1: extract features (requires Gemma 4 E2B + GPU)
python -m medusa.cache --config medusa/configs/default.yaml --split train
python -m medusa.cache --config medusa/configs/default.yaml --split validation

# Phase 2: train heads
python -m medusa.train --config medusa/configs/default.yaml

# Tests (no GPU required)
pytest medusa/tests/ -v
```
