# MTP Draft Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a <50M-param speculative decoding draft model that uses Gemma 4 E2b intermediate features as cross-attention queries and predicts a tree of draft tokens in parallel, trained on HotpotQA.

**Architecture:** Cross-attention decoder where Q = KromHC-fused teacher hidden states + sinusoidal step embedding per draft position, K=V = projected context token embeddings (frozen teacher embedding). A single frozen teacher LM head + trainable LoRA outputs logits for all draft positions simultaneously. Tree candidates are constructed from relative log-prob thresholding at inference.

**Tech Stack:** Python 3.10+, PyTorch 2.11, HuggingFace `transformers` 5.9, `datasets` 4.8, `pytest` 9.0

## Global Constraints

- All new code lives in `mtp_draft/` (new top-level package alongside `rbf_ffn/`)
- `KromHCHeadMixer` from `rbf_ffn/models/head_mixer.py` requires `n_heads` to be a power of 2 — use **4 teacher layers** `[3, 8, 14, 17]` (not 3) so `n_heads=4` satisfies this
- No bias on any projection linear layer (Llama convention, consistent with rbf_ffn)
- All test files live in `mtp_draft/tests/`; run with `pytest mtp_draft/tests/`
- Hardware target: 16 GB GPU, 16 GB RAM, 15 GB disk
- Teacher model: `"google/gemma-4-e2b-it"`, `d_teacher=2048`
- Frozen parameters: teacher embedding weight, teacher LM head weight (only LoRA A, B are trainable in the head)
- `ffn_hidden` and `n_blocks` are always read from `MTPConfig` — never hardcoded in model code
- Pre-norm with `nn.RMSNorm`, consistent with rbf_ffn

---

## File Map

| File | Responsibility |
|---|---|
| `mtp_draft/__init__.py` | empty package marker |
| `mtp_draft/models/__init__.py` | empty package marker |
| `mtp_draft/tests/__init__.py` | empty package marker |
| `mtp_draft/config.py` | `MTPConfig` dataclass + `load_config()` |
| `mtp_draft/configs/default.yaml` | default hyperparameters |
| `mtp_draft/models/fusion.py` | `TeacherFeatureFusion`: KromHC over N teacher layers |
| `mtp_draft/models/step_embed.py` | `StepEmbedding`: sinusoidal + 2-layer MLP |
| `mtp_draft/models/cross_attn_block.py` | `CrossAttnBlock`: KV-shared cross-attn + SwiGLU FFN, pre-norm |
| `mtp_draft/models/lora_lm_head.py` | `LoRALMHead`: frozen weight buffer + trainable A, B |
| `mtp_draft/models/draft_model.py` | `MTPDraftModel`: wires all components end-to-end |
| `mtp_draft/data.py` | `build_prompt()`, `FeatureDataset`, `get_dataloaders()` |
| `mtp_draft/cache.py` | `extract_and_cache()`: Phase 1 feature extraction + int8 shards |
| `mtp_draft/train.py` | `train()`: Phase 2 training loop |
| `mtp_draft/tree.py` | `build_tree()`: relative log-prob threshold tree construction |
| `mtp_draft/tests/test_fusion.py` | unit tests for `TeacherFeatureFusion` |
| `mtp_draft/tests/test_step_embed.py` | unit tests for `StepEmbedding` |
| `mtp_draft/tests/test_cross_attn_block.py` | unit tests for `CrossAttnBlock` |
| `mtp_draft/tests/test_lora_lm_head.py` | unit tests for `LoRALMHead` |
| `mtp_draft/tests/test_draft_model.py` | integration smoke tests for `MTPDraftModel` |
| `mtp_draft/tests/test_tree.py` | unit tests for `build_tree` |
| `pyproject.toml` | add `mtp_draft` to packages and testpaths |

---

## Task 1: Package Scaffold, Config, and pyproject.toml

**Files:**
- Create: `mtp_draft/__init__.py`
- Create: `mtp_draft/models/__init__.py`
- Create: `mtp_draft/tests/__init__.py`
- Create: `mtp_draft/config.py`
- Create: `mtp_draft/configs/default.yaml`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `MTPConfig` dataclass importable as `from mtp_draft.config import MTPConfig`
  - `load_config(path: str) -> MTPConfig`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p mtp_draft/models mtp_draft/tests mtp_draft/configs
touch mtp_draft/__init__.py mtp_draft/models/__init__.py mtp_draft/tests/__init__.py
```

- [ ] **Step 2: Write `mtp_draft/config.py`**

```python
# mtp_draft/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class MTPConfig:
    # Draft model dimensions
    d_draft: int = 512
    n_blocks: int = 4
    ffn_hidden: int = 1366        # SwiGLU hidden; 0 = auto as int(8/3 * d_draft)
    n_heads: int = 8
    dropout: float = 0.0
    use_xsa: bool = False         # XSA orthogonalisation (no-op in cross-attn, for future)

    # Teacher
    teacher_model_id: str = "google/gemma-4-e2b-it"
    teacher_layers: list[int] = field(default_factory=lambda: [3, 8, 14, 17])
    d_teacher: int = 2048

    # Training
    max_draft: int = 8
    lambda_decay: float = 0.8
    lora_rank: int = 16
    max_prompt_len: int = 256

    # Inference
    tau: float = 2.0
    max_tree_nodes: int = 256

    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 16
    n_epochs: int = 3
    warmup_steps: int = 200
    seed: int = 42

    # Data / cache
    cache_dir: str = "mtp_draft/cache"
    cache_n_answer_positions: int = 8
    cache_shard_size: int = 5000

    def __post_init__(self) -> None:
        if self.ffn_hidden == 0:
            self.ffn_hidden = int(8 / 3 * self.d_draft)
        n = len(self.teacher_layers)
        assert (n & (n - 1)) == 0, (
            f"len(teacher_layers) must be a power of 2 for KromHC; got {n}"
        )


def load_config(path: str) -> MTPConfig:
    data = yaml.safe_load(Path(path).read_text())
    return MTPConfig(**data)
```

- [ ] **Step 3: Write `mtp_draft/configs/default.yaml`**

```yaml
d_draft: 512
n_blocks: 4
ffn_hidden: 1366
n_heads: 8
dropout: 0.0
use_xsa: false
teacher_model_id: "google/gemma-4-e2b-it"
teacher_layers: [3, 8, 14, 17]
d_teacher: 2048
max_draft: 8
lambda_decay: 0.8
lora_rank: 16
max_prompt_len: 256
tau: 2.0
max_tree_nodes: 256
lr: 3.0e-4
weight_decay: 0.1
grad_clip: 1.0
batch_size: 16
n_epochs: 3
warmup_steps: 200
seed: 42
cache_dir: "mtp_draft/cache"
cache_n_answer_positions: 8
cache_shard_size: 5000
```

- [ ] **Step 4: Update `pyproject.toml`** — add `mtp_draft` to packages and testpaths

Open `pyproject.toml` and make these two edits:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["rbf_ffn*", "kromhc_transformer*", "grokking*", "flow_matching*", "sigreg*", "mamba_lm*", "idlm*", "mtp_draft*"]

[tool.pytest.ini_options]
testpaths = ["rbf_ffn/tests", "kromhc_transformer/tests", "grokking/tests", "sigreg/tests", "idlm/tests", "mtp_draft/tests"]
```

- [ ] **Step 5: Write the failing config test**

```python
# mtp_draft/tests/test_config.py
import pytest
from mtp_draft.config import MTPConfig, load_config
import tempfile, yaml, os

def test_default_config_valid():
    cfg = MTPConfig()
    assert cfg.d_draft == 512
    assert cfg.n_blocks == 4
    assert cfg.ffn_hidden == 1366
    assert len(cfg.teacher_layers) == 4

def test_ffn_hidden_auto():
    cfg = MTPConfig(d_draft=384, ffn_hidden=0)
    assert cfg.ffn_hidden == int(8 / 3 * 384)

def test_teacher_layers_power_of_two():
    with pytest.raises(AssertionError, match="power of 2"):
        MTPConfig(teacher_layers=[3, 8, 17])  # 3 layers = not power of 2

def test_load_config_roundtrip():
    cfg = MTPConfig(d_draft=256, n_blocks=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"d_draft": 256, "n_blocks": 2}, f)
        path = f.name
    loaded = load_config(path)
    os.unlink(path)
    assert loaded.d_draft == 256
    assert loaded.n_blocks == 2
```

- [ ] **Step 6: Run config tests**

```bash
pytest mtp_draft/tests/test_config.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add mtp_draft/ pyproject.toml
git commit -m "feat(mtp): scaffold package, MTPConfig, default config"
```

---

## Task 2: KromHC Multi-Layer Feature Fusion

**Files:**
- Create: `mtp_draft/models/fusion.py`
- Create: `mtp_draft/tests/test_fusion.py`

**Interfaces:**
- Consumes: `KromHCHeadMixer` from `rbf_ffn.models.head_mixer`
- Produces:
  - `TeacherFeatureFusion(n_teacher_layers: int, d_teacher: int, d_draft: int, mixer_hidden: int = 32)`
  - `forward(hidden_states: Tensor) -> Tensor`
    - input: `(B, n_teacher_layers, d_teacher)`
    - output: `(B, d_draft)`

- [ ] **Step 1: Write the failing test**

```python
# mtp_draft/tests/test_fusion.py
import torch
import pytest
from mtp_draft.models.fusion import TeacherFeatureFusion

B, N_LAYERS, D_TEACHER, D_DRAFT = 3, 4, 2048, 512


@pytest.fixture
def fusion():
    return TeacherFeatureFusion(n_teacher_layers=N_LAYERS, d_teacher=D_TEACHER, d_draft=D_DRAFT)


def test_output_shape(fusion):
    x = torch.randn(B, N_LAYERS, D_TEACHER)
    out = fusion(x)
    assert out.shape == (B, D_DRAFT)


def test_output_differs_per_batch(fusion):
    x = torch.randn(B, N_LAYERS, D_TEACHER)
    out = fusion(x)
    assert not torch.allclose(out[0], out[1])


def test_gradients_flow(fusion):
    x = torch.randn(B, N_LAYERS, D_TEACHER, requires_grad=False)
    out = fusion(x)
    loss = out.sum()
    loss.backward()
    for name, p in fusion.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


def test_power_of_two_check():
    with pytest.raises(AssertionError):
        TeacherFeatureFusion(n_teacher_layers=3, d_teacher=64, d_draft=32)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest mtp_draft/tests/test_fusion.py -v
```

Expected: ImportError (module not yet created).

- [ ] **Step 3: Write `mtp_draft/models/fusion.py`**

```python
# mtp_draft/models/fusion.py
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
```

- [ ] **Step 4: Run tests**

```bash
pytest mtp_draft/tests/test_fusion.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mtp_draft/models/fusion.py mtp_draft/tests/test_fusion.py
git commit -m "feat(mtp): TeacherFeatureFusion with KromHC head mixing"
```

---

## Task 3: Step Embedding

**Files:**
- Create: `mtp_draft/models/step_embed.py`
- Create: `mtp_draft/tests/test_step_embed.py`

**Interfaces:**
- Produces:
  - `StepEmbedding(d_model: int, max_steps: int = 64)`
  - `forward(steps: Tensor) -> Tensor`
    - input: `(B, S)` integer step indices (1-indexed, values in `1..max_draft`)
    - output: `(B, S, d_model)`

- [ ] **Step 1: Write the failing test**

```python
# mtp_draft/tests/test_step_embed.py
import torch
import pytest
from mtp_draft.models.step_embed import StepEmbedding

B, S, D = 2, 8, 512


@pytest.fixture
def embed():
    return StepEmbedding(d_model=D, max_steps=16)


def test_output_shape(embed):
    steps = torch.arange(1, S + 1).unsqueeze(0).expand(B, -1)
    out = embed(steps)
    assert out.shape == (B, S, D)


def test_different_steps_differ(embed):
    steps = torch.tensor([[1, 2]])
    out = embed(steps)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_same_step_same_output(embed):
    s1 = torch.tensor([[3, 3]])
    out = embed(s1)
    assert torch.allclose(out[0, 0], out[0, 1])


def test_gradients_flow(embed):
    steps = torch.arange(1, S + 1).unsqueeze(0).expand(B, -1)
    out = embed(steps)
    out.sum().backward()
    for name, p in embed.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest mtp_draft/tests/test_step_embed.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `mtp_draft/models/step_embed.py`**

```python
# mtp_draft/models/step_embed.py
from __future__ import annotations
import math
import torch
import torch.nn as nn


def _sinusoidal(steps: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    steps: (B, S) integer indices
    Returns (B, S, d_model) sinusoidal embeddings (DDPM-style).
    """
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=steps.device, dtype=torch.float32) / half
    )  # (half,)
    args = steps.float().unsqueeze(-1) * freqs  # (B, S, half)
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, S, d_model)


class StepEmbedding(nn.Module):
    """
    Maps integer draft-step indices to d_model vectors via:
        sinusoidal(i) → Linear(d_model, 4*d_model) → SiLU → Linear(4*d_model, d_model)

    Input:  (B, S) integer step indices (1-indexed)
    Output: (B, S, d_model)
    """

    def __init__(self, d_model: int, max_steps: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        """steps: (B, S) → (B, S, d_model)"""
        emb = _sinusoidal(steps, self.d_model)  # (B, S, d_model)
        return self.mlp(emb)
```

- [ ] **Step 4: Run tests**

```bash
pytest mtp_draft/tests/test_step_embed.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mtp_draft/models/step_embed.py mtp_draft/tests/test_step_embed.py
git commit -m "feat(mtp): StepEmbedding — sinusoidal + MLP step conditioning"
```

---

## Task 4: Cross-Attention Block

**Files:**
- Create: `mtp_draft/models/cross_attn_block.py`
- Create: `mtp_draft/tests/test_cross_attn_block.py`

**Interfaces:**
- Consumes: `MTPConfig` from `mtp_draft.config`
- Produces:
  - `CrossAttnBlock(cfg: MTPConfig)`
  - `forward(query: Tensor, context: Tensor) -> Tensor`
    - `query`: `(B, S, d_draft)` — S draft positions
    - `context`: `(B, N, d_draft)` — N context tokens
    - output: `(B, S, d_draft)`

- [ ] **Step 1: Write the failing test**

```python
# mtp_draft/tests/test_cross_attn_block.py
import torch
import pytest
from mtp_draft.config import MTPConfig
from mtp_draft.models.cross_attn_block import CrossAttnBlock

B, S, N = 2, 8, 64  # batch, draft positions, context length


@pytest.fixture
def cfg():
    return MTPConfig(d_draft=64, n_heads=4, ffn_hidden=128, dropout=0.0)


@pytest.fixture
def block(cfg):
    return CrossAttnBlock(cfg)


def test_output_shape(block, cfg):
    query = torch.randn(B, S, cfg.d_draft)
    context = torch.randn(B, N, cfg.d_draft)
    out = block(query, context)
    assert out.shape == (B, S, cfg.d_draft)


def test_query_positions_independent(block, cfg):
    """Changing one query position must not change another (no causal mask between draft positions)."""
    query = torch.randn(B, S, cfg.d_draft)
    context = torch.randn(B, N, cfg.d_draft)

    out1 = block(query, context)

    query2 = query.clone()
    query2[:, 0, :] += 1.0  # perturb only position 0
    out2 = block(query2, context)

    # position 0 changes
    assert not torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-4)
    # position 1 must NOT change (no causal dependency)
    assert torch.allclose(out1[:, 1, :], out2[:, 1, :], atol=1e-4)


def test_different_context_changes_output(block, cfg):
    query = torch.randn(B, S, cfg.d_draft)
    ctx1 = torch.randn(B, N, cfg.d_draft)
    ctx2 = torch.randn(B, N, cfg.d_draft)
    assert not torch.allclose(block(query, ctx1), block(query, ctx2), atol=1e-4)


def test_no_bias_on_projections(block):
    for name, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            assert m.bias is None, f"{name} has unexpected bias"


def test_gradients_flow(block, cfg):
    query = torch.randn(B, S, cfg.d_draft, requires_grad=True)
    context = torch.randn(B, N, cfg.d_draft, requires_grad=True)
    out = block(query, context)
    out.sum().backward()
    assert query.grad is not None
    assert context.grad is not None
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest mtp_draft/tests/test_cross_attn_block.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `mtp_draft/models/cross_attn_block.py`**

```python
# mtp_draft/models/cross_attn_block.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtp_draft.config import MTPConfig


class CrossAttnBlock(nn.Module):
    """
    Pre-norm cross-attention block with KV-sharing and SwiGLU FFN.

    K and V both come from a single kv_proj applied to the context
    (same projection, no RoPE — context positions are not relative to
    draft positions). No causal mask between draft query positions.

    Input:
        query:   (B, S, d_draft)  — S independent draft positions
        context: (B, N, d_draft)  — N context tokens

    Output: (B, S, d_draft)
    """

    def __init__(self, cfg: MTPConfig) -> None:
        super().__init__()
        D = cfg.d_draft
        H = cfg.n_heads
        assert D % H == 0, f"d_draft ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self._dropout = cfg.dropout

        # Cross-attention projections
        self.q_proj = nn.Linear(D, D, bias=False)
        self.kv_proj = nn.Linear(D, D, bias=False)   # K = V = kv_proj(context)
        self.o_proj = nn.Linear(D, D, bias=False)

        # FFN (SwiGLU)
        H_ffn = cfg.ffn_hidden
        self.gate_proj = nn.Linear(D, H_ffn, bias=False)
        self.up_proj = nn.Linear(D, H_ffn, bias=False)
        self.down_proj = nn.Linear(H_ffn, D, bias=False)

        # Pre-norm
        self.norm_q = nn.RMSNorm(D)
        self.norm_ctx = nn.RMSNorm(D)
        self.norm_ffn = nn.RMSNorm(D)

    def _attn(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, S, D = query.shape
        N = context.shape[1]
        H, hd = self.n_heads, self.head_dim

        q = self.q_proj(query).view(B, S, H, hd).transpose(1, 2)   # (B, H, S, hd)
        kv = self.kv_proj(context).view(B, N, H, hd).transpose(1, 2)  # (B, H, N, hd)
        # K and V are the same projected tensor (KV-sharing)
        k = kv
        v = kv

        dp = self._dropout if self.training else 0.0
        # is_causal=False: draft positions attend freely to all context positions
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=False)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query:   (B, S, d_draft)
        context: (B, N, d_draft)
        returns: (B, S, d_draft)
        """
        query = query + self._attn(self.norm_q(query), self.norm_ctx(context))
        query = query + self._ffn(self.norm_ffn(query))
        return query
```

- [ ] **Step 4: Run tests**

```bash
pytest mtp_draft/tests/test_cross_attn_block.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mtp_draft/models/cross_attn_block.py mtp_draft/tests/test_cross_attn_block.py
git commit -m "feat(mtp): CrossAttnBlock — KV-shared cross-attention + SwiGLU FFN"
```

---

## Task 5: LoRA LM Head

**Files:**
- Create: `mtp_draft/models/lora_lm_head.py`
- Create: `mtp_draft/tests/test_lora_lm_head.py`

**Interfaces:**
- Produces:
  - `LoRALMHead(frozen_weight: Tensor, lora_rank: int = 16)`
    - `frozen_weight`: `(vocab, d_teacher)` — teacher LM head weight
  - `forward(x: Tensor) -> Tensor`
    - input: `(B, S, d_teacher)`
    - output: `(B, S, vocab)`

- [ ] **Step 1: Write the failing test**

```python
# mtp_draft/tests/test_lora_lm_head.py
import torch
import pytest
from mtp_draft.models.lora_lm_head import LoRALMHead

VOCAB, D_TEACHER, RANK = 100, 64, 4
B, S = 2, 8


@pytest.fixture
def frozen_weight():
    return torch.randn(VOCAB, D_TEACHER)


@pytest.fixture
def head(frozen_weight):
    return LoRALMHead(frozen_weight, lora_rank=RANK)


def test_output_shape(head):
    x = torch.randn(B, S, D_TEACHER)
    out = head(x)
    assert out.shape == (B, S, VOCAB)


def test_frozen_weight_no_grad(head):
    """Registered buffer must not accumulate gradients."""
    x = torch.randn(B, S, D_TEACHER)
    head(x).sum().backward()
    assert head.weight.grad is None


def test_lora_params_have_grad(head):
    x = torch.randn(B, S, D_TEACHER)
    head(x).sum().backward()
    assert head.lora_A.grad is not None
    assert head.lora_B.grad is not None


def test_lora_B_init_zero(frozen_weight):
    """B=0 means LoRA starts as a no-op: output should equal frozen-only output."""
    head = LoRALMHead(frozen_weight, lora_rank=RANK)
    assert torch.all(head.lora_B == 0)
    x = torch.randn(1, 1, D_TEACHER)
    out_lora = head(x)
    out_frozen = x @ frozen_weight.T
    assert torch.allclose(out_lora, out_frozen, atol=1e-5)


def test_weight_not_updated_after_backward(head):
    """Frozen weight buffer must be unchanged after an optimizer step."""
    import copy
    w_before = head.weight.clone()
    opt = torch.optim.AdamW([head.lora_A, head.lora_B], lr=1e-3)
    x = torch.randn(B, S, D_TEACHER)
    head(x).sum().backward()
    opt.step()
    assert torch.allclose(head.weight, w_before)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest mtp_draft/tests/test_lora_lm_head.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `mtp_draft/models/lora_lm_head.py`**

```python
# mtp_draft/models/lora_lm_head.py
from __future__ import annotations
import math
import torch
import torch.nn as nn


class LoRALMHead(nn.Module):
    """
    Frozen teacher LM head weight with a trainable low-rank adapter.

    logits = x @ (W + B @ A).T
    where W is frozen, A and B are the LoRA matrices.

    B is initialised to zero so the adapter starts as a no-op.

    Input:  (B, S, d_teacher)
    Output: (B, S, vocab)
    """

    def __init__(self, frozen_weight: torch.Tensor, lora_rank: int = 16) -> None:
        super().__init__()
        vocab, d = frozen_weight.shape
        self.register_buffer("weight", frozen_weight.detach())
        self.lora_A = nn.Parameter(torch.empty(lora_rank, d))
        self.lora_B = nn.Parameter(torch.zeros(vocab, lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_teacher) → (B, S, vocab)"""
        W = self.weight + self.lora_B @ self.lora_A   # (vocab, d_teacher)
        return x @ W.T
```

- [ ] **Step 4: Run tests**

```bash
pytest mtp_draft/tests/test_lora_lm_head.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mtp_draft/models/lora_lm_head.py mtp_draft/tests/test_lora_lm_head.py
git commit -m "feat(mtp): LoRALMHead — frozen teacher LM head + trainable LoRA A, B"
```

---

## Task 6: MTPDraftModel

**Files:**
- Create: `mtp_draft/models/draft_model.py`
- Create: `mtp_draft/tests/test_draft_model.py`

**Interfaces:**
- Consumes:
  - `TeacherFeatureFusion` from `mtp_draft.models.fusion`
  - `StepEmbedding` from `mtp_draft.models.step_embed`
  - `CrossAttnBlock` from `mtp_draft.models.cross_attn_block`
  - `LoRALMHead` from `mtp_draft.models.lora_lm_head`
  - `MTPConfig` from `mtp_draft.config`
- Produces:
  - `MTPDraftModel(cfg: MTPConfig, teacher_embedding_weight: Tensor, teacher_lm_head_weight: Tensor)`
  - `forward(teacher_hiddens: Tensor, context_ids: Tensor) -> Tensor`
    - `teacher_hiddens`: `(B, n_teacher_layers, d_teacher)`
    - `context_ids`: `(B, seq_len)` integer token ids
    - output: `(B, max_draft, vocab)`
  - `trainable_parameters() -> list[nn.Parameter]` — returns only non-frozen params

- [ ] **Step 1: Write the failing test**

```python
# mtp_draft/tests/test_draft_model.py
import torch
import pytest
from mtp_draft.config import MTPConfig
from mtp_draft.models.draft_model import MTPDraftModel

VOCAB, D_TEACHER = 200, 64
B, SEQ_LEN = 2, 32

@pytest.fixture
def cfg():
    return MTPConfig(
        d_draft=64, n_blocks=2, ffn_hidden=128, n_heads=4,
        d_teacher=D_TEACHER, max_draft=4,
        teacher_layers=[0, 1, 2, 3],  # 4 layers = power of 2
        lora_rank=4,
    )

@pytest.fixture
def model(cfg):
    emb_w = torch.randn(VOCAB, D_TEACHER)
    lm_w = torch.randn(VOCAB, D_TEACHER)
    return MTPDraftModel(cfg, emb_w, lm_w)

@pytest.fixture
def inputs(cfg):
    hiddens = torch.randn(B, len(cfg.teacher_layers), cfg.d_teacher)
    ctx_ids = torch.randint(0, VOCAB, (B, SEQ_LEN))
    return hiddens, ctx_ids


def test_output_shape(model, inputs, cfg):
    hiddens, ctx_ids = inputs
    out = model(hiddens, ctx_ids)
    assert out.shape == (B, cfg.max_draft, VOCAB)


def test_frozen_embedding_no_grad(model, inputs):
    hiddens, ctx_ids = inputs
    model(hiddens, ctx_ids).sum().backward()
    assert model.token_embedding.weight.grad is None


def test_frozen_lm_head_weight_no_grad(model, inputs):
    hiddens, ctx_ids = inputs
    model(hiddens, ctx_ids).sum().backward()
    assert model.lm_head.weight.grad is None


def test_trainable_params_have_grad(model, inputs):
    hiddens, ctx_ids = inputs
    model(hiddens, ctx_ids).sum().backward()
    for p in model.trainable_parameters():
        assert p.grad is not None


def test_param_count_under_50m(model):
    n = sum(p.numel() for p in model.trainable_parameters())
    assert n < 50_000_000, f"Trainable params {n:,} exceed 50M"
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest mtp_draft/tests/test_draft_model.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `mtp_draft/models/draft_model.py`**

```python
# mtp_draft/models/draft_model.py
from __future__ import annotations
import torch
import torch.nn as nn
from mtp_draft.config import MTPConfig
from mtp_draft.models.fusion import TeacherFeatureFusion
from mtp_draft.models.step_embed import StepEmbedding
from mtp_draft.models.cross_attn_block import CrossAttnBlock
from mtp_draft.models.lora_lm_head import LoRALMHead


class MTPDraftModel(nn.Module):
    """
    Multi-Token Prediction draft model.

    Forward pass:
      1. Fuse teacher hidden states → Q_fused (B, d_draft) via KromHC
      2. Add per-position step embeddings → Q (B, max_draft, d_draft)
      3. Embed context_ids with frozen teacher embedding, project to d_draft → context
      4. N cross-attention blocks: query = Q, context = context
      5. Project output back to d_teacher, apply frozen LM head + LoRA

    teacher_embedding_weight and teacher_lm_head_weight are registered as
    buffers (no gradient). Only the components listed in trainable_parameters()
    carry gradients.
    """

    def __init__(
        self,
        cfg: MTPConfig,
        teacher_embedding_weight: torch.Tensor,
        teacher_lm_head_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Frozen teacher embedding
        vocab, d_emb = teacher_embedding_weight.shape
        self.token_embedding = nn.Embedding(vocab, d_emb)
        self.token_embedding.weight = nn.Parameter(
            teacher_embedding_weight.detach(), requires_grad=False
        )

        # Context projection: d_teacher → d_draft
        self.ctx_proj = nn.Linear(cfg.d_teacher, cfg.d_draft, bias=False)

        # KromHC multi-layer feature fusion
        self.fusion = TeacherFeatureFusion(
            n_teacher_layers=len(cfg.teacher_layers),
            d_teacher=cfg.d_teacher,
            d_draft=cfg.d_draft,
        )

        # Step conditioning
        self.step_embed = StepEmbedding(d_model=cfg.d_draft, max_steps=cfg.max_draft + 1)

        # Cross-attention blocks
        self.blocks = nn.ModuleList([CrossAttnBlock(cfg) for _ in range(cfg.n_blocks)])

        # Output projection: d_draft → d_teacher (to match LM head input dim)
        self.out_proj = nn.Linear(cfg.d_draft, cfg.d_teacher, bias=False)

        # Frozen LM head + trainable LoRA
        self.lm_head = LoRALMHead(teacher_lm_head_weight, cfg.lora_rank)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        teacher_hiddens: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        teacher_hiddens: (B, n_teacher_layers, d_teacher)
        context_ids:     (B, seq_len)
        returns:         (B, max_draft, vocab)
        """
        B = teacher_hiddens.shape[0]
        device = teacher_hiddens.device

        # 1. Fuse teacher features → (B, d_draft)
        q_fused = self.fusion(teacher_hiddens)

        # 2. Step embeddings for each draft position
        steps = torch.arange(1, self.cfg.max_draft + 1, device=device)
        steps = steps.unsqueeze(0).expand(B, -1)          # (B, max_draft)
        step_embs = self.step_embed(steps)                 # (B, max_draft, d_draft)
        query = q_fused.unsqueeze(1) + step_embs           # (B, max_draft, d_draft)

        # 3. Context: embed tokens (frozen) then project to d_draft
        ctx_emb = self.token_embedding(context_ids)        # (B, seq_len, d_teacher)
        context = self.ctx_proj(ctx_emb)                   # (B, seq_len, d_draft)

        # 4. Cross-attention blocks
        for block in self.blocks:
            query = block(query, context)                  # (B, max_draft, d_draft)

        # 5. Project to d_teacher, apply LM head + LoRA
        out = self.out_proj(query)                         # (B, max_draft, d_teacher)
        return self.lm_head(out)                           # (B, max_draft, vocab)
```

- [ ] **Step 4: Run tests**

```bash
pytest mtp_draft/tests/test_draft_model.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mtp_draft/models/draft_model.py mtp_draft/tests/test_draft_model.py
git commit -m "feat(mtp): MTPDraftModel — end-to-end cross-attention draft model"
```

---

## Task 7: HotpotQA Data Pipeline

**Files:**
- Create: `mtp_draft/data.py`
- Create: `mtp_draft/tests/test_data.py`

**Interfaces:**
- Produces:
  - `build_prompt(example: dict, tokenizer, max_prompt_len: int) -> tuple[list[int], list[int]]`
    - returns `(prompt_ids, answer_ids)`
  - `FeatureDataset(shard_paths: list[Path], cfg: MTPConfig)`
    - `__getitem__` returns `(hiddens: Tensor(n_layers, d_teacher), context_ids: Tensor(max_prompt_len,), targets: Tensor(max_draft,), valid_len: Tensor scalar)`
  - `get_dataloaders(cfg: MTPConfig) -> tuple[DataLoader, DataLoader]`

- [ ] **Step 1: Write the failing test**

```python
# mtp_draft/tests/test_data.py
import torch, tempfile
from pathlib import Path
import pytest
from mtp_draft.config import MTPConfig
from mtp_draft.data import build_prompt, FeatureDataset

VOCAB, D_TEACHER = 50, 16

@pytest.fixture
def cfg():
    return MTPConfig(
        d_teacher=D_TEACHER, max_prompt_len=32, max_draft=4,
        teacher_layers=[0, 1, 2, 3], cache_n_answer_positions=2,
    )


def _fake_tokenizer(text: str) -> list[int]:
    """Trivial whitespace tokenizer for testing (returns word indices mod VOCAB)."""
    return [hash(w) % VOCAB for w in text.split()]


class _MockTokenizer:
    def encode(self, text, add_special_tokens=True):
        return _fake_tokenizer(text)


def _make_example():
    return {
        "question": "Where was Einstein born?",
        "context": {
            "title": ["Einstein", "Physics"],
            "sentences": [
                ["Albert Einstein was born in Ulm."],
                ["He developed the theory of relativity."],
            ],
        },
        "answer": "Ulm",
    }


def test_build_prompt_returns_nonempty():
    tok = _MockTokenizer()
    prompt_ids, answer_ids = build_prompt(_make_example(), tok, max_prompt_len=64)
    assert len(prompt_ids) > 0
    assert len(answer_ids) > 0


def test_build_prompt_truncates_to_max(cfg):
    tok = _MockTokenizer()
    prompt_ids, _ = build_prompt(_make_example(), tok, max_prompt_len=cfg.max_prompt_len)
    assert len(prompt_ids) <= cfg.max_prompt_len


def _make_shard(cfg, tmp_path: Path) -> Path:
    n_layers = len(cfg.teacher_layers)
    shard = []
    for _ in range(3):
        features = torch.randn(cfg.cache_n_answer_positions, n_layers, cfg.d_teacher)
        scale = features.abs().max() / 127.0
        features_int8 = (features / scale).clamp(-128, 127).to(torch.int8)
        shard.append({
            "features_int8": features_int8,
            "scale": scale,
            "prompt_ids": torch.randint(0, VOCAB, (20,)),
            "answer_ids": torch.randint(0, VOCAB, (8,)),
            "handoff": torch.tensor(15, dtype=torch.long),
        })
    path = tmp_path / "train_shard_0000.pt"
    torch.save(shard, path)
    return path


def test_feature_dataset_length(cfg, tmp_path):
    path = _make_shard(cfg, tmp_path)
    ds = FeatureDataset([path], cfg)
    # 3 examples × cache_n_answer_positions=2 anchor positions each
    assert len(ds) == 3 * cfg.cache_n_answer_positions


def test_feature_dataset_item_shapes(cfg, tmp_path):
    path = _make_shard(cfg, tmp_path)
    ds = FeatureDataset([path], cfg)
    hiddens, ctx_ids, targets, valid_len = ds[0]
    assert hiddens.shape == (len(cfg.teacher_layers), cfg.d_teacher)
    assert ctx_ids.shape == (cfg.max_prompt_len,)
    assert targets.shape == (cfg.max_draft,)


def test_feature_dataset_dequantizes(cfg, tmp_path):
    path = _make_shard(cfg, tmp_path)
    ds = FeatureDataset([path], cfg)
    hiddens, _, _, _ = ds[0]
    assert hiddens.dtype == torch.bfloat16
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest mtp_draft/tests/test_data.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `mtp_draft/data.py`**

```python
# mtp_draft/data.py
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mtp_draft.config import MTPConfig


def build_prompt(
    example: dict,
    tokenizer,
    max_prompt_len: int,
) -> tuple[list[int], list[int]]:
    """
    Format a HotpotQA example into (prompt_ids, answer_ids).

    Prompt format:
        Question: {q}\n\nContext:\n{para_1}\n...\n{para_n}\n\nAnswer:

    Context paragraphs are truncated (last paragraphs dropped first) to fit
    within max_prompt_len. The question is always preserved.

    Returns:
        prompt_ids:  token ids for the full prompt (len <= max_prompt_len)
        answer_ids:  token ids for the answer string (no special tokens)
    """
    question = example["question"]
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    answer = example["answer"]

    q_prefix = f"Question: {question}\n\nContext:\n"
    a_suffix = "\n\nAnswer:"

    q_ids = tokenizer.encode(q_prefix)
    a_sep_ids = tokenizer.encode(a_suffix)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    budget = max_prompt_len - len(q_ids) - len(a_sep_ids) - 2

    ctx_ids: list[int] = []
    for title, sent_list in zip(titles, sentences):
        para = title + ": " + " ".join(sent_list) + "\n"
        para_ids = tokenizer.encode(para, add_special_tokens=False)
        if len(ctx_ids) + len(para_ids) > budget:
            break
        ctx_ids.extend(para_ids)

    prompt_ids = q_ids + ctx_ids + a_sep_ids
    return prompt_ids, answer_ids


class FeatureDataset(Dataset):
    """
    Dataset over pre-cached teacher features.

    Each item corresponds to one (example, anchor_position) pair.
    The anchor_position j ∈ {0, …, cache_n_answer_positions-1} indexes
    into the cached feature tensor for each example.

    Returns:
        hiddens:    (n_teacher_layers, d_teacher) bfloat16 — dequantised
        context_ids: (max_prompt_len,) long — left-padded with 0
        targets:    (max_draft,) long — tokens after anchor; -100 for padding
        valid_len:  scalar int — number of valid (non-padded) targets
    """

    def __init__(self, shard_paths: list[Path], cfg: MTPConfig) -> None:
        self.cfg = cfg
        self.items: list[tuple[dict, int]] = []
        for path in shard_paths:
            shard: list[dict] = torch.load(path, weights_only=False)
            for entry in shard:
                n_positions = entry["features_int8"].shape[0]
                for j in range(n_positions):
                    self.items.append((entry, j))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        entry, j = self.items[idx]
        cfg = self.cfg

        # Dequantise features
        f_int8 = entry["features_int8"][j].float()   # (n_layers, d_teacher)
        hiddens = (f_int8 * entry["scale"]).bfloat16()

        # Context: prompt tokens up to handoff + j, left-padded to max_prompt_len
        handoff: int = int(entry["handoff"].item())
        raw_ctx = entry["prompt_ids"][:handoff + j]
        L = len(raw_ctx)
        if L >= cfg.max_prompt_len:
            context_ids = raw_ctx[-cfg.max_prompt_len:]
        else:
            pad = torch.zeros(cfg.max_prompt_len - L, dtype=torch.long)
            context_ids = torch.cat([pad, raw_ctx])

        # Targets: tokens after anchor position in full sequence
        full_ids = torch.cat([entry["prompt_ids"], entry["answer_ids"]])
        start = handoff + j + 1
        raw_targets = full_ids[start:start + cfg.max_draft]
        valid_len = len(raw_targets)
        if valid_len < cfg.max_draft:
            pad = torch.full((cfg.max_draft - valid_len,), -100, dtype=torch.long)
            targets = torch.cat([raw_targets, pad])
        else:
            targets = raw_targets

        return hiddens, context_ids, targets, torch.tensor(valid_len, dtype=torch.long)


def get_dataloaders(cfg: MTPConfig) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from pre-cached shards.

    Expects shards named `train_shard_*.pt` and `validation_shard_*.pt`
    in `cfg.cache_dir`.
    """
    cache = Path(cfg.cache_dir)
    train_shards = sorted(cache.glob("train_shard_*.pt"))
    val_shards = sorted(cache.glob("validation_shard_*.pt"))

    train_ds = FeatureDataset(train_shards, cfg)
    val_ds = FeatureDataset(val_shards, cfg)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader
```

- [ ] **Step 4: Run tests**

```bash
pytest mtp_draft/tests/test_data.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mtp_draft/data.py mtp_draft/tests/test_data.py
git commit -m "feat(mtp): HotpotQA data pipeline — build_prompt, FeatureDataset, get_dataloaders"
```

---

## Task 8: Feature Extraction Cache

**Files:**
- Create: `mtp_draft/cache.py`
- Create: `mtp_draft/tests/test_cache.py`

**Interfaces:**
- Consumes: `build_prompt` from `mtp_draft.data`; `MTPConfig` from `mtp_draft.config`
- Produces:
  - `extract_and_cache(cfg: MTPConfig, split: str = "train") -> None`
    - writes int8 `.pt` shards to `cfg.cache_dir/{split}_shard_NNNN.pt`
  - `_quantise_int8(t: Tensor) -> tuple[Tensor, Tensor]` — returns `(int8_tensor, scale)`
  - `_dequantise_int8(t: Tensor, scale: Tensor) -> Tensor`

- [ ] **Step 1: Write the failing test**

```python
# mtp_draft/tests/test_cache.py
import torch, tempfile
from pathlib import Path
import pytest
from mtp_draft.cache import _quantise_int8, _dequantise_int8

def test_quantise_roundtrip():
    x = torch.randn(4, 2048)
    q, scale = _quantise_int8(x)
    assert q.dtype == torch.int8
    x_hat = _dequantise_int8(q, scale)
    # Max absolute error should be small relative to data range
    assert (x - x_hat).abs().max() < x.abs().max() * 0.02


def test_quantise_int8_range():
    x = torch.randn(4, 2048) * 10
    q, scale = _quantise_int8(x)
    assert q.abs().max() <= 127


def test_scale_positive():
    x = torch.randn(4, 2048)
    _, scale = _quantise_int8(x)
    assert scale > 0
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest mtp_draft/tests/test_cache.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `mtp_draft/cache.py`**

```python
# mtp_draft/cache.py
"""
Phase 1: extract teacher hidden states from HotpotQA and save int8 shards.

Usage:
    python -m mtp_draft.cache --config mtp_draft/configs/default.yaml --split train
    python -m mtp_draft.cache --config mtp_draft/configs/default.yaml --split validation
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from mtp_draft.config import MTPConfig, load_config
from mtp_draft.data import build_prompt

if TYPE_CHECKING:
    pass


def _quantise_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor int8 quantisation. Returns (int8_tensor, scale)."""
    scale = t.float().abs().max() / 127.0
    if scale == 0:
        scale = torch.tensor(1.0)
    q = (t.float() / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def _dequantise_int8(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of _quantise_int8. Returns float32 tensor."""
    return t.float() * scale


def extract_and_cache(cfg: MTPConfig, split: str = "train") -> None:
    """
    Load Gemma 4 E2b, run over HotpotQA `split`, extract hidden states at
    cfg.teacher_layers for up to cfg.cache_n_answer_positions per example,
    quantise to int8, and write sharded .pt files to cfg.cache_dir.

    Shard file format (list of dicts):
        {
            "features_int8": Tensor(cache_n_answer_positions, n_layers, d_teacher) int8,
            "scale":         Tensor scalar float32,
            "prompt_ids":    Tensor(prompt_len,) long,
            "answer_ids":    Tensor(answer_len,) long,
            "handoff":       int (index of last prompt token),
        }
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(cfg.cache_dir, exist_ok=True)

    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split)
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
        device_map="cuda",
    ).eval()

    # teacher_layers are 0-indexed layer numbers;
    # hidden_states[0] = embedding output, hidden_states[k+1] = layer k output
    layer_indices = [l + 1 for l in cfg.teacher_layers]

    shard_data: list[dict] = []
    shard_idx = 0
    cache_dir = Path(cfg.cache_dir)

    for example in tqdm(dataset, desc=f"Extracting {split}"):
        prompt_ids, answer_ids = build_prompt(example, tokenizer, cfg.max_prompt_len)
        if not prompt_ids or not answer_ids:
            continue

        handoff = len(prompt_ids) - 1  # index of last prompt token (0-based)
        n_pos = min(cfg.cache_n_answer_positions, len(answer_ids) + 1)

        # Positions to cache: handoff, handoff+1, ..., handoff+n_pos-1
        # (handoff = last prompt token; handoff+k = k-th answer token)
        full_ids = prompt_ids + list(answer_ids[:n_pos])
        input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1,) each (1, seq_len, d_model)
        all_hidden = outputs.hidden_states

        pos_features = []
        for k in range(n_pos):
            pos = handoff + k
            layer_feats = torch.stack([
                all_hidden[li][0, pos, :].float()
                for li in layer_indices
            ])  # (n_layers, d_teacher)
            pos_features.append(layer_feats)

        features = torch.stack(pos_features)  # (n_pos, n_layers, d_teacher)
        # Pad to cache_n_answer_positions if shorter
        if n_pos < cfg.cache_n_answer_positions:
            pad = torch.zeros(
                cfg.cache_n_answer_positions - n_pos,
                len(cfg.teacher_layers),
                cfg.d_teacher,
            )
            features = torch.cat([features, pad], dim=0)

        q_feat, scale = _quantise_int8(features)

        shard_data.append({
            "features_int8": q_feat.cpu(),
            "scale": scale.cpu(),
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "answer_ids": torch.tensor(list(answer_ids), dtype=torch.long),
            "handoff": torch.tensor(handoff, dtype=torch.long),
        })

        if len(shard_data) == cfg.cache_shard_size:
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
    parser.add_argument("--config", default="mtp_draft/configs/default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    extract_and_cache(cfg, split=args.split)
```

- [ ] **Step 4: Run quantisation unit tests (does not load teacher)**

```bash
pytest mtp_draft/tests/test_cache.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mtp_draft/cache.py mtp_draft/tests/test_cache.py
git commit -m "feat(mtp): feature extraction cache — int8 shards from Gemma hidden states"
```

---

## Task 9: Tree Construction

**Files:**
- Create: `mtp_draft/tree.py`
- Create: `mtp_draft/tests/test_tree.py`

**Interfaces:**
- Produces:
  - `build_tree(logits: Tensor, tau: float, max_tree_nodes: int = 256) -> list[list[int]]`
    - `logits`: `(max_draft, vocab)` — raw logits for each draft position
    - returns: list of candidate token sequences (variable-length lists), sorted descending by cumulative log-prob

- [ ] **Step 1: Write the failing test**

```python
# mtp_draft/tests/test_tree.py
import torch
import pytest
from mtp_draft.tree import build_tree

VOCAB, MAX_DRAFT = 100, 4


def _make_peaked_logits(top_token: int, secondary_token: int, gap: float = 5.0) -> torch.Tensor:
    """Logits with one clear top token and one secondary token gap nats below."""
    logits = torch.full((MAX_DRAFT, VOCAB), -100.0)
    logits[:, top_token] = 0.0
    logits[:, secondary_token] = -gap
    return logits


def test_single_path_when_tau_zero():
    """tau=0.0 → only the top token at each position → exactly 1 path."""
    logits = _make_peaked_logits(top_token=5, secondary_token=10)
    paths = build_tree(logits, tau=0.0)
    assert len(paths) == 1
    assert all(t == 5 for t in paths[0])


def test_two_candidates_per_position_when_tau_large():
    """With gap=2 and tau=3, both tokens qualify at every position."""
    logits = _make_peaked_logits(top_token=5, secondary_token=10, gap=2.0)
    paths = build_tree(logits, tau=3.0, max_tree_nodes=10000)
    # 2 candidates × 4 positions → 2^4 = 16 paths
    assert len(paths) == 16


def test_paths_sorted_by_score():
    """Paths must be returned highest cumulative log-prob first."""
    logits = _make_peaked_logits(top_token=5, secondary_token=10, gap=2.0)
    paths = build_tree(logits, tau=3.0, max_tree_nodes=10000)
    # First path: all top tokens
    assert paths[0] == [5] * MAX_DRAFT


def test_max_tree_nodes_respected():
    """Output must not exceed max_tree_nodes paths."""
    logits = torch.randn(MAX_DRAFT, VOCAB)
    paths = build_tree(logits, tau=100.0, max_tree_nodes=10)
    assert len(paths) <= 10


def test_empty_logits_returns_one_path():
    """Even with very negative logits, at least one path (the top tokens) is returned."""
    logits = torch.full((MAX_DRAFT, VOCAB), -1e9)
    logits[:, 0] = 0.0
    paths = build_tree(logits, tau=0.5)
    assert len(paths) >= 1
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest mtp_draft/tests/test_tree.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `mtp_draft/tree.py`**

```python
# mtp_draft/tree.py
from __future__ import annotations
import torch


def build_tree(
    logits: torch.Tensor,
    tau: float,
    max_tree_nodes: int = 256,
) -> list[list[int]]:
    """
    Construct speculative decoding candidate tree from draft logits.

    For each draft position i, selects candidate tokens whose log-probability
    is within tau nats of the top token. The full candidate tree is the
    Cartesian product of per-position candidate sets, pruned to max_tree_nodes
    paths by cumulative log-probability (highest first).

    Args:
        logits:         (max_draft, vocab) raw logits
        tau:            relative threshold in log-prob space (nats)
        max_tree_nodes: maximum number of candidate paths to return

    Returns:
        List of candidate token sequences (list[int]), sorted highest
        cumulative log-prob first. Each sequence has length max_draft.
    """
    max_draft = logits.shape[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)   # (max_draft, vocab)

    # paths: list of (cumulative_log_prob, token_sequence)
    paths: list[tuple[float, list[int]]] = [(0.0, [])]

    for i in range(max_draft):
        lp_i = log_probs[i]
        top_lp = lp_i.max().item()
        threshold = top_lp - tau
        # Always include the top token even if tau=0
        mask = lp_i >= threshold
        candidates = mask.nonzero(as_tuple=True)[0].tolist()
        if not candidates:
            candidates = [int(lp_i.argmax().item())]

        new_paths: list[tuple[float, list[int]]] = []
        for cum_lp, tokens in paths:
            for tok in candidates:
                new_lp = cum_lp + lp_i[tok].item()
                new_paths.append((new_lp, tokens + [tok]))

        # Prune to max_tree_nodes by score
        new_paths.sort(key=lambda x: -x[0])
        paths = new_paths[:max_tree_nodes]

    return [tokens for _, tokens in paths]
```

- [ ] **Step 4: Run tests**

```bash
pytest mtp_draft/tests/test_tree.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mtp_draft/tree.py mtp_draft/tests/test_tree.py
git commit -m "feat(mtp): build_tree — relative log-prob threshold candidate tree"
```

---

## Task 10: Training Loop

**Files:**
- Create: `mtp_draft/train.py`

**Interfaces:**
- Consumes:
  - `MTPDraftModel` from `mtp_draft.models.draft_model`
  - `get_dataloaders` from `mtp_draft.data`
  - `MTPConfig`, `load_config` from `mtp_draft.config`
- Produces:
  - `train(cfg: MTPConfig) -> None` — runs full training, saves checkpoint
  - `make_lr_lambda(warmup_steps: int, total_steps: int)` — cosine with warmup

No unit tests for the training loop itself; a smoke test is included that runs
2 steps with tiny synthetic data to verify the backward pass doesn't error.

- [ ] **Step 1: Write the smoke test**

```python
# mtp_draft/tests/test_train_smoke.py
import torch
import pytest
from mtp_draft.config import MTPConfig
from mtp_draft.models.draft_model import MTPDraftModel
from mtp_draft.train import training_step

VOCAB, D_TEACHER = 100, 32
B = 2


@pytest.fixture
def cfg():
    return MTPConfig(
        d_draft=32, n_blocks=1, ffn_hidden=64, n_heads=2,
        d_teacher=D_TEACHER, max_draft=3,
        teacher_layers=[0, 1, 2, 3], lora_rank=2, dropout=0.0,
    )


@pytest.fixture
def model(cfg):
    emb_w = torch.randn(VOCAB, D_TEACHER)
    lm_w = torch.randn(VOCAB, D_TEACHER)
    return MTPDraftModel(cfg, emb_w, lm_w)


def test_training_step_returns_scalar(model, cfg):
    hiddens = torch.randn(B, len(cfg.teacher_layers), cfg.d_teacher)
    ctx_ids = torch.randint(0, VOCAB, (B, cfg.max_prompt_len))
    targets = torch.randint(0, VOCAB, (B, cfg.max_draft))
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    loss = training_step(model, hiddens, ctx_ids, targets, cfg, opt)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_training_step_updates_params(model, cfg):
    hiddens = torch.randn(B, len(cfg.teacher_layers), cfg.d_teacher)
    ctx_ids = torch.randint(0, VOCAB, (B, cfg.max_prompt_len))
    targets = torch.randint(0, VOCAB, (B, cfg.max_draft))
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)

    params_before = [p.clone() for p in model.trainable_parameters()]
    training_step(model, hiddens, ctx_ids, targets, cfg, opt)
    params_after = list(model.trainable_parameters())

    any_changed = any(
        not torch.allclose(b, a)
        for b, a in zip(params_before, params_after)
    )
    assert any_changed
```

- [ ] **Step 2: Run smoke test to confirm failure**

```bash
pytest mtp_draft/tests/test_train_smoke.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `mtp_draft/train.py`**

```python
# mtp_draft/train.py
"""
Phase 2: train the MTP draft model on pre-cached HotpotQA features.

Usage:
    python -m mtp_draft.train --config mtp_draft/configs/default.yaml
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from mtp_draft.config import MTPConfig, load_config
from mtp_draft.data import get_dataloaders
from mtp_draft.models.draft_model import MTPDraftModel


def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def training_step(
    model: MTPDraftModel,
    hiddens: torch.Tensor,
    context_ids: torch.Tensor,
    targets: torch.Tensor,
    cfg: MTPConfig,
    optimizer: AdamW,
) -> torch.Tensor:
    """
    Single training step. Computes distance-weighted cross-entropy loss,
    back-propagates, clips gradients, and steps the optimizer.

    hiddens:     (B, n_layers, d_teacher)
    context_ids: (B, max_prompt_len)
    targets:     (B, max_draft) — -100 for positions beyond the answer
    Returns scalar loss tensor.
    """
    model.train()
    logits = model(hiddens, context_ids)   # (B, max_draft, vocab)

    loss = torch.tensor(0.0, device=logits.device)
    for i in range(cfg.max_draft):
        weight = cfg.lambda_decay ** i
        loss_i = F.cross_entropy(logits[:, i, :], targets[:, i], ignore_index=-100)
        loss = loss + weight * loss_i
    loss = loss / cfg.max_draft

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return loss.detach()


def train(cfg: MTPConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen teacher weights (CPU, no grad)
    print("Loading teacher weights for embedding and LM head...")
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id, torch_dtype=torch.bfloat16
    )
    teacher_embed_w = teacher.model.embed_tokens.weight.detach().clone()
    teacher_lm_head_w = teacher.lm_head.weight.detach().clone()
    del teacher
    print("Teacher weights extracted, model unloaded.")

    model = MTPDraftModel(cfg, teacher_embed_w, teacher_lm_head_w).to(device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"Trainable parameters: {n_trainable:,}")

    train_loader, val_loader = get_dataloaders(cfg)
    total_steps = cfg.n_epochs * len(train_loader)

    optimizer = AdamW(
        model.trainable_parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = LambdaLR(optimizer, make_lr_lambda(cfg.warmup_steps, total_steps))

    best_val_loss = float("inf")
    ckpt_dir = Path(cfg.cache_dir).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.n_epochs):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for hiddens, ctx_ids, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            hiddens = hiddens.to(device)
            ctx_ids = ctx_ids.to(device)
            targets = targets.to(device)

            loss = training_step(model, hiddens, ctx_ids, targets, cfg, optimizer)
            scheduler.step()
            total_train_loss += loss.item()
            n_batches += 1

        avg_train = total_train_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for hiddens, ctx_ids, targets, _ in val_loader:
                hiddens = hiddens.to(device)
                ctx_ids = ctx_ids.to(device)
                targets = targets.to(device)
                logits = model(hiddens, ctx_ids)
                for i in range(cfg.max_draft):
                    w = cfg.lambda_decay ** i
                    val_loss += w * F.cross_entropy(
                        logits[:, i, :], targets[:, i], ignore_index=-100
                    ).item()
                n_val += 1
        avg_val = val_loss / max(1, n_val * cfg.max_draft)

        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt = {
                "epoch": epoch,
                "model_state": {
                    k: v for k, v in model.state_dict().items()
                    if "token_embedding" not in k and "lm_head.weight" not in k
                },
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Checkpoint saved (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mtp_draft/configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
```

- [ ] **Step 4: Run smoke tests**

```bash
pytest mtp_draft/tests/test_train_smoke.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest mtp_draft/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add mtp_draft/train.py mtp_draft/tests/test_train_smoke.py
git commit -m "feat(mtp): training loop — distance-weighted MTP loss, cosine LR, checkpoint"
```

---

## Execution Order

Run phases in sequence:

```bash
# Phase 1: extract features (needs GPU + Gemma downloaded, ~2 hours)
python -m mtp_draft.cache --config mtp_draft/configs/default.yaml --split train
python -m mtp_draft.cache --config mtp_draft/configs/default.yaml --split validation

# Phase 2: train draft model (~GPU hours, teacher not loaded)
python -m mtp_draft.train --config mtp_draft/configs/default.yaml
```
