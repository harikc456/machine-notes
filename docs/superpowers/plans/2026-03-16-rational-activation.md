# Rational Activation Experiment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `RationalFFN` model variant (learnable rational activation replacing SwiGLU) to the existing WikiText-103 ablation infrastructure and wire it up for comparison against the SwiGLU baseline.

**Architecture:** A new `rational_ffn.py` defines `RationalActivation` (element-wise P(x)/Q(x) with learnable `a`/`b` parameters) and `RationalFFN` (up_proj → act → down_proj). `RationalBlock` in `transformer_block.py` slots this FFN into the pre-norm residual structure. `CausalLM` in `model.py` gains a dict-based block dispatch for `model_type="rational"`. A new YAML config enables the run.

**Tech Stack:** PyTorch 2.10+, existing `rbf_ffn` package, pytest, tiktoken

---

## Chunk 1: rational_ffn.py — RationalActivation and RationalFFN

### Task 1: RationalActivation and RationalFFN

**Files:**
- Create: `rbf_ffn/models/rational_ffn.py`
- Create: `rbf_ffn/tests/test_rational_ffn.py`

- [ ] **Step 1: Write failing tests for RationalActivation**

Create `rbf_ffn/tests/test_rational_ffn.py` with this exact content:

```python
# rbf_ffn/tests/test_rational_ffn.py
import torch
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.rational_ffn import RationalActivation, RationalFFN

B, N, D = 2, 16, 32


def make_cfg():
    # model_type="rational" is a valid dataclass value (no runtime validation);
    # CausalLM dispatch for this value is wired in Chunk 2.
    return RBFFFNConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="rational",
    )


def test_rational_activation_shape():
    act = RationalActivation()
    x = torch.randn(B, N, D)
    assert act(x).shape == (B, N, D)


def test_rational_activation_gradients():
    act = RationalActivation()
    x = torch.randn(B, N, D)
    act(x).sum().backward()
    assert act.a.grad is not None
    assert act.b.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest rbf_ffn/tests/test_rational_ffn.py -v
```

Expected: `ImportError: cannot import name 'RationalActivation' from 'rbf_ffn.models.rational_ffn'` (module does not exist yet).

- [ ] **Step 3: Create rational_ffn.py with RationalActivation**

Create `rbf_ffn/models/rational_ffn.py` with this exact content:

```python
# rbf_ffn/models/rational_ffn.py
import torch
import torch.nn as nn

from rbf_ffn.config import RBFFFNConfig


class RationalActivation(nn.Module):
    """
    Learnable rational activation f(x) = P(x) / Q(x).

    P(x) = a0 + a1·x + a2·x² + a3·x³  (Horner's method)
    Q(x) = 1 + |x·(b0 + x·b1)|

    Applied element-wise; a and b are shared across all positions and channels.
    Q(x) >= 1 always — division is numerically safe.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([0.4401, 0.5, 0.507, 0.05]))
        self.b = nn.Parameter(torch.tensor([0.0, 0.01]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_x = self.a[0] + x * (self.a[1] + x * (self.a[2] + x * self.a[3]))
        q_x = 1.0 + torch.abs(x * (self.b[0] + x * self.b[1]))
        return p_x / q_x
```

- [ ] **Step 4: Run RationalActivation tests to verify they pass**

```bash
pytest rbf_ffn/tests/test_rational_ffn.py::test_rational_activation_shape \
       rbf_ffn/tests/test_rational_ffn.py::test_rational_activation_gradients -v
```

Expected: 2 passed.

- [ ] **Step 5: Write failing tests for RationalFFN**

Append to `rbf_ffn/tests/test_rational_ffn.py`:

```python
def test_rational_ffn_shape():
    ffn = RationalFFN(make_cfg())
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_rational_ffn_no_bias():
    ffn = RationalFFN(make_cfg())
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None
```

- [ ] **Step 6: Run RationalFFN tests to verify they fail**

```bash
pytest rbf_ffn/tests/test_rational_ffn.py::test_rational_ffn_shape \
       rbf_ffn/tests/test_rational_ffn.py::test_rational_ffn_no_bias -v
```

Expected: `ImportError: cannot import name 'RationalFFN'` — class not yet defined.

- [ ] **Step 7: Add RationalFFN to rational_ffn.py**

Append to `rbf_ffn/models/rational_ffn.py`:

```python

class RationalFFN(nn.Module):
    """
    Feed-forward network with learnable rational activation.

        up_proj → RationalActivation → down_proj

    No bias on projections (Llama convention).
    Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.up_proj   = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = RationalActivation()
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))
```

- [ ] **Step 8: Run all tests in test_rational_ffn.py to verify they pass**

```bash
pytest rbf_ffn/tests/test_rational_ffn.py -v
```

Expected: 4 passed. (`test_rational_block_shape` is not yet in the file.)

- [ ] **Step 9: Run the full test suite to check for regressions**

```bash
pytest rbf_ffn/tests/ -v
```

Expected: all pre-existing tests pass; the 4 new tests also pass.

- [ ] **Step 10: Commit**

```bash
git add rbf_ffn/models/rational_ffn.py rbf_ffn/tests/test_rational_ffn.py
git commit -m "feat: add RationalActivation and RationalFFN"
```

---

## Chunk 2: RationalBlock, CausalLM integration, and config

> **Prerequisite:** Chunk 1 must be complete — `rbf_ffn/models/rational_ffn.py` and `rbf_ffn/tests/test_rational_ffn.py` must exist and all 4 Chunk 1 tests must pass before starting this chunk.

### Task 2: RationalBlock

**Files:**
- Modify: `rbf_ffn/models/transformer_block.py`
- Modify: `rbf_ffn/tests/test_rational_ffn.py`

- [ ] **Step 1: Write failing test for RationalBlock**

Add the `RationalBlock` import and test to `rbf_ffn/tests/test_rational_ffn.py`.

**Add** this import after the existing imports at the top of the file:

```python
from rbf_ffn.models.transformer_block import RationalBlock
```

**Append** this test at the bottom of the file:

```python
def test_rational_block_shape():
    block = RationalBlock(make_cfg())
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest rbf_ffn/tests/test_rational_ffn.py::test_rational_block_shape -v
```

Expected: `ImportError: cannot import name 'RationalBlock' from 'rbf_ffn.models.transformer_block'`.

- [ ] **Step 3: Add RationalFFN import and RationalBlock class to transformer_block.py**

In `rbf_ffn/models/transformer_block.py`:

**Add** this import after the existing `from rbf_ffn.models.rbf_ffn import RBFFFN` line:

```python
from rbf_ffn.models.rational_ffn import RationalFFN
```

**Append** this class at the end of the file:

```python

class RationalBlock(nn.Module):
    """
    Transformer block with RationalFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is RationalFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = RationalFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

- [ ] **Step 4: Run all tests in test_rational_ffn.py to verify they pass**

```bash
pytest rbf_ffn/tests/test_rational_ffn.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/models/transformer_block.py rbf_ffn/tests/test_rational_ffn.py
git commit -m "feat: add RationalBlock to transformer_block"
```

---

### Task 3: CausalLM dict dispatch

**Files:**
- Modify: `rbf_ffn/models/model.py`
- Modify: `rbf_ffn/tests/test_model.py`

- [ ] **Step 1: Write failing test for rational optimizer groups**

Append to `rbf_ffn/tests/test_model.py`:

```python
def test_rational_params_in_adamw():
    """RationalActivation a and b must be in AdamW (1-D), not Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("rational")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.act.a) in adamw_ids,    "act.a should be in AdamW"
        assert id(block.ffn.act.a) not in muon_ids, "act.a should not be in Muon"
        assert id(block.ffn.act.b) in adamw_ids,    "act.b should be in AdamW"
        assert id(block.ffn.act.b) not in muon_ids, "act.b should not be in Muon"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest rbf_ffn/tests/test_model.py::test_rational_params_in_adamw -v
```

Expected: `AttributeError: 'RBFFFN' object has no attribute 'act'` — `make_model("rational")` still routes to `RBFBlock` via the old `if/else`.

- [ ] **Step 3: Update model.py — import**

In `rbf_ffn/models/model.py`, find the import line:

```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock
```

Replace it with:

```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock
```

- [ ] **Step 4: Update model.py — dict dispatch**

In `rbf_ffn/models/model.py`, find line 59:

```python
        BlockClass = LlamaBlock if cfg.model_type == "baseline" else RBFBlock
```

Replace it with:

```python
        BlockClass = {
            "baseline": LlamaBlock,
            "rbf":      RBFBlock,
            "rational": RationalBlock,
        }[cfg.model_type]
```

Note: this dict raises `KeyError` for unrecognised `model_type` values — intentional, fails loudly instead of silently routing to the wrong block.

- [ ] **Step 5: Update model.py — CausalLM docstring**

In `rbf_ffn/models/model.py`, find the `CausalLM` docstring lines:

```
    Block type is selected by cfg.model_type:
        "baseline" → LlamaBlock (SwiGLU FFN)
        "rbf"      → RBFBlock   (RBF-FFN)
```

Replace with:

```
    Block type is selected by cfg.model_type:
        "baseline" → LlamaBlock  (SwiGLU FFN)
        "rbf"      → RBFBlock    (RBF-FFN)
        "rational" → RationalBlock (RationalFFN)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest rbf_ffn/tests/test_model.py::test_rational_params_in_adamw -v
```

Expected: PASS.

- [ ] **Step 7: Run the full test suite to check for regressions**

```bash
pytest rbf_ffn/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add rbf_ffn/models/model.py rbf_ffn/tests/test_model.py
git commit -m "feat: add rational model_type to CausalLM block dispatch"
```

---

### Task 4: Config wiring

**Files:**
- Modify: `rbf_ffn/config.py`
- Create: `rbf_ffn/configs/rational_ffn.yaml`

- [ ] **Step 1: Update config.py comments**

In `rbf_ffn/config.py`, find:

```python
    model_type: str = "rbf"        # "baseline" | "rbf"
    ffn_hidden: int = 688          # SwiGLU hidden dim; ignored by RBF model
```

Replace with:

```python
    model_type: str = "rbf"        # "baseline" | "rbf" | "rational"
    ffn_hidden: int = 688          # FFN hidden dim (SwiGLU / RationalFFN); ignored by RBF model
```

- [ ] **Step 2: Create rational_ffn.yaml**

Create `rbf_ffn/configs/rational_ffn.yaml` with this exact content:

```yaml
# Rational activation ablation — gate_variant, sigma_variant, K, centers,
# sigma_init, and sinkhorn_iters are inert for model_type: rational but
# present for a uniform config schema across all runs.
model_type: rational
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
n_epochs: 3
batch_size: 16
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
grad_accum_steps: 1
```

- [ ] **Step 3: Verify config loads correctly**

```bash
python -c "
from rbf_ffn.config import load_config
cfg = load_config('rbf_ffn/configs/rational_ffn.yaml')
print(cfg.model_type, cfg.ffn_hidden, cfg.grad_accum_steps)
"
```

Expected output: `rational 688 1`

- [ ] **Step 4: Run the full test suite one final time**

```bash
pytest rbf_ffn/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/config.py rbf_ffn/configs/rational_ffn.yaml
git commit -m "feat: add rational_ffn.yaml config and update config.py comments"
```
