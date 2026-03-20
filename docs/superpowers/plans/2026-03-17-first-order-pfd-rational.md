# FirstOrderPFDRationalFFN Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `FirstOrderPFDRationalFFN` — a parameter-efficient gated FFN using a shared projection and PFD rational activation with a phase-shift vector — as a new `model_type: "first_order_pfd_rational"` variant.

**Architecture:** Single `up_proj` produces `u`; gate is `PFDRationalActivation(sin(u + phi))` where `phi` is a learnable vector; output is `down_proj(gate * u)`. Two matrices instead of three (~33% fewer FFN params vs SwiGLU).

**Tech Stack:** PyTorch, pytest, YAML config

---

## Chunk 1: FFN class + unit tests

### Task 1: Write failing unit tests for `FirstOrderPFDRationalFFN`

**Files:**
- Create: `rbf_ffn/tests/test_first_order_pfd_rational_ffn.py`

- [ ] **Step 1: Create the test file**

```python
# rbf_ffn/tests/test_first_order_pfd_rational_ffn.py
import torch
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.rational_ffn import FirstOrderPFDRationalFFN

B, N, D = 2, 16, 32


def make_cfg():
    return RBFFFNConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="first_order_pfd_rational", pfd_n=4,
    )


def test_ffn_output_shape():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_ffn_no_bias():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_phi_receives_gradient():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    x = torch.randn(B, N, D)
    ffn(x).sum().backward()
    assert ffn.phi.grad is not None


def test_input_gradient_flows():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    x = torch.randn(B, N, D, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None


def test_pfd_act_receives_gradient():
    ffn = FirstOrderPFDRationalFFN(make_cfg())
    x = torch.randn(B, N, D)
    ffn(x).sum().backward()
    assert ffn.act.a.grad is not None
```

- [ ] **Step 2: Run tests to confirm they fail (import error expected)**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_first_order_pfd_rational_ffn.py -v
```

Expected: `ImportError: cannot import name 'FirstOrderPFDRationalFFN'`

---

### Task 2: Implement `FirstOrderPFDRationalFFN`

**Files:**
- Modify: `rbf_ffn/models/rational_ffn.py` (append after line 135, end of file)

- [ ] **Step 1: Add the class to `rational_ffn.py`**

Append after the final line of `PFDRationalGatedFFN`:

```python


class FirstOrderPFDRationalFFN(nn.Module):
    """
    First-order gated FFN with PFD rational activation and shared projection.

        u    = up_proj(x)
        gate = PFDRationalActivation(sin(u + phi))
        out  = down_proj(gate * u)

    phi is a learnable vector of shape (ffn_hidden,) — phase shift that decouples
    the gate signal from the value despite sharing the same projection u.

    2 large matrices instead of 3 (no gate_proj) — ~33% fewer FFN params vs SwiGLU.
    No bias (Llama convention). Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: RBFFFNConfig, n: int = 4):
        super().__init__()
        self.up_proj   = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.phi       = nn.Parameter(torch.randn(cfg.ffn_hidden) * 0.02)
        self.act       = PFDRationalActivation(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up_proj(x)
        gate = self.act(torch.sin(u + self.phi))
        return self.down_proj(gate * u)
```

- [ ] **Step 2: Run tests to confirm they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_first_order_pfd_rational_ffn.py -v
```

Expected: 5 passed

- [ ] **Step 3: Commit**

```bash
git add rbf_ffn/models/rational_ffn.py rbf_ffn/tests/test_first_order_pfd_rational_ffn.py
git commit -m "feat: add FirstOrderPFDRationalFFN with PFD rational activation and phi phase shift"
```

---

## Chunk 2: Block, config, model dispatch

### Task 3: Write failing block-level tests

**Files:**
- Modify: `rbf_ffn/tests/test_transformer_block.py`

- [ ] **Step 1: Append to `test_transformer_block.py`**

Add to the **import line** (line 5) — extend the existing import:
```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock, FirstOrderPFDRationalBlock
```

Then append the following tests at the end of the file:

```python

# ── FirstOrderPFDRationalBlock ────────────────────────────────────────────────

def make_first_order_pfd_cfg():
    return RBFFFNConfig(d_model=D, n_heads=H, dropout=0.0,
                        model_type="first_order_pfd_rational", pfd_n=4)


def test_first_order_pfd_rational_block_shape():
    block = FirstOrderPFDRationalBlock(make_first_order_pfd_cfg())
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_first_order_pfd_rational_block_gradient_flow():
    block = FirstOrderPFDRationalBlock(make_first_order_pfd_cfg())
    x = torch.randn(B, N, D)
    block(x).sum().backward()
    assert block.ffn.phi.grad is not None


def test_first_order_pfd_rational_block_residual_connection():
    """Zero FFN and attn output projections → output equals input."""
    block = FirstOrderPFDRationalBlock(make_first_order_pfd_cfg())
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)
```

- [ ] **Step 2: Run to confirm they fail (import error expected)**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py::test_first_order_pfd_rational_block_shape -v
```

Expected: `ImportError: cannot import name 'FirstOrderPFDRationalBlock'`

---

### Task 4: Implement `FirstOrderPFDRationalBlock` and wire up config + model dispatch

**Files:**
- Modify: `rbf_ffn/models/transformer_block.py`
- Modify: `rbf_ffn/config.py`
- Modify: `rbf_ffn/models/model.py`

- [ ] **Step 1: Update import in `transformer_block.py` (line 8)**

Replace:
```python
from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN, PFDRationalFFN, PFDRationalGatedFFN
```
With:
```python
from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN, PFDRationalFFN, PFDRationalGatedFFN, FirstOrderPFDRationalFFN
```

- [ ] **Step 2: Append `FirstOrderPFDRationalBlock` to `transformer_block.py`**

Append after the final line of `PFDRationalGLUBlock`:

```python


class FirstOrderPFDRationalBlock(nn.Module):
    """
    Transformer block with FirstOrderPFDRationalFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is FirstOrderPFDRationalFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = FirstOrderPFDRationalFFN(cfg, n=cfg.pfd_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

- [ ] **Step 3: Update `config.py` model_type comment (line 30)**

Replace:
```python
    model_type: str = "rbf"        # "baseline" | "rbf" | "rational" | "rationalglu" | "pfd_rational" | "pfd_rationalglu"
```
With:
```python
    model_type: str = "rbf"        # "baseline" | "rbf" | "rational" | "rationalglu" | "pfd_rational" | "pfd_rationalglu" | "first_order_pfd_rational"
```

- [ ] **Step 4: Update import in `model.py` (line 6)**

Replace:
```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock, PFDRationalBlock, PFDRationalGLUBlock
```
With:
```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock, PFDRationalBlock, PFDRationalGLUBlock, FirstOrderPFDRationalBlock
```

- [ ] **Step 5: Update `CausalLM` docstring in `model.py`**

In the docstring, the block dispatch list currently ends with:
```
        "pfd_rationalglu"→ PFDRationalGLUBlock (PFDRationalGatedFFN)
```
Add a new line after it:
```
        "first_order_pfd_rational" → FirstOrderPFDRationalBlock (FirstOrderPFDRationalFFN)
```

- [ ] **Step 6: Add dispatch entry to `BlockClass` dict in `model.py`**

In `CausalLM.__init__`, the `BlockClass` dict currently ends with:
```python
            "pfd_rationalglu": PFDRationalGLUBlock,
```
Add after it:
```python
            "first_order_pfd_rational": FirstOrderPFDRationalBlock,
```

- [ ] **Step 7: Run block tests to confirm they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -v -k "first_order"
```

Expected: 3 passed

- [ ] **Step 8: Run full test suite to confirm no regressions**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/ -v --ignore=rbf_ffn/tests/test_train.py
```

Expected: all previously passing tests still pass

- [ ] **Step 9: Commit**

```bash
git add rbf_ffn/models/transformer_block.py rbf_ffn/config.py rbf_ffn/models/model.py rbf_ffn/tests/test_transformer_block.py
git commit -m "feat: add FirstOrderPFDRationalBlock and wire up config/model dispatch"
```

---

## Chunk 3: Model-level tests + YAML config

### Task 5: Write and run model-level tests

**Files:**
- Modify: `rbf_ffn/tests/test_model.py`

- [ ] **Step 1: Append to `test_model.py`**

```python


def test_first_order_pfd_rational_output_shape():
    model = make_model("first_order_pfd_rational")
    tokens = torch.randint(0, VOCAB, (B, N))
    assert model(tokens).shape == (B, N, VOCAB)


def test_first_order_pfd_rational_params_in_adamw():
    """phi and PFD activation params must be in AdamW, not Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("first_order_pfd_rational")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.phi)       in adamw_ids,    "phi should be in AdamW"
        assert id(block.ffn.phi)       not in muon_ids, "phi should not be in Muon"
        assert id(block.ffn.act.a)     in adamw_ids,    "act.a should be in AdamW"
        assert id(block.ffn.act.a)     not in muon_ids, "act.a should not be in Muon"
        assert id(block.ffn.act.b)     in adamw_ids,    "act.b should be in AdamW"
        assert id(block.ffn.act.b)     not in muon_ids, "act.b should not be in Muon"
        assert id(block.ffn.act.c)     in adamw_ids,    "act.c should be in AdamW"
        assert id(block.ffn.act.c)     not in muon_ids, "act.c should not be in Muon"
        assert id(block.ffn.act.gamma) in adamw_ids,    "act.gamma should be in AdamW"
        assert id(block.ffn.act.gamma) not in muon_ids, "act.gamma should not be in Muon"
```

- [ ] **Step 2: Run new model tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_model.py -v -k "first_order"
```

Expected: 2 passed

- [ ] **Step 3: Run full test suite**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/ -v --ignore=rbf_ffn/tests/test_train.py
```

Expected: all tests pass

---

### Task 6: Add YAML config

**Files:**
- Create: `rbf_ffn/configs/first_order_pfd_rational_ffn.yaml`

- [ ] **Step 1: Create the config file**

```yaml
# FirstOrderPFDRational ablation — gate_variant, sigma_variant, K, centers,
# sigma_init, and sinkhorn_iters are inert for model_type: first_order_pfd_rational
# but present for a uniform config schema across all runs.
model_type: first_order_pfd_rational
gate_variant: G0
sigma_variant: global
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
pfd_n: 4
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

- [ ] **Step 2: Verify config loads without error**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -c "
from rbf_ffn.config import load_config
cfg = load_config('rbf_ffn/configs/first_order_pfd_rational_ffn.yaml')
print('model_type:', cfg.model_type)
print('pfd_n:', cfg.pfd_n)
print('ffn_hidden:', cfg.ffn_hidden)
"
```

Expected output:
```
model_type: first_order_pfd_rational
pfd_n: 4
ffn_hidden: 688
```

- [ ] **Step 3: Commit**

```bash
git add rbf_ffn/tests/test_model.py rbf_ffn/configs/first_order_pfd_rational_ffn.yaml
git commit -m "feat: add model tests and YAML config for first_order_pfd_rational"
```
