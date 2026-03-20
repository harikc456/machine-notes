# RationalGLU Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `RationalGatedFFN` variant (`down(RationalAct(gate) * up)`) to the WikiText-103 ablation suite as `model_type="rationalglu"`, parameter-matched to SwiGLU.

**Architecture:** `RationalGatedFFN` and `RationalGLUBlock` are appended to the existing `rational_ffn.py` and `transformer_block.py` files. `model.py` gains one dispatch entry. A new `rationalglu_ffn.yaml` config enables training runs.

**Tech Stack:** Python, PyTorch, pytest, YAML

---

## Chunk 1: FFN and Block

### Task 1: RationalGatedFFN

**Files:**
- Modify: `rbf_ffn/models/rational_ffn.py` — append `RationalGatedFFN`
- Modify: `rbf_ffn/tests/test_rational_ffn.py` — update import, add `make_gated_cfg()`, add 3 tests

**Spec:** `docs/superpowers/specs/2026-03-16-rationalglu-design.md`

---

- [ ] **Step 1: Write failing tests**

In `rbf_ffn/tests/test_rational_ffn.py`, update line 4:

```python
from rbf_ffn.models.rational_ffn import RationalActivation, RationalFFN, RationalGatedFFN
```

Then append after the last existing test:

```python
def make_gated_cfg():
    return RBFFFNConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="rationalglu",
    )


def test_rational_gated_ffn_shape():
    ffn = RationalGatedFFN(make_gated_cfg())
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_rational_gated_ffn_no_bias():
    ffn = RationalGatedFFN(make_gated_cfg())
    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_rational_gated_ffn_gate_gradient():
    ffn = RationalGatedFFN(make_gated_cfg())
    x = torch.randn(B, N, D)
    ffn(x).sum().backward()
    assert ffn.act.a.grad is not None
    assert ffn.act.b.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes && pytest rbf_ffn/tests/test_rational_ffn.py::test_rational_gated_ffn_shape rbf_ffn/tests/test_rational_ffn.py::test_rational_gated_ffn_no_bias rbf_ffn/tests/test_rational_ffn.py::test_rational_gated_ffn_gate_gradient -v
```

Expected: `ImportError: cannot import name 'RationalGatedFFN' from 'rbf_ffn.models.rational_ffn'`

- [ ] **Step 3: Implement RationalGatedFFN**

In `rbf_ffn/models/rational_ffn.py`, append after the `RationalFFN` class:

```python
class RationalGatedFFN(nn.Module):
    """
    Gated FFN with learnable rational activation replacing SiLU.

        gate = RationalActivation(gate_proj(x))
        out  = down_proj(gate * up_proj(x))

    Matches SwiGLU parameter count at ffn_hidden=688.
    No bias (Llama convention). Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj   = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = RationalActivation()
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes && pytest rbf_ffn/tests/test_rational_ffn.py -v
```

Expected: all tests pass (8 total including the 3 new ones).

- [ ] **Step 5: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes && git add rbf_ffn/models/rational_ffn.py rbf_ffn/tests/test_rational_ffn.py && git commit -m "feat: add RationalGatedFFN with gate gradient tests"
```

---

### Task 2: RationalGLUBlock

**Files:**
- Modify: `rbf_ffn/models/transformer_block.py` — update import, append `RationalGLUBlock`
- Modify: `rbf_ffn/tests/test_transformer_block.py` — update import, add `make_rationalglu()`, add 3 tests

---

- [ ] **Step 1: Write failing tests**

In `rbf_ffn/tests/test_transformer_block.py`, update line 5:

```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock
```

Then append a `make_rationalglu()` helper and 3 tests after the last existing test (line 95):

```python
def make_rationalglu() -> RationalGLUBlock:
    return RationalGLUBlock(RBFFFNConfig(d_model=D, n_heads=H, dropout=0.0, model_type="rationalglu"))


def test_rationalglu_block_shape():
    block = make_rationalglu()
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_rationalglu_block_gradient_flow():
    block = make_rationalglu()
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None


def test_rationalglu_block_residual_connection():
    block = make_rationalglu()
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes && pytest rbf_ffn/tests/test_transformer_block.py::test_rationalglu_block_shape rbf_ffn/tests/test_transformer_block.py::test_rationalglu_block_gradient_flow rbf_ffn/tests/test_transformer_block.py::test_rationalglu_block_residual_connection -v
```

Expected: `ImportError: cannot import name 'RationalGLUBlock' from 'rbf_ffn.models.transformer_block'`

- [ ] **Step 3: Implement RationalGLUBlock**

In `rbf_ffn/models/transformer_block.py`, update line 8:

```python
from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN
```

Then append after the `RationalBlock` class:

```python
class RationalGLUBlock(nn.Module):
    """
    Transformer block with RationalGatedFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is RationalGatedFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = RationalGatedFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes && pytest rbf_ffn/tests/test_transformer_block.py -v
```

Expected: all tests pass (including the 3 new ones).

- [ ] **Step 5: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes && git add rbf_ffn/models/transformer_block.py rbf_ffn/tests/test_transformer_block.py && git commit -m "feat: add RationalGLUBlock to transformer_block"
```

---

## Chunk 2: Dispatch, Config, and YAML

### Task 3: CausalLM dict dispatch

**Files:**
- Modify: `rbf_ffn/models/model.py` — extend import, add dispatch entry, update docstring
- Modify: `rbf_ffn/tests/test_model.py` — add 2 tests

---

- [ ] **Step 1: Write failing tests**

In `rbf_ffn/tests/test_model.py`, append after `test_rational_params_in_adamw`:

```python
def test_rationalglu_output_shape():
    model = make_model("rationalglu")
    tokens = torch.randint(0, VOCAB, (B, N))
    assert model(tokens).shape == (B, N, VOCAB)


def test_rationalglu_params_in_adamw():
    """RationalGatedFFN act.a and act.b must be in AdamW (1-D), not Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("rationalglu")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.act.a) in adamw_ids,    "act.a should be in AdamW"
        assert id(block.ffn.act.a) not in muon_ids, "act.a should not be in Muon"
        assert id(block.ffn.act.b) in adamw_ids,    "act.b should be in AdamW"
        assert id(block.ffn.act.b) not in muon_ids, "act.b should not be in Muon"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes && pytest rbf_ffn/tests/test_model.py::test_rationalglu_output_shape rbf_ffn/tests/test_model.py::test_rationalglu_params_in_adamw -v
```

Expected: `KeyError: 'rationalglu'`

- [ ] **Step 3: Update model.py**

In `rbf_ffn/models/model.py`, update line 6:

```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock
```

Update the dispatch dict (lines 60-64) to add the `"rationalglu"` entry:

```python
        BlockClass = {
            "baseline":    LlamaBlock,
            "rbf":         RBFBlock,
            "rational":    RationalBlock,
            "rationalglu": RationalGLUBlock,
        }[cfg.model_type]
```

Update the `CausalLM` docstring (lines 52-56) to add:

```
        "rationalglu" → RationalGLUBlock (RationalGatedFFN)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes && pytest rbf_ffn/tests/test_model.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes && git add rbf_ffn/models/model.py rbf_ffn/tests/test_model.py && git commit -m "feat: add rationalglu model_type to CausalLM block dispatch"
```

---

### Task 4: Config wiring

**Files:**
- Modify: `rbf_ffn/config.py` — update `model_type` comment
- Create: `rbf_ffn/configs/rationalglu_ffn.yaml`

---

- [ ] **Step 1: Update config.py comment**

In `rbf_ffn/config.py`, update line 30:

```python
    model_type: str = "rbf"        # "baseline" | "rbf" | "rational" | "rationalglu"
```

- [ ] **Step 2: Create rationalglu_ffn.yaml**

Create `rbf_ffn/configs/rationalglu_ffn.yaml`:

```yaml
# RationalGLU ablation — gate_variant, sigma_variant, K, centers,
# sigma_init, and sinkhorn_iters are inert for model_type: rationalglu
# but present for a uniform config schema across all runs.
model_type: rationalglu
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

- [ ] **Step 3: Verify config loads**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -c "
from rbf_ffn.config import load_config
cfg = load_config('rbf_ffn/configs/rationalglu_ffn.yaml')
print(cfg.model_type, cfg.ffn_hidden, cfg.grad_accum_steps)
"
```

Expected: `rationalglu 688 1`

- [ ] **Step 4: Run full test suite**

```bash
cd /home/harikrishnan-c/projects/machine-notes && pytest rbf_ffn/tests/ -v --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes && git add rbf_ffn/config.py rbf_ffn/configs/rationalglu_ffn.yaml && git commit -m "feat: add rationalglu_ffn.yaml config"
```
