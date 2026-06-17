# GQA Attention Variant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `n_kv_heads` config field to enable Grouped Query Attention (and MQA) across all five existing attention variants via `repeat_interleave` expansion.

**Architecture:** Add `n_kv_heads: int = 0` to `ModelConfig` (0 resolves to `n_heads` in `__post_init__`). Each attention class sizes K/V projections to `n_kv_heads * head_dim` and expands back to `n_heads` with `repeat_interleave` before attention. When `n_kv_heads == n_heads`, `n_groups == 1` and the expand is a no-op.

**Tech Stack:** Python, PyTorch, pytest

## Global Constraints

- All five `ATTN_REGISTRY` variants must support `n_kv_heads` — no new registry keys.
- `n_kv_heads=0` is the sentinel for MHA (resolves to `n_heads` in `__post_init__`).
- `n_heads % n_kv_heads != 0` must raise `ValueError`.
- No bias on any projection.
- All tests live in `rbf_ffn/tests/test_transformer_block.py`.

---

### Task 1: Config field + validation

**Files:**
- Modify: `rbf_ffn/config.py`
- Test: `rbf_ffn/tests/test_transformer_block.py`

**Interfaces:**
- Produces: `ModelConfig.n_kv_heads: int` — always a valid, resolved value after `__post_init__` (never 0 in downstream code).

- [ ] **Step 1: Write the failing tests**

Add to `rbf_ffn/tests/test_transformer_block.py`:

```python
# ── n_kv_heads config ─────────────────────────────────────────────────────────

def test_n_kv_heads_default_resolves_to_n_heads():
    cfg = ModelConfig(d_model=32, n_heads=4)
    assert cfg.n_kv_heads == 4


def test_n_kv_heads_zero_resolves_to_n_heads():
    cfg = ModelConfig(d_model=32, n_heads=4, n_kv_heads=0)
    assert cfg.n_kv_heads == 4


def test_n_kv_heads_explicit():
    cfg = ModelConfig(d_model=32, n_heads=4, n_kv_heads=2)
    assert cfg.n_kv_heads == 2


def test_n_kv_heads_mqa():
    cfg = ModelConfig(d_model=32, n_heads=4, n_kv_heads=1)
    assert cfg.n_kv_heads == 1


def test_n_kv_heads_indivisible_raises():
    with pytest.raises(ValueError, match="n_kv_heads"):
        ModelConfig(d_model=32, n_heads=4, n_kv_heads=3)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py::test_n_kv_heads_default_resolves_to_n_heads rbf_ffn/tests/test_transformer_block.py::test_n_kv_heads_indivisible_raises -v
```

Expected: FAIL — `ModelConfig` has no `n_kv_heads` attribute.

- [ ] **Step 3: Add the field and validation to `rbf_ffn/config.py`**

After the `qkv_gain_targets` field (line 37), add:

```python
    n_kv_heads: int = 0            # 0 = match n_heads (MHA). 1 = MQA. 1 < n < n_heads = GQA.
```

In `__post_init__`, after the `model_type` block and before the `qkv_gain` block, add:

```python
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -k "n_kv_heads" -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/config.py rbf_ffn/tests/test_transformer_block.py
git commit -m "feat(config): add n_kv_heads field for GQA/MQA support"
```

---

### Task 2: CausalSelfAttention + ExclusiveSelfAttention (SDPA, separate Q/K/V)

**Files:**
- Modify: `rbf_ffn/models/attention.py`
- Test: `rbf_ffn/tests/test_transformer_block.py`

**Interfaces:**
- Consumes: `ModelConfig.n_kv_heads` (always resolved, Task 1).
- Produces: `CausalSelfAttention` and `ExclusiveSelfAttention` with K/V sized to `n_kv_heads * head_dim` and expanded before SDPA.

- [ ] **Step 1: Write the failing tests**

Add to `rbf_ffn/tests/test_transformer_block.py`:

```python
# ── GQA shape tests ───────────────────────────────────────────────────────────

def make_gqa_cfg(attn_type: str) -> ModelConfig:
    return ModelConfig(
        d_model=D, n_heads=H, n_kv_heads=2, dropout=0.0,
        attn_type=attn_type, ffn_type="swiglu",
        ffn_hidden=86, pfd_n=4,
    )


@pytest.mark.parametrize("attn_type", ["standard", "xsa"])
def test_gqa_sdpa_output_shape(attn_type):
    block = TransformerBlock(make_gqa_cfg(attn_type))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_gqa_standard_gradient_flows():
    block = TransformerBlock(make_gqa_cfg("standard"))
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py::test_gqa_sdpa_output_shape rbf_ffn/tests/test_transformer_block.py::test_gqa_standard_gradient_flows -v
```

Expected: FAIL — projections output `D` but split expects `n_kv_heads * head_dim`.

- [ ] **Step 3: Replace `CausalSelfAttention` in `rbf_ffn/models/attention.py`**

Replace the entire `CausalSelfAttention` class (lines 245–314) with:

```python
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    Supports Grouped Query Attention (GQA) via cfg.n_kv_heads. When
    n_kv_heads < n_heads, K and V projections output n_kv_heads * head_dim
    and are expanded to n_heads via repeat_interleave before SDPA.
    n_kv_heads == n_heads is standard MHA (no overhead).

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, KV, bias=False)
        self.v_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(split_q(F.silu(self.q_proj(x)) if self._qkv_silu else self.q_proj(x)))
        k = self.rope(split_kv(F.silu(self.k_proj(x)) if self._qkv_silu else self.k_proj(x)))
        v = split_kv(F.silu(self.v_proj(x)) if self._qkv_silu else self.v_proj(x))

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)
```

- [ ] **Step 4: Replace `ExclusiveSelfAttention` in `rbf_ffn/models/attention.py`**

Replace the entire `ExclusiveSelfAttention` class (lines 153–242) with:

```python
class ExclusiveSelfAttention(nn.Module):
    """
    Exclusive Self-Attention (XSA) with GQA support.

    Runs causal MHA to produce Y, then projects each output vector onto the
    subspace orthogonal to the normalised value vector:

        Vn = V / ||V||          (per head, per position, post-expansion)
        Z  = Y - (Y · Vn) Vn

    Supports GQA via cfg.n_kv_heads — K and V are projected to
    n_kv_heads * head_dim then expanded before SDPA. The Gram-Schmidt step
    uses the expanded V so each Q head is orthogonalised against its KV group.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, KV, bias=False)
        self.v_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            k_raw = F.silu(k_raw)
            v_raw = F.silu(v_raw)

        q = self.rope(split_q(q_raw))
        k = self.rope(split_kv(k_raw))
        v = split_kv(v_raw)

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)

        Vn = F.normalize(v, dim=-1)
        Z  = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn

        out = Z.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)
```

- [ ] **Step 5: Run tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -k "gqa" -v
```

Expected: `test_gqa_sdpa_output_shape[standard]`, `test_gqa_sdpa_output_shape[xsa]`, `test_gqa_standard_gradient_flows` all PASS.

Also run the full suite to catch regressions:

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -v
```

Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/models/attention.py rbf_ffn/tests/test_transformer_block.py
git commit -m "feat(attention): add GQA support to CausalSelfAttention and ExclusiveSelfAttention"
```

---

### Task 3: KVSharedAttention + KVSharedExclusiveSelfAttention

**Files:**
- Modify: `rbf_ffn/models/attention.py`
- Test: `rbf_ffn/tests/test_transformer_block.py`

**Interfaces:**
- Consumes: `ModelConfig.n_kv_heads` (Task 1), `make_gqa_cfg` helper (Task 2).
- Produces: `KVSharedAttention` and `KVSharedExclusiveSelfAttention` with `kv_proj` sized to `n_kv_heads * head_dim`.

- [ ] **Step 1: Write the failing tests**

Add to `rbf_ffn/tests/test_transformer_block.py`:

```python
@pytest.mark.parametrize("attn_type", ["kv_shared", "xsa_kv_shared"])
def test_gqa_kv_shared_output_shape(attn_type):
    block = TransformerBlock(make_gqa_cfg(attn_type))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py::test_gqa_kv_shared_output_shape -v
```

Expected: FAIL — `kv_proj` outputs `D` but split expects `n_kv_heads * head_dim`.

- [ ] **Step 3: Replace `KVSharedAttention` in `rbf_ffn/models/attention.py`**

Replace the entire `KVSharedAttention` class with:

```python
class KVSharedAttention(nn.Module):
    """
    Multi-head causal self-attention with K = V (shared projection) and GQA support.

    Q is produced by its own projection; K and V both come from a single shared
    `kv_proj` (output size n_kv_heads * head_dim). RoPE is applied to Q and K.
    K and V are expanded from n_kv_heads to n_heads via repeat_interleave before SDPA.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.kv_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_raw = self.q_proj(x)
        kv_raw = self.kv_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            kv_raw = F.silu(kv_raw)

        q = self.rope(split_q(q_raw))
        v = split_kv(kv_raw)
        k = self.rope(v)

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)
```

- [ ] **Step 4: Replace `KVSharedExclusiveSelfAttention` in `rbf_ffn/models/attention.py`**

Replace the entire `KVSharedExclusiveSelfAttention` class with:

```python
class KVSharedExclusiveSelfAttention(nn.Module):
    """
    Exclusive Self-Attention (XSA) with K = V (shared projection) and GQA support.

    Combines KVSharedAttention's projection scheme (kv_proj outputs
    n_kv_heads * head_dim) with XSA's Gram-Schmidt orthogonalisation step.
    K and V are expanded from n_kv_heads to n_heads before SDPA; the XSA
    step uses the expanded V.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.kv_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_raw = self.q_proj(x)
        kv_raw = self.kv_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            kv_raw = F.silu(kv_raw)

        q = self.rope(split_q(q_raw))
        v = split_kv(kv_raw)
        k = self.rope(v)

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)

        Vn = F.normalize(v, dim=-1)
        Z  = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn

        out = Z.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)
```

- [ ] **Step 5: Run tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -k "gqa" -v
```

Expected: all 5 GQA tests PASS.

Full suite:

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -v
```

Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/models/attention.py rbf_ffn/tests/test_transformer_block.py
git commit -m "feat(attention): add GQA support to KVSharedAttention and KVSharedExclusiveSelfAttention"
```

---

### Task 4: PolarAttention

**Files:**
- Modify: `rbf_ffn/models/attention.py`
- Test: `rbf_ffn/tests/test_transformer_block.py`

**Interfaces:**
- Consumes: `ModelConfig.n_kv_heads` (Task 1), `make_gqa_cfg` (Task 2).
- Produces: `PolarAttention` with K/V sized to `n_kv_heads * head_dim`; `k_scale` resized to `n_kv_heads`; expansion happens after polar decomposition and before the attention matmul.

**Note on `k_scale`:** Previously `k_scale` had shape `(n_heads,)`. With GQA it becomes `(n_kv_heads,)` — one scale per KV head. It is applied to `r_k` before expansion, then `r_k` is expanded alongside `k_dir` and `v`.

- [ ] **Step 1: Write the failing test**

Add to `rbf_ffn/tests/test_transformer_block.py`:

```python
def test_gqa_polar_output_shape():
    block = TransformerBlock(make_gqa_cfg("polar"))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py::test_gqa_polar_output_shape -v
```

Expected: FAIL — projection shape mismatch.

- [ ] **Step 3: Replace `PolarAttention` in `rbf_ffn/models/attention.py`**

Replace the entire `PolarAttention` class with:

```python
class PolarAttention(nn.Module):
    """
    Polar-coordinates causal self-attention with GQA support.

    Decomposes Q and K into direction (unit vector) and magnitude, computes
    cosine similarity as the base geometric score, then re-weights by the
    outer product of magnitudes scaled by per-head learnable confidence
    parameters q_scale (shape: n_heads) and k_scale (shape: n_kv_heads).

    With GQA (n_kv_heads < n_heads), K and V projections output
    n_kv_heads * head_dim. Polar decomposition runs at n_kv_heads size;
    k_scale is applied before expansion; k_dir, r_k, and v are then
    expanded to n_heads via repeat_interleave before the attention matmul.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, KV, bias=False)
        self.v_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.q_scale = nn.Parameter(torch.ones(H))
        self.k_scale = nn.Parameter(torch.ones(self.n_kv_heads))
        self._qkv_silu = cfg.qkv_silu
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            k_raw = F.silu(k_raw)
            v_raw = F.silu(v_raw)

        q = q_raw.view(B, N, self.n_heads, self.head_dim)
        k = k_raw.view(B, N, self.n_kv_heads, self.head_dim)
        v = v_raw.view(B, N, self.n_kv_heads, self.head_dim)

        r_q = torch.norm(q, p=2, dim=-1, keepdim=True)   # (B, N, H, 1)
        r_k = torch.norm(k, p=2, dim=-1, keepdim=True)   # (B, N, n_kv_heads, 1)
        q_dir = q / (r_q + 1e-6)
        k_dir = k / (r_k + 1e-6)

        q_dir = q_dir.transpose(1, 2)   # (B, H, N, head_dim)
        k_dir = k_dir.transpose(1, 2)   # (B, n_kv_heads, N, head_dim)
        v     = v.transpose(1, 2)        # (B, n_kv_heads, N, head_dim)
        r_q   = r_q.transpose(1, 2)     # (B, H, N, 1)
        r_k   = r_k.transpose(1, 2)     # (B, n_kv_heads, N, 1)

        if hasattr(self, "q_gain"):
            q_dir = q_dir * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k_dir = k_dir * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        # Apply k_scale before expansion (one scalar per KV head)
        r_k = r_k * self.k_scale.view(1, -1, 1, 1)

        # Expand KV tensors to full head count
        k_dir = k_dir.repeat_interleave(self.n_groups, dim=1)   # (B, H, N, head_dim)
        v     = v.repeat_interleave(self.n_groups, dim=1)        # (B, H, N, head_dim)
        r_k   = r_k.repeat_interleave(self.n_groups, dim=1)     # (B, H, N, 1)

        # Cosine similarity: (B, H, N, N)
        attn_weights = torch.matmul(q_dir, k_dir.transpose(-2, -1))

        # Re-weight by magnitude product with per-head confidence scalars
        scale_q = self.q_scale.view(1, -1, 1, 1)                # (1, H, 1, 1)
        attn_weights = attn_weights * (r_q * scale_q) * r_k.transpose(-2, -1)

        # Causal mask
        mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril()
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)                        # (B, H, N, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -k "gqa" -v
```

Expected: all 6 GQA tests PASS (`standard`, `xsa`, `kv_shared`, `xsa_kv_shared`, `polar` shape tests + gradient test).

Full suite:

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/models/attention.py rbf_ffn/tests/test_transformer_block.py
git commit -m "feat(attention): add GQA support to PolarAttention"
```

---

### Task 5: ATTN_REGISTRY parametrized test + MQA smoke test

**Files:**
- Test: `rbf_ffn/tests/test_transformer_block.py`

**Interfaces:**
- Consumes: `make_gqa_cfg` (Task 2), all five updated attention classes (Tasks 2–4).

- [ ] **Step 1: Write the tests**

Add to `rbf_ffn/tests/test_transformer_block.py`:

```python
@pytest.mark.parametrize("attn_type", list(ATTN_REGISTRY.keys()))
def test_gqa_all_registry_variants_shape(attn_type):
    """Every ATTN_REGISTRY entry must forward correctly with n_kv_heads=2."""
    block = TransformerBlock(make_gqa_cfg(attn_type))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


@pytest.mark.parametrize("attn_type", list(ATTN_REGISTRY.keys()))
def test_mqa_all_registry_variants_shape(attn_type):
    """n_kv_heads=1 (MQA) must forward correctly for every variant."""
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_kv_heads=1, dropout=0.0,
        attn_type=attn_type, ffn_type="swiglu",
        ffn_hidden=86, pfd_n=4,
    )
    block = TransformerBlock(cfg)
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)
```

Also add the ATTN_REGISTRY import at the top of the test file (alongside the existing `FFN_REGISTRY` import):

```python
from rbf_ffn.models.attention import ATTN_REGISTRY
```

- [ ] **Step 2: Run**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_transformer_block.py -v
```

Expected: all tests PASS, including 10 new parametrized tests (5 GQA + 5 MQA).

- [ ] **Step 3: Commit**

```bash
git add rbf_ffn/tests/test_transformer_block.py
git commit -m "test(attention): add parametrized GQA/MQA coverage for all ATTN_REGISTRY variants"
```
