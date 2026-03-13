# RBF-FFN: Radial Basis Function Feed-Forward Network

**Date:** 2026-03-12
**Status:** Design approved

---

## Motivation

The standard transformer FFN (Up Projection → SwiGLU → Down Projection) uses a learned gating mechanism to selectively pass information. This design replaces it with an RBF-based expansion that offers:

1. **Curse-of-dimensionality mitigation** — LayerNorm constrains each scalar feature to a known marginal distribution (approximately zero-mean, unit-variance Gaussian). Because the RBF kernel is applied element-wise to individual scalars, this makes the 1D kernel distances stable and meaningful across training.
2. **Interpretable feature expansion** — static grid centers make each kernel activation correspond to a fixed region of the normalized input space.
3. **Parameter efficiency** — removing the up projection and relying on RBF expansion (factor K) reduces total FFN parameters from ~12·d_model² (SwiGLU) to ~K·d_model² (K=5 → ~5·d_model²).

---

## Architecture

### Full Forward Pass

```
x: (B, N, d_model)
  │
  ▼
LayerNorm       d_model → d_model            zero-mean, unit-variance
  │
  ▼
RBF             d_model → d_model·K          element-wise; K static centers, 1 learnable σ
  │
  ▼
Gate            sigmoid(w ⊙ x + b) ⊙ x      w, b ∈ ℝ^(d_model·K), learnable
  │
  ▼
Down Projection d_model·K → d_model          learnable
  │
  ▼
x: (B, N, d_model)
```

### Integration into Transformer Block (Pre-norm)

```python
x = x + attn(norm1(x))
x = x + rbf_ffn(norm2(x))
```

The `norm2` pre-block LayerNorm is kept unchanged. The LayerNorm inside RBF-FFN is an additional normalization applied specifically before the RBF kernel. While this results in double normalization, it serves a distinct purpose: the internal LayerNorm has its own learnable affine parameters (γ, β) that can rescale activations relative to the fixed RBF center positions, decoupling the RBF input distribution from whatever the attention sublayer produces.

---

## RBF Layer

### Kernel Formula

Applied independently to each scalar feature `x_i` in the `d_model`-dimensional vector:

```
φ_k(x_i) = exp( -(x_i - c_k)² / (2σ²) )     k = 1, ..., K
```

Output for one token: reshape `(d_model, K)` → flatten to `(d_model·K,)`.

### Centers

Static grid, registered as a non-learnable buffer:

```
c = [-1.0, -0.5, 0.0, 0.5, 1.0]    (K=5)
```

After LayerNorm, ~95% of values fall in `[-2, 2]`, so this grid covers the high-density region. Centers do not receive gradients.

Note: tail inputs beyond `±2` receive attenuated response (e.g., at `x=2.0`, nearest center `c=1.0`, σ=0.5 → φ ≈ 0.135, ~13.5% of peak; all other centers effectively zero). This is intentional — outlier activations are attenuated rather than fully suppressed, acting as soft saturation. If broader coverage is needed in future work, edge guard centers at `±2.0` can be added.

### Bandwidth

Single learnable scalar `σ`, initialized to grid spacing:

```
σ_init = 0.5
σ = softplus(σ_raw)    # enforces σ > 0 at all times
```

---

## Gate Layer

```
gate = sigmoid(w ⊙ x + b)
out  = gate ⊙ x
```

- `w ∈ ℝ^(d_model·K)` — learnable per-feature scale, initialized to `ones`
- `b ∈ ℝ^(d_model·K)` — learnable per-feature bias, initialized to `zeros`
- The gate receives RBF output which is always in `[0, 1]` (non-negative by construction of the Gaussian kernel). For inputs `x ∈ [0, 1]`, `sigmoid(x)·x` ranges from `0` to `sigmoid(1)·1 ≈ 0.73`, giving a smooth, monotonically increasing pass-through. At initialization (w=ones, b=zeros) the gate is approximately linear over the RBF output range.
- **Degeneracy risk:** gate weights `w` are unconstrained. If `w` grows large and positive, sigmoid saturates to 1 (gate becomes identity); if large and negative, sigmoid saturates to 0 (sublayer goes dead). Monitor gate weight norms during training; apply weight decay to `w` as a first-line mitigation.

---

## Parameter Budget

| Component         | SwiGLU FFN         | RBF-FFN (K=5)        |
|-------------------|--------------------|----------------------|
| Up projection     | 2 × d_model × 4d   | —                    |
| Down projection   | 4d × d_model       | K·d_model × d_model  |
| Gate              | fused in SwiGLU    | 2 × d_model·K        |
| σ                 | —                  | 1                    |
| **Total (approx)**| **12·d_model²**    | **~5·d_model²**      |

Note: SwiGLU count assumes `hidden_dim = 4·d_model` (two up projections of size d×4d, one down projection of size 4d×d → 8d²+4d²=12d²). Some implementations use `(8/3)·d_model` for FLOPs parity with a standard FFN, yielding ~10.7·d_model². All ablations in this project use `hidden_dim = 4·d_model` for consistency.

K can be increased to match SwiGLU parameter count for fair ablations.

---

## Initialization Summary

| Parameter       | Init value          | Constraint         |
|-----------------|---------------------|--------------------|
| Centers `c`     | `[-1,-0.5,0,0.5,1]` | frozen buffer      |
| `σ_raw`         | `softplus⁻¹(0.5)`   | `σ = softplus(σ_raw) > 0` |
| Gate `w`        | `ones`              | unconstrained      |
| Gate `b`        | `zeros`             | unconstrained      |
| Down projection weight | Kaiming uniform | unconstrained   |
| Down projection bias   | `zeros`         | unconstrained   |

---

## Ablation Roadmap

### σ Ablations (run in order; promote if prior stage shows promise)

| Stage | Change                           | Purpose                              |
|-------|----------------------------------|--------------------------------------|
| σ-A   | Global σ (current)               | Baseline — does the idea work?       |
| σ-B   | Per-center σ (K scalars)         | Can centers specialize bandwidth?    |
| σ-C   | Per-dim-per-center σ (d_model·K) | Full expressiveness upper bound      |

### Gating Ablations (independent study; compare all four against each other)

The goal is to understand whether the gate benefits from (a) cross-kernel interaction and (b) using the pre-RBF input `x` rather than the RBF output as the gate signal.

| ID   | Gate input                        | Mechanism                                                        | Down proj input  |
|------|-----------------------------------|------------------------------------------------------------------|------------------|
| G0   | RBF output (element-wise)         | `sigmoid(w ⊙ rbf_out + b) ⊙ rbf_out`                           | `d_model·K`      |
| G1-A | RBF output (flattened, d_model·K) | `Linear(d_model·K → d_model·K) → sigmoid → ⊙ rbf_out`          | `d_model·K`      |
| G1-B | Pre-RBF input `x` (d_model)       | `Linear(d_model → d_model·K) → sigmoid → ⊙ rbf_out`            | `d_model·K`      |
| G2   | RBF output (per-feature K values) | Sinkhorn over K dim (20 iters) → weighted sum → `(d_model,)`   | `d_model`        |

**G0** is the approved baseline design.

**G1-A** adds cross-kernel mixing: the flattened RBF output passes through a square linear layer before the sigmoid, allowing each gate value to see responses from all K centers of the same feature. Adds `(d_model·K)²` parameters — expensive at large K; consider restricting to small K for this ablation.

**G1-B** is the architecturally cleaner design: the gate is computed from the pre-RBF normalized input `x`, keeping gate and gated signal from separate branches. Adds `d_model × d_model·K` parameters.

**G2 (Sinkhorn)** replaces gating entirely with a soft aggregation over K centers per feature:
```
rbf_out: (B, N, d_model, K)      # reshape before Sinkhorn
→ Sinkhorn normalization over K dim, 20 iterations
→ weighted sum over K            # (B, N, d_model)
→ Down Projection: d_model → d_model
```
Sinkhorn produces a doubly stochastic weight matrix over the K kernel responses, enforcing that each center contributes meaningfully (prevents winner-take-all collapse). Down projection is `d_model → d_model` (no K expansion), making G2 the most parameter-efficient variant overall.

If G2 shows promise, refer to the **mHC-lite paper** for faster doubly stochastic matrix approximations to reduce the cost of 20-iteration Sinkhorn at larger scale.

---

## Project Structure

```
rbf_ffn/
  experiments/        # one subfolder per run, named by date + config
  models/             # RBFLayer, GateLayer, RBFFFN module definitions
  tests/              # unit tests for each component
  train.py            # training entry point
  config.py           # hyperparameter dataclass
```

---

## Validation Plan

1. Implement `RBFLayer`, `GateLayer`, `RBFFFN` as drop-in for standard FFN
2. Train on a small benchmark (WikiText-103 LM or CIFAR-10 ViT)
3. Compare perplexity / accuracy vs SwiGLU baseline at matched compute
4. Log per-experiment results under `rbf_ffn/experiments/`
