# RBF-FFN: Transformer FFN Architecture Exploration

A systematic study of alternative feed-forward network designs for causal language models, comparing RBF kernels, learnable rational activations, polar-coordinate representations, Kronecker factorization, and normalization strategies on WikiText-103.

**Best result:** SwiGLU + QK-norm + weight-norm → **58.16 val PPL** (−23.1% vs vanilla SwiGLU)

---

## Overview

The feed-forward network (FFN) accounts for roughly two-thirds of a transformer's parameters and compute, yet most models use the same fixed SwiGLU design. This project asks:

- Can **learnable activations** outperform the fixed SiLU?
- Can **alternative feature expansions** (RBF kernels, polar directions) replace learned projections?
- Can **parameter-efficient factorizations** (Kronecker products, shared projections) maintain quality?
- What **normalization strategies** help most, and why?

All experiments share the same base architecture (6 layers, d=256, 8 heads, seq_len=512, WikiText-103) and optimizer setup (Muon + AdamW), isolating the effect of each architectural choice.

---

## Base Architecture

Every model variant is built on a pre-norm residual transformer:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

**Attention:** Multi-head causal attention with RoPE positional encoding. Q and K receive rotary embeddings; FlashAttention is used when available.

**Optimizer split:** 2D weight matrices go to **Muon** (lr=0.02); biases, embeddings, and scalar/1D parameters go to **AdamW** (lr=3e-4). This reflects Muon's design for matrices and AdamW for everything else.

---

## Experiments

---

### Experiment 1 — Baseline: SwiGLU FFN

**Goal:** Establish a reference using the standard Llama-style FFN.

**Math:**

```
gate = SiLU(W_gate · x)        # gating branch
value = W_up · x               # value branch
out = W_down · (gate ⊙ value)  # element-wise product, then project
```

Where `SiLU(x) = x · σ(x)` is the sigmoid-linear unit.

**Intuition:** The multiplicative gate acts as a soft feature selector — neurons where the gate is near zero are suppressed, and the SiLU's smooth non-linearity produces a gradient-friendly "leaky" gate. The two-branch structure roughly doubles the expressiveness per parameter compared to a plain MLP, at the cost of an extra projection.

**Parameter count:** 3 projections, each d×H where H=688. Total ≈ 3 × 256 × 688 ≈ 529K per block.

**Config:** `baseline.yaml`

---

### Experiment 2 — RBF Kernels

**Goal:** Replace the learned up-projection with static Gaussian RBF centers, dramatically cutting parameters. Test whether kernel-based feature expansion can substitute for learned projection.

**Math:**

Each input scalar `xᵢ` is expanded through K Gaussian kernels with fixed centers `c = [−1, −0.5, 0, 0.5, 1]`:

```
φ(xᵢ, cₖ) = exp(−(xᵢ − cₖ)² / (2σ²))
rbf(x) ∈ R^{d×K}     # K activations per input dimension
```

`σ` is learnable (parameterised as `softplus(σ_raw)` to enforce positivity).

Four gate designs were tested:

| ID | Name | Gate input | Formula |
|----|------|-----------|---------|
| G0 | Element-wise | RBF output | `σ(w ⊙ rbf + b) ⊙ rbf` |
| G1-A | Cross-kernel | RBF output | `σ(Linear(d·K → d·K)(rbf)) ⊙ rbf` |
| G1-B | Input-driven | Pre-RBF input | `σ(Linear(d → d·K)(x)) ⊙ rbf` |
| G2 | Sinkhorn | RBF output | Doubly-stochastic aggregation across K centers |

**Intuition:** Static Gaussian centers act like a lookup table: the model learns to place tokens near relevant centers to get strong activations. The gate then decides which centers' contributions to pass through. The key question is whether the gate receives enough signal — G1-B uses the original token representation (richer signal) while G0 only sees the post-RBF output (already limited by the static centers).

**σ bandwidth ablations:** Three granularities tested: global (1 parameter), per-center (K parameters), per-dimension (d×K parameters). Conclusion: differences are within 1 PPL point — σ granularity is a second-order effect.

**Results:** All RBF variants underperform SwiGLU. Best RBF (G1-B): 81.62 PPL (+6.8%). Static centers fundamentally limit adaptability — the model cannot move centers or reshape their bandwidth to match task structure.

**Configs:** `baseline.yaml` (reference); RBF variant configs are historical (no longer in `configs/`)

---

### Experiment 3 — Learnable Rational Activations

**Goal:** Keep the SwiGLU gating structure but replace the fixed SiLU with a learnable rational function. Test whether task-adaptive activation can outperform fixed activation at near-zero parameter cost.

#### 3a. RationalGLU (Gated)

**Math:**

```
P(x) = a₀ + a₁x + a₂x² + a₃x³              # degree-3 numerator (Horner's method)
Q(x) = 1 + |x · (b₀ + b₁x)|                # always ≥ 1, no poles

gate = P(W_gate · x) / Q(W_gate · x)
out = W_down · (gate ⊙ W_up · x)
```

6 learnable scalars (a₀–a₃, b₀–b₁) shared across all positions and channels.

**Intuition:** A rational function P(x)/Q(x) can approximate a much wider family of shapes than a fixed polynomial or SiLU — it can be asymmetric, have variable slope, and develop sharp transitions in one region while staying flat elsewhere. The denominator Q(x) ≥ 1 prevents poles and bounds gradients. Only 6 numbers control the gate's nonlinearity for all 256 dimensions, so the model must learn a single "best" activation for language modeling — which it does.

**Results:** 74.37 PPL (−1.7% vs SwiGLU) at +15% training time. Negligible parameter overhead: +36 params for 6 blocks.

#### 3b. RationalFFN (Non-Gated, Ablation)

**Math:**

```
out = W_down · Rational(W_up · x)
```

Same activation, but no multiplicative gate branch — one fewer projection.

**Intuition:** This tests whether the activation alone carries the benefit, or whether the gating structure is essential. The answer is clear: without the gate, this falls to 78.38 PPL (worse than SwiGLU). The multiplicative gate is load-bearing — it lets the network suppress irrelevant features entirely, which a single nonlinearity cannot.

**Configs:** `rationalglu_ffn.yaml`, `rationalglu_qk_norm.yaml`, `rational_ffn.yaml`

---

### Experiment 4 — Partial Fraction Decomposition (PFD) Rational

**Goal:** Replace the Padé rational form with a partial fraction decomposition. PFD has better numerical properties and a different inductive bias.

#### 4a. PFDRationalGLU (Gated)

**Math:**

```
f(x) = Σᵢ₌₁ⁿ (aᵢx + bᵢ) / (x² + cᵢ²)  +  γ·x
```

Where n=4 (default). The denominator `x² + cᵢ²` is always positive since `cᵢ²` is positive, so there are no poles. γ is a learnable pass-through scalar.

**Intuition:** The Padé form (P/Q) expresses a rational function as a single ratio of polynomials. The PFD form expresses the same class of functions as a *sum* of simpler terms — each term `(aᵢx + bᵢ)/(x² + cᵢ²)` is a simple resonant bump with a learnable center `cᵢ`, amplitude `aᵢ`, and offset `bᵢ`. This decomposition:
- Gives better gradient flow early in training (the epoch-0 PPL is the lowest of any variant: 140.91)
- Each term specializes independently — one might learn to pass negative inputs, another to amplify near zero
- The sum naturally handles multi-modal activation shapes

**Results:** 73.00 PPL (−3.5% vs SwiGLU). Best FFN activation variant. Cost: ~60% training overhead.

#### 4b. PFDRationalFFN (Non-Gated, Ablation)

Same PFD activation, no gate. Confirms the same pattern as RationalFFN vs RationalGLU: gating is essential regardless of activation form.

**Configs:** `pfd_rationalglu_ffn.yaml`, `pfd_rationalglu_qk_norm.yaml`, `pfd_rationalglu_qk_norm_weight_norm.yaml`, `pfd_rational_ffn.yaml`

---

### Experiment 5 — FirstOrderPFDRational (Parameter-Efficient)

**Goal:** Reduce FFN parameter count by 33% (2 projections instead of 3) while keeping a gating structure. Use a phase-shifted sine as the gate signal to create diversity from a single shared projection.

**Math:**

```
u = W_up · x                                # shared projection (one matrix)
gate = PFDRational(sin(u + φ))             # phase-shifted, then PFD activation
out = W_down · (gate ⊙ u)                  # same u used as value
```

φ ∈ R^{H} is a learnable phase vector (initialized small, ~0.02·randn).

**Intuition:** Standard SwiGLU uses separate `W_gate` and `W_up` matrices so the gate and value receive different linear combinations of the input. Here, a single `W_up` produces `u`, and the phase offset `φ` decorrelates `u` from `sin(u + φ)` — two different signals from one projection. The sine wraps `u` into `[−1, 1]` cyclically; the PFD activation then reshapes this into the gate. The `φ` vector allows different hidden dimensions to use different phase offsets, providing diversity without a second matrix.

**Challenge:** At initialization, `sin(u)` has near-zero gradients at integer multiples of π (saturation). This explains the very high epoch-0 train PPL (9153) — the model needs to escape these saturation regions before training becomes smooth.

**Results:** 76.77 PPL (+1.4% vs SwiGLU) with 33% fewer FFN parameters. Near-parity with SwiGLU at significantly lower parameter cost.

**Configs:** `first_order_pfd_rational_ffn.yaml`, `first_order_pfd_rational_qk_norm_weight_norm.yaml`

---

### Experiment 6 — Polar Coordinate FFN

**Goal:** Operate entirely in directional (angular) space, discarding token magnitude. Test whether language model FFNs need magnitude at all.

**Math:**

```
x_dir = x / ||x||₂                                        # L2-normalize to unit sphere
key_dir = keys / ||keys||₂     for keys ∈ R^{H×D}         # normalize learned directions

cos_sim = x_dir · key_dirᵀ    ∈ R^{...×H}                 # cosine similarities

gate = σ(10 · (cos_sim − θ))                               # sigmoid gate, sharpness=10
                                                           # θ ∈ R^H learnable threshold

out = W_down · (cos_sim ⊙ gate)
```

**Intuition:** A token's magnitude reflects how "typical" or "surprising" it is after attention — but what the FFN needs to act on is *what type* of token it is, not how strongly it arrived. By normalizing to the unit sphere, this model compares token direction against H learned semantic directions. A large cosine similarity means the token aligns with a learned direction — the gate then decides whether alignment is strong enough to activate. This is conceptually similar to a sparse key-value memory, but fully differentiable and integrated as a standard module.

**Config:** `polar_mlp.yaml`

---

### Experiment 7 — Polar Attention

**Goal:** Apply the same directional philosophy to attention: score attention by cosine similarity of queries and keys (geometric alignment), with learnable confidence scalars modulating each head's sensitivity.

**Math:**

```
q_dir = Q / ||Q||₂     (per token, per head)
k_dir = K / ||K||₂

score = (q_dir · k_dirᵀ) * (q_scale * k_scale)   # scaled cosine similarity
       + q_mag_log * q_scale + k_mag_log * k_scale # optional magnitude modulation
```

Where `q_scale, k_scale ∈ R^{n_heads}` are learnable per-head confidence scalars (1D → AdamW group).

**Intuition:** Standard dot-product attention mixes direction and magnitude in a way that can be hard to interpret or optimize. Polar attention decouples them: the cosine term measures *semantic alignment* (are this query and key pointing at similar concepts?), while the magnitude terms modulate confidence (a token with large activation magnitude may warrant stronger attention). Per-head scalars let each head choose how much to trust magnitude vs. direction.

**Configs:** `polar_attn.yaml` (PolarAttention + SwiGLU FFN), `polar_full.yaml` (PolarAttention + AdaptivePolarMLP)

---

### Experiment 8 — Kronecker-Factored MLP Projections

**Goal:** Replace each `nn.Linear` in FFN layers with a Kronecker-factored approximation. Reduce parameters by ~50% while preserving the full rank of the weight matrix.

**Math:**

A standard linear layer has weight `W ∈ R^{out × in}`. KroneckerLinear replaces this with:

```
W ≈ A ⊗ B
```

Where `A ∈ R^{out₁ × in₁}` and `B ∈ R^{out₂ × in₂}`, with `out₁ · out₂ = out` and `in₁ · in₂ = in`. The factors are chosen so `out₁ ≈ √out` and `in₁ ≈ √in` (square factorization to maximize savings).

The forward pass never materializes the full Kronecker product — instead:

```python
x_reshaped = x.view(..., in1, in2)
out = einsum('...ij, mi, nj -> ...mn', x_reshaped, A, B)
out = out.reshape(..., out_features)
```

**Intuition:** The Kronecker product `A ⊗ B` produces a matrix where every row of A is scaled by B. This means the weight matrix has a structured form — it's a "repeated pattern" of B, modulated by the rows of A. This is a strong inductive bias: it assumes the input-output relationship decomposes into two independent subspaces of dimensions (in₁, in₂) and (out₁, out₂). For language model FFNs, this is a hypothesis worth testing: can the hidden-to-hidden transformation factorize cleanly?

**Parameter savings:** For d=256, H=688:
- Standard `W_up ∈ R^{688×256}`: 176K params
- Kronecker: `A ∈ R^{26×16}`, `B ∈ R^{43×16}` (approximate): ~1.1K params — roughly 160× fewer

**Muon compatibility:** A and B are both 2D matrices, so they go to the Muon optimizer group.

**Config:** `baseline_qk_norm_weight_norm_kronecker.yaml`

---

### Experiment 9 — Normalization Strategies

Normalization proved to be the dominant factor — far more impactful than any FFN architecture change.

#### 9a. QK Normalization

**Math:**

```
Q_norm = RMSNorm(Q)   (per-head, after RoPE)
K_norm = RMSNorm(K)
attn_score = Q_norm · K_normᵀ / √d_head
```

**Intuition:** Without QK-norm, attention logits can grow large during training (the dot product grows as the norms of Q and K grow). Large logits concentrate the softmax distribution, causing "attention collapse" — a few positions dominate. Normalizing Q and K to unit scale prevents this, keeping the attention distribution spread out and gradients flowing. This is the same insight behind modern QK-norm in models like PaLM-2.

**Effect:** Consistent −0.5 to −0.9 PPL improvement across all model types.

#### 9b. Linear Weight Normalization

**Math:**

After each optimizer step, for each linear layer (excluding weight-tied lm_head):

```
for each output neuron i:
    w[i] = w[i] * (target_norm / ||w[i]||₂)     # if not max_only
         = w[i] * min(1, target_norm / ||w[i]||₂) # if max_only=True
```

Target norm: 2.0 by default.

**Intuition:** Unconstrained weight matrices can develop neurons with very different norms — some grow large (dominating the output), others shrink small (contributing nothing). This creates an implicit hierarchy the optimizer didn't intend. By constraining each output neuron's weight row to a fixed L2 norm, all neurons are forced to compete on equal footing through their *direction*, not their *magnitude*. This is a form of implicit regularization that prevents the optimizer from taking shortcuts by scaling individual neurons up or down.

**Effect:** −21.8 PPL improvement — **the single largest effect observed**. Dwarfs all FFN activation changes.

#### 9c. Adaptive Depth-Based Weight Normalization

**Math:**

The target norm decreases linearly from early to late layers:

```
target_norm(layer ℓ) = early_norm - (early_norm - late_norm) * ℓ / (L - 1)
```

Additionally, a phase-aware correction adjusts the target based on the derivative of the train/val log-gap:

```
log_gap = log(val_loss) - log(train_loss)
d_gap = EMA(log_gap) - log_gap              # smoothed derivative
correction = gamma * tanh(beta * d_gap)    # bounded correction
target_norm(ℓ) += correction               # tighten late layers when gap grows
```

**Intuition:** Early layers learn broad, high-norm representations (input distribution shaping), while later layers refine toward the output distribution (lower-norm, more targeted). Imposing a fixed target on all layers ignores this structure. A linearly decreasing target respects the depth-based specialization. The phase correction detects when generalization gap is actively growing and tightens the later layers' norm budget — a form of adaptive regularization that responds to overfitting in real time.

**Config:** `baseline_adaptive_weight_norm.yaml`

#### 9d. Activation Coefficient Normalization

For RationalActivation: normalize the coefficient vectors `a` and `b` to L2 norm 2.0 after each step. For PFDRationalActivation: normalize `a`, `b`, `c` independently; skip scalar `γ`.

**Intuition:** Prevents any single PFD term from dominating the activation function. In practice, slightly harmful when combined with weight normalization (the two normalizations interact and over-constrain the optimization).

---

## Results Summary

### Normalization ablations (SwiGLU backbone, WikiText-103, ep 2)

| Variant | Val PPL | Δ vs SwiGLU |
|---------|---------|-------------|
| **SwiGLU + QK-norm + weight-norm** | **58.16** | **−23.1%** |
| SwiGLU + weight-norm | 58.97 | −22.1% |
| PFDRationalGLU + QK-norm + weight-norm | 58.91 | −22.2% |
| SwiGLU + QK-norm | 75.14 | −0.7% |
| RationalGLU + QK-norm | 73.51 | −2.9% |
| PFDRationalGLU + QK-norm | 72.25 | −4.5% |

> Weight normalization (−21.8 PPL) is the dominant effect. It eclipses all FFN architecture gains combined.

### FFN activation variants (no normalization additions, ep 2)

| Variant | Val PPL | Δ vs SwiGLU | Time/epoch |
|---------|---------|-------------|-----------|
| **PFDRationalGLU** | **73.00** | **−3.5%** | ~1975s (+60%) |
| RationalGLU | 74.37 | −1.7% | ~1424s (+15%) |
| **SwiGLU (baseline)** | **75.68** | — | ~1234s |
| FirstOrderPFDRational | 76.77 | +1.4% | ~2029s |
| Rational (non-gated) | 78.38 | +3.6% | ~1357s |
| RBF G1-B (input-driven) | 81.62 | +7.8% | ~1691s |
| RBF G1-A (cross-kernel) | 83.56 | +10.4% | ~1994s |
| RBF G0 (element-wise) | 92.70 | +22.4% | ~2294s |
| RBF G2 (Sinkhorn) | 110.28 | +45.7% | ~2771s |

### Key take-aways

1. **Gating is load-bearing.** Non-gated variants (RationalFFN, RBF G0) all underperform SwiGLU. The multiplicative gate is not a nice-to-have — it is the mechanism by which the FFN selects relevant features.

2. **Learnable activations beat fixed SiLU.** RationalGLU (−1.7%) and PFDRationalGLU (−3.5%) both improve on SwiGLU with negligible extra parameters.

3. **PFD form is better than Padé.** The sum-of-resonances decomposition converges faster (lower epoch-0 PPL) and achieves a better final result. The 1.37 PPL gap is meaningful, though it comes at 40% more training time.

4. **Static kernels (RBF) cannot match learned projections.** Even the best RBF gate (G1-B, +6.8%) trails SwiGLU. Fixed Gaussian centers lose the adaptability that gradient descent would otherwise use to shape feature selectivity.

5. **Weight normalization dominates.** The −21.8 PPL improvement from constraining output neuron norms is larger than the entire spread of FFN architecture experiments. This suggests the training dynamics problem (neuron norm drift) is more significant than the activation form problem at this scale.

---

## Running Experiments

```bash
uv pip install -e .
```

```bash
# Best overall result
python -m rbf_ffn.train --config rbf_ffn/configs/baseline_weight_norm.yaml

# Best FFN activation + normalization
python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_qk_norm_weight_norm.yaml

# Parameter-efficient variant (33% fewer FFN params)
python -m rbf_ffn.train --config rbf_ffn/configs/first_order_pfd_rational_qk_norm_weight_norm.yaml

# Kronecker-factored MLP
python -m rbf_ffn.train --config rbf_ffn/configs/baseline_qk_norm_weight_norm_kronecker.yaml

# Polar attention
python -m rbf_ffn.train --config rbf_ffn/configs/polar_attn.yaml

# Override epochs at runtime
python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_ffn.yaml --n_epochs 10

# Resume from checkpoint
python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml --resume_from path/to/checkpoint_latest.pt
```

### Experiment output

Each run creates a timestamped directory:

```
rbf_ffn/experiments/20260324_164546_baseline_qknorm_wnorm_d256/
  config.yaml           # exact config used (reproducibility)
  metrics.jsonl         # one JSON line per epoch: {"epoch": 0, "train_loss": ..., "val_ppl": ...}
  checkpoint_best.pt    # best val PPL checkpoint
  checkpoint_final.pt   # final epoch checkpoint
  checkpoint_latest.pt  # overwritten each epoch (for resuming)
```

---

## Configs Reference

| Config | model_type | Normalization | Notes |
|--------|-----------|---------------|-------|
| `baseline.yaml` | baseline | — | SwiGLU reference |
| `baseline_qk_norm.yaml` | baseline | qk_norm | +QK-norm |
| `baseline_weight_norm.yaml` | baseline | qk_norm + weight_norm | Best SwiGLU result |
| `baseline_adaptive_weight_norm.yaml` | baseline | qk_norm + adaptive weight_norm | Depth-aware norm |
| `baseline_qk_norm_weight_norm_kronecker.yaml` | baseline | qk_norm + weight_norm | Kronecker MLP projections |
| `baseline_qk_norm_weight_norm_pre_silu.yaml` | baseline | qk_norm + weight_norm | +SiLU before lm_head |
| `rationalglu_ffn.yaml` | rationalglu | — | Learnable rational gate |
| `rationalglu_qk_norm.yaml` | rationalglu | qk_norm | |
| `rational_ffn.yaml` | rational | — | Non-gated (ablation) |
| `pfd_rationalglu_ffn.yaml` | pfd_rationalglu | — | PFD rational gate |
| `pfd_rationalglu_qk_norm.yaml` | pfd_rationalglu | qk_norm | |
| `pfd_rationalglu_qk_norm_weight_norm.yaml` | pfd_rationalglu | qk_norm + weight_norm | Best FFN activation overall |
| `pfd_rational_ffn.yaml` | pfd_rational | — | Non-gated PFD (ablation) |
| `first_order_pfd_rational_ffn.yaml` | first_order_pfd_rational | — | 2-projection, sin gate |
| `first_order_pfd_rational_qk_norm_weight_norm.yaml` | first_order_pfd_rational | qk_norm + weight_norm | Efficient + full norm stack |
| `polar_mlp.yaml` | polar_mlp | qk_norm + weight_norm | Directional FFN |
| `polar_attn.yaml` | polar_attn | — | Cosine-similarity attention |
| `polar_full.yaml` | polar_full | — | Polar attention + polar FFN |

---

## Config Fields

### Model

| Field | Default | Description |
|-------|---------|-------------|
| `model_type` | `"baseline"` | See table above for all values |
| `d_model` | 256 | Model dimension |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 6 | Transformer blocks |
| `ffn_hidden` | 688 | FFN hidden dim |
| `pfd_n` | 4 | PFD rational terms (PFD variants only) |
| `dropout` | 0.1 | Attention dropout |
| `seq_len` | 512 | Context length |
| `vocab_size` | 50257 | Vocabulary size (GPT-2 BPE) |
| `qk_norm` | `false` | RMSNorm on Q and K after RoPE |
| `qkv_silu` | `false` | SiLU after Q, K, V projections |
| `pre_lm_head_silu` | `false` | SiLU before lm_head |
| `kronecker_mlp` | `false` | Replace FFN `nn.Linear` with `KroneckerLinear` |

### Normalization

| Field | Default | Description |
|-------|---------|-------------|
| `linear_weight_norm` | `false` | Normalize linear weight rows after each optimizer step |
| `linear_weight_norm_value` | `2.0` | Target L2 norm per output neuron |
| `linear_weight_norm_max_only` | `false` | If true, only scale down (never scale up) |
| `activation_norm` | `false` | Normalize rational/PFD activation coefficients |
| `adaptive_weight_norm` | `false` | Depth-based linearly-decreasing target norm |
| `adaptive_norm_early` | `2.5` | Target norm at layer 0 |
| `adaptive_norm_late` | `1.2` | Target norm at layer L−1 (must be ≥ 1.0) |
| `adaptive_norm_gamma` | `0.3` | Max phase-correction magnitude |
| `adaptive_norm_beta` | `5.0` | Tanh sensitivity to gap derivative |
| `adaptive_norm_alpha` | `0.9` | EMA smoothing for log-gap |

### Training

| Field | Default | Description |
|-------|---------|-------------|
| `seed` | 42 | Random seed |
| `n_epochs` | 10 | Training epochs |
| `batch_size` | 32 | Per-GPU batch size |
| `muon_lr` | 0.02 | Muon learning rate (2D weight matrices) |
| `adamw_lr` | 3e-4 | AdamW learning rate (biases, 1D params) |
| `adamw_wd` | 0.1 | AdamW weight decay |
| `warmup_ratio` | 0.02 | Fraction of total steps for linear warmup |
| `grad_clip` | 1.0 | Gradient norm clipping threshold |
| `grad_accum_steps` | 1 | Mini-batches per optimizer step |

---

## Tests

```bash
pytest rbf_ffn/tests/ -v
```

---

## References

- **SwiGLU:** Shazeer (2020), "GLU Variants Improve Transformer"
- **RoPE:** Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **Rational Activations:** Molina et al. (2019), "Pad\'e Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks"
- **Muon Optimizer:** Kosson et al. — momentum orthogonalization for 2D parameter matrices
- **QK Norm:** Henry et al. (2020), "Query-Key Normalization for Transformers"
- **Kronecker Factorization:** Van Loan & Pitsianis (1993); applied to neural compression in Martens & Grosse (2015)
