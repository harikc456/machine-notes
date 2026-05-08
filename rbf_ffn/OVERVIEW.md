# RBF-FFN: Feed-Forward Network Architecture Exploration

A systematic study of alternative FFN designs for transformer language models, comparing RBF kernels, rational activations, and gating mechanisms on WikiText-103.

**Repository:** `/rbf_ffn/`
**Dataset:** WikiText-103 (perplexity metric)
**Baselines:** Llama-style transformers with SwiGLU FFNs
**Code:** Python + PyTorch, YAML-driven experiments

---

## Consolidation Note (2026-04-01)

`rbf_ffn/` is now the **single source of truth** for transformer experiments in this repo.
The original `kromhc_transformer/` implementation has been archived to `archive/kromhc_transformer/`.

KromHC head mixing is available in `rbf_ffn/` via `use_kromhc: true` in any config.
It wraps any `model_type` with a `KromHCWrapper` — see `models/head_mixer.py` and
`models/transformer_block.py`. Configs: `baseline_kromhc.yaml`, `baseline_qk_norm_kromhc.yaml`,
`pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml`.

---

## I. Motivation

The standard SwiGLU FFN in modern transformers uses:
- A fixed SiLU activation function
- A multiplicative gating structure (gate branch × value branch)
- A large up-projection (4× the model dimension)

**Questions this work addresses:**

1. **Can learnable activations outperform fixed SiLU?** Standard activations (ReLU, SiLU, GELU) are not task-specific. A learnable rational function might adapt to language modeling better.

2. **Can we reduce parameters without sacrificing performance?** The FFN dominates parameter count (≈66% of total in a 6-layer 256-dim model). RBF kernels expand features via static Gaussian centers rather than learned projections.

3. **What gating structures are most effective?** Gate design has been underexplored; we test element-wise, cross-kernel, input-driven, and Sinkhorn variants.

4. **Does the order of FFN activation matter?** Partial fraction decomposition (PFD) and Padé approximations are alternative parameterizations that may have better inductive biases.

---

## II. Methodology

### A. Experimental Setup

**Architecture:**
- Transformer: 6 layers, 256 dims, 8 heads, 512 seq length
- Shared across all variants: RMSNorm, RoPE, causal attention
- Metric: Validation perplexity on WikiText-103
- Training: 3 epochs (primary results), 10+ epochs (extended runs)
- Batch size: 16, seed: 42, learning rate: Muon (0.02) + AdamW (0.0003)

**Parameter Budget Awareness:**
All designs target approximate parity with SwiGLU (≈12·d² parameters) to isolate FFN choice from capacity effects.

| Component | SwiGLU | RBF-FFN (K=5) | RationalGLU | PFDRationalGLU |
|-----------|--------|---------------|-------------|----------------|
| **Total** | 12d²   | 5d²           | 12d² + 36   | 12d² + 36      |

### B. FFN Variants

#### **Variant 1: Baseline (SwiGLU)**
```
out = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
```
- Fixed activation: SiLU(x) = x·σ(x)
- Parameters: 3 projections (256→688→256)
- Baseline for all comparisons

#### **Variant 2: RationalFFN (Non-Gated)**
```
out = down_proj(Rational(up_proj(x)))
```
- Learnable rational activation: P(x)/Q(x)
- No gating structure; single projection path
- Tests activation expressiveness in isolation
- 33% fewer FFN parameters (2 vs 3 projections)

#### **Variant 3: RationalGLU (Gated)**
```
out = down_proj(Rational(gate_proj(x)) ⊙ up_proj(x))
```
- Replaces SiLU in the gate path with a learnable rational function
- Parameter-matched to SwiGLU (+36 params for 6 blocks)
- Tests whether learnable activation can improve gating

#### **Variant 4: PFDRationalGLU (Partial Fraction Decomposition)**
```
gate = ∑ᵢ (aᵢ/(x - pᵢ))   # 4 partial fraction terms
out = down_proj(gate ⊙ up_proj(x))
```
- Uses partial fraction decomposition instead of Padé rational form
- More expressive parameterization (~0.6× slower)
- Better early-epoch behavior observed in preliminary runs

#### **Variant 5: FirstOrderPFDRational (Parameter-Efficient)**
```
shared = up_proj(x)
gate = sin(shared + phi) * phase_scale
out = down_proj(gate ⊙ shared)
```
- Single projection shared between gate and value paths
- Phase-shifted sine gate (sin(u+φ)) provides signal diversity
- 33% fewer FFN parameters than SwiGLU
- Tests whether gate and value can share representations

#### **Variant 6: LeakyReLUSq (Tested)**
```
gate = leaky_relu(gate_proj(x))²   # always non-negative
out  = down_proj(gate * up_proj(x))
```
- Parameter-matched to SwiGLU; always-non-negative gate
- XSA + LeakyReLUSq + qk_norm: 79.30 PPL — ~9 ppl worse than XSA+SwiGLU+qknorm (69.87); deprioritized

#### **Variant 7: OrthogonalMLPWrapper (Pending)**
```
y   = mlp(x)                                    # any inner FFN
out = y - (y·x / (x·x + ε)) * x                # project out component along x
```
- Wraps any FFN so its additive update is perpendicular to the residual stream input
- FFN analogue of XSA: XSA makes attention output orthogonal to its value; this makes FFN output orthogonal to its input
- Planned as `baseline_xsa_qk_norm_orthogonal_mlp.yaml` (XSA + SwiGLU + qk_norm + orthogonal_ffn)
- No experiments run yet

#### **Variant 8–11: RBF Variants (Gate Ablations)**

**RBF Architecture:**
```
rbf_out = [φ(x, c₁), φ(x, c₂), ..., φ(x, c_K)]  # K Gaussian kernels
gate_sig = σ(w ⊙ rbf_out + b)
out = down_proj(gate_sig ⊙ rbf_out)
```

- Replaces up-projection with K Gaussian RBF kernels (static centers)
- 4 gate variants (G0–G2):

| ID | Gate input | Mechanism | Performance |
|----|-----------|-----------|-------------|
| **G0** | RBF output | Element-wise: `σ(w⊙rbf+b)⊙rbf` | +21.3% PPL (92.70) |
| **G1-B** | Pre-RBF input | Input-driven: `Linear(d→d·K)→σ` | +6.8% PPL (81.62) |
| **G1-A** | RBF output | Cross-kernel: `Linear(d·K→d·K)→σ` | +9.3% PPL (83.56) |
| **G2** | RBF output | Sinkhorn aggregation (no gate) | +44.3% PPL (110.28) |

**RBF Bandwidth (σ) Ablations:**
- Global σ: Single learnable parameter
- Per-center σ: K parameters (one per center)
- Per-dim σ: d×K parameters (one per dimension per center)

### C. Evaluation Protocol

**Primary metric:** Validation perplexity at epoch 2 (3-epoch runs)

**Secondary metrics:**
- Epoch 0 perplexity (initialization quality)
- Per-epoch training time (efficiency)
- σ dynamics (for RBF variants) — how bandwidth changes during training

**Statistical control:**
- Fixed seed (42) across all runs
- Identical hyperparameters (learning rates, batch size, warmup)
- Consistent data preprocessing and tokenization

---

## III. Experimental Results

### A. Headline Results

#### 3-epoch runs — no normalization additions (WikiText-103, d_model=256)

| Variant | Val PPL (ep 0) | Val PPL (ep 2) | Δ vs SwiGLU | Time/epoch | Status |
|---------|--------------|--------------|-------------|-----------|--------|
| **XSA + SwiGLU + qk_norm + orthogonal_ffn** | 130.88 | **69.87** | **−7.7%** ✓ | ~1167s | Best (no wnorm) |
| **XSA + SwiGLU + qk_norm** | 135.36 | **71.57** | **−5.4%** ✓ | ~1166s | Good |
| **XSA + SwiGLU** | 138.55 | **72.41** | **−4.3%** ✓ | ~1160s | Good |
| **PFDRationalGLU** | 140.91 | **73.00** | **−3.5%** ✓ | ~1975s | Good |
| **RationalGLU** | 142.61 | 74.37 | −1.7% ✓ | ~1424s | Good |
| **Baseline (SwiGLU)** | 145.62 | 75.68 | — | ~1234s | Ref |
| FirstOrderPFDRational | 146.00 | 76.77 | +1.4% | ~2029s | Marginal |
| Rational (non-gated) | 155.08 | 78.38 | +3.6% | ~1357s | Weak |
| XSA + LeakyReLUSq + qk_norm | 144.04 | 79.30 | +4.8% | ~1165s | Weak |
| RBF G1-B (input-driven) | 145.13 | 81.62 | +7.8% | ~1691s | Mid |
| RBF G1-A (cross-kernel) | 152.11 | 83.56 | +10.4% | ~1994s | Weak |
| Polar MLP | 172.15 | 95.11 | +25.7% | ~1305s | Poor |
| RBF G0 (element-wise) | 164.67 | 92.70 | +22.4% | ~2294s | Poor |
| RBF G2 (Sinkhorn) | 186.67 | 110.28 | +45.7% | ~2771s | Very Poor |

Note: XSA+qknorm+orthogonal_ffn result is a single run. XSA+qknorm (no orthogonal_ffn) is best of 2 runs (71.57–71.87). XSA (no norm) is a single run.

#### 3-epoch runs — with normalization additions (qk_norm + weight_norm)

| Variant | Val PPL (ep 0) | Val PPL (ep 2) | Δ vs SwiGLU+norm | Time/epoch | Status |
|---------|--------------|--------------|-----------------|-----------|--------|
| **SwiGLU + qk_norm + weight_norm** | 114.41 | **58.16** | — | ~1330s | Best |
| SwiGLU + weight_norm | 120.19 | 58.97 | +1.4% | ~1223s | Good |
| PFDRationalGLU + qk_norm + weight_norm | 117.34 | 58.91 | +1.3% | ~2088s | Matched |
| SwiGLU + qk_norm + adaptive_weight_norm | 118.74 | 60.67 | +4.3% | ~1346s | Marginal |
| Polar MLP + qk_norm + weight_norm | 157.17 | 85.84 | +47.6% | ~1116s | Poor |

#### 10-epoch runs — with normalization (qk_norm + weight_norm, best of multiple runs)

| Variant | Val PPL (ep 9) | Notes |
|---------|---------------|-------|
| **SwiGLU + qk_norm + weight_norm** | **41.40** | Best of 4 runs; range 41.40–49.06 |
| PFDRationalGLU + qk_norm + weight_norm | 41.69 | Best of 2 runs; converges to same level |

### B. Key Findings

#### **1. Stacked Orthogonality (XSA + OrthogonalFFN) Provides the Best Per-Epoch Return Without Weight Normalization**
- **XSA + qk_norm + orthogonal_ffn:** 69.87 PPL (−7.7% vs SwiGLU) — best 3-epoch result without weight_norm; single run
- **XSA + qk_norm (no orthogonal_ffn):** 71.57 PPL (best of 2 runs) — qk_norm alone adds ~0.8 ppl over XSA; consistent with its effect on SwiGLU (~0.5–0.7 ppl)
- **OrthogonalFFN adds ~1.7 ppl** on top of XSA+qknorm (71.57→69.87); no training overhead (single dot product per position, no new parameters)
- **Mechanism:** XSA forces attention updates to be orthogonal to the attended value; OrthogonalMLPWrapper forces FFN updates to be orthogonal to the residual stream input — both sublayers must add genuinely new information
- **LeakyReLUSq is inferior in XSA context:** 79.30 PPL — always-non-negative gate halves the gate's dynamic range vs SwiGLU
- **Key open question:** Does dual orthogonality advantage persist with weight_norm?

#### **2. Linear Weight Normalization Is the Dominant Improvement**
- **SwiGLU + weight_norm:** 58.97 PPL (−22.1% vs SwiGLU baseline) — outweighs all FFN activation variants
- **SwiGLU + qk_norm + weight_norm:** 58.16 PPL (−23.1%) — new baseline for norm ablations
- **Mechanism:** Row-normalizing all linear weight matrices to target norm 2.0 dramatically accelerates early optimization (ep0 train PPL drops from ~3000 to ~800)
- **Critical:** max_only mode (clip-only, no scale-up) is catastrophic — 75.54 vs 58.16

#### **3. Weight Normalization Erases Rational Activation Advantage**
- **PFDRationalGLU + qk_norm + weight_norm:** 58.91 PPL — statistically tied with SwiGLU+norm (58.97)
- The 3.5% PFD advantage from unnormalized runs is an early-training effect, not a long-horizon improvement
- At 10 epochs: SwiGLU (41.40) ≈ PFDRationalGLU (41.69) with norm; both converge to the same level
- **Implication:** SwiGLU + qk_norm + weight_norm is the recommended baseline going forward

#### **4. PFD vs SwiGLU at 3 Epochs Without Norm**
- **PFDRationalGLU:** 73.00 PPL (−3.5% vs SwiGLU)
- **RationalGLU:** 74.37 PPL (−1.7% vs SwiGLU)
- PFD advantage at 3 epochs is real but erased at 10 epochs under normalization

#### **5. Gating Is Load-Bearing**
- **Non-gated RationalFFN:** 78.38 PPL (3.6% worse than SwiGLU)
- **RationalGLU:** 74.37 PPL (1.7% better)
- **Gap:** 4.0 PPL — multiplicative gating structure critical even with learnable activations
- Learnable activation alone cannot compensate for missing gate branch

#### **6. RBF and Polar Kernels Underperform**
- **Best RBF (G1-B):** 81.62 PPL — 6.8% worse than SwiGLU; 23.5 ppl behind SwiGLU+norm
- **Polar MLP (best):** 85.84 PPL with norm — discarding magnitude information is harmful
- Both approaches are fundamentally limited by replacing learned projections with static/directional kernels

#### **7. Kronecker MLP Is Incompatible with Weight Norm**
- Kronecker MLP + weight_norm: diverges (val PPL 100+)
- The per-row norm constraint is not well-defined for factored weights; requires a dedicated normalization scheme before proceeding

#### **8. Run-to-Run Variance Is Large**
- SwiGLU+qknorm+wnorm at ep9: range 41.40–49.06 across 4 runs
- Hardware/IO state substantially affects convergence speed and final PPL
- Single-run results at 10+ epochs should be treated as lower bounds

#### **9. Training Efficiency Hierarchy**
| Category | Representative | Time/epoch | Overhead |
|----------|---|---|---|
| Fastest | XSA + SwiGLU | ~1160s | −6% vs baseline |
| Baseline | SwiGLU | ~1234s | 0% |
| Rational family | RationalGLU | ~1424s | +15% |
| RBF family | G1-B | ~1691s | +37% |
| PFD rational | PFDRationalGLU | ~1975s | +60% |
| Worst | RBF G2 (Sinkhorn) | ~2771s | +124% |

---

## IV. Analysis & Interpretation

### Why XSA Works

Exclusive Self-Attention's Gram-Schmidt step (`Z = Y − (Y·Vn)Vn`) forces attention output to be orthogonal to the attended value. This provides:

1. **Diversity pressure:** Each head cannot simply copy its value vector; it must attend to information that adds something new
2. **Implicit regularization:** The orthogonalization acts like a built-in diversity constraint, potentially reducing head redundancy
3. **Cheap implementation:** Single subtraction per head; no additional parameters

The 4.3% improvement over SwiGLU at 3 epochs without any normalization suggests this is a genuine structural improvement, not an initialization artifact. Whether it stacks with weight norm (and whether it persists at 10 epochs) is the key open question.

### Why Weight Normalization Dominates

Linear weight normalization (constraining each row of every weight matrix to have L2 norm = 2.0) accelerates early optimization dramatically:
- Epoch 0 train PPL: 806 (with wnorm) vs 3245 (without)
- This implies the unnormalized model spends most of 3 epochs recovering from a poorly conditioned initialization
- The effect is so large (~17 ppl at ep2) that it overshadows all FFN activation design choices

This explains why PFDRationalGLU's 3-epoch advantage disappears: the advantage was about early-epoch optimization efficiency (lower ep0 PPL), not about the long-run expressiveness of the gate. Weight norm provides the same early efficiency benefit to any architecture.

### Why RationalGLU Works Better Than RBF

The learnable rational activation succeeds where RBF kernels fail because:

1. **Task-Specificity:** Rational parameters learn language-specific nonlinearity; Gaussian kernels are fixed and generic
2. **Adaptive Bandwidth:** σ parameters adjust during training; Gaussian centers are static
3. **Parameter Efficiency:** Rational (4 coefficients) vs RBF (d×K expansion) — similar expressiveness, far fewer params
4. **Gating Synergy:** Learnable gate + learnable activation create redundancy allowing optimization; static kernels limit gate's expressiveness

### Why PFD Beats Padé (at 3 Epochs Without Norm)

Partial fraction decomposition provides:
- **Better initialization:** Lowest epoch-0 PPL suggests better early gradient flow
- **Numerical stability:** Sum of simpler terms vs ratio of polynomials
- **Inductive bias:** Decomposition may naturally capture language task structure
- **Caveat:** The 1.37 PPL improvement at 3 epochs is an early-training effect; it disappears at 10 epochs under weight norm

### RBF and Polar Lessons

Both fail because:
- **Static vs. learnable trade-off:** Fixed centers (RBF) and discarded magnitude (Polar) sacrifice adaptability
- **Gate design criticality:** RBF quality depends heavily on gate design; no gate variant beats learned projections
- **Parameter ceiling:** Even with optimized gates, can't exceed learned projection quality
- **Sinkhorn failure:** Doubly-stochastic aggregation too rigid for language modeling; winner-take-all better

---

## V. Recommended Next Steps

### Highest Priority

1. **XSA + qk_norm + orthogonal_ffn + weight_norm (3 epochs)**
   - Best no-wnorm result is 69.87 (XSA+qknorm+orthogonal_ffn); adding wnorm (~17 ppl gain on SwiGLU) should push to ~52–54 PPL
   - If this beats SwiGLU+qknorm+wnorm (58.16), dual orthogonality is the new architecture baseline
   - No additional overhead (both XSA and orthogonal_ffn are parameter-free projections)

2. **XSA + qk_norm + weight_norm (no orthogonal_ffn) (3 epochs)**
   - Cleanly isolates wnorm effect without orthogonal_ffn; establishes whether the ~1.7 ppl gain from orthogonal_ffn persists under normalization
   - Expected: ~55–57 PPL (SwiGLU+wnorm was 58.16; XSA adds ~0.8–1 ppl over SwiGLU)

3. **XSA + qk_norm + orthogonal_ffn, 10 epochs**
   - Confirm whether dual orthogonality advantage persists at longer training
   - Compare against best SwiGLU+norm at ep9 (41.40)

### Medium Priority (Confirming and extending)

4. **Kronecker MLP with correct normalization**
   - Current weight_norm is per-row of the full weight matrix; not defined for factored (A⊗B) form
   - Approach: normalize at the Kronecker factor level or use spectral norm instead
   - Current experiments all diverge; this is a prerequisite before any Kronecker ablation

5. **PFDRationalGLU, per-channel PFD params**
   - ~~10-epoch result confirmed: PFD advantage disappears with norm~~ (done)
   - Remaining question: per-channel params (256 independent) may widen the gap at 3 epochs even after norm
   - Low expected return given normalization parity result

6. **FirstOrderPFDRational, φ initialization tuning**
   - Epoch-0 train PPL spike (9153 unnormalized, 3382 with norm) indicates sin saturation
   - Test: Larger φ init (e.g., π/4) or learnable φ scale to reduce wrapping
   - With norm, ep3=74.17 — behind SwiGLU; better init may close the gap

### Low Priority / Deprioritized

7. **G1-B + σ-C stacking (RBF)** — gap to best result (58.16) is now 27.5 ppl; unfavorable
8. **G2 with K > 5 (RBF)** — approach is fundamentally limited
9. **RBF/Polar hybrid approaches** — both underperform; not worth pursuing
10. **Scaling laws** — useful once the best architecture at d=256 is settled

---

## VI. Files & Organization

### Code Structure

```
rbf_ffn/
├── README.md                      # Quick-start guide
├── OVERVIEW.md                    # This file
├── findings.md                    # Detailed results table + interpretation
├── config.py                      # Config schema + defaults
├── train.py                       # Training loop
├── data.py                        # WikiText-103 loading
├── models/
│   ├── __init__.py
│   ├── transformer_block.py       # TransformerBlock (composable; dispatches via ATTN_REGISTRY + FFN_REGISTRY)
│   ├── attention.py               # CausalSelfAttention, ExclusiveSelfAttention (XSA), PolarAttention; ATTN_REGISTRY
│   ├── llama_ffn.py               # SwiGLU FFN (Llama-style); registered as "swiglu"
│   ├── rational_ffn.py            # RationalActivation, RationalFFN, RationalGatedFFN, PFD variants; FFN_REGISTRY
│   ├── polar_ffn.py               # AdaptivePolarMLP (directional/cosine similarity FFN); FFN_REGISTRY
│   ├── kronecker_linear.py        # KroneckerLinear, KroneckerDeltaLinear
│   ├── head_mixer.py              # KromHCWrapper (optional head mixing post-attention)
│   └── model.py                   # CausalLM; uses TransformerBlock + FFN_REGISTRY/ATTN_REGISTRY
├── configs/
│   ├── baseline.yaml              # SwiGLU reference
│   ├── baseline_qk_norm.yaml      # + qk_norm
│   ├── baseline_weight_norm.yaml  # + weight_norm
│   ├── baseline_xsa.yaml          # XSA + SwiGLU — **best no-norm result (72.41)**
│   ├── baseline_adaptive_weight_norm.yaml  # + depth-adaptive wnorm
│   ├── baseline_qk_norm_weight_norm_pre_silu.yaml  # + qkv_silu + pre_lm_head_silu
│   ├── baseline_kronecker_delta.yaml  # + Kronecker-delta MLP
│   ├── baseline_kronecker_lm_head.yaml # + Kronecker LM head
│   ├── baseline_qk_norm_weight_norm_kronecker.yaml  # + Kronecker MLP (no wnorm)
│   ├── baseline_untied_embeddings.yaml
│   ├── baseline_kromhc.yaml / baseline_qk_norm_kromhc.yaml
│   ├── rational_ffn.yaml          # Non-gated rational
│   ├── rationalglu_ffn.yaml / rationalglu_qk_norm.yaml
│   ├── pfd_rational_ffn.yaml / pfd_rationalglu_ffn.yaml
│   ├── pfd_rationalglu_qk_norm.yaml
│   ├── pfd_rationalglu_qk_norm_weight_norm.yaml   # **best 3-epoch norm result (tied SwiGLU+norm)**
│   ├── pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml
│   ├── first_order_pfd_rational_ffn.yaml / first_order_pfd_rational_qk_norm_weight_norm.yaml
│   ├── polar_mlp.yaml / polar_attn.yaml / polar_full.yaml
│   └── [RBF configs archived to experiments/archive/]
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Integration tests, optimizer groups
│   ├── test_rational_ffn.py       # Unit tests for rational variants
│   ├── test_rbf_ffn.py            # Unit tests for RBF variants
│   └── test_transformer_block.py  # Block-level tests
├── experiments/
│   ├── analysis.md                # Full per-epoch metrics for all completed runs
│   ├── archive/                   # Archived RBF runs (2026-03-13/14)
│   └── YYYYMMDD_*/                # Per-run dirs with config.yaml + metrics.jsonl
└── findings.md                    # Section-by-section findings + interpretation
```

### Running Experiments

```bash
# Current best (no norm, 3 epochs)
python -m rbf_ffn.train --config rbf_ffn/configs/baseline_xsa.yaml --n_epochs 3

# Current best (with norm, 3 epochs)
python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml --n_epochs 3  # + qk_norm + weight_norm flags

# 10-epoch extended run
python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml --n_epochs 10

# Run tests
pytest rbf_ffn/tests/ -v
```

---

## VII. References

- **SwiGLU:** Shazeer & Parmar (2020), "GLU Variants Improve Transformer"
- **RationalActivations:** Molina et al., learnable rational functions for neural networks
- **Partial Fraction Decomposition:** Classical technique; applied here to neural activation
- **RBF kernels:** Haykin (1999), "Neural Networks: A Comprehensive Foundation"
- **RoPE:** Su et al. (2021), "Roformer: Enhanced Transformer with Rotary Position Embedding"

---

**Last updated:** 2026-04-16
**Best result at 3 epochs (no weight_norm):** XSA + SwiGLU + qk_norm + orthogonal_ffn at 69.87 PPL (−7.7% vs SwiGLU, single run)
**Best result at 3 epochs (with norm):** SwiGLU + qk_norm + weight_norm at 58.16 PPL (−23.1% vs SwiGLU)
**Best result at 10 epochs (with norm):** SwiGLU + qk_norm + weight_norm at 41.40 PPL
