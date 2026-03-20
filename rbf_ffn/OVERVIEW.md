# RBF-FFN: Feed-Forward Network Architecture Exploration

A systematic study of alternative FFN designs for transformer language models, comparing RBF kernels, rational activations, and gating mechanisms on WikiText-103.

**Repository:** `/rbf_ffn/`
**Dataset:** WikiText-103 (perplexity metric)
**Baselines:** Llama-style transformers with SwiGLU FFNs
**Code:** Python + PyTorch, YAML-driven experiments

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

#### **Variant 6–9: RBF Variants (Gate Ablations)**

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

### A. Headline Results (3-epoch runs, WikiText-103)

| Variant | Val PPL (ep 0) | Val PPL (ep 2) | Δ vs Baseline | Time/epoch | Status |
|---------|--------------|--------------|---------------|-----------|--------|
| **PFDRationalGLU** | 140.91 | **73.00** | **−3.5%** ✓ | ~1975s | Best |
| **RationalGLU** | 142.61 | 74.37 | −1.7% ✓ | ~1424s | Good |
| **Baseline (SwiGLU)** | 145.62 | 75.68 | — | ~1234s | Ref |
| FirstOrderPFDRational | 146.00 | 76.77 | +1.4% | ~2029s | Marginal |
| Rational (non-gated) | 155.08 | 78.38 | +3.6% | ~1357s | Weak |
| RBF G1-B (input-driven) | 145.13 | 81.62 | +7.8% | ~1691s | Mid |
| RBF G1-A (cross-kernel) | 152.11 | 83.56 | +10.4% | ~1994s | Weak |
| RBF G0 (element-wise) | 164.67 | 92.70 | +22.4% | ~2294s | Poor |
| RBF G2 (Sinkhorn) | 186.67 | 110.28 | +45.7% | ~2771s | Very Poor |

### B. Key Findings

#### **1. Rational Activations Outperform SiLU**
- **RationalGLU:** 74.37 PPL (−1.7% vs SwiGLU)
- **Improvement source:** Learnable P(x)/Q(x) adapts better to language modeling than fixed SiLU
- **Parameter cost:** Only +36 params for 6 blocks (negligible relative to 12d² FFN)
- **Training overhead:** ~15% slowdown due to polynomial evaluation (acceptable)

#### **2. PFD Outperforms Padé Rational Form**
- **PFDRationalGLU:** 73.00 PPL (−3.5% vs SwiGLU) — **best overall**
- **vs RationalGLU:** 1.37 PPL gain from switching to partial fraction decomposition
- **Trade-off:** 60% training overhead (1975s vs 1234s baseline) for the extra expressiveness
- **Early signal:** Lowest epoch-0 PPL (140.91) suggests favorable initialization

#### **3. Gating Is Load-Bearing**
- **Non-gated RationalFFN:** 78.38 PPL (3.6% worse than SwiGLU)
- **RationalGLU:** 74.37 PPL (1.7% better)
- **Gap:** 4.0 PPL — multiplicative gating structure critical even with learnable activations
- Learnable activation alone cannot compensate for missing gate branch

#### **4. RBF Kernels Underperform — Gate Design Matters Critically**
- **Best RBF (G1-B):** 81.62 PPL — 6.8% worse than SwiGLU
- **Worst RBF (G2):** 110.28 PPL — 44% worse than SwiGLU
- **Why:** Replacing learned up-projection with static Gaussian centers loses expressiveness; gate design becomes crucial
  - **G1-B** uses input-driven gate (pre-RBF input) — best RBF variant
  - **G0** uses element-wise gate (post-RBF output) — poor, local signal only
  - **G2** uses Sinkhorn (no gate, just aggregation) — worst, too rigid for language modeling
- **Conclusion:** Static kernel expansion is fundamentally limited; task-specific learned projections are important

#### **5. Parameter Efficiency Trade-Off**
- **FirstOrderPFDRational (sin gate, shared projection):** 76.77 PPL, 33% fewer FFN params
- **vs SwiGLU:** +1.4% PPL worse, but with significant parameter savings
- **Viability:** Acceptable for parameter-constrained settings (e.g., mobile, edge inference)
- **Challenge:** High epoch-0 train PPL (9153) suggests sin saturation at initialization; mitigation strategies needed

#### **6. σ Granularity Is Second-Order**
- **Per-dim σ (σ-C):** 91.79 PPL — marginal 0.91 PPL win over global σ
- **Per-center σ (σ-B):** 92.50 PPL — minimal specialization benefit
- **Global σ (σ-A):** 92.70 PPL — practical default
- **Insight:** Model prefers narrower bandwidth than grid spacing default (σ=0.5); single global parameter sufficient

#### **7. Training Efficiency Hierarchy**
| Category | Representative | Time/epoch | Overhead |
|----------|---|---|---|
| Fastest | SwiGLU baseline | ~1234s | 0% |
| Rational family | RationalGLU | ~1424s | +15% |
| RBF family | G1-B | ~1691s | +37% |
| PFD rational | PFDRationalGLU | ~1975s | +60% |
| Worst | RBF G2 (Sinkhorn) | ~2771s | +124% |

---

## IV. Analysis & Interpretation

### Why RationalGLU Works Better Than RBF

The learnable rational activation succeeds where RBF kernels fail because:

1. **Task-Specificity:** Rational parameters learn language-specific nonlinearity; Gaussian kernels are fixed and generic
2. **Adaptive Bandwidth:** σ parameters adjust during training; Gaussian centers are static
3. **Parameter Efficiency:** Rational (4 coefficients) vs RBF (d×K expansion) — similar expressiveness, far fewer params
4. **Gating Synergy:** Learnable gate + learnable activation create redundancy allowing optimization; static kernels limit gate's expressiveness

### Why PFD Beats Padé

Partial fraction decomposition provides:
- **Better initialization:** Lowest epoch-0 PPL suggests better early gradient flow
- **Numerical stability:** Sum of simpler terms vs ratio of polynomials
- **Inductive bias:** Decomposition may naturally capture language task structure
- **Trade-off:** 1.37 PPL improvement warrants ~40% training cost increase

### RBF Lessons

RBF approach fails because:
- **Static vs. learnable trade-off:** Fixed centers sacrifice adaptability
- **Gate design criticality:** RBF quality depends heavily on gate design; no single gate beats learnable activation
- **Parameter ceiling:** Even with 5 optimized gates, can't exceed learned projection quality
- **Sinkhorn failure:** Doubly-stochastic aggregation too rigid for language modeling; winner-take-all better

---

## V. Recommended Next Steps

### Highest Priority (Likely +0.5–2.0% improvement)

1. **PFDRationalGLU, 10+ epochs**
   - Confirm whether 3.5% improvement holds at longer training horizon or is early-training effect
   - Extrapolate validation trajectory

2. **PFDRationalGLU, per-channel PFD params**
   - Current: Shared params across all 256 dimensions
   - Test: Per-channel (256 independent) or per-head (8 independent) rational params
   - Hypothesis: Channel specialization may widen gap further

3. **FirstOrderPFDRational, φ initialization tuning**
   - Epoch-0 train PPL spike (9153) indicates sin saturation
   - Test: Larger φ init (e.g., π/4) or learnable φ scale to reduce wrapping
   - Target: Better early training dynamics without sacrificing final perplexity

### Medium Priority (Confirming results, ablations)

4. **RationalGLU, 10+ epochs**
   - Extend to longer training; confirm 1.7% improvement holds
   - Per-channel rational params (test hypothesis 2 for RationalGLU too)

5. **G1-B + σ-C stacking (RBF)**
   - Best two RBF variants together
   - Expect 80–81 PPL (marginal improvement over G1-B alone at 81.62)

6. **G2 with K > 5 (RBF)**
   - Test K=10 or K=20
   - Hypothesis: More centers give Sinkhorn more expressiveness

### Exploratory (Low priority, speculative)

7. **Hybrid approaches**
   - RBF preprocessing + rational gate (combine insights)
   - Rational activation on RBF output

8. **Scaling laws**
   - Repeat with larger d_model (512, 1024)
   - Do rational advantages scale?
   - Is RBF failure fundamental or depth-dependent?

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
│   ├── transformer_block.py       # LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock, FirstOrderPFDBlock
│   ├── rational_ffn.py            # RationalActivation, RationalFFN, RationalGatedFFN
│   ├── rbf_ffn.py                 # RBFBlock, gates (G0, G1A, G1B, G2), bandwidth variants
│   ├── attention.py               # CausalSelfAttention
│   └── model.py                   # CausalLM, block dispatch
├── configs/
│   ├── baseline.yaml              # SwiGLU reference
│   ├── rational_ffn.yaml          # Non-gated rational
│   ├── rationalglu_ffn.yaml       # Gated rational (Padé)
│   ├── pfd_rationalglu_ffn.yaml   # Gated rational (PFD) — **best variant**
│   ├── pfd_rationalglu_ffn_small.yaml  # Small model version
│   ├── g0_baseline.yaml           # RBF G0 gate
│   ├── g1a_cross_kernel.yaml      # RBF G1-A gate
│   ├── g1b_input_driven.yaml      # RBF G1-B gate
│   ├── g2_sinkhorn.yaml           # RBF G2 gate
│   ├── sigma_b_per_center.yaml    # RBF σ per-center
│   └── sigma_c_per_dim.yaml       # RBF σ per-dim
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Integration tests, optimizer groups
│   ├── test_rational_ffn.py        # Unit tests for rational variants
│   ├── test_rbf_ffn.py            # Unit tests for RBF variants
│   └── test_transformer_block.py   # Block-level tests
├── experiments/
│   ├── 20260313_..._G0_*/         # Per-run directories with config.yaml + metrics.jsonl
│   └── ...
└── findings.md                    # Detailed results table + interpretation
```

### Running Experiments

```bash
# Baseline (reference)
python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml --n_epochs 10

# Best variant
python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_ffn.yaml --n_epochs 10

# Quick test (3 epochs)
python -m rbf_ffn.train --config rbf_ffn/configs/rationalglu_ffn.yaml --n_epochs 3

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

**Last updated:** 2026-03-17
**Best result:** PFDRationalGLU at 73.00 PPL (−3.5% vs SwiGLU)
