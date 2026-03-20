# Findings: KromHC Transformer on WikiText-103

## 1. Literature Survey

### 1.1 Landscape Overview

KromHC (January 2026) introduces Kronecker-factored doubly-stochastic permutations for mixing
attention heads. The method scales to 128+ heads with O(K × hidden_dim) parameters where
K = log₂(n_heads), enabling content-dependent, balanced information flow across heads.

### 1.2 Related Work

- **Multi-head attention** (Vaswani et al., 2017): Standard baseline; heads operate independently after split.
- **Head analysis** (Clark et al., 2019): Attention heads specialise (syntax, coreference, etc.).
- **Head pruning** (Michel et al., 2019): Many heads are redundant — suggests mixing is non-trivial.
- **KromHC** (Jan 2026): First application of Kronecker-factored doubly-stochastic mixing to heads.

## 2. Gap Analysis

### 2.1 Theoretical
- No prior proof that head permutation mixing preserves language modelling capacity.
- Scaling laws for KromHC overhead unknown.

### 2.2 Methodological
- Head mixing never benchmarked on standard LM tasks (WikiText-103).
- No ablation isolating the doubly-stochastic constraint vs. unconstrained mixing.

### 2.3 Empirical
- No comparison to baseline at multiple scales.
- Effect of QK norm + head mixing interaction unexplored.

## 3. Feasibility Assessment

### 3.1 Scientific Validity

KromHC is theoretically sound: Kronecker products of permutation matrices remain permutation
matrices; convex combination of permutations is doubly-stochastic (Birkhoff–von Neumann).

### 3.2 Novelty Assessment

Novel application. Prior work analyses or prunes heads; KromHC is the first to mix them
dynamically with content-dependent doubly-stochastic matrices.

### 3.3 Computational Feasibility

- **Parameter overhead**: ~K × (d_context × 32 + 32 × 2) per layer; negligible vs. d_model² projections.
- **Runtime overhead**: One (B×N, n, n) matmul per block; fast on modern hardware.
- **Memory**: Stores H for analysis but not needed for inference.
- **Verdict**: Fits comfortably on RTX 5060 Ti (16 GB VRAM) up to 200M params.

### 3.4 Overall Verdict

**Feasible and tractable.** Well-motivated by head specialisation literature; low overhead; clear experimental path.

## 4. Theoretical Contributions

*Pending: formal analysis of doubly-stochastic constraint and head information flow.*

## 5. Experimental Findings

### 5.1 Experiment Log

| Phase | Variant | Seeds | Primary Metric | Mean ± Std | H0 Decision | Status |
|-------|---------|-------|----------------|-----------|-------------|--------|
| 1: POC | KromHC | 1 | loss decreasing | TBD | N/A | Pending |
| 2: Baseline | Baseline | 3 | test_ppl | TBD | — | Pending |
| 2: Comparison | KromHC | 3 | test_ppl | TBD | TBD | Pending |
| 3: Ablation w/ mixing | KromHC | 3 | test_ppl + entropy | TBD | TBD | Pending |
| 3: Ablation w/o mixing | KromHC (disabled) | 3 | test_ppl + entropy | TBD | TBD | Pending |

### 5.2 Key Results

*To be populated after experiments run.*

### 5.3 Negative Results & Lessons

*To be documented.*

## 6. Conclusions & Next Steps

*Post-experiment.*

## References

1. Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
2. Clark et al. (2019). What Does BERT Look At? ACL Workshop.
3. Michel et al. (2019). Are Sixteen Heads Really Better than One? NeurIPS.
4. KromHC (January 2026). Kronecker-Factored Head Mixing.
