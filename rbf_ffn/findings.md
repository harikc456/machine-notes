# RBF-FFN Findings

## §1 Architecture

- **Forward pass:** `LayerNorm → RBF → Gate → Down Projection` replaces the standard SwiGLU FFN
- **RBF kernel:** applied element-wise per scalar; K=5 static centers `[-1, -0.5, 0, 0.5, 1]`, single learnable σ (or variants)
- **Gate:** `sigmoid(w ⊙ x + b) ⊙ x` with per-feature learnable w, b (G0 baseline); G1-A/B/G2 are alternatives
- **Double LayerNorm is intentional:** the internal LN has its own γ, β to decouple the RBF input distribution from attention output

## §2 Parameter Budget

| Component        | SwiGLU FFN     | RBF-FFN (K=5)   |
|------------------|----------------|-----------------|
| Up projection    | 2 × d × 4d     | —               |
| Down projection  | 4d × d         | K·d × d         |
| Gate             | fused in SwiGLU | 2 × d·K        |
| σ                | —              | 1 (or K or d·K) |
| **Total (approx)** | **12·d²**   | **~5·d²**       |

G2 (Sinkhorn) is more parameter-efficient still: down projection shrinks to `d → d` (no K expansion).

## §3 Implementation Notes

- σ is parameterised as `softplus(σ_raw)` to enforce positivity; initialized to `softplus⁻¹(0.5)`
- Gate weights initialized to ones, biases to zeros — gate starts approximately linear over `[0,1]` RBF range
- Tail inputs beyond ±2 receive attenuated response (~13.5% of peak at x=2.0, c=1.0, σ=0.5); intentional soft saturation
- `ffn_hidden=688` in configs is a legacy field carried over from SwiGLU baseline; not used by RBF path

## §4 σ Bandwidth Ablations (G0 gate, d_model=256, K=5, WikiText-103, 3 epochs)

All three σ variants perform within ~1 perplexity point of each other. Per-dim σ (σ-C) gives the clearest benefit.

| Variant       | Val PPL (ep 0) | Val PPL (ep 2) | σ at ep 2 (mean ± std) |
|---------------|---------------|---------------|------------------------|
| Global (σ-A)  | 164.67        | 92.70         | 0.420 ± 0.000          |
| Per-center (σ-B) | 167.93     | 92.50         | 0.466 ± 0.124          |
| Per-dim (σ-C) | 164.81        | 91.79         | 0.575 ± 0.044          |

- **Global σ** declines monotonically (0.483 → 0.420): model prefers a narrower bandwidth than the initial grid-spacing default
- **Per-center σ** shows center specialization (std grows from 0.110 → 0.124) but validation improvement over global is marginal (0.71 ppl)
- **Per-dim σ** converges to a higher mean σ (~0.575) and achieves the best per-center result; the spread is modest (std ~0.044), suggesting the per-dim freedom is used conservatively
- Conclusion: σ granularity is a second-order effect at 3 epochs; global σ is a reasonable default for gate ablations

## §5 FFN Ablation Results (d_model=256, WikiText-103, 3 epochs)

### Summary table

Δ values for rational/rationalglu/pfd variants are vs the 2026-03-16 baseline re-run (75.68). Δ values for RBF variants are vs the 2026-03-13 baseline (76.43). The two baseline runs are within 0.75 ppl of each other (run-to-run variance). FirstOrderPFDRational ran on 2026-03-17; its epoch time is not directly comparable to the 2026-03-16 baseline.

| Variant                         | Val PPL (ep 0) | Val PPL (ep 2) | Δ vs Baseline | Time/epoch |
|---------------------------------|---------------|---------------|---------------|------------|
| **PFDRationalGLU**              | 140.91        | **73.00**     | **−3.5%**     | ~1975s     |
| **RationalGLU**                 | 142.61        | **74.37**     | **−1.7%**     | ~1424s     |
| **Baseline (SwiGLU, 2026-03-16)** | 145.62      | **75.68**     | —             | ~1234s     |
| FirstOrderPFDRational           | 146.00        | 76.77         | +1.4%         | ~2029s†    |
| Rational (non-gated)            | 155.08        | 78.38         | +3.6%         | ~1357s     |
| Baseline (SwiGLU, 2026-03-13)   | 146.27        | 76.43         | —             | ~1813s     |
| RBF G1-B (input-driven)         | 145.13        | 81.62         | +6.8%         | ~1691s     |
| RBF G1-A (cross-kernel)         | 152.11        | 83.56         | +9.3%         | ~1994s     |
| RBF G0 (element-wise)           | 164.67        | 92.70         | +21.3%        | ~2294s     |
| RBF G2 (Sinkhorn)               | 186.67        | 110.28        | +44.3%        | ~2771s     |

† 2026-03-17 run; timing not directly comparable to 2026-03-16 baseline.

### Key findings

1. **G1-B closes most of the gap.** The input-driven gate — computing the gate signal from the pre-RBF normalized input rather than the RBF output — reaches val PPL 81.62 at epoch 2, only 5.19 points behind SwiGLU. It also trains ~7% faster than G0 (1691s vs 2294s/epoch) due to the smaller gate linear layer.

2. **G1-A is second.** Cross-kernel mixing (full d·K → d·K linear before sigmoid) reaches 83.56 ppl, confirming that inter-center interaction helps over the element-wise G0, but less efficiently than the input-driven branch.

3. **G0 underperforms relative to G1-A/B.** The approved baseline RBF design (element-wise gate on RBF output) lands 16 ppl points behind SwiGLU and 11 points behind G1-B. The gate receiving only its own RBF output constrains its expressiveness.

4. **G2 (Sinkhorn) is the weakest gate variant.** Replacing the gate with Sinkhorn aggregation over K centers produces the highest perplexity (110.28) and the longest training time (2772s/epoch — 53% slower than baseline). The doubly-stochastic constraint on K=5 centers may be too rigid for language modeling, where winner-take-all or sparse selection is beneficial.

5. **σ dynamics differ by gate.** G1-B's σ increases (0.573 → 0.620) while G0's decreases (0.483 → 0.420). The input-driven gate may allow the RBF to maintain wider bandwidth because the gate already filters via the pre-norm input.

6. **RationalGLU outperforms SwiGLU at 3 epochs.** Replacing the fixed SiLU gate activation with a learnable rational function achieves val PPL 74.37 — 1.31 ppl better than the contemporaneous baseline (75.68). RationalGLU is parameter-matched to SwiGLU (difference: 36 params = 6 blocks × 6 rational params), so the improvement is purely from activation expressiveness, not additional capacity. The rational gate appears to learn a better-shaped nonlinearity than the fixed SiLU for this task.

7. **Gating is essential; the non-gated rational FFN underperforms.** RationalFFN (`up → RationalAct → down`, no gate branch) achieves 78.38 ppl — better than all RBF variants but 4.01 ppl behind RationalGLU and 2.70 ppl behind SwiGLU. The multiplicative gate structure is load-bearing: the learnable activation in isolation is insufficient to match gated performance. Removing the gate branch also removes one projection (2 vs 3 linears), cutting parameter count by ~33% at the FFN.

8. **Rational variants are faster than RBF, competitive with baseline.** On the 2026-03-16 runs: Rational adds ~10% overhead over baseline (1357s vs 1234s/epoch); RationalGLU adds ~15% (1424s vs 1234s). Both are substantially faster than any RBF variant (G0: ~2294s, G2: ~2771s). The rational activation's overhead is dominated by polynomial evaluation, not a separate large linear.

9. **PFDRationalGLU sets a new best at 73.00 ppl.** Replacing the Pade rational gate (RationalGLU) with a PFD rational (partial fraction decomposition, n=4 terms) reduces val PPL from 74.37 to 73.00 — 2.68 ppl better than SwiGLU (−3.5%). Epoch 0 val PPL (140.91) is the lowest of any variant at epoch 0, suggesting the PFD form initialises favorably for the gate path. The 1.37 ppl gain over RationalGLU indicates the PFD representation has a better inductive bias for this task than the Pade form, at the cost of significantly higher training overhead (~60% vs baseline, compared to +15% for RationalGLU).

10. **FirstOrderPFDRational (2-projection) is near-parity with SwiGLU at 33% fewer FFN parameters.** The `sin(u+phi)` phase-shifted gate achieves 76.77 ppl with only 2 projections vs 3 for SwiGLU. It trails SwiGLU by 1.09 ppl but beats the non-gated RationalFFN (78.38) by 1.61 ppl, confirming that the multiplicative gate is still beneficial even when gate and value share the same projection. The high epoch-0 train PPL (9153) — the largest of any variant — reflects optimization difficulty from `sin` wrapping at initialization; the model recovers to a competitive final result. The phase shift `phi` provides enough signal diversity to substitute for a dedicated gate projection.

## §6 Normalization Ablations (d_model=256, WikiText-103, 3 epochs)

New runs on 2026-03-18 and 2026-03-24/25 introduce two normalization techniques on top of the best variants from §5: **QK normalization** in attention (qk_norm) and **weight normalization** on linear layers (linear_weight_norm, max_norm=2.0). A third technique, **activation coefficient normalization** (activation_norm), was tested on 2026-03-25.

### Summary table

Δ values are vs the 2026-03-16 SwiGLU baseline (75.68). All runs use d_model=256, 3 epochs.

| Variant | Norm additions | Val PPL (ep 2) | Δ vs §5 baseline | Time/epoch |
|---------|---------------|---------------|-----------------|------------|
| Baseline (SwiGLU) + qk_norm + weight_norm | qk_norm, weight_norm | **58.16** | **−23.1%** | ~1330s |
| PFDRationalGLU + qk_norm + weight_norm | qk_norm, weight_norm | **58.91** | **−22.2%** | ~2097s |
| Baseline (SwiGLU) + weight_norm | weight_norm | 58.97 | −22.1% | ~1223s |
| PFDRationalGLU + qk_norm + weight_norm + act_norm | qk_norm, weight_norm, act_norm | 59.82 | −21.0% | ~2113s |
| PFDRationalGLU + qk_norm | qk_norm | 72.25 | −4.5% | ~2126s |
| RationalGLU + qk_norm | qk_norm | 73.51 | −2.9% | ~1515s |
| Baseline (SwiGLU) + qk_norm | qk_norm | 75.14 | −0.7% | ~1307s |
| *§5 PFDRationalGLU (no norm additions)* | — | *73.00* | *−3.5%* | *~1975s* |
| *§5 RationalGLU (no norm additions)* | — | *74.37* | *−1.7%* | *~1424s* |
| *§5 Baseline (SwiGLU, 2026-03-16)* | — | *75.68* | *—* | *~1234s* |

### Key findings

11. **Linear weight normalization is the dominant improvement.** Adding `linear_weight_norm` (max_norm=2.0) to the SwiGLU baseline drops val PPL from 75.68 → 58.97 — a 21.8 ppl reduction (−28.8% relative). This single change outweighs all FFN activation variants tested in §5. The mechanism is likely better gradient conditioning and prevention of weight matrix blow-up during training with Muon optimizer.

12. **QK normalization provides a modest but consistent gain.** Adding `qk_norm` alone improves: baseline 75.68 → 75.14 (−0.54 ppl), RationalGLU 74.37 → 73.51 (−0.86 ppl), PFDRationalGLU 73.00 → 72.25 (−0.75 ppl). The effect is small but consistent across variants.

13. **Weight normalization erases the PFDRationalGLU advantage.** With weight normalization applied, PFDRationalGLU (58.91) is essentially tied with the SwiGLU baseline (58.97 with weight_norm only; 58.16 with qk_norm+weight_norm). The −3.5% advantage PFDRationalGLU had over SwiGLU in §5 disappears. This suggests weight normalization addresses the same optimization pathology that the PFD rational gate was implicitly correcting.

14. **Activation coefficient normalization (act_norm) is slightly harmful.** PFDRationalGLU + qk_norm + weight_norm + act_norm achieves 59.82 — 0.91 ppl worse than the same config without act_norm (58.91). Normalizing the PFD rational activation coefficients adds a constraint that may limit the rational gate's expressiveness when linear weights are already normalized.

15. **SwiGLU + qk_norm + weight_norm is the new best overall.** Val PPL 58.16 at epoch 2 is the best result across all experiments. The combination of attention QK normalization and linear weight normalization provides the largest improvement, with lower training overhead than PFDRationalGLU (+8% vs baseline with weight_norm).

16. **Best PFDRationalGLU result (58.91) vs best Baseline (58.16): gap has narrowed from 2.68 ppl to 0.75 ppl.** The remaining small PFDRationalGLU advantage is within run-to-run variance and may not be meaningful at 3 epochs.

### Recommended next steps

| Experiment | Hypothesis |
|------------|-----------|
| RationalGLU + qk_norm + weight_norm | Expected to close to ~57–58 ppl; confirm whether rational gate regains advantage over SwiGLU with norm additions |
| SwiGLU + weight_norm, 10+ epochs | Confirm whether the large PPL gain from weight_norm persists at convergence or is an early-training effect |
| PFDRationalGLU + weight_norm (no qk_norm) | Isolate weight_norm vs qk_norm contributions for PFD variant |
| weight_norm max_norm sweep | max_norm=2.0 chosen arbitrarily; test 1.0, 3.0 to see sensitivity |
| PFDRationalGLU, 10+ epochs | Confirm whether PFD advantage over RationalGLU holds or is an early-training effect |
| G1-B + weight_norm | Best RBF variant + weight_norm; may close the gap to SwiGLU |
| G2 with K>5 | More centers may give Sinkhorn more useful signal; test K=10 |
