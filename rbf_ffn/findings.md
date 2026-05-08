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

### Recommended next steps

| Experiment | Hypothesis | Status |
|------------|-----------|--------|
| XSA + qk_norm (no wnorm) | Whether qknorm improves XSA | **Done** — 71.57 (§8.2) |
| XSA + qk_norm + orthogonal_ffn | Whether FFN orthogonalization stacks with XSA | **Done** — 69.87 (§8.5); ~1.7 ppl gain; single run, confirm |
| XSA + qk_norm + orthogonal_ffn + weight_norm | Whether dual orthogonality stacks with wnorm | **Done** — 55.57 ep2, 49.91 ep9 (§8.9) |
| XSA + qk_norm + weight_norm (no orthogonal_ffn) | Isolate XSA+wnorm without orthogonal_ffn | **Done** — 56.88 (§8.8) |
| XSA + MoE + qk_norm + weight_norm | Whether sparse MoE FFN improves over SwiGLU | **Done** — 47.31 ep2 (§9.2); new best at 3 epochs, but 4× FFN params |
| XSA + PFDRationalGLU + qk_norm + orthogonal_ffn | Stack best attention + best FFN + orthogonality | Priority 1 |
| MoE parameter-matched ablation | Isolate MoE structural gain from param count | Priority 2 — run SwiGLU at 4× FFN width (ffn_hidden=2752) |
| PFDRationalGLU, 10+ epochs | Confirm whether PFD advantage over RationalGLU holds or is an early-training effect | Done (§7) — advantage disappears |
| PFDRationalGLU, per-channel PFD params | Current params are shared; per-channel or per-head may widen the gap further | Low priority |
| FirstOrderPFDRational, phi initialization | ep0 train PPL spike (9153) suggests sin saturation; test larger phi init or learnable scale | Low priority |
| G1-B, 10+ epochs | Confirm gap closure; check if PPL crosses SwiGLU at longer horizon | Deprioritized (27 ppl behind best) |
| G2 with K>5 | More centers may give Sinkhorn more useful signal; test K=10 | Deprioritized |

## §6 Normalization Ablations (d_model=256, WikiText-103, 3 epochs unless noted)

Building on §5, these runs add normalization on top of the best FFN variants.

### §6.1 QK norm + weight norm — headline results

| Variant | Val PPL (ep 0) | Val PPL (ep 2) | Δ vs unnorm baseline | Time/epoch |
|---------|----------------|----------------|----------------------|------------|
| **SwiGLU + qk_norm + weight_norm** | 114.41 | **58.16** | **−23.1%** | ~1330s |
| SwiGLU + weight_norm | 120.19 | 58.97 | −22.1% | ~1223s |
| PFDRationalGLU + qk_norm + weight_norm | 117.34 | 58.91 | −22.2% | ~2088s |
| PFDRationalGLU + qk_norm + weight_norm + act_norm | 121.58 | 59.82 | −21.0% | ~2113s |
| SwiGLU + qk_norm + adaptive_weight_norm | 118.74 | 60.67 | −19.8% | ~1346s |
| PFDRationalGLU + qk_norm | 136.46 | 72.25 | −4.5% | ~2126s |
| RationalGLU + qk_norm | 137.24 | 73.51 | −2.9% | ~1515s |
| SwiGLU + qk_norm | 141.84 | 75.14 | −0.7% | ~1307s |

**Key conclusions:**

- **Linear weight normalization is the dominant improvement** (−21.8 ppl over baseline alone, outweighing all FFN activation variants)
- **Weight normalization erases PFDRationalGLU's advantage.** PFDRationalGLU + weight_norm (58.91) is statistically tied with SwiGLU + weight_norm (58.97). The rational gate advantage from §5 (−3.5%) disappears under normalization.
- **QK norm adds a consistent but modest −0.5 to −0.9 ppl** on top of weight norm. Worth including; negligible cost.
- **Activation coefficient normalization (act_norm) is slightly harmful** (+0.91 ppl on PFDRationalGLU+qknorm+wnorm). Do not use with weight_norm.
- **Adaptive weight norm underperforms fixed norm** (60.67 vs 58.16). Depth-scaling the target norm with a phase-aware derivative correction does not help over a uniform target of 2.0.

### §6.2 max_only weight norm mode (2026-03-29)

`linear_weight_norm_max_only: true` truncates rows only when their norm *exceeds* the target (no scaling up). Result: val PPL 75.54 at ep2 — catastrophic regression vs full weight_norm (58.16). The bidirectional constraint is essential; clipping alone is insufficient.

### §6.3 FirstOrderPFDRational + qk_norm + weight_norm (2026-03-29, 10 epochs)

| Epoch | Val PPL |
|-------|---------|
| 0 | 139.02 |
| 1 | 100.71 |
| 2 | 85.06 |
| 3 | 74.17 |

At ep3, FirstOrderPFDRational+qknorm+wnorm reaches 74.17 — behind the unnormalized SwiGLU baseline (75.68 at ep2 ≈ ep3 PPL ~68). The sin gate does not benefit from weight norm as strongly as SwiGLU.

### §6.4 QKV SiLU + pre-LM-head SiLU (2026-03-30)

SwiGLU + qk_norm + weight_norm + qkv_silu + pre_lm_head_silu: ep5 = 54.33, compared to SwiGLU+qknorm+wnorm ep5 ≈ 53-60 (depending on run). Adding SiLU to QKV projections and before the LM head slows convergence without clear long-term benefit.

### §6.5 Kronecker MLP (2026-03-30)

| Config | ep 2 Val PPL | Notes |
|--------|-------------|-------|
| SwiGLU + qknorm + wnorm + kronecker_mlp | 100.19 | Diverged |
| SwiGLU + qknorm + kronecker_mlp (no wnorm) | 113.09 | Diverged |

Replacing MLP projections (up_proj, down_proj) with KroneckerLinear while applying weight normalization causes training divergence. The Kronecker factorization is incompatible with the current weight norm constraint — the per-row norm normalization of a factored weight is not well-defined. Needs rethinking before further experimentation.

## §7 Extended Training (10 epochs, SwiGLU vs PFDRationalGLU)

Both variants trained with qk_norm + weight_norm for 10 epochs. Multiple runs to measure variance.

| Run | Variant | Val PPL ep2 | Val PPL ep9 |
|-----|---------|------------|------------|
| 20260327_214545 | PFDRationalGLU+qknorm+wnorm | 80.55 | **41.72** |
| 20260328_223729 | PFDRationalGLU+qknorm+wnorm | 79.54 | **41.69** |
| 20260328_180644 | SwiGLU+qknorm+wnorm | 78.96 | **41.40** |
| 20260328_132202 | SwiGLU+qknorm+wnorm | 58.19† | 44.68 |
| 20260331_122637 | SwiGLU+qknorm+wnorm | — | 49.06 |
| 20260331_154942 | SwiGLU+qknorm+wnorm | — | 47.97 |
| 20260324_164546 | SwiGLU+qknorm+wnorm | 58.16 | **44.14** |
| 20260325_133825 | PFDRationalGLU+qknorm+wnorm+actnorm | — | 44.77 |

† 20260328_132202 shows anomalous ep2=58.19 then regression to 78.42 at ep3, indicating a training instability (likely hardware I/O interrupt); ep9=44.68 is trustworthy.

**Key conclusion:** PFDRationalGLU's 3-epoch advantage (−3.5% vs SwiGLU) is an early-training effect. At 10 epochs, SwiGLU+qknorm+wnorm matches or beats PFDRationalGLU+qknorm+wnorm (~41.40 vs 41.69 on best runs). The PFD rational gate provides no lasting efficiency advantage at moderate training length. For long training, SwiGLU+qknorm+wnorm is preferred (lower training cost, same convergence).

**Run-to-run variance is substantial:** SwiGLU+qknorm+wnorm ep9 ranges from 41.40 to 49.06 across runs on the same hardware. Hardware/IO state has a large effect on both convergence and per-epoch time. Best-of-N reporting is more informative than single-run results.

Trajectory of best 10-epoch SwiGLU+qknorm+wnorm run (20260328_180644):

| Epoch | Val PPL | Time (s) |
|-------|---------|----------|
| 0 | 126.09 | 1170 |
| 1 | 91.49 | 1151 |
| 2 | 78.96 | 1146 |
| 3 | 69.22 | 1147 |
| 4 | 62.22 | 1174 |
| 5 | 54.53 | 1145 |
| 6 | 49.30 | 1150 |
| 7 | 45.16 | 1150 |
| 8 | 42.37 | 1156 |
| 9 | **41.40** | 1159 |

## §8 Attention Variants

### §8.1 Exclusive Self-Attention (XSA, 2026-04-15)

XSA runs standard causal attention to produce Y, then subtracts from each output the component lying along its own value direction:

```
Vn = V / ||V||          (per head, per position)
Z  = Y - (Y · Vn) Vn   (Gram-Schmidt step)
out = o_proj(Z)
```

Tested as `xsa_swiglu_d256` — XSA attention + SwiGLU FFN, **no normalization** (no qk_norm, no weight_norm).

| Epoch | Train PPL | Val PPL | Time (s) |
|-------|-----------|---------|----------|
| 0 | 3422 | 138.55 | 1181 |
| 1 | 121.27 | 91.65 | 1160 |
| 2 | 86.35 | **72.41** | 1137 |

**Comparison to unnormalized 3-epoch baselines:**

| Variant | Val PPL (ep 2) | Δ vs SwiGLU |
|---------|----------------|-------------|
| XSA + SwiGLU | **72.41** | **−4.3%** |
| PFDRationalGLU | 73.00 | −3.5% |
| RationalGLU | 74.37 | −1.7% |
| SwiGLU (baseline) | 75.68 | — |

XSA outperforms all FFN activation variants at 3 epochs without any normalization, suggesting the attention orthogonalization provides a stronger inductive bias than rational gate activations. Per-epoch time (~1160s avg) is similar to baseline SwiGLU (~1234s at ep1-2), with slight speedup likely from the in-place subtraction replacing the projection dropout path.

XSA + qk_norm results (with and without orthogonal_ffn) are covered in §8.2 and §8.5 below. XSA + qknorm + weight_norm remains the obvious next experiment.

### §8.2 XSA + QK Norm (no weight_norm, no orthogonal_ffn) (2026-04-15)

Two runs of XSA + SwiGLU with `qk_norm: true`, `orthogonal_ffn: false`:

| Run | Val PPL (ep 0) | Val PPL (ep 1) | Val PPL (ep 2) | Time/epoch |
|-----|----------------|----------------|----------------|------------|
| 20260415_135856 | 134.90 | 90.72 | 71.87 | ~1161s |
| 20260415_145723 | 135.36 | 90.62 | **71.57** | ~1166s |

**Key conclusions:**

- **QK norm adds ~0.5–1 ppl improvement over XSA without qk_norm** (72.41 → 71.57–71.87). Modest effect, consistent with qk_norm's ~0.5 ppl gain on SwiGLU (§6.1).
- **Run-to-run variance is low** (71.57–71.87, spread of 0.3 ppl) — tighter than longer runs; these two runs are a reliable estimate.
- The XSA + qk_norm baseline (no orthogonal_ffn) is **71.57 PPL** (best of 2).

### §8.3 XSA + LeakyReLUSq + QK Norm (2026-04-15)

`LeakyReLUSq` is a gated FFN using `leaky_relu(x)²` as the gate activation:

```
gate = leaky_relu(gate_proj(x))²   # always non-negative
out  = down_proj(gate * up_proj(x))
```

Two runs with `qk_norm: true`:

| Run | Val PPL (ep 0) | Val PPL (ep 1) | Val PPL (ep 2) | Time/epoch |
|-----|----------------|----------------|----------------|------------|
| 20260415_180252 | 144.04 | 99.21 | 79.97 | ~1163s |
| 20260415_190149 | 144.31 | 98.97 | **79.30** | ~1165s |

**Key conclusions:**

- **LeakyReLUSq significantly underperforms SwiGLU in the XSA context** (79.30 vs 71.57 for XSA+qknorm+SwiGLU). The ~8 ppl gap is larger than the SwiGLU-vs-SwiGLU run-to-run variance.
- **Epoch 0 PPL (144) is similar to XSA+SwiGLU without qknorm (138)** but higher than XSA+SwiGLU+qknorm (130–135), suggesting the LeakyReLUSq gate initializes slightly worse.
- **Always-non-negative activation is a likely culprit**: `leaky_relu(x)²` collapses negative gate values to near-zero, halving the effective dynamic range of the gate. SwiGLU's `silu(x) = x·σ(x)` can produce negative gate values, providing richer gating signal.
- LeakyReLUSq is deprioritized; SwiGLU remains the FFN of choice for XSA experiments.

### §8.4 Large Vocabulary Experiments (2026-04-04)

Two runs using `vocab_size: 65536` and `tie_embeddings: false` (different tokenizer from all prior experiments):

| Run | Variant | Val PPL (ep 2) | Notes |
|-----|---------|----------------|-------|
| 20260404_095151 | SwiGLU + qknorm + wnorm + untied_emb | 36.79 | Anomalous — likely resumed checkpoint† |
| 20260404_105557 | SwiGLU + qknorm + wnorm + kronecker_lm_head + untied_emb | 138.21 | Diverged |

† Epoch 0 train PPL = 147.59 vs ~800 expected for a fresh wnorm run; training resumed from a partially-trained checkpoint, making the ep2 result misleading.

**Key conclusions:**

- **Results are not directly comparable to prior experiments** (50257 vocab) — different tokenizer, different PPL scale.
- **Kronecker LM head with vocab=65536 diverges.** The Kronecker factorization of the LM head (`V×H → A⊗B`) is unstable with weight_norm and untied embeddings at this scale. Epoch 0 val PPL = 263.95, never recovers.
- These experiments open a separate evaluation axis (65536-vocab tokenizer). Future work should establish a clean baseline on this tokenizer before comparing variants.

### §8.5 Orthogonal FFN Wrapper (2026-04-15)

`OrthogonalMLPWrapper` wraps any FFN so its output is orthogonal to its input (the pre-norm residual stream):

```
y          = mlp(x)
y_parallel = (y · x / (x · x + ε)) * x   # component along x
out        = y - y_parallel               # perpendicular component only
```

The additive FFN update is guaranteed to lie in the subspace orthogonal to `x` at every position and layer. This is the FFN analogue of XSA: XSA subtracts the component of attention output along the value vector; `OrthogonalMLPWrapper` subtracts the component of FFN output along the input token representation.

Tested as XSA + SwiGLU + qk_norm + `orthogonal_ffn: true` (run `20260415_231629_387840_xsa_swiglu_qknorm_orthogonal_d256`):

| Epoch | Train PPL | Val PPL | Time (s) |
|-------|-----------|---------|----------|
| 0 | 3076 | 130.88 | 1188 |
| 1 | 115.8 | 88.28 | 1156 |
| 2 | 83.20 | **69.87** | 1157 |

**Comparison to XSA + qk_norm without orthogonal_ffn (§8.2):**

| Variant | Val PPL (ep 2) | Δ |
|---------|----------------|---|
| XSA + qk_norm + orthogonal_ffn | **69.87** | — |
| XSA + qk_norm (best of 2) | 71.57 | +1.70 |
| XSA (no norm) | 72.41 | +2.54 |
| SwiGLU baseline | 75.68 | +5.81 |

**Key conclusions:**

- **Orthogonal FFN adds ~1.7 ppl improvement over XSA + qk_norm** (71.57 → 69.87). This is a single run; the improvement exceeds the XSA+qknorm run-to-run spread (~0.3 ppl between those two runs), making it likely real, but confirmation is warranted.
- **Dual orthogonality stacks additively.** XSA contributes ~0.8 ppl over SwiGLU (72.41→71.57 with qknorm), and orthogonal_ffn adds a further ~1.7 ppl. The two orthogonality constraints (attention and FFN) appear to provide complementary signal — consistent with them acting on different sources of redundancy in the residual stream.
- **Epoch 0 PPL (130.88) is the lowest of all XSA+qknorm variants**, suggesting orthogonal_ffn improves early gradient flow, perhaps by preventing the FFN from collapsing toward identity-like updates.
- **No training overhead**: per-epoch time (~1167s) matches the other XSA+qknorm runs (~1161–1166s). The projection is a single dot product per position with no new parameters.
- **XSA + qk_norm + orthogonal_ffn at 69.87 PPL is the new best no-weight-norm result** (−7.7% vs SwiGLU baseline).
- Priority next: XSA + qk_norm + orthogonal_ffn + weight_norm (expected ~52–54 PPL at ep2 if wnorm gives similar ~17 ppl gain as on SwiGLU).

### §8.6 Polar Attention + Polar FFN (2026-03-27)

Three polar configurations tested:

| Config | Val PPL (ep 2) | Notes |
|--------|----------------|-------|
| polar_mlp (no norm) | 95.11 | standard attn + Polar FFN, no norm |
| polar_mlp + qknorm + wnorm | 85.84 | standard attn + Polar FFN + norm |
| polar_full (polar attn + polar FFN) | — | diverged after ep0 (214.20) |

**Polar MLP:** Projects token vectors onto the unit sphere, computes cosine similarity against learned key directions, gates with a fixed-sharpness sigmoid (threshold learnable). All magnitude information is discarded. Result is poor — 95.11 without norm, 85.84 with norm — far behind SwiGLU at comparable configurations. Discarding magnitude is harmful for language modeling.

**Polar attention:** Combined polar-attn + polar-FFN diverged at ep0 (val PPL 214.20) and was abandoned. Polar attention is likely unstable when both attention and FFN discard magnitude.

### §8.7 Gated Orthogonal FFN Variant (2026-04-16)

`GatedOrthogonalMLPWrapper` extends `OrthogonalMLPWrapper` by replacing the hard subtraction of the parallel component with a learned gate. Given `y = mlp(x)` and scalar projection `c = (y · x) / (||x||² + ε)`, the gate controls how much of the input direction is amplified or erased rather than unconditionally removing it.

Tested as XSA + SwiGLU + qk_norm + `gated_orthogonal_ffn: true`, `gate_activation: tanh` (run `20260416_095111_442806_xsa_swiglu_qknorm_gated_orthogonal_d256`):

| Epoch | Val PPL | Time (s) |
|-------|---------|----------|
| 0 | 130.33 | 1204 |
| 1 | 89.08 | 1193 |
| 2 | **69.92** | 1201 |

**Comparison to plain OrthogonalMLPWrapper (§8.5):**

| Variant | Val PPL (ep 2) |
|---------|----------------|
| XSA + qk_norm + orthogonal_ffn (plain) | **69.87** |
| XSA + qk_norm + gated_orthogonal_ffn | 69.92 |

**Key conclusion:** Gated orthogonal provides no improvement over the plain wrapper (69.92 vs 69.87, within noise). The tanh gate over the scalar projection does not help — the model appears not to benefit from the ability to amplify or partially retain the input direction. The plain `OrthogonalMLPWrapper` is preferred.

### §8.8 XSA + QK Norm + Weight Norm (Priority 2 Complete, 2026-04-30)

First run isolating XSA + weight_norm without orthogonal_ffn (run `20260430_132442_213125_xsa_swiglu_qknorm_wnorm_d256`):

| Epoch | Val PPL | Time (s) |
|-------|---------|----------|
| 0 | 114.11 | 1393 |
| 1 | 72.06 | 1363 |
| 2 | **56.88** | 1364 |

**Comparison to SwiGLU + qknorm + wnorm (§6.1):**

| Variant | Val PPL (ep 0) | Val PPL (ep 2) | Δ vs SwiGLU+norm |
|---------|----------------|----------------|------------------|
| **XSA + qk_norm + weight_norm** | 114.11 | **56.88** | **−1.28** |
| SwiGLU + qk_norm + weight_norm | 114.41 | 58.16 | — |

**Key conclusions:**

- **XSA + wnorm (56.88) beats SwiGLU + wnorm (58.16) by 1.28 ppl.** XSA's orthogonality advantage persists under weight normalization — the gain is structural, not an optimization artifact.
- **Epoch 0 PPL (114.11) is nearly identical to SwiGLU+wnorm (114.41)**, confirming weight norm dominates initialization quality regardless of attention variant.
- **Note: slower run (~1373s/epoch vs ~1330s for SwiGLU+wnorm)** — hardware state degraded relative to April 15–16 runs; this is not architectural overhead.
- This closes Priority 2: XSA adds ~1.3 ppl over SwiGLU under weight normalization.

### §8.9 XSA + QK Norm + Weight Norm + Orthogonal FFN (Priority 1 Complete, 10 epochs, 2026-04-16)

Run `20260416_105724_190084_xsa_swiglu_qknorm_wnorm_orthogonal_d256` — 10-epoch trajectory:

| Epoch | Val PPL | Time (s) |
|-------|---------|----------|
| 0 | 110.73 | 1238 |
| 1 | 70.29 | 1213 |
| 2 | **55.57** | 1197 |
| 3 | 53.79 | 1188 |
| 4 | 52.96 | 1164 |
| 5 | 52.07 | 1177 |
| 6 | 51.42 | 1189 |
| 7 | 50.92 | 1190 |
| 8 | 50.30 | 1190 |
| 9 | **49.91** | 1192 |

**ep2 comparison:**

| Variant | Val PPL (ep 2) | Δ vs SwiGLU+wnorm |
|---------|----------------|-------------------|
| XSA + qk_norm + wnorm + orthogonal_ffn | **55.57** | **−2.59** |
| XSA + qk_norm + wnorm (§8.8) | 56.88 | −1.28 |
| SwiGLU + qk_norm + wnorm | 58.16 | — |

**ep9 comparison:**

| Variant | Val PPL (ep 9) | Notes |
|---------|----------------|-------|
| SwiGLU + qk_norm + wnorm | **41.40** | best of 4 runs |
| XSA + qk_norm + wnorm + orthogonal_ffn | 49.91 | single run, still declining |

**Key conclusions:**

- **At ep2, dual orthogonality + wnorm achieves 55.57** — 2.59 ppl better than SwiGLU+wnorm (58.16). Orthogonal_ffn adds 1.31 ppl over XSA+wnorm alone (56.88 → 55.57), slightly less than its 1.70 ppl gain without wnorm (§8.5).
- **At ep9, 49.91 trails SwiGLU+wnorm best (41.40) by 8.5 ppl.** This is a single run and PPL is still declining at ep9 — but the convergence trajectory is substantially slower than SwiGLU+wnorm (which reached 41.40 at ep9 on its best run). The orthogonal_ffn's ep2 advantage does not translate to a 10-epoch advantage.
- **Epoch 0 PPL (110.73) is the lowest of all wnorm variants**, consistent with orthogonal_ffn improving early gradient flow.
- The Priority 1 expected range (~52–54 PPL at ep2) was slightly optimistic; actual is 55.57. Weight norm's ~17 ppl gain stacks additively with both XSA and orthogonal_ffn at ep2.

## §9 Mixture of Experts FFN (2026-05-04/08)

`SparseMoEFFN` routes each token independently to `top_k` of `n_experts` SwiGLU experts, outputting the softmax-weighted sum of the selected experts' outputs.

**Parameter note:** With n_experts=4, top_k=2, ffn_hidden=688, d_model=256: total FFN params ≈ 4 × 528K = 2.1M — roughly **4× the parameter count of a standard SwiGLU FFN** (528K). Results are not parameter-matched; gains must be interpreted in this context.

### §9.1 XSA + MoE + QK Norm (no weight_norm, 2026-05-04)

Run `20260504_225655_397364_xsa_moe_qknorm_d256` (n_experts=4, top_k=2):

| Epoch | Val PPL | Time (s) |
|-------|---------|----------|
| 0 | 127.12 | 1741 |
| 1 | 84.43 | 1706 |
| 2 | **64.73** | 1749 |

**Comparison to no-wnorm ep2 results:**

| Variant | Val PPL (ep 2) | FFN params | Time/epoch |
|---------|----------------|------------|------------|
| **XSA + MoE + qk_norm** | **64.73** | ~4× SwiGLU | ~1732s |
| XSA + qk_norm + orthogonal_ffn | 69.87 | 1× SwiGLU | ~1167s |
| XSA + qk_norm (best of 2) | 71.57 | 1× SwiGLU | ~1163s |
| SwiGLU (baseline) | 75.68 | 1× SwiGLU | ~1234s |

MoE reaches 64.73 — 5.14 ppl better than the best parameter-matched no-wnorm result (XSA+qknorm+orthogonal, 69.87), at the cost of 4× FFN parameters and ~49% higher per-epoch time.

### §9.2 XSA + MoE + QK Norm + Weight Norm (2026-05-08)

Run `20260508_085651_652695_xsa_moe_qknorm_wnorm_d256`:

| Epoch | Val PPL | Time (s) |
|-------|---------|----------|
| 0 | 102.68 | 1758 |
| 1 | 62.37 | 1725 |
| 2 | **47.31** | 1721 |

**Comparison to wnorm ep2 results:**

| Variant | Val PPL (ep 2) | FFN params | Time/epoch |
|---------|----------------|------------|------------|
| **XSA + MoE + qk_norm + wnorm** | **47.31** | ~4× SwiGLU | ~1735s |
| XSA + qk_norm + wnorm + orthogonal_ffn | 55.57 | 1× SwiGLU | ~1197s |
| XSA + qk_norm + wnorm | 56.88 | 1× SwiGLU | ~1373s |
| SwiGLU + qk_norm + wnorm | 58.16 | 1× SwiGLU | ~1330s |

**Key conclusions:**

1. **XSA + MoE + wnorm achieves 47.31 at ep2 — the new best 3-epoch result by a wide margin** (10.85 ppl better than SwiGLU+wnorm at 58.16). Even the best 10-epoch SwiGLU+wnorm result (41.40) is only 6.1 ppl ahead of this 3-epoch MoE result.
2. **Gain is not parameter-free.** MoE uses ~4× the FFN parameters and ~30% more time/epoch. The correct null comparison is SwiGLU at equivalent parameter budget (ffn_hidden≈2752), which has not been run.
3. **Weight normalization works well with MoE.** Epoch 0 val PPL (102.68) is lower than SwiGLU+wnorm (114.41); the router appears to benefit from normalized expert weights.
4. **Router load balance not yet monitored.** Expert collapse (tokens concentrating on 1–2 experts) is a known MoE failure mode. Monitoring router entropy is a priority before drawing strong conclusions.
5. **Priority next: parameter-matched ablation.** Run SwiGLU at ffn_hidden=2752 (4× budget) and MoE with smaller experts (n_experts=4, ffn_hidden=172 ≈ 1× budget) to isolate the structural benefit of sparse routing from raw parameter count.
