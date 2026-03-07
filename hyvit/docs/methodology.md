# HyViT: Hyperbolic Vision Transformer

## Motivation

Euclidean space is a poor fit for representing hierarchical structure — the volume of a ball grows polynomially with radius, whereas tree-structured data has exponentially many nodes at each depth. Hyperbolic space solves this: its volume grows exponentially, making it a natural embedding for hierarchies. CIFAR-10 has latent hierarchical structure (e.g., vehicles / animals / sub-categories). The hypothesis is that hyperbolic representations can achieve equivalent classification accuracy with fewer model parameters, or better accuracy at the same parameter budget, by virtue of a more geometrically appropriate inductive bias.

---

## Architecture Overview

```
Image (B, C, H, W)
  → Euclidean patch projection (Linear, R^d_model)
  → project_to_hyperboloid   →  H^{d_model}    ← manifold entry
  → N × LorentzTransformerBlock
  → LorentzLayerNorm
  → log_map_origin (CLS token)  →  T_o H  ≅  R^{d_model}   ← manifold exit
  → Linear classifier
```

The model enters the hyperboloid once (at the patch embedding) and exits once (before classification). All N transformer blocks operate natively on H^{d_model}. There are no intermediate exp/log map roundtrips.

---

## Mathematical Foundations

**The Lorentz model.** A point **x** ∈ H^n lives in R^{n+1} satisfying:

```
⟨x, x⟩_L = -x₀² + x₁² + ... + xₙ² = -1,   x₀ > 0
```

The Lorentz inner product is `⟨x, y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ`. The geodesic distance is `d(x,y) = arcosh(-⟨x,y⟩_L)`.

This model is chosen over the Poincaré ball because:
- The similarity measure is a **single sign flip** from the standard Euclidean dot product, making implementation and gradient flow straightforward
- Numerically stable: no Möbius additions, no divisions by `(1 - ‖x‖²)` that blow up near the boundary
- exp/log maps at the origin are simple closed forms

---

## Core Components

### LorentzLinear

Maps H^{d_in} → H^{d_out} without explicit exp/log maps:
```
y_s = W x_s + b            (Euclidean linear on spatial components)
y   = (sqrt(1 + ‖y_s‖²), y_s)   ∈ H^{d_out}
```
The time component is determined by the manifold constraint, not learned. All parameters are in `W` and `b`, so standard SGD applies.

### Lorentz Attention

Score function replaces the Euclidean dot product with the Lorentz inner product:
```
score(q, k) = -⟨q, k⟩_L / sqrt(d_head)
```
For points on the manifold, `⟨q,k⟩_L ≤ -1`, so scores are always ≥ `1/sqrt(d_head)`. Softmax normalization proceeds identically to standard attention.

Aggregation uses the **Lorentz centroid** — a closed-form approximation to the Fréchet mean:
```
z = Σᵢ attn_wᵢ · vᵢ    (ambient weighted sum in R^{d_head+1})
output = lorentz_normalize(z)    (project back to H^{d_head})
```
This is differentiable, avoids iterative Fréchet mean computation, and has been shown to be a valid approximation in prior hyperbolic network literature.

Head splitting/merging is done by slicing the spatial components, reshaping into heads, then recomputing the time component via the constraint `x₀ = sqrt(1 + ‖x_s‖²)`. Each head's sub-vector is itself a valid hyperboloid point.

### LorentzLayerNorm

Applies standard LayerNorm to the spatial components only (the time component is not a free variable), then reprojects:
```
x_s_norm = LayerNorm(x_s)     (learn gamma, beta over d_model dims)
output   = project_to_hyperboloid(x_s_norm)
```

### Hyperbolic Residual Connection

```
x = lorentz_normalize(x + sub_layer_output)
```
The ambient sum temporarily exits the manifold; `lorentz_normalize` reprojects by dividing by `sqrt(-⟨z,z⟩_L)` and enforcing `z₀ > 0`. This is faster and more stable than geodesic midpoints, and gradients flow cleanly through the normalization.

### Block Structure (pre-norm)

```
x = lorentz_normalize(x + MHSA(LorentzLayerNorm(x)))
x = lorentz_normalize(x + FFN(LorentzLayerNorm(x)))
```
The FFN applies GELU to spatial components between two LorentzLinear layers, with an intermediate `project_to_hyperboloid` after the activation.

### Patch Embedding

The "lift and reproject" pattern:
1. Flatten non-overlapping patches (4×4, giving 64 tokens for 32×32 input)
2. Euclidean linear: `R^{C·p²} → R^{d_model}`
3. `project_to_hyperboloid` → H^{d_model}
4. Prepend CLS token (learnable Euclidean vector, lifted to manifold)
5. Add positional embeddings to spatial components, reproject

Positional embeddings live in Euclidean space and are added before the final `project_to_hyperboloid`. This avoids defining a notion of "hyperbolic position encoding."

### Optimizer

Standard AdamW. The manifold structure is encoded in the forward pass (activations satisfy the hyperboloid constraint by construction), not in the parameters. Parameters are unconstrained Euclidean weights, so Riemannian optimizers are unnecessary. This is a key practical advantage over Poincaré-ball models where parameters sometimes live on the manifold.

---

## Scale

HyViT-Tiny: d_model=192, 12 blocks, 3 heads, mlp_ratio=4. Approximately 5M parameters. Trained for 100 epochs on CIFAR-10 with AdamW (lr=3e-4, weight decay=0.05), cosine LR schedule with 10-epoch warmup, label smoothing=0.1, gradient clipping at 1.0.

---

## Novelty Claims

1. **Fully hyperbolic internal representation.** All token representations remain on H^{d_model} throughout every transformer block. Prior hyperbolic transformer work typically uses hybrid designs — Euclidean attention with hyperbolic embeddings, or hyperbolic layers only at the output. HyViT has no Euclidean intermediate representations between input and output.

2. **Lorentz model for vision.** Prior hyperbolic ViT / image work (to the extent it exists) has used Poincaré ball models, which require Möbius operations and are numerically sensitive near the boundary. Grounding the architecture in the Lorentz model gives a cleaner structural parallel to standard dot-product attention — the score function is literally `-x₀y₀ + Σxᵢyᵢ` vs. `Σxᵢyᵢ`, a single sign flip.

3. **Manifold-preserving head split/merge.** Splitting multi-head attention without leaving the manifold requires recomputing the time component per head from the split spatial components. This keeps each head's representation a valid manifold point without an exp/log roundtrip.

4. **Manifold-free parameterization.** By structuring LorentzLinear so that parameters are Euclidean (`W`, `b` act on spatial components only) and the manifold constraint is imposed by construction in the forward pass, standard first-order optimizers apply without modification. This contrasts with geoopt-based Riemannian Adam approaches and simplifies the training stack considerably.

5. **Empirical question.** Whether hyperbolic geometry provides a meaningful inductive bias for vision classification (as opposed to graph/tree tasks where the benefit is well-established) remains an open empirical question. HyViT provides a controlled comparison: architecture identical to ViT-Tiny except the attention geometry, evaluated against a matched Euclidean baseline on CIFAR-10.
