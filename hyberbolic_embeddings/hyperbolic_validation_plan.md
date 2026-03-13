# Hyperbolic Geometry Validation Plan

## Goal
Validate the core claim: **Does hyperbolic geometry improve hierarchical representation quality vs. Euclidean space?**

If this fails, abandon the HYPER-G architecture.

---

## Dataset: Synthetic Tree Hierarchy

**Structure**: Balanced tree with known ground truth
- Nodes: 1000-5000
- Branching factor: 4
- Depth: 5-7
- Features: Random walk + Gaussian noise

**Ground truth**: Tree distance between any two nodes (shortest path)

**Why synthetic**: Controlled hierarchy. If hyperbolic can't win here, it won't win on messy real data.

---

## Model: Minimal Embedding Comparison

### Architecture Options

**1. Euclidean Baseline**
```python
class EuclideanEmbedding(nn.Module):
    def __init__(self, num_nodes, dim):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)

    def distance(self, u_idx, v_idx):
        u = self.emb(u_idx)
        v = self.emb(v_idx)
        return torch.norm(u - v, dim=-1)
```

**2. Hyperbolic Test (Poincaré Ball)**
```python
from geoopt import PoincareBall

class HyperbolicEmbedding(nn.Module):
    def __init__(self, num_nodes, dim, c=-1.0):
        super().__init__()
        self.ball = PoincareBall(c=c)
        # Initialize in tangent space at origin
        self.emb_tangent = nn.Embedding(num_nodes, dim)

    def forward(self, idx):
        # Map from tangent space to manifold
        tangent = torch.tanh(self.emb_tangent(idx)) * 0.9
        return self.ball.expmap0(tangent)

    def distance(self, u_idx, v_idx):
        u = self.forward(u_idx)
        v = self.forward(v_idx)
        return self.ball.dist(u, v)
```

---

## Task: Link Prediction

**Positive examples**: Connected node pairs (tree edges)
**Negative examples**: Random unconnected pairs

**Loss**:
```python
# Binary cross-entropy with distance-based logits
def compute_loss(model, pos_pairs, neg_pairs):
    pos_dist = model.distance(pos_pairs[:, 0], pos_pairs[:, 1])
    neg_dist = model.distance(neg_pairs[:, 0], neg_pairs[:, 1])

    # Lower distance = higher probability of edge
    pos_loss = F.softplus(pos_dist)  # -log(sigmoid(-dist))
    neg_loss = F.softplus(-neg_dist)  # -log(sigmoid(dist))

    return pos_loss.mean() + neg_loss.mean()
```

---

## Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Dimension | 16-64 | Hyperbolic needs d≥2; test if higher helps |
| Learning rate | 1e-3 | Standard for embedding tasks |
| Optimizer | RiemannianAdam (hyperbolic), Adam (Euclidean) | Required for manifold optimization |
| Batch size | 256-1024 | As large as fits |
| Epochs | 500-2000 | Train until convergence |
| Curvature (c) | -1.0 (fixed initially) | Learn later if basic setup works |

**Optimization Stability**:
- Clip gradients: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`
- Monitor for NaN: Check every 10 steps, halt if found
- Radial constraint: `||emb|| < 0.95 / sqrt(|c|)` after each step

---

## Evaluation Metrics

### Primary Metrics
1. **Link Prediction AUROC**: Area under ROC curve on held-out edges
2. **Distance Correlation**: Spearman ρ between learned distance and tree distance

### Secondary Metrics
3. **Embedding Norm vs. Depth**: Plot `||emb||` vs. tree depth
   - Hyperbolic: Should show exponential growth (or at least monotonic increase)
   - Euclidean: No expected relationship

4. **Visualization**: UMAP/tsne of embeddings, colored by depth
   - Hyperbolic: Should show radial stratification by depth
   - Euclidean: Random or mixed

### Success Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| AUROC | Hyperbolic > Euclidean + 5% | Hyperbolic captures structure better |
| Distance Correlation | ρ > 0.5 for hyperbolic | Learned distances correlate with hierarchy |
| Norm Growth | Monotonic with depth | Proper hyperbolic usage |

**If ANY metric fails**: Debug before proceeding.

---

## Implementation Checklist

### Phase 1: Setup (30 min)
- [ ] Install `geoopt`: `pip install geoopt`
- [ ] Install `networkx`: `pip install networkx`
- [ ] Verify imports work
- [ ] Create directory: `experiments/tree_validation/`

### Phase 2: Data Generation (30 min)
- [ ] Generate balanced tree with NetworkX
- [ ] Create train/val/test splits
- [ ] Generate positive and negative edge samples
- [ ] Save to disk for reproducibility

### Phase 3: Baseline (Euclidean) (1 hour)
- [ ] Implement `EuclideanEmbedding`
- [ ] Implement training loop
- [ ] Train until convergence
- [ ] Record AUROC and distance correlation
- [ ] Save model checkpoint

### Phase 4: Hyperbolic Test (2-3 hours)
- [ ] Implement `HyperbolicEmbedding` with `geoopt`
- [ ] Implement RiemannianAdam optimizer
- [ ] Add gradient clipping
- [ ] Add radial constraint enforcement
- [ ] Train until convergence
- [ ] Record same metrics as baseline
- [ ] **Check**: No NaN gradients? Embeddings within bounds?

### Phase 5: Analysis (30 min)
- [ ] Compare AUROC: Euclidean vs. Hyperbolic
- [ ] Compute distance correlation for both
- [ ] Plot embedding norm vs. depth
- [ ] Generate visualization
- [ ] Write results summary

---

## Expected Outcomes & Next Steps

### Scenario A: Hyperbolic Wins
**Result**: Hyperbolic AUROC > Euclidean + 5%, distance correlation strong

**Interpretation**: Hyperbolic geometry is beneficial for hierarchy. Proceed to:
1. Test on WordNet (real hierarchy)
2. Add graph structure (message passing)
3. Scale to multi-modal

### Scenario B: No Difference
**Result**: Metrics within noise

**Possible Causes**:
- Implementation bug (check gradient flow)
- Tree too small/simple
- Dimension too low
- Training insufficient

**Next Steps**:
- Verify implementation with `geoopt` examples
- Try larger tree (10k+ nodes)
- Try learned curvature
- If still no difference: Consider abandoning hyperbolic approach

### Scenario C: Hyperbolic Worse
**Result**: Euclidean outperforms

**Possible Causes**:
- Hyperbolic optimization is harder (needs more tuning)
- Initialization issue
- Curvature mismatch

**Next Steps**:
- Hyperparameter sweep (LR, init scale, curvature)
- Longer training
- Different curvature values
- If still worse: **Major red flag**—hyperbolic may not help

### Scenario D: Training Fails (NaN, divergence)
**Result**: Hyperbolic model unstable

**Possible Causes**:
- Embeddings hit boundary
- Gradient explosion
- Numerical precision issues

**Next Steps**:
- Add aggressive radial clipping
- Reduce learning rate 10×
- Use FP64 instead of FP32
- If still unstable: Hyperbolic optimization is impractical

---

## Key Files to Create

```
experiments/
├── tree_validation/
│   ├── data.py              # Tree generation
│   ├── models.py            # Euclidean & Hyperbolic embeddings
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Metrics computation
│   └── run_experiment.py    # Main script
└── results/
    ├── euclidean/           # Baseline results
    └── hyperbolic/          # Test results
```

---

## Resume Instructions

**When resuming**:
1. Check this file for current status
2. Run `experiments/tree_validation/run_experiment.py`
3. Compare results in `experiments/results/`
4. Make decision based on Scenario A/B/C/D

**Time estimate**: 4-6 hours total

---

## Questions for Future Us

1. Did hyperbolic show radial stratification (norm vs depth)?
2. Was training stable without manual intervention?
3. Did we test multiple curvature values?
4. Should we try WordNet next, or fix issues first?

---

*Last updated: Session break point*
*Next action: Implement Phase 1-2 (setup and data generation)*
