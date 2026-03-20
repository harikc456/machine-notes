---
name: KromHC Transformer on WikiText-103
description: Design for building and training a transformer with Kronecker head mixing on WikiText-103
type: design
---

# KromHC Transformer on WikiText-103 — Design Specification

## 1. Overview

Build and train a transformer language model on WikiText-103 with **Kronecker-factored head mixing (KromHC)** in the attention layer. The research track is isolated from the existing `rbf_ffn/` FFN experiments, enabling parallel exploration of attention head mixing as an orthogonal research direction.

### Research Goals (Sequential Phases)
1. **POC**: Validate end-to-end training with KromHC (30 min)
2. **Baseline Comparison**: Measure test perplexity vs. standard attention (8h, 3 seeds)
3. **Ablation Suite**: Isolate head mixer contribution via variants (24h, 3 seeds × 2 variants)
4. **Publication-Ready**: Rigorous comparison with statistical testing and scaling experiments (72h)

---

## 2. Module Architecture

### 2.1 Directory Structure

```
kromhc_transformer/
├── __init__.py
├── config.py                    # RBFFFNConfig-like config class
├── models/
│   ├── __init__.py
│   ├── attention.py             # CausalSelfAttention (RoPE + QK norm)
│   ├── head_mixer.py            # KromHCHeadMixer implementation
│   ├── llama_ffn.py             # SwiGLU FFN
│   ├── transformer_block.py      # KromHCBlock: norm1 → attn → mixer → residual → norm2 → ffn → residual
│   └── model.py                 # CausalLM with block dispatch
├── data.py                      # Data loading (import from rbf_ffn or duplicate)
├── train.py                     # Training loop with dual optimizer (Muon + AdamW)
├── configs/
│   ├── baseline.yaml            # Standard attention, no KromHC
│   ├── kromhc_small.yaml        # KromHC enabled, 50M params
│   ├── kromhc_medium.yaml       # KromHC enabled, 100M params
│   └── kromhc_large.yaml        # KromHC enabled, 200M params (optional scaling)
├── experiments/
│   ├── README.md                # Experiment protocols & running instructions
│   ├── scripts/
│   │   ├── requirements.txt      # Pinned dependencies
│   │   ├── utils.py             # Shared utilities (set_seeds, logging)
│   │   ├── exp_01_poc.py        # Phase 1: POC validation
│   │   ├── exp_02_baseline_vs_kromhc.py  # Phase 2: Comparison study
│   │   ├── exp_03_ablations.py  # Phase 3: Ablation variants
│   │   └── exp_04_final.py      # Phase 4: Publication-ready rigorous experiments
│   └── results/
│       ├── *.json               # Structured experiment logs
│       └── plots/               # Publication-quality plots (PDF + PNG)
├── findings.md                  # Main research deliverable (updated post-experiments)
└── tests/
    ├── test_head_mixer.py       # KromHCHeadMixer unit tests
    ├── test_transformer_block.py # KromHCBlock integration tests
    └── test_model.py            # End-to-end model tests
```

### 2.2 Core Components

#### **CausalSelfAttention** (adapted from `rbf_ffn/`)
- Multi-head causal self-attention with RoPE
- **QK Normalization**: RMSNorm applied to Q and K (default enabled via `cfg.qk_norm=True`)
- FlashAttention support (graceful fallback to Math backend)
- Input/output: (B, seq_len, d_model) → (B, seq_len, d_model)
- Returns: attention output only (mixing happens downstream)

#### **KromHCHeadMixer** (new)
- Takes attention head outputs: (B×seq_len, n_heads, head_dim)
- Builds Kronecker-factored doubly-stochastic permutation matrices:
  - Auto-factor n_heads into powers of 2 (e.g., 8 → [2, 2, 2])
  - Per-factor: tiny permutation basis (2! = 2 for factor=2) + context-dependent weights
  - Small MLPs generate convex weights from token-level context (mean of heads)
  - Final H: Kronecker product chain → (B×seq_len, n_heads, n_heads)
- Output: mixed heads (B×seq_len, n_heads, head_dim) + mixing matrix H (for analysis)
- Parameters: O(K × hidden_dim) where K = log₂(n_heads), hidden_dim ≈ 32

#### **KromHCBlock** (new transformer block)
```
x = x + CausalSelfAttention(norm1(x))        # (B, seq_len, d_model)
x_attn = reshape to (B×seq_len, n_heads, head_dim)
x_mixed, H = KromHCHeadMixer(x_attn)         # Mix heads
x = x + project(reshape(x_mixed))             # Residual + project back
x = x + SwiGLU(norm2(x))                      # FFN
return x
```

**Pre-norm, no biases** (consistent with `rbf_ffn/`)

#### **CausalLM** (adapted from `rbf_ffn/`)
- Block dispatch:
  - `model_type="baseline"` → standard `LlamaBlock` (no head mixing)
  - `model_type="kromhc"` → new `KromHCBlock` (head mixing enabled)
- Architecture: token_embedding → N × Block → RMSNorm → lm_head (weight-tied)
- Same optimizer grouping as `rbf_ffn/` (Muon for 2D params, AdamW otherwise)

---

## 3. Configuration

**RBFFFNConfig extended for KromHC:**

New fields:
```python
model_type: str = "kromhc"      # "baseline" | "kromhc"
use_kromhc: bool = True          # Soft toggle (for ablation experiments)
qk_norm: bool = True             # Default: enable QK normalization
d_model: int = 256               # 256, 512, or 1024 depending on scale
n_heads: int = 8                 # Must be power of 2
n_layers: int = 6 | 12 | 24      # Depth scaling
ffn_hidden: int = 688            # FFN intermediate dim (4× d_model typical)
```

Reused fields (same as `rbf_ffn/`):
- `seq_len: int = 512`
- `vocab_size: int = 50257` (r50k_base)
- `batch_size: int = 32`
- `n_epochs: int = 10`
- `dropout: float = 0.1`
- Optimizer: `muon_lr=0.02`, `adamw_lr=3e-4`, `adamw_wd=0.1`
- LR schedule: warmup (2%) + cosine annealing

**Example configs:**

`baseline.yaml`:
```yaml
model_type: baseline
qk_norm: true
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
batch_size: 32
n_epochs: 10
```

`kromhc_small.yaml`:
```yaml
model_type: kromhc
use_kromhc: true
qk_norm: true
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
batch_size: 32
n_epochs: 10
```

---

## 4. Dataset & Training

### 4.1 Dataset
- **WikiText-103** (standard): split into train/val/test
- **Tokenizer**: r50k_base (tiktoken)
- **Sequence length**: 512 tokens
- **Caching**: Download & tokenize once, reuse cached tensors

### 4.2 Training Loop
- **Optimizer**: Dual (Muon for 2D, AdamW for rest) — matches `rbf_ffn/`
- **LR schedule**: Linear warmup (2% steps) + cosine annealing to 0.1× base LR
- **Mixed precision**: bfloat16 on CUDA
- **Gradient clipping**: 1.0
- **Metrics**: loss (nats/token), perplexity, wall-clock time, VRAM peak
- **Checkpointing**: Save every epoch (optional; mainly final state)

### 4.3 Hardware Assumptions
- GPU: NVIDIA RTX 5060 Ti (16 GB VRAM)
- RAM: 16 GB (sufficient for batch_size=32, seq_len=512)
- Scaling: models up to 200M params fit with gradient accumulation if needed

---

## 5. Experiment Phases

### Phase 1: POC (30 min, ~1 GPU hour)
**Hypothesis**: KromHC integrates without crashes; training loop is sound.

- **Model**: 50M params (d_model=256, n_heads=8, n_layers=6)
- **Data**: WikiText-103 train split only (fast iteration)
- **Duration**: 1 epoch
- **Metrics**: loss progression, no exceptions
- **Pass criterion**: Loss decreases smoothly, no OOM, no NaNs
- **Output**: Confirms architecture is correct

### Phase 2: Baseline vs. KromHC Comparison (8 hours, 3 seeds)
**Hypothesis**: KromHC does not degrade test perplexity vs. baseline (H0: ppl_kromhc > ppl_baseline + 1.0 nats).

- **Models**:
  - Baseline: 100M params, no KromHC
  - KromHC: 100M params, with head mixing
- **Data**: Full WikiText-103 (train + val + test)
- **Duration**: 10 epochs per seed, 3 seeds (42, 43, 44)
- **Primary metrics**:
  - test_ppl (main decision)
  - val_ppl (per-epoch tracking)
  - train_loss
- **Statistical test**: Two-sample t-test (mean ± std across seeds)
- **Pass criterion**: H0 not rejected (KromHC ≤ baseline + 1% margin)
- **Outputs**: JSON artifacts, test curves

### Phase 3: Ablations (24 hours, 3 seeds × 2 variants)
**Hypothesis**: Head mixing contributes measurably to model expressiveness.

- **Variants**:
  - A: KromHC with full mixing (baseline KromHC)
  - B: KromHC with mixing disabled (use_kromhc=False) — equivalent to baseline but same code path
- **Metric** (new): Mixing matrix entropy, head information variance, effective rank of head outputs
- **Data**: Full WikiText-103, 10 epochs
- **Duration**: ~24h (3 seeds × 2 variants)
- **Output**: Ablation table showing ppl + mixing diagnostics

### Phase 4: Publication-Ready (72 hours, 3 seeds × 3 sizes)
**Hypothesis**: KromHC scales consistently; results are statistically significant and reproducible.

- **Model sizes**:
  - Small: 50M params
  - Medium: 100M params
  - Large: 200M params (if VRAM allows)
- **Data**: Full WikiText-103
- **Duration**: 10 epochs × 3 seeds × 3 sizes
- **Metrics**: test_ppl ± std, wall-clock time, VRAM, FLOPs estimate
- **Analysis**: Scaling curves, confidence intervals, significance tests
- **Outputs**: Publication-quality plots (PDF + PNG), detailed results table

---

## 6. Testing Strategy

### Unit Tests
- **KromHCHeadMixer**:
  - Output shape correctness: (bs, n_heads, head_dim)
  - Doubly-stochastic property: row & column sums = 1
  - Gradient flow: backprop through mixing matrix
  - Edge cases: n_heads ∈ {2, 4, 8, 16}

### Integration Tests
- **KromHCBlock**:
  - Forward pass: input (B, seq_len, d_model) → output (B, seq_len, d_model)
  - Residual connections: output ≈ input for untrained weights
  - Backward pass: no gradient overflow/underflow
  - Dtype consistency: float32/bfloat16

### Smoke Tests (in experiment script)
- Full training loop: 10 steps on tiny data (16 batch_size, 10 samples)
- Confirms: data loading, forward/backward, optimization, no memory leaks
- Runs before full experiments

---

## 7. Deliverables

1. **Code**:
   - `kromhc_transformer/` module (fully functional)
   - Tests passing
   - Experiment scripts ready to run

2. **Documentation**:
   - `experiments/README.md`: Running instructions, hardware requirements, expected timings
   - `findings.md`: Literature, gap analysis, feasibility assessment, experimental results
   - Inline code comments for novel components

3. **Outputs** (post-experiments):
   - JSON experiment artifacts (all phases)
   - Plots: loss curves, test ppl vs. model size, mixing matrix visualizations
   - Statistical analysis: confidence intervals, significance tests

---

## 8. Success Criteria

| Criterion | Status |
|-----------|--------|
| POC trains end-to-end without crashes | Required ✓ |
| Phase 2: KromHC ppl ≤ baseline + margin | Exploratory (5% margin acceptable) |
| Phase 3: Ablations show measurable difference | Expected |
| Phase 4: Results reproducible across seeds | Required for publication |
| All tests pass | Required ✓ |
| Code follows `rbf_ffn/` conventions | Required ✓ |
| Experiment protocols match Claude.md workflow | Required ✓ |

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| KromHC overhead → higher loss | High | Ablation phase (3) isolates impact |
| VRAM OOM on large models | Medium | Gradient accumulation; scale down if needed |
| Head mixing breaks gradient flow | Medium | Unit tests on gradient flow; small model POC |
| Hyperparameter sensitivity | Medium | Use same hyper-params as `rbf_ffn/` baseline |

---

## 10. Iteration Plan

1. **Week 1**: Code (days 1–2), POC experiments (day 3)
2. **Week 2**: Phase 2 baseline comparison (2–3 days)
3. **Week 3**: Phase 3 ablations (2–3 days)
4. **Week 4**: Phase 4 publication-ready runs + analysis (3–4 days)

Each phase gates the next. If POC fails, stop and debug. If Phase 2 shows major degradation, revisit architecture before ablations.

---

## 11. References

- **KromHC paper**: Jan 2026, Kronecker-factored head mixing
- **Baseline**: `rbf_ffn/models/` (CausalSelfAttention, transformer blocks)
- **WikiText-103**: Standard language modeling benchmark
- **Optimization**: Dual Muon + AdamW (matches `rbf_ffn/` precedent)
