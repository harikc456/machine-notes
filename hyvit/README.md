# HyViT — Hyperbolic Vision Transformer

A Vision Transformer where attention operates in the **Lorentz model of hyperbolic space** instead of Euclidean space. The core hypothesis: images have hierarchical structure (pixels → edges → textures → parts → objects), and hyperbolic geometry represents hierarchies with exponentially lower distortion than flat space.

## Key Idea

Standard dot-product attention: `score(q, k) = qᵀk / √d`

Lorentz attention: `score(q, k) = -⟨q, k⟩_L / √d`  where `⟨q,k⟩_L = -q₀k₀ + Σᵢ qᵢkᵢ`

This is a **single sign flip** on the time-like component. The rest of the architecture — multi-head, softmax, aggregation — is structurally identical to a standard ViT, except aggregation uses a Lorentz centroid (project weighted sum back to hyperboloid) instead of a plain weighted sum.

## Architecture

```
Image (B, C, H, W)
  → Euclidean patch projection  [Linear]
  → project_to_hyperboloid      [lift to H^{d_model}]
  → N × LorentzTransformerBlock [attention + FFN in hyperbolic space]
  → LorentzLayerNorm
  → log_map_origin (CLS token)  [map back to R^{d_model}]
  → Linear classifier
```

## Project Structure

```
hyvit/
├── README.md
├── requirements.txt
├── config.py                   # HyViTTinyConfig, HyViTSmallConfig
├── train.py                    # training script
├── geometry/
│   └── lorentz.py              # manifold primitives (exp, log, project, distance)
├── models/
│   ├── lorentz_layers.py       # LorentzLinear, LorentzLayerNorm, LorentzCentroid
│   ├── lorentz_attention.py    # LorentzMultiheadAttention
│   ├── lorentz_block.py        # LorentzFFN, LorentzTransformerBlock
│   ├── patch_embed.py          # HyperbolicPatchEmbed
│   ├── hyvit.py                # HyViT (full model)
│   └── euclidean_vit.py        # EuclideanViT (baseline)
├── optim/
│   └── riemannian.py           # build_optimizer (AdamW with decay groups)
├── data/
│   └── cifar.py                # CIFAR-10 loaders
└── tests/                      # pytest test suite
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 5060 Ti 16GB)
- PyTorch 2.9+ with CUDA 12.8

### Install

```bash
# If using uv (recommended):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install geoopt>=0.5.1 numpy>=2.2.3 einops>=0.8.1

# Or with pip:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install geoopt>=0.5.1 numpy>=2.2.3 einops>=0.8.1
```

> CIFAR-10 (~170MB) will be auto-downloaded to `data/cifar10/` on first run.

### Verify installation

```bash
cd hyvit/
python3 -m pytest tests/ -v
# Expected: all tests pass
```

---

## Running Experiments

All commands should be run from the `hyvit/` directory.

### Experiment 1: Train HyViT-Tiny (baseline hyperbolic run)

```bash
python3 train.py --config tiny --model hyvit
```

- ~5.4M parameters
- batch_size=128, lr=3e-4, 100 epochs, cosine LR with 10-epoch warmup
- Checkpoints saved to `checkpoints/hyvit_tiny/`
- Expected runtime: ~2–3 min/epoch on RTX 5060 Ti

### Experiment 2: Train Euclidean ViT baseline (same architecture, flat space)

```bash
python3 train.py --config tiny --model euclidean
```

- Same d_model=192, n_heads=3, n_blocks=12 as HyViT-Tiny
- Checkpoints saved to `checkpoints/euclidean_tiny/`

### Experiment 3: Train HyViT-Small

```bash
python3 train.py --config small --model hyvit
```

- ~21.4M parameters
- batch_size=64, lr=1e-4, 200 epochs
- Requires ~8GB VRAM

### Resume from checkpoint

```bash
python3 train.py --config tiny --model hyvit --resume checkpoints/hyvit_tiny/best.pt
```

---

## What to Watch During Training

The training script prints one line per epoch:

```
Epoch 001/100 | train_loss=2.3021 train_acc=10.2% | val_loss=2.2985 val_acc=10.8% | lr=3.00e-05 | 142s
```

**Signs the model is learning correctly:**
- `train_acc` and `val_acc` should climb above random (10%) within 5–10 epochs
- `val_loss` should decrease steadily for the first 30–50 epochs
- No `NaN` in loss (if you see NaN, reduce `lr` by 10×)

**Expected final accuracy (100 epochs, CIFAR-10):**
| Model | Expected val_acc |
|-------|-----------------|
| EuclideanViT-Tiny | ~78–82% |
| HyViT-Tiny | ~75–80% (hypothesis: comparable or better with fewer params) |

> Note: These are ballpark figures for ViT-Tiny on CIFAR-10 without heavy augmentation. The gap — or lack thereof — is the scientific result.

---

## Comparing Results

After training both models, compare:

```
EuclideanViT-Tiny best val_acc: checkpoints/euclidean_tiny/best.pt  → cfg["best_val_acc"]
HyViT-Tiny best val_acc:        checkpoints/hyvit_tiny/best.pt      → cfg["best_val_acc"]
```

To print the best accuracy from a checkpoint:

```bash
python3 -c "
import torch
ckpt = torch.load('checkpoints/hyvit_tiny/best.pt', map_location='cpu')
print(f'Best val acc: {ckpt[\"best_val_acc\"]*100:.2f}%  (epoch {ckpt[\"epoch\"]+1})')
"
```

**What to measure beyond top-1 accuracy:**
1. **Sample efficiency** — plot val_acc vs epoch. Does HyViT reach 70% faster?
2. **Parameter efficiency** — accuracy per million parameters
3. **Convergence stability** — does loss curve oscillate more/less?

---

## Config Reference

Edit `config.py` to tune hyperparameters.

| Parameter | Tiny default | Notes |
|-----------|-------------|-------|
| `d_model` | 192 | Lorentz dim is `d_model+1` |
| `n_heads` | 3 | Must divide `d_model` |
| `n_blocks` | 12 | Depth |
| `mlp_ratio` | 4 | FFN hidden = `d_model * mlp_ratio` |
| `lr` | 3e-4 | Reduce to 1e-4 if NaN |
| `batch_size` | 128 | Reduce to 64 if OOM |
| `dropout` | 0.1 | Increase to 0.2 for regularization |
| `label_smoothing` | 0.1 | Standard for image classification |

---

## Troubleshooting

**NaN loss on first step**
- Reduce `lr` by 10× (e.g. `lr=3e-5`)
- Increase `warmup_epochs` to 20

**CUDA out of memory**
- Reduce `batch_size` (64 → 32)
- Switch to `--config tiny` if on `--config small`

**Training is slow**
- Confirm GPU is being used: `nvidia-smi` should show GPU utilization >90%
- Reduce `num_workers` to 2 if CPU is the bottleneck

**Accuracy stuck at ~10% (random)**
- Check that LR schedule is working: first epoch LR should be `lr / warmup_epochs`
- Verify CIFAR-10 data is normalized correctly (check `data/cifar.py` mean/std)
