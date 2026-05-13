# machine-notes

A personal ML research lab notebook — independent experiments in transformer architectures, alternative sequence models, and training dynamics. Each subdirectory is a self-contained project with its own config system, training loop, and experiment outputs.

## Projects

| Project | Topic | Dataset | Status |
|---------|-------|---------|--------|
| [`rbf_ffn/`](#rbf_ffn) | Transformer FFN architecture exploration | WikiText-103 | Active |
| [`sigreg/`](#sigreg) | Sketched Isotropic Gaussian Regularization for hidden states | WikiText-103 | Active |
| [`mamba_lm/`](#mamba_lm) | Pure-PyTorch Mamba language model | WikiText-103 | Active |
| [`grokking/`](#grokking) | Grokking on modular arithmetic | Synthetic | Active |
| [`hyvit/`](#hyvit) | Hyperbolic Vision Transformer (Lorentz space) | CIFAR-10 | Active |
| [`flow_matching/`](#flow_matching) | Flow matching experiments | — | Active |
| [`archive/`](#archive) | Archived earlier experiments | — | Archived |

---

## Setup

```bash
# Install all packages (uses uv)
uv sync

# Or with pip
pip install -e .
```

Python 3.10+, CUDA-capable GPU recommended.

---

## rbf_ffn

Systematic study of alternative FFN designs for causal language models: RBF kernels, learnable rational activations (Padé and PFD), polar-coordinate representations, Kronecker-factored projections, and normalization strategies.

**Best result:** SwiGLU + QK-norm + weight-norm → **58.16 val PPL** (−23.1% vs vanilla SwiGLU)

**Key finding:** Weight normalization (−21.8 PPL) dominates all FFN architecture changes combined. Gating is load-bearing — non-gated variants consistently underperform.

```bash
# Best overall result
python -m rbf_ffn.train --config rbf_ffn/configs/baseline_weight_norm.yaml

# Best FFN activation variant
python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_qk_norm_weight_norm.yaml

# Resume from checkpoint
python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml --resume_from path/to/checkpoint_latest.pt
```

See [`rbf_ffn/README.md`](rbf_ffn/README.md) for full experiment details, results table, and config reference.

---

## sigreg

Transformers trained with an auxiliary **SIGReg** (Sketched Isotropic Gaussian Regularization) loss that pushes hidden-state distributions toward a standard Gaussian. Uniquely: no residual connections or normalization layers by default — signal routes through weights alone.

Two loss types: *strong* (all moments, via characteristic functions) and *weak* (2nd moment only, via covariance distance).

```bash
python -m sigreg.train --config sigreg/configs/baseline.yaml
python -m sigreg.train --config sigreg/configs/weak_loss.yaml
```

See [`sigreg/README.md`](sigreg/README.md) for architecture details and config reference.

---

## mamba_lm

Pure-PyTorch Mamba (selective state space) language model trained on WikiText-103. No `mamba-ssm` package required — the selective scan is a plain Python loop compiled by `torch.compile`.

Reference: Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023).

```bash
python -m mamba_lm.train --config mamba_lm/configs/baseline.yaml
```

See [`mamba_lm/README.md`](mamba_lm/README.md) for architecture, config reference, and checkpoint format.

---

## grokking

Reproduces the grokking phenomenon from [Power et al. 2022](https://arxiv.org/abs/2201.02177): a small transformer first memorizes a modular arithmetic task, then suddenly generalizes long after training loss plateaus.

Supports multiple operations (`add`, `sub`, `mul`, `div`, `x2_plus_xy_plus_y2`) and both AdamW and Muon optimizers.

```bash
python -m grokking.train --config grokking/configs/adamw_add.yaml
python -m grokking.train --config grokking/configs/muon_add.yaml
```

See [`grokking/README.md`](grokking/README.md) for expected behavior and config reference.

---

## hyvit

Vision Transformer where attention operates in the **Lorentz model of hyperbolic space**. Core hypothesis: images have hierarchical structure, and hyperbolic geometry represents hierarchies with exponentially lower distortion than flat space.

The change from standard ViT is a single sign flip: `score(q,k) = −⟨q,k⟩_L / √d` using the Lorentz inner product, with Lorentz centroids for attention aggregation.

```bash
# From hyvit/
python3 train.py --config tiny --model hyvit      # hyperbolic
python3 train.py --config tiny --model euclidean   # Euclidean baseline
```

See [`hyvit/README.md`](hyvit/README.md) for setup, architecture diagram, and training tips.

---

## flow_matching

Flow matching experiments. See [`flow_matching/`](flow_matching/) for configs and training scripts.

---

## archive

Earlier experiments (Kronecker-factored transformer variants) preserved for reference.

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5060 Ti — 16 GB VRAM |
| RAM | 16 GB |
| CPU | AMD Ryzen 7 — 3 GHz |

---

## Tests

```bash
pytest rbf_ffn/tests/ grokking/tests/ sigreg/tests/ -v
```
