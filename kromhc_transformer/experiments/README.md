# KromHC Transformer Experiments

## Overview

Four-phase experiments comparing KromHC head mixing against a standard transformer baseline on WikiText-103 (language modeling).

## Prerequisites

- GPU: NVIDIA RTX 5060 Ti (16 GB VRAM) recommended
- Python 3.11+
- `pip install -r scripts/requirements.txt`

## Running Experiments

> Run phases in order. Do not skip a failing phase.

### Phase 1 — POC (~30 min, 1 GPU-hour)

**Hypothesis**: KromHC trains end-to-end without errors.

```bash
cd /path/to/machine-notes
python -m kromhc_transformer.experiments.scripts.exp_01_poc --seed 42
```

**Pass criterion**: Loss decreases over 1 epoch, no OOM, no NaN.

---

### Phase 2 — Baseline vs. KromHC (~8 hours, 3 seeds)

**Hypothesis**: KromHC test perplexity within 1% of baseline.

```bash
python -m kromhc_transformer.experiments.scripts.exp_02_baseline_vs_kromhc --seed 42
python -m kromhc_transformer.experiments.scripts.exp_02_baseline_vs_kromhc --seed 43
python -m kromhc_transformer.experiments.scripts.exp_02_baseline_vs_kromhc --seed 44
```

**Pass criterion**: KromHC ppl ≤ baseline ppl × 1.01 (mean ± std across seeds).

---

### Phase 3 — Ablations (~24 hours, 3 seeds × 2 variants)

```bash
python -m kromhc_transformer.experiments.scripts.exp_03_ablations --seed 42
```

---

### Phase 4 — Publication-Ready (~72 hours)

```bash
python -m kromhc_transformer.experiments.scripts.exp_04_final --seed 42
```

---

## Outputs

Results saved to `experiments/results/<timestamp>/`:
- `<model>_<seed>.json` — metrics, config, hardware
- `model_<seed>.pt` — checkpoint

## Troubleshooting

- **OOM**: Reduce `batch_size` in config
- **Slow first run**: WikiText-103 downloads once (~5 min), then uses cache
