# mamba_lm

Pure-PyTorch Mamba language model trained on WikiText-103. No `mamba-ssm` package required — the selective scan is a plain Python loop compiled by `torch.compile`.

**Reference:** Gu & Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (2023).

---

## Structure

```
mamba_lm/
├── config.py          # MambaConfig dataclass + YAML loader
├── data.py            # WikiText-103 download, tokenisation, DataLoader
├── train.py           # Training entry point
├── configs/
│   └── baseline.yaml  # Default config (d_model=256, 6 layers)
├── models/
│   └── mamba.py       # MambaBlock, CausalMambaLM, optimizer groups
└── experiments/       # Created at runtime; one subdir per run
```

## Setup

```bash
pip install torch tiktoken datasets pyyaml tqdm
```

The first run downloads and tokenises WikiText-103 (~5 min) and caches the result in `mamba_lm/data_cache/`.

## Usage

```bash
# Train from scratch
python -m mamba_lm.train --config mamba_lm/configs/baseline.yaml

# Override epochs
python -m mamba_lm.train --config mamba_lm/configs/baseline.yaml --n_epochs 5

# Resume from a checkpoint
python -m mamba_lm.train --resume mamba_lm/experiments/<name> --resume_from latest
# --resume_from: best | final | latest
```

Each run creates a timestamped directory under `mamba_lm/experiments/` containing:

| File | Contents |
|---|---|
| `config.yaml` | Copy of the config used |
| `metrics.jsonl` | Per-epoch train/val loss and perplexity |
| `checkpoint_best.pt` | Checkpoint with lowest val loss |
| `checkpoint_latest.pt` | Checkpoint after the most recent epoch |
| `checkpoint_final.pt` | Checkpoint after the last epoch |

## Configuration

All fields from `MambaConfig` can be set in a YAML file.

| Field | Default | Description |
|---|---|---|
| `d_model` | 256 | Model dimension |
| `n_layers` | 6 | Number of Mamba blocks |
| `d_state` | 16 | SSM state dimension (N in the paper) |
| `d_conv` | 4 | Depthwise conv kernel size |
| `expand` | 2 | Inner expansion factor (`d_inner = d_model * expand`) |
| `dt_rank` | auto | Δ projection rank; defaults to `ceil(d_model / 16)` |
| `vocab_size` | 50257 | GPT-2 / r50k_base vocabulary |
| `seq_len` | 512 | Sequence length for training chunks |
| `tie_embeddings` | true | Tie input embedding and LM head weights |
| `n_epochs` | 10 | Training epochs |
| `batch_size` | 32 | Per-step batch size |
| `lr` | 3e-4 | Peak learning rate |
| `weight_decay` | 0.1 | AdamW weight decay (applied to 2-D weights only) |
| `warmup_ratio` | 0.02 | Fraction of optimizer steps used for linear warmup |
| `grad_clip` | 1.0 | Gradient norm clip value |
| `grad_accum_steps` | 1 | Gradient accumulation steps |

## Model Architecture

```
token_embedding  (vocab_size → d_model)
     │
  ×N MambaBlock
     │  RMSNorm
     │  in_proj  → split into x, z  (d_model → 2 × d_inner)
     │  causal depthwise Conv1d + SiLU
     │  Selective State Space (S6 scan)
     │  gating: y * SiLU(z)
     │  out_proj  (d_inner → d_model)
     │  + residual
     │
  RMSNorm
     │
  lm_head  (d_model → vocab_size, tied with embedding)
```

Weight decay is disabled for `A_log`, `D` (SSM parameters), biases, norm scales, and the embedding.

## Baseline Config

`configs/baseline.yaml` — a small model for quick iteration:

- **d_model** 256, **n_layers** 6, **d_state** 16 → ~10M parameters
- **seq_len** 512, **batch_size** 16, **lr** 3e-4
- 3 epochs, cosine LR with 2% warmup
