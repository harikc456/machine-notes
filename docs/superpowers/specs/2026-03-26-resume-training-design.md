# Resume Training Design

**Date:** 2026-03-26
**Status:** Approved

## Overview

Add the ability to resume an interrupted training run, continuing in the same experiment directory, appending to its `metrics.jsonl`, and preserving the best checkpoint.

## CLI

Two new arguments added to `parse_args()`:

```
--resume <exp_dir>         Path to the experiment directory to resume.
--resume_from [best|final] Which checkpoint to load (default: best).
```

When `--resume` is given, `config_path` is set to `<exp_dir>/config.yaml` (the saved copy) so the config is always consistent with the original run. `n_epochs` is interpreted as **total epochs**, not additional epochs — the run stops at `cfg.n_epochs` total.

## `train()` signature

```python
def train(
    cfg: RBFFFNConfig,
    config_path: Path,
    n_epochs: int | None = None,
    resume_checkpoint: Path | None = None,
) -> Path:
```

The resolved checkpoint path (`<exp_dir>/checkpoint_best.pt` or `checkpoint_final.pt`) is passed in as `resume_checkpoint`.

## Checkpoint format

Two new keys added to the checkpoint dict:

```python
{
    # existing keys unchanged
    "best_val_loss": best_val_loss,
    "ema_log_gap":   ema_log_gap,
}
```

Both are read back with `.get()` fallbacks (`float("inf")` and `0.0` respectively) for backward compatibility with checkpoints that predate this change.

`global_step` is not saved — it is local to each epoch loop and the scheduler state already encodes LR schedule position.

## Resume logic inside `train()`

When `resume_checkpoint is not None`:

1. **Experiment directory** — use `resume_checkpoint.parent` as `exp_dir`; skip `get_experiment_dir()` and `shutil.copy()` (config already present).
2. **Checkpoint load** — after model, optimizers, and schedulers are constructed, load the checkpoint and restore all state dicts. Extract:
   - `start_epoch = ckpt["epoch"] + 1`
   - `best_val_loss = ckpt.get("best_val_loss", float("inf"))`
   - `ema_log_gap = ckpt.get("ema_log_gap", 0.0)`
3. **Early exit** — if `start_epoch >= cfg.n_epochs`, print a message and return `exp_dir` immediately.
4. **Epoch loop** — `for epoch in range(start_epoch, cfg.n_epochs)` instead of `range(cfg.n_epochs)`.
5. **Progress bar** — total set to `(cfg.n_epochs - start_epoch) * steps_per_epoch` so it reflects remaining steps only.
6. **`metrics.jsonl`** — already opened in append mode (`"a"`); no change needed.

## What is not in scope

- Mid-epoch resume (resume always starts from the beginning of the next epoch).
- Automatic resume on crash (caller must pass `--resume` explicitly).
- Keeping multiple best checkpoints.
