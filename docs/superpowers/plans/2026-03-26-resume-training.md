# Resume Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow interrupted training runs to be resumed from a checkpoint, continuing in the same experiment directory.

**Architecture:** All changes live in `rbf_ffn/train.py`. The `train()` function gains an optional `resume_checkpoint` param; when set it skips directory creation, loads all state dicts, and starts the epoch loop from `ckpt["epoch"] + 1`. The checkpoint dict gains `best_val_loss` and `ema_log_gap` for complete state restoration. CLI gains `--resume` and `--resume_from` args.

**Tech Stack:** PyTorch, Python 3.12, pytest

---

### Task 1: Add `best_val_loss` and `ema_log_gap` to checkpoint

**Files:**
- Modify: `rbf_ffn/train.py:216-226` (`save_checkpoint` closure)
- Test: `rbf_ffn/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `rbf_ffn/tests/test_train.py`:

```python
def test_checkpoint_contains_resume_keys(tmp_path):
    """Checkpoints must include best_val_loss and ema_log_gap for resume."""
    cfg = _tiny_cfg()
    exp_dir = _run_train(cfg, tmp_path)
    ckpt = torch.load(exp_dir / "checkpoint_final.pt", map_location="cpu",
                      weights_only=False)
    assert "best_val_loss" in ckpt
    assert "ema_log_gap" in ckpt
    assert isinstance(ckpt["best_val_loss"], float)
    assert isinstance(ckpt["ema_log_gap"], float)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /home/harikrishnan-c/projects/torch/bin/activate
python -m pytest rbf_ffn/tests/test_train.py::test_checkpoint_contains_resume_keys -v
```

Expected: FAIL with `AssertionError` (key missing).

- [ ] **Step 3: Update `save_checkpoint` in `train.py`**

In `rbf_ffn/train.py`, the `save_checkpoint` closure (around line 216) currently saves:

```python
def save_checkpoint(name: str, epoch: int, val_loss: float, val_ppl: float):
    torch.save({
        "model":           model.state_dict(),
        "optimizer_muon":  muon.state_dict(),
        "optimizer_adamw": adamw.state_dict(),
        "scheduler_muon":  sched_muon.state_dict(),
        "scheduler_adamw": sched_adamw.state_dict(),
        "epoch":    epoch,
        "val_loss": val_loss,
        "val_ppl":  val_ppl,
    }, exp_dir / name)
```

Replace with:

```python
def save_checkpoint(name: str, epoch: int, val_loss: float, val_ppl: float):
    torch.save({
        "model":           model.state_dict(),
        "optimizer_muon":  muon.state_dict(),
        "optimizer_adamw": adamw.state_dict(),
        "scheduler_muon":  sched_muon.state_dict(),
        "scheduler_adamw": sched_adamw.state_dict(),
        "epoch":         epoch,
        "val_loss":      val_loss,
        "val_ppl":       val_ppl,
        "best_val_loss": best_val_loss,
        "ema_log_gap":   ema_log_gap,
    }, exp_dir / name)
```

`best_val_loss` and `ema_log_gap` are already in the enclosing scope; the closure captures them by reference so their current values are saved at call time.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest rbf_ffn/tests/test_train.py::test_checkpoint_contains_resume_keys -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite to check nothing is broken**

```bash
python -m pytest rbf_ffn/tests/test_train.py -v
```

Expected: 13 passed.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/train.py rbf_ffn/tests/test_train.py
git commit -m "feat(train): add best_val_loss and ema_log_gap to checkpoint"
```

---

### Task 2: Add `resume_checkpoint` param and resume logic to `train()`

**Files:**
- Modify: `rbf_ffn/train.py`
- Test: `rbf_ffn/tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

Add to `rbf_ffn/tests/test_train.py`:

```python
def _run_resume(exp_dir: Path, tmp_path, cfg: RBFFFNConfig,
                resume_from: str = "best", n_epochs: int | None = None):
    """Run train() in resume mode, reusing exp_dir."""
    resume_ckpt = exp_dir / f"checkpoint_{resume_from}.pt"
    loaders = _fake_loaders(cfg)
    with patch("rbf_ffn.train.get_dataloaders", return_value=loaders), \
         patch("rbf_ffn.train.Muon", _MuonStub):
        return train(
            cfg,
            config_path=exp_dir / "config.yaml",
            n_epochs=n_epochs,
            resume_checkpoint=resume_ckpt,
        )


def test_resume_continues_from_correct_epoch(tmp_path):
    """After a 1-epoch run, resuming with n_epochs=2 trains exactly one more epoch."""
    cfg = _tiny_cfg(n_epochs=1)
    exp_dir = _run_train(cfg, tmp_path)

    # Resume for 1 more epoch (total=2)
    cfg2 = _tiny_cfg(n_epochs=2)
    result_dir = _run_resume(exp_dir, tmp_path, cfg2)

    assert result_dir == exp_dir
    rows = [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 2
    assert rows[0]["epoch"] == 0
    assert rows[1]["epoch"] == 1


def test_resume_appends_to_existing_metrics(tmp_path):
    """metrics.jsonl is appended to, not overwritten, on resume."""
    cfg = _tiny_cfg(n_epochs=1)
    exp_dir = _run_train(cfg, tmp_path)
    original_lines = (exp_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(original_lines) == 1

    cfg2 = _tiny_cfg(n_epochs=2)
    _run_resume(exp_dir, tmp_path, cfg2)

    all_lines = (exp_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert all_lines[0] == original_lines[0]   # first line unchanged
    assert len(all_lines) == 2


def test_resume_already_complete_exits_early(tmp_path):
    """Resuming when start_epoch >= n_epochs returns exp_dir without training."""
    cfg = _tiny_cfg(n_epochs=2)
    exp_dir = _run_train(cfg, tmp_path)
    line_count_before = len(
        (exp_dir / "metrics.jsonl").read_text().strip().splitlines()
    )

    # Try to resume with same n_epochs — nothing should happen
    cfg2 = _tiny_cfg(n_epochs=2)
    result_dir = _run_resume(exp_dir, tmp_path, cfg2)

    assert result_dir == exp_dir
    line_count_after = len(
        (exp_dir / "metrics.jsonl").read_text().strip().splitlines()
    )
    assert line_count_after == line_count_before
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest rbf_ffn/tests/test_train.py::test_resume_continues_from_correct_epoch rbf_ffn/tests/test_train.py::test_resume_appends_to_existing_metrics rbf_ffn/tests/test_train.py::test_resume_already_complete_exits_early -v
```

Expected: 3 FAIL (TypeError — `train()` does not accept `resume_checkpoint`).

- [ ] **Step 3: Update `train()` signature**

Change the function signature from:

```python
def train(cfg: RBFFFNConfig, config_path: Path, n_epochs: int | None = None) -> Path:
```

to:

```python
def train(
    cfg: RBFFFNConfig,
    config_path: Path,
    n_epochs: int | None = None,
    resume_checkpoint: Path | None = None,
) -> Path:
```

- [ ] **Step 4: Branch experiment directory creation**

Replace the current block (around line 179):

```python
exp_dir = get_experiment_dir(cfg)
shutil.copy(config_path, exp_dir / "config.yaml")
metrics_path = exp_dir / "metrics.jsonl"
print(f"Experiment dir: {exp_dir}")
```

with:

```python
if resume_checkpoint is not None:
    exp_dir = resume_checkpoint.parent
else:
    exp_dir = get_experiment_dir(cfg)
    shutil.copy(config_path, exp_dir / "config.yaml")
metrics_path = exp_dir / "metrics.jsonl"
print(f"Experiment dir: {exp_dir}")
```

- [ ] **Step 5: Initialise `start_epoch` before the pbar and load checkpoint after schedulers**

After the schedulers are built (after the `sched_muon` / `sched_adamw` lines, around line 207), insert:

```python
# ── Resume ────────────────────────────────────────────────────────────────
start_epoch = 0
if resume_checkpoint is not None:
    ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    muon.load_state_dict(ckpt["optimizer_muon"])
    adamw.load_state_dict(ckpt["optimizer_adamw"])
    sched_muon.load_state_dict(ckpt["scheduler_muon"])
    sched_adamw.load_state_dict(ckpt["scheduler_adamw"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    ema_log_gap   = ckpt.get("ema_log_gap", 0.0)
    print(f"Resuming from epoch {start_epoch} / {cfg.n_epochs}")
    if start_epoch >= cfg.n_epochs:
        print("Already at target epoch count. Nothing to do.")
        return exp_dir
```

- [ ] **Step 6: Update pbar total and epoch loop**

Change the pbar line from:

```python
pbar = tqdm(total=total_steps, desc="training", unit="step", dynamic_ncols=True)
```

to:

```python
remaining_steps = (cfg.n_epochs - start_epoch) * steps_per_epoch
pbar = tqdm(total=remaining_steps, desc="training", unit="step", dynamic_ncols=True)
```

Change the epoch loop from:

```python
for epoch in range(cfg.n_epochs):
```

to:

```python
for epoch in range(start_epoch, cfg.n_epochs):
```

- [ ] **Step 7: Run new tests to verify they pass**

```bash
python -m pytest rbf_ffn/tests/test_train.py::test_resume_continues_from_correct_epoch rbf_ffn/tests/test_train.py::test_resume_appends_to_existing_metrics rbf_ffn/tests/test_train.py::test_resume_already_complete_exits_early -v
```

Expected: 3 PASS

- [ ] **Step 8: Run full test suite**

```bash
python -m pytest rbf_ffn/tests/test_train.py -v
```

Expected: 16 passed.

- [ ] **Step 9: Commit**

```bash
git add rbf_ffn/train.py rbf_ffn/tests/test_train.py
git commit -m "feat(train): add resume_checkpoint param and resume logic to train()"
```

---

### Task 3: Add `--resume` and `--resume_from` CLI args

**Files:**
- Modify: `rbf_ffn/train.py` (`parse_args()` and `__main__` block)
- Test: `rbf_ffn/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `rbf_ffn/tests/test_train.py`:

```python
def test_parse_args_resume_defaults(monkeypatch):
    """--resume_from defaults to 'best' when not specified."""
    import sys
    from rbf_ffn.train import parse_args
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "cfg.yaml",
                                      "--resume", "experiments/foo"])
    args = parse_args()
    assert args.resume == "experiments/foo"
    assert args.resume_from == "best"


def test_parse_args_resume_from_final(monkeypatch):
    """--resume_from final is accepted."""
    import sys
    from rbf_ffn.train import parse_args
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "cfg.yaml",
                                      "--resume", "experiments/foo",
                                      "--resume_from", "final"])
    args = parse_args()
    assert args.resume_from == "final"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest rbf_ffn/tests/test_train.py::test_parse_args_resume_defaults rbf_ffn/tests/test_train.py::test_parse_args_resume_from_final -v
```

Expected: 2 FAIL (unrecognised argument).

- [ ] **Step 3: Update `parse_args()`**

Replace:

```python
def parse_args():
    p = argparse.ArgumentParser(description="Train RBF-FFN on WikiText-103")
    p.add_argument("--config",   required=True, help="Path to YAML config")
    p.add_argument("--n_epochs", type=int, default=None, help="Override n_epochs")
    return p.parse_args()
```

with:

```python
def parse_args():
    p = argparse.ArgumentParser(description="Train RBF-FFN on WikiText-103")
    p.add_argument("--config",      required=True, help="Path to YAML config")
    p.add_argument("--n_epochs",    type=int, default=None, help="Override n_epochs")
    p.add_argument("--resume",      type=str, default=None,
                   help="Experiment directory to resume training from")
    p.add_argument("--resume_from", choices=["best", "final"], default="best",
                   help="Which checkpoint to load when resuming (default: best)")
    return p.parse_args()
```

- [ ] **Step 4: Update `__main__` block**

Replace:

```python
if __name__ == "__main__":
    args   = parse_args()
    path   = Path(args.config)
    cfg    = load_config(path)
    train(cfg, config_path=path, n_epochs=args.n_epochs)
```

with:

```python
if __name__ == "__main__":
    args = parse_args()

    resume_checkpoint = None
    if args.resume is not None:
        resume_dir        = Path(args.resume)
        path              = resume_dir / "config.yaml"
        cfg               = load_config(path)
        resume_checkpoint = resume_dir / f"checkpoint_{args.resume_from}.pt"
    else:
        path = Path(args.config)
        cfg  = load_config(path)

    train(cfg, config_path=path, n_epochs=args.n_epochs,
          resume_checkpoint=resume_checkpoint)
```

- [ ] **Step 5: Run new tests to verify they pass**

```bash
python -m pytest rbf_ffn/tests/test_train.py::test_parse_args_resume_defaults rbf_ffn/tests/test_train.py::test_parse_args_resume_from_final -v
```

Expected: 2 PASS

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest rbf_ffn/tests/test_train.py -v
```

Expected: 18 passed.

- [ ] **Step 7: Commit**

```bash
git add rbf_ffn/train.py rbf_ffn/tests/test_train.py
git commit -m "feat(train): add --resume and --resume_from CLI args"
```
