# No Auxiliary Loss Option — Design Spec

**Date:** 2026-04-08
**Status:** Approved

## Motivation

Setting `sigreg_weight: 0.0` already skips the aux loss computation in `train.py`, but the model still runs `collect_hidden=True` on every forward pass, allocating and returning hidden state tensors that are never used. This is wasted work. The fix makes `sigreg_weight: 0.0` fully correct with zero overhead.

## Change

**File:** `sigreg/train.py`

Change the forward call inside the training loop from:

```python
logits, hidden_states = model(inputs, collect_hidden=True)
```

to:

```python
logits, hidden_states = model(inputs, collect_hidden=(cfg.sigreg_weight > 0.0))
```

## Behaviour

| `sigreg_weight` | `collect_hidden` | aux loss computed |
|---|---|---|
| `> 0.0` | `True` | yes |
| `0.0` | `False` | no (guard already skips it) |

## What Is Not Changed

- `SIGRegConfig` — no new fields
- YAML configs — no changes required
- `model.py`, `block.py`, `losses.py` — untouched

## Usage

To run CE-only (no SIGReg), set in any config:

```yaml
sigreg_weight: 0.0
```

## Testing

- With `sigreg_weight > 0.0`: `hidden_states` is non-empty and aux loss is computed (existing behaviour, unchanged)
- With `sigreg_weight = 0.0`: `hidden_states` is `[]` and `total_loss == ce_loss`
