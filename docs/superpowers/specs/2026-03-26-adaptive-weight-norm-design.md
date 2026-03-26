# Adaptive Weight Norm Based on Network Depth

**Date:** 2026-03-26
**Status:** Draft
**Scope:** `rbf_ffn` and `grokking` projects

---

## 1. Motivation

The current `linear_weight_norm` implementation applies a single global target norm to every Linear layer after each optimizer step. This treats all layers as equivalent, ignoring the structural asymmetry of transformers:

- **Early layers** build compositional intermediate representations and benefit from a larger norm budget to span a rich feature space.
- **Late layers** are closest to the output logits and are the primary substrate for memorization circuits (in the sense of Nanda et al. 2023). Constraining them more tightly biases the network toward generalizable, low-norm solutions.

Additionally, the uniform norm is static — it applies the same pressure regardless of whether the network is in the memorization phase or actively grokking. A phase-aware correction that tightens late layers only when memorization is deepening (and relaxes during grokking) is more surgical and avoids permanent over-tightening.

**Critical constraint:** Weight row norms below 1.0 flatten the local loss landscape (low curvature), killing gradient signal. The design guarantees `target_norm ≥ 1.0` at all layers at all times.

---

## 2. Design

### 2.1 Two-Component Formula

```
log_gap(t)      = log(val_loss(t) / train_loss(t))
ema_gap(t)      = α · log_gap(t) + (1 - α) · ema_gap(t-1)
Δgap(t)         = ema_gap(t) - ema_gap(t-1)

static(l)       = norm_late + (norm_early - norm_late) · (1 - l / (L - 1))
correction(l,t) = γ · (l / (L - 1)) · tanh(β · Δgap(t))

target_norm(l, t) = max(1.0, static(l) - correction(l, t))
```

Where `l` is the zero-indexed layer index and `L` is the total number of transformer blocks.

### 2.2 Component 1 — Static Depth Schedule

`static(l)` is a linear interpolation from `norm_early` (at layer 0) to `norm_late` (at layer L-1):

- Early layers receive the full norm budget → rich intermediate representations.
- Late layers operate near the floor → memorization circuits are harder to maintain.
- `norm_late ≥ 1.0` is enforced by configuration validation to guarantee safe curvature.

### 2.3 Component 2 — Phase-Aware Derivative Correction

`correction(l, t)` acts as a **high-pass filter** on the generalization gap:

- It fires only during **phase transitions**, not in steady state (Δgap ≈ 0 when stable).
- The `(l / (L-1))` weight makes the correction **strongest at late layers** (where memorization lives) and zero at the earliest layer.
- `tanh` bounds the correction to `[-γ, +γ]`, preventing runaway tightening from a sudden loss spike.

**Behavior by regime:**

| Regime | Δgap | Correction | Early layers | Late layers |
|---|---|---|---|---|
| Pre-training (both losses high) | ≈ 0 | ≈ 0 | `norm_early` | `norm_late` |
| Memorization deepening | > 0 | positive | unchanged | tightened |
| Grokking transition | < 0 | negative | unchanged | slightly relaxed |
| Post-grokking stable | ≈ 0 | ≈ 0 | `norm_early` | `norm_late` |

### 2.4 Ablation Structure

Setting `γ = 0` disables the phase correction and reduces the system to a pure static depth schedule — enabling clean ablation between the two components.

Setting `norm_early = norm_late` recovers the original uniform weight norm behavior.

---

## 3. Hyperparameters

| Parameter | Role | Default | Safe range |
|---|---|---|---|
| `norm_early` | Target norm for layer 0 | 2.5 | 1.5–4.0 |
| `norm_late` | Target norm for layer L-1 (floor) | 1.2 | 1.0–2.0 |
| `γ` | Max phase correction magnitude | 0.3 | 0.0–0.5 |
| `β` | Sensitivity to gap derivative | 5.0 | 2.0–15.0 |
| `α` | EMA smoothing for log-gap | 0.9 | 0.7–0.98 |

**Constraint:** `norm_late ≥ 1.0` is validated at config load time.

---

## 4. Architecture

### 4.1 Changes to `rbf_ffn`

**`rbf_ffn/config.py`**
Replace `linear_weight_norm_value: float` with five new fields:
- `adaptive_norm_early: float = 2.5` — target norm at layer 0
- `adaptive_norm_late: float = 1.2` — target norm at layer L-1
- `adaptive_norm_gamma: float = 0.3` — max phase correction magnitude
- `adaptive_norm_beta: float = 5.0` — tanh sensitivity to gap derivative
- `adaptive_norm_alpha: float = 0.9` — EMA smoothing factor for log-gap

Add `__post_init__` validation: `adaptive_norm_late ≥ 1.0` and `adaptive_norm_early > adaptive_norm_late`.

**`rbf_ffn/train.py`**
- Add EMA state variable `ema_log_gap` initialized to 0.0.
- Replace `apply_linear_weight_norm(model, cfg.linear_weight_norm_value)` with `apply_adaptive_weight_norm(model, cfg, ema_log_gap)`.
- Update `ema_log_gap` once per epoch after val loss is computed.
- Replace the `_wnorm` tag in `get_experiment_dir` to `_adpwnorm`.

**`rbf_ffn/train.py` — new function:**
```python
@torch.no_grad()
def apply_adaptive_weight_norm(
    model: CausalLM,
    cfg: RBFFFNConfig,
    delta_log_gap: float,
) -> None:
    """Apply per-layer adaptive weight norm.

    Each transformer block's linear layers are normalised to a target norm
    that decreases linearly from norm_early (layer 0) to norm_late (layer L-1),
    with a phase-aware derivative correction applied most strongly to late layers.
    """
    tied_id = id(model.token_embedding.weight)
    L = len(model.blocks)

    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        static = cfg.norm_late + (cfg.norm_early - cfg.norm_late) * (1.0 - frac)
        correction = cfg.adaptive_norm_gamma * frac * math.tanh(cfg.adaptive_norm_beta * delta_log_gap)
        target = max(1.0, static - correction)

        for module in block.modules():
            if isinstance(module, nn.Linear):
                if id(module.weight) == tied_id:
                    continue
                norms = module.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
                module.weight.data.mul_(target / norms)
```

### 4.2 Changes to `grokking`

The grokking project currently has no weight norm. This feature will be added as an opt-in config flag mirroring the `rbf_ffn` design, applied in `grokking/train.py` after each optimizer step. Val loss is computed every step in grokking (logged in metrics), so the EMA update can run every step.

---

## 5. Testing

- **Unit test:** `apply_adaptive_weight_norm` produces norms ≥ 1.0 for arbitrary `delta_log_gap` values, including large positive (memorization spike) and large negative (rapid grokking).
- **Ablation test:** `γ=0` produces identical norms to a hand-computed static depth schedule.
- **Recovery test:** `norm_early == norm_late` reproduces the original `apply_linear_weight_norm` behavior.
- **Config validation test:** `norm_late < 1.0` raises `ValueError` at load time.

---

## 6. Open Questions

- Should the EMA `α` be auto-set relative to steps-per-epoch (so behavior is consistent across batch sizes), or remain a raw hyperparameter?
- For `grokking` (only 2 layers), the depth schedule has only two points (l=0 and l=1). Worth testing, but the dynamic range of the schedule is limited.
