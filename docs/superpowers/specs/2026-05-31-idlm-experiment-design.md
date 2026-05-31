# I-DLM Experiment Design

**Date:** 2026-05-31
**Status:** Approved
**Reference:** arXiv:2604.11035 (Yu, Jian, Wang, Zhou et al. — Together AI / UIUC / UT Austin / Princeton / Stanford, Apr 2026)
**Wiki:** `wiki/entities/i-dlm.md`

## Goal

Implement a small-scale reproduction of the I-DLM (Introspective Diffusion Language Model) pipeline on WikiText-103:

1. Fine-tune a pretrained `rbf_ffn` AR checkpoint using the I-DLM introspective-consistency training objective (LoRA adapters only, frozen AR base)
2. Evaluate the trained model using Introspective Strided Decoding (ISD) — measuring acceptance rate α, perplexity, and compute efficiency (TPF/OH)

## Project Structure

```
idlm/
├── config.py          # IDLMConfig dataclass
├── train.py           # training entry point
├── generate.py        # ISD evaluation entry point
├── data.py            # WikiText-103 loader (mirrors rbf_ffn/data.py)
├── models/
│   ├── lora.py        # LoRALinear wrapper + apply_lora() helper
│   └── idlm_model.py  # IDLMCausalLM: frozen AR base + LoRA at mask positions
├── configs/
│   └── baseline.yaml  # first experiment config
├── tests/
│   └── test_model.py  # shape, gradient, LoRA-isolation tests
└── README.md
```

New top-level project — does not extend `rbf_ffn/`. The AR checkpoint is an input, not a dependency.

## Architecture

### IDLMCausalLM

Wraps the frozen `rbf_ffn` `CausalLM` checkpoint. LoRA adapters (`r=8`, `alpha=16`) are applied to `q_proj` and `v_proj` in every attention layer. The AR base weights are frozen (`requires_grad=False`); only LoRA parameters are trained.

**Position-aware LoRA activation:**
- A boolean `use_lora: Tensor[B, L]` is passed into the forward call
- At each position, the LoRA delta is multiplied by `use_lora[:, i]` before being added to the base output
- Mask positions (`use_lora=True`): decode pathway q — adapted weights active
- Introspection positions (`use_lora=False`): base AR weights only — causal anchor distribution p

The AR logit shift (predict token i+1 from hidden state at position i) is inherited from the base checkpoint unchanged.

### LoRA Implementation

```python
# lora.py
class LoRALinear(nn.Module):
    # base: frozen Linear
    # lora_A, lora_B: trainable, initialized so delta=0
    # forward(x, use_lora: float scalar 0 or 1) -> base(x) + use_lora * (lora_B @ lora_A)(x) * scale
```

`apply_lora(model, target_modules, rank, alpha)` walks the model and replaces named Linear layers with `LoRALinear` in-place.

## Training Objective

All training is on WikiText-103 sequences of length L=512.

### Input Construction (all-masked objective)

For each batch:
1. Sample `x_0` (length L) from WikiText-103
2. Replace all tokens with `[MASK]` → `x_t`
3. Concatenate: input = `[x_t | x_0]`, total length 2L
4. Apply strict causal masking throughout (no bidirectional attention)

The `x_0` half attends to the `x_t` half through the causal mask — this is the introspection pathway.

### Loss Terms

```
L_mask  = CrossEntropy at x_t positions  (decode pathway, LoRA active)
L_clean = CrossEntropy at x_0 positions  (introspection pathway, LoRA zeroed)
λ       = stop_gradient(L_mask / L_clean)
L       = L_mask + λ * L_clean
```

The auto-balanced coefficient λ dynamically rescales L_clean to match L_mask magnitude each step, preventing the introspection pathway from dominating training.

### LoRA activation during training

- `x_t` positions: `use_lora=1` → full adapted weights (q distribution)
- `x_0` positions: `use_lora=0` → frozen base weights (p distribution)

Both pathways are computed in a single forward pass over the 2L sequence.

## ISD Evaluation (`generate.py`)

Introspective Strided Decoding over WikiText-103 test continuations.

**Algorithm (stride S, prompt length P, generation length G):**

1. Take first P tokens of a test example as prefix
2. Fill positions P..P+S-1 with `[MASK]`
3. Forward pass (full 2L-style, but generation portion only):
   - Mask positions: sample proposed token from q (LoRA active)
   - Previously accepted positions: compute p (LoRA zeroed), accept x_k with prob `min(1, p(x_k)/q(x_k))`; on rejection, resample from corrected distribution
4. Slide window by S; repeat until G tokens generated

**Metrics per run (logged to JSON):**

| Metric | Description |
|--------|-------------|
| α | `(1/G) Σ_k min(1, p_k(x_k)/q_k(x_k))` — introspective acceptance rate |
| PPL | Perplexity of generated continuation under base AR model vs. ground truth |
| TPF/OH | `S / (1 + rejection_overhead)` — compute efficiency; > 1 means ISD beats sequential AR |

## Config (`IDLMConfig`)

```python
# checkpoint
ar_checkpoint: str          # path to rbf_ffn checkpoint (baseline_weight_norm recommended)
lora_rank: int = 8
lora_alpha: float = 16.0
lora_target_modules: list = ["q_proj", "v_proj"]

# training
seq_len: int = 512
batch_size: int = 8
max_steps: int = 10_000
lr: float = 3e-4            # AdamW on LoRA params only
warmup_steps: int = 200
grad_clip: float = 1.0

# evaluation / ISD
eval_every: int = 500
stride: int = 4
num_eval_examples: int = 200
prompt_len: int = 64
gen_len: int = 128
```

**`baseline.yaml`** points at the `rbf_ffn` `baseline_weight_norm` checkpoint (58.16 val PPL — best result in the repo) as the AR base.

## Training Loop

- Optimizer: AdamW over LoRA parameters only
- Checkpoints: save LoRA weights only (~0.4M params at r=8, α=16)
- Eval: ISD on 200 WikiText-103 test examples every 500 steps — logs α, PPL, TPF/OH to `idlm/experiments/<run_id>/results.json`
- Hardware target: RTX 5060 Ti 16 GB (same as all other experiments)

## Tests

`tests/test_model.py` covers:

1. Output shape: `(B, 2L, vocab_size)` for 2L input
2. LoRA isolation: `use_lora=0` output matches frozen base exactly
3. Gradient isolation: only LoRA params have non-zero gradients after backward
4. Loss terms: L_mask and L_clean are both finite; λ is positive
5. Auto-balance: λ = stop_gradient(L_mask / L_clean) — verify no grad flows through λ
6. ISD shape: generated sequence has correct length G
7. Acceptance rate bounds: 0 ≤ α ≤ 1

## Success Criteria

- Training converges (L_mask and L_clean both decrease over 10k steps)
- α > 0.7 at convergence (paper reports 0.699 for SDAR-8B; our 6-layer model should be in range)
- TPF/OH approaches or exceeds 1.0 as α improves
- All tests pass
