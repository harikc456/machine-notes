# Latent-CoT Diffusion-Encoder Reconstruction Experiment

## Goal

Test whether a reasoning latent `z` produced from the **question alone**, via
an iterative diffusion-style refinement (not one-shot pooling), is
expressive enough for an autoregressive decoder to reconstruct the GSM8K
reasoning trace text. This is a new, standalone probe — distinct from
`latent_cot`'s existing 4-way kill-test (`floor`/`z`/`ceiling`/`z_shuffled`),
which encodes the *trace* into `z` and decodes only the final answer number.
Here the encoder never sees the trace at all, and the decode target is the
full trace text.

## Non-goals

- Not modifying `latent_cot/model.py`'s `ReasoningEncoder` or the existing
  4-way kill-test conditions (`floor`/`z`/`ceiling`/`z_shuffled`) — those
  stay exactly as they are.
- Not wiring this into `run_killtest.py`'s orchestrated table — its metric
  (token-level reconstruction accuracy) isn't comparable to the kill-test's
  exact-match answer accuracy.
- Not implementing a real denoising-score-matching / DDPM loss. The
  "diffusion" here is purely architectural (iterative refinement from
  noise, conditioned on the question); the only supervision signal is the
  trace-reconstruction loss, backpropagated through the fully unrolled
  refinement chain.
- Not adding a shuffled-trace control for this experiment (there's no trace
  input to shuffle — the encoder only sees the question).
- Not scaling beyond GSM8K or the existing `google/gemma-4-E2B-it` LoRA
  backbone.

## Architecture

One shared `google/gemma-4-E2B-it` backbone, LoRA-tuned (same tuning
approach as the existing kill-test: LoRA-only, bfloat16, gradient
checkpointing, single 16 GB GPU budget).

### Encoder — `DiffusionReasoningEncoder`

Input: **the question only.** Output: `z`, shape `(K, d_z)` (same bottleneck
shape convention as the existing kill-test: defaults `n_slots=16, d_z=32`).

```
z = torch.randn(B, K, d_z)                       # z_T ~ N(0, I)
question_cond = backbone(question_ids, question_mask,
                          output_hidden_states=True).hidden_states[-1]
for t in reversed(range(T)):                      # T = 6, fixed
    z = refine_block(z, time_embed(t), question_cond, question_mask)
z0 = z                                             # fed to the decoder
```

- `T = 6` refinement steps, fully unrolled in the autograd graph. No
  stop-gradient anywhere in the chain — the trace-reconstruction loss
  backprops through all 6 steps into `refine_block`'s parameters.
- `question_cond`: the shared LoRA backbone's last-layer hidden states over
  the question tokens — the exact same mechanism `_encode_z` already uses
  for trace hidden states in `model.py` (one backbone forward pass over the
  question, reused as fixed conditioning context for every refinement
  step — not recomputed per step).
- `refine_block`: standard diffusion-style block —
  - sinusoidal timestep embedding (`time_embed(t)`), injected additively
    (or FiLM-style scale/shift) into the `K` slot representations,
  - self-attention over the `K` slots,
  - cross-attention from the `K` slots to `question_cond` (masked by
    `question_mask`),
  - feed-forward + residual connections around each sub-block, pre-LN.
  - Runs in float32 (matches the existing `ReasoningEncoder`'s precision
    convention); `z` is cast to the backbone's dtype before being used as a
    soft prefix.

### Decoder

The same shared LoRA backbone, autoregressive, teacher-forced during
training. Input: `z0` projected to `d_model` via a `Linear(d_z, d_model)`
and prepended as `K` soft-prefix embeddings, followed by the question
tokens (mirrors the existing `z`-condition soft-prefix mechanics in
`model.py`: `inputs_embeds = cat([z_up, question_emb, target_emb])`).
Target: the reasoning trace text, EOS-appended, truncated to
`max_trace_tokens`.

## Training

- Single loss: cross-entropy on trace tokens (standard teacher-forced LM
  loss), identical in shape to the existing kill-test's answer-loss
  computation but supervising the trace instead of the answer.
- Backprop flows through the decoder and through all `T = 6` unrolled
  encoder steps into `refine_block`.
- Same optimizer/schedule conventions as `latent_cot/train.py`: AdamW,
  linear warmup + linear decay, grad clipping, LoRA-only trainable
  parameters (plus the new encoder/projection modules, which train in
  float32 like the existing `ReasoningEncoder`).
- Reuses GSM8K loading/parsing (`load_gsm8k`, `split_answer`,
  `strip_calc_annotations`) from `latent_cot/data.py` unchanged.

## Eval metric

Teacher-forced token-level accuracy on trace reconstruction:
`mean(argmax(logits) == trace_target_ids)` over non-pad, non-ignored
positions. No generation-based eval (no ROUGE/BLEU) — matches the existing
kill-test's philosophy of a cheap, fast feasibility signal first.

## New code

- `latent_cot/diffusion_encoder.py` — `DiffusionReasoningEncoder` and its
  `refine_block` (timestep embedding, self-attn, cross-attn, feed-forward).
- `latent_cot/model.py` — MODIFY: add a `reconstruct` condition to
  `LatentCoTModel` (new branch in `__init__`, `forward`, and `generate`)
  that uses `DiffusionReasoningEncoder` instead of `ReasoningEncoder`, and
  decodes the trace instead of the answer. Existing conditions
  (`floor`/`z`/`ceiling`/`z_shuffled`) and their code paths are untouched.
- `latent_cot/config.py` — MODIFY: add `"reconstruct"` to
  `VALID_CONDITIONS`; add `diffusion_steps: int = 6` to `ExperimentConfig`.
- `latent_cot/data.py` — MODIFY: `Collator` gets a `reconstruct` branch
  that tokenizes the trace as the decode target (EOS-appended, truncated to
  `max_trace_tokens`) instead of the answer label.
- `latent_cot/train.py` — MODIFY: `train_and_eval` branches on
  `cfg.condition == "reconstruct"` to run the teacher-forced token-accuracy
  eval path instead of the generation + exact-match path.
- Run via `python -m latent_cot.train --condition reconstruct` — standalone,
  not part of `run_killtest.py`.

## How to read the result

- High token accuracy (well above the majority-class-token floor for GSM8K
  trace text) → the question alone, through a diffusion-style refined `z`,
  carries enough signal for the decoder to reconstruct plausible reasoning.
  A natural next probe (separate experiment, not in scope here) would check
  whether the reconstructed trace's *arithmetic* is actually correct, not
  just fluent.
- Low/near-random token accuracy → the diffusion encoder isn't extracting
  anything useful from the question alone, or the decoder isn't leveraging
  `z`. Isolating which (e.g. by comparing against feeding `z0` = pure noise,
  no refinement) is a separate follow-up, not in scope here.

## Error handling

No new defensive handling beyond what `latent_cot/model.py` already has
(e.g. the vision-tower LoRA-targeting workaround). Single model, single
config, local exploratory run — consistent with the existing kill-test's
error-handling posture.
