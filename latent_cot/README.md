# Latent-CoT Reasoning-Encoding Kill-Test

## Hypothesis
A fixed-size continuous encoding `z` (shape `K × d_z`, with `K·d_z` far below the
trace's token footprint) can carry a GSM8K reasoning trace well enough for an
answer decoder to produce the correct final number — approaching the accuracy of
feeding the full text chain-of-thought. If this fails, "reason in latent space
instead of tokens" is dead cheaply. If it holds, it earns the right to scale to
long-trace datasets (where the token savings are the actual payoff — GSM8K traces
are short, so this tests the *mechanism*, not the *payoff*).

## Design
Shared `google/gemma-4-E2B-it` backbone, LoRA-tuned. Encoder `E`: K learnable
queries cross-attend the trace's last-layer hidden states → `K × d_z` bottleneck.
`z` is projected to `d_model` and prepended as K soft-prefix embeddings to the
answer decoder (same backbone).

## Conditions
- **floor** — `question → answer`. No reasoning. Lower reference.
- **z** — `question + z → answer`. The idea under test.
- **ceiling** — `question + full trace text → answer`. Upper reference.
- **z_shuffled** — like `z`, but reasoning steps are shuffled before encoding.
  The control: if `z_shuffled ≈ z`, then `z` isn't using reasoning structure and
  a positive `z` result is an illusion (the model is answering from the question).

## How to read the result
- `z ≫ floor` and `z ≈ ceiling` → reasoning compresses into `z`. Success.
- `z ≈ floor` → the bottleneck can't hold reasoning. Idea killed (cheaply).
- `z ≈ z_shuffled` → `z` isn't carrying reasoning; disregard any apparent gain.

## Gotchas baked into the design
- **Bottleneck must bite:** defaults `K=16, d_z=32` (512 scalars ≪ trace footprint).
  Do not inflate these until the squeeze is real, or "it works" proves nothing.
- **LoRA can memorize answers** — that's exactly what `z_shuffled` controls for.
- **Short traces:** GSM8K reasoning is ~40–80 tokens, so token savings here are
  modest by design. This is a feasibility gate, not the payoff demo.

## Run
```bash
# fast unit tests (no model)
pytest latent_cot/tests -q

# slow tests (loads Gemma; needs GPU)
pytest latent_cot/tests -q --run-slow

# tiny end-to-end smoke (8 train / 8 eval)
python -m latent_cot.train --condition z --max-train-samples 8 --epochs 1

# full four-way kill-test (writes latent_cot/runs/)
python -m latent_cot.run_killtest
```

## Config
All knobs live in `latent_cot/config.py` (`ExperimentConfig`). Override via a YAML
file passed with `--config`, or the CLI flags on `train.py`.
