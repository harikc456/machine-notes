# Latent-CoT Diffusion-Encoder Reconstruction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `reconstruct` condition to the `latent_cot` package in which a question-only diffusion-style encoder produces a reasoning latent `z`, and the shared LoRA backbone autoregressively reconstructs the GSM8K reasoning trace text from `question + z`, trained end-to-end on trace-reconstruction loss.

**Architecture:** `DiffusionReasoningEncoder` (new module) runs `T=6` fully-unrolled refinement steps starting from Gaussian noise, cross-attending the backbone's question hidden states at every step, to produce `z0` (shape `K x d_z`). `z0` is projected to `d_model` and prepended as a soft prefix to `question + trace` tokens fed into the same shared LoRA backbone, teacher-forced against the trace text. Eval is teacher-forced token-level accuracy — no generation, no separate diffusion loss.

**Tech Stack:** Python >=3.10, PyTorch, transformers, peft (LoRA), pytest. Reuses `latent_cot/config.py`, `latent_cot/data.py`, `latent_cot/model.py`, `latent_cot/train.py` — this plan modifies each, plus one new file.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-28-latent-cot-diffusion-reconstruction-design.md`. Follow it exactly; this plan implements it task-by-task.
- **Do not touch** the existing `floor`/`z`/`ceiling`/`z_shuffled` conditions, `ReasoningEncoder`, or `run_killtest.py`. This is a fifth, independent condition.
- New trainable modules (the diffusion encoder, its projection) run in **float32**; cast `z` to the backbone's dtype (bfloat16) before use as a soft prefix — same convention `ReasoningEncoder`/`_encode_z` already follow in `model.py`.
- **Never hardcode `d_model`.** Pass it in from `LatentCoTModel.__init__` (already resolved there via `base.config.hidden_size` / `text_config.hidden_size`).
- The bottleneck shape stays `n_slots x d_z` (defaults `16 x 32`) — same as the existing kill-test, not something this plan changes.
- Tests that load the model/tokenizer are marked `@pytest.mark.slow` (repo already registers this marker; deselected by default). Pure-tensor tests (no model, no tokenizer) must stay fast and unmarked.
- Determinism: seed via `cfg.seed` — already handled by `train.py`'s `_seed_all`; no new seeding logic needed.
- `python -m latent_cot.train --condition reconstruct` is the only entry point — no changes to `run_killtest.py`.

---

## File Structure

- `latent_cot/diffusion_encoder.py` — NEW. `sinusoidal_embedding(timesteps, dim)`, `RefineBlock` (self-attn + cross-attn + feed-forward + timestep conditioning), `DiffusionReasoningEncoder` (unrolls `RefineBlock` `T` times from noise).
- `latent_cot/config.py` — MODIFY. Add `"reconstruct"` to `VALID_CONDITIONS`; add `diffusion_steps: int` field + validation.
- `latent_cot/data.py` — MODIFY. `Collator` gets a `reconstruct` branch: tokenizes `question_ids`/`question_mask` (encoder input) and `recon_ids`/`recon_mask` (trace-as-target, EOS-appended).
- `latent_cot/model.py` — MODIFY. `LatentCoTModel` gets: a `DiffusionReasoningEncoder` instance for `condition == "reconstruct"`, `_encode_z_diffusion`, `_reconstruct_forward` (shared by `forward()` and eval), `logits_and_labels()` (teacher-forced eval, no generation).
- `latent_cot/train.py` — MODIFY. `train_and_eval` branches: `reconstruct` → teacher-forced token-accuracy eval; everything else → unchanged generation + exact-match path.
- `latent_cot/README.md` — MODIFY. Document the new condition and how to run it.
- `latent_cot/tests/test_diffusion_encoder.py` — NEW. Fast, model-free shape/gradient tests.
- `latent_cot/tests/test_config.py` — MODIFY. Cover the new field/condition.
- `latent_cot/tests/test_data.py` — MODIFY. Cover the new `Collator` branch.
- `latent_cot/tests/test_model.py` — MODIFY. Cover `LatentCoTModel` forward/backward for `reconstruct`.
- `latent_cot/tests/test_train.py` — MODIFY. End-to-end smoke test for `reconstruct`.

---

### Task 1: `DiffusionReasoningEncoder`

**Files:**
- Create: `latent_cot/diffusion_encoder.py`
- Test: `latent_cot/tests/test_diffusion_encoder.py`

**Interfaces:**
- Consumes: nothing (pure `torch.nn.Module`, no backbone/tokenizer dependency).
- Produces: `DiffusionReasoningEncoder(d_model: int, n_slots: int, d_z: int, n_heads: int, n_steps: int)`, callable as `forward(question_hidden: Tensor[B,Tq,d_model], question_kpm: Tensor[B,Tq] bool, True=pad) -> Tensor[B, n_slots, d_z]` (float32). Later tasks (`model.py`) construct it as `DiffusionReasoningEncoder(self.d_model, cfg.n_slots, cfg.d_z, cfg.encoder_heads, cfg.diffusion_steps).float()` and call it the same way `ReasoningEncoder` is called today.

- [ ] **Step 1: Write the failing tests**

Create `latent_cot/tests/test_diffusion_encoder.py`:

```python
import torch
from latent_cot.diffusion_encoder import sinusoidal_embedding, DiffusionReasoningEncoder


def test_sinusoidal_embedding_shape_and_finite():
    t = torch.tensor([0, 1, 5])
    emb = sinusoidal_embedding(t, dim=16)
    assert emb.shape == (3, 16)
    assert torch.isfinite(emb).all()


def test_sinusoidal_embedding_distinct_timesteps_differ():
    t = torch.tensor([0, 3])
    emb = sinusoidal_embedding(t, dim=16)
    assert not torch.allclose(emb[0], emb[1])


def test_encoder_output_shape_and_finite():
    B, Tq, d_model, K, d_z = 2, 5, 64, 16, 32
    enc = DiffusionReasoningEncoder(d_model, K, d_z, n_heads=8, n_steps=3)
    question_hidden = torch.randn(B, Tq, d_model)
    kpm = torch.zeros(B, Tq, dtype=torch.bool)
    kpm[0, 4:] = True  # last position padded for example 0
    z0 = enc(question_hidden, kpm)
    assert z0.shape == (B, K, d_z)
    assert torch.isfinite(z0).all()


def test_encoder_gradient_flows_through_all_steps():
    B, Tq, d_model, K, d_z = 2, 5, 32, 4, 8
    enc = DiffusionReasoningEncoder(d_model, K, d_z, n_heads=2, n_steps=4)
    question_hidden = torch.randn(B, Tq, d_model, requires_grad=True)
    kpm = torch.zeros(B, Tq, dtype=torch.bool)
    z0 = enc(question_hidden, kpm)
    z0.sum().backward()
    # gradient must reach both the conditioning input and every refine-block param
    assert question_hidden.grad is not None and torch.isfinite(question_hidden.grad).all()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_encoder_is_stochastic_across_calls():
    # z_T ~ N(0, I) is redrawn every forward call -> two calls on the same
    # input should not produce identical output (guards against accidentally
    # caching / fixing the initial noise).
    torch.manual_seed(0)
    B, Tq, d_model, K, d_z = 1, 3, 16, 4, 8
    enc = DiffusionReasoningEncoder(d_model, K, d_z, n_heads=2, n_steps=2)
    question_hidden = torch.randn(B, Tq, d_model)
    kpm = torch.zeros(B, Tq, dtype=torch.bool)
    z_a = enc(question_hidden, kpm)
    z_b = enc(question_hidden, kpm)
    assert not torch.allclose(z_a, z_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest latent_cot/tests/test_diffusion_encoder.py -q`
Expected: FAIL/ERROR — `latent_cot.diffusion_encoder` doesn't exist yet.

- [ ] **Step 3: Implement `latent_cot/diffusion_encoder.py`**

```python
from __future__ import annotations
import math
import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer/DDPM sinusoidal timestep embedding.
    timesteps: (B,) int/long tensor. Returns (B, dim) float tensor."""
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class RefineBlock(nn.Module):
    """One diffusion-style refinement step: timestep conditioning, self-attn
    over the K latent slots, cross-attn to question hidden states, feed-forward.
    Shared across all T steps (only the timestep embedding differs per step),
    matching standard diffusion-network practice."""

    def __init__(self, d_z: int, d_model: int, n_heads: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(d_z, d_z), nn.SiLU(), nn.Linear(d_z, d_z)
        )
        self.ln_self = nn.LayerNorm(d_z)
        self.self_attn = nn.MultiheadAttention(d_z, n_heads, batch_first=True)
        self.ln_cross = nn.LayerNorm(d_z)
        self.cross_attn = nn.MultiheadAttention(
            d_z, n_heads, kdim=d_model, vdim=d_model, batch_first=True
        )
        self.ln_ff = nn.LayerNorm(d_z)
        self.ff = nn.Sequential(
            nn.Linear(d_z, 4 * d_z), nn.GELU(), nn.Linear(4 * d_z, d_z)
        )

    def forward(
        self, z: torch.Tensor, t_emb: torch.Tensor,
        question_hidden: torch.Tensor, question_kpm: torch.Tensor,
    ) -> torch.Tensor:
        z = z + self.time_mlp(t_emb).unsqueeze(1)  # broadcast over K slots

        h = self.ln_self(z)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        z = z + attn_out

        h = self.ln_cross(z)
        cross_out, _ = self.cross_attn(
            h, question_hidden, question_hidden,
            key_padding_mask=question_kpm, need_weights=False,
        )
        z = z + cross_out

        h = self.ln_ff(z)
        z = z + self.ff(h)
        return z


class DiffusionReasoningEncoder(nn.Module):
    """Produces a reasoning latent z (K x d_z) from the question ALONE, via
    T fully-unrolled refinement steps starting from Gaussian noise. No
    ground-truth z, no denoising-score-matching loss: the only supervision
    is whatever loss the caller backprops through `forward`'s output. Runs
    in float32, matching ReasoningEncoder's precision convention."""

    def __init__(self, d_model: int, n_slots: int, d_z: int, n_heads: int, n_steps: int):
        super().__init__()
        self.n_slots = n_slots
        self.d_z = d_z
        self.n_steps = n_steps
        self.block = RefineBlock(d_z, d_model, n_heads)

    def forward(self, question_hidden: torch.Tensor, question_kpm: torch.Tensor) -> torch.Tensor:
        question_hidden = question_hidden.float()
        B = question_hidden.size(0)
        z = torch.randn(B, self.n_slots, self.d_z, device=question_hidden.device)
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B,), t, device=question_hidden.device, dtype=torch.long)
            t_emb = sinusoidal_embedding(t_batch, self.d_z)
            z = self.block(z, t_emb, question_hidden, question_kpm)
        return z
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest latent_cot/tests/test_diffusion_encoder.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add latent_cot/diffusion_encoder.py latent_cot/tests/test_diffusion_encoder.py
git commit -m "feat(latent_cot): add DiffusionReasoningEncoder (question-only, unrolled refinement)"
```

---

### Task 2: Config — `reconstruct` condition + `diffusion_steps`

**Files:**
- Modify: `latent_cot/config.py:6` (`VALID_CONDITIONS`), `latent_cot/config.py:14-18` (bottleneck fields), `latent_cot/config.py:53-63` (`__post_init__`)
- Test: `latent_cot/tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ExperimentConfig.diffusion_steps: int` (default `6`); `"reconstruct"` accepted by `ExperimentConfig(condition=...)`. Task 4 reads `cfg.diffusion_steps` when constructing `DiffusionReasoningEncoder`.

- [ ] **Step 1: Write the failing tests**

Add to `latent_cot/tests/test_config.py`:

```python
def test_reconstruct_condition_valid():
    cfg = ExperimentConfig(condition="reconstruct")
    assert cfg.condition == "reconstruct"


def test_diffusion_steps_default_and_validation():
    cfg = ExperimentConfig()
    assert cfg.diffusion_steps == 6
    with pytest.raises(ValueError):
        ExperimentConfig(diffusion_steps=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest latent_cot/tests/test_config.py -q`
Expected: FAIL — `"reconstruct"` not in `VALID_CONDITIONS`; `diffusion_steps` doesn't exist.

- [ ] **Step 3: Implement the config changes**

In `latent_cot/config.py`, change:

```python
VALID_CONDITIONS = {"floor", "z", "ceiling", "z_shuffled"}
```
to:
```python
VALID_CONDITIONS = {"floor", "z", "ceiling", "z_shuffled", "reconstruct"}
```

Add a field next to `encoder_heads` (in the "Reasoning-encoding bottleneck" group):

```python
    encoder_heads: int = 8     # heads in the encoder's cross-attention
    diffusion_steps: int = 6  # T: refinement steps for DiffusionReasoningEncoder
```

Add validation in `__post_init__`, alongside the existing `encoder_heads` check:

```python
        if self.encoder_heads < 1:
            raise ValueError(f"encoder_heads must be >= 1, got {self.encoder_heads}")
        if self.diffusion_steps < 1:
            raise ValueError(f"diffusion_steps must be >= 1, got {self.diffusion_steps}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest latent_cot/tests/test_config.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add latent_cot/config.py latent_cot/tests/test_config.py
git commit -m "feat(latent_cot): add reconstruct condition and diffusion_steps to ExperimentConfig"
```

---

### Task 3: Collator `reconstruct` branch

**Files:**
- Modify: `latent_cot/data.py:107-187` (`Collator` class)
- Test: `latent_cot/tests/test_data.py`

**Interfaces:**
- Consumes: `ExperimentConfig` with `condition="reconstruct"` (Task 2); `_pad`, `_mask_from` (existing helpers in `data.py`).
- Produces: `Collator(tok, cfg, "reconstruct", include_answer)` returns a batch dict with keys `question_ids`, `question_mask`, `recon_ids`, `recon_mask` (both padded right — this condition is always teacher-forced, no batched-generation left-padding requirement). `recon_ids`/`recon_mask` are always present regardless of `include_answer` (there is no generation-only eval mode for this condition). Task 4's model code consumes exactly these four keys.

- [ ] **Step 1: Write the failing tests**

Add to `latent_cot/tests/test_data.py` (near the other `@pytest.mark.slow` collator tests):

```python
@pytest.mark.slow
def test_collator_reconstruct_shapes(tok):
    cfg = ExperimentConfig(condition="reconstruct")
    coll = Collator(tok, cfg, "reconstruct", include_answer=True)
    batch = coll(_ROWS)
    B = len(_ROWS)
    for k in ("question_ids", "question_mask", "recon_ids", "recon_mask"):
        assert batch[k].shape[0] == B and batch[k].ndim == 2
    assert "trace_ids" not in batch  # encoder never sees the trace
    assert "answer_ids" not in batch


@pytest.mark.slow
def test_collator_reconstruct_present_even_without_answer(tok):
    """Unlike z/floor/ceiling, reconstruct has no generation-only eval mode:
    recon_ids must be present regardless of include_answer."""
    cfg = ExperimentConfig(condition="reconstruct")
    coll = Collator(tok, cfg, "reconstruct", include_answer=False)
    batch = coll(_ROWS)
    assert "recon_ids" in batch and "recon_mask" in batch


@pytest.mark.slow
def test_collator_reconstruct_targets_end_with_eos(tok):
    cfg = ExperimentConfig(condition="reconstruct")
    coll = Collator(tok, cfg, "reconstruct", include_answer=True)
    batch = coll(_ROWS)
    for row_ids, row_mask in zip(batch["recon_ids"].tolist(), batch["recon_mask"].tolist()):
        last_real = max(i for i, m in enumerate(row_mask) if m == 1)
        assert row_ids[last_real] == tok.eos_token_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest latent_cot/tests/test_data.py --run-slow -k reconstruct -q`
Expected: FAIL/ERROR — `Collator` has no `"reconstruct"` branch, raises or falls through incorrectly.

- [ ] **Step 3: Implement the `Collator` branch**

In `latent_cot/data.py`, add a `_recon_ids` helper next to `_answer_ids`:

```python
    def _recon_ids(self, trace: str) -> list[int]:
        ids = self.tok(trace, add_special_tokens=False)["input_ids"]
        return (ids + [self.eos_id])[: self.cfg.max_trace_tokens]
```

In `Collator.__call__`, add a branch before the existing `# z / z_shuffled` section (so it doesn't fall through into that code):

```python
        if c == "reconstruct":
            q_ids, recon_ids = [], []
            for r in rows:
                q_ids.append(self._enc(f"{r['question']}\nAnswer:",
                                       self.cfg.max_question_tokens, add_special=True))
                recon_ids.append(self._recon_ids(r["trace"]))
            qi = _pad(q_ids, self.pad_id)
            ri = _pad(recon_ids, self.pad_id)
            batch["question_ids"] = qi
            batch["question_mask"] = _mask_from(q_ids, qi.size(1))
            batch["recon_ids"] = ri
            batch["recon_mask"] = _mask_from(recon_ids, ri.size(1))
            return batch

```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest latent_cot/tests/test_data.py --run-slow -k reconstruct -q`
Expected: PASS (3 tests).

Also run the full data test file to confirm no regressions:
Run: `pytest latent_cot/tests/test_data.py --run-slow -q`
Expected: PASS (all tests, including pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add latent_cot/data.py latent_cot/tests/test_data.py
git commit -m "feat(latent_cot): add reconstruct branch to Collator"
```

---

### Task 4: `LatentCoTModel` integration

**Files:**
- Modify: `latent_cot/model.py:68-74` (`__init__` bottleneck setup), `latent_cot/model.py:104-112` (near `_encode_z`), `latent_cot/model.py:140-170` (`forward`)
- Test: `latent_cot/tests/test_model.py`

**Interfaces:**
- Consumes: `DiffusionReasoningEncoder` (Task 1); `Collator`'s `reconstruct` batch keys `question_ids`/`question_mask`/`recon_ids`/`recon_mask` (Task 3); `cfg.diffusion_steps` (Task 2).
- Produces: `LatentCoTModel(cfg)` with `condition="reconstruct"` supports `model(batch) -> loss` (via `forward`) and `model.logits_and_labels(batch) -> (logits, labels)` (teacher-forced, no-grad, for eval). Task 5's `train.py` calls `logits_and_labels`.

- [ ] **Step 1: Write the failing tests**

Add to `latent_cot/tests/test_model.py`:

```python
@pytest.mark.slow
def test_reconstruct_forward_returns_scalar_loss_with_grad():
    from latent_cot.model import LatentCoTModel
    cfg = ExperimentConfig(condition="reconstruct", n_slots=4, d_z=16, lora_r=4,
                           diffusion_steps=2, batch_size=2,
                           max_trace_tokens=64, max_question_tokens=32)
    model = LatentCoTModel(cfg)
    coll = Collator(model.tokenizer, cfg, "reconstruct", include_answer=True)
    loss = model(coll(_ROWS))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.trainable_parameters() if p.grad is not None]
    assert len(grads) > 0


@pytest.mark.slow
def test_reconstruct_logits_and_labels_shapes_and_no_grad():
    from latent_cot.model import LatentCoTModel
    cfg = ExperimentConfig(condition="reconstruct", n_slots=4, d_z=16, lora_r=4,
                           diffusion_steps=2, batch_size=2,
                           max_trace_tokens=64, max_question_tokens=32)
    model = LatentCoTModel(cfg)
    coll = Collator(model.tokenizer, cfg, "reconstruct", include_answer=True)
    batch = coll(_ROWS)
    logits, labels = model.logits_and_labels(batch)
    assert logits.shape[0] == labels.shape[0] == 2
    assert logits.shape[1] == labels.shape[1]
    assert not logits.requires_grad
    # some positions must be supervised (not all -100)
    assert (labels != -100).any()
```

`test_model.py` already imports `pytest`, `ExperimentConfig`, and `Collator` at module scope (added inline after the first test, ahead of the existing `@pytest.mark.slow` tests) — no new imports needed; the two tests above go directly after the existing `test_generate_returns_strings` test using those same names.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest latent_cot/tests/test_model.py --run-slow -k reconstruct -q`
Expected: FAIL/ERROR — `condition="reconstruct"` not handled in `LatentCoTModel.__init__`/`forward`; `logits_and_labels` doesn't exist.

- [ ] **Step 3: Implement the model changes**

In `latent_cot/model.py`, add the import:

```python
from latent_cot.diffusion_encoder import DiffusionReasoningEncoder
```

In `LatentCoTModel.__init__`, change:

```python
        if self.condition in ("z", "z_shuffled"):
            self.encoder = ReasoningEncoder(
                self.d_model, cfg.n_slots, cfg.d_z, cfg.encoder_heads
            ).float()
            self.up = nn.Linear(cfg.d_z, self.d_model).float()
```
to:
```python
        if self.condition in ("z", "z_shuffled"):
            self.encoder = ReasoningEncoder(
                self.d_model, cfg.n_slots, cfg.d_z, cfg.encoder_heads
            ).float()
            self.up = nn.Linear(cfg.d_z, self.d_model).float()

        if self.condition == "reconstruct":
            self.diff_encoder = DiffusionReasoningEncoder(
                self.d_model, cfg.n_slots, cfg.d_z, cfg.encoder_heads, cfg.diffusion_steps
            ).float()
            self.up = nn.Linear(cfg.d_z, self.d_model).float()
```

Add `_encode_z_diffusion` and `_reconstruct_forward` right after the existing `_encode_z` method (after line 112):

```python
    def _encode_z_diffusion(self, question_ids, question_mask) -> torch.Tensor:
        out = self.backbone(
            input_ids=question_ids, attention_mask=question_mask, output_hidden_states=True
        )
        hidden = out.hidden_states[-1]                 # (B, Tq, d_model) bf16
        kpm = question_mask == 0                        # True = pad
        z = self.diff_encoder(hidden, kpm)               # fp32 (B, K, d_z)
        z_up = self.up(z)                                # fp32 (B, K, d_model)
        return z_up.to(self._embed(question_ids[:, :1]).dtype)

    def _reconstruct_forward(self, batch: dict):
        """Shared by `forward()` (returns .loss) and `logits_and_labels()`
        (returns .logits + labels for teacher-forced eval). `batch` must
        already be moved to self.device."""
        z_up = self._encode_z_diffusion(batch["question_ids"], batch["question_mask"])
        q_emb = self._embed(batch["question_ids"])
        r_emb = self._embed(batch["recon_ids"])
        inputs_embeds = torch.cat([z_up, q_emb, r_emb], dim=1)

        B, K = z_up.shape[0], z_up.shape[1]
        z_mask = torch.ones(B, K, dtype=torch.long, device=self.device)
        attn = torch.cat([z_mask, batch["question_mask"], batch["recon_mask"]], dim=1)

        prefix_len = K + q_emb.size(1)
        ignore = torch.full((B, prefix_len), -100, dtype=torch.long, device=self.device)
        recon_labels = batch["recon_ids"].masked_fill(batch["recon_mask"] == 0, -100)
        labels = torch.cat([ignore, recon_labels], dim=1)

        z_ids = self._placeholder_ids(B, K)
        full_ids = torch.cat([z_ids, batch["question_ids"], batch["recon_ids"]], dim=1)
        per_layer_inputs = self._per_layer_inputs(full_ids)

        out = self.backbone(
            inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels,
            per_layer_inputs=per_layer_inputs,
        )
        return out, labels

    @torch.no_grad()
    def logits_and_labels(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._move(batch)
        out, labels = self._reconstruct_forward(batch)
        return out.logits, labels
```

In `forward`, insert a new branch between the `floor`/`ceiling` early return and the existing `z_up = self._encode_z(...)` line — i.e. change:

```python
    def forward(self, batch: dict) -> torch.Tensor:
        batch = self._move(batch)
        if self.condition in ("floor", "ceiling"):
            return self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss

        z_up = self._encode_z(batch["trace_ids"], batch["trace_mask"])
```

to:

```python
    def forward(self, batch: dict) -> torch.Tensor:
        batch = self._move(batch)
        if self.condition in ("floor", "ceiling"):
            return self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss

        if self.condition == "reconstruct":
            out, _ = self._reconstruct_forward(batch)
            return out.loss

        z_up = self._encode_z(batch["trace_ids"], batch["trace_mask"])
```

Everything after that line (the rest of the existing `z`/`z_shuffled` forward body, through the final `return self.backbone(...)` call) stays exactly as it is today — this change only inserts the new `if` block, it does not touch any other line in `forward`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest latent_cot/tests/test_model.py --run-slow -k reconstruct -q`
Expected: PASS (2 tests).

Run the full model test file to confirm no regressions to existing conditions:
Run: `pytest latent_cot/tests/test_model.py --run-slow -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add latent_cot/model.py latent_cot/tests/test_model.py
git commit -m "feat(latent_cot): wire DiffusionReasoningEncoder into LatentCoTModel as reconstruct condition"
```

---

### Task 5: `train.py` token-accuracy eval path + README

**Files:**
- Modify: `latent_cot/train.py:79-92` (`train_and_eval` eval section)
- Modify: `latent_cot/README.md`
- Test: `latent_cot/tests/test_train.py`

**Interfaces:**
- Consumes: `model.logits_and_labels(batch)` (Task 4); `Collator` with `condition="reconstruct"` (Task 3).
- Produces: `train_and_eval(cfg)` returns, for `cfg.condition == "reconstruct"`, a dict with keys `condition`, `token_accuracy` (float in `[0,1]`), `n_eval` (int), `final_train_loss` (float) — same shape convention as the existing dict, with `token_accuracy` replacing `eval_accuracy`.

- [ ] **Step 1: Write the failing test**

Add to `latent_cot/tests/test_train.py`:

```python
@pytest.mark.slow
def test_train_and_eval_reconstruct_smoke():
    from latent_cot.config import ExperimentConfig
    from latent_cot.train import train_and_eval
    cfg = ExperimentConfig(
        condition="reconstruct", n_slots=4, d_z=16, lora_r=4, diffusion_steps=2,
        epochs=1, batch_size=2, grad_accum_steps=1,
        max_train_samples=8, max_eval_samples=8,
        max_trace_tokens=64, max_question_tokens=32,
    )
    result = train_and_eval(cfg)
    assert set(result) >= {"condition", "token_accuracy", "n_eval", "final_train_loss"}
    assert 0.0 <= result["token_accuracy"] <= 1.0
    assert result["n_eval"] == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest latent_cot/tests/test_train.py --run-slow -k reconstruct -q`
Expected: FAIL — current `train_and_eval` always calls `model.generate(...)` and computes `exact_match`, which doesn't apply to `reconstruct` (no `label_text`-comparable output) and doesn't produce a `token_accuracy` key.

- [ ] **Step 3: Implement the eval branch**

In `latent_cot/train.py`, replace the eval section of `train_and_eval`:

```python
    # ---- eval ----
    model.eval()
    preds, golds = [], []
    for batch in eval_loader:
        out = model.generate(batch, max_new_tokens=cfg.max_answer_tokens)
        preds.extend(out)
        golds.extend(batch["label_text"])

    return {
        "condition": cfg.condition,
        "eval_accuracy": exact_match(preds, golds),
        "n_eval": len(golds),
        "final_train_loss": final_loss,
    }
```

with:

```python
    # ---- eval ----
    model.eval()
    if cfg.condition == "reconstruct":
        correct, total, n_eval = 0, 0, 0
        with torch.no_grad():
            for batch in eval_loader:
                logits, labels = model.logits_and_labels(batch)
                preds_tok = logits[:, :-1, :].argmax(-1)
                targets = labels[:, 1:].to(preds_tok.device)
                mask = targets != -100
                correct += (preds_tok == targets)[mask].sum().item()
                total += mask.sum().item()
                n_eval += targets.size(0)
        return {
            "condition": cfg.condition,
            "token_accuracy": (correct / total) if total else 0.0,
            "n_eval": n_eval,
            "final_train_loss": final_loss,
        }

    preds, golds = [], []
    for batch in eval_loader:
        out = model.generate(batch, max_new_tokens=cfg.max_answer_tokens)
        preds.extend(out)
        golds.extend(batch["label_text"])

    return {
        "condition": cfg.condition,
        "eval_accuracy": exact_match(preds, golds),
        "n_eval": len(golds),
        "final_train_loss": final_loss,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest latent_cot/tests/test_train.py --run-slow -k reconstruct -q`
Expected: PASS.

Run the full train test file to confirm no regressions:
Run: `pytest latent_cot/tests/test_train.py --run-slow -q`
Expected: PASS (all tests).

- [ ] **Step 5: Update the README**

In `latent_cot/README.md`, add a new section after "## Gotchas baked into the design" (before "## Run"):

```markdown
## Reconstruction probe (`reconstruct` condition)

A separate, standalone diagnostic — not part of the four-way kill-test
above. `DiffusionReasoningEncoder` produces `z` from the **question
alone** (never sees the trace) via `T=6` fully-unrolled refinement steps
starting from Gaussian noise, cross-attending the backbone's question
hidden states at each step. The same shared LoRA backbone then
autoregressively reconstructs the reasoning trace text from
`question + z`, teacher-forced. No separate diffusion/denoising loss —
only the trace-reconstruction cross-entropy, backpropagated through the
entire unrolled encoder. Eval metric is teacher-forced token-level
accuracy, not exact-match or generation.

Full design: `docs/superpowers/specs/2026-07-28-latent-cot-diffusion-reconstruction-design.md`.

Run standalone (not part of `run_killtest.py`):
```bash
python -m latent_cot.train --condition reconstruct --max-train-samples 8 --epochs 1
```
```

- [ ] **Step 6: Commit**

```bash
git add latent_cot/train.py latent_cot/tests/test_train.py latent_cot/README.md
git commit -m "feat(latent_cot): token-accuracy eval path for reconstruct condition + docs"
```

---

## Post-plan verification

- [ ] **Full fast suite:** `pytest latent_cot/tests -m "not slow" -q` → all pass.
- [ ] **Full slow suite, one file at a time** (GPU memory convention already documented in `latent_cot/README.md`):
  - `pytest latent_cot/tests/test_diffusion_encoder.py --run-slow -q`
  - `pytest latent_cot/tests/test_config.py --run-slow -q`
  - `pytest latent_cot/tests/test_data.py --run-slow -q`
  - `pytest latent_cot/tests/test_model.py --run-slow -q`
  - `pytest latent_cot/tests/test_train.py --run-slow -q`
- [ ] **Confirm existing kill-test unaffected:** `pytest latent_cot/tests/test_model.py --run-slow -k "not reconstruct" -q` still passes with the same behavior as before this plan (no changes to `floor`/`z`/`ceiling`/`z_shuffled` code paths).
