import torch
from latent_cot.model import ReasoningEncoder


def test_encoder_output_shape_and_finite():
    B, T, d_model, K, d_z = 2, 7, 64, 16, 32
    enc = ReasoningEncoder(d_model, K, d_z, n_heads=8)
    hidden = torch.randn(B, T, d_model)
    kpm = torch.zeros(B, T, dtype=torch.bool)
    kpm[0, 5:] = True  # last two positions are padding for example 0
    z = enc(hidden, kpm)
    assert z.shape == (B, K, d_z)
    assert torch.isfinite(z).all()


import pytest
from latent_cot.config import ExperimentConfig
from latent_cot.data import Collator

_ROWS = [
    {"question": "2+2?", "trace": "add two and two\nget four", "label": "4"},
    {"question": "3+5?", "trace": "add three and five\nget eight", "label": "8"},
]


@pytest.mark.slow
@pytest.mark.parametrize("condition", ["floor", "ceiling", "z", "z_shuffled"])
def test_forward_returns_scalar_loss_with_grad(condition):
    from latent_cot.model import LatentCoTModel
    cfg = ExperimentConfig(condition=condition, n_slots=4, d_z=16, lora_r=4,
                           batch_size=2, max_trace_tokens=64, max_question_tokens=32)
    model = LatentCoTModel(cfg)
    coll = Collator(model.tokenizer, cfg, condition, include_answer=True)
    loss = model(coll(_ROWS))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.trainable_parameters() if p.grad is not None]
    assert len(grads) > 0


@pytest.mark.slow
@pytest.mark.parametrize("condition", ["floor", "z"])
def test_generate_returns_strings(condition):
    from latent_cot.model import LatentCoTModel
    cfg = ExperimentConfig(condition=condition, n_slots=4, d_z=16, lora_r=4,
                           max_trace_tokens=64, max_question_tokens=32)
    model = LatentCoTModel(cfg)
    coll = Collator(model.tokenizer, cfg, condition, include_answer=False)
    out = model.generate(coll(_ROWS), max_new_tokens=8)
    assert isinstance(out, list) and len(out) == 2 and all(isinstance(s, str) for s in out)


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
