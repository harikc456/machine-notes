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
