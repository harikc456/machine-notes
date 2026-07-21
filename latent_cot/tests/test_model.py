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
