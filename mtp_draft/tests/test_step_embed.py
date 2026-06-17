import torch
import pytest
from mtp_draft.models.step_embed import StepEmbedding

B, S, D = 2, 8, 512


@pytest.fixture
def embed():
    return StepEmbedding(d_model=D, max_steps=16)


def test_output_shape(embed):
    steps = torch.arange(1, S + 1).unsqueeze(0).expand(B, -1)
    out = embed(steps)
    assert out.shape == (B, S, D)


def test_different_steps_differ(embed):
    steps = torch.tensor([[1, 2]])
    out = embed(steps)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_same_step_same_output(embed):
    s1 = torch.tensor([[3, 3]])
    out = embed(s1)
    assert torch.allclose(out[0, 0], out[0, 1])


def test_gradients_flow(embed):
    steps = torch.arange(1, S + 1).unsqueeze(0).expand(B, -1)
    out = embed(steps)
    out.sum().backward()
    for name, p in embed.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
