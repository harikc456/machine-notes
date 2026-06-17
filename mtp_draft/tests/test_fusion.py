import torch
import pytest
from mtp_draft.models.fusion import TeacherFeatureFusion

B, N_LAYERS, D_TEACHER, D_DRAFT = 3, 4, 2048, 512


@pytest.fixture
def fusion():
    return TeacherFeatureFusion(n_teacher_layers=N_LAYERS, d_teacher=D_TEACHER, d_draft=D_DRAFT)


def test_output_shape(fusion):
    x = torch.randn(B, N_LAYERS, D_TEACHER)
    out = fusion(x)
    assert out.shape == (B, D_DRAFT)


def test_output_differs_per_batch(fusion):
    x = torch.randn(B, N_LAYERS, D_TEACHER)
    out = fusion(x)
    assert not torch.allclose(out[0], out[1])


def test_gradients_flow(fusion):
    x = torch.randn(B, N_LAYERS, D_TEACHER, requires_grad=False)
    out = fusion(x)
    loss = out.sum()
    loss.backward()
    for name, p in fusion.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


def test_power_of_two_check():
    with pytest.raises(AssertionError):
        TeacherFeatureFusion(n_teacher_layers=3, d_teacher=64, d_draft=32)
