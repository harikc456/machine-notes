import torch
import pytest
from mtp_draft.models.lora_lm_head import LoRALMHead

VOCAB, D_TEACHER, RANK = 100, 64, 4
B, S = 2, 8


@pytest.fixture
def frozen_weight():
    return torch.randn(VOCAB, D_TEACHER)


@pytest.fixture
def head(frozen_weight):
    return LoRALMHead(frozen_weight, lora_rank=RANK)


def test_output_shape(head):
    x = torch.randn(B, S, D_TEACHER)
    out = head(x)
    assert out.shape == (B, S, VOCAB)


def test_frozen_weight_no_grad(head):
    """Registered buffer must not accumulate gradients."""
    x = torch.randn(B, S, D_TEACHER)
    head(x).sum().backward()
    assert head.weight.grad is None


def test_lora_params_have_grad(head):
    x = torch.randn(B, S, D_TEACHER)
    head(x).sum().backward()
    assert head.lora_A.grad is not None
    assert head.lora_B.grad is not None


def test_lora_B_init_zero(frozen_weight):
    """B=0 means LoRA starts as a no-op: output should equal frozen-only output."""
    head = LoRALMHead(frozen_weight, lora_rank=RANK)
    assert torch.all(head.lora_B == 0)
    x = torch.randn(1, 1, D_TEACHER)
    out_lora = head(x)
    out_frozen = x @ frozen_weight.T
    assert torch.allclose(out_lora, out_frozen, atol=1e-5)


def test_weight_not_updated_after_backward(head):
    """Frozen weight buffer must be unchanged after an optimizer step."""
    import copy
    w_before = head.weight.clone()
    opt = torch.optim.AdamW([head.lora_A, head.lora_B], lr=1e-3)
    x = torch.randn(B, S, D_TEACHER)
    head(x).sum().backward()
    opt.step()
    assert torch.allclose(head.weight, w_before)
