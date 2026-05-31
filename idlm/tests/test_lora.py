# idlm/tests/test_lora.py
import math
import torch
import torch.nn as nn
import pytest
from idlm.models.lora import LoRALinear, apply_lora

B, N, D_IN, D_OUT, RANK = 2, 16, 32, 32, 4


def make_base_linear() -> nn.Linear:
    lin = nn.Linear(D_IN, D_OUT, bias=False)
    lin.weight.requires_grad_(False)
    return lin


def test_output_shape():
    lora = LoRALinear(make_base_linear(), rank=RANK, alpha=8.0)
    x = torch.randn(B, N, D_IN)
    lora.current_mask = torch.ones(B, N, 1)
    out = lora(x)
    assert out.shape == (B, N, D_OUT)


def test_lora_zero_mask_matches_base():
    """With current_mask=0, output must equal the frozen base exactly."""
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    # Set lora_B non-zero so any delta would matter
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.1)
    x = torch.randn(B, N, D_IN)
    lora.current_mask = torch.zeros(B, N, 1)
    with torch.no_grad():
        out_lora = lora(x)
        out_base = base(x)
    assert torch.allclose(out_lora, out_base, atol=1e-6)


def test_lora_full_mask_differs_from_base():
    """With current_mask=1 and non-zero lora_B, output differs from base."""
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.01)
    x = torch.randn(B, N, D_IN)
    lora.current_mask = torch.ones(B, N, 1)
    out_lora = lora(x)
    out_base = base(x)
    assert not torch.allclose(out_lora, out_base, atol=1e-6)


def test_lora_init_delta_zero():
    lora = LoRALinear(make_base_linear(), rank=RANK, alpha=8.0)
    assert torch.all(lora.lora_B.weight == 0)


def test_only_lora_params_have_grad():
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    lora.current_mask = torch.ones(B, N, 1)
    x = torch.randn(B, N, D_IN)
    lora(x).sum().backward()
    assert base.weight.grad is None
    assert lora.lora_A.weight.grad is not None
    assert lora.lora_B.weight.grad is not None


def test_apply_lora_replaces_target_modules():
    model = nn.Module()
    model.q_proj = nn.Linear(D_IN, D_OUT, bias=False)
    model.v_proj = nn.Linear(D_IN, D_OUT, bias=False)
    model.other  = nn.Linear(D_IN, D_OUT, bias=False)
    apply_lora(model, ["q_proj", "v_proj"], rank=RANK, alpha=8.0)
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert isinstance(model.other,  nn.Linear)  # untouched


def test_per_position_mask():
    base = make_base_linear()
    lora = LoRALinear(base, rank=RANK, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.1)
    x = torch.randn(B, N, D_IN)
    mask = torch.zeros(B, N, 1)
    mask[:, :N // 2, :] = 1.0
    lora.current_mask = mask
    out = lora(x)
    base_out = base(x)
    assert torch.allclose(out[:, N // 2:], base_out[:, N // 2:], atol=1e-6)
    assert not torch.allclose(out[:, :N // 2], base_out[:, :N // 2], atol=1e-6)
