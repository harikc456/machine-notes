# idlm/tests/test_model.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.model import CausalLM
from idlm.models.idlm_model import IDLMCausalLM

B, N, V, D, H, L_layers = 2, 16, 256, 32, 4, 2
MASK_ID = 50256


def make_ar_model() -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L_layers,
        vocab_size=V, seq_len=N,
        ffn_hidden=86, dropout=0.0,
    )
    return CausalLM(cfg)


def make_idlm(ar_model: CausalLM) -> IDLMCausalLM:
    return IDLMCausalLM(ar_model, lora_rank=4, lora_alpha=8.0,
                        lora_target_modules=["q_proj", "v_proj"])


def test_output_shape():
    """Forward over 2N tokens returns (B, 2N, V) logits."""
    model = make_idlm(make_ar_model())
    tokens = torch.randint(0, V, (B, 2 * N))
    mask = torch.zeros(B, 2 * N, 1)
    mask[:, :N, :] = 1.0
    logits = model(tokens, mask)
    assert logits.shape == (B, 2 * N, V)


def test_ar_weights_frozen():
    """All non-LoRA parameters must have requires_grad=False."""
    ar = make_ar_model()
    model = make_idlm(ar)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_zero_mask_equals_base_ar():
    """With use_lora_mask=0 everywhere, output equals frozen AR model output."""
    ar = make_ar_model()
    model = make_idlm(ar)
    from idlm.models.lora import LoRALinear
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.weight.fill_(0.05)
    tokens = torch.randint(0, V, (B, 2 * N))
    mask = torch.zeros(B, 2 * N, 1)
    with torch.no_grad():
        idlm_out = model(tokens, mask)
        ar_out, _ = ar(tokens)
    assert torch.allclose(idlm_out, ar_out, atol=1e-5)


def test_only_lora_grads_flow():
    """Backward should only leave gradients on LoRA params."""
    ar = make_ar_model()
    model = make_idlm(ar)
    tokens = torch.randint(0, V, (B, 2 * N))
    mask = torch.ones(B, 2 * N, 1)
    logits = model(tokens, mask)
    logits.sum().backward()
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.grad is not None, f"{name} missing grad"
        else:
            assert param.grad is None, f"{name} should have no grad"


def test_lora_count():
    """Number of LoRA layers = n_layers * len(target_modules)."""
    from idlm.models.lora import LoRALinear
    ar = make_ar_model()
    model = make_idlm(ar)
    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    # 2 target modules (q_proj, v_proj) * L_layers = 4
    assert len(lora_layers) == L_layers * 2


# --- Loss tests ---

def compute_idlm_loss(model, x_0, device, mask_id=MASK_ID):
    """Helper: build 2L input, run forward, return (loss, l_mask, l_clean, lam)."""
    import torch.nn.functional as F
    B, L = x_0.shape
    x_t = torch.full_like(x_0, mask_id)
    tokens = torch.cat([x_t, x_0], dim=1)          # (B, 2L)
    mask = torch.zeros(B, 2 * L, 1, device=device)
    mask[:, :L, :] = 1.0
    logits = model(tokens, mask)                    # (B, 2L, V)
    l_mask = F.cross_entropy(
        logits[:, :L].reshape(-1, logits.size(-1)),
        x_0.reshape(-1),
    )
    l_clean = F.cross_entropy(
        logits[:, L:2 * L - 1].reshape(-1, logits.size(-1)),
        x_0[:, 1:].reshape(-1),
    )
    with torch.no_grad():
        lam = (l_mask / (l_clean + 1e-8)).detach()
    loss = l_mask + lam * l_clean
    return loss, l_mask, l_clean, lam


def test_loss_terms_finite():
    model = make_idlm(make_ar_model())
    # Use V-1 as toy MASK_ID (fits in toy vocab of size V=256); targets avoid it
    x_0 = torch.randint(0, V - 1, (B, N))
    loss, l_mask, l_clean, lam = compute_idlm_loss(model, x_0, torch.device("cpu"), mask_id=V - 1)
    assert torch.isfinite(loss)
    assert torch.isfinite(l_mask)
    assert torch.isfinite(l_clean)
    assert lam > 0


def test_lambda_has_no_grad():
    """λ must not appear in the computation graph."""
    model = make_idlm(make_ar_model())
    x_0 = torch.randint(0, V - 1, (B, N))
    loss, l_mask, l_clean, lam = compute_idlm_loss(model, x_0, torch.device("cpu"), mask_id=V - 1)
    assert not lam.requires_grad


def test_loss_backward_updates_lora_only():
    """After loss.backward(), only LoRA params should have gradients."""
    model = make_idlm(make_ar_model())
    x_0 = torch.randint(0, V - 1, (B, N))
    loss, *_ = compute_idlm_loss(model, x_0, torch.device("cpu"), mask_id=V - 1)
    loss.backward()
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.grad is not None, f"{name} missing grad after loss.backward()"
        else:
            assert param.grad is None, f"{name} should have no grad"
