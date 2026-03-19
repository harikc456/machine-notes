import pytest
import torch
from kromhc_transformer.models.model import CausalLM
from kromhc_transformer.config import KromHCConfig

def test_causal_lm_kromhc_shape():
    cfg = KromHCConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=1000,
                       seq_len=32, ffn_hidden=128, model_type="kromhc", dropout=0.0)
    model = CausalLM(cfg)
    tokens = torch.randint(0, 1000, (2, 32))
    logits = model(tokens)
    assert logits.shape == (2, 32, 1000)

def test_causal_lm_baseline_shape():
    cfg = KromHCConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=1000,
                       seq_len=32, ffn_hidden=128, model_type="baseline", dropout=0.0)
    model = CausalLM(cfg)
    tokens = torch.randint(0, 1000, (2, 32))
    logits = model(tokens)
    assert logits.shape == (2, 32, 1000)

def test_causal_lm_weight_tying():
    cfg = KromHCConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=1000, model_type="kromhc")
    model = CausalLM(cfg)
    assert model.lm_head.weight is model.token_embedding.weight

def test_causal_lm_unknown_model_type_raises():
    cfg = KromHCConfig(model_type="unknown")
    with pytest.raises(KeyError):
        CausalLM(cfg)

def test_causal_lm_gradient_flow():
    cfg = KromHCConfig(d_model=64, n_heads=4, n_layers=2, vocab_size=100,
                       seq_len=16, ffn_hidden=128, model_type="kromhc", dropout=0.0)
    model = CausalLM(cfg)
    tokens = torch.randint(0, 100, (2, 16))
    logits = model(tokens)
    logits.sum().backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for {name}"
