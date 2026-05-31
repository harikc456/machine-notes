# idlm/tests/test_isd.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.model import CausalLM
from idlm.models.idlm_model import IDLMCausalLM
from idlm.config import IDLMConfig
from idlm.generate import isd_acceptance_rate, isd_generate, compute_tpf_oh

B, N, V, D, H, L_layers = 2, 16, 256, 32, 4, 2
MASK_ID = 50256
DEVICE = torch.device("cpu")


def make_idlm() -> IDLMCausalLM:
    cfg = ModelConfig(d_model=D, n_heads=H, n_layers=L_layers,
                      vocab_size=V, seq_len=N, ffn_hidden=86, dropout=0.0)
    ar = CausalLM(cfg)
    return IDLMCausalLM(ar, lora_rank=4, lora_alpha=8.0,
                        lora_target_modules=["q_proj", "v_proj"])


def make_eval_cfg() -> IDLMConfig:
    return IDLMConfig(
        ar_checkpoint="dummy.pt",
        stride=2,
        prompt_len=4,
        gen_len=8,
        num_eval_examples=2,
        vocab_size=V,
    )


def test_isd_generate_output_length():
    """Generated sequence has prompt_len + gen_len tokens."""
    model = make_idlm()
    cfg = make_eval_cfg()
    prompt = list(range(cfg.prompt_len))
    tokens = isd_generate(model, prompt, cfg, DEVICE)
    assert len(tokens) == cfg.prompt_len + cfg.gen_len


def test_alpha_in_bounds():
    """Acceptance rate alpha must be in [0, 1]."""
    model = make_idlm()
    cfg = make_eval_cfg()
    seq = list(range(cfg.prompt_len + cfg.gen_len))
    alpha = isd_acceptance_rate(model, seq, cfg, DEVICE)
    assert 0.0 <= alpha <= 1.0


def test_tpf_oh_positive():
    alpha = 0.85
    stride = 4
    tpf = compute_tpf_oh(alpha, stride)
    assert tpf > 0


def test_tpf_oh_high_alpha():
    """At alpha=1.0 (perfect acceptance), TPF/OH = stride."""
    tpf = compute_tpf_oh(1.0, stride=4)
    assert abs(tpf - 4.0) < 1e-6


def test_tpf_oh_low_alpha():
    """At alpha=0.0 (all rejected), efficiency is very low."""
    tpf = compute_tpf_oh(0.0, stride=4)
    assert tpf < 1.0
