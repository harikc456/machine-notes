# rbf_ffn/tests/test_model.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.model import CausalLM

B, N = 2, 16
VOCAB = 256    # small for fast tests
D, H, L = 32, 4, 2


def make_model(model_type: str = "baseline") -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        model_type=model_type,
        ffn_hidden=86,   # 8/3 * 32 ≈ 85
        pfd_n=4,
        dropout=0.0,
    )
    return CausalLM(cfg)


def test_baseline_output_shape():
    model = make_model("baseline")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, hs = model(tokens)
    assert logits.shape == (B, N, VOCAB)
    assert hs == []


def test_weight_tying():
    """LM head weight must be the same tensor object as the embedding weight."""
    model = make_model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_weight_tying_shared_memory():
    """A write to embedding weight must be reflected in lm_head weight."""
    model = make_model()
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = 999.0
    assert model.lm_head.weight[0, 0].item() == 999.0


def test_no_duplicate_params_in_optimizer_groups():
    """The tied embedding/lm_head weight must appear exactly once across groups."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids)), "Duplicate parameters in optimizer groups"


def test_embedding_in_adamw_not_muon():
    """Token embedding weight must be in AdamW group (not Muon)."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model()
    muon_params, _ = build_optimizer_groups(model)
    emb_id = id(model.token_embedding.weight)
    assert emb_id not in {id(p) for p in muon_params}


def test_gradient_flows_baseline():
    model = make_model("baseline")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None


def test_rational_params_in_adamw():
    """RationalActivation a and b must be in AdamW (1-D), not Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("rational")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.act.a) in adamw_ids,    "act.a should be in AdamW"
        assert id(block.ffn.act.a) not in muon_ids, "act.a should not be in Muon"
        assert id(block.ffn.act.b) in adamw_ids,    "act.b should be in AdamW"
        assert id(block.ffn.act.b) not in muon_ids, "act.b should not be in Muon"


def test_rationalglu_output_shape():
    model = make_model("rationalglu")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)


def test_rationalglu_params_in_adamw():
    """RationalGatedFFN act.a and act.b must be in AdamW (1-D), not Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("rationalglu")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.act.a) in adamw_ids,    "act.a should be in AdamW"
        assert id(block.ffn.act.a) not in muon_ids, "act.a should not be in Muon"
        assert id(block.ffn.act.b) in adamw_ids,    "act.b should be in AdamW"
        assert id(block.ffn.act.b) not in muon_ids, "act.b should not be in Muon"


def test_first_order_pfd_rational_output_shape():
    model = make_model("first_order_pfd_rational")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)


def test_first_order_pfd_rational_params_in_adamw():
    """phi and PFD activation params must be in AdamW, not Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("first_order_pfd_rational")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.phi)       in adamw_ids,    "phi should be in AdamW"
        assert id(block.ffn.phi)       not in muon_ids, "phi should not be in Muon"
        assert id(block.ffn.act.a)     in adamw_ids,    "act.a should be in AdamW"
        assert id(block.ffn.act.a)     not in muon_ids, "act.a should not be in Muon"
        assert id(block.ffn.act.b)     in adamw_ids,    "act.b should be in AdamW"
        assert id(block.ffn.act.b)     not in muon_ids, "act.b should not be in Muon"
        assert id(block.ffn.act.c)     in adamw_ids,    "act.c should be in AdamW"
        assert id(block.ffn.act.c)     not in muon_ids, "act.c should not be in Muon"
        assert id(block.ffn.act.gamma) in adamw_ids,    "act.gamma should be in AdamW"
        assert id(block.ffn.act.gamma) not in muon_ids, "act.gamma should not be in Muon"


def test_polar_mlp_output_shape():
    model = make_model("polar_mlp")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)


def test_polar_mlp_gradient_flows():
    """Smoke test: backward pass through full model, loss is finite, grads propagate."""
    model = make_model("polar_mlp")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    loss = logits.sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_polar_mlp_thresholds_in_adamw():
    """thresholds (1-D) must be in AdamW, not Muon; keys and down_proj.weight (2-D) must be in Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("polar_mlp")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.thresholds)        in adamw_ids,    "thresholds should be in AdamW"
        assert id(block.ffn.thresholds)        not in muon_ids, "thresholds should not be in Muon"
        assert id(block.ffn.keys)              in muon_ids,     "keys should be in Muon"
        assert id(block.ffn.keys)              not in adamw_ids,"keys should not be in AdamW"
        assert id(block.ffn.down_proj.weight)  in muon_ids,     "down_proj.weight should be in Muon"
        assert id(block.ffn.down_proj.weight)  not in adamw_ids,"down_proj.weight should not be in AdamW"


def _make_delta_model() -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        model_type="baseline",
        ffn_hidden=86,
        dropout=0.0,
        kronecker_delta_mlp=True,
        kronecker_delta_rank=4,
    )
    return CausalLM(cfg)


def test_kronecker_delta_output_shape():
    model = _make_delta_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)


def test_delta_params_in_adamw_not_muon():
    """delta_C and delta_D must be routed to AdamW, not Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_delta_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.up_proj.delta_C)   in adamw_ids,    "up_proj.delta_C should be in AdamW"
        assert id(block.ffn.up_proj.delta_C)   not in muon_ids, "up_proj.delta_C should not be in Muon"
        assert id(block.ffn.up_proj.delta_D)   in adamw_ids,    "up_proj.delta_D should be in AdamW"
        assert id(block.ffn.up_proj.delta_D)   not in muon_ids, "up_proj.delta_D should not be in Muon"
        assert id(block.ffn.down_proj.delta_C) in adamw_ids,    "down_proj.delta_C should be in AdamW"
        assert id(block.ffn.down_proj.delta_C) not in muon_ids, "down_proj.delta_C should not be in Muon"
        assert id(block.ffn.down_proj.delta_D) in adamw_ids,    "down_proj.delta_D should be in AdamW"
        assert id(block.ffn.down_proj.delta_D) not in muon_ids, "down_proj.delta_D should not be in Muon"


def test_kronecker_core_params_in_muon():
    """A and B of KroneckerDeltaLinear must still go to Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_delta_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.ffn.up_proj.A)   in muon_ids,     "up_proj.A should be in Muon"
        assert id(block.ffn.up_proj.A)   not in adamw_ids,"up_proj.A should not be in AdamW"
        assert id(block.ffn.down_proj.B) in muon_ids,     "down_proj.B should be in Muon"
        assert id(block.ffn.down_proj.B) not in adamw_ids,"down_proj.B should not be in AdamW"


def test_no_duplicate_params_delta_model():
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_delta_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids)), "Duplicate parameters in optimizer groups"


def _make_kromhc_model() -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        model_type="baseline",
        ffn_hidden=86,
        dropout=0.0,
        use_kromhc=True,
    )
    return CausalLM(cfg)


def test_kromhc_output_shape():
    model = _make_kromhc_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, hs = model(tokens)
    assert logits.shape == (B, N, VOCAB)
    assert len(hs) == L


def test_kromhc_H_shape():
    model = _make_kromhc_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    _, hs = model(tokens)
    for H_mat in hs:
        assert H_mat.shape == (B, N, H, H)  # H=4 heads


def test_kromhc_no_duplicate_params():
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_kromhc_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids))


def test_kromhc_gradient_flows():
    """Gradients must reach mixer_proj and weight_gens despite H being detached."""
    model = _make_kromhc_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    logits.sum().backward()
    for block in model.blocks:
        assert block.mixer_proj.weight.grad is not None
        for gen in block.head_mixer.weight_gens:
            for p in gen.parameters():
                assert p.grad is not None
