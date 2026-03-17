# rbf_ffn/tests/test_model.py
import torch
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.model import CausalLM

B, N = 2, 16
VOCAB = 256    # small for fast tests
D, H, L = 32, 4, 2


def make_model(model_type: str = "rbf", gate_variant: str = "G0") -> CausalLM:
    cfg = RBFFFNConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        model_type=model_type, gate_variant=gate_variant,
        ffn_hidden=86,   # 8/3 * 32 ≈ 85
        dropout=0.0,
    )
    return CausalLM(cfg)


def test_baseline_output_shape():
    model = make_model("baseline")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits = model(tokens)
    assert logits.shape == (B, N, VOCAB)


@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_rbf_output_shape(variant):
    tokens = torch.randint(0, VOCAB, (B, N))
    assert make_model("rbf", variant)(tokens).shape == (B, N, VOCAB)


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


def test_all_2d_non_embedding_non_sigma_in_muon():
    """Every 2D param that is not the embedding and not sigma_raw must be in Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = make_model("rbf", "G0")
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    emb_id = id(model.token_embedding.weight)
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if "sigma_raw" in name or id(param) == emb_id:
            assert id(param) in adamw_ids, f"{name} should be AdamW"
        elif param.ndim == 2:
            assert id(param) in muon_ids, f"{name} (2D) should be Muon"


def test_gradient_flows_baseline():
    model = make_model("baseline")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits = model(tokens)
    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None


def test_gradient_flows_rbf():
    model = make_model("rbf")
    tokens = torch.randint(0, VOCAB, (B, N))
    model(tokens).sum().backward()
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
    assert model(tokens).shape == (B, N, VOCAB)


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
    assert model(tokens).shape == (B, N, VOCAB)


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
