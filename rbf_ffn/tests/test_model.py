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


def _make_untied_model() -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        model_type="baseline",
        ffn_hidden=86,
        dropout=0.0,
        tie_embeddings=False,
    )
    return CausalLM(cfg)


def _make_kronecker_lm_head_model() -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        model_type="baseline",
        ffn_hidden=86,
        dropout=0.0,
        tie_embeddings=False,
        lm_head_kronecker=True,
    )
    return CausalLM(cfg)


def test_untied_embeddings_lm_head_is_independent():
    model = _make_untied_model()
    assert model.lm_head.weight is not model.token_embedding.weight


def test_untied_embeddings_output_shape():
    model = _make_untied_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)


def test_untied_embeddings_lm_head_in_muon():
    """Untied lm_head.weight is 2-D and not the embedding, so it must go to Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_untied_model()
    muon_params, _ = build_optimizer_groups(model)
    muon_ids = {id(p) for p in muon_params}
    assert id(model.lm_head.weight) in muon_ids


def test_kronecker_lm_head_output_shape():
    model = _make_kronecker_lm_head_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)


def test_kronecker_lm_head_gradient_flows():
    model = _make_kronecker_lm_head_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    logits.sum().backward()
    assert model.lm_head.A.grad is not None
    assert model.lm_head.B.grad is not None


def test_kronecker_lm_head_factors_in_muon():
    """KroneckerLMHead A and B are 2-D and must be routed to Muon."""
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_kronecker_lm_head_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    assert id(model.lm_head.A) in muon_ids,     "lm_head.A should be in Muon"
    assert id(model.lm_head.A) not in adamw_ids, "lm_head.A should not be in AdamW"
    assert id(model.lm_head.B) in muon_ids,     "lm_head.B should be in Muon"
    assert id(model.lm_head.B) not in adamw_ids, "lm_head.B should not be in AdamW"


def test_no_duplicate_params_untied():
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_untied_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids)), "Duplicate parameters in optimizer groups"


def test_no_duplicate_params_kronecker_lm_head():
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_kronecker_lm_head_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids)), "Duplicate parameters in optimizer groups"


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


def _make_looped_model(n_repeats: int = 4, n_fixed: int = 2) -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H,
        vocab_size=VOCAB, seq_len=N,
        attn_type="standard", ffn_type="swiglu",
        ffn_hidden=86,
        dropout=0.0,
        use_loop=True,
        loop_n_repeats=n_repeats,
        loop_n_fixed=n_fixed,
    )
    return CausalLM(cfg)


def test_looped_output_shape():
    model = _make_looped_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, hs = model(tokens)
    assert logits.shape == (B, N, VOCAB)
    assert hs == []


def test_looped_block_count():
    """blocks list = loop_n_fixed head + 1 LoopBlock + loop_n_fixed tail."""
    from rbf_ffn.models.transformer_block import LoopBlock
    model = _make_looped_model(n_repeats=4, n_fixed=2)
    assert len(model.blocks) == 2 + 1 + 2  # 5 entries
    loop_blocks = [b for b in model.blocks if isinstance(b, LoopBlock)]
    assert len(loop_blocks) == 1
    assert loop_blocks[0].n_repeats == 4


def test_looped_layer_enc_shape():
    """layer_enc buffer must have shape (n_repeats, d_model)."""
    from rbf_ffn.models.transformer_block import LoopBlock
    model = _make_looped_model()
    loop_block = next(b for b in model.blocks if isinstance(b, LoopBlock))
    assert loop_block.layer_enc.shape == (4, D)


def test_looped_layer_enc_count():
    """layer_enc must have exactly loop_n_repeats rows."""
    from rbf_ffn.models.transformer_block import LoopBlock
    model = _make_looped_model(n_repeats=6)
    loop_block = next(b for b in model.blocks if isinstance(b, LoopBlock))
    assert loop_block.layer_enc.shape[0] == 6


def test_looped_layer_enc_is_not_parameter():
    """layer_enc must be a buffer, not a learnable parameter."""
    from rbf_ffn.models.transformer_block import LoopBlock
    model = _make_looped_model()
    loop_block = next(b for b in model.blocks if isinstance(b, LoopBlock))
    param_names = {n for n, _ in loop_block.named_parameters()}
    assert "layer_enc" not in param_names
    buffer_names = {n for n, _ in loop_block.named_buffers()}
    assert "layer_enc" in buffer_names


def test_looped_layer_enc_absolute_positions():
    """layer_enc rows must encode absolute positions starting at loop_n_fixed."""
    import math
    from rbf_ffn.models.transformer_block import LoopBlock, _sinusoidal_layer_encoding
    model = _make_looped_model(n_fixed=2, n_repeats=4)
    loop_block = next(b for b in model.blocks if isinstance(b, LoopBlock))
    expected = _sinusoidal_layer_encoding(start=2, n=4, d_model=D)
    assert torch.allclose(loop_block.layer_enc.cpu(), expected)


def test_looped_gradient_flows():
    model = _make_looped_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None
    from rbf_ffn.models.transformer_block import LoopBlock
    loop_block = next(b for b in model.blocks if isinstance(b, LoopBlock))
    # layer_enc is a non-trainable buffer; check the shared block's params instead
    for p in loop_block.inner_block.parameters():
        assert p.grad is not None


def test_looped_layer_enc_differentiates_steps():
    """Each loop step must have a distinct encoding (sinusoidal positions are unique)."""
    from rbf_ffn.models.transformer_block import LoopBlock
    model = _make_looped_model(n_repeats=4)
    loop_block = next(b for b in model.blocks if isinstance(b, LoopBlock))
    enc = loop_block.layer_enc
    for i in range(enc.shape[0]):
        for j in range(i + 1, enc.shape[0]):
            assert not torch.allclose(enc[i], enc[j])


def test_looped_no_duplicate_params():
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_looped_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids))


def test_looped_and_kromhc_raises():
    with pytest.raises(ValueError, match="use_loop and use_kromhc"):
        ModelConfig(use_loop=True, use_kromhc=True)


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
