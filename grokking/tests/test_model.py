from __future__ import annotations
import torch
import pytest
from grokking.config import GrokConfig
from grokking.model import GrokTransformer, build_optimizer_groups


def _tiny_cfg(**kwargs) -> GrokConfig:
    return GrokConfig(p=7, d_model=32, n_heads=2, n_layers=1, **kwargs)


# ── Forward pass ──────────────────────────────────────────────────────────────

def test_forward_output_shape():
    cfg = _tiny_cfg()
    model = GrokTransformer(cfg)
    tokens = torch.zeros(4, 4, dtype=torch.long)   # batch=4, seq=4
    out = model(tokens)
    assert out.shape == (4, 7)    # (B, p)

def test_forward_output_is_float():
    cfg = _tiny_cfg()
    model = GrokTransformer(cfg)
    tokens = torch.zeros(2, 4, dtype=torch.long)
    out = model(tokens)
    assert out.dtype == torch.float32

def test_forward_accepts_valid_token_ids():
    cfg = _tiny_cfg()                         # p=7, vocab=9 (7 + op + eq)
    model = GrokTransformer(cfg)
    tokens = torch.tensor([[0, 7, 6, 8]])     # a=0, op=7, b=6, eq=8
    out = model(tokens)
    assert out.shape == (1, 7)


# ── Optimizer groups ──────────────────────────────────────────────────────────

def test_muon_params_are_2d():
    cfg = _tiny_cfg()
    model = GrokTransformer(cfg)
    muon, _ = build_optimizer_groups(model)
    assert len(muon) > 0
    for p in muon:
        assert p.ndim == 2

def test_no_embeddings_in_muon():
    cfg = _tiny_cfg()
    model = GrokTransformer(cfg)
    emb_ids = {id(model.token_embedding.weight), id(model.pos_embedding.weight)}
    muon, _ = build_optimizer_groups(model)
    for p in muon:
        assert id(p) not in emb_ids

def test_optimizer_groups_no_overlap():
    cfg = _tiny_cfg()
    model = GrokTransformer(cfg)
    muon, adamw = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    assert len(muon_ids & adamw_ids) == 0

def test_optimizer_groups_cover_all_params():
    cfg = _tiny_cfg()
    model = GrokTransformer(cfg)
    muon, adamw = build_optimizer_groups(model)
    all_ids   = {id(p) for p in model.parameters()}
    group_ids = {id(p) for p in muon} | {id(p) for p in adamw}
    assert all_ids == group_ids
