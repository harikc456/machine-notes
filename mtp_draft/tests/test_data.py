import torch
from pathlib import Path
import pytest

from mtp_draft.config import MTPConfig
from mtp_draft.data import build_prompt, FeatureDataset

VOCAB, D_TEACHER = 50, 16


@pytest.fixture
def cfg():
    return MTPConfig(
        d_teacher=D_TEACHER,
        max_prompt_len=32,
        max_draft=4,
        teacher_layers=[0, 1, 2, 3],
        cache_n_answer_positions=2,
    )


def _fake_tokenizer(text: str) -> list[int]:
    """Trivial whitespace tokenizer for testing (returns word indices mod VOCAB)."""
    return [hash(w) % VOCAB for w in text.split()]


class _MockTokenizer:
    def encode(self, text, add_special_tokens=True):
        return _fake_tokenizer(text)


def _make_example():
    return {
        "question": "Where was Einstein born?",
        "context": {
            "title": ["Einstein", "Physics"],
            "sentences": [
                ["Albert Einstein was born in Ulm."],
                ["He developed the theory of relativity."],
            ],
        },
        "answer": "Ulm",
    }


def test_build_prompt_returns_nonempty():
    tok = _MockTokenizer()
    prompt_ids, answer_ids = build_prompt(_make_example(), tok, max_prompt_len=64)
    assert len(prompt_ids) > 0
    assert len(answer_ids) > 0


def test_build_prompt_truncates_to_max(cfg):
    tok = _MockTokenizer()
    prompt_ids, _ = build_prompt(_make_example(), tok, max_prompt_len=cfg.max_prompt_len)
    assert len(prompt_ids) <= cfg.max_prompt_len


def test_build_prompt_truncates_paragraphs():
    """Truncation fires when paragraphs exceed the budget."""
    tok = _MockTokenizer()
    # Make an example with 20 identical long paragraphs
    long_para = " ".join([f"word{i}" for i in range(50)])
    example = {
        "question": "What?",
        "context": {
            "title": [f"Title{i}" for i in range(20)],
            "sentences": [[long_para] for _ in range(20)],
        },
        "answer": "yes",
    }
    prompt_ids, _ = build_prompt(example, tok, max_prompt_len=64)
    assert len(prompt_ids) <= 64
    # At most 2-3 paragraphs can fit in 64 tokens; 20 paragraphs means truncation fired
    assert len(prompt_ids) < 64  # paragraphs were cut off


def _make_shard(cfg, tmp_path: Path) -> Path:
    n_layers = len(cfg.teacher_layers)
    shard = []
    for _ in range(3):
        features = torch.randn(cfg.cache_n_answer_positions, n_layers, cfg.d_teacher)
        scale = features.abs().max() / 127.0
        features_int8 = (features / scale).clamp(-127, 127).to(torch.int8)
        shard.append(
            {
                "features_int8": features_int8,
                "scale": scale,
                "prompt_ids": torch.randint(0, VOCAB, (20,)),
                "answer_ids": torch.randint(0, VOCAB, (8,)),
                "handoff": torch.tensor(15, dtype=torch.long),
            }
        )
    path = tmp_path / "train_shard_0000.pt"
    torch.save(shard, path)
    return path


def test_feature_dataset_length(cfg, tmp_path):
    path = _make_shard(cfg, tmp_path)
    ds = FeatureDataset([path], cfg)
    # 3 examples × cache_n_answer_positions=2 anchor positions each
    assert len(ds) == 3 * cfg.cache_n_answer_positions


def test_feature_dataset_item_shapes(cfg, tmp_path):
    path = _make_shard(cfg, tmp_path)
    ds = FeatureDataset([path], cfg)
    hiddens, ctx_ids, targets, valid_len = ds[0]
    assert hiddens.shape == (len(cfg.teacher_layers), cfg.d_teacher)
    assert ctx_ids.shape == (cfg.max_prompt_len,)
    assert targets.shape == (cfg.max_draft,)


def test_feature_dataset_dequantizes(cfg, tmp_path):
    path = _make_shard(cfg, tmp_path)
    ds = FeatureDataset([path], cfg)
    hiddens, _, _, _ = ds[0]
    assert hiddens.dtype == torch.bfloat16
