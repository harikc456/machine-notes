"""
Tests for sigreg/data.py.

All tests are offline — no network calls, no disk I/O to rbf_ffn/data_cache/.
_build_tokenizer and _load_split are mocked where needed.
"""
import torch
import pytest
from unittest.mock import MagicMock, patch


# ── _build_tokenizer ──────────────────────────────────────────────────────────

def test_build_tokenizer_loads_from_cache_when_files_exist(tmp_path):
    """If vocab.json and merges.txt exist, load without retraining."""
    tok_dir = tmp_path / "bpe65536_tokenizer"
    tok_dir.mkdir()
    (tok_dir / "vocab.json").write_text('{"a": 0}')
    (tok_dir / "merges.txt").write_text("#version: 0.2\n")

    mock_tok = MagicMock()
    with patch("sigreg.data.ByteLevelBPETokenizer", return_value=mock_tok) as MockTok:
        from sigreg.data import _build_tokenizer
        result = _build_tokenizer(tmp_path)

    MockTok.assert_called_once_with(
        str(tok_dir / "vocab.json"),
        str(tok_dir / "merges.txt"),
    )
    assert result is mock_tok


def test_build_tokenizer_trains_and_saves_when_no_cache(tmp_path):
    """If no cache files, train on WikiText-103 and save."""
    mock_tok = MagicMock()

    with patch("sigreg.data.ByteLevelBPETokenizer", return_value=mock_tok) as MockTok, \
         patch("sigreg.data._load_wikitext_split_texts", return_value=["hello world", "foo bar"]):
        from sigreg.data import _build_tokenizer
        result = _build_tokenizer(tmp_path)

    # Constructor called with no args (fresh tokenizer)
    MockTok.assert_called_once_with()
    mock_tok.train_from_iterator.assert_called_once_with(
        ["hello world", "foo bar"],
        vocab_size=65536,
        min_frequency=2,
        special_tokens=[],
    )
    tok_dir = tmp_path / "bpe65536_tokenizer"
    mock_tok.save_model.assert_called_once_with(str(tok_dir))
    assert result is mock_tok


# ── _load_split ───────────────────────────────────────────────────────────────

def test_load_split_uses_bpe65536_cache_filename(tmp_path):
    """Cache file must be named {split}_bpe65536_{seq_len}.pt."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value.ids = list(range(100))

    with patch("sigreg.data._CACHE_DIR", tmp_path), \
         patch("sigreg.data._load_wikitext_split_texts", return_value=["hello world"]):
        from sigreg.data import _load_split
        _load_split("train", seq_len=10, tokenizer=mock_tok)

    assert (tmp_path / "train_bpe65536_10.pt").exists()


def test_load_split_loads_from_cache_when_pt_exists(tmp_path):
    """If the .pt file exists, return it without re-tokenising."""
    cached = torch.zeros(5, 10, dtype=torch.long)
    cache_file = tmp_path / "validation_bpe65536_10.pt"
    torch.save(cached, cache_file)

    mock_tok = MagicMock()
    with patch("sigreg.data._CACHE_DIR", tmp_path):
        from sigreg.data import _load_split
        result = _load_split("validation", seq_len=10, tokenizer=mock_tok)

    mock_tok.encode.assert_not_called()
    assert result.shape == (5, 10)


def test_load_split_returns_long_tensor(tmp_path):
    """Returned tensor must be dtype=torch.long."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value.ids = list(range(50))

    with patch("sigreg.data._CACHE_DIR", tmp_path), \
         patch("sigreg.data._load_wikitext_split_texts", return_value=["text"]):
        from sigreg.data import _load_split
        result = _load_split("test", seq_len=10, tokenizer=mock_tok)

    assert result.dtype == torch.long


# ── get_dataloaders ───────────────────────────────────────────────────────────

def _fake_load_split(split, seq_len, tokenizer):
    return torch.zeros(32, seq_len, dtype=torch.long)


class _Cfg:
    seq_len = 16
    batch_size = 4
    seed = 42


def test_get_dataloaders_returns_three_loaders():
    mock_tok = MagicMock()
    with patch("sigreg.data._build_tokenizer", return_value=mock_tok), \
         patch("sigreg.data._load_split", side_effect=_fake_load_split):
        from sigreg.data import get_dataloaders
        result = get_dataloaders(_Cfg())
    assert len(result) == 3


def test_train_loader_has_persistent_workers_and_prefetch():
    mock_tok = MagicMock()
    with patch("sigreg.data._build_tokenizer", return_value=mock_tok), \
         patch("sigreg.data._load_split", side_effect=_fake_load_split):
        from sigreg.data import get_dataloaders
        train_loader, _, _ = get_dataloaders(_Cfg())
    assert train_loader.persistent_workers is True
    assert train_loader.prefetch_factor == 2


def test_val_and_test_loaders_have_no_persistent_workers():
    mock_tok = MagicMock()
    with patch("sigreg.data._build_tokenizer", return_value=mock_tok), \
         patch("sigreg.data._load_split", side_effect=_fake_load_split):
        from sigreg.data import get_dataloaders
        _, val_loader, test_loader = get_dataloaders(_Cfg())
    assert val_loader.persistent_workers is False
    assert test_loader.persistent_workers is False
