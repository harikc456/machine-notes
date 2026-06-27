from __future__ import annotations
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chat.kv_quant_utils import ChatStats, ModelEntry, get_stats, load_models_yaml


def test_load_models_yaml_parses_entries(tmp_path):
    yaml_text = textwrap.dedent("""
    models:
      - id: Qwen/Qwen3-0.6B-Instruct
        label: "Qwen3 0.6B"
        head_dim: 128
        default_bits: 4
      - id: google/gemma-4-2b-it
        label: "Gemma4 2B"
        head_dim: 128
        default_bits: 4
    """)
    f = tmp_path / "models.yaml"
    f.write_text(yaml_text)
    entries = load_models_yaml(f)
    assert len(entries) == 2
    assert isinstance(entries[0], ModelEntry)
    assert entries[0].id == "Qwen/Qwen3-0.6B-Instruct"
    assert entries[0].label == "Qwen3 0.6B"
    assert entries[0].head_dim == 128
    assert entries[0].default_bits == 4
    assert entries[1].id == "google/gemma-4-2b-it"


def test_load_models_yaml_default_path():
    # Default path resolves to chat/models.yaml in the repo
    entries = load_models_yaml()
    assert len(entries) == 6
    ids = [e.id for e in entries]
    assert "Qwen/Qwen3-0.6B-Instruct" in ids
    assert "google/gemma-4-2b-it" in ids


def test_get_stats_baseline_mode():
    # baseline: n_layers=28, n_kv_heads=8, seq_len=512, head_dim=128
    # baseline_bytes = 28 * 8 * 512 * 128 * 2 * 2 = 58_720_256
    stats = get_stats(
        cache=None,
        n_new_tokens=50,
        elapsed=1.0,
        n_layers=28,
        n_kv_heads=8,
        seq_len=512,
        head_dim=128,
    )
    assert isinstance(stats, ChatStats)
    assert stats.tokens_per_sec == pytest.approx(50.0)
    assert stats.compression_ratio == pytest.approx(1.0)
    assert stats.kv_memory_mb == pytest.approx(stats.baseline_memory_mb)
    assert stats.baseline_memory_mb == pytest.approx(58_720_256 / 1e6, rel=1e-4)


def test_get_stats_quantized_mode():
    mock_cache = MagicMock()
    mock_cache.compressed_bytes.return_value = 5_000_000  # 5 MB
    # baseline_bytes = 28 * 8 * 512 * 128 * 2 * 2 = 58_720_256
    stats = get_stats(
        cache=mock_cache,
        n_new_tokens=100,
        elapsed=2.0,
        n_layers=28,
        n_kv_heads=8,
        seq_len=512,
        head_dim=128,
    )
    assert stats.tokens_per_sec == pytest.approx(50.0)
    assert stats.kv_memory_mb == pytest.approx(5.0)
    assert stats.baseline_memory_mb == pytest.approx(58_720_256 / 1e6, rel=1e-4)
    assert stats.compression_ratio == pytest.approx(58_720_256 / 5_000_000, rel=1e-3)


def test_get_stats_zero_elapsed_does_not_divide_by_zero():
    stats = get_stats(
        cache=None,
        n_new_tokens=10,
        elapsed=0.0,
        n_layers=2,
        n_kv_heads=4,
        seq_len=64,
        head_dim=64,
    )
    assert stats.tokens_per_sec >= 0
    assert not (stats.tokens_per_sec != stats.tokens_per_sec)  # not NaN
