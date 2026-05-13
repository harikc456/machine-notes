import json
import pytest
from pathlib import Path
from rbf_ffn.generate_leaderboard_html import generate_html
from rbf_ffn._leaderboard_data import load_experiment


@pytest.fixture
def sample_exps(tmp_path):
    exps = []
    for i, (attn, ffn, ppl) in enumerate([
        ("xsa", "swiglu", 35.0),
        ("std", "swiglu", 41.0),
        ("xsa", "moe",    47.0),
    ]):
        d = tmp_path / f"2026010{i}_120000_abc{i}_exp{i}"
        d.mkdir()
        (d / "metrics.jsonl").write_text(
            f'{{"epoch": 0, "train_ppl": {ppl+10:.1f}, "val_ppl": {ppl:.1f}, "epoch_time_s": 3600.0}}\n'
        )
        (d / "config.yaml").write_text(
            f"attn_type: {attn}\nffn_type: {ffn}\nqk_norm: true\nd_model: 256\n"
        )
        (d / "params.json").write_text('{"n_params": 30000000}')
        exps.append(load_experiment(d))
    return exps


def test_generate_html_returns_string(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    assert isinstance(html, str)


def test_generate_html_has_doctype(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    assert html.strip().startswith("<!DOCTYPE html>")


def test_generate_html_contains_all_column_headers(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    for header in ["attn", "ffn", "qk", "wn", "orth_layers", "MoE", "params",
                   "ep", "best_ppl", "@ep", "final_ppl", "trn_ppl", "hrs"]:
        assert header in html, f"Missing column header: {header}"


def test_generate_html_embeds_config_text(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    assert "data-config" in html
    assert "attn_type" in html


def test_generate_html_contains_sort_js(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    assert "sortTable" in html


def test_generate_html_contains_filter_input(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    assert 'id="filter-input"' in html


def test_generate_html_contains_generated_timestamp(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    assert "2026-05-13 12:00:00" in html


def test_generate_html_rank1_gold_class(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    assert "rank-1" in html


def test_generate_html_no_external_urls(sample_exps):
    html = generate_html(sample_exps, generated_at="2026-05-13 12:00:00")
    assert "cdn." not in html
    assert "fonts.googleapis" not in html
    assert "https://" not in html
