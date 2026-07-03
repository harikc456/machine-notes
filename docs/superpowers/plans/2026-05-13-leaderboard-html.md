# Leaderboard HTML Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a self-contained interactive HTML leaderboard from experiment artifacts.

**Architecture:** Extract shared data loading from `leaderboard.py` into `_leaderboard_data.py`; a new `generate_leaderboard_html.py` imports the same module and writes a single-file HTML page with embedded CSS + vanilla JS for sort, filter, gold/silver/bronze highlights, and per-row config expansion.

**Tech Stack:** Python 3.10+, PyYAML (already in use), no new runtime deps; pytest for tests.

---

### Task 1: Extract shared data loading into `_leaderboard_data.py`

**Files:**
- Create: `rbf_ffn/_leaderboard_data.py`
- Create: `rbf_ffn/tests/test_leaderboard_data.py`

- [ ] **Step 1: Write the failing tests**

Create `rbf_ffn/tests/test_leaderboard_data.py`:

```python
import json
import pytest
from pathlib import Path
from rbf_ffn._leaderboard_data import load_experiment, load_all_experiments, fmt, fmt_params


@pytest.fixture
def exp_dir(tmp_path):
    d = tmp_path / "20260101_120000_abc123_xsa_swiglu_qknorm_wnorm_d256"
    d.mkdir()
    (d / "metrics.jsonl").write_text(
        '{"epoch": 0, "train_loss": 5.0, "train_ppl": 148.4, "val_loss": 4.5, "val_ppl": 90.0, "epoch_time_s": 3600.0, "effective_batch_size": 16}\n'
        '{"epoch": 1, "train_loss": 4.0, "train_ppl": 54.6, "val_loss": 3.8, "val_ppl": 44.7, "epoch_time_s": 3600.0, "effective_batch_size": 16}\n'
    )
    (d / "config.yaml").write_text(
        "attn_type: xsa\nffn_type: swiglu\nqk_norm: true\nlinear_weight_norm: true\n"
        "d_model: 256\nn_layers: 6\nffn_hidden: 688\n"
    )
    (d / "params.json").write_text('{"n_params": 30478464}')
    return d


def test_load_experiment_returns_dict(exp_dir):
    result = load_experiment(exp_dir)
    assert isinstance(result, dict)


def test_load_experiment_best_ppl(exp_dir):
    result = load_experiment(exp_dir)
    assert result["best_val_ppl"] == pytest.approx(44.7)
    assert result["best_epoch"] == 1


def test_load_experiment_epochs_done(exp_dir):
    result = load_experiment(exp_dir)
    assert result["epochs_done"] == 2


def test_load_experiment_total_time(exp_dir):
    result = load_experiment(exp_dir)
    assert result["total_time_h"] == pytest.approx(2.0)


def test_load_experiment_params(exp_dir):
    result = load_experiment(exp_dir)
    assert result["n_params"] == 30478464


def test_load_experiment_config_text(exp_dir):
    result = load_experiment(exp_dir)
    assert "xsa" in result["config_text"]


def test_load_experiment_missing_metrics_returns_none(tmp_path):
    d = tmp_path / "empty_exp"
    d.mkdir()
    (d / "config.yaml").write_text("attn_type: xsa\n")
    assert load_experiment(d) is None


def test_load_experiment_infers_attn_from_dirname(tmp_path):
    d = tmp_path / "20260101_120000_abc123_xsa_swiglu_d256"
    d.mkdir()
    (d / "metrics.jsonl").write_text(
        '{"epoch": 0, "train_ppl": 100.0, "val_ppl": 80.0, "epoch_time_s": 100.0}\n'
    )
    (d / "config.yaml").write_text("d_model: 256\n")
    result = load_experiment(d)
    assert result["attn_type"] == "xsa"


def test_load_experiment_infers_standard_from_dirname(tmp_path):
    d = tmp_path / "20260101_120000_abc123_standard_swiglu_d256"
    d.mkdir()
    (d / "metrics.jsonl").write_text(
        '{"epoch": 0, "train_ppl": 100.0, "val_ppl": 80.0, "epoch_time_s": 100.0}\n'
    )
    (d / "config.yaml").write_text("d_model: 256\n")
    result = load_experiment(d)
    assert result["attn_type"] == "std"


def test_load_all_experiments_skips_non_dirs(tmp_path):
    (tmp_path / ".gitkeep").touch()
    (tmp_path / "analysis.md").write_text("notes")
    results = load_all_experiments(tmp_path)
    assert results == []


def test_fmt_none():
    assert fmt(None) == "—"

def test_fmt_float():
    assert fmt(3.14159) == "3.14"

def test_fmt_bool_true():
    assert fmt(True) == "Y"

def test_fmt_bool_false():
    assert fmt(False) == "N"

def test_fmt_list():
    assert fmt([1, 3]) == "[1, 3]"

def test_fmt_params_millions():
    assert fmt_params(30_478_464) == "30.5M"

def test_fmt_params_billions():
    assert fmt_params(1_500_000_000) == "1.50B"

def test_fmt_params_none():
    assert fmt_params(None) == "—"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_leaderboard_data.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'rbf_ffn._leaderboard_data'`

- [ ] **Step 3: Create `rbf_ffn/_leaderboard_data.py`**

```python
#!/usr/bin/env python3
"""Shared data loading for leaderboard scripts."""

import ast
import json
from pathlib import Path

import yaml

CONFIG_FIELDS = [
    "attn_type",
    "ffn_type",
    "qk_norm",
    "linear_weight_norm",
    "orthogonal_ffn_layers",
    "moe_n_experts",
    "moe_top_k",
    "n_layers",
    "d_model",
]

_SKIP_DIRS = {"archive"}


def load_experiment(exp_dir: Path) -> dict | None:
    metrics_path = exp_dir / "metrics.jsonl"
    config_path = exp_dir / "config.yaml"

    if not metrics_path.exists() or not config_path.exists():
        return None

    rows = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append(ast.literal_eval(line))
    if not rows:
        return None

    config_text = config_path.read_text()
    config = yaml.safe_load(config_text) or {}

    name_parts = set(exp_dir.name.split("_"))
    if not config.get("attn_type"):
        if "xsa" in name_parts:
            config["attn_type"] = "xsa"
        elif "standard" in name_parts:
            config["attn_type"] = "std"
    if not config.get("ffn_type"):
        for candidate in ("swiglu", "moe", "rational", "rationalglu", "polar"):
            if candidate in name_parts:
                config["ffn_type"] = candidate
                break

    best_row = min(rows, key=lambda r: r.get("val_ppl", float("inf")))
    final_row = rows[-1]
    total_time_h = sum(r.get("epoch_time_s", 0) for r in rows) / 3600

    params_path = exp_dir / "params.json"
    n_params = None
    if params_path.exists():
        n_params = json.loads(params_path.read_text()).get("n_params")

    return {
        "name": exp_dir.name,
        "config_text": config_text,
        "epochs_done": len(rows),
        "best_val_ppl": best_row.get("val_ppl"),
        "best_epoch": best_row.get("epoch"),
        "final_val_ppl": final_row.get("val_ppl"),
        "final_train_ppl": final_row.get("train_ppl"),
        "total_time_h": total_time_h,
        "n_params": n_params,
        **{f: config.get(f) for f in CONFIG_FIELDS},
    }


def load_all_experiments(experiments_dir: Path, min_epochs: int = 1) -> list[dict]:
    results = []
    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in _SKIP_DIRS:
            continue
        data = load_experiment(d)
        if data and data["epochs_done"] >= min_epochs:
            results.append(data)
    return results


def fmt(val, digits: int = 2) -> str:
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Y" if val else "N"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    if isinstance(val, list):
        return str(val)
    return str(val)


def fmt_params(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_leaderboard_data.py -v
```

Expected: all 20 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/_leaderboard_data.py rbf_ffn/tests/test_leaderboard_data.py
git commit -m "feat: extract shared leaderboard data loading into _leaderboard_data.py"
```

---

### Task 2: Update `leaderboard.py` to import from `_leaderboard_data`

**Files:**
- Modify: `rbf_ffn/leaderboard.py`

- [ ] **Step 1: Replace the data loading section in `leaderboard.py`**

Replace the contents of `rbf_ffn/leaderboard.py` with:

```python
#!/usr/bin/env python3
"""Model leaderboard from experiment artifacts in rbf_ffn/experiments/."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from rbf_ffn._leaderboard_data import load_all_experiments, fmt, fmt_params

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def main():
    parser = argparse.ArgumentParser(description="Leaderboard of rbf_ffn experiments")
    parser.add_argument("--sort", default="best_val_ppl", choices=["best_val_ppl", "final_val_ppl", "name"])
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--min-epochs", type=int, default=1)
    args = parser.parse_args()

    exps = load_all_experiments(EXPERIMENTS_DIR, min_epochs=args.min_epochs)

    if args.filter:
        exps = [e for e in exps if args.filter.lower() in e["name"].lower()]

    if args.sort in ("best_val_ppl", "final_val_ppl"):
        exps.sort(key=lambda e: e[args.sort] or float("inf"))
    else:
        exps.sort(key=lambda e: e[args.sort] or "")

    if args.top:
        exps = exps[: args.top]

    console = Console(width=200)
    table = Table(
        title=f"[bold]rbf_ffn Leaderboard[/bold]  ({len(exps)} experiments, sorted by {args.sort})",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        highlight=True,
    )

    table.add_column("#", style="dim", justify="right", width=3)
    table.add_column("Experiment", style="cyan", no_wrap=True, max_width=45)
    table.add_column("attn", justify="center", width=9)
    table.add_column("ffn", justify="center", width=12)
    table.add_column("qk", justify="center", width=3)
    table.add_column("wn", justify="center", width=3)
    table.add_column("orth_layers", justify="center", width=14)
    table.add_column("MoE", justify="center", width=6)
    table.add_column("params", justify="right", width=8)
    table.add_column("ep", justify="right", width=4)
    table.add_column("best_ppl", justify="right", style="green bold", width=9)
    table.add_column("@ep", justify="right", style="dim", width=4)
    table.add_column("final_ppl", justify="right", width=10)
    table.add_column("trn_ppl", justify="right", style="dim", width=9)
    table.add_column("hrs", justify="right", style="dim", width=5)

    for rank, e in enumerate(exps, 1):
        parts = e["name"].split("_")
        short = "_".join(parts[3:]) if len(parts) > 3 else e["name"]

        n_exp = e.get("moe_n_experts")
        top_k = e.get("moe_top_k")
        moe_str = f"{n_exp}/{top_k}" if n_exp is not None else "—"

        table.add_row(
            str(rank),
            short,
            fmt(e.get("attn_type")),
            fmt(e.get("ffn_type")),
            fmt(e.get("qk_norm")),
            fmt(e.get("linear_weight_norm")),
            fmt(e.get("orthogonal_ffn_layers")),
            moe_str,
            fmt_params(e.get("n_params")),
            fmt(e.get("epochs_done"), 0),
            fmt(e.get("best_val_ppl")),
            fmt(e.get("best_epoch"), 0),
            fmt(e.get("final_val_ppl")),
            fmt(e.get("final_train_ppl")),
            fmt(e.get("total_time_h")),
        )

    console.print(table)
    console.print(f"[dim]Experiments dir: {EXPERIMENTS_DIR}[/dim]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify terminal leaderboard still works**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python3 rbf_ffn/leaderboard.py --top 5
```

Expected: table prints with 5 rows, no import errors.

- [ ] **Step 3: Commit**

```bash
git add rbf_ffn/leaderboard.py
git commit -m "refactor: leaderboard.py imports data loading from _leaderboard_data"
```

---

### Task 3: Write `generate_leaderboard_html.py` with tests

**Files:**
- Create: `rbf_ffn/generate_leaderboard_html.py`
- Create: `rbf_ffn/tests/test_generate_leaderboard_html.py`

- [ ] **Step 1: Write failing tests**

Create `rbf_ffn/tests/test_generate_leaderboard_html.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_generate_leaderboard_html.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'rbf_ffn.generate_leaderboard_html'`

- [ ] **Step 3: Create `rbf_ffn/generate_leaderboard_html.py`**

```python
#!/usr/bin/env python3
"""Generate a self-contained HTML leaderboard from experiment artifacts."""

import argparse
import html as html_module
from datetime import datetime
from pathlib import Path

from rbf_ffn._leaderboard_data import load_all_experiments, fmt, fmt_params

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

_COLUMNS = [
    ("#",          "#",           "num"),
    ("Experiment", "experiment",  "str"),
    ("attn",       "attn",        "str"),
    ("ffn",        "ffn",         "str"),
    ("qk",         "qk",          "str"),
    ("wn",         "wn",          "str"),
    ("orth_layers","orth_layers",  "str"),
    ("MoE",        "moe",         "str"),
    ("params",     "params",      "str"),
    ("ep",         "ep",          "num"),
    ("best_ppl",   "best_ppl",    "num"),
    ("@ep",        "at_ep",       "num"),
    ("final_ppl",  "final_ppl",   "num"),
    ("trn_ppl",    "trn_ppl",     "num"),
    ("hrs",        "hrs",         "num"),
]


def _row_cells(rank: int, e: dict) -> list[str]:
    parts = e["name"].split("_")
    short = "_".join(parts[3:]) if len(parts) > 3 else e["name"]

    n_exp = e.get("moe_n_experts")
    top_k = e.get("moe_top_k")
    moe_str = f"{n_exp}/{top_k}" if n_exp is not None else "—"

    return [
        str(rank),
        short,
        fmt(e.get("attn_type")),
        fmt(e.get("ffn_type")),
        fmt(e.get("qk_norm")),
        fmt(e.get("linear_weight_norm")),
        fmt(e.get("orthogonal_ffn_layers")),
        moe_str,
        fmt_params(e.get("n_params")),
        fmt(e.get("epochs_done"), 0),
        fmt(e.get("best_val_ppl")),
        fmt(e.get("best_epoch"), 0),
        fmt(e.get("final_val_ppl")),
        fmt(e.get("final_train_ppl")),
        fmt(e.get("total_time_h")),
    ]


def generate_html(exps: list[dict], generated_at: str) -> str:
    # Build table rows HTML
    rows_html = []
    for rank, e in enumerate(exps, 1):
        cells = _row_cells(rank, e)
        rank_class = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "")
        config_escaped = html_module.escape(e.get("config_text", ""), quote=True)
        row = f'  <tr class="{rank_class}" data-config="{config_escaped}" onclick="toggleConfig(this)">\n'
        for cell in cells:
            row += f"    <td>{html_module.escape(str(cell))}</td>\n"
        row += "  </tr>\n"
        row += f'  <tr class="config-row" style="display:none"><td colspan="{len(_COLUMNS)}"><pre class="config-pre"></pre></td></tr>\n'
        rows_html.append(row)

    thead_cells = "".join(
        f'<th onclick="sortTable({i})" data-col="{col_id}" data-type="{col_type}">'
        f'{label}<span class="sort-ind"></span></th>\n'
        for i, (label, col_id, col_type) in enumerate(_COLUMNS)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>rbf_ffn Leaderboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 13px;
    padding: 24px;
  }}
  h1 {{ color: #a0c4ff; margin-bottom: 16px; font-size: 18px; letter-spacing: 1px; }}
  #filter-input {{
    background: #16213e;
    border: 1px solid #444;
    color: #e0e0e0;
    padding: 6px 10px;
    font-family: inherit;
    font-size: 13px;
    width: 320px;
    margin-bottom: 14px;
    border-radius: 3px;
  }}
  #filter-input:focus {{ outline: none; border-color: #a0c4ff; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{
    border-collapse: collapse;
    white-space: nowrap;
    width: 100%;
  }}
  th {{
    background: #16213e;
    color: #a0c4ff;
    padding: 7px 10px;
    cursor: pointer;
    user-select: none;
    text-align: right;
    border-bottom: 2px solid #444;
  }}
  th:first-child, th:nth-child(2) {{ text-align: left; }}
  th:hover {{ background: #1e2f5e; }}
  .sort-ind {{ margin-left: 4px; font-size: 10px; color: #ffcc00; }}
  td {{
    padding: 5px 10px;
    border-bottom: 1px solid #2a2a4a;
    text-align: right;
  }}
  td:first-child, td:nth-child(2) {{ text-align: left; }}
  tr:hover > td {{ background: #1e2f5e; cursor: pointer; }}
  .rank-1 > td {{ background: #3a2e00; }}
  .rank-1 > td:first-child {{ color: #ffd700; font-weight: bold; }}
  .rank-2 > td {{ background: #2a2a2a; }}
  .rank-2 > td:first-child {{ color: #c0c0c0; font-weight: bold; }}
  .rank-3 > td {{ background: #2a1e0a; }}
  .rank-3 > td:first-child {{ color: #cd7f32; font-weight: bold; }}
  .rank-1:hover > td, .rank-2:hover > td, .rank-3:hover > td {{ filter: brightness(1.3); }}
  .config-row td {{ padding: 0; background: #0f0f1e; }}
  .config-pre {{
    padding: 12px 16px;
    color: #88ccaa;
    font-size: 12px;
    white-space: pre-wrap;
    word-break: break-word;
    border-left: 3px solid #a0c4ff;
    margin: 4px 8px;
  }}
  footer {{
    margin-top: 18px;
    font-size: 11px;
    color: #555;
  }}
</style>
</head>
<body>
<h1>rbf_ffn Leaderboard &mdash; {len(exps)} experiments</h1>
<input id="filter-input" type="text" placeholder="Filter experiments..." oninput="filterTable(this.value)">
<div class="table-wrap">
<table id="lb-table">
<thead><tr>
{thead_cells}</tr></thead>
<tbody>
{"".join(rows_html)}</tbody>
</table>
</div>
<footer>Generated: {generated_at}</footer>

<script>
(function() {{
  let sortCol = 10;  // best_ppl column index
  let sortAsc = true;

  function cellValue(row, col) {{
    const cells = row.querySelectorAll('td');
    return cells[col] ? cells[col].textContent.trim() : '';
  }}

  function parseVal(v, colType) {{
    if (v === '—' || v === '') return colType === 'num' ? Infinity : '';
    if (colType === 'num') {{
      const n = parseFloat(v.replace(/[MB K]/g, ''));
      return isNaN(n) ? Infinity : n;
    }}
    return v.toLowerCase();
  }}

  window.sortTable = function(col) {{
    const table = document.getElementById('lb-table');
    const ths = table.querySelectorAll('th');
    const colType = ths[col].dataset.type;

    if (sortCol === col) {{ sortAsc = !sortAsc; }}
    else {{ sortCol = col; sortAsc = true; }}

    ths.forEach((th, i) => {{
      th.querySelector('.sort-ind').textContent =
        i === col ? (sortAsc ? ' ▲' : ' ▼') : '';
    }});

    const tbody = table.querySelector('tbody');
    // Gather data rows (skip config-rows)
    const pairs = [];
    const rows = Array.from(tbody.rows);
    for (let i = 0; i < rows.length; i += 2) {{
      pairs.push([rows[i], rows[i+1]]);
    }}

    pairs.sort((a, b) => {{
      const va = parseVal(cellValue(a[0], col), colType);
      const vb = parseVal(cellValue(b[0], col), colType);
      if (va < vb) return sortAsc ? -1 : 1;
      if (va > vb) return sortAsc ? 1 : -1;
      return 0;
    }});

    pairs.forEach(([dr, cr]) => {{ tbody.appendChild(dr); tbody.appendChild(cr); }});
    rerank();
  }};

  function rerank() {{
    const tbody = document.getElementById('lb-table').querySelector('tbody');
    let visibleRank = 0;
    Array.from(tbody.rows).forEach(row => {{
      if (row.classList.contains('config-row')) return;
      if (row.style.display === 'none') return;
      visibleRank++;
      row.classList.remove('rank-1', 'rank-2', 'rank-3');
      if (visibleRank <= 3) row.classList.add('rank-' + visibleRank);
      row.cells[0].textContent = visibleRank;
    }});
  }}

  window.filterTable = function(query) {{
    const q = query.toLowerCase();
    const tbody = document.getElementById('lb-table').querySelector('tbody');
    const rows = Array.from(tbody.rows);
    for (let i = 0; i < rows.length; i += 2) {{
      const dataRow = rows[i];
      const configRow = rows[i + 1];
      const text = dataRow.textContent.toLowerCase();
      const visible = q === '' || text.includes(q);
      dataRow.style.display = visible ? '' : 'none';
      // hide config panel too when row is filtered out
      if (!visible) configRow.style.display = 'none';
    }};
    rerank();
  }};

  window.toggleConfig = function(row) {{
    const configRow = row.nextElementSibling;
    if (!configRow || !configRow.classList.contains('config-row')) return;
    const isOpen = configRow.style.display !== 'none';
    // Close all open config rows
    document.querySelectorAll('.config-row').forEach(r => r.style.display = 'none');
    if (!isOpen) {{
      configRow.querySelector('.config-pre').textContent = row.dataset.config;
      configRow.style.display = '';
    }}
  }};

  // Apply default sort indicator on load
  const ths = document.querySelectorAll('#lb-table th');
  if (ths[sortCol]) ths[sortCol].querySelector('.sort-ind').textContent = ' ▲';
}})();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML leaderboard")
    parser.add_argument("--out", default=str(Path(__file__).parent / "leaderboard.html"))
    parser.add_argument("--min-epochs", type=int, default=1)
    args = parser.parse_args()

    exps = load_all_experiments(EXPERIMENTS_DIR, min_epochs=args.min_epochs)
    exps.sort(key=lambda e: e.get("best_val_ppl") or float("inf"))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = generate_html(exps, generated_at=generated_at)

    out_path = Path(args.out)
    out_path.write_text(content)
    print(f"Written: {out_path.name} ({len(exps)} experiments)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_generate_leaderboard_html.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Generate the HTML and open it**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python3 rbf_ffn/generate_leaderboard_html.py
```

Expected output: `Written: leaderboard.html (N experiments)`

Then open in browser:
```bash
xdg-open rbf_ffn/leaderboard.html
```

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/generate_leaderboard_html.py rbf_ffn/tests/test_generate_leaderboard_html.py
git commit -m "feat: add generate_leaderboard_html.py — self-contained interactive HTML leaderboard"
```

---

### Task 4: Run full test suite and verify nothing regressed

**Files:** none

- [ ] **Step 1: Run all leaderboard tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python -m pytest rbf_ffn/tests/test_leaderboard_data.py rbf_ffn/tests/test_generate_leaderboard_html.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Smoke-test terminal leaderboard**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python3 rbf_ffn/leaderboard.py --top 5
```

Expected: 5-row table printed without errors.

- [ ] **Step 3: Smoke-test HTML generation**

```bash
cd /home/harikrishnan-c/projects/machine-notes && python3 rbf_ffn/generate_leaderboard_html.py && wc -l rbf_ffn/leaderboard.html
```

Expected: `Written: leaderboard.html (N experiments)` and a line count > 100.
