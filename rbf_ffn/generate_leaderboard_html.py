#!/usr/bin/env python3
"""Generate a self-contained HTML leaderboard from experiment artifacts."""

import argparse
import html as html_module
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
  let sortCol = 10;
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
      if (!visible) configRow.style.display = 'none';
    }};
    rerank();
  }};

  window.toggleConfig = function(row) {{
    const configRow = row.nextElementSibling;
    if (!configRow || !configRow.classList.contains('config-row')) return;
    const isOpen = configRow.style.display !== 'none';
    document.querySelectorAll('.config-row').forEach(r => r.style.display = 'none');
    if (!isOpen) {{
      configRow.querySelector('.config-pre').textContent = row.dataset.config;
      configRow.style.display = '';
    }}
  }};

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
