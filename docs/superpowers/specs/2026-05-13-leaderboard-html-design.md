# Leaderboard HTML Page — Design Spec

**Date:** 2026-05-13  
**Status:** Approved

## Overview

A Python script (`rbf_ffn/generate_leaderboard_html.py`) that reads experiment artifacts and writes a self-contained `rbf_ffn/leaderboard.html` file. No server, no new runtime dependencies. The HTML file is fully interactive via embedded vanilla JS and CSS.

## Data Loading

Shared data loading logic is extracted from `leaderboard.py` into `rbf_ffn/_leaderboard_data.py`:
- `load_all_experiments(experiments_dir) -> list[dict]` — iterates experiment dirs, reads `metrics.jsonl`, `config.yaml`, and `params.json`
- Both `leaderboard.py` and `generate_leaderboard_html.py` import from this module

`leaderboard.py` is updated to import `load_all_experiments` instead of containing its own copy.

## Output File

`rbf_ffn/leaderboard.html` — single self-contained file, no external CDN links, no external fonts. CSS and JS are inlined in `<style>` and `<script>` tags.

## Table Columns

Same as the terminal leaderboard:

| Column | Notes |
|---|---|
| # | Rank |
| Experiment | Descriptive part of dir name (timestamp prefix stripped) |
| attn | attn_type |
| ffn | ffn_type |
| qk | qk_norm |
| wn | linear_weight_norm |
| orth_layers | orthogonal_ffn_layers |
| MoE | `n_experts/top_k` or `—` |
| params | Formatted (e.g. `30.5M`) |
| ep | Completed epochs |
| best_ppl | Best val perplexity (default sort key, ascending) |
| @ep | Epoch where best_ppl was achieved |
| final_ppl | Val perplexity at last epoch |
| trn_ppl | Train perplexity at last epoch |
| hrs | Total wall-clock hours |

## Interactivity

### Sort
- Click any `<th>` to sort by that column (ascending first, then toggle descending)
- Active sort column shows a `▲` / `▼` indicator
- Default sort: `best_ppl` ascending

### Filter
- Search box above the table
- Live filtering as user types — matches against the full experiment name and visible cell text
- Case-insensitive

### Row Highlights
- Rank 1: gold background
- Rank 2: silver
- Rank 3: bronze
- Highlights re-apply after sort/filter based on current visible rank

### Expandable Config
- Click any row to toggle an inline expansion panel below it
- Panel shows the raw `config.yaml` text for that experiment in a `<pre>` block
- Only one row expanded at a time; clicking another row closes the previous one
- The full `config.yaml` content is embedded in the HTML as a `data-config` attribute on each `<tr>`

## Rendering

- Dark background (`#1a1a2e`) with a light table to match a typical ML research terminal aesthetic
- Monospace font throughout
- Responsive: horizontal scroll on narrow viewports rather than wrapping
- Generated timestamp shown in the page footer

## Script Interface

```
python3 rbf_ffn/generate_leaderboard_html.py [--out PATH]
```

- `--out` defaults to `rbf_ffn/leaderboard.html`
- Prints a confirmation line: `Written: leaderboard.html (N experiments)`

## Files Changed

| File | Change |
|---|---|
| `rbf_ffn/_leaderboard_data.py` | New — shared data loading |
| `rbf_ffn/leaderboard.py` | Updated to import from `_leaderboard_data` |
| `rbf_ffn/generate_leaderboard_html.py` | New — HTML generator |
| `rbf_ffn/leaderboard.html` | Generated output (gitignored or committed as desired) |
