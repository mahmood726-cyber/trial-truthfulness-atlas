"""Atlas dashboard HTML — self-contained, inline SVG, no external resources."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict

import pandas as pd

FLAG_COLUMNS = [
    ("outcome_drift", "Outcome drift"),
    ("n_drift", "N drift"),
    ("direction_concordance", "Direction concordance"),
    ("results_posting", "Results posting"),
]


def _summary_counts(atlas: pd.DataFrame, column: str) -> Dict[str, int]:
    counts = atlas[column].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _summary_block_html(atlas: pd.DataFrame) -> str:
    parts = ['<section class="summary"><h2>Per-flag summary</h2>']
    for col, label in FLAG_COLUMNS:
        counts = _summary_counts(atlas, col)
        items = " ".join(
            f'<span class="pill pill-{escape(k)}">{escape(k)}: {v}</span>'
            for k, v in sorted(counts.items())
        )
        parts.append(f'<div class="summary-row"><strong>{escape(label)}:</strong> {items}</div>')
    parts.append("</section>")
    return "".join(parts)


def _table_html(atlas: pd.DataFrame) -> str:
    cols = ["nct_id", "Study", "review_id", "bridge_method",
            "outcome_drift", "n_drift", "direction_concordance", "results_posting"]
    head = "".join(f"<th>{escape(c)}</th>" for c in cols)
    rows = []
    for _, r in atlas.iterrows():
        cells = "".join(
            f"<td>{escape(str(r.get(c)) if pd.notna(r.get(c)) else '—')}</td>"
            for c in cols
        )
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="atlas"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
       margin: 2em auto; color: #1a1a1a; padding: 0 1em; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }
.summary { background: #f6f8fa; padding: 1em; border-radius: 6px; margin: 1em 0; }
.summary-row { margin: 0.4em 0; }
.pill { display: inline-block; padding: 0.15em 0.5em; margin: 0.1em;
        border-radius: 999px; background: #e0e7ef; font-size: 0.85em; }
.pill-flagged, .pill-required_not_posted, .pill-substantively_different,
.pill-flipped, .pill-unbridgeable { background: #ffd9d9; }
.pill-not_flagged, .pill-identical, .pill-concordant, .pill-posted { background: #d9f0d9; }
.pill-unscoreable, .pill-refinement, .pill-not_required { background: #f0e6c0; }
table.atlas { width: 100%; border-collapse: collapse; margin-top: 1em; font-size: 0.88em; }
table.atlas th, table.atlas td { border: 1px solid #ddd; padding: 0.4em; text-align: left; }
table.atlas thead { background: #333; color: #fff; }
table.atlas tbody tr:nth-child(even) { background: #fafafa; }
""".strip()


def render(atlas: pd.DataFrame, title: str) -> str:
    summary = _summary_block_html(atlas)
    table = _table_html(atlas)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        f"<style>{_CSS}</style></head>\n"
        "<body>\n"
        f"<h1>{escape(title)}</h1>\n"
        f"<p><strong>Trials:</strong> {len(atlas)}</p>\n"
        f"{summary}\n"
        f"{table}\n"
        "</body>\n</html>\n"
    )


def write(atlas: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(atlas, title=title), encoding="utf-8")
