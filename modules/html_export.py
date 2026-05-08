"""Interactive HTML report export.
BatterySense AI
Author: AU P. Vajeeston, 2026
"""

from __future__ import annotations

import base64
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.io as pio


def _image_to_data_uri(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower().replace(".", "") or "png"
    return f"data:image/{suffix};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def _df_to_html_table(df: pd.DataFrame | None, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No data available.</p>"
    display_df = df.head(max_rows).copy()
    table = display_df.to_html(index=False, classes="data-table", border=0, justify="left")
    return f"<div class='table-wrap'>{table}</div>"


def _mapping_table(mapping: dict[str, str | None]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v or 'Not mapped'))}</td></tr>"
        for k, v in mapping.items()
    )
    return f"<div class='table-wrap'><table class='data-table'><thead><tr><th>Semantic field</th><th>Mapped column</th></tr></thead><tbody>{rows}</tbody></table></div>"


def _inline_markdown_to_html(escaped_text: str) -> str:
    escaped_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped_text)
    escaped_text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", escaped_text)
    escaped_text = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped_text)
    return escaped_text


def _markdown_table_to_html(raw_lines: list[str]) -> str:
    if len(raw_lines) < 2:
        return ""
    rows = []
    for line in raw_lines:
        line = line.strip().strip("|")
        rows.append([c.strip() for c in line.split("|")])
    if len(rows) < 2:
        return ""
    header = rows[0]
    body = rows[2:] if all(set(c.replace(" ", "")) <= {"-", ":"} for c in rows[1]) else rows[1:]
    head_html = "".join(f"<th>{_inline_markdown_to_html(html.escape(c))}</th>" for c in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_inline_markdown_to_html(html.escape(c))}</td>" for c in row) + "</tr>"
        for row in body
    )
    return f"<div class='table-wrap'><table class='data-table'><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table></div>"


def _markdown_to_basic_html(text: str) -> str:
    """Small markdown-like renderer for headings, lists, code fences and tables."""
    raw_lines = (text or "").splitlines()
    out: list[str] = []
    in_ul = False
    in_ol = False
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            close_lists()
            out.append(_markdown_table_to_html(table_lines))
            table_lines = []

    for raw in raw_lines:
        stripped_raw = raw.strip()
        if stripped_raw.startswith("```"):
            flush_table()
            if not in_code:
                close_lists()
                in_code = True
                code_lines = []
            else:
                out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                in_code = False
            continue
        if in_code:
            code_lines.append(raw)
            continue

        if stripped_raw.startswith("|") and stripped_raw.endswith("|"):
            table_lines.append(stripped_raw)
            continue
        else:
            flush_table()

        if not stripped_raw:
            close_lists()
            continue

        escaped = html.escape(stripped_raw)
        if stripped_raw.startswith("# "):
            close_lists(); out.append(f"<h2>{_inline_markdown_to_html(html.escape(stripped_raw[2:].strip()))}</h2>")
        elif stripped_raw.startswith("## "):
            close_lists(); out.append(f"<h3>{_inline_markdown_to_html(html.escape(stripped_raw[3:].strip()))}</h3>")
        elif stripped_raw.startswith("### "):
            close_lists(); out.append(f"<h4>{_inline_markdown_to_html(html.escape(stripped_raw[4:].strip()))}</h4>")
        elif stripped_raw.startswith("- ") or stripped_raw.startswith("* "):
            if in_ol:
                out.append("</ol>"); in_ol = False
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{_inline_markdown_to_html(html.escape(stripped_raw[2:].strip()))}</li>")
        elif re.match(r"^\d+\.\s+", stripped_raw):
            if in_ul:
                out.append("</ul>"); in_ul = False
            if not in_ol:
                out.append("<ol>"); in_ol = True
            item = re.sub(r"^\d+\.\s+", "", stripped_raw)
            out.append(f"<li>{_inline_markdown_to_html(html.escape(item))}</li>")
        else:
            close_lists(); out.append(f"<p>{_inline_markdown_to_html(escaped)}</p>")

    flush_table()
    if in_code:
        out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    close_lists()
    return "\n".join(out)


def save_html_report(
    title: str,
    uploaded_file_names: list[str],
    dataset_profile: dict[str, Any],
    mapping: dict[str, str | None],
    analysis: dict[str, Any],
    plots: dict[str, Any],
    interpretation_text: str,
    output_dir: str | Path = "outputs/reports",
    banner_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"battery_ai_report_{safe_timestamp}.html"

    summary = analysis.get("summary", {})
    banner_uri = _image_to_data_uri(banner_path)
    metrics = analysis.get("cell_metrics", pd.DataFrame())
    anomalies = analysis.get("anomalies", pd.DataFrame())
    deviation_report = analysis.get("deviation_report", {}) or {}
    deviation_std = deviation_report.get("std_table") if isinstance(deviation_report, dict) else pd.DataFrame()
    deviation_per_cell = deviation_report.get("per_cell_deviation") if isinstance(deviation_report, dict) else pd.DataFrame()

    key_summary = {
        "Best cell": summary.get("best_performing_cell", "N/A"),
        "Worst cell": summary.get("worst_performing_cell", "N/A"),
        "Mean retention, max baseline": summary.get("mean_last_capacity_retention_pct", "N/A"),
        "Mean retention, first baseline": summary.get("mean_last_capacity_retention_from_first_pct", "N/A"),
        "Mean calculated CE": summary.get("mean_calculated_ce_pct", "N/A"),
    }

    def _fmt_metric_value(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    plot_sections: list[str] = []
    for name, fig in plots.items():
        plot_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config={"displaylogo": False, "responsive": True})
        section_title = name.replace("_", " ").title()
        plot_sections.append(f"<section><h2>{html.escape(section_title)}</h2>{plot_html}</section>")

    file_list = "".join(f"<li>{html.escape(name)}</li>" for name in uploaded_file_names)
    column_list = ", ".join(html.escape(str(c)) for c in dataset_profile.get("column_names", []))

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
:root {{ --bg: #f7f9fc; --card: #ffffff; --text: #162033; --muted: #667085; --line: #e6eaf2; --accent: #1f77b4; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, Arial, sans-serif; line-height: 1.55; }}
header {{ padding: 36px 48px; background: linear-gradient(120deg, #122033, #1f4b7a); color: white; }}
header h1 {{ margin: 0 0 8px; font-size: 34px; }}
header p {{ margin: 0; opacity: 0.9; }}
.banner {{ width: 100%; max-height: 260px; object-fit: contain; margin-bottom: 18px; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
section {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 24px; margin: 18px 0; box-shadow: 0 8px 24px rgba(16, 24, 40, 0.05); }}
h2 {{ margin-top: 0; color: #12395d; }}
h3 {{ color: #164a77; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
.metric {{ background: #f3f7fb; border: 1px solid var(--line); border-radius: 14px; padding: 16px; }}
.metric .label {{ color: var(--muted); font-size: 13px; }}
.metric .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
.table-wrap {{ width: 100%; overflow-x: auto; margin: 10px 0 18px; border: 1px solid var(--line); border-radius: 12px; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 760px; }}
.data-table th, .data-table td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; white-space: nowrap; }}
.data-table th {{ background: #f3f7fb; position: sticky; top: 0; }}
code {{ background: #f3f7fb; padding: 2px 4px; border-radius: 4px; }}
pre {{ background: #0f172a; color: #e5e7eb; padding: 14px; border-radius: 12px; overflow-x: auto; }}
.muted {{ color: var(--muted); }}
footer {{ color: var(--muted); text-align: center; padding: 28px; }}
</style>
</head>
<body>
<header>
  {f'<img class="banner" src="{banner_uri}" alt="BatterySense AI banner" />' if banner_uri else ""}
  <h1>{html.escape(title)}</h1>
  <p>Generated: {html.escape(timestamp)}</p>
</header>
<main>
<section>
  <h2>Dataset summary</h2>
  <div class="grid">
    <div class="metric"><div class="label">Rows</div><div class="value">{dataset_profile.get('rows', 'N/A')}</div></div>
    <div class="metric"><div class="label">Columns</div><div class="value">{dataset_profile.get('columns', 'N/A')}</div></div>
    <div class="metric"><div class="label">Cells/samples</div><div class="value">{summary.get('n_cells_or_samples', 'N/A')}</div></div>
    <div class="metric"><div class="label">Anomaly flags</div><div class="value">{summary.get('n_anomalies', 'N/A')}</div></div>
    {''.join(f'<div class="metric"><div class="label">{html.escape(str(k))}</div><div class="value">{html.escape(_fmt_metric_value(v))}</div></div>' for k, v in key_summary.items())}
  </div>
  <h3>Uploaded files</h3>
  <ul>{file_list}</ul>
  <h3>Columns</h3>
  <p>{column_list}</p>
</section>

<section><h2>Detected / selected column mapping</h2>{_mapping_table(mapping)}</section>
<section><h2>Key metrics table</h2>{_df_to_html_table(metrics, max_rows=200)}</section>
<section>
  <h2>Multi-cell deviation / reproducibility report</h2>
  <p class="muted">Standard deviation and relative standard deviation across cells for CE, first-cycle capacities, upper/max capacities, final retention, and degradation.</p>
  {_df_to_html_table(deviation_std if isinstance(deviation_std, pd.DataFrame) else pd.DataFrame(), max_rows=200)}
  <h3>Per-cell deviation from mean</h3>
  {_df_to_html_table(deviation_per_cell if isinstance(deviation_per_cell, pd.DataFrame) else pd.DataFrame(), max_rows=200)}
</section>
<section><h2>Detected anomalies</h2>{_df_to_html_table(anomalies, max_rows=200)}</section>
{''.join(plot_sections)}
<section><h2>AI / automatic scientific interpretation</h2>{_markdown_to_basic_html(interpretation_text)}</section>
</main>
<footer>© 2026 AU P. Vajeeston · BatterySense AI · Free for non-commercial research, education, and personal use. Commercial use requires prior written permission.</footer>
</body>
</html>
"""
    report_file.write_text(html_text, encoding="utf-8")
    return report_file
