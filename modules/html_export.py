"""Interactive HTML report export.
BatterySense AI
**Author: AU P. Vajeeston, 2026** """

from __future__ import annotations

import base64
import html
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
    return display_df.to_html(index=False, classes="data-table", border=0, justify="left")


def _mapping_table(mapping: dict[str, str | None]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v or 'Not mapped'))}</td></tr>"
        for k, v in mapping.items()
    )
    return f"<table class='data-table'><thead><tr><th>Semantic field</th><th>Mapped column</th></tr></thead><tbody>{rows}</tbody></table>"


def _markdown_to_basic_html(text: str) -> str:
    """Small markdown-like renderer for headings and bullet lists.

    This avoids adding extra dependencies. The original text is escaped first.
    """
    lines = html.escape(text or "").splitlines()
    out: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue

        if stripped.startswith("# "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h2>{stripped[2:]}</h2>")
        elif stripped.startswith("## "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h3>{stripped[3:]}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{stripped[2:]}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{stripped}</p>")

    if in_list:
        out.append("</ul>")
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
    """Save one complete self-contained report shell with interactive Plotly figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"battery_ai_report_{safe_timestamp}.html"

    summary = analysis.get("summary", {})
    banner_uri = _image_to_data_uri(banner_path)
    metrics = analysis.get("cell_metrics", pd.DataFrame())
    anomalies = analysis.get("anomalies", pd.DataFrame())

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
:root {{
  --bg: #f7f9fc;
  --card: #ffffff;
  --text: #162033;
  --muted: #667085;
  --line: #e6eaf2;
  --accent: #1f77b4;
}}
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
.data-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.data-table th, .data-table td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
.data-table th {{ background: #f3f7fb; }}
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
  </div>
  <h3>Uploaded files</h3>
  <ul>{file_list}</ul>
  <h3>Columns</h3>
  <p>{column_list}</p>
</section>

<section>
  <h2>Detected / selected column mapping</h2>
  {_mapping_table(mapping)}
</section>

<section>
  <h2>Key metrics table</h2>
  {_df_to_html_table(metrics, max_rows=200)}
</section>

<section>
  <h2>Detected anomalies</h2>
  {_df_to_html_table(anomalies, max_rows=200)}
</section>

{''.join(plot_sections)}

<section>
  <h2>AI / automatic scientific interpretation</h2>
  {_markdown_to_basic_html(interpretation_text)}
</section>
</main>
<footer>Battery AI Report Tool</footer>
</body>
</html>
"""
    report_file.write_text(html_text, encoding="utf-8")
    return report_file
