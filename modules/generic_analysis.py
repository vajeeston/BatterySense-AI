"""Generic non-battery dataset analysis utilities.
BatterySense AI
Author: AU P. Vajeeston, 2026

This module is intentionally compact. It avoids rendering large nested JSON
objects in Streamlit, because that can freeze the browser and produce messy PDF
exports for wide overview tables.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd


def _round_numeric_df(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(ndigits)
    return out.replace([np.inf, -np.inf], np.nan)


def detect_battery_overview_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Find useful battery-overview-style columns in generic summary tables."""
    lower_map = {str(c).lower(): str(c) for c in df.columns}

    def pick(keywords: list[str], limit: int = 20) -> list[str]:
        found: list[str] = []
        for low, original in lower_map.items():
            if all(k in low for k in keywords):
                found.append(original)
        return found[:limit]

    return {
        "identity_columns": [c for c in df.columns if str(c).lower() in {"file_name", "nw_file_name", "cell", "cell_name", "sample_name", "slurry_name", "group"}],
        "first_cycle_columns": pick(["first", "cycle"], 20) + pick(["1st", "cycle"], 20) + pick(["fce"], 10),
        "capacity_columns": pick(["capacity"], 30) + pick(["ch_"], 20) + pick(["dch"], 20),
        "ce_columns": pick(["ce"], 40),
        "cycle_life_columns": pick(["cycle_loss"], 20) + pick(["cycle_dch"], 20) + pick(["number", "cycle"], 10),
        "rate_recovery_columns": pick(["fast", "charge"], 10) + pick(["capacity", "recovery"], 10),
    }


def generic_dataset_summary(df: pd.DataFrame, max_columns: int = 100) -> dict[str, Any]:
    """Return a compact general-purpose dataset summary for non-battery files.

    The output is designed for tables, not raw JSON rendering.
    """
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    missing = (
        pd.DataFrame({
            "column": df.columns.astype(str),
            "missing_count": df.isna().sum().to_numpy(),
            "missing_percent": (df.isna().mean() * 100).round(2).to_numpy(),
            "dtype": [str(t) for t in df.dtypes.to_numpy()],
        })
        .sort_values(["missing_percent", "missing_count"], ascending=False)
        .reset_index(drop=True)
    )

    if not numeric.empty:
        numeric_desc = numeric.describe().T.reset_index(names="column")
        numeric_desc["missing_percent"] = numeric.isna().mean().reindex(numeric_desc["column"]).to_numpy() * 100
        numeric_desc = _round_numeric_df(numeric_desc, 4)
    else:
        numeric_desc = pd.DataFrame()

    categorical_summary_rows: list[dict[str, Any]] = []
    for col in categorical.columns[:max_columns]:
        vc = df[col].astype("string").fillna("NaN").value_counts(dropna=False).head(5)
        top = "; ".join([f"{idx}: {val}" for idx, val in vc.items()])
        categorical_summary_rows.append({
            "column": str(col),
            "unique_values": int(df[col].nunique(dropna=True)),
            "missing_percent": round(float(df[col].isna().mean() * 100), 2),
            "top_values": top,
        })
    categorical_summary = pd.DataFrame(categorical_summary_rows)

    corr_pairs = pd.DataFrame()
    if numeric.shape[1] >= 2:
        # Limit correlation calculation for very wide datasets to prevent slow UI.
        numeric_for_corr = numeric.loc[:, numeric.notna().mean().sort_values(ascending=False).head(60).index]
        corr = numeric_for_corr.corr(numeric_only=True).replace([np.inf, -np.inf], np.nan)
        rows: list[dict[str, Any]] = []
        cols = list(corr.columns)
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                val = corr.loc[c1, c2]
                if pd.notna(val):
                    rows.append({"column_1": c1, "column_2": c2, "correlation": float(val), "abs_correlation": abs(float(val))})
        corr_pairs = pd.DataFrame(rows)
        if not corr_pairs.empty:
            corr_pairs = corr_pairs.sort_values("abs_correlation", ascending=False).drop(columns="abs_correlation").head(30)
            corr_pairs = _round_numeric_df(corr_pairs, 4)

    battery_overview_cols = detect_battery_overview_columns(df)

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric.columns.astype(str).tolist(),
        "categorical_columns": categorical.columns.astype(str).tolist(),
        "missing_table": missing,
        "numeric_describe_table": numeric_desc,
        "categorical_summary_table": categorical_summary,
        "correlation_table": corr_pairs,
        "battery_overview_columns": battery_overview_cols,
    }


def compare_groups(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Aggregate numeric columns by a grouping column."""
    if group_col not in df.columns:
        return pd.DataFrame()
    value_cols = [c for c in value_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        return pd.DataFrame()
    grouped = df.groupby(group_col, dropna=False)[value_cols].agg(["count", "mean", "std", "min", "median", "max"])
    grouped.columns = ["_".join(map(str, c)).strip("_") for c in grouped.columns]
    return _round_numeric_df(grouped.reset_index(), 4)


def build_generic_interpretation(summary: dict[str, Any], group_col: str | None = None) -> str:
    """Create a deterministic, concise generic interpretation."""
    rows = summary.get("rows", 0)
    cols = summary.get("columns", 0)
    nnum = len(summary.get("numeric_columns", []))
    ncat = len(summary.get("categorical_columns", []))
    missing_table = summary.get("missing_table", pd.DataFrame())
    corr_table = summary.get("correlation_table", pd.DataFrame())

    lines = [
        "# Generic dataset interpretation",
        "",
        f"The uploaded dataset contains **{rows} rows** and **{cols} columns**, including **{nnum} numeric columns** and **{ncat} categorical columns**.",
    ]
    if isinstance(missing_table, pd.DataFrame) and not missing_table.empty:
        high_missing = missing_table[missing_table["missing_percent"] > 20].head(10)
        if not high_missing.empty:
            cols_txt = ", ".join(high_missing["column"].astype(str).tolist())
            lines.append(f"Several columns contain more than 20% missing values, including: {cols_txt}. These columns should be reviewed before modelling or comparison.")
        else:
            lines.append("No column among the displayed variables has severe missingness above 20%, based on the compact summary.")
    if isinstance(corr_table, pd.DataFrame) and not corr_table.empty:
        top = corr_table.iloc[0]
        lines.append(f"The strongest displayed numeric correlation is between **{top['column_1']}** and **{top['column_2']}** with r = **{top['correlation']}**.")
    overview_cols = summary.get("battery_overview_columns", {})
    useful = {k: v for k, v in overview_cols.items() if v}
    if useful:
        lines.append("Battery-overview-style columns were detected, so this file appears suitable for screening-level comparison of first-cycle behavior, capacity loss, CE, cycle-life, and group effects.")
    if group_col:
        lines.append(f"Group-level comparison was requested using **{group_col}**. Use the group table to identify materials or process groups with higher average capacity, lower capacity loss, or lower variability.")
    lines.extend([
        "",
        "Recommended next steps:",
        "- Remove or separately handle columns with high missingness before modelling.",
        "- Use group comparison for columns such as FCE, first-cycle lithiation/delithiation capacity, cycle loss, and average CE.",
        "- For raw cycle-level files, switch to Battery cycling analysis rather than Generic dataset analysis.",
    ])
    return "\n".join(lines)


def recommend_group_column(df: pd.DataFrame) -> str | None:
    """Choose a sensible default grouping column for overview datasets."""
    preferred = ["Group", "sample_name", "slurry_name", "cell_name", "file_name", "NW_file_name"]
    for col in preferred:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            return col
    cats = df.select_dtypes(exclude="number").columns.astype(str).tolist()
    for col in cats:
        if df[col].nunique(dropna=True) > 1:
            return col
    return None


def recommend_numeric_columns(df: pd.DataFrame, max_cols: int = 8) -> list[str]:
    """Pick useful numeric columns for screening-level overview comparisons."""
    preferred_exact = [
        "FCE(%)",
        "First_cycle_lithi(mAh/g)",
        "First_cycle_delithi(mAh/g)",
        "First_cycle_loss(mAh/g)",
        "Second_cycle_lithi(mAh/g)",
        "cycle_loss_50(%)",
        "cycle_loss_100(%)",
        "cycle_loss_200(%)",
        "Fast_charge_capability(%)",
        "Capacity Recovery(%)",
        "Last_cycle_Dch_capacity(mAh/g)",
        "Last_cycle_CE(%)",
        "Number of cycles",
    ]
    nums = [c for c in df.select_dtypes(include="number").columns.astype(str).tolist()]
    chosen = [c for c in preferred_exact if c in nums]
    if len(chosen) < max_cols:
        keywords = ["fce", "first", "delithi", "lithi", "cycle_loss", "capacity", "ce", "recovery"]
        for c in nums:
            low = c.lower()
            if c not in chosen and any(k in low for k in keywords):
                chosen.append(c)
            if len(chosen) >= max_cols:
                break
    return chosen[:max_cols]


def build_generic_key_findings(summary: dict[str, Any], group_table: pd.DataFrame | None = None, group_col: str | None = None) -> pd.DataFrame:
    """Create a compact table of key findings for reports."""
    rows: list[dict[str, Any]] = []
    missing_table = summary.get("missing_table", pd.DataFrame())
    corr_table = summary.get("correlation_table", pd.DataFrame())
    rows.append({"finding": "Dataset size", "value": f"{summary.get('rows', 0)} rows × {summary.get('columns', 0)} columns"})
    rows.append({"finding": "Column types", "value": f"{len(summary.get('numeric_columns', []))} numeric; {len(summary.get('categorical_columns', []))} categorical"})
    if isinstance(missing_table, pd.DataFrame) and not missing_table.empty:
        high_missing = missing_table[missing_table["missing_percent"] > 20]
        rows.append({"finding": "High-missing columns (>20%)", "value": int(len(high_missing))})
        if not high_missing.empty:
            rows.append({"finding": "Highest missingness", "value": f"{high_missing.iloc[0]['column']} ({high_missing.iloc[0]['missing_percent']}%)"})
    if isinstance(corr_table, pd.DataFrame) and not corr_table.empty:
        top = corr_table.iloc[0]
        rows.append({"finding": "Strongest numeric correlation", "value": f"{top['column_1']} vs {top['column_2']} (r={top['correlation']})"})
    if group_col and isinstance(group_table, pd.DataFrame) and not group_table.empty:
        rows.append({"finding": "Group comparison", "value": f"Grouped by {group_col}; {len(group_table)} group rows"})
    return pd.DataFrame(rows)


def make_generic_plots(df: pd.DataFrame, summary: dict[str, Any], group_col: str | None = None, value_cols: list[str] | None = None) -> dict[str, Any]:
    """Generate lightweight Plotly figures for generic / overview datasets."""
    import plotly.express as px

    plots: dict[str, Any] = {}
    missing_table = summary.get("missing_table", pd.DataFrame())
    if isinstance(missing_table, pd.DataFrame) and not missing_table.empty:
        mt = missing_table.sort_values("missing_percent", ascending=False).head(25)
        fig = px.bar(mt, x="column", y="missing_percent", title="Top missing-value columns", template="plotly_white")
        fig.update_layout(xaxis_title="Column", yaxis_title="Missing values (%)", xaxis_tickangle=-45)
        plots["missing_values"] = fig

    numeric_desc = summary.get("numeric_describe_table", pd.DataFrame())
    if isinstance(numeric_desc, pd.DataFrame) and not numeric_desc.empty and "std" in numeric_desc.columns:
        nd = numeric_desc.sort_values("std", ascending=False).head(25)
        fig = px.bar(nd, x="column", y="std", title="Numeric columns with highest standard deviation", template="plotly_white")
        fig.update_layout(xaxis_title="Column", yaxis_title="Standard deviation", xaxis_tickangle=-45)
        plots["numeric_variability"] = fig

    corr_table = summary.get("correlation_table", pd.DataFrame())
    if isinstance(corr_table, pd.DataFrame) and not corr_table.empty:
        ct = corr_table.head(20).copy()
        ct["pair"] = ct["column_1"].astype(str) + " vs " + ct["column_2"].astype(str)
        fig = px.bar(ct, x="pair", y="correlation", title="Strongest numeric correlations", template="plotly_white")
        fig.update_layout(xaxis_title="Column pair", yaxis_title="Correlation coefficient", xaxis_tickangle=-45)
        plots["top_correlations"] = fig

    if group_col and value_cols and group_col in df.columns:
        safe_value_cols = [c for c in value_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        # Limit to six selected variables to keep the report readable.
        for col in safe_value_cols[:6]:
            tmp = df[[group_col, col]].dropna()
            if tmp.empty:
                continue
            # Avoid very high-cardinality group charts.
            order = tmp.groupby(group_col)[col].mean().sort_values(ascending=False).head(20).index
            tmp = tmp[tmp[group_col].isin(order)]
            fig = px.box(tmp, x=group_col, y=col, title=f"{col} by {group_col}", template="plotly_white")
            fig.update_layout(xaxis_title=group_col, yaxis_title=col, xaxis_tickangle=-45)
            plots[f"group_box_{col}".replace("/", "_").replace("%", "pct").replace(" ", "_")] = fig
            break
    for fig in plots.values():
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(showline=True, linewidth=1, linecolor="#1F2937", mirror=True),
            yaxis=dict(showline=True, linewidth=1, linecolor="#1F2937", mirror=True),
            font=dict(color="#162033"),
        )
    return plots


def save_generic_html_report(
    title: str,
    uploaded_file_names: list[str],
    summary: dict[str, Any],
    group_table: pd.DataFrame | None,
    key_findings: pd.DataFrame,
    interpretation: str,
    plots: dict[str, Any] | None,
    output_dir: str | Path = "outputs/reports",
    banner_path: str | Path | None = None,
) -> Path:
    """Export a complete generic analysis report as HTML."""
    import base64
    import html as _html
    from datetime import datetime
    import plotly.io as pio

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"batterysense_generic_report_{safe_timestamp}.html"

    def img_uri(path: str | Path | None) -> str:
        if not path:
            return ""
        p = Path(path)
        if not p.exists():
            return ""
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")

    def table_html(df: pd.DataFrame | None, max_rows: int = 100) -> str:
        if df is None or df.empty:
            return "<p class='muted'>No data available.</p>"
        return "<div class='table-wrap'>" + df.head(max_rows).to_html(index=False, classes="data-table", border=0, escape=True) + "</div>"

    def md_to_html(text: str) -> str:
        # Small markdown renderer for headings, bold and bullets.
        out = []
        in_ul = False
        for raw in (text or "").splitlines():
            s = raw.strip()
            if not s:
                if in_ul:
                    out.append("</ul>"); in_ul = False
                continue
            esc = _html.escape(s).replace("**", "")
            if s.startswith("# "):
                if in_ul: out.append("</ul>"); in_ul = False
                out.append(f"<h2>{_html.escape(s[2:].strip())}</h2>")
            elif s.startswith("## "):
                if in_ul: out.append("</ul>"); in_ul = False
                out.append(f"<h3>{_html.escape(s[3:].strip())}</h3>")
            elif s.startswith("- "):
                if not in_ul:
                    out.append("<ul>"); in_ul = True
                out.append(f"<li>{_html.escape(s[2:].strip())}</li>")
            else:
                if in_ul: out.append("</ul>"); in_ul = False
                out.append(f"<p>{esc}</p>")
        if in_ul:
            out.append("</ul>")
        return "\n".join(out)

    banner = img_uri(banner_path)
    files = "".join(f"<li>{_html.escape(str(f))}</li>" for f in uploaded_file_names)
    plot_sections = []
    for name, fig in (plots or {}).items():
        plot_sections.append(f"<section><h2>{_html.escape(name.replace('_',' ').title())}</h2>{pio.to_html(fig, include_plotlyjs='cdn', full_html=False, config={'displaylogo': False, 'responsive': True})}</section>")

    html_text = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{_html.escape(title)}</title>
<style>
body{{margin:0;background:#f7f9fc;color:#162033;font-family:Inter,Arial,sans-serif;line-height:1.55}}header{{padding:34px 48px;background:linear-gradient(120deg,#122033,#1f4b7a);color:white}}main{{max-width:1180px;margin:0 auto;padding:28px}}section{{background:white;border:1px solid #e6eaf2;border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 8px 24px rgba(16,24,40,.05)}}.banner{{width:100%;max-height:240px;object-fit:contain;margin-bottom:16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}}.metric{{background:#f3f7fb;border:1px solid #e6eaf2;border-radius:14px;padding:16px}}.metric .label{{color:#667085;font-size:13px}}.metric .value{{font-size:22px;font-weight:700}}.table-wrap{{width:100%;overflow-x:auto;border:1px solid #e6eaf2;border-radius:12px;margin:10px 0 18px}}.data-table{{width:100%;border-collapse:collapse;font-size:13px;min-width:760px}}.data-table th,.data-table td{{border-bottom:1px solid #e6eaf2;padding:8px 10px;text-align:left;vertical-align:top;white-space:nowrap}}.data-table th{{background:#f3f7fb}}.muted{{color:#667085}}footer{{text-align:center;color:#667085;padding:28px}}
</style></head><body><header>{f'<img class="banner" src="{banner}" alt="BatterySense AI banner">' if banner else ''}<h1>{_html.escape(title)}</h1><p>Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}</p></header><main>
<section><h2>Dataset overview</h2><div class='grid'><div class='metric'><div class='label'>Rows</div><div class='value'>{summary.get('rows',0)}</div></div><div class='metric'><div class='label'>Columns</div><div class='value'>{summary.get('columns',0)}</div></div><div class='metric'><div class='label'>Numeric columns</div><div class='value'>{len(summary.get('numeric_columns', []))}</div></div><div class='metric'><div class='label'>Categorical columns</div><div class='value'>{len(summary.get('categorical_columns', []))}</div></div></div><h3>Uploaded files</h3><ul>{files}</ul></section>
<section><h2>Key findings</h2>{table_html(key_findings, 50)}</section>
<section><h2>Missing-value summary</h2>{table_html(summary.get('missing_table'), 60)}</section>
<section><h2>Numeric descriptive statistics</h2>{table_html(summary.get('numeric_describe_table'), 80)}</section>
<section><h2>Categorical summary</h2>{table_html(summary.get('categorical_summary_table'), 80)}</section>
<section><h2>Strongest numeric correlations</h2>{table_html(summary.get('correlation_table'), 60)}</section>
<section><h2>Group comparison</h2>{table_html(group_table, 100)}</section>
{''.join(plot_sections)}
<section><h2>Interpretation</h2>{md_to_html(interpretation)}</section>
</main><footer>© 2026 AU P. Vajeeston. BatterySense AI is free for non-commercial research, education, and personal use. Commercial use requires prior written permission.</footer></body></html>"""
    report_file.write_text(html_text, encoding="utf-8")
    return report_file


def save_generic_pdf_report(
    title: str,
    uploaded_file_names: list[str],
    summary: dict[str, Any],
    group_table: pd.DataFrame | None,
    key_findings: pd.DataFrame,
    interpretation: str,
    plots: dict[str, Any] | None,
    output_dir: str | Path = "outputs/reports",
) -> Path:
    """Export a compact generic PDF report."""
    from datetime import datetime
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    import html as _html
    import re as _re

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_path / f"batterysense_generic_report_{safe_timestamp}.pdf"
    styles = getSampleStyleSheet()
    page_size = landscape(A4)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=page_size, rightMargin=.9*cm, leftMargin=.9*cm, topMargin=1*cm, bottomMargin=1*cm)

    def clean_text(x: object) -> str:
        s = "" if x is None else str(x)
        s = s.replace("`", "")
        s = _html.escape(s)
        s = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return s

    def fmt(v: object, max_len: int = 42) -> str:
        try:
            if pd.isna(v): return ""
        except Exception:
            pass
        if isinstance(v, float):
            s = f"{v:.4g}"
        else:
            s = str(v)
        return s if len(s) <= max_len else s[:max_len-3] + "..."

    def add_table(story: list[Any], title_: str, df: pd.DataFrame | None, max_rows: int = 30, max_cols: int = 8) -> None:
        story.append(Paragraph(clean_text(title_), styles["Heading2"]))
        if df is None or df.empty:
            story.append(Paragraph("No data available.", styles["Normal"])); story.append(Spacer(1, 8)); return
        show = df.head(max_rows).iloc[:, :max_cols].copy()
        data = [[fmt(c, 24) for c in show.columns]] + [[fmt(v) for v in row] for row in show.values.tolist()]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#12395D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.25,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.extend([t, Spacer(1, 10)])

    story: list[Any] = [Paragraph(clean_text(title), styles["Title"]), Paragraph(f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", styles["Normal"]), Spacer(1, 10)]
    overview = pd.DataFrame([
        {"Item": "Rows", "Value": summary.get("rows", 0)},
        {"Item": "Columns", "Value": summary.get("columns", 0)},
        {"Item": "Numeric columns", "Value": len(summary.get("numeric_columns", []))},
        {"Item": "Categorical columns", "Value": len(summary.get("categorical_columns", []))},
        {"Item": "Uploaded files", "Value": ", ".join(uploaded_file_names)},
    ])
    add_table(story, "Dataset overview", overview, 10, 2)
    add_table(story, "Key findings", key_findings, 20, 2)
    add_table(story, "Missing-value summary", summary.get("missing_table"), 30, 4)
    add_table(story, "Numeric descriptive statistics", summary.get("numeric_describe_table"), 30, 7)
    add_table(story, "Categorical summary", summary.get("categorical_summary_table"), 20, 4)
    add_table(story, "Strongest numeric correlations", summary.get("correlation_table"), 25, 3)
    add_table(story, "Group comparison", group_table, 30, 8)

    story.append(PageBreak())
    story.append(Paragraph("Interpretation", styles["Heading2"]))
    for raw in (interpretation or "").splitlines():
        s = raw.strip()
        if not s: story.append(Spacer(1, 4)); continue
        if s.startswith("#"):
            story.append(Paragraph(clean_text(s.lstrip('#').strip()), styles["Heading3"]))
        elif s.startswith("- "):
            story.append(Paragraph("• " + clean_text(s[2:]), styles["BodyText"]))
        else:
            story.append(Paragraph(clean_text(s), styles["BodyText"]))

    if plots:
        fig_dir = output_path / "generic_pdf_figures"; fig_dir.mkdir(parents=True, exist_ok=True)
        story.append(PageBreak()); story.append(Paragraph("Figures", styles["Heading2"]))
        for name, fig in list(plots.items())[:5]:
            try:
                img = fig_dir / f"{name}.png"
                fig.write_image(str(img), width=1200, height=700, scale=2)
                story.append(Paragraph(clean_text(name.replace("_", " ").title()), styles["Heading3"]))
                story.append(Image(str(img), width=24*cm, height=14*cm)); story.append(Spacer(1, 8))
            except Exception:
                continue

    def footer(canvas, _doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.grey)
        canvas.drawCentredString(page_size[0]/2, .45*cm, "© 2026 AU P. Vajeeston · BatterySense AI · Free for non-commercial use; commercial use requires written permission.")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf_path
