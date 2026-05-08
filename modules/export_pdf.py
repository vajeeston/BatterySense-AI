"""PDF export utilities.
BatterySense AI
Author: AU P. Vajeeston, 2026

The PDF export uses ReportLab for a reliable local export. Interactive Plotly
content remains in the HTML report; the PDF contains summary tables and, when
kaleido is installed, static snapshots of the figures.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import html
import re
from typing import Any

import pandas as pd


def _strip_code_fence(line: str) -> str:
    stripped = str(line).strip()
    if stripped.startswith("```"):
        return ""
    return str(line)


def _markdown_inline_to_reportlab(text: object) -> str:
    """Convert a safe subset of markdown inline syntax to ReportLab markup.

    ReportLab can fail when unknown font names are emitted, so inline code is
    rendered as plain escaped text instead of Courier markup.
    """
    raw = _strip_code_fence("" if text is None else str(text))
    raw = raw.replace("`", "")
    escaped = html.escape(raw)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _shorten(text: object, max_len: int = 90) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _format_table_value(value: object, max_len: int = 18) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4g}"
    return _shorten(value, max_len)


def save_pdf_report(
    title: str,
    uploaded_file_names: list[str],
    dataset_profile: dict[str, Any],
    analysis: dict[str, Any],
    plots: dict[str, Any],
    interpretation_text: str,
    output_dir: str | Path = "outputs/reports",
) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_path / f"battery_ai_report_{safe_timestamp}.pdf"

    styles = getSampleStyleSheet()
    page_size = landscape(A4)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=page_size, rightMargin=0.9 * cm, leftMargin=0.9 * cm, topMargin=1.0 * cm, bottomMargin=1.0 * cm)
    story = [
        Paragraph(_markdown_inline_to_reportlab(title), styles["Title"]),
        Paragraph(f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", styles["Normal"]),
        Spacer(1, 12),
    ]

    story.append(Paragraph("Dataset summary", styles["Heading2"]))
    summary_rows = [
        ["Rows", dataset_profile.get("rows", "N/A")],
        ["Columns", dataset_profile.get("columns", "N/A")],
        ["Uploaded files", ", ".join(uploaded_file_names)],
        ["Cells/samples", analysis.get("summary", {}).get("n_cells_or_samples", "N/A")],
        ["Anomaly flags", analysis.get("summary", {}).get("n_anomalies", "N/A")],
        ["Excluded rows", analysis.get("summary", {}).get("excluded_rows", "N/A")],
        ["Best cell", analysis.get("summary", {}).get("best_performing_cell", "N/A")],
        ["Worst cell", analysis.get("summary", {}).get("worst_performing_cell", "N/A")],
    ]
    table = Table(summary_rows, colWidths=[4 * cm, 18 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF3F8")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.extend([table, Spacer(1, 12)])

    metrics = analysis.get("cell_metrics", pd.DataFrame())
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        story.append(Paragraph("Key metrics", styles["Heading2"]))
        cols = [c for c in [
            "cell_or_sample",
            "n_points",
            "first_charge_or_lithiation_capacity",
            "first_discharge_or_delithiation_capacity",
            "first_coulombic_efficiency_pct",
            "max_discharge_or_delithiation_capacity",
            "last_retention_capacity",
            "last_capacity_retention_from_first_pct",
            "last_capacity_retention_from_max_pct",
            "degradation_rate_pct_per_cycle",
            "average_coulombic_efficiency_pct",
            "last_accumulated_ce_pct",
            "total_accumulated_irreversible_capacity",
        ] if c in metrics.columns]
        table_data = [[_shorten(c, 22) for c in cols]] + [[_format_table_value(v, 16) for v in row] for row in metrics[cols].head(25).values.tolist()]
        col_width = max(1.6 * cm, min(2.7 * cm, 25 * cm / max(1, len(cols))))
        t = Table(table_data, repeatRows=1, colWidths=[col_width] * len(cols))
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12395D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.extend([t, Spacer(1, 12)])

    deviation_report = analysis.get("deviation_report", {}) or {}
    deviation_std = deviation_report.get("std_table") if isinstance(deviation_report, dict) else pd.DataFrame()
    deviation_per_cell = deviation_report.get("per_cell_deviation") if isinstance(deviation_report, dict) else pd.DataFrame()
    if isinstance(deviation_std, pd.DataFrame) and not deviation_std.empty:
        story.append(Paragraph("Multi-cell deviation / reproducibility report", styles["Heading2"]))
        std_cols = [c for c in ["parameter", "n_cells", "mean", "std", "rsd_pct", "min", "max", "range"] if c in deviation_std.columns]
        table_data = [std_cols] + [[_format_table_value(v, 20) for v in row] for row in deviation_std[std_cols].head(30).values.tolist()]
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12395D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.extend([t, Spacer(1, 12)])
    if isinstance(deviation_per_cell, pd.DataFrame) and not deviation_per_cell.empty:
        story.append(Paragraph("Per-cell deviation from mean", styles["Heading3"]))
        pc_cols = deviation_per_cell.columns[:8].tolist()
        table_data = [pc_cols] + [[_format_table_value(v, 16) for v in row] for row in deviation_per_cell[pc_cols].head(20).values.tolist()]
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3F8")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.extend([t, Spacer(1, 12)])

    story.append(PageBreak())
    story.append(Paragraph("Scientific interpretation", styles["Heading2"]))
    in_code_block = False
    for raw_line in (interpretation_text or "").splitlines():
        line = str(raw_line).rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if not stripped:
            story.append(Spacer(1, 5))
        elif stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            story.append(Paragraph(_markdown_inline_to_reportlab(heading), styles["Heading3"]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph("• " + _markdown_inline_to_reportlab(stripped[2:].strip()), styles["BodyText"]))
        elif re.match(r"^\d+\.\s+", stripped):
            story.append(Paragraph(_markdown_inline_to_reportlab(stripped), styles["BodyText"]))
        elif stripped.startswith("|") and stripped.endswith("|"):
            # Avoid rendering markdown table syntax in the PDF interpretation; proper tables are already included above.
            continue
        else:
            story.append(Paragraph(_markdown_inline_to_reportlab(_shorten(stripped, 1000)), styles["BodyText"]))

    if plots:
        story.append(PageBreak())
        story.append(Paragraph("Static figure snapshots", styles["Heading2"]))
        fig_dir = output_path / "pdf_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in list(plots.items())[:8]:
            try:
                img_path = fig_dir / f"{name}.png"
                fig.write_image(str(img_path), width=1200, height=700, scale=2)
                story.append(Paragraph(_markdown_inline_to_reportlab(name.replace("_", " ").title()), styles["Heading3"]))
                story.append(Image(str(img_path), width=24 * cm, height=14 * cm))
                story.append(Spacer(1, 10))
            except Exception:
                continue

    def _footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(page_size[0] / 2, 0.45 * cm, "© 2026 AU P. Vajeeston · BatterySense AI · Free for non-commercial use; commercial use requires written permission.")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return pdf_path
