"""PDF export utilities.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
The PDF export uses ReportLab for a reliable local export. Interactive Plotly
content remains in the HTML report; the PDF contains summary tables and, when
kaleido is installed, static snapshots of the figures.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _shorten(text: object, max_len: int = 90) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


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
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_path / f"battery_ai_report_{safe_timestamp}.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=1.2 * cm, leftMargin=1.2 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    story = [Paragraph(title, styles["Title"]), Paragraph(f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", styles["Normal"]), Spacer(1, 12)]

    story.append(Paragraph("Dataset summary", styles["Heading2"]))
    summary_rows = [
        ["Rows", dataset_profile.get("rows", "N/A")],
        ["Columns", dataset_profile.get("columns", "N/A")],
        ["Uploaded files", ", ".join(uploaded_file_names)],
        ["Cells/samples", analysis.get("summary", {}).get("n_cells_or_samples", "N/A")],
        ["Anomaly flags", analysis.get("summary", {}).get("n_anomalies", "N/A")],
        ["Excluded rows", analysis.get("summary", {}).get("excluded_rows", "N/A")],
    ]
    table = Table(summary_rows, colWidths=[4 * cm, 12 * cm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF3F8")), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([table, Spacer(1, 12)])

    metrics = analysis.get("cell_metrics", pd.DataFrame())
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        story.append(Paragraph("Key metrics", styles["Heading2"]))
        cols = [c for c in ["cell_or_sample", "n_points", "first_discharge_capacity", "last_capacity_retention_pct", "degradation_rate_pct_per_cycle", "average_coulombic_efficiency_pct"] if c in metrics.columns]
        table_data = [cols] + [[_shorten(v, 20) for v in row] for row in metrics[cols].head(25).values.tolist()]
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12395D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        story.extend([t, Spacer(1, 12)])

    story.append(PageBreak())
    story.append(Paragraph("Scientific interpretation", styles["Heading2"]))
    for line in (interpretation_text or "").splitlines():
        if not line.strip():
            story.append(Spacer(1, 6))
        elif line.startswith("#"):
            story.append(Paragraph(line.replace("#", "").strip(), styles["Heading3"]))
        else:
            story.append(Paragraph(_shorten(line, 900), styles["BodyText"]))

    # Optional static figure snapshots.
    if plots:
        story.append(PageBreak())
        story.append(Paragraph("Static figure snapshots", styles["Heading2"]))
        fig_dir = output_path / "pdf_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in list(plots.items())[:8]:
            try:
                img_path = fig_dir / f"{name}.png"
                fig.write_image(str(img_path), width=1000, height=600, scale=2)
                story.append(Paragraph(name.replace("_", " ").title(), styles["Heading3"]))
                story.append(Image(str(img_path), width=17 * cm, height=10.2 * cm))
                story.append(Spacer(1, 10))
            except Exception:
                continue

    doc.build(story)
    return pdf_path
