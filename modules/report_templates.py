"""Report layout templates.

Templates are intentionally data-driven. A report can be single-page,
multi-page, or dashboard style depending on the user's choices and the number of
plots/tables available.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportSection:
    title: str
    kind: str
    content_key: str | None = None
    description: str = ""


@dataclass
class ReportTemplate:
    name: str
    description: str
    sections: list[ReportSection] = field(default_factory=list)
    plots_per_page: int = 4


def get_report_templates() -> dict[str, ReportTemplate]:
    return {
        "single_page": ReportTemplate(
            name="single_page",
            description="Compact one-page report for quick screening.",
            plots_per_page=4,
            sections=[
                ReportSection("Executive summary", "text", "interpretation"),
                ReportSection("Key metrics", "table", "cell_metrics"),
                ReportSection("Main plots", "plots", "all"),
            ],
        ),
        "multipage": ReportTemplate(
            name="multipage",
            description="Detailed report with separate pages/sections for overview, metrics, plots, and discussion.",
            plots_per_page=3,
            sections=[
                ReportSection("Dataset overview", "profile", "dataset_profile"),
                ReportSection("Quality and protocol segmentation", "table", "quality_summary"),
                ReportSection("Key metrics", "table", "cell_metrics"),
                ReportSection("Anomalies", "table", "anomalies"),
                ReportSection("Interactive plots", "plots", "all"),
                ReportSection("Scientific interpretation", "text", "interpretation"),
            ],
        ),
        "dashboard": ReportTemplate(
            name="dashboard",
            description="Dashboard-style report with KPI cards and many interactive plots.",
            plots_per_page=6,
            sections=[
                ReportSection("KPI dashboard", "kpis", "summary"),
                ReportSection("Plots", "plots", "all"),
                ReportSection("Interpretation", "text", "interpretation"),
            ],
        ),
    }


def build_plot_pages(plot_names: list[str], plots_per_page: int = 4) -> list[list[str]]:
    plots_per_page = max(1, int(plots_per_page))
    return [plot_names[i:i + plots_per_page] for i in range(0, len(plot_names), plots_per_page)]
