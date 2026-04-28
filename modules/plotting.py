"""Plotly figure generation for battery/cell analysis.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGO_PATH = PROJECT_ROOT / "assets" / "log_small.png"


def _mapped(mapping: dict[str, str | None], key: str) -> str | None:
    value = mapping.get(key)
    return str(value) if value not in (None, "") else None


def _group_column(df: pd.DataFrame, mapping: dict[str, str | None]) -> str:
    for key in ("cell_name", "sample_name"):
        column = _mapped(mapping, key)
        if column and column in df.columns:
            return column
    if "source_file" in df.columns:
        return "source_file"
    df["analysis_group"] = "All data"
    return "analysis_group"


def _logo_data_uri(logo_path: str | Path | None = None) -> str | None:
    path = Path(logo_path) if logo_path else DEFAULT_LOGO_PATH
    if not path.exists():
        return None
    suffix = path.suffix.lower().replace(".", "") or "png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def add_plot_logo(fig: go.Figure, logo_path: str | Path | None = None) -> go.Figure:
    """Add the small app logo to bottom-right of a Plotly figure."""
    uri = _logo_data_uri(logo_path)
    if not uri:
        return fig
    fig.add_layout_image(
        dict(
            source=uri,
            xref="paper",
            yref="paper",
            x=0.995,
            y=0.005,
            sizex=0.10,
            sizey=0.10,
            xanchor="right",
            yanchor="bottom",
            opacity=0.82,
            layer="above",
        )
    )
    fig.update_layout(margin=dict(r=30, b=40))
    return fig


def _finalize(fig: go.Figure, logo_path: str | Path | None = None) -> go.Figure:
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="Cell/sample",
        modebar_remove=["select2d", "lasso2d"],
    )
    return add_plot_logo(fig, logo_path=logo_path)


def _line_plot(df: pd.DataFrame, x: str | None, y: str | None, color: str, title: str, y_label: str, logo_path: str | Path | None = None) -> go.Figure | None:
    if not x or not y or x not in df.columns or y not in df.columns:
        return None
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return None
    fig = px.line(plot_df, x=x, y=y, color=color if color in plot_df.columns else None, markers=True, title=title, labels={x: "Cycle number", y: y_label, color: "Cell/sample"})
    return _finalize(fig, logo_path)


def generate_plots(data: pd.DataFrame, analysis: dict[str, Any], mapping: dict[str, str | None], logo_path: str | Path | None = None) -> dict[str, go.Figure]:
    df = data.copy()
    group_col = _group_column(df, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    charge_col = _mapped(mapping, "charge_capacity")
    discharge_col = _mapped(mapping, "discharge_capacity")
    ce_col = _mapped(mapping, "coulombic_efficiency")
    if (ce_col is None or ce_col not in df.columns) and "calculated_coulombic_efficiency_pct" in df.columns:
        ce_col = "calculated_coulombic_efficiency_pct"

    figures: dict[str, go.Figure] = {}
    for key, fig in {
        "charge_capacity_vs_cycle": _line_plot(df, cycle_col, charge_col, group_col, "Charge capacity vs cycle number", "Charge capacity", logo_path),
        "discharge_capacity_vs_cycle": _line_plot(df, cycle_col, discharge_col, group_col, "Discharge capacity vs cycle number", "Discharge capacity", logo_path),
        "coulombic_efficiency_vs_cycle": _line_plot(df, cycle_col, ce_col, group_col, "Coulombic efficiency vs cycle number", "Coulombic efficiency (%)", logo_path),
        "capacity_retention_vs_cycle": _line_plot(df, cycle_col, "capacity_retention_pct", group_col, "Capacity retention vs cycle number", "Capacity retention (%)", logo_path),
    }.items():
        if fig is not None:
            figures[key] = fig

    # Segment visualization if available.
    if cycle_col and discharge_col and "protocol_segment" in df.columns:
        seg_df = df.dropna(subset=[cycle_col, discharge_col]).copy()
        if not seg_df.empty:
            fig = px.scatter(seg_df, x=cycle_col, y=discharge_col, color="protocol_segment", symbol=group_col, title="Detected protocol segments", labels={cycle_col: "Cycle number", discharge_col: "Discharge capacity"})
            figures["protocol_segment_detection"] = _finalize(fig, logo_path)

    metrics = analysis.get("cell_metrics", pd.DataFrame())
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        bar_specs = [
            ("summary_final_retention_bar", "last_capacity_retention_pct", "Final capacity retention comparison", "Final retention (%)", False),
            ("degradation_rate_comparison", "degradation_rate_pct_per_cycle", "Degradation rate comparison", "Retention slope (% per cycle)", True),
            ("summary_average_ce_bar", "average_coulombic_efficiency_pct", "Average Coulombic efficiency comparison", "Average CE (%)", False),
        ]
        for name, col, title, ylabel, asc in bar_specs:
            if col in metrics.columns:
                fig = px.bar(metrics.sort_values(col, ascending=asc), x="cell_or_sample", y=col, title=title, labels={"cell_or_sample": "Cell/sample", col: ylabel})
                fig.update_layout(xaxis_tickangle=-45)
                figures[name] = _finalize(fig, logo_path)

    if "cc_fraction_pct" in df.columns and "cv_fraction_pct" in df.columns and cycle_col in df.columns:
        plot_df = df.dropna(subset=[cycle_col, "cc_fraction_pct", "cv_fraction_pct"]).copy()
        if not plot_df.empty:
            fig = go.Figure()
            for cell_name, group in plot_df.groupby(group_col, dropna=False):
                fig.add_trace(go.Scatter(x=group[cycle_col], y=group["cc_fraction_pct"], mode="lines+markers", name=f"{cell_name} CC %"))
                fig.add_trace(go.Scatter(x=group[cycle_col], y=group["cv_fraction_pct"], mode="lines+markers", name=f"{cell_name} CV %", line={"dash": "dash"}))
            fig.update_layout(title="CC/CV contribution to charge capacity", xaxis_title="Cycle number", yaxis_title="Contribution (%)")
            figures["cc_cv_contribution"] = _finalize(fig, logo_path)

    anomalies = analysis.get("anomalies", pd.DataFrame())
    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        anomaly_points = anomalies.dropna(subset=["cycle"]).copy() if "cycle" in anomalies.columns else pd.DataFrame()
        if not anomaly_points.empty:
            fig = px.scatter(anomaly_points, x="cycle", y="cell_or_sample", color="type", symbol="severity", title="Detected anomaly map", hover_data=["details"], labels={"cycle": "Cycle", "cell_or_sample": "Cell/sample", "type": "Anomaly type"})
            figures["anomaly_visualization"] = _finalize(fig, logo_path)

    return figures


def comparison_plot(data: pd.DataFrame, mapping: dict[str, str | None], selected_cells: list[str], y_metric: str = "capacity_retention_pct", logo_path: str | Path | None = None) -> go.Figure | None:
    df = data.copy()
    group_col = _group_column(df, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    if not cycle_col or y_metric not in df.columns or group_col not in df.columns:
        return None
    plot_df = df[df[group_col].astype(str).isin([str(x) for x in selected_cells])].dropna(subset=[cycle_col, y_metric])
    if plot_df.empty:
        return None
    fig = px.line(plot_df, x=cycle_col, y=y_metric, color=group_col, markers=True, title=f"One-to-one comparison: {y_metric}", labels={cycle_col: "Cycle", y_metric: y_metric, group_col: "Cell/sample"})
    return _finalize(fig, logo_path)


def group_comparison_table(metrics: pd.DataFrame, group_mapping: dict[str, str]) -> pd.DataFrame:
    if metrics.empty or not group_mapping:
        return pd.DataFrame()
    work = metrics.copy()
    work["comparison_group"] = work["cell_or_sample"].astype(str).map(group_mapping).fillna("Unassigned")
    num_cols = work.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return pd.DataFrame()
    grouped = work.groupby("comparison_group")[num_cols].agg(["count", "mean", "std", "min", "max"])
    grouped.columns = ["_".join(map(str, col)).strip("_") for col in grouped.columns]
    return grouped.reset_index()
