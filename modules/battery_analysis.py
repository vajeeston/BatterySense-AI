"""Battery/cell cycling analysis functions.

The functions in this module calculate metrics from data. AI report generation is
kept separate so that the model receives summarized results, not raw datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from modules.data_cleaning import analysis_ready_dataframe


@dataclass
class AnalysisSettings:
    capacity_drop_threshold_pct: float = 10.0
    ce_low_threshold_pct: float = 80.0
    ce_high_threshold_pct: float = 105.0
    cycle_life_retention_threshold_pct: float = 80.0


def _mapped(mapping: dict[str, str | None], key: str) -> str | None:
    value = mapping.get(key)
    return str(value) if value not in (None, "") else None


def _group_column(df: pd.DataFrame, mapping: dict[str, str | None]) -> str | None:
    for key in ("cell_name", "sample_name"):
        column = _mapped(mapping, key)
        if column and column in df.columns:
            return column
    if "source_file" in df.columns:
        return "source_file"
    return None


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _linear_slope(x: pd.Series, y: pd.Series) -> float | None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2:
        return None
    try:
        slope, _intercept = np.polyfit(valid["x"].astype(float), valid["y"].astype(float), 1)
        return float(slope)
    except Exception:  # noqa: BLE001
        return None


def add_derived_metrics(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Add calculated CE and retention columns when possible."""
    data = df.copy()
    group_col = _group_column(data, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    charge_col = _mapped(mapping, "charge_capacity")
    discharge_col = _mapped(mapping, "discharge_capacity")
    ce_col = _mapped(mapping, "coulombic_efficiency")
    cc_col = _mapped(mapping, "cc_capacity")
    cv_col = _mapped(mapping, "cv_capacity")

    sort_cols = [col for col in [group_col, cycle_col] if col and col in data.columns]
    if sort_cols:
        data = data.sort_values(sort_cols).reset_index(drop=True)

    if ce_col is None and charge_col and discharge_col and charge_col in data.columns and discharge_col in data.columns:
        denom = data[charge_col].replace(0, np.nan)
        data["calculated_coulombic_efficiency_pct"] = (data[discharge_col] / denom) * 100
        ce_col = "calculated_coulombic_efficiency_pct"

    if discharge_col and discharge_col in data.columns:
        if group_col and group_col in data.columns:
            first_discharge = data.groupby(group_col, dropna=False)[discharge_col].transform("first")
        else:
            first_discharge = pd.Series(data[discharge_col].iloc[0], index=data.index)
        data["capacity_retention_pct"] = (data[discharge_col] / first_discharge.replace(0, np.nan)) * 100

    if cc_col and cv_col and cc_col in data.columns and cv_col in data.columns:
        total = data[cc_col] + data[cv_col]
        data["cc_fraction_pct"] = data[cc_col] / total.replace(0, np.nan) * 100
        data["cv_fraction_pct"] = data[cv_col] / total.replace(0, np.nan) * 100

    return data


def _estimate_cycle_life(
    cycles: pd.Series,
    retention: pd.Series,
    threshold_pct: float,
) -> float | None:
    valid = pd.DataFrame({"cycle": cycles, "retention": retention}).dropna()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None

    crossed = valid[valid["retention"] <= threshold_pct]
    if not crossed.empty:
        return _safe_float(crossed.sort_values("cycle").iloc[0]["cycle"])

    if len(valid) < 3 or valid["cycle"].nunique() < 2:
        return None

    slope = _linear_slope(valid["cycle"], valid["retention"])
    if slope is None or slope >= 0:
        return None

    intercept = float(valid["retention"].mean() - slope * valid["cycle"].mean())
    estimated_cycle = (threshold_pct - intercept) / slope
    last_cycle = float(valid["cycle"].max())
    if estimated_cycle <= last_cycle:
        return None
    return float(estimated_cycle)


def summarize_cells(
    data: pd.DataFrame,
    mapping: dict[str, str | None],
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Calculate cell/sample-level metrics."""
    group_col = _group_column(data, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    charge_col = _mapped(mapping, "charge_capacity")
    discharge_col = _mapped(mapping, "discharge_capacity")
    ce_col = _mapped(mapping, "coulombic_efficiency") or "calculated_coulombic_efficiency_pct"

    if group_col and group_col in data.columns:
        groups = data.groupby(group_col, dropna=False)
    else:
        data = data.copy()
        data["analysis_group"] = "All data"
        group_col = "analysis_group"
        groups = data.groupby(group_col, dropna=False)

    rows: list[dict[str, Any]] = []
    for group_name, group in groups:
        group_sorted = group.sort_values(cycle_col) if cycle_col and cycle_col in group.columns else group.copy()
        first_row = group_sorted.iloc[0] if not group_sorted.empty else pd.Series(dtype="object")
        last_row = group_sorted.iloc[-1] if not group_sorted.empty else pd.Series(dtype="object")

        first_charge = _safe_float(first_row.get(charge_col)) if charge_col else None
        first_discharge = _safe_float(first_row.get(discharge_col)) if discharge_col else None
        first_ce = _safe_float(first_row.get(ce_col)) if ce_col in group_sorted.columns else None
        last_discharge = _safe_float(last_row.get(discharge_col)) if discharge_col else None
        max_discharge = _safe_float(group_sorted[discharge_col].max()) if discharge_col and discharge_col in group_sorted.columns else None
        avg_ce = _safe_float(group_sorted[ce_col].dropna().mean()) if ce_col in group_sorted.columns else None
        last_retention = _safe_float(last_row.get("capacity_retention_pct")) if "capacity_retention_pct" in group_sorted.columns else None

        degradation_rate = None
        cycle_life = None
        if cycle_col and cycle_col in group_sorted.columns and "capacity_retention_pct" in group_sorted.columns:
            degradation_rate = _linear_slope(group_sorted[cycle_col], group_sorted["capacity_retention_pct"])
            cycle_life = _estimate_cycle_life(
                group_sorted[cycle_col],
                group_sorted["capacity_retention_pct"],
                threshold_pct=settings.cycle_life_retention_threshold_pct,
            )

        max_single_cycle_drop = None
        abnormal_drop_count = 0
        if cycle_col and cycle_col in group_sorted.columns and "capacity_retention_pct" in group_sorted.columns:
            retention_diff = group_sorted["capacity_retention_pct"].diff()
            drops = retention_diff[retention_diff <= -abs(settings.capacity_drop_threshold_pct)]
            abnormal_drop_count = int(len(drops))
            max_single_cycle_drop = float(abs(retention_diff.min())) if retention_diff.notna().any() else None

        rows.append(
            {
                "cell_or_sample": str(group_name),
                "n_points": int(len(group_sorted)),
                "first_cycle": _safe_float(first_row.get(cycle_col)) if cycle_col else None,
                "last_cycle": _safe_float(last_row.get(cycle_col)) if cycle_col else None,
                "first_charge_capacity": first_charge,
                "first_discharge_capacity": first_discharge,
                "first_coulombic_efficiency_pct": first_ce,
                "max_discharge_capacity": max_discharge,
                "last_discharge_capacity": last_discharge,
                "last_capacity_retention_pct": last_retention,
                "degradation_rate_pct_per_cycle": degradation_rate,
                "average_coulombic_efficiency_pct": avg_ce,
                "cycle_life_estimate_to_80pct": cycle_life,
                "abnormal_capacity_drop_count": abnormal_drop_count,
                "max_single_cycle_drop_pct": max_single_cycle_drop,
            }
        )

    metrics = pd.DataFrame(rows)
    return metrics.sort_values(
        by=["last_capacity_retention_pct", "average_coulombic_efficiency_pct"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def detect_anomalies(
    data: pd.DataFrame,
    mapping: dict[str, str | None],
    settings: AnalysisSettings,
) -> pd.DataFrame:
    """Flag suspicious values without deleting them."""
    group_col = _group_column(data, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    charge_col = _mapped(mapping, "charge_capacity")
    discharge_col = _mapped(mapping, "discharge_capacity")
    ce_col = _mapped(mapping, "coulombic_efficiency") or "calculated_coulombic_efficiency_pct"

    anomalies: list[dict[str, Any]] = []

    if group_col and group_col in data.columns:
        groups = data.groupby(group_col, dropna=False)
    else:
        data = data.copy()
        data["analysis_group"] = "All data"
        group_col = "analysis_group"
        groups = data.groupby(group_col, dropna=False)

    for group_name, group in groups:
        group_sorted = group.sort_values(cycle_col) if cycle_col and cycle_col in group.columns else group.copy()

        # Missing cycle numbers / gaps.
        if cycle_col and cycle_col in group_sorted.columns:
            cycles = group_sorted[cycle_col].dropna().astype(float).sort_values()
            if len(cycles) >= 2:
                diffs = cycles.diff().dropna()
                gaps = diffs[diffs > 1]
                for idx, gap in gaps.items():
                    anomalies.append(
                        {
                            "cell_or_sample": str(group_name),
                            "cycle": float(cycles.loc[idx]),
                            "type": "missing_cycle_gap",
                            "severity": "medium",
                            "details": f"Gap of {gap:.0f} cycles before cycle {cycles.loc[idx]:.0f}.",
                        }
                    )

        # Negative capacities.
        for col_name, label in [(charge_col, "charge"), (discharge_col, "discharge")]:
            if col_name and col_name in group_sorted.columns:
                bad = group_sorted[group_sorted[col_name] < 0]
                for _, row in bad.iterrows():
                    anomalies.append(
                        {
                            "cell_or_sample": str(group_name),
                            "cycle": _safe_float(row.get(cycle_col)) if cycle_col else None,
                            "type": "negative_capacity",
                            "severity": "high",
                            "details": f"Negative {label} capacity in column '{col_name}'.",
                        }
                    )

        # Abnormal Coulombic efficiency.
        if ce_col in group_sorted.columns:
            ce_bad = group_sorted[
                (group_sorted[ce_col] < settings.ce_low_threshold_pct)
                | (group_sorted[ce_col] > settings.ce_high_threshold_pct)
            ]
            for _, row in ce_bad.iterrows():
                anomalies.append(
                    {
                        "cell_or_sample": str(group_name),
                        "cycle": _safe_float(row.get(cycle_col)) if cycle_col else None,
                        "type": "coulombic_efficiency_out_of_range",
                        "severity": "medium",
                        "details": (
                            f"CE={row.get(ce_col):.2f}% outside "
                            f"{settings.ce_low_threshold_pct:.1f}-{settings.ce_high_threshold_pct:.1f}%."
                        ) if pd.notna(row.get(ce_col)) else "CE is missing or invalid.",
                    }
                )

        # Abnormal capacity drops based on retention.
        if cycle_col and cycle_col in group_sorted.columns and "capacity_retention_pct" in group_sorted.columns:
            group_sorted = group_sorted.copy()
            group_sorted["retention_drop_pct"] = group_sorted["capacity_retention_pct"].diff()
            drops = group_sorted[group_sorted["retention_drop_pct"] <= -abs(settings.capacity_drop_threshold_pct)]
            for _, row in drops.iterrows():
                anomalies.append(
                    {
                        "cell_or_sample": str(group_name),
                        "cycle": _safe_float(row.get(cycle_col)),
                        "type": "abnormal_capacity_drop",
                        "severity": "high",
                        "details": f"Retention dropped by {abs(row['retention_drop_pct']):.2f}% from previous point.",
                    }
                )

    return pd.DataFrame(anomalies)


def _iqr_outlier_labels(series: pd.Series) -> set[str]:
    values = series.dropna()
    if len(values) < 4:
        return set()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return set()
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return set(series[(series < lower) | (series > upper)].index.astype(str))


def summarize_overall(metrics: pd.DataFrame, anomalies: pd.DataFrame, data: pd.DataFrame) -> dict[str, Any]:
    """Create high-level summary values for UI cards and AI context."""
    summary: dict[str, Any] = {
        "n_rows_analyzed": int(len(data)),
        "n_cells_or_samples": int(len(metrics)),
        "n_anomalies": int(len(anomalies)),
        "best_performing_cell": None,
        "worst_performing_cell": None,
        "mean_last_capacity_retention_pct": None,
        "mean_degradation_rate_pct_per_cycle": None,
    }

    if not metrics.empty:
        if "last_capacity_retention_pct" in metrics.columns:
            ranked = metrics.dropna(subset=["last_capacity_retention_pct"])
            if not ranked.empty:
                best = ranked.sort_values("last_capacity_retention_pct", ascending=False).iloc[0]
                worst = ranked.sort_values("last_capacity_retention_pct", ascending=True).iloc[0]
                summary["best_performing_cell"] = str(best["cell_or_sample"])
                summary["worst_performing_cell"] = str(worst["cell_or_sample"])
                summary["mean_last_capacity_retention_pct"] = _safe_float(ranked["last_capacity_retention_pct"].mean())

        if "degradation_rate_pct_per_cycle" in metrics.columns:
            summary["mean_degradation_rate_pct_per_cycle"] = _safe_float(metrics["degradation_rate_pct_per_cycle"].mean())

    return summary


def add_cell_outlier_anomalies(metrics: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """Add cell-level outlier flags based on final retention and degradation."""
    extra: list[dict[str, Any]] = []
    if metrics.empty:
        return anomalies

    indexed = metrics.set_index("cell_or_sample", drop=False)
    for col, label in [
        ("last_capacity_retention_pct", "final retention"),
        ("degradation_rate_pct_per_cycle", "degradation rate"),
    ]:
        if col not in indexed.columns:
            continue
        outlier_cells = _iqr_outlier_labels(indexed[col])
        for cell in outlier_cells:
            value = indexed.loc[cell, col]
            extra.append(
                {
                    "cell_or_sample": str(cell),
                    "cycle": None,
                    "type": "cell_level_outlier",
                    "severity": "medium",
                    "details": f"Cell/sample is an IQR outlier for {label}: {value:.3f}.",
                }
            )

    if not extra:
        return anomalies
    return pd.concat([anomalies, pd.DataFrame(extra)], ignore_index=True)


def _quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize protocol segments and excluded rows."""
    out: dict[str, Any] = {
        "rows_before_quality_filter": int(len(df)),
        "rows_after_quality_filter": int(df.get("analysis_include", pd.Series(True, index=df.index)).sum()) if len(df) else 0,
        "excluded_rows": 0,
        "segment_counts": {},
        "quality_flag_counts": {},
    }
    if "analysis_include" in df.columns:
        out["excluded_rows"] = int((~df["analysis_include"].fillna(False)).sum())
    if "protocol_segment" in df.columns:
        out["segment_counts"] = {str(k): int(v) for k, v in df["protocol_segment"].value_counts(dropna=False).items()}
    if "quality_flag" in df.columns:
        exploded = df["quality_flag"].replace("", pd.NA).dropna().astype(str).str.split(";").explode()
        out["quality_flag_counts"] = {str(k): int(v) for k, v in exploded.value_counts().items()}
    return out


def run_battery_analysis(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    settings_dict: dict[str, Any] | None = None,
    include_segments: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full battery analysis workflow.

    ``df`` may contain quality/protocol columns from data_cleaning. By default,
    rows marked ``analysis_include=False`` are excluded from metrics and plots.
    """
    settings = AnalysisSettings(**(settings_dict or {}))
    processed_all = add_derived_metrics(df, mapping)
    analysis_df = analysis_ready_dataframe(processed_all, include_segments=include_segments)
    processed = add_derived_metrics(analysis_df, mapping)
    metrics = summarize_cells(processed, mapping, settings)
    anomalies = detect_anomalies(processed, mapping, settings)
    # Carry quality exclusions into anomaly/diagnostic table.
    if "analysis_include" in processed_all.columns and "quality_flag" in processed_all.columns:
        excluded = processed_all[~processed_all["analysis_include"].fillna(False)].copy()
        if not excluded.empty:
            group_col = _group_column(excluded, mapping)
            cycle_col = _mapped(mapping, "cycle_number")
            qrows = []
            for _, row in excluded.iterrows():
                qrows.append({
                    "cell_or_sample": str(row.get(group_col, "All data")) if group_col else "All data",
                    "cycle": _safe_float(row.get(cycle_col)) if cycle_col else None,
                    "type": row.get("quality_flag") or "excluded_quality_point",
                    "severity": "high" if "incomplete_terminal_cycle" in str(row.get("quality_flag", "")) else "medium",
                    "details": row.get("exclude_reason") or "Excluded by data-cleaning quality filter.",
                })
            anomalies = pd.concat([anomalies, pd.DataFrame(qrows)], ignore_index=True)
    anomalies = add_cell_outlier_anomalies(metrics, anomalies)
    summary = summarize_overall(metrics, anomalies, processed)
    summary.update(_quality_summary(processed_all))

    return {
        "processed_data": processed,
        "processed_data_all_rows": processed_all,
        "cell_metrics": metrics,
        "anomalies": anomalies,
        "summary": summary,
        "settings": settings.__dict__,
        "include_segments": include_segments,
    }
