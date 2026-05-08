"""Battery/cell cycling analysis functions for BatterySense AI.

The functions in this module calculate metrics locally from the uploaded data.
AI report generation receives summarized results only, not raw cycle-level data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from modules.data_cleaning import analysis_ready_dataframe


@dataclass
class AnalysisSettings:
    battery_cell_type: str = "full_cell"
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
    valid = pd.DataFrame({"x": x, "y": y}).dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2:
        return None
    try:
        slope, _intercept = np.polyfit(valid["x"].astype(float), valid["y"].astype(float), 1)
        return float(slope)
    except Exception:  # noqa: BLE001
        return None


def _to_numeric_series(data: pd.DataFrame, column: str | None) -> pd.Series | None:
    if not column or column not in data.columns:
        return None
    return pd.to_numeric(data[column], errors="coerce")


def _parse_duration_to_hours(value: Any) -> float | None:
    """Parse common tester duration formats to hours."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        hours = value.total_seconds() / 3600.0
        return float(hours) if hours > 0 else None
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        if val <= 0:
            return None
        # Most tester exports store duration as seconds. Small values are often already hours.
        return val / 3600.0 if val > 48 else val

    text = str(value).strip().lower()
    if not text:
        return None
    try:
        td = pd.to_timedelta(text)
        if pd.notna(td):
            hours = td.total_seconds() / 3600.0
            return float(hours) if hours > 0 else None
    except Exception:  # noqa: BLE001
        pass

    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = [float(x) for x in parts]
            hours = h + m / 60.0 + s / 3600.0
            return hours if hours > 0 else None
        if len(parts) == 2:
            m, s = [float(x) for x in parts]
            hours = m / 60.0 + s / 3600.0
            return hours if hours > 0 else None
    except Exception:  # noqa: BLE001
        pass

    match = re.findall(r"([0-9]*\.?[0-9]+)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)", text)
    if match:
        seconds = 0.0
        for number, unit in match:
            value_f = float(number)
            if unit.startswith("h"):
                seconds += value_f * 3600.0
            elif unit.startswith("m"):
                seconds += value_f * 60.0
            else:
                seconds += value_f
        hours = seconds / 3600.0
        return hours if hours > 0 else None
    return None


def _duration_column_to_hours(data: pd.DataFrame, column: str | None) -> pd.Series | None:
    if not column or column not in data.columns:
        return None
    return data[column].apply(_parse_duration_to_hours)




def _cell_type_label(battery_cell_type: str) -> str:
    labels = {
        "full_cell": "Full cell",
        "half_cell": "Generic half cell",
        "anode_half_cell": "Anode half-cell",
        "cathode_half_cell": "Cathode half-cell",
    }
    return labels.get(battery_cell_type, "Full cell")


def _ce_inputs(
    charge: pd.Series,
    discharge: pd.Series,
    battery_cell_type: str,
) -> tuple[pd.Series, pd.Series, str, str, str]:
    """Return CE numerator, denominator, reversible label, denominator label, and formula note.

    The app keeps the raw mapped columns as charge/lithiation and discharge/delithiation,
    then chooses the electrochemical convention based on cell type.

    Full cell and cathode half-cell usually use discharge/charge. Anode half-cell
    usually reports reversible CE as delithiation/lithiation; in common tester
    exports this corresponds to charge/discharge. Generic half-cell keeps the
    previous neutral convention of discharge/charge unless anode/cathode is selected.
    """
    if battery_cell_type == "anode_half_cell":
        return (
            charge,
            discharge,
            "delithiation_capacity",
            "lithiation_capacity",
            "Anode half-cell: delithiation/lithiation × 100; common export assumption: charge/discharge × 100",
        )
    if battery_cell_type == "cathode_half_cell":
        return (
            discharge,
            charge,
            "lithiation_discharge_capacity",
            "delithiation_charge_capacity",
            "Cathode half-cell: lithiation/delithiation × 100; common export assumption: discharge/charge × 100",
        )
    if battery_cell_type == "half_cell":
        return (
            discharge,
            charge,
            "reversible_capacity",
            "inserted_capacity",
            "Generic half-cell: reversible/inserted capacity × 100; default neutral assumption: discharge/charge × 100",
        )
    return (
        discharge,
        charge,
        "discharge_capacity",
        "charge_capacity",
        "Full cell: discharge/charge × 100",
    )


def _first_positive_by_group(values: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """Return each row's first finite positive value within its cell/sample group."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    if groups is None:
        valid = numeric[(numeric > 0) & numeric.notna()]
        first = valid.iloc[0] if not valid.empty else np.nan
        out.loc[:] = first
        return out
    for _, idx in groups.groupby(groups, dropna=False).groups.items():
        idx = list(idx)
        valid = numeric.loc[idx][(numeric.loc[idx] > 0) & numeric.loc[idx].notna()]
        out.loc[idx] = valid.iloc[0] if not valid.empty else np.nan
    return out


def _max_stabilized_by_group(values: pd.Series, data: pd.DataFrame, groups: pd.Series | None = None) -> pd.Series:
    """Return each row's stabilized maximum reference capacity.

    Preference order:
    1. maximum finite positive capacity in rows tagged as protocol_segment == 'cycling';
    2. maximum finite positive capacity in all valid rows for that cell/sample.

    This is useful for real battery cycling data where early formation cycles are not
    always the most representative capacity baseline.
    """
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=values.index, dtype="float64")

    def choose_baseline(idx: list[int]) -> float:
        idx = list(idx)
        valid_all = numeric.loc[idx]
        valid_all = valid_all[(valid_all > 0) & valid_all.notna()]
        if "protocol_segment" in data.columns:
            segment = data.loc[idx, "protocol_segment"].astype(str).str.lower()
            cycling_idx = segment[segment.eq("cycling")].index
            valid_cycling = numeric.loc[cycling_idx]
            valid_cycling = valid_cycling[(valid_cycling > 0) & valid_cycling.notna()]
            if not valid_cycling.empty:
                return float(valid_cycling.max())
        if not valid_all.empty:
            return float(valid_all.max())
        return np.nan

    if groups is None:
        out.loc[:] = choose_baseline(list(values.index))
        return out
    for _, idx in groups.groupby(groups, dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx] = choose_baseline(idx)
    return out

def add_derived_metrics(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    battery_cell_type: str = "full_cell",
) -> pd.DataFrame:
    """Add locally calculated CE, irreversible capacity, accumulated CE, SOH, and C-rate.

    BatterySense AI calculates CE and related metrics locally from the core
    experimental columns: cycle, charge/lithiation capacity, and
    discharge/delithiation capacity. Uploaded CE columns are preserved as
    ``uploaded_coulombic_efficiency_pct`` but not used as the primary metric.
    """
    data = df.copy()
    group_col = _group_column(data, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    charge_col = _mapped(mapping, "charge_capacity")
    discharge_col = _mapped(mapping, "discharge_capacity")
    uploaded_ce_col = _mapped(mapping, "coulombic_efficiency")
    cc_col = _mapped(mapping, "cc_capacity")
    cv_col = _mapped(mapping, "cv_capacity")
    charge_time_col = _mapped(mapping, "charge_time") or _mapped(mapping, "time")
    discharge_time_col = _mapped(mapping, "discharge_time")

    sort_cols = [col for col in [group_col, cycle_col] if col and col in data.columns]
    if sort_cols:
        data = data.sort_values(sort_cols).reset_index(drop=True)

    if uploaded_ce_col and uploaded_ce_col in data.columns:
        data["uploaded_coulombic_efficiency_pct"] = pd.to_numeric(data[uploaded_ce_col], errors="coerce")

    charge = _to_numeric_series(data, charge_col)
    discharge = _to_numeric_series(data, discharge_col)
    if charge is not None:
        data["charge_or_lithiation_capacity"] = charge
    if discharge is not None:
        data["discharge_or_delithiation_capacity"] = discharge

    if charge is not None and discharge is not None:
        ce_numerator, ce_denominator, reversible_label, denominator_label, formula_note = _ce_inputs(charge, discharge, battery_cell_type)
        ce = (ce_numerator / ce_denominator.replace(0, np.nan)) * 100.0
        irreversible = ce_denominator - ce_numerator

        data["ce_numerator_capacity"] = ce_numerator
        data["ce_denominator_capacity"] = ce_denominator
        data["reversible_capacity_for_ce"] = ce_numerator
        data["inserted_capacity_for_ce"] = ce_denominator
        data["ce_reversible_capacity_label"] = reversible_label
        data["ce_denominator_capacity_label"] = denominator_label
        data["calculated_coulombic_efficiency_pct"] = ce
        data["CE(%)"] = ce
        data["100-CE"] = 100.0 - ce
        data["irreversible_capacity"] = irreversible
        data["battery_cell_type"] = _cell_type_label(battery_cell_type)
        data["ce_calculation_basis"] = formula_note

        if group_col and group_col in data.columns:
            data["Acc_irreversible_capacity_u_mAh_per_cm2"] = irreversible.groupby(data[group_col], dropna=False).cumsum()
            cum_numerator = ce_numerator.groupby(data[group_col], dropna=False).cumsum()
            cum_denominator = ce_denominator.groupby(data[group_col], dropna=False).cumsum()
            data["Accumulated CE"] = (cum_numerator / cum_denominator.replace(0, np.nan)) * 100.0
            data["CE_20_pt_AVG"] = data.groupby(group_col, dropna=False)["calculated_coulombic_efficiency_pct"].transform(
                lambda s: s.rolling(window=20, min_periods=1).mean()
            )
        else:
            data["Acc_irreversible_capacity_u_mAh_per_cm2"] = irreversible.cumsum()
            data["Accumulated CE"] = (ce_numerator.cumsum() / ce_denominator.cumsum().replace(0, np.nan)) * 100.0
            data["CE_20_pt_AVG"] = data["calculated_coulombic_efficiency_pct"].rolling(window=20, min_periods=1).mean()

    # Capacity-retention baseline handling.
    # Use the electrochemically reversible/output capacity when available:
    # full cell = discharge capacity; anode half-cell = delithiation capacity
    # under the chosen convention; cathode half-cell = lithiation/discharge capacity.
    retention_capacity = None
    if "reversible_capacity_for_ce" in data.columns:
        retention_capacity = pd.to_numeric(data["reversible_capacity_for_ce"], errors="coerce")
    elif discharge is not None:
        retention_capacity = discharge

    if retention_capacity is not None:
        groups_for_ref = data[group_col] if group_col and group_col in data.columns else None
        first_reference = _first_positive_by_group(retention_capacity, groups_for_ref)
        max_stabilized_reference = _max_stabilized_by_group(retention_capacity, data, groups_for_ref)
        data["retention_capacity"] = retention_capacity
        data["retention_first_reference_capacity"] = first_reference
        data["retention_max_stabilized_reference_capacity"] = max_stabilized_reference
        data["capacity_retention_from_first_pct"] = (retention_capacity / first_reference.replace(0, np.nan)) * 100.0
        data["capacity_retention_from_max_pct"] = (retention_capacity / max_stabilized_reference.replace(0, np.nan)) * 100.0
        # Main retention/SOH uses the stabilized maximum baseline, while first-cycle
        # retention is kept for transparent comparison in figures and tables.
        data["capacity_retention_pct"] = data["capacity_retention_from_max_pct"]
        data["SOH_pct"] = data["capacity_retention_pct"]

    charge_hours = _duration_column_to_hours(data, charge_time_col)
    discharge_hours = _duration_column_to_hours(data, discharge_time_col)
    if charge_hours is not None:
        data["charge_time_h"] = charge_hours
        data["estimated_charge_c_rate_from_time"] = 1.0 / charge_hours.replace(0, np.nan)
    if discharge_hours is not None:
        data["discharge_time_h"] = discharge_hours
        data["estimated_discharge_c_rate_from_time"] = 1.0 / discharge_hours.replace(0, np.nan)
    if charge_hours is not None or discharge_hours is not None:
        basis = discharge_hours if discharge_hours is not None else charge_hours
        data["estimated_c_rate_from_time"] = 1.0 / basis.replace(0, np.nan)

    if cc_col and cv_col and cc_col in data.columns and cv_col in data.columns:
        cc = pd.to_numeric(data[cc_col], errors="coerce")
        cv = pd.to_numeric(data[cv_col], errors="coerce")
        total = cc + cv
        data["cc_fraction_pct"] = cc / total.replace(0, np.nan) * 100
        data["cv_fraction_pct"] = cv / total.replace(0, np.nan) * 100

    return data


def _estimate_cycle_life(cycles: pd.Series, retention: pd.Series, threshold_pct: float) -> float | None:
    valid = pd.DataFrame({"cycle": cycles, "retention": retention}).dropna().replace([np.inf, -np.inf], np.nan).dropna()
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


def summarize_cells(data: pd.DataFrame, mapping: dict[str, str | None], settings: AnalysisSettings) -> pd.DataFrame:
    group_col = _group_column(data, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    charge_col = _mapped(mapping, "charge_capacity")
    discharge_col = _mapped(mapping, "discharge_capacity")
    ce_col = "calculated_coulombic_efficiency_pct"

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
        if group_sorted.empty:
            continue
        first_row = group_sorted.iloc[0]
        last_row = group_sorted.iloc[-1]
        first_charge = _safe_float(first_row.get(charge_col)) if charge_col else None
        first_discharge = _safe_float(first_row.get(discharge_col)) if discharge_col else None
        first_ce = _safe_float(first_row.get(ce_col)) if ce_col in group_sorted.columns else None
        last_discharge = _safe_float(last_row.get(discharge_col)) if discharge_col else None
        retention_capacity_col = "retention_capacity" if "retention_capacity" in group_sorted.columns else discharge_col
        last_retention_capacity = _safe_float(last_row.get(retention_capacity_col)) if retention_capacity_col else None
        max_retention_capacity = _safe_float(pd.to_numeric(group_sorted[retention_capacity_col], errors="coerce").max()) if retention_capacity_col and retention_capacity_col in group_sorted.columns else None
        first_ref_capacity = _safe_float(first_row.get("retention_first_reference_capacity")) if "retention_first_reference_capacity" in group_sorted.columns else first_discharge
        max_ref_capacity = _safe_float(first_row.get("retention_max_stabilized_reference_capacity")) if "retention_max_stabilized_reference_capacity" in group_sorted.columns else max_retention_capacity
        max_charge = _safe_float(pd.to_numeric(group_sorted[charge_col], errors="coerce").max()) if charge_col and charge_col in group_sorted.columns else None
        max_discharge = _safe_float(pd.to_numeric(group_sorted[discharge_col], errors="coerce").max()) if discharge_col and discharge_col in group_sorted.columns else None
        avg_ce = _safe_float(group_sorted[ce_col].dropna().mean()) if ce_col in group_sorted.columns else None
        avg_ce20 = _safe_float(group_sorted["CE_20_pt_AVG"].dropna().mean()) if "CE_20_pt_AVG" in group_sorted.columns else None
        last_acc_ce = _safe_float(group_sorted["Accumulated CE"].dropna().iloc[-1]) if "Accumulated CE" in group_sorted.columns and not group_sorted["Accumulated CE"].dropna().empty else None
        total_irrev = _safe_float(group_sorted["Acc_irreversible_capacity_u_mAh_per_cm2"].dropna().iloc[-1]) if "Acc_irreversible_capacity_u_mAh_per_cm2" in group_sorted.columns and not group_sorted["Acc_irreversible_capacity_u_mAh_per_cm2"].dropna().empty else None
        avg_est_c_rate = _safe_float(group_sorted["estimated_c_rate_from_time"].dropna().mean()) if "estimated_c_rate_from_time" in group_sorted.columns else None
        last_retention = _safe_float(last_row.get("capacity_retention_pct")) if "capacity_retention_pct" in group_sorted.columns else None
        last_retention_first = _safe_float(last_row.get("capacity_retention_from_first_pct")) if "capacity_retention_from_first_pct" in group_sorted.columns else None
        last_retention_max = _safe_float(last_row.get("capacity_retention_from_max_pct")) if "capacity_retention_from_max_pct" in group_sorted.columns else None
        last_ce = _safe_float(last_row.get(ce_col)) if ce_col in group_sorted.columns else None

        degradation_rate = None
        cycle_life = None
        if cycle_col and cycle_col in group_sorted.columns and "capacity_retention_pct" in group_sorted.columns:
            degradation_rate = _linear_slope(group_sorted[cycle_col], group_sorted["capacity_retention_pct"])
            cycle_life = _estimate_cycle_life(group_sorted[cycle_col], group_sorted["capacity_retention_pct"], settings.cycle_life_retention_threshold_pct)

        max_single_cycle_drop = None
        abnormal_drop_count = 0
        if cycle_col and cycle_col in group_sorted.columns and "capacity_retention_pct" in group_sorted.columns:
            retention_diff = group_sorted["capacity_retention_pct"].diff()
            drops = retention_diff[retention_diff <= -abs(settings.capacity_drop_threshold_pct)]
            abnormal_drop_count = int(len(drops))
            max_single_cycle_drop = float(abs(retention_diff.min())) if retention_diff.notna().any() else None

        rows.append({
            "cell_or_sample": str(group_name),
            "cell_type": _cell_type_label(settings.battery_cell_type),
            "n_points": int(len(group_sorted)),
            "first_cycle": _safe_float(first_row.get(cycle_col)) if cycle_col else None,
            "last_cycle": _safe_float(last_row.get(cycle_col)) if cycle_col else None,
            "first_charge_or_lithiation_capacity": first_charge,
            "first_discharge_or_delithiation_capacity": first_discharge,
            "first_charge_capacity": first_charge,
            "first_discharge_capacity": first_discharge,
            "first_coulombic_efficiency_pct": first_ce,
            "last_coulombic_efficiency_pct": last_ce,
            "retention_first_reference_capacity": first_ref_capacity,
            "retention_max_stabilized_reference_capacity": max_ref_capacity,
            "max_retention_capacity": max_retention_capacity,
            "max_charge_or_lithiation_capacity": max_charge,
            "max_discharge_or_delithiation_capacity": max_discharge,
            "last_retention_capacity": last_retention_capacity,
            "max_discharge_capacity": max_discharge,
            "last_discharge_capacity": last_discharge,
            "last_capacity_retention_from_first_pct": last_retention_first,
            "last_capacity_retention_from_max_pct": last_retention_max,
            "last_capacity_retention_pct": last_retention,
            "degradation_rate_pct_per_cycle": degradation_rate,
            "average_coulombic_efficiency_pct": avg_ce,
            "average_ce_20_point_rolling_pct": avg_ce20,
            "last_accumulated_ce_pct": last_acc_ce,
            "total_accumulated_irreversible_capacity": total_irrev,
            "average_estimated_c_rate_from_time": avg_est_c_rate,
            "cycle_life_estimate_to_80pct": cycle_life,
            "abnormal_capacity_drop_count": abnormal_drop_count,
            "max_single_cycle_drop_pct": max_single_cycle_drop,
        })

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    return metrics.sort_values(
        by=["last_capacity_retention_pct", "average_coulombic_efficiency_pct"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def detect_anomalies(data: pd.DataFrame, mapping: dict[str, str | None], settings: AnalysisSettings) -> pd.DataFrame:
    group_col = _group_column(data, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    charge_col = _mapped(mapping, "charge_capacity")
    discharge_col = _mapped(mapping, "discharge_capacity")
    ce_col = "calculated_coulombic_efficiency_pct"
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

        if cycle_col and cycle_col in group_sorted.columns:
            cycles = pd.to_numeric(group_sorted[cycle_col], errors="coerce").dropna().sort_values()
            if len(cycles) >= 2:
                diffs = cycles.diff().dropna()
                gaps = diffs[diffs > 1]
                for idx, gap in gaps.items():
                    anomalies.append({
                        "cell_or_sample": str(group_name),
                        "cycle": float(cycles.loc[idx]),
                        "type": "missing_cycle_gap",
                        "severity": "medium",
                        "details": f"Gap of {gap:.0f} cycles before cycle {cycles.loc[idx]:.0f}.",
                    })

        for col_name, label in [(charge_col, "charge/lithiation"), (discharge_col, "discharge/delithiation")]:
            if col_name and col_name in group_sorted.columns:
                values = pd.to_numeric(group_sorted[col_name], errors="coerce")
                bad = group_sorted[values < 0]
                for _, row in bad.iterrows():
                    anomalies.append({
                        "cell_or_sample": str(group_name),
                        "cycle": _safe_float(row.get(cycle_col)) if cycle_col else None,
                        "type": "negative_capacity",
                        "severity": "high",
                        "details": f"Negative {label} capacity in column '{col_name}'.",
                    })

        if ce_col in group_sorted.columns:
            ce_values = pd.to_numeric(group_sorted[ce_col], errors="coerce")
            ce_bad = group_sorted[(ce_values < settings.ce_low_threshold_pct) | (ce_values > settings.ce_high_threshold_pct)]
            for _, row in ce_bad.iterrows():
                val = row.get(ce_col)
                anomalies.append({
                    "cell_or_sample": str(group_name),
                    "cycle": _safe_float(row.get(cycle_col)) if cycle_col else None,
                    "type": "calculated_coulombic_efficiency_out_of_range",
                    "severity": "medium",
                    "details": f"Calculated CE={val:.2f}% outside {settings.ce_low_threshold_pct:.1f}-{settings.ce_high_threshold_pct:.1f}%." if pd.notna(val) else "Calculated CE is missing or invalid.",
                })

        if cycle_col and cycle_col in group_sorted.columns and "capacity_retention_pct" in group_sorted.columns:
            tmp = group_sorted.copy()
            tmp["retention_drop_pct"] = tmp["capacity_retention_pct"].diff()
            drops = tmp[tmp["retention_drop_pct"] <= -abs(settings.capacity_drop_threshold_pct)]
            for _, row in drops.iterrows():
                anomalies.append({
                    "cell_or_sample": str(group_name),
                    "cycle": _safe_float(row.get(cycle_col)),
                    "type": "abnormal_capacity_drop",
                    "severity": "high",
                    "details": f"Retention dropped by {abs(row['retention_drop_pct']):.2f}% from previous point.",
                })

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
    summary: dict[str, Any] = {
        "n_rows_analyzed": int(len(data)),
        "n_cells_or_samples": int(len(metrics)),
        "n_anomalies": int(len(anomalies)),
        "best_performing_cell": None,
        "worst_performing_cell": None,
        "mean_last_capacity_retention_pct": None,
        "mean_last_capacity_retention_from_first_pct": None,
        "mean_last_capacity_retention_from_max_pct": None,
        "mean_degradation_rate_pct_per_cycle": None,
        "mean_calculated_ce_pct": None,
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
        if "last_capacity_retention_from_first_pct" in metrics.columns:
            summary["mean_last_capacity_retention_from_first_pct"] = _safe_float(metrics["last_capacity_retention_from_first_pct"].mean())
        if "last_capacity_retention_from_max_pct" in metrics.columns:
            summary["mean_last_capacity_retention_from_max_pct"] = _safe_float(metrics["last_capacity_retention_from_max_pct"].mean())
        if "degradation_rate_pct_per_cycle" in metrics.columns:
            summary["mean_degradation_rate_pct_per_cycle"] = _safe_float(metrics["degradation_rate_pct_per_cycle"].mean())
        if "average_coulombic_efficiency_pct" in metrics.columns:
            summary["mean_calculated_ce_pct"] = _safe_float(metrics["average_coulombic_efficiency_pct"].mean())
    return summary


def add_cell_outlier_anomalies(metrics: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    extra: list[dict[str, Any]] = []
    if metrics.empty:
        return anomalies
    indexed = metrics.set_index("cell_or_sample", drop=False)
    for col, label in [("last_capacity_retention_pct", "final retention"), ("degradation_rate_pct_per_cycle", "degradation rate")]:
        if col not in indexed.columns:
            continue
        for cell in _iqr_outlier_labels(indexed[col]):
            value = indexed.loc[cell, col]
            extra.append({
                "cell_or_sample": str(cell),
                "cycle": None,
                "type": "cell_level_outlier",
                "severity": "medium",
                "details": f"Cell/sample is an IQR outlier for {label}: {value:.3f}.",
            })
    if not extra:
        return anomalies
    return pd.concat([anomalies, pd.DataFrame(extra)], ignore_index=True)


def _quality_summary(df: pd.DataFrame) -> dict[str, Any]:
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




def build_multi_cell_deviation_report(metrics: pd.DataFrame, analysis_type: str = "multi_cell") -> dict[str, Any]:
    """Build reproducibility/deviation tables for multi-cell datasets.

    The table is intentionally metric-based rather than raw-data based. It asks:
    how much do repeated cells differ in first-cycle behavior, peak/upper
    capacity, CE stability, and final retention.
    """
    empty = {"enabled": False, "summary": {}, "per_cell_deviation": pd.DataFrame(), "std_table": pd.DataFrame()}
    if metrics is None or metrics.empty:
        return empty
    if analysis_type != "multi_cell" or len(metrics) < 2:
        return empty

    std_specs = [
        ("first_coulombic_efficiency_pct", "STD in first-cycle CE (%)"),
        ("average_coulombic_efficiency_pct", "STD in average CE (%)"),
        ("first_charge_or_lithiation_capacity", "STD in first-cycle charge/lithiation capacity"),
        ("first_discharge_or_delithiation_capacity", "STD in first-cycle discharge/delithiation capacity"),
        ("max_charge_or_lithiation_capacity", "STD in upper/max charge/lithiation capacity"),
        ("max_discharge_or_delithiation_capacity", "STD in upper/max discharge/delithiation capacity"),
        ("last_capacity_retention_pct", "STD in final retention (%)"),
        ("degradation_rate_pct_per_cycle", "STD in degradation rate (%/cycle)"),
    ]
    rows: list[dict[str, Any]] = []
    for col, label in std_specs:
        if col not in metrics.columns:
            continue
        values = pd.to_numeric(metrics[col], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "parameter": label,
            "n_cells": int(values.count()),
            "mean": _safe_float(values.mean()),
            "std": _safe_float(values.std(ddof=1)) if len(values) >= 2 else 0.0,
            "rsd_pct": _safe_float(values.std(ddof=1) / values.mean() * 100.0) if len(values) >= 2 and values.mean() not in (0, np.nan) else None,
            "min": _safe_float(values.min()),
            "max": _safe_float(values.max()),
            "range": _safe_float(values.max() - values.min()),
        })
    std_table = pd.DataFrame(rows)

    deviation_cols = [
        "first_coulombic_efficiency_pct",
        "average_coulombic_efficiency_pct",
        "first_charge_or_lithiation_capacity",
        "first_discharge_or_delithiation_capacity",
        "max_charge_or_lithiation_capacity",
        "max_discharge_or_delithiation_capacity",
        "last_capacity_retention_pct",
        "degradation_rate_pct_per_cycle",
    ]
    per_cell = metrics[["cell_or_sample"] + [c for c in deviation_cols if c in metrics.columns]].copy()
    for col in [c for c in deviation_cols if c in per_cell.columns]:
        vals = pd.to_numeric(per_cell[col], errors="coerce")
        mean = vals.mean()
        per_cell[f"{col}_deviation_from_mean"] = vals - mean
        per_cell[f"{col}_deviation_pct"] = ((vals - mean) / mean * 100.0) if pd.notna(mean) and mean != 0 else np.nan

    summary = {
        "analysis_type": analysis_type,
        "n_cells_compared": int(len(metrics)),
        "highest_first_cycle_ce_std_parameter": None,
        "capacity_reproducibility_note": "Lower standard deviation and RSD indicate better cell-to-cell reproducibility.",
    }
    if not std_table.empty:
        summary["largest_relative_deviation_parameter"] = str(std_table.sort_values("rsd_pct", ascending=False, na_position="last").iloc[0]["parameter"])
        summary["mean_rsd_pct"] = _safe_float(pd.to_numeric(std_table["rsd_pct"], errors="coerce").mean())

    return {"enabled": True, "summary": summary, "per_cell_deviation": per_cell, "std_table": std_table}

def run_battery_analysis(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    settings_dict: dict[str, Any] | None = None,
    include_segments: list[str] | None = None,
    battery_cell_type: str = "full_cell",
    analysis_type: str = "multi_cell",
) -> dict[str, Any]:
    """Run the complete local battery-analysis workflow."""
    settings_data = dict(settings_dict or {})
    settings_data["battery_cell_type"] = battery_cell_type
    settings = AnalysisSettings(**settings_data)

    processed_all = add_derived_metrics(df, mapping, battery_cell_type=battery_cell_type)
    # Important: derive retention baselines before segment filtering, so cycle 1 /
    # first valid capacity remains available even when the user analyzes only cycling
    # rows. This avoids incorrectly using cycle 3/4 as the retention baseline.
    processed = analysis_ready_dataframe(processed_all, include_segments=include_segments).copy()
    metrics = summarize_cells(processed, mapping, settings)
    anomalies = detect_anomalies(processed, mapping, settings)

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
    deviation_report = build_multi_cell_deviation_report(metrics, analysis_type=analysis_type)
    summary = summarize_overall(metrics, anomalies, processed)
    summary.update(_quality_summary(processed_all))
    summary["battery_cell_type"] = _cell_type_label(battery_cell_type)
    summary["calculated_columns"] = [
        "CE(%)",
        "100-CE",
        "irreversible_capacity",
        "Acc_irreversible_capacity_u_mAh_per_cm2",
        "Accumulated CE",
        "CE_20_pt_AVG",
        "capacity_retention_from_first_pct",
        "capacity_retention_from_max_pct",
        "SOH_pct",
        "estimated_c_rate_from_time",
    ]

    return {
        "processed_data": processed,
        "processed_data_all_rows": processed_all,
        "cell_metrics": metrics,
        "anomalies": anomalies,
        "summary": summary,
        "settings": settings.__dict__,
        "include_segments": include_segments,
        "battery_cell_type": battery_cell_type,
        "analysis_type": analysis_type,
        "deviation_report": deviation_report,
    }
