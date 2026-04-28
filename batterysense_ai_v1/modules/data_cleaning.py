"""Data cleaning and protocol segmentation utilities for battery datasets.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
Conservative design: suspicious points are flagged and excluded from analysis by
default, but the rows remain in the processed table for auditability.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_CLEANING_SETTINGS: dict[str, Any] = {
    "drop_all_empty_rows": True,
    "drop_duplicate_rows": False,
    "remove_suspicious_points": True,
    "robust_outlier_mad_threshold": 8.0,
    "extreme_quantile": 0.999,
    "absolute_ce_min_pct": 0.0,
    "absolute_ce_max_pct": 150.0,
    "terminal_zero_capacity_is_incomplete": True,
    "terminal_change_threshold_pct": 30.0,
    "minimum_terminal_reference_points": 3,
    "formation_max_cycles": 3,
    "formation_ce_threshold_pct": 95.0,
    "rate_test_cv_threshold": 0.08,
    "rate_test_window_cycles": 5,
}


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in columns:
        if column not in cleaned.columns:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
            continue
        series = cleaned[column].astype(str).str.strip()
        series = series.str.replace("%", "", regex=False).str.replace(" ", "", regex=False)
        series = series.str.replace(r"(?<=\d),(?=\d+$)", ".", regex=True)
        cleaned[column] = pd.to_numeric(series, errors="coerce")
    return cleaned


def _mapped(mapping: dict[str, str | None] | None, key: str) -> str | None:
    value = (mapping or {}).get(key)
    return str(value) if value not in (None, "") else None


def _group_column(df: pd.DataFrame, mapping: dict[str, str | None] | None) -> str | None:
    for key in ("cell_name", "sample_name"):
        col = _mapped(mapping, key)
        if col and col in df.columns:
            return col
    return "source_file" if "source_file" in df.columns else None


def _append_flag(df: pd.DataFrame, mask: pd.Series, flag: str, reason: str) -> None:
    mask = mask.reindex(df.index, fill_value=False).fillna(False)
    if not mask.any():
        return
    df.loc[mask, "quality_flag"] = df.loc[mask, "quality_flag"].astype(str).replace("nan", "")
    df.loc[mask, "quality_flag"] = df.loc[mask, "quality_flag"].apply(lambda x: flag if not x else f"{x};{flag}")
    df.loc[mask, "exclude_reason"] = df.loc[mask, "exclude_reason"].astype(str).replace("nan", "")
    df.loc[mask, "exclude_reason"] = df.loc[mask, "exclude_reason"].apply(lambda x: reason if not x else f"{x}; {reason}")
    df.loc[mask, "analysis_include"] = False


def _robust_outlier_mask(series: pd.Series, mad_threshold: float = 8.0, extreme_quantile: float = 0.999) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = pd.Series(False, index=series.index)
    valid = s.dropna()
    if len(valid) < 6:
        return mask | (s.abs() > 1e12).fillna(False)
    med = valid.median()
    mad = np.median(np.abs(valid - med))
    if mad and np.isfinite(mad):
        mask |= (0.6745 * (s - med).abs() / mad) > mad_threshold
    high = valid.abs().quantile(extreme_quantile)
    if pd.notna(high) and high > 0:
        mask |= s.abs() > high * 10
    return mask.fillna(False)


def _detect_incomplete_terminal_points(df: pd.DataFrame, mapping: dict[str, str | None], settings: dict[str, Any]) -> pd.DataFrame:
    data = df.copy()
    group_col = _group_column(data, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    cap_cols = [c for c in [_mapped(mapping, "discharge_capacity"), _mapped(mapping, "charge_capacity")] if c in data.columns]
    if not cap_cols:
        return data
    groups = data.groupby(group_col, dropna=False) if group_col else [("All data", data)]
    threshold = float(settings.get("terminal_change_threshold_pct", 30.0))
    min_ref = int(settings.get("minimum_terminal_reference_points", 3))
    for _, group in groups:
        g = group.sort_values(cycle_col) if cycle_col and cycle_col in group.columns else group.copy()
        if len(g) < min_ref + 1:
            continue
        last_idx = g.index[-1]
        prev = g.iloc[-(min_ref + 1):-1]
        reasons: list[str] = []
        for cap_col in cap_cols:
            last_val = pd.to_numeric(pd.Series([g.loc[last_idx, cap_col]]), errors="coerce").iloc[0]
            prev_vals = pd.to_numeric(prev[cap_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if pd.isna(last_val) or len(prev_vals) < min_ref:
                continue
            baseline = prev_vals.median()
            if baseline == 0 or pd.isna(baseline):
                continue
            rel_change = abs(last_val - baseline) / abs(baseline) * 100
            if bool(settings.get("terminal_zero_capacity_is_incomplete", True)) and abs(last_val) < 1e-12 and abs(baseline) > 0:
                reasons.append(f"last {cap_col} is zero while previous median is non-zero")
            elif rel_change >= threshold:
                reasons.append(f"last {cap_col} differs from previous median by {rel_change:.1f}%")
        if reasons:
            _append_flag(data, pd.Series(data.index == last_idx, index=data.index), "incomplete_terminal_cycle", "; ".join(reasons))
    return data


def detect_protocol_segments(df: pd.DataFrame, mapping: dict[str, str | None], settings: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = {**DEFAULT_CLEANING_SETTINGS, **(settings or {})}
    data = df.copy()
    data["protocol_segment"] = "cycling"
    group_col = _group_column(data, mapping)
    cycle_col = _mapped(mapping, "cycle_number")
    ce_col = _mapped(mapping, "coulombic_efficiency")
    discharge_col = _mapped(mapping, "discharge_capacity")
    current_col = _mapped(mapping, "current")

    textual_cols = [c for c in data.columns if any(k in str(c).lower() for k in ["step", "mode", "procedure", "protocol", "rate", "c-rate", "crate"])]
    if textual_cols:
        text = data[textual_cols].astype(str).agg(" ".join, axis=1).str.lower()
        data.loc[text.str.contains("formation|form", regex=True, na=False), "protocol_segment"] = "formation"
        data.loc[text.str.contains(r"rate|c-rate|crate|c/|1c|2c|5c|10c", regex=True, na=False), "protocol_segment"] = "rate_test"

    groups = data.groupby(group_col, dropna=False) if group_col else [("All data", data)]
    for _, group in groups:
        g = group.sort_values(cycle_col) if cycle_col and cycle_col in group.columns else group.copy()
        if g.empty:
            continue
        if cycle_col and cycle_col in g.columns:
            cycles = pd.to_numeric(g[cycle_col], errors="coerce")
            min_cycle = cycles.min(skipna=True)
            if pd.notna(min_cycle):
                formation_mask = cycles <= min_cycle + int(cfg.get("formation_max_cycles", 3)) - 1
                data.loc[g.index[formation_mask.fillna(False)], "protocol_segment"] = "formation"
        if ce_col and ce_col in g.columns:
            ce = pd.to_numeric(g[ce_col], errors="coerce")
            early = g.head(max(int(cfg.get("formation_max_cycles", 3)), 1))
            early_ce = ce.loc[early.index]
            low_ce_idx = early.index[(early_ce < float(cfg.get("formation_ce_threshold_pct", 95.0))).fillna(False)]
            data.loc[low_ce_idx, "protocol_segment"] = "formation"
        rate_mask = pd.Series(False, index=g.index)
        if discharge_col and discharge_col in g.columns and len(g) >= 8:
            cap = pd.to_numeric(g[discharge_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            win = int(cfg.get("rate_test_window_cycles", 5))
            local_cv = cap.rolling(win, min_periods=3).std() / cap.rolling(win, min_periods=3).mean().abs()
            rate_mask |= local_cv > float(cfg.get("rate_test_cv_threshold", 0.08))
        if current_col and current_col in g.columns and len(g) >= 8:
            cur = pd.to_numeric(g[current_col], errors="coerce").replace([np.inf, -np.inf], np.nan).abs()
            if cur.nunique(dropna=True) > 2:
                rate_mask |= cur.pct_change().abs().fillna(0) > 0.15
        rate_idx = g.index[rate_mask.fillna(False) & (data.loc[g.index, "protocol_segment"] != "formation")]
        data.loc[rate_idx, "protocol_segment"] = "rate_test"
    return data


def add_quality_flags(df: pd.DataFrame, mapping: dict[str, str | None] | None, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = {**DEFAULT_CLEANING_SETTINGS, **(settings or {})}
    data = df.copy()
    data["analysis_include"] = True
    data["quality_flag"] = ""
    data["exclude_reason"] = ""
    numeric_keys = ["cycle_number", "charge_capacity", "discharge_capacity", "coulombic_efficiency", "voltage", "current", "cc_capacity", "cv_capacity"]
    numeric_cols = [c for c in (_mapped(mapping, k) for k in numeric_keys) if c and c in data.columns]
    for col in [c for c in [_mapped(mapping, "cycle_number"), _mapped(mapping, "discharge_capacity")] if c in data.columns]:
        values = pd.to_numeric(data[col], errors="coerce")
        _append_flag(data, values.isna() | ~np.isfinite(values), "missing_or_nonfinite_critical_value", f"{col} is missing or non-finite")
    if cfg.get("remove_suspicious_points", True):
        for col in numeric_cols:
            mask = _robust_outlier_mask(data[col], float(cfg["robust_outlier_mad_threshold"]), float(cfg["extreme_quantile"]))
            _append_flag(data, mask, "extreme_numeric_outlier", f"{col} is an extreme numeric outlier")
    ce_col = _mapped(mapping, "coulombic_efficiency")
    if ce_col and ce_col in data.columns:
        ce = pd.to_numeric(data[ce_col], errors="coerce")
        mask = (ce < float(cfg["absolute_ce_min_pct"])) | (ce > float(cfg["absolute_ce_max_pct"]))
        _append_flag(data, mask.fillna(False), "implausible_ce", f"{ce_col} outside physically plausible range")
    if mapping:
        data = _detect_incomplete_terminal_points(data, mapping, cfg)
    return data


def clean_dataframe(
    df: pd.DataFrame,
    numeric_columns: Iterable[str] | None = None,
    drop_all_empty_rows: bool = True,
    drop_duplicate_rows: bool = False,
    mapping: dict[str, str | None] | None = None,
    cleaning_settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = {**DEFAULT_CLEANING_SETTINGS, **(cleaning_settings or {})}
    cleaned = df.copy().replace(r"^\s*$", np.nan, regex=True)
    if drop_all_empty_rows:
        cleaned = cleaned.dropna(how="all")
    if numeric_columns:
        cleaned = coerce_numeric_columns(cleaned, numeric_columns)
    if drop_duplicate_rows:
        cleaned = cleaned.drop_duplicates()
    if mapping:
        cleaned = add_quality_flags(cleaned, mapping, cfg)
        cleaned = detect_protocol_segments(cleaned, mapping, cfg)
    return cleaned.reset_index(drop=True)


def analysis_ready_dataframe(df: pd.DataFrame, include_segments: Iterable[str] | None = None) -> pd.DataFrame:
    data = df.copy()
    if "analysis_include" in data.columns:
        data = data[data["analysis_include"]]
    if include_segments is not None and "protocol_segment" in data.columns:
        data = data[data["protocol_segment"].isin(list(include_segments))]
    return data.reset_index(drop=True)
