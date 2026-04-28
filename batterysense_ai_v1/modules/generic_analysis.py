"""Generic non-battery dataset analysis utilities.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generic_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a general-purpose dataset summary for non-battery files."""
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric.columns.astype(str).tolist(),
        "categorical_columns": categorical.columns.astype(str).tolist(),
        "missing_percent": {str(k): float(v) for k, v in (df.isna().mean() * 100).round(2).items()},
        "numeric_describe": numeric.describe().T.reset_index(names="column").replace([np.inf, -np.inf], np.nan).to_dict("records") if not numeric.empty else [],
        "categorical_top_values": {
            str(col): df[col].astype(str).value_counts(dropna=False).head(10).to_dict()
            for col in categorical.columns[:20]
        },
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
    return grouped.reset_index()
