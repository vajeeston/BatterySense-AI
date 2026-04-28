"""Statistical analysis helpers: correlation, PCA, tests, and regression.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    return numeric.corr(method=method)


def run_pca(df: pd.DataFrame, columns: list[str] | None = None, n_components: int = 3) -> dict[str, Any]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        return {"error": f"scikit-learn is required for PCA: {exc}"}
    numeric = df[columns].copy() if columns else df.select_dtypes(include="number").copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    if numeric.shape[1] < 2 or numeric.shape[0] < 3:
        return {"error": "PCA requires at least 2 numeric columns and 3 rows."}
    n_components = min(n_components, numeric.shape[1], numeric.shape[0])
    X = StandardScaler().fit_transform(numeric)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.round(5).tolist(),
        "components": pd.DataFrame(pca.components_, columns=numeric.columns, index=[f"PC{i+1}" for i in range(n_components)]),
        "scores": pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)], index=numeric.index),
    }


def linear_regression(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_absolute_error
    except Exception as exc:  # noqa: BLE001
        return {"error": f"scikit-learn is required for regression: {exc}"}
    cols = [target] + features
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < max(5, len(features) + 2):
        return {"error": "Not enough complete rows for regression."}
    X, y = work[features], work[target]
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    return {
        "target": target,
        "features": features,
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "intercept": float(model.intercept_),
        "coefficients": {features[i]: float(model.coef_[i]) for i in range(len(features))},
    }


def hypothesis_test_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> dict[str, Any]:
    """Run t-test for two groups or Kruskal-Wallis for >2 groups."""
    try:
        from scipy import stats
    except Exception as exc:  # noqa: BLE001
        return {"error": f"scipy is required for hypothesis tests: {exc}"}
    if value_col not in df.columns or group_col not in df.columns:
        return {"error": "Selected value or group column not found."}
    groups = [g[value_col].dropna().astype(float).values for _, g in df.groupby(group_col) if len(g[value_col].dropna()) >= 2]
    if len(groups) < 2:
        return {"error": "Need at least two groups with at least two observations each."}
    if len(groups) == 2:
        stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
        return {"test": "Welch t-test", "statistic": float(stat), "p_value": float(p)}
    stat, p = stats.kruskal(*groups, nan_policy="omit")
    return {"test": "Kruskal-Wallis", "statistic": float(stat), "p_value": float(p)}
