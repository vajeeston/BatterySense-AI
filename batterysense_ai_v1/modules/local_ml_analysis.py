"""Local scikit-learn analysis: anomaly detection, clustering, and life prediction.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numeric_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, Any]:
    from sklearn.preprocessing import StandardScaler
    X = df[columns].copy() if columns else df.select_dtypes(include="number").copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    return X, scaler.fit_transform(X)


def isolation_forest_anomalies(df: pd.DataFrame, columns: list[str] | None = None, contamination: float = 0.05) -> pd.DataFrame:
    try:
        from sklearn.ensemble import IsolationForest
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame({"error": [f"scikit-learn is required: {exc}"]})
    Xdf, X = _numeric_matrix(df, columns)
    if Xdf.shape[0] < 10 or Xdf.shape[1] < 1:
        return pd.DataFrame()
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X)
    scores = model.decision_function(X)
    out = df.copy()
    out["ml_anomaly_label"] = labels
    out["ml_anomaly_score"] = scores
    return out[out["ml_anomaly_label"] == -1].copy()


def kmeans_clustering(df: pd.DataFrame, columns: list[str] | None = None, n_clusters: int = 3) -> pd.DataFrame:
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame({"error": [f"scikit-learn is required: {exc}"]})
    Xdf, X = _numeric_matrix(df, columns)
    if Xdf.shape[0] < n_clusters or Xdf.shape[1] < 1:
        return pd.DataFrame()
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(X)
    out = df.copy()
    out["cluster"] = labels
    return out


def predict_lifetime_from_metrics(metrics: pd.DataFrame, target_col: str = "cycle_life_estimate_to_80pct") -> dict[str, Any]:
    """Train a local random-forest lifetime model when enough historical labels exist."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except Exception as exc:  # noqa: BLE001
        return {"error": f"scikit-learn is required: {exc}"}
    if target_col not in metrics.columns:
        return {"error": f"Target column {target_col!r} not found."}
    numeric = metrics.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    work = numeric.dropna(subset=[target_col]).copy()
    features = [c for c in work.columns if c != target_col and work[c].notna().sum() >= 5]
    if len(work) < 10 or len(features) < 2:
        return {"error": "Need at least 10 labeled cells and 2 usable numeric features for lifetime prediction."}
    X = work[features].fillna(work[features].median())
    y = work[target_col]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    cv = min(5, len(work))
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    model.fit(X, y)
    return {"features": features, "cv_r2_mean": float(np.mean(scores)), "cv_r2_std": float(np.std(scores)), "model": model}
