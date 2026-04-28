"""SQLite database for saving historical analysis runs.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def init_database(db_path: str | Path = "outputs/database/analysis_runs.sqlite") -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                app_version TEXT,
                report_title TEXT,
                uploaded_files TEXT,
                settings_json TEXT,
                summary_json TEXT,
                metrics_json TEXT,
                anomalies_json TEXT,
                html_report_path TEXT,
                pdf_report_path TEXT
            )
            """
        )
    return path


def save_analysis_run(
    db_path: str | Path,
    report_title: str,
    uploaded_files: list[str],
    settings: dict[str, Any],
    analysis: dict[str, Any],
    html_report_path: str | None = None,
    pdf_report_path: str | None = None,
    app_version: str = "0.2.0",
) -> int:
    path = init_database(db_path)
    metrics = analysis.get("cell_metrics")
    anomalies = analysis.get("anomalies")
    metrics_json = metrics.to_json(orient="records") if hasattr(metrics, "to_json") else "[]"
    anomalies_json = anomalies.to_json(orient="records") if hasattr(anomalies, "to_json") else "[]"
    with sqlite3.connect(path) as con:
        cur = con.execute(
            """
            INSERT INTO analysis_runs
            (created_at, app_version, report_title, uploaded_files, settings_json, summary_json,
             metrics_json, anomalies_json, html_report_path, pdf_report_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(),
                app_version,
                report_title,
                json.dumps(uploaded_files),
                json.dumps(settings, default=str),
                json.dumps(analysis.get("summary", {}), default=str),
                metrics_json,
                anomalies_json,
                html_report_path,
                pdf_report_path,
            ),
        )
        return int(cur.lastrowid)


def list_analysis_runs(db_path: str | Path = "outputs/database/analysis_runs.sqlite", limit: int = 50) -> list[dict[str, Any]]:
    path = init_database(db_path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM analysis_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
