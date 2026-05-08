"""Data loading and dataset profiling utilities.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
This module is intentionally independent from Streamlit. The app passes uploaded
file-like objects here and receives pandas DataFrames plus metadata back.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass
class LoadedDataset:
    """Container for one loaded table."""

    name: str
    dataframe: pd.DataFrame
    source_file: str
    source_sheet: str | None = None


def _read_csv_from_bytes(raw: bytes) -> pd.DataFrame:
    """Read CSV bytes using a few common encodings and automatic delimiter sniffing."""
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                BytesIO(raw),
                encoding=encoding,
                sep=None,          # let pandas detect comma/semicolon/tab
                engine="python",
            )
        except Exception as exc:  # noqa: BLE001 - we want to try all encodings
            last_error = exc

    raise ValueError(f"Could not read CSV file. Last error: {last_error}")


def _read_xlsx_from_bytes(raw: bytes, file_name: str) -> list[LoadedDataset]:
    """Read all visible sheets from an XLSX file."""
    try:
        sheets = pd.read_excel(BytesIO(raw), sheet_name=None, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read Excel file '{file_name}': {exc}") from exc

    datasets: list[LoadedDataset] = []
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        clean_name = f"{Path(file_name).stem}__{sheet_name}"
        datasets.append(
            LoadedDataset(
                name=clean_name,
                dataframe=df,
                source_file=file_name,
                source_sheet=str(sheet_name),
            )
        )
    return datasets


def load_uploaded_files(uploaded_files: Iterable[Any]) -> list[LoadedDataset]:
    """Load Streamlit uploaded CSV/XLSX files into a list of LoadedDataset objects.

    Parameters
    ----------
    uploaded_files:
        Iterable of Streamlit UploadedFile objects or objects with ``name`` and
        ``getvalue()`` attributes.
    """
    datasets: list[LoadedDataset] = []

    for uploaded in uploaded_files:
        file_name = getattr(uploaded, "name", "uploaded_file")
        suffix = Path(file_name).suffix.lower()
        raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()

        if suffix == ".csv":
            df = _read_csv_from_bytes(raw)
            datasets.append(
                LoadedDataset(
                    name=Path(file_name).stem,
                    dataframe=df,
                    source_file=file_name,
                )
            )
        elif suffix in {".xlsx", ".xlsm"}:
            datasets.extend(_read_xlsx_from_bytes(raw, file_name=file_name))
        else:
            raise ValueError(f"Unsupported file type: {file_name}")

    if not datasets:
        raise ValueError("No readable tables were found in the uploaded files.")

    return datasets


def combine_datasets(datasets: list[LoadedDataset]) -> pd.DataFrame:
    """Combine loaded datasets into one DataFrame while preserving provenance."""
    frames: list[pd.DataFrame] = []
    for item in datasets:
        df = item.dataframe.copy()
        # Use the file stem as the default cell/sample label so plots and reports
        # display clean names like cell_01 instead of cell_01.csv.
        # Keep the original file name separately for traceability.
        df["source_file"] = Path(item.source_file).stem
        df["source_file_original"] = item.source_file
        if item.source_sheet is not None:
            df["source_sheet"] = item.source_sheet
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact profile that can be displayed in the UI and report."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_columns = [
        col
        for col in df.columns
        if col not in numeric_columns and col not in datetime_columns
    ]

    missing_values = df.isna().sum().sort_values(ascending=False)
    missing_percent = (df.isna().mean() * 100).round(2).sort_values(ascending=False)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(col) for col in df.columns],
        "missing_values": {str(k): int(v) for k, v in missing_values.items()},
        "missing_percent": {str(k): float(v) for k, v in missing_percent.items()},
        "numeric_columns": [str(col) for col in numeric_columns],
        "categorical_columns": [str(col) for col in categorical_columns],
        "datetime_columns": [str(col) for col in datetime_columns],
    }


def profile_loaded_datasets(datasets: list[LoadedDataset]) -> dict[str, dict[str, Any]]:
    """Profile each loaded dataset separately."""
    return {item.name: profile_dataframe(item.dataframe) for item in datasets}
