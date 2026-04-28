"""Column detection and canonicalization helpers for battery datasets.
BatterySense AI
**Author: AU P. Vajeeston, 2026**
Supports common exports from Neware, Arbin, BioLogic, and generic CSV/XLSX
files by mapping tester-specific names to semantic fields.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

import pandas as pd


BATTERY_COLUMN_ALIASES: dict[str, list[str]] = {
    "sample_name": ["sample", "sample name", "sample_name", "sample id", "sample_id", "material", "batch", "group", "sample_full_name", "legend_title"],
    "cell_name": ["cell", "cell name", "cell_name", "cell id", "cell_id", "cell number", "cell_file_name", "file", "filename", "test name", "channel", "barcode"],
    "cycle_number": ["cycle", "cycles", "cycle number", "cycle_number", "cycle index", "cycle_index", "cycle no", "cycle_no", "n cycle", "cycle_count", "cycle id"],
    "charge_capacity": ["charge capacity", "charge_capacity", "chg capacity", "charge cap", "lithiation capacity", "lithiation_capacity", "cha cap", "capacity charge", "charge_capacity(mah/g)", "ch. cap.", "q charge", "q_charge"],
    "discharge_capacity": ["discharge capacity", "discharge_capacity", "dchg capacity", "discharge cap", "delithiation capacity", "delithiation_capacity", "dis cap", "capacity discharge", "discharge_capacity(mah/g)", "dch. cap.", "q discharge", "q_discharge"],
    "coulombic_efficiency": ["coulombic efficiency", "coulombic_efficiency", "ce", "ce(%)", "ce (%)", "efficiency", "coulombic efficiency (%)", "coulomb efficiency", "coulombic_efficiency/%"],
    "voltage": ["voltage", "volt", "v", "ewe", "potential", "cell voltage", "voltage(v)", "ewe/v"],
    "current": ["current", "i", "amps", "ampere", "current(a)", "current_ma", "current ma", "i/a", "current/ma"],
    "cc_capacity": ["cc capacity", "cc_capacity", "constant current capacity", "cc charge capacity", "charge_cc_capacity", "cc cap", "cc_charge_capacity", "cc capacity(ah)"],
    "cv_capacity": ["cv capacity", "cv_capacity", "constant voltage capacity", "cv charge capacity", "charge_cv_capacity", "cv cap", "cv_charge_capacity", "cv capacity(ah)"],
    "c_rate": ["c rate", "c-rate", "crate", "rate", "rate type", "current rate", "discharge rate", "charge rate"],
    "step_type": ["step", "step type", "step_type", "mode", "status", "state", "procedure", "test step", "control mode"],
    "time": ["time", "test time", "date time", "datetime", "total time", "time(s)", "time / s"],
}

DISPLAY_NAMES: dict[str, str] = {
    "sample_name": "Sample name",
    "cell_name": "Cell name",
    "cycle_number": "Cycle number",
    "charge_capacity": "Charge capacity",
    "discharge_capacity": "Discharge capacity",
    "coulombic_efficiency": "Coulombic efficiency",
    "voltage": "Voltage",
    "current": "Current",
    "cc_capacity": "CC capacity",
    "cv_capacity": "CV capacity",
    "c_rate": "C-rate / rate label",
    "step_type": "Step/protocol type",
    "time": "Time / timestamp",
}

NUMERIC_SEMANTIC_COLUMNS = {"cycle_number", "charge_capacity", "discharge_capacity", "coulombic_efficiency", "voltage", "current", "cc_capacity", "cv_capacity"}
CANONICAL_COLUMN_NAMES: dict[str, str] = {key: key for key in BATTERY_COLUMN_ALIASES}


def normalize_name(name: object) -> str:
    text = str(name).strip().lower().replace("μ", "u")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_name(name: object) -> str:
    return normalize_name(name).replace(" ", "")


def _score_column(column: str, aliases: Iterable[str]) -> float:
    col_norm = normalize_name(column)
    col_compact = compact_name(column)
    best_score = 0.0
    for alias in aliases:
        alias_norm = normalize_name(alias)
        alias_compact = compact_name(alias)
        if col_compact == alias_compact:
            score = 1.0
        elif alias_compact and alias_compact in col_compact:
            score = 0.92
        elif col_compact and col_compact in alias_compact:
            score = 0.86
        else:
            score = SequenceMatcher(None, col_norm, alias_norm).ratio()
        best_score = max(best_score, score)
    return best_score


def detect_columns(df: pd.DataFrame, min_score: float = 0.68) -> dict[str, str | None]:
    suggestions: dict[str, str | None] = {}
    columns = [str(col) for col in df.columns]
    for semantic_name, aliases in BATTERY_COLUMN_ALIASES.items():
        ranked = sorted(((column, _score_column(column, aliases)) for column in columns), key=lambda item: item[1], reverse=True)
        suggestions[semantic_name] = ranked[0][0] if ranked and ranked[0][1] >= min_score else None
    if suggestions.get("cell_name") is None and "source_file" in df.columns:
        suggestions["cell_name"] = "source_file"
    return suggestions


def canonicalize_battery_columns(df: pd.DataFrame, mapping: dict[str, str | None], overwrite: bool = False) -> tuple[pd.DataFrame, dict[str, str | None]]:
    data = df.copy()
    new_mapping = dict(mapping)
    for semantic, original in mapping.items():
        canonical = CANONICAL_COLUMN_NAMES.get(semantic)
        if not canonical or not original or original not in data.columns:
            continue
        if overwrite or canonical not in data.columns:
            data[canonical] = data[original]
        new_mapping[semantic] = canonical
    if new_mapping.get("cell_name") is None and "source_file" in data.columns:
        new_mapping["cell_name"] = "source_file"
    return data, new_mapping


def get_ranked_candidates(df: pd.DataFrame, semantic_name: str, max_candidates: int = 5) -> list[tuple[str, float]]:
    aliases = BATTERY_COLUMN_ALIASES.get(semantic_name, [])
    ranked = sorted(((str(column), _score_column(str(column), aliases)) for column in df.columns), key=lambda item: item[1], reverse=True)
    return ranked[:max_candidates]


def validate_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> dict[str, str]:
    warnings: dict[str, str] = {}
    df_columns = set(str(col) for col in df.columns)
    for semantic_name, column in mapping.items():
        if column in (None, ""):
            continue
        if str(column) not in df_columns:
            warnings[semantic_name] = f"Mapped column '{column}' is not present in the dataset."
    if not mapping.get("cycle_number"):
        warnings["cycle_number"] = "Cycle number was not detected. Some cycle-based analysis and plots will be skipped."
    if not mapping.get("discharge_capacity"):
        warnings["discharge_capacity"] = "Discharge capacity was not detected. Retention and degradation metrics will be limited."
    return warnings


def numeric_columns_from_mapping(mapping: dict[str, str | None]) -> list[str]:
    return [str(column) for semantic, column in mapping.items() if semantic in NUMERIC_SEMANTIC_COLUMNS and column not in (None, "")]
