"""AI-assisted scientific report generation with selectable backends.
BatterySense AI
**Author: AU P. Vajeeston, 2026**

Important design rule:
- Python calculates the metrics locally first.
- The AI backend receives summarized metrics only, not raw cycle-level data.

Supported report-writing backends:
1. ``openai``: cloud report generation through the OpenAI API.
2. ``ollama``: local LLM report generation through a locally running Ollama server.
3. ``rule_based``: deterministic local scientific summary with no LLM and no API.

If a selected AI backend is unavailable, the module returns a deterministic fallback
report rather than failing the Streamlit app.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import pandas as pd

ReportBackend = Literal["openai", "ollama", "rule_based"]

SYSTEM_INSTRUCTIONS = """You are a scientific battery data analyst.
Write concise, technically accurate, research-style interpretations of battery/cell cycling data.
Do not invent experimental details that are not present in the supplied summary.
When evidence is limited, clearly state that interpretation is tentative.
Use clear section headings and actionable recommendations.
"""

REQUIRED_SECTIONS = [
    "Executive summary",
    "Dataset overview",
    "Key performance metrics",
    "Cell/sample comparison",
    "Degradation analysis",
    "Coulombic efficiency interpretation",
    "Anomaly discussion",
    "Possible scientific/technical explanations",
    "Recommended next experiments",
    "Final conclusion",
]


def _round_value(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return round(value, ndigits)
    return value


def dataframe_to_records(df: pd.DataFrame, max_rows: int = 25) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records with limited row count."""
    if df is None or df.empty:
        return []
    limited = df.head(max_rows).copy()
    records = limited.where(pd.notna(limited), None).to_dict(orient="records")
    return [
        {str(key): _round_value(value) for key, value in row.items()}
        for row in records
    ]


def build_ai_context(
    dataset_profile: dict[str, Any],
    mapping: dict[str, str | None],
    analysis: dict[str, Any],
    uploaded_file_names: list[str],
) -> dict[str, Any]:
    """Build a compact, raw-data-free context object for AI generation."""
    metrics = analysis.get("cell_metrics", pd.DataFrame())
    anomalies = analysis.get("anomalies", pd.DataFrame())

    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        best_cells = metrics.sort_values("last_capacity_retention_pct", ascending=False, na_position="last").head(5)
        worst_cells = metrics.sort_values("last_capacity_retention_pct", ascending=True, na_position="last").head(5)
    else:
        best_cells = pd.DataFrame()
        worst_cells = pd.DataFrame()

    return {
        "uploaded_files": uploaded_file_names,
        "dataset_profile": {
            "rows": dataset_profile.get("rows"),
            "columns": dataset_profile.get("columns"),
            "column_names": dataset_profile.get("column_names"),
            "numeric_columns": dataset_profile.get("numeric_columns"),
            "categorical_columns": dataset_profile.get("categorical_columns"),
            "top_missing_values": dict(list(dataset_profile.get("missing_values", {}).items())[:20]),
        },
        "column_mapping": mapping,
        "overall_summary": analysis.get("summary", {}),
        "top_cells_by_retention": dataframe_to_records(best_cells, max_rows=5),
        "lowest_cells_by_retention": dataframe_to_records(worst_cells, max_rows=5),
        "cell_metrics": dataframe_to_records(metrics, max_rows=25),
        "anomalies": dataframe_to_records(anomalies, max_rows=30),
        "analysis_settings": analysis.get("settings", {}),
    }


def build_report_prompt(context: dict[str, Any]) -> str:
    """Build a backend-independent report prompt from summarized metrics."""
    prompt = {
        "task": "Write a scientific/technical battery cycling report interpretation using only the supplied summarized results.",
        "important_constraints": [
            "Do not claim that raw data was inspected by the language model.",
            "Do not invent cell chemistry, electrolyte, electrode loading, temperature, or C-rate unless present in the summary.",
            "Distinguish measured/calculated results from tentative scientific explanations.",
            "Use markdown headings and concise paragraphs.",
        ],
        "required_sections": REQUIRED_SECTIONS,
        "data_summary": context,
    }
    return json.dumps(prompt, indent=2)


def fallback_report(context: dict[str, Any], reason: str | None = None) -> str:
    """Create a deterministic report summary without calling an AI API or local LLM."""
    summary = context.get("overall_summary", {})
    profile = context.get("dataset_profile", {})
    best = summary.get("best_performing_cell") or "Not available"
    worst = summary.get("worst_performing_cell") or "Not available"
    mean_retention = summary.get("mean_last_capacity_retention_pct")
    mean_degradation = summary.get("mean_degradation_rate_pct_per_cycle")
    n_anomalies = summary.get("n_anomalies", 0)
    mean_retention_text = "Not available" if mean_retention is None else f"{mean_retention:.2f}%"
    mean_degradation_text = "Not available" if mean_degradation is None else f"{mean_degradation:.4f}% per cycle"

    top_cells = context.get("top_cells_by_retention", [])
    low_cells = context.get("lowest_cells_by_retention", [])
    top_cell_lines = "\n".join(
        f"- {row.get('cell_or_sample', row.get('cell_name', 'Unknown'))}: "
        f"{row.get('last_capacity_retention_pct', 'N/A')}% final retention"
        for row in top_cells[:5]
    ) or "- Not available"
    low_cell_lines = "\n".join(
        f"- {row.get('cell_or_sample', row.get('cell_name', 'Unknown'))}: "
        f"{row.get('last_capacity_retention_pct', 'N/A')}% final retention"
        for row in low_cells[:5]
    ) or "- Not available"

    anomaly_text = (
        f"The analysis detected **{n_anomalies}** anomaly flag(s). Review the anomaly table and raw tester files before "
        "making final conclusions."
        if n_anomalies
        else "No anomaly flags were detected with the current thresholds. This does not guarantee that the raw data are error-free."
    )

    note = f"\n\n**Backend status:** {reason}\n" if reason else ""

    return f"""# Scientific interpretation
{note}
## Executive summary
The uploaded dataset contains **{profile.get('rows', 'N/A')} rows** and **{profile.get('columns', 'N/A')} columns**. The local Python analysis identified **{summary.get('n_cells_or_samples', 0)} cell/sample group(s)** and **{n_anomalies} potential anomaly flag(s)**.

## Dataset overview
The report was generated from summarized metrics calculated locally in Python. The interpretation layer did not receive raw cycle-level data.

## Key performance metrics
- Best-performing cell/sample based on final capacity retention: **{best}**.
- Worst-performing cell/sample based on final capacity retention: **{worst}**.
- Mean final capacity retention: **{mean_retention_text}**.
- Mean degradation slope: **{mean_degradation_text}**.

## Cell/sample comparison
Top cells by final retention:
{top_cell_lines}

Lowest cells by final retention:
{low_cell_lines}

## Degradation analysis
Capacity retention was calculated from the first available discharge capacity of each cell/sample. A more negative degradation slope indicates faster capacity loss. Review the degradation comparison plot to identify cells with rapid fading or unstable cycling.

## Coulombic efficiency interpretation
Average Coulombic efficiency is useful for identifying irreversible side reactions, unstable SEI formation, electrolyte depletion, or measurement issues. Values far below or above the configured range should be checked against raw files and test-channel logs.

## Anomaly discussion
{anomaly_text}

The anomaly detector flags missing cycle gaps, negative capacities, Coulombic-efficiency values outside the configured range, abnormal retention drops, and cell-level statistical outliers. These flags are diagnostic aids and should be confirmed against the original measurement files.

## Possible scientific/technical explanations
Possible explanations for poor retention or unstable Coulombic efficiency include side reactions, unstable SEI growth, electrode delamination, increasing polarization, electrolyte wetting issues, temperature variation, channel/contact problems, or protocol differences. These explanations are tentative unless metadata confirm the relevant mechanism.

## Recommended next experiments
1. Repeat or extend cycling for the best and worst cells to confirm reproducibility.
2. Inspect formation-cycle behavior and early-cycle CE for cells with strong capacity loss.
3. Compare electrolyte amount, electrode loading, current density, rest steps, and temperature history for outlier cells.
4. If CC/CV data are available, compare CV fraction growth with aging to identify increasing polarization.
5. Add experimental metadata to the input table so future reports can connect performance with process parameters.

## Final conclusion
The automatic analysis provides an initial ranking and quality-control screen for battery cycling datasets. Scientific conclusions should be finalized after confirming metadata, test protocols, and any flagged suspicious data points.
"""


def generate_openai_report(
    context: dict[str, Any],
    model: str = "gpt-5.2",
    max_output_tokens: int = 1800,
) -> str:
    """Generate a report using the OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_report(context, reason="OPENAI_API_KEY was not found. A local rule-based report was generated instead.")

    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        return fallback_report(context, reason=f"The OpenAI Python package could not be imported: {exc}")

    try:
        client = OpenAI(api_key=api_key, timeout=60.0)
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_report_prompt(context),
            max_output_tokens=max_output_tokens,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        return fallback_report(context, reason="The OpenAI response did not contain output_text. A local rule-based report was generated instead.")
    except Exception as exc:  # noqa: BLE001
        return fallback_report(context, reason=f"OpenAI report generation failed: {exc}")


def list_ollama_models(host: str = "http://localhost:11434") -> tuple[list[str], str | None]:
    """Return locally available Ollama model names, or an error message."""
    try:
        from ollama import Client
    except Exception as exc:  # noqa: BLE001
        return [], f"The ollama Python package is not installed: {exc}"

    try:
        client = Client(host=host)
        response = client.list()
        models: list[str] = []

        # ollama-python response objects have changed slightly across versions.
        raw_models = getattr(response, "models", None)
        if raw_models is None and isinstance(response, dict):
            raw_models = response.get("models", [])

        for item in raw_models or []:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("model") or item.get("name")
            if name:
                models.append(str(name))

        return sorted(set(models)), None
    except Exception as exc:  # noqa: BLE001
        return [], f"Could not connect to Ollama at {host}: {exc}"


def generate_ollama_report(
    context: dict[str, Any],
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
    max_output_tokens: int = 1800,
) -> str:
    """Generate a report using a local Ollama model."""
    try:
        from ollama import Client
    except Exception as exc:  # noqa: BLE001
        return fallback_report(context, reason=f"The ollama Python package could not be imported: {exc}")

    prompt = build_report_prompt(context)
    try:
        client = Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            options={
                "num_predict": int(max_output_tokens),
                "temperature": 0.2,
            },
        )

        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None and isinstance(response, dict):
            content = response.get("message", {}).get("content")

        if content:
            return str(content)
        return fallback_report(context, reason="The local Ollama response did not contain message content. A local rule-based report was generated instead.")
    except Exception as exc:  # noqa: BLE001
        return fallback_report(context, reason=f"Local Ollama report generation failed: {exc}")


def generate_ai_report(
    context: dict[str, Any],
    backend: ReportBackend = "openai",
    openai_model: str = "gpt-5.2",
    ollama_model: str = "llama3.2",
    ollama_host: str = "http://localhost:11434",
    max_output_tokens: int = 1800,
) -> str:
    """Generate an interpretation using the selected backend."""
    if backend == "rule_based":
        return fallback_report(context, reason="Rule-based local analysis was selected. No API or local LLM was used.")
    if backend == "ollama":
        return generate_ollama_report(
            context,
            model=ollama_model,
            host=ollama_host,
            max_output_tokens=max_output_tokens,
        )
    return generate_openai_report(
        context,
        model=openai_model,
        max_output_tokens=max_output_tokens,
    )
