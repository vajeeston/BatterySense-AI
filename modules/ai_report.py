"""AI-assisted scientific report generation with selectable backends.
BatterySense AI
Author: AU P. Vajeeston, 2026

Python calculates all metrics locally first. AI receives only compact,
Python-calculated evidence and must not interpret raw cycle-level data directly.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

import pandas as pd

ReportBackend = Literal["openai", "ollama", "rule_based"]

SYSTEM_INSTRUCTIONS = """You are BatterySense AI, a senior battery research data analyst.
Write professional, evidence-based scientific interpretations of battery cycling data.
Use only the supplied Python-calculated evidence. Do not invent cells, experiments,
chemistry, C-rates, methods, or anomaly counts. Always name the actual cells/samples
supplied in the evidence. If metadata are missing, state that mechanistic explanation
is tentative.
"""

REQUIRED_SECTIONS = [
    "Executive summary",
    "Data and protocol overview",
    "Formation and early-cycle behaviour",
    "Capacity evolution and SOH",
    "Coulombic efficiency and irreversible capacity",
    "Cell-to-cell comparison and reproducibility",
    "Anomaly and data-quality assessment",
    "Scientific interpretation",
    "Recommended next experiments",
    "Final conclusion",
]

KEY_METRIC_COLUMNS = [
    "cell_or_sample",
    "n_points",
    "first_cycle",
    "last_cycle",
    "first_charge_or_lithiation_capacity",
    "first_discharge_or_delithiation_capacity",
    "first_coulombic_efficiency_pct",
    "max_charge_or_lithiation_capacity",
    "max_discharge_or_delithiation_capacity",
    "last_retention_capacity",
    "last_capacity_retention_from_first_pct",
    "last_capacity_retention_from_max_pct",
    "last_capacity_retention_pct",
    "degradation_rate_pct_per_cycle",
    "average_coulombic_efficiency_pct",
    "last_coulombic_efficiency_pct",
    "last_accumulated_ce_pct",
    "total_accumulated_irreversible_capacity",
    "estimated_cycle_life_to_80pct",
]


def _round_value(value: Any, ndigits: int = 4) -> Any:
    if value is None or isinstance(value, (str, int)):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return round(float(value), ndigits)
    except Exception:
        return value


def _fmt(value: Any, ndigits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    try:
        return f"{float(value):.{ndigits}f}{suffix}"
    except Exception:
        return str(value)


def dataframe_to_records(df: pd.DataFrame | None, max_rows: int = 25, columns: list[str] | None = None) -> list[dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    limited = df.copy()
    if columns:
        limited = limited[[c for c in columns if c in limited.columns]]
    limited = limited.head(max_rows)
    records = limited.where(pd.notna(limited), None).to_dict(orient="records")
    return [{str(k): _round_value(v) for k, v in row.items()} for row in records]


def _markdown_table(records: list[dict[str, Any]], columns: list[str] | None = None, max_rows: int = 20) -> str:
    if not records:
        return "Not available."
    if columns is None:
        columns = list(records[0].keys())
    columns = [c for c in columns if any(c in r for r in records)]
    if not columns:
        return "Not available."
    rows = records[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "N/A")
            vals.append(_fmt(v, 4) if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def _safe_group_column(data: pd.DataFrame) -> str | None:
    for col in ["cell_or_sample", "source_file", "cell_name", "sample_name"]:
        if col in data.columns:
            return col
    return None


def _summarize_protocol_segments(processed: pd.DataFrame, max_rows: int = 60) -> pd.DataFrame:
    if processed is None or not isinstance(processed, pd.DataFrame) or processed.empty:
        return pd.DataFrame()
    data = processed.copy()
    group_col = _safe_group_column(data)
    if not group_col:
        data["cell_or_sample"] = "All data"
        group_col = "cell_or_sample"
    segment_col = "protocol_segment" if "protocol_segment" in data.columns else None
    if not segment_col:
        data["protocol_segment"] = "not_detected"
        segment_col = "protocol_segment"
    cycle_col = "cycle_number" if "cycle_number" in data.columns else "Cycle" if "Cycle" in data.columns else None
    cap_col = "retention_capacity" if "retention_capacity" in data.columns else "discharge_or_delithiation_capacity" if "discharge_or_delithiation_capacity" in data.columns else None
    charge_col = "charge_or_lithiation_capacity" if "charge_or_lithiation_capacity" in data.columns else None
    ce_col = "calculated_coulombic_efficiency_pct" if "calculated_coulombic_efficiency_pct" in data.columns else "CE(%)" if "CE(%)" in data.columns else None
    ret_col = "capacity_retention_pct" if "capacity_retention_pct" in data.columns else None

    rows: list[dict[str, Any]] = []
    for (cell, segment), grp in data.groupby([group_col, segment_col], dropna=False):
        g = grp.copy()
        if cycle_col and cycle_col in g.columns:
            g = g.sort_values(cycle_col)
        row: dict[str, Any] = {"cell_or_sample": str(cell), "segment": str(segment), "n_points": int(len(g))}
        if cycle_col and cycle_col in g.columns:
            cycles = pd.to_numeric(g[cycle_col], errors="coerce").dropna()
            row["cycle_min"] = _round_value(cycles.min()) if not cycles.empty else None
            row["cycle_max"] = _round_value(cycles.max()) if not cycles.empty else None
        for label, col in [
            ("charge_mean", charge_col),
            ("charge_max", charge_col),
            ("output_capacity_start", cap_col),
            ("output_capacity_end", cap_col),
            ("output_capacity_max", cap_col),
            ("ce_mean_pct", ce_col),
            ("retention_end_pct", ret_col),
        ]:
            if not col or col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
            if vals.empty:
                row[label] = None
            elif label.endswith("_start"):
                row[label] = _round_value(vals.iloc[0])
            elif label.endswith("_end"):
                row[label] = _round_value(vals.iloc[-1])
            elif label.endswith("_max"):
                row[label] = _round_value(vals.max())
            else:
                row[label] = _round_value(vals.mean())
        rows.append(row)
    return pd.DataFrame(rows).head(max_rows)


def _summarize_anomaly_counts(anomalies: pd.DataFrame) -> list[dict[str, Any]]:
    if anomalies is None or not isinstance(anomalies, pd.DataFrame) or anomalies.empty:
        return []
    cols = [c for c in ["type", "severity"] if c in anomalies.columns]
    if not cols:
        return []
    counts = anomalies.groupby(cols, dropna=False).size().reset_index(name="count")
    return dataframe_to_records(counts.sort_values("count", ascending=False), max_rows=30)


def build_ai_context(dataset_profile: dict[str, Any], mapping: dict[str, str | None], analysis: dict[str, Any], uploaded_file_names: list[str]) -> dict[str, Any]:
    metrics = analysis.get("cell_metrics", pd.DataFrame())
    anomalies = analysis.get("anomalies", pd.DataFrame())
    processed = analysis.get("processed_data", pd.DataFrame())
    deviation_report = analysis.get("deviation_report", {}) or {}
    deviation_std = deviation_report.get("std_table") if isinstance(deviation_report, dict) else pd.DataFrame()
    deviation_per_cell = deviation_report.get("per_cell_deviation") if isinstance(deviation_report, dict) else pd.DataFrame()
    segment_summary = _summarize_protocol_segments(processed)

    if isinstance(metrics, pd.DataFrame) and not metrics.empty and "last_capacity_retention_pct" in metrics.columns:
        best_cells = metrics.sort_values("last_capacity_retention_pct", ascending=False, na_position="last").head(5)
        worst_cells = metrics.sort_values("last_capacity_retention_pct", ascending=True, na_position="last").head(5)
    else:
        best_cells = pd.DataFrame()
        worst_cells = pd.DataFrame()

    metric_records = dataframe_to_records(metrics, max_rows=30, columns=KEY_METRIC_COLUMNS)
    context: dict[str, Any] = {
        "uploaded_files": uploaded_file_names,
        "dataset_profile": {
            "rows": dataset_profile.get("rows"),
            "columns": dataset_profile.get("columns"),
            "column_names": dataset_profile.get("column_names"),
            "top_missing_values": dict(list(dataset_profile.get("missing_values", {}).items())[:20]),
        },
        "column_mapping": mapping,
        "overall_summary": analysis.get("summary", {}),
        "cell_names": [str(r.get("cell_or_sample")) for r in metric_records if r.get("cell_or_sample") is not None],
        "top_cells_by_retention": dataframe_to_records(best_cells, max_rows=5, columns=KEY_METRIC_COLUMNS),
        "lowest_cells_by_retention": dataframe_to_records(worst_cells, max_rows=5, columns=KEY_METRIC_COLUMNS),
        "cell_metrics": metric_records,
        "protocol_segment_summary": dataframe_to_records(segment_summary, max_rows=60),
        "deviation_summary": deviation_report.get("summary", {}) if isinstance(deviation_report, dict) else {},
        "deviation_std_table": dataframe_to_records(deviation_std if isinstance(deviation_std, pd.DataFrame) else pd.DataFrame(), max_rows=30),
        "per_cell_deviation": dataframe_to_records(deviation_per_cell if isinstance(deviation_per_cell, pd.DataFrame) else pd.DataFrame(), max_rows=25),
        "anomaly_counts": _summarize_anomaly_counts(anomalies),
        "anomalies": dataframe_to_records(anomalies, max_rows=30),
        "analysis_settings": analysis.get("settings", {}),
    }
    context["evidence_brief"] = build_evidence_brief(context)
    return context


def build_evidence_brief(context: dict[str, Any]) -> str:
    profile = context.get("dataset_profile", {})
    summary = context.get("overall_summary", {})
    cell_names = context.get("cell_names", [])
    cell_metrics = context.get("cell_metrics", [])
    deviation = context.get("deviation_std_table", [])
    segments = context.get("protocol_segment_summary", [])
    anomaly_counts = context.get("anomaly_counts", [])

    lines = [
        "EVIDENCE SHEET: PYTHON-CALCULATED BATTERYSENSE AI RESULTS",
        "",
        "Dataset:",
        f"- Uploaded files: {', '.join(context.get('uploaded_files', [])) or 'N/A'}",
        f"- Rows: {profile.get('rows', 'N/A')}; columns: {profile.get('columns', 'N/A')}",
        f"- Number of detected cells/samples: {summary.get('n_cells_or_samples', len(cell_names) or 'N/A')}",
        f"- Actual cell/sample names: {', '.join(cell_names) or 'N/A'}",
        f"- Battery/cell type selected: {summary.get('battery_cell_type', 'N/A')}",
        f"- Protocol segment counts: {summary.get('segment_counts', {})}",
        f"- Excluded rows after quality filtering: {summary.get('excluded_rows', 'N/A')}",
        "",
        "Overall decision metrics:",
        f"- Best-performing cell by final retention: {summary.get('best_performing_cell', 'N/A')}",
        f"- Worst-performing cell by final retention: {summary.get('worst_performing_cell', 'N/A')}",
        f"- Mean final retention using max-stabilized baseline: {_fmt(summary.get('mean_last_capacity_retention_pct'), 2, '%')}",
        f"- Mean final retention using first-valid baseline: {_fmt(summary.get('mean_last_capacity_retention_from_first_pct'), 2, '%')}",
        f"- Mean calculated CE: {_fmt(summary.get('mean_calculated_ce_pct'), 2, '%')}",
        f"- Mean degradation slope: {_fmt(summary.get('mean_degradation_rate_pct_per_cycle'), 4, '% per cycle')}",
        f"- Number of anomaly flags: {summary.get('n_anomalies', 'N/A')}",
        "",
        "Per-cell metrics table:",
        _markdown_table(cell_metrics, columns=KEY_METRIC_COLUMNS, max_rows=30),
        "",
        "Protocol/segment summary table:",
        _markdown_table(segments, max_rows=60),
        "",
        "Multi-cell reproducibility/deviation table:",
        _markdown_table(deviation, max_rows=30),
        "",
        "Anomaly count summary:",
        _markdown_table(anomaly_counts, max_rows=30),
    ]
    return "\n".join(lines)


def _compose_prompt_intro(custom_prompt: str | None = None) -> str:
    """Return user-editable instructions with strict evidence rules."""
    cp = (custom_prompt or "").strip()
    if cp:
        return f"""USER-SELECTED REPORT PROMPT:
{cp}

Use the user-selected prompt above as the report style and focus, but obey all accuracy rules below."""
    return "Write the final scientific interpretation section for a BatterySense AI report."


def build_report_prompt(context: dict[str, Any], custom_prompt: str | None = None) -> str:
    sections = "\n".join(f"- {section}" for section in REQUIRED_SECTIONS)
    evidence = context.get("evidence_brief") or build_evidence_brief(context)
    actual_cells = ", ".join(context.get("cell_names", [])) or "the detected cells/samples"
    n_cells = context.get("overall_summary", {}).get("n_cells_or_samples", len(context.get("cell_names", [])))
    prompt_intro = _compose_prompt_intro(custom_prompt)
    return f"""{prompt_intro}

Use ONLY the evidence sheet below. The numerical analysis has already been done locally by Python. Do not recalculate and do not infer hidden experimental conditions.

NON-NEGOTIABLE ACCURACY RULES:
- The dataset contains exactly {n_cells} detected cell/sample group(s): {actual_cells}. Do not write any other number of cells, batteries, groups, or samples.
- Use the actual cell names exactly as written at least once in the cell-comparison section. Do not write "Cell 1 (implied)" or rename cells.
- Do not invent values such as average CE, retention, degradation rate, cycle count, anomaly counts, C-rate, chemistry, electrolyte, temperature, loading, tester type, EIS, post-mortem analysis, field testing, or machine-learning results.
- Do not describe JSON, keys, schemas, code, or software structure.
- If a value is missing, write "not available" and explain what metadata would be needed.
- Mention that CE, 100-CE, irreversible capacity, accumulated CE, CE_20_pt_AVG, SOH, estimated C-rate, and both retention baselines are locally calculated from mapped charge/lithiation and discharge/delithiation columns.
- Explain both retention baselines: first-valid capacity and max-stabilized cycling capacity. The main ranking/SOH uses the max-stabilized baseline.
- Discuss reproducibility quantitatively when the deviation table is available, especially STD/RSD in first-cycle CE, first-cycle capacities, upper/max capacities, final retention, and degradation rate.
- Separate observed evidence from tentative electrochemical explanations.

STYLE RULES:
- Start exactly with: # Scientific interpretation
- Use markdown headings beginning with ## for sections.
- Use short paragraphs and a few bullet points where helpful.
- Include important numerical values directly in the text.
- Keep the tone scientific, cautious, and suitable for a battery research report.

Required sections:
{sections}

{evidence}

Now write the scientific interpretation only. Do not include the evidence sheet, markdown tables, code fences, JSON, or a preamble.
"""


def _looks_like_json_schema_explanation(text: str) -> bool:
    lowered = text.lower()
    bad_phrases = [
        "this is a json object",
        "top-level keys",
        "breakdown of the structure",
        "each anomaly is represented",
        "analysis settings are stored",
        "json structure",
        "schema",
    ]
    return any(phrase in lowered for phrase in bad_phrases)


def _looks_hallucinated_or_generic(text: str, context: dict[str, Any]) -> bool:
    lowered = text.lower()
    bad_phrases = [
        "cell 1 (implied)",
        "cell 2 (implied)",
        "cell 3 (implied)",
        "six lithium-ion batteries",
        "six batteries",
        "19 battery cells",
        "field trials",
        "cell balancing",
        "average ce across all cells was 95.6",
        "evidence sheet:",
        "```json",
        "```text",
    ]
    if any(p in lowered for p in bad_phrases):
        return True
    if _looks_like_json_schema_explanation(text):
        return True
    cell_names = [str(x) for x in context.get("cell_names", []) if x]
    if len(cell_names) >= 2:
        mentioned = sum(1 for name in cell_names if name.lower() in lowered)
        if mentioned == 0:
            return True
    return False


def _cell_metric_bullets(context: dict[str, Any], max_cells: int = 8) -> str:
    records = context.get("cell_metrics", []) or []
    if not records:
        return "- Per-cell metrics were not available in the generated context."
    lines = []
    for row in records[:max_cells]:
        cell = row.get("cell_or_sample", "N/A")
        lines.append(
            f"- **{cell}**: first charge/lithiation {_fmt(row.get('first_charge_or_lithiation_capacity'), 2)}, "
            f"first discharge/delithiation {_fmt(row.get('first_discharge_or_delithiation_capacity'), 2)}, "
            f"first CE {_fmt(row.get('first_coulombic_efficiency_pct'), 2, '%')}, "
            f"max discharge/delithiation {_fmt(row.get('max_discharge_or_delithiation_capacity'), 2)}, "
            f"last capacity {_fmt(row.get('last_retention_capacity'), 2)}, "
            f"retention from first {_fmt(row.get('last_capacity_retention_from_first_pct'), 2, '%')}, "
            f"retention from max-stabilized {_fmt(row.get('last_capacity_retention_from_max_pct'), 2, '%')}, "
            f"average CE {_fmt(row.get('average_coulombic_efficiency_pct'), 3, '%')}, "
            f"degradation slope {_fmt(row.get('degradation_rate_pct_per_cycle'), 4, '% per cycle')}."
        )
    return "\n".join(lines)


def _deviation_bullets(context: dict[str, Any], max_items: int = 10) -> str:
    records = context.get("deviation_std_table", []) or []
    if not records:
        return "- Multi-cell deviation statistics were not available or only one cell was analyzed."
    lines = []
    for row in records[:max_items]:
        param = row.get("parameter", "parameter")
        lines.append(
            f"- **{param}**: mean {_fmt(row.get('mean'), 3)}, STD {_fmt(row.get('std'), 3)}, "
            f"RSD {_fmt(row.get('rsd_pct'), 2, '%')}, range {_fmt(row.get('range'), 3)}."
        )
    return "\n".join(lines)


def fallback_report(context: dict[str, Any], reason: str | None = None) -> str:
    summary = context.get("overall_summary", {})
    best = summary.get("best_performing_cell") or "Not available"
    worst = summary.get("worst_performing_cell") or "Not available"
    n_anomalies = summary.get("n_anomalies", 0)
    backend_note = f"\n**Backend status:** {reason}\n" if reason else ""
    per_cell = _cell_metric_bullets(context)
    deviation = _deviation_bullets(context)

    return f"""# Scientific interpretation
{backend_note}
## Executive summary
The BatterySense AI local analysis identified **{summary.get('n_cells_or_samples', 'N/A')} cell/sample group(s)**. Based on the max-stabilized capacity-retention baseline, the best-performing cell/sample is **{best}**, while the weakest cell/sample is **{worst}**. The mean final retention is **{_fmt(summary.get('mean_last_capacity_retention_pct'), 2, '%')}**, the mean first-baseline retention is **{_fmt(summary.get('mean_last_capacity_retention_from_first_pct'), 2, '%')}**, and the mean calculated Coulombic efficiency is **{_fmt(summary.get('mean_calculated_ce_pct'), 2, '%')}**.

## Data and protocol overview
The uploaded data were cleaned, standardized, and analyzed locally. CE and related quantities were calculated from the mapped charge/lithiation and discharge/delithiation capacities using the selected cell-type convention. The dataset includes the following protocol-segment counts: **{summary.get('segment_counts', {})}**. Excluded rows after quality filtering: **{summary.get('excluded_rows', 'N/A')}**.

## Formation and early-cycle behaviour
First-cycle charge/lithiation capacity, discharge/delithiation capacity, and first-cycle CE are important indicators of irreversible losses during initial conditioning. In this analysis, the first-cycle and upper-capacity values are reported per cell so that formation reproducibility can be assessed directly.

{per_cell}

## Capacity evolution and SOH
BatterySense AI reports two capacity-retention baselines. The first-valid baseline shows change relative to the first valid output capacity, while the max-stabilized baseline compares each later point with the maximum stable capacity observed during cycling. The max-stabilized baseline is used for the main SOH and final ranking because it is often more representative after formation stabilization.

## Coulombic efficiency and irreversible capacity
Average CE, accumulated CE, CE_20_pt_AVG, and accumulated irreversible capacity should be interpreted together. Stable CE near 100% generally indicates good reversibility, whereas low or unstable CE can indicate irreversible side reactions, wetting/formation issues, or measurement instability. These mechanistic explanations remain tentative without metadata such as temperature, electrolyte, loading, and protocol details.

## Cell-to-cell comparison and reproducibility
The deviation table quantifies reproducibility using STD and RSD across cells. Lower STD/RSD in first-cycle CE, first-cycle capacities, upper/max capacities, final retention, and degradation slope indicates more reproducible cell behaviour.

{deviation}

## Anomaly and data-quality assessment
BatterySense AI detected **{n_anomalies}** anomaly flag(s). These should be treated as diagnostic flags rather than final failure assignments. Confirm flagged points against the original tester files and channel logs before making final conclusions.

## Scientific interpretation
The observed differences between cells should be interpreted primarily from retention, degradation slope, CE stability, and reproducibility metrics. Large first-cycle capacity or CE differences suggest formation or electrode/process variability, while large divergence in final retention or degradation slope indicates differences in long-term cycling stability.

## Recommended next experiments
1. Repeat or extend cycling for the best and worst cells to confirm reproducibility.
2. Inspect early-cycle CE and irreversible capacity to separate formation losses from long-term degradation.
3. Compare electrode loading, electrolyte amount, rest steps, current density, and temperature history for cells with poorer retention.
4. If available, add CC/CV split, dQ/dV, impedance, and metadata to improve mechanistic interpretation.
5. Validate anomaly flags against original tester logs before excluding or diagnosing a cell.

## Final conclusion
The report provides an evidence-based ranking and quality-control screen. The strongest conclusion is the relative performance ranking from the locally calculated metrics; mechanistic explanations should be confirmed with protocol metadata and follow-up diagnostics.
"""


def _repair_prompt(context: dict[str, Any], previous_text: str, custom_prompt: str | None = None) -> str:
    return (
        "The previous answer was too generic or contradicted the BatterySense evidence. "
        "Rewrite it as a professional battery research interpretation using the exact cell names and exact numbers. "
        "Do not include JSON, code fences, evidence tables, or software explanations. "
        "Start exactly with '# Scientific interpretation'.\n\n"
        + build_report_prompt(context, custom_prompt=custom_prompt)
        + "\n\nPrevious answer to avoid:\n" + previous_text[:1200]
    )


def generate_openai_report(context: dict[str, Any], model: str = "gpt-5.2", max_output_tokens: int = 3500, custom_prompt: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_report(context, reason="OPENAI_API_KEY was not found. A local rule-based report was generated instead.")
    try:
        from openai import OpenAI
    except Exception as exc:
        return fallback_report(context, reason=f"The OpenAI Python package could not be imported: {exc}")
    try:
        client = OpenAI(api_key=api_key, timeout=90.0)
        prompt = build_report_prompt(context, custom_prompt=custom_prompt)
        response = client.responses.create(model=model, instructions=SYSTEM_INSTRUCTIONS, input=prompt, max_output_tokens=max_output_tokens)
        output_text = str(getattr(response, "output_text", "") or "").strip()
        if output_text and not _looks_hallucinated_or_generic(output_text, context):
            return output_text
        if output_text:
            retry = client.responses.create(model=model, instructions=SYSTEM_INSTRUCTIONS, input=_repair_prompt(context, output_text, custom_prompt=custom_prompt), max_output_tokens=max_output_tokens)
            retry_text = str(getattr(retry, "output_text", "") or "").strip()
            if retry_text and not _looks_hallucinated_or_generic(retry_text, context):
                return retry_text
        return fallback_report(context, reason="The AI response did not pass the evidence-consistency check, so BatterySense AI generated the deterministic evidence-based interpretation below.")
    except Exception as exc:
        return fallback_report(context, reason=f"OpenAI report generation failed: {exc}")


def list_ollama_models(host: str = "http://localhost:11434") -> tuple[list[str], str | None]:
    try:
        from ollama import Client
    except Exception as exc:
        return [], f"The ollama Python package is not installed: {exc}"
    try:
        client = Client(host=host)
        response = client.list()
        models: list[str] = []
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
    except Exception as exc:
        return [], f"Could not connect to Ollama at {host}: {exc}"


def _ollama_chat(client: Any, model: str, prompt: str, max_output_tokens: int, temperature: float = 0.10) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_INSTRUCTIONS}, {"role": "user", "content": prompt}],
        stream=False,
        options={
            "num_predict": int(max_output_tokens),
            "temperature": temperature,
            "top_p": 0.85,
            "repeat_penalty": 1.08,
            "num_ctx": 8192,
        },
    )
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(response, dict):
        content = response.get("message", {}).get("content")
    return str(content or "").strip()


def generate_ollama_report(context: dict[str, Any], model: str = "llama3.2", host: str = "http://localhost:11434", max_output_tokens: int = 3500, custom_prompt: str | None = None) -> str:
    try:
        from ollama import Client
    except Exception as exc:
        return fallback_report(context, reason=f"The ollama Python package could not be imported: {exc}")
    try:
        client = Client(host=host)
        prompt = build_report_prompt(context, custom_prompt=custom_prompt)
        content_text = _ollama_chat(client, model, prompt, max_output_tokens=max_output_tokens, temperature=0.10)
        if content_text and not _looks_hallucinated_or_generic(content_text, context):
            return content_text
        if content_text:
            retry_text = _ollama_chat(client, model, _repair_prompt(context, content_text, custom_prompt=custom_prompt), max_output_tokens=max_output_tokens, temperature=0.05)
            if retry_text and not _looks_hallucinated_or_generic(retry_text, context):
                return retry_text
        return fallback_report(context, reason="The local Ollama model response did not pass the evidence-consistency check, so BatterySense AI generated the deterministic evidence-based interpretation below.")
    except Exception as exc:
        return fallback_report(context, reason=f"Local Ollama report generation failed: {exc}")


def generate_ai_report(context: dict[str, Any], backend: ReportBackend = "openai", openai_model: str = "gpt-5.2", ollama_model: str = "llama3.2", ollama_host: str = "http://localhost:11434", max_output_tokens: int = 3500, custom_prompt: str | None = None) -> str:
    if backend == "rule_based":
        return fallback_report(context, reason="Rule-based local analysis was selected. No API or local LLM was used.")
    if backend == "ollama":
        return generate_ollama_report(context, model=ollama_model, host=ollama_host, max_output_tokens=max_output_tokens, custom_prompt=custom_prompt)
    return generate_openai_report(context, model=openai_model, max_output_tokens=max_output_tokens, custom_prompt=custom_prompt)
