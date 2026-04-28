"""Streamlit app for BatterySense AI battery/cell cycling reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.ai_report import build_ai_context, generate_ai_report, list_ollama_models
from modules.battery_analysis import run_battery_analysis
from modules.column_detection import DISPLAY_NAMES, canonicalize_battery_columns, detect_columns, numeric_columns_from_mapping, validate_mapping
from modules.data_cleaning import DEFAULT_CLEANING_SETTINGS, clean_dataframe
from modules.data_loader import combine_datasets, load_uploaded_files, profile_dataframe, profile_loaded_datasets
from modules.export_pdf import save_pdf_report
from modules.generic_analysis import generic_dataset_summary
from modules.html_export import save_html_report
from modules.local_ml_analysis import isolation_forest_anomalies, kmeans_clustering
from modules.plotting import comparison_plot, generate_plots, group_comparison_table
from modules.project_database import list_analysis_runs, save_analysis_run
from modules.report_templates import get_report_templates


PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
DB_PATH = PROJECT_ROOT / "outputs" / "database" / "analysis_runs.sqlite"
LOGO_PATH = PROJECT_ROOT / "assets" / "log_small.png"
BANNER_PATH = PROJECT_ROOT / "assets" / "batterysense_banner.png"

load_dotenv()


def load_settings() -> dict[str, Any]:
    defaults = {
        "app_name": "BatterySense AI",
        "default_ai_backend": "Rule-based local summary",
        "default_openai_model": "gpt-5.2",
        "default_ollama_model": "llama3.2",
        "ollama_host": "http://localhost:11434",
        "max_ai_output_tokens": 1800,
        "report_title": "BatterySense AI Battery Cycling Analysis Report",
        "cleaning": DEFAULT_CLEANING_SETTINGS,
        "analysis": {
            "capacity_drop_threshold_pct": 10.0,
            "ce_low_threshold_pct": 80.0,
            "ce_high_threshold_pct": 105.0,
            "cycle_life_retention_threshold_pct": 80.0,
        },
    }
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            defaults.update(loaded)
            defaults["cleaning"].update(loaded.get("cleaning", {}))
            defaults["analysis"].update(loaded.get("analysis", {}))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load config/settings.json. Using defaults. Error: {exc}")
    return defaults


def show_dataset_profile(profile: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", profile.get("rows", 0))
    col2.metric("Columns", profile.get("columns", 0))
    col3.metric("Numeric columns", len(profile.get("numeric_columns", [])))
    col4.metric("Categorical columns", len(profile.get("categorical_columns", [])))
    with st.expander("Column names", expanded=False):
        st.write(profile.get("column_names", []))
    missing = pd.DataFrame({"missing_values": profile.get("missing_values", {}), "missing_percent": profile.get("missing_percent", {})}).reset_index(names="column")
    st.dataframe(missing, use_container_width=True, hide_index=True)


def select_column_mapping(df: pd.DataFrame, detected_mapping: dict[str, str | None]) -> dict[str, str | None]:
    st.subheader("Column detection and manual correction")
    st.caption("The app suggests likely battery columns. Correct any field before analysis.")
    columns = [str(col) for col in df.columns]
    options = [""] + columns
    mapping: dict[str, str | None] = {}
    grid = st.columns(2)
    for idx, semantic_name in enumerate(DISPLAY_NAMES):
        detected = detected_mapping.get(semantic_name) or ""
        default_index = options.index(detected) if detected in options else 0
        with grid[idx % 2]:
            selected = st.selectbox(DISPLAY_NAMES[semantic_name], options=options, index=default_index, key=f"mapping_{semantic_name}")
            mapping[semantic_name] = selected or None
    for warning in validate_mapping(df, mapping).values():
        st.warning(warning)
    return mapping


def render_metrics(summary: dict[str, Any]) -> None:
    st.subheader("Analysis summary")
    cols = st.columns(5)
    cols[0].metric("Cells / samples", summary.get("n_cells_or_samples", 0))
    cols[1].metric("Anomaly flags", summary.get("n_anomalies", 0))
    cols[2].metric("Excluded rows", summary.get("excluded_rows", 0))
    cols[3].metric("Best cell", summary.get("best_performing_cell") or "N/A")
    cols[4].metric("Worst cell", summary.get("worst_performing_cell") or "N/A")
    c1, c2 = st.columns(2)
    mean_retention = summary.get("mean_last_capacity_retention_pct")
    mean_degradation = summary.get("mean_degradation_rate_pct_per_cycle")
    c1.metric("Mean final retention", "N/A" if mean_retention is None else f"{mean_retention:.2f}%")
    c2.metric("Mean degradation slope", "N/A" if mean_degradation is None else f"{mean_degradation:.4f}% / cycle")


def configure_sidebar(settings: dict[str, Any]) -> dict[str, Any]:
    with st.sidebar:
        st.header("Settings")
        analysis_mode = st.radio("Analysis mode", ["Battery cycling analysis", "Generic dataset analysis"], index=0)
        backend_labels = {"Rule-based local summary": "rule_based", "Local LLM with Ollama": "ollama", "OpenAI API": "openai"}
        selected_backend_label = st.radio("AI report backend", list(backend_labels.keys()), index=list(backend_labels.keys()).index(settings.get("default_ai_backend", "Rule-based local summary")))
        ai_backend = backend_labels[selected_backend_label]
        openai_model = settings.get("default_openai_model", "gpt-5.2")
        ollama_model = settings.get("default_ollama_model", "llama3.2")
        ollama_host = settings.get("ollama_host", "http://localhost:11434")
        if ai_backend == "openai":
            openai_model = st.text_input("OpenAI model", value=openai_model)
        elif ai_backend == "ollama":
            ollama_host = st.text_input("Ollama host", value=ollama_host)
            local_models, err = list_ollama_models(ollama_host)
            if local_models:
                ollama_model = st.selectbox("Local Ollama model", local_models, index=local_models.index(ollama_model) if ollama_model in local_models else 0)
            else:
                ollama_model = st.text_input("Local Ollama model", value=ollama_model)
                st.warning(f"No Ollama models detected. {err or ''}")

        max_output_tokens = st.number_input("Max interpretation output tokens", 500, 6000, int(settings.get("max_ai_output_tokens", 1800)), 100)

        st.subheader("Protocol sections to analyze")
        include_segments = st.multiselect("Include segments in metrics/plots", ["formation", "rate_test", "cycling"], default=["cycling"], help="Use cycling only for publication-style long-term metrics. Include formation/rate-test when you want full-protocol plots.")

        st.subheader("Cleaning and quality filters")
        cleaning_settings = dict(settings.get("cleaning", DEFAULT_CLEANING_SETTINGS))
        cleaning_settings["drop_all_empty_rows"] = st.checkbox("Drop all-empty rows", bool(cleaning_settings.get("drop_all_empty_rows", True)))
        cleaning_settings["drop_duplicate_rows"] = st.checkbox("Drop duplicate rows", bool(cleaning_settings.get("drop_duplicate_rows", False)))
        cleaning_settings["remove_suspicious_points"] = st.checkbox("Exclude NaN/non-finite/extreme suspicious points", bool(cleaning_settings.get("remove_suspicious_points", True)))
        cleaning_settings["terminal_change_threshold_pct"] = st.number_input("Incomplete final-cycle threshold (%)", 5.0, 95.0, float(cleaning_settings.get("terminal_change_threshold_pct", 30.0)), 5.0)
        cleaning_settings["formation_max_cycles"] = st.number_input("Formation cycles estimate", 1, 10, int(cleaning_settings.get("formation_max_cycles", 3)), 1)

        st.subheader("Battery analysis thresholds")
        analysis_settings = dict(settings.get("analysis", {}))
        analysis_settings["capacity_drop_threshold_pct"] = st.number_input("Abnormal capacity drop threshold (%)", 1.0, 50.0, float(analysis_settings.get("capacity_drop_threshold_pct", 10.0)), 1.0)
        analysis_settings["ce_low_threshold_pct"] = st.number_input("Low CE threshold (%)", 0.0, 100.0, float(analysis_settings.get("ce_low_threshold_pct", 80.0)), 1.0)
        analysis_settings["ce_high_threshold_pct"] = st.number_input("High CE threshold (%)", 100.0, 150.0, float(analysis_settings.get("ce_high_threshold_pct", 105.0)), 1.0)
        analysis_settings["cycle_life_retention_threshold_pct"] = st.number_input("Cycle-life retention threshold (%)", 1.0, 100.0, float(analysis_settings.get("cycle_life_retention_threshold_pct", 80.0)), 1.0)

    return {
        "analysis_mode": analysis_mode,
        "ai_backend": ai_backend,
        "openai_model": openai_model,
        "ollama_model": ollama_model,
        "ollama_host": ollama_host,
        "max_output_tokens": int(max_output_tokens),
        "include_segments": include_segments,
        "cleaning_settings": cleaning_settings,
        "analysis_settings": analysis_settings,
    }


def main() -> None:
    settings = load_settings()
    st.set_page_config(page_title=settings["app_name"], layout="wide", initial_sidebar_state="expanded")
    if BANNER_PATH.exists():
        st.image(str(BANNER_PATH), use_container_width=True)
    st.title("BatterySense AI")
    st.markdown("Professional AI-assisted battery data analysis, quality control, visualization, and report generation.")

    ui = configure_sidebar(settings)

    uploaded_files = st.file_uploader("Upload one or more CSV/XLSX files", type=["csv", "xlsx", "xlsm"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Upload cycling data files to begin, or test the app with `sample_data/sample_battery_cycling.csv`.")
        return

    try:
        loaded_datasets = load_uploaded_files(uploaded_files)
        uploaded_file_names = sorted({item.source_file for item in loaded_datasets})
        combined_raw = combine_datasets(loaded_datasets)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load uploaded files: {exc}")
        return

    st.header("1. Data preview")
    st.dataframe(combined_raw.head(200), use_container_width=True)

    st.header("2. Dataset inspection")
    combined_profile = profile_dataframe(combined_raw)
    show_dataset_profile(combined_profile)
    with st.expander("Per-file / per-sheet profiles", expanded=False):
        per_dataset_profiles = profile_loaded_datasets(loaded_datasets)
        tabs = st.tabs(list(per_dataset_profiles.keys()))
        for tab, (name, profile) in zip(tabs, per_dataset_profiles.items()):
            with tab:
                st.write(f"**{name}**")
                show_dataset_profile(profile)

    if ui["analysis_mode"] == "Generic dataset analysis":
        st.header("Generic analysis")
        generic = generic_dataset_summary(combined_raw)
        st.json({k: v for k, v in generic.items() if k != "numeric_describe"})
        if generic["numeric_describe"]:
            st.dataframe(pd.DataFrame(generic["numeric_describe"]), use_container_width=True)
        return

    st.header("3. Column standardization")
    detected_mapping = detect_columns(combined_raw)
    user_mapping = select_column_mapping(combined_raw, detected_mapping)
    standardized_raw, mapping = canonicalize_battery_columns(combined_raw, user_mapping)
    st.caption("Canonical columns are added internally so downstream modules can handle Neware, Arbin, BioLogic, and custom exports consistently.")

    numeric_cols = numeric_columns_from_mapping(mapping)
    cleaned = clean_dataframe(
        standardized_raw,
        numeric_columns=numeric_cols,
        drop_all_empty_rows=ui["cleaning_settings"]["drop_all_empty_rows"],
        drop_duplicate_rows=ui["cleaning_settings"]["drop_duplicate_rows"],
        mapping=mapping,
        cleaning_settings=ui["cleaning_settings"],
    )

    st.header("4. Data cleaning and protocol segmentation")
    qcols = [c for c in ["source_file", "source_sheet", "cell_name", "cycle_number", "protocol_segment", "analysis_include", "quality_flag", "exclude_reason"] if c in cleaned.columns]
    if qcols:
        st.dataframe(cleaned[qcols].head(500), use_container_width=True, hide_index=True)
    if "protocol_segment" in cleaned.columns:
        st.bar_chart(cleaned["protocol_segment"].value_counts())
    excluded = cleaned[~cleaned.get("analysis_include", pd.Series(True, index=cleaned.index)).fillna(False)]
    if not excluded.empty:
        st.warning(f"{len(excluded)} row(s) were excluded from plots/metrics by the quality filter. They remain visible in the processed table and anomaly report.")

    st.header("5. Battery analysis")
    analysis = run_battery_analysis(cleaned, mapping, settings_dict=ui["analysis_settings"], include_segments=ui["include_segments"])
    processed = analysis["processed_data"]
    metrics = analysis["cell_metrics"]
    anomalies = analysis["anomalies"]
    render_metrics(analysis["summary"])
    st.subheader("Cell/sample metrics")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.subheader("Detected anomalies and excluded points")
    st.dataframe(anomalies, use_container_width=True, hide_index=True)

    st.header("6. Comparison tools")
    group_col = "cell_or_sample"
    if not metrics.empty and group_col in metrics.columns:
        cells = metrics[group_col].astype(str).tolist()
        selected_cells = st.multiselect("One-to-one cell comparison", cells, default=cells[: min(2, len(cells))])
        cmp_metric = st.selectbox("Comparison y-axis", [c for c in ["capacity_retention_pct", "discharge_capacity", "charge_capacity", "calculated_coulombic_efficiency_pct", "coulombic_efficiency"] if c in processed.columns])
        cmp_fig = comparison_plot(processed, mapping, selected_cells, cmp_metric, logo_path=LOGO_PATH)
        if cmp_fig:
            st.plotly_chart(cmp_fig, use_container_width=True, config={"displaylogo": False})
        with st.expander("Group comparison", expanded=False):
            st.caption("Assign cells to groups such as Material A triplet vs Material B triplet.")
            group_text = st.text_area("Group mapping, one per line: cell_name = group_name", value="")
            group_mapping = {}
            for line in group_text.splitlines():
                if "=" in line:
                    cell, grp = line.split("=", 1)
                    group_mapping[cell.strip()] = grp.strip()
            if group_mapping:
                group_table = group_comparison_table(metrics, group_mapping)
                st.dataframe(group_table, use_container_width=True, hide_index=True)

    st.header("7. Local ML checks")
    with st.expander("Run optional local scikit-learn analysis", expanded=False):
        if st.button("Run local ML anomaly detection"):
            ml_anoms = isolation_forest_anomalies(processed, contamination=0.05)
            st.dataframe(ml_anoms.head(200), use_container_width=True)
        if st.button("Run local clustering"):
            clusters = kmeans_clustering(metrics, n_clusters=min(3, max(1, len(metrics))))
            st.dataframe(clusters, use_container_width=True)

    st.header("8. Interactive plots")
    plots = generate_plots(processed, analysis, mapping, logo_path=LOGO_PATH)
    if not plots:
        st.warning("No plots could be generated. Check cycle/capacity/CE mappings.")
    else:
        plot_tabs = st.tabs([name.replace("_", " ").title() for name in plots])
        for tab, (name, fig) in zip(plot_tabs, plots.items()):
            with tab:
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})

    st.header("9. AI / automatic scientific interpretation")
    ai_context = build_ai_context(combined_profile, mapping, analysis, uploaded_file_names)
    if "interpretation_text" not in st.session_state:
        st.session_state.interpretation_text = ""
    if st.button("Generate interpretation", type="primary"):
        with st.spinner("Generating report interpretation..."):
            st.session_state.interpretation_text = generate_ai_report(
                ai_context,
                backend=ui["ai_backend"],
                openai_model=ui["openai_model"],
                ollama_model=ui["ollama_model"],
                ollama_host=ui["ollama_host"],
                max_output_tokens=ui["max_output_tokens"],
            )
    if st.session_state.interpretation_text:
        st.markdown(st.session_state.interpretation_text)

    st.header("10. Export reports")
    templates = get_report_templates()
    report_title = st.text_input("Report title", value=settings.get("report_title", "BatterySense AI Battery Cycling Analysis Report"))
    report_template = st.selectbox("Report template", list(templates.keys()), format_func=lambda x: f"{x} — {templates[x].description}")
    export_pdf = st.checkbox("Also export PDF", value=True)
    save_to_db = st.checkbox("Save historical run in local SQLite database", value=True)

    if st.button("Export report"):
        interpretation = st.session_state.interpretation_text or generate_ai_report(
            ai_context,
            backend=ui["ai_backend"],
            openai_model=ui["openai_model"],
            ollama_model=ui["ollama_model"],
            ollama_host=ui["ollama_host"],
            max_output_tokens=ui["max_output_tokens"],
        )
        try:
            html_path = save_html_report(
                title=report_title,
                uploaded_file_names=uploaded_file_names,
                dataset_profile=combined_profile,
                mapping=mapping,
                analysis=analysis,
                plots=plots,
                interpretation_text=interpretation,
                output_dir=OUTPUT_REPORT_DIR,
                banner_path=BANNER_PATH,
            )
            pdf_path = None
            if export_pdf:
                pdf_path = save_pdf_report(report_title, uploaded_file_names, combined_profile, analysis, plots, interpretation, output_dir=OUTPUT_REPORT_DIR)
            if save_to_db:
                save_analysis_run(DB_PATH, report_title, uploaded_file_names, {**ui, "report_template": report_template}, analysis, str(html_path), str(pdf_path) if pdf_path else None)
            st.success(f"HTML report saved: {html_path}")
            st.download_button("Download HTML report", data=html_path.read_bytes(), file_name=html_path.name, mime="text/html")
            if pdf_path:
                st.success(f"PDF report saved: {pdf_path}")
                st.download_button("Download PDF report", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to export report: {exc}")

    with st.expander("Historical analysis runs", expanded=False):
        st.dataframe(pd.DataFrame(list_analysis_runs(DB_PATH, limit=20)), use_container_width=True)


if __name__ == "__main__":
    main()
