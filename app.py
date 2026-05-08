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
from modules.data_cleaning import DEFAULT_CLEANING_SETTINGS, apply_protocol_segment_overrides, clean_dataframe
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
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "config" / "prompt_templates.json"
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
        "max_ai_output_tokens": 5000,
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




def load_prompt_templates() -> dict[str, Any]:
    """Load editable AI prompt templates from config/prompt_templates.json."""
    if PROMPT_TEMPLATE_PATH.exists():
        try:
            loaded = json.loads(PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load config/prompt_templates.json. Error: {exc}")
    return {
        "battery_cycling_report": {
            "name": "Battery cycling scientific report",
            "prompt": "You are a battery research data analyst. Write a scientific report using only calculated BatterySense AI metrics. Do not invent values.",
        },
        "lab_qc_progress_report": {
            "name": "Lab QC progress report",
            "prompt": "You are a battery lab quality-control analyst. Write a concise monthly progress report using only the calculated QC evidence. Explain CE, STD, CV, mass loading, lithiation/delithiation variability, long-term cycling indicators, and recommended actions.",
        },
    }


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
    c1, c2, c3, c4 = st.columns(4)
    mean_retention = summary.get("mean_last_capacity_retention_pct")
    mean_retention_first = summary.get("mean_last_capacity_retention_from_first_pct")
    mean_degradation = summary.get("mean_degradation_rate_pct_per_cycle")
    mean_ce = summary.get("mean_calculated_ce_pct")
    c1.metric("Mean final retention, max baseline", "N/A" if mean_retention is None else f"{mean_retention:.2f}%")
    c2.metric("Mean final retention, first baseline", "N/A" if mean_retention_first is None else f"{mean_retention_first:.2f}%")
    c3.metric("Mean degradation slope", "N/A" if mean_degradation is None else f"{mean_degradation:.4f}% / cycle")
    c4.metric("Mean calculated CE", "N/A" if mean_ce is None else f"{mean_ce:.2f}%")



def render_footer(settings: dict[str, Any]) -> None:
    year = settings.get("copyright_year", 2026)
    owner = settings.get("copyright_owner", "AU P. Vajeeston")
    st.markdown("---")
    st.markdown(
        f"<div style=\"text-align:center; color:#6B7280; font-size:0.85rem; padding:0.75rem;\">"
        f"© {year} {owner}. BatterySense AI is free for non-commercial research, education, and personal use. "
        "Commercial use requires prior written permission.</div>",
        unsafe_allow_html=True,
    )


def render_progress_radial(placeholder: Any, percent: int, label: str) -> None:
    """Render a compact radial progress indicator in Streamlit."""
    percent = max(0, min(100, int(percent)))
    placeholder.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.85rem; margin:0.5rem 0 1rem 0;">
          <div style="
            width:72px; height:72px; border-radius:50%;
            background: conic-gradient(#00A878 {percent * 3.6}deg, #E5E7EB 0deg);
            display:flex; align-items:center; justify-content:center;
            box-shadow: 0 4px 14px rgba(16,24,40,0.12);">
            <div style="
              width:52px; height:52px; border-radius:50%; background:white;
              display:flex; align-items:center; justify-content:center;
              font-weight:700; color:#12395d; font-size:0.9rem;">
              {percent}%
            </div>
          </div>
          <div>
            <div style="font-weight:700; color:#12395d;">{label}</div>
            <div style="font-size:0.85rem; color:#667085;">BatterySense AI processing status</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def configure_sidebar(settings: dict[str, Any]) -> dict[str, Any]:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=120)
        st.header("Settings")
        analysis_mode = st.radio("Analysis mode", ["Battery cycling analysis", "Generic dataset analysis"], index=0)
        dataset_analysis_type = st.radio(
            "Battery dataset comparison type",
            ["Single cell", "Multi cells / reproducibility"],
            index=1,
            help="Use Single cell for one cell only. Use Multi cells to generate STD and deviation reports across repeated cells or groups.",
        )
        battery_cell_options = {
            "Full cell": "full_cell",
            "Anode half-cell": "anode_half_cell",
            "Cathode half-cell": "cathode_half_cell",
            "Generic half-cell / custom": "half_cell",
        }
        default_cell_type = settings.get("default_battery_cell_type", "full_cell")
        default_label = next((label for label, value in battery_cell_options.items() if value == default_cell_type), "Full cell")
        battery_cell_label = st.radio(
            "Uploaded battery data type",
            list(battery_cell_options.keys()),
            index=list(battery_cell_options.keys()).index(default_label),
            help=(
                "Full cell: CE = discharge/charge × 100. "
                "Anode half-cell: CE = delithiation/lithiation × 100, commonly charge/discharge. "
                "Cathode half-cell: CE = lithiation/delithiation × 100, commonly discharge/charge. "
                "Generic half-cell uses the neutral reversible/inserted capacity convention."
            ),
        )
        battery_cell_type = battery_cell_options[battery_cell_label]
        st.caption(
            {
                "full_cell": "CE basis: discharge capacity / charge capacity × 100.",
                "anode_half_cell": "CE basis: delithiation capacity / lithiation capacity × 100. For many tester exports this is charge capacity / discharge capacity.",
                "cathode_half_cell": "CE basis: lithiation capacity / delithiation capacity × 100. For many tester exports this is discharge capacity / charge capacity.",
                "half_cell": "CE basis: reversible capacity / inserted capacity × 100 using the selected mapped columns.",
            }[battery_cell_type]
        )
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

        max_output_tokens = st.number_input("Max interpretation output tokens", 800, 9000, int(settings.get("max_ai_output_tokens", 3500)), 100)

        st.subheader("AI prompt settings")
        prompt_templates = load_prompt_templates()
        prompt_keys = list(prompt_templates.keys())
        default_prompt_key = settings.get("default_prompt_template", "lab_qc_progress_report" if analysis_mode == "Generic dataset analysis" else "battery_cycling_report")
        default_prompt_index = prompt_keys.index(default_prompt_key) if default_prompt_key in prompt_keys else 0
        selected_prompt_key = st.selectbox(
            "Prompt template",
            options=prompt_keys,
            index=default_prompt_index if prompt_keys else None,
            format_func=lambda key: prompt_templates.get(key, {}).get("name", key),
            help="Choose a built-in prompt style. The text below is editable and will be used for OpenAI/Ollama report writing.",
        ) if prompt_keys else ""
        default_prompt_text = prompt_templates.get(selected_prompt_key, {}).get("prompt", "") if selected_prompt_key else ""
        use_custom_prompt = st.checkbox("Use editable custom prompt", value=True, help="Turn this on to send the prompt below to the AI together with calculated evidence tables.")
        custom_prompt = st.text_area(
            "Custom report prompt",
            value=default_prompt_text,
            height=220,
            help="Edit this text to control how the AI writes the interpretation. BatterySense still sends calculated summary/evidence, not the full raw dataset.",
        ) if use_custom_prompt else ""
        st.caption("Tip for local models: keep the prompt strict: do not explain JSON, do not invent values, and use only BatterySense evidence.")

        st.subheader("Protocol sections to analyze")
        include_segments = st.multiselect(
            "Include segments in metrics/plots",
            ["formation", "rate_test", "cycling"],
            default=settings.get("default_include_segments", ["formation", "rate_test", "cycling"]),
            help=(
                "Default uses all protocol sections so cycle 1 and formation behavior remain visible. "
                "Capacity-retention baselines are always calculated before this segment filter."
            ),
        )
        protocol_override_text = st.text_area(
            "Optional manual protocol cycle ranges",
            value=settings.get("default_protocol_overrides", ""),
            height=120,
            help=(
                "Use this when the automatic detector cannot know the exact tester protocol. "
                "Example: cell_01: formation=1; rate_test=2-14; cycling=15-\n"
                "cell_02: formation=1; rate_test=2-22; cycling=23-\n"
                "Use 'all:' to apply to every cell."
            ),
            placeholder=(
                "cell_01: formation=1; rate_test=2-14; cycling=15-\n"
                "cell_02: formation=1; rate_test=2-22; cycling=23-\n"
                "cell_03: formation=1; rate_test=2-14; cycling=15-"
            ),
        )

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
        "analysis_type": "multi_cell" if dataset_analysis_type == "Multi cells / reproducibility" else "single_cell",
        "battery_cell_type": battery_cell_type,
        "battery_cell_label": battery_cell_label,
        "ai_backend": ai_backend,
        "openai_model": openai_model,
        "ollama_model": ollama_model,
        "ollama_host": ollama_host,
        "max_output_tokens": int(max_output_tokens),
        "prompt_template_key": selected_prompt_key,
        "custom_prompt": custom_prompt,
        "include_segments": include_segments,
        "protocol_override_text": protocol_override_text,
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
        render_footer(settings)
        return

    progress_placeholder = st.empty()
    render_progress_radial(progress_placeholder, 3, "Starting analysis")
    progress_box = st.status("Preparing BatterySense AI analysis...", expanded=True)
    try:
        render_progress_radial(progress_placeholder, 8, "Loading uploaded files")
        progress_box.write("Loading uploaded CSV/XLSX files...")
        loaded_datasets = load_uploaded_files(uploaded_files)
        uploaded_file_names = sorted({item.source_file for item in loaded_datasets})
        combined_raw = combine_datasets(loaded_datasets)
        progress_box.write(f"Loaded {len(uploaded_file_names)} file(s); combined rows: {len(combined_raw)}.")
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
        from modules.generic_analysis import (
            build_generic_interpretation,
            build_generic_key_findings,
            compare_groups,
            make_generic_plots,
            recommend_group_column,
            recommend_numeric_columns,
            save_generic_html_report,
            save_generic_pdf_report,
        )

        st.header("Generic dataset analysis")
        st.caption("This mode is for overview/summary tables and non-cycle-level datasets. It creates compact analysis tables, group comparisons, figures, and exportable HTML/PDF reports.")
        render_progress_radial(progress_placeholder, 35, "Summarizing generic dataset")
        progress_box.write("Creating compact generic summary tables and avoiding raw JSON rendering.")
        generic = generic_dataset_summary(combined_raw)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Rows", generic.get("rows", 0))
        g2.metric("Columns", generic.get("columns", 0))
        g3.metric("Numeric", len(generic.get("numeric_columns", [])))
        g4.metric("Categorical", len(generic.get("categorical_columns", [])))

        cat_cols = generic.get("categorical_columns", [])
        num_cols = generic.get("numeric_columns", [])
        preferred_groups = [c for c in ["Group", "sample_name", "slurry_name", "cell_name", "file_name", "NW_file_name"] if c in combined_raw.columns]
        group_options = [""] + preferred_groups + [c for c in cat_cols if c not in preferred_groups]
        recommended_group = recommend_group_column(combined_raw)
        default_group_index = group_options.index(recommended_group) if recommended_group in group_options else 0
        default_value_cols = recommend_numeric_columns(combined_raw, max_cols=8)

        st.subheader("Analysis controls")
        c_group, c_values = st.columns([1, 2])
        with c_group:
            group_col = st.selectbox(
                "Group column",
                options=group_options,
                index=default_group_index,
                help="Choose a grouping column to compare materials, samples, slurry names, or file names. The app selects a likely default when available.",
            )
        with c_values:
            value_cols = st.multiselect(
                "Numeric columns to compare",
                options=num_cols,
                default=[c for c in default_value_cols if c in num_cols],
                help="These variables are used for group comparison and overview plots.",
            )

        render_progress_radial(progress_placeholder, 52, "Calculating generic statistics")
        group_table = pd.DataFrame()
        if group_col and value_cols:
            group_table = compare_groups(combined_raw, group_col, value_cols)

        key_findings = build_generic_key_findings(generic, group_table, group_col or None)
        plots = make_generic_plots(combined_raw, generic, group_col or None, value_cols)
        interpretation = build_generic_interpretation(generic, group_col or None)

        st.subheader("Key findings")
        st.dataframe(key_findings, use_container_width=True, hide_index=True)

        tabs = st.tabs(["Missing values", "Numeric statistics", "Categorical summary", "Correlations", "Battery overview columns", "Group comparison", "Figures", "Interpretation / Export"])
        with tabs[0]:
            st.dataframe(generic["missing_table"].head(120), use_container_width=True, hide_index=True)
        with tabs[1]:
            st.caption("Compact descriptive statistics. Full raw JSON is intentionally not displayed because it can freeze the browser for wide datasets.")
            st.dataframe(generic["numeric_describe_table"].head(160), use_container_width=True, hide_index=True)
        with tabs[2]:
            if not generic["categorical_summary_table"].empty:
                st.dataframe(generic["categorical_summary_table"], use_container_width=True, hide_index=True)
            else:
                st.info("No categorical columns detected.")
        with tabs[3]:
            if not generic["correlation_table"].empty:
                st.dataframe(generic["correlation_table"], use_container_width=True, hide_index=True)
            else:
                st.info("Not enough numeric columns for correlation analysis.")
        with tabs[4]:
            overview_rows = []
            for section, cols in generic.get("battery_overview_columns", {}).items():
                overview_rows.append({"section": section, "detected_columns": ", ".join(map(str, cols[:30])) if cols else ""})
            st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)
        with tabs[5]:
            if group_col and value_cols and not group_table.empty:
                st.caption(f"Grouped by `{group_col}`. Values are count, mean, standard deviation, min, median, and max for selected numeric variables.")
                st.dataframe(group_table.head(300), use_container_width=True, hide_index=True)
            else:
                st.info("Choose a group column and at least one numeric variable to create a group-comparison report.")
        with tabs[6]:
            if plots:
                for name, fig in plots.items():
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            else:
                st.info("No plots were generated for this dataset.")
        with tabs[7]:
            st.subheader("Automatic generic interpretation")
            st.markdown(interpretation)

            st.subheader("Export generic report")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                html_clicked = st.button("Generate generic HTML report", type="primary", key="generic_html_export")
            with export_col2:
                pdf_clicked = st.button("Generate generic PDF report", key="generic_pdf_export")

            if html_clicked:
                render_progress_radial(progress_placeholder, 86, "Exporting generic HTML report")
                html_path = save_generic_html_report(
                    title="BatterySense AI Generic Dataset Analysis Report",
                    uploaded_file_names=uploaded_file_names,
                    summary=generic,
                    group_table=group_table,
                    key_findings=key_findings,
                    interpretation=interpretation,
                    plots=plots,
                    output_dir=OUTPUT_REPORT_DIR,
                    banner_path=BANNER_PATH,
                )
                st.success(f"Generic HTML report saved: {html_path}")
                st.download_button("Download generic HTML report", data=html_path.read_bytes(), file_name=html_path.name, mime="text/html")

            if pdf_clicked:
                render_progress_radial(progress_placeholder, 92, "Exporting generic PDF report")
                try:
                    pdf_path = save_generic_pdf_report(
                        title="BatterySense AI Generic Dataset Analysis Report",
                        uploaded_file_names=uploaded_file_names,
                        summary=generic,
                        group_table=group_table,
                        key_findings=key_findings,
                        interpretation=interpretation,
                        plots=plots,
                        output_dir=OUTPUT_REPORT_DIR,
                    )
                    st.success(f"Generic PDF report saved: {pdf_path}")
                    st.download_button("Download generic PDF report", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to export generic PDF report: {exc}")
                    st.info("The HTML report is recommended if Plotly static image export/Kaleido is not available.")

        render_progress_radial(progress_placeholder, 100, "Generic analysis completed")
        progress_box.update(label="Generic dataset analysis completed", state="complete", expanded=False)
        render_footer(settings)
        return

    st.header("3. Column standardization")
    render_progress_radial(progress_placeholder, 28, "Detecting columns")
    progress_box.write("Detecting and standardizing battery columns...")
    detected_mapping = detect_columns(combined_raw)
    user_mapping = select_column_mapping(combined_raw, detected_mapping)
    standardized_raw, mapping = canonicalize_battery_columns(combined_raw, user_mapping)
    st.caption("Canonical columns are added internally so downstream modules can handle Neware, Arbin, BioLogic, and custom exports consistently.")

    numeric_cols = numeric_columns_from_mapping(mapping)
    render_progress_radial(progress_placeholder, 42, "Cleaning and segmenting data")
    progress_box.write("Cleaning data and applying quality filters...")
    cleaned = clean_dataframe(
        standardized_raw,
        numeric_columns=numeric_cols,
        drop_all_empty_rows=ui["cleaning_settings"]["drop_all_empty_rows"],
        drop_duplicate_rows=ui["cleaning_settings"]["drop_duplicate_rows"],
        mapping=mapping,
        cleaning_settings=ui["cleaning_settings"],
    )
    if ui.get("protocol_override_text", "").strip():
        progress_box.write("Applying manual protocol segment cycle-range overrides...")
        cleaned = apply_protocol_segment_overrides(cleaned, ui["protocol_override_text"], mapping)

    st.header("4. Data cleaning and protocol segmentation")
    qcols = [c for c in ["source_file", "source_sheet", "cell_name", "cycle_number", "protocol_segment", "analysis_include", "quality_flag", "exclude_reason"] if c in cleaned.columns]
    if qcols:
        st.dataframe(cleaned[qcols].head(500), use_container_width=True, hide_index=True)
    if "protocol_segment" in cleaned.columns:
        st.bar_chart(cleaned["protocol_segment"].value_counts())
        seg_cols = [c for c in ["cell_name", "source_file"] if c in cleaned.columns]
        segment_group_col = seg_cols[0] if seg_cols else None
        if segment_group_col and "cycle_number" in cleaned.columns:
            seg_summary = (
                cleaned.assign(cycle_number_numeric=pd.to_numeric(cleaned["cycle_number"], errors="coerce"))
                .groupby([segment_group_col, "protocol_segment"], dropna=False)
                .agg(
                    n_points=("protocol_segment", "size"),
                    cycle_min=("cycle_number_numeric", "min"),
                    cycle_max=("cycle_number_numeric", "max"),
                )
                .reset_index()
                .rename(columns={segment_group_col: "cell_or_sample"})
            )
            st.subheader("Detected protocol segment ranges")
            st.caption("Check these ranges before interpreting rate-test and cycling metrics. Use the manual protocol cycle-range box in the sidebar if the automatic detector needs correction.")
            st.dataframe(seg_summary, use_container_width=True, hide_index=True)
    excluded = cleaned[~cleaned.get("analysis_include", pd.Series(True, index=cleaned.index)).fillna(False)]
    if not excluded.empty:
        st.warning(f"{len(excluded)} row(s) were excluded from plots/metrics by the quality filter. They remain visible in the processed table and anomaly report.")

    st.header("5. Battery analysis")
    render_progress_radial(progress_placeholder, 62, "Calculating local battery metrics")
    progress_box.write("Calculating CE, irreversible capacity, accumulated CE, SOH, C-rate, and cell metrics locally...")
    analysis = run_battery_analysis(cleaned, mapping, settings_dict=ui["analysis_settings"], include_segments=ui["include_segments"], battery_cell_type=ui["battery_cell_type"], analysis_type=ui["analysis_type"])
    processed = analysis["processed_data"]
    metrics = analysis["cell_metrics"]
    anomalies = analysis["anomalies"]
    render_progress_radial(progress_placeholder, 100, "Analysis completed")
    progress_box.update(label="BatterySense AI analysis completed", state="complete", expanded=False)
    render_metrics(analysis["summary"])
    derived_cols = [c for c in [
        "cycle_number",
        "protocol_segment",
        "charge_or_lithiation_capacity",
        "discharge_or_delithiation_capacity",
        "retention_capacity",
        "retention_first_reference_capacity",
        "retention_max_stabilized_reference_capacity",
        "capacity_retention_from_first_pct",
        "capacity_retention_from_max_pct",
        "ce_numerator_capacity",
        "ce_denominator_capacity",
        "CE(%)",
        "100-CE",
        "irreversible_capacity",
        "Acc_irreversible_capacity_u_mAh_per_cm2",
        "Accumulated CE",
        "CE_20_pt_AVG",
        "SOH_pct",
        "estimated_c_rate_from_time",
        "battery_cell_type",
        "ce_calculation_basis",
    ] if c in processed.columns]
    if derived_cols:
        with st.expander("Locally calculated battery columns", expanded=False):
            st.caption("CE and related columns are calculated locally from the mapped charge/lithiation and discharge/delithiation capacities using the selected full-cell, anode half-cell, cathode half-cell, or generic half-cell convention. Uploaded CE is preserved separately only for reference.")
            st.dataframe(processed[derived_cols].head(500), use_container_width=True, hide_index=True)
    st.subheader("Cell/sample metrics")
    st.caption(
        "Main deciding parameters are shown here, including first-cycle capacity/CE, last CE, "
        "retention from first valid capacity, retention from max-stabilized capacity, degradation rate, "
        "average CE, accumulated CE, and accumulated irreversible capacity."
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    deviation_report = analysis.get("deviation_report", {})
    if ui.get("analysis_type") == "multi_cell" and deviation_report.get("enabled"):
        st.subheader("Multi-cell deviation / reproducibility report")
        st.caption(
            "This report compares cell-to-cell reproducibility using standard deviation and relative standard deviation (RSD) "
            "for first-cycle CE, first-cycle lithiation/delithiation capacities, upper/max capacities, final retention, and degradation."
        )
        std_table = deviation_report.get("std_table")
        per_cell_dev = deviation_report.get("per_cell_deviation")
        if isinstance(std_table, pd.DataFrame) and not std_table.empty:
            st.dataframe(std_table, use_container_width=True, hide_index=True)
        if isinstance(per_cell_dev, pd.DataFrame) and not per_cell_dev.empty:
            with st.expander("Per-cell deviation from multi-cell mean", expanded=False):
                st.dataframe(per_cell_dev, use_container_width=True, hide_index=True)
    elif ui.get("analysis_type") == "multi_cell":
        st.info("Multi-cell deviation report requires at least two detected cells/samples.")

    st.subheader("Detected anomalies and excluded points")
    st.dataframe(anomalies, use_container_width=True, hide_index=True)

    st.header("6. Comparison tools")
    group_col = "cell_or_sample"
    if not metrics.empty and group_col in metrics.columns:
        cells = metrics[group_col].astype(str).tolist()
        selected_cells = st.multiselect("One-to-one cell comparison", cells, default=cells[: min(2, len(cells))])
        cmp_metric = st.selectbox("Comparison y-axis", [c for c in ["capacity_retention_pct", "discharge_or_delithiation_capacity", "charge_or_lithiation_capacity", "calculated_coulombic_efficiency_pct", "CE_20_pt_AVG", "Accumulated CE", "SOH_pct", "estimated_c_rate_from_time"] if c in processed.columns])
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
    interpretation_progress = st.empty()
    if st.button("Generate interpretation", type="primary"):
        render_progress_radial(interpretation_progress, 15, "Starting interpretation")
        with st.spinner("Generating report interpretation..."):
            render_progress_radial(interpretation_progress, 55, "Writing scientific interpretation")
            st.session_state.interpretation_text = generate_ai_report(
                ai_context,
                backend=ui["ai_backend"],
                openai_model=ui["openai_model"],
                ollama_model=ui["ollama_model"],
                ollama_host=ui["ollama_host"],
                max_output_tokens=ui["max_output_tokens"],
                custom_prompt=ui.get("custom_prompt", ""),
            )
            render_progress_radial(interpretation_progress, 100, "Interpretation completed")
    if st.session_state.interpretation_text:
        st.markdown(st.session_state.interpretation_text)

    st.header("10. Export reports")
    templates = get_report_templates()
    report_title = st.text_input("Report title", value=settings.get("report_title", "BatterySense AI Battery Cycling Analysis Report"))
    report_template = st.selectbox("Report template", list(templates.keys()), format_func=lambda x: f"{x} — {templates[x].description}")
    export_pdf = st.checkbox("Also export PDF", value=True)
    save_to_db = st.checkbox("Save historical run in local SQLite database", value=True)

    export_progress = st.empty()
    if st.button("Export report"):
        render_progress_radial(export_progress, 10, "Preparing report export")
        interpretation = st.session_state.interpretation_text or generate_ai_report(
            ai_context,
            backend=ui["ai_backend"],
            openai_model=ui["openai_model"],
            ollama_model=ui["ollama_model"],
            ollama_host=ui["ollama_host"],
            max_output_tokens=ui["max_output_tokens"],
            custom_prompt=ui.get("custom_prompt", ""),
        )
        try:
            render_progress_radial(export_progress, 35, "Saving interactive HTML report")
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
                render_progress_radial(export_progress, 70, "Saving PDF report")
                pdf_path = save_pdf_report(report_title, uploaded_file_names, combined_profile, analysis, plots, interpretation, output_dir=OUTPUT_REPORT_DIR)
            if save_to_db:
                save_analysis_run(DB_PATH, report_title, uploaded_file_names, {**ui, "report_template": report_template}, analysis, str(html_path), str(pdf_path) if pdf_path else None)
            render_progress_radial(export_progress, 100, "Reports exported")
            st.success(f"HTML report saved: {html_path}")
            st.download_button("Download HTML report", data=html_path.read_bytes(), file_name=html_path.name, mime="text/html")
            if pdf_path:
                st.success(f"PDF report saved: {pdf_path}")
                st.download_button("Download PDF report", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to export report: {exc}")

    with st.expander("Historical analysis runs", expanded=False):
        st.dataframe(pd.DataFrame(list_analysis_runs(DB_PATH, limit=20)), use_container_width=True)

    render_footer(settings)


if __name__ == "__main__":
    main()

