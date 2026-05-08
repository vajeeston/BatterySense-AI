<p align="center">
  <img src="assets/batterysense_banner.png" alt="BatterySense AI banner" width="100%">
</p>

# BatterySense AI

**Author: AU P. Vajeeston**
-   Copyright (c) 2026 AU P. Vajeeston

BatterySense AI is a professional Python/Streamlit application for AI-assisted battery cycling data analysis, quality control, visualization, and report generation. 
It is designed for common battery research workflows where data may come from different cell testers such as Neware, Arbin, BioLogic, or custom CSV/XLSX exports.

The app calculates metrics locally in Python first. The AI layer receives only summarized metrics and writes a scientific interpretation. 
Users can choose between OpenAI API, local Ollama models, or a fully local rule-based report.

## Main features

- Upload CSV, XLSX, or XLSM files
- Upload multiple files and multi-sheet Excel workbooks
- Automatic data preview and dataset profiling
- Column detection and canonicalization for common tester exports
- Manual column correction for difficult datasets
- Formation, rate-test, and cycling segment detection
- Conservative data cleaning:
  - remove all-empty rows
  - coerce numeric columns
  - flag NaN and non-finite critical values
  - flag artificial extreme numeric values
  - detect incomplete final cycles
  - preserve excluded rows with quality flags
- Battery metrics:
  - first cycle charge/discharge capacity
  - first cycle Coulombic efficiency
  - capacity retention
  - degradation slope
  - average Coulombic efficiency
  - cycle-life estimate to configurable retention threshold
  - best/worst cell ranking
  - abnormal capacity-drop detection
- Interactive Plotly plots with BatterySense logo watermark
- Plotly modebar logo removed in the Streamlit app and exported HTML
- One-to-one cell comparison
- Group comparison, e.g. Material A triplet vs Material B triplet
- Optional local ML:
  - Isolation Forest anomaly detection
  - K-means clustering
  - lifetime-prediction utility for historical metrics
- Generic non-battery dataset summary mode
- HTML and PDF report export
- Local SQLite database for historical analysis runs
- Modular architecture for future desktop/PyQt5 or PyInstaller packaging

## Project structure

```text
batterysense_ai/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   ├── batterysense_banner.png
│   └── log_small.png
├── config/
│   └── settings.json
├── docs/
│   └── USER_GUIDE.md
├── modules/
│   ├── ai_report.py
│   ├── battery_analysis.py
│   ├── column_detection.py
│   ├── data_cleaning.py
│   ├── data_loader.py
│   ├── export_pdf.py
│   ├── generic_analysis.py
│   ├── html_export.py
│   ├── local_ml_analysis.py
│   ├── plotting.py
│   ├── project_database.py
│   ├── report_templates.py
│   └── statistical_analysis.py
├── outputs/
│   ├── database/
│   ├── figures/
│   └── reports/
├── examples/
└── sample_data/
```

## Installation

### You can install BatterySense AI in a Conda environment like this.
```bash
conda create -n batterysense-ai python=3.11
conda activate batterysense-ai
```
Go to the project folder:

```bash
cd path/to/batterysense_ai
```
Install the required Python packages:

```bash
pip install -r requirements.txt
```
Run the app:

```bash
streamlit run app.py
```
Then open the local Streamlit link shown in the terminal, usually:
```bash
http://localhost:8501
```





For  Windows PowerShell :

```bash
cd batterysense_ai
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -r requirements.txt
streamlit run app.py
```

## AI backend options

### Rule-based local summary

This works without API keys or local models.

### OpenAI API

```bash
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

### Local AI with Ollama

Install Ollama, then run:

```bash
ollama pull llama3.2
streamlit run app.py
```

Choose **Local LLM with Ollama** in the sidebar.

## Recommended workflow

1. Upload one or more CSV/XLSX files.
2. Inspect dataset profile and preview.
3. Confirm or correct detected battery columns.
4. Review data-cleaning quality flags and protocol segmentation.
5. Choose which protocol segments to include in metrics and plots.
6. Review metrics, anomalies, comparison tools, and plots.
7. Generate AI or local interpretation.
8. Export HTML/PDF reports and optionally save the run to the local database.

## Notes on data cleaning

BatterySense AI does not silently delete suspicious data. It adds:

- `analysis_include`
- `quality_flag`
- `exclude_reason`
- `protocol_segment`

Rows marked as `analysis_include=False` are excluded from metrics and plots by default, but they remain in the processed data and anomaly tables.

## License

BatterySense AI

Copyright (c) 2026 AU P. Vajeeston

This software is licensed under the PolyForm Noncommercial License 1.0.0.

For commercial licensing, contact:

AU P. Vajeeston, vajeeston@gmail.com



## Latest analysis update: locally calculated battery metrics

BatterySense AI now treats the uploaded battery tester file as the raw experimental source and calculates the main derived battery metrics locally in Python. The app primarily requires the following mapped columns:

- cycle number
- charge capacity / lithiation capacity
- discharge capacity / delithiation capacity
- optional charge time
- optional discharge time

The following columns are calculated locally and used for analysis, plotting, and reports:

- `CE(%)`
- `100-CE`
- `irreversible_capacity`
- `Acc_irreversible_capacity_u_mAh_per_cm2`
- `Accumulated CE`
- `CE_20_pt_AVG`
- `SOH_pct`
- `estimated_c_rate_from_time`, when charge/discharge time data are available

Uploaded Coulombic-efficiency columns are preserved only as reference data. By default, the app uses the locally calculated CE for metrics and interpretation.

The sidebar includes a **battery data type** selector. **Full cell** is the default.

Supported CE conventions:

- **Full cell:** CE = discharge capacity / charge capacity × 100
- **Anode half-cell:** CE = delithiation capacity / lithiation capacity × 100. In many tester exports this corresponds to charge capacity / discharge capacity.
- **Cathode half-cell:** CE = lithiation capacity / delithiation capacity × 100. In many tester exports this corresponds to discharge capacity / charge capacity.
- **Generic half-cell / custom:** CE = reversible capacity / inserted capacity × 100 using the selected mapped columns.

Always verify the mapped charge/lithiation and discharge/delithiation columns for your tester export before final interpretation.

A small progress/status panel is displayed during upload, cleaning, local metric calculation, and report preparation.

---

© 2026 AU P. Vajeeston. BatterySense AI is free for non-commercial research, education, and personal use. Commercial use requires prior written permission.


## Latest update: retention baselines, visible metrics, and export rendering

This version improves handling of formation/rate/cycling datasets:

- Cycle 1 and formation cycles are no longer removed simply because formation CE differs from later cycling CE.
- Capacity retention is calculated in two ways:
  - `capacity_retention_from_first_pct`: normalized to the first valid reversible/output capacity.
  - `capacity_retention_from_max_pct`: normalized to the maximum stabilized capacity in the cycling segment when available.
- The primary `capacity_retention_pct` and `SOH_pct` use the max-stabilized baseline by default.
- A new Plotly figure compares both retention baselines in the same graph.
- The key metrics table now exposes first capacity, last capacity, first/last CE, both retention baselines, average CE, accumulated CE, and accumulated irreversible capacity.
- HTML and PDF exports now render basic markdown bold text instead of showing raw `**bold**` markers.
- A small radial progress indicator is shown during loading, cleaning, local metric calculation, and export.


## Latest update: single-cell and multi-cell deviation analysis

BatterySense AI now lets users choose whether an uploaded dataset should be treated as a **single-cell analysis** or a **multi-cell / reproducibility analysis**. In multi-cell mode, cell labels are taken from the uploaded file name without the extension, for example `cell_01` instead of `cell_01.csv`.

The multi-cell deviation report calculates standard deviation and relative standard deviation across cells for important deciding parameters, including first-cycle Coulombic efficiency, first-cycle lithiation/charge capacity, first-cycle delithiation/discharge capacity, upper/max lithiation and delithiation capacity, final retention, and degradation rate. This helps users judge how strongly cells differ from each other in repeated-cell or triplicate experiments.

The app also includes separate local radial progress indicators near **Generate interpretation** and **Export report**, a BatterySense logo at the top of the settings panel, and Plotly figures with visible axis lines for clearer scientific presentation.

### AI report prompt improvement

This version uses an evidence-first AI report prompt. The app sends Python-calculated tables for per-cell metrics, formation/rate/cycling segment summaries, deviation/reproducibility statistics, anomaly counts, retention baselines, CE, accumulated CE, irreversible capacity, and SOH. The model is instructed to use exact cell names and exact values only. If an AI response is generic or contradicts the evidence, the app retries and can fall back to the rule-based scientific report.

## Protocol segmentation update

BatterySense AI now uses a recovery-jump detector for formation/rate-test/cycling separation. This is useful for common cell-test protocols where cycle 1 is formation, cycles 2 onward are a rate test, and normal cycling starts after the capacity recovers from the final high-rate cycle.

A manual override box is also available in the sidebar. Example:

```text
cell_01: formation=1; rate_test=2-14; cycling=15-
cell_02: formation=1; rate_test=2-22; cycling=23-
cell_03: formation=1; rate_test=2-14; cycling=15-
```

Use this when the exact tester protocol is known. The detected segment range table in the app should be checked before interpreting the rate-test and cycling sections.


## Editable AI prompts

BatterySense AI includes editable prompt templates for report writing. Templates are stored in `config/prompt_templates.json` and can be selected from the sidebar under **AI prompt settings**. Users can modify the free-text prompt before generating an interpretation with OpenAI or a local Ollama model.

The AI receives calculated evidence and summary tables only, not the full raw dataset.
