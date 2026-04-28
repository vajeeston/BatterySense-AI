<p align="center">
  <img src="assets/batterysense_banner.png" alt="BatterySense AI banner" width="100%">
</p>

# BatterySense AI

**Author: AU P. Vajeeston**
-   Copyright (c) 2026 AU P. Vajeeston

BatterySense AI is a professional Python/Streamlit application for AI-assisted battery cycling data analysis, quality control, visualization, and report generation. It is designed for common battery research workflows where data may come from different cell testers such as Neware, Arbin, BioLogic, or custom CSV/XLSX exports.

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


