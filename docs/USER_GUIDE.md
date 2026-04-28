<p align="center">
  <img src="assets/batterysense_banner.png" alt="BatterySense AI banner" width="100%">
</p>


# BatterySense AI User Guide

**Author: AU P. Vajeeston**

## 1. What BatterySense AI does

BatterySense AI helps battery researchers convert raw cycling data into clean metrics, interactive plots, and scientific reports. It is designed for CSV/XLSX data from different cell testers and supports multi-file and multi-sheet workflows.

## 2. Starting the app

Open a terminal in the project folder and run:

```bash
streamlit run app.py
```

The app opens in your browser.

## 3. Uploading data

Use the upload box to select one or more files. Supported formats are:

- `.csv`
- `.xlsx`
- `.xlsm`

For Excel workbooks, each readable sheet is loaded as a separate table and combined with source-file and source-sheet metadata.

## 4. Column mapping

BatterySense AI automatically searches for common battery columns such as cycle number, charge capacity, discharge capacity, Coulombic efficiency, voltage, current, CC capacity, and CV capacity.

Always check the suggested mapping. If a field is wrong, select the correct column manually.

## 5. Data cleaning

The app flags suspicious data using conservative rules:

- empty rows
- missing or non-finite critical values
- implausible Coulombic efficiency values
- artificial extremely high numeric values
- incomplete terminal cycles
- abnormal capacity drops

Flagged rows are not permanently deleted. They are excluded from analysis and plotting by default, but remain visible in quality tables.

## 6. Protocol segmentation

The app tries to separate:

- formation
- rate test
- cycling

It uses available step/rate labels when present. If such labels are missing, it uses early-cycle behavior, Coulombic efficiency, local capacity variation, and current changes as heuristic indicators.

You can choose which segments to include in the sidebar. For long-term cyclability, use only `cycling`. For full-protocol visualization, include formation and rate-test sections.

## 7. Battery analysis

The app calculates:

- first cycle capacities
- first cycle CE
- capacity retention
- degradation slope
- average CE
- estimated cycle life to the chosen retention threshold
- best/worst cell
- anomaly flags

## 8. Comparison tools

Use one-to-one comparison to compare selected cells directly.

Use group comparison for triplicate or material-group studies. Enter one mapping per line:

```text
Cell_01 = Material_A
Cell_02 = Material_A
Cell_03 = Material_A
Cell_04 = Material_B
Cell_05 = Material_B
Cell_06 = Material_B
```

## 9. AI report generation

The app supports:

- local rule-based summary
- local Ollama model
- OpenAI API

The AI receives summarized metrics only, not raw cycle-level data.

## 10. Exporting reports

The app exports:

- interactive HTML report
- PDF summary report
- optional local SQLite record of the analysis run

Reports are saved in:

```text
outputs/reports/
```

Historical runs are saved in:

```text
outputs/database/analysis_runs.sqlite
```

## 11. Tips for reliable results

- Check column mapping before running analysis.
- Review excluded rows and anomaly flags.
- Confirm whether final zero-capacity rows are real failures or incomplete tester exports.
- For publications, document your filtering criteria and protocol-segment choices.
- Add metadata columns such as material, electrolyte, loading, formation protocol, temperature, and C-rate when possible.
