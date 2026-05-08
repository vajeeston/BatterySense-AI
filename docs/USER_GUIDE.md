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

## Full-cell and half-cell mode

When uploading battery data, select the correct cell type in the sidebar.

- **Full cell** is the default. CE is calculated as `discharge capacity / charge capacity × 100`.
- **Anode half-cell** uses the common reversible anode convention: `delithiation capacity / lithiation capacity × 100`. For many tester exports this means `charge capacity / discharge capacity × 100`.
- **Cathode half-cell** uses the common cathode convention: `lithiation capacity / delithiation capacity × 100`. For many tester exports this means `discharge capacity / charge capacity × 100`.
- **Generic half-cell / custom** uses `reversible capacity / inserted capacity × 100` based on the selected mapped columns.

The app does not require CE to be present in the uploaded file. Instead, it calculates CE and related columns locally from the mapped capacity columns and the selected cell type. This gives more consistent analysis across Neware, Arbin, BioLogic, and other tester exports.

## Locally calculated columns

BatterySense AI calculates these values locally:

- CE(%)
- 100-CE
- irreversible capacity
- accumulated irreversible capacity
- accumulated CE
- 20-point moving average CE
- SOH / capacity retention
- estimated C-rate from time, when charge/discharge time columns are available

If the uploaded file already contains CE, it is kept as a reference column, but the local calculated CE is used for analysis.

## Progress window

During analysis, a small status window shows the current processing step, including file loading, column detection, cleaning, local metric calculation, and completion.


## Retention baseline options

BatterySense AI now reports two retention calculations:

1. **From first valid capacity**: useful for comparing the whole protocol from cycle 1.
2. **From max-stabilized cycling capacity**: useful when formation or early cycles stabilize later and the maximum cycling capacity is a more realistic reference.

The main SOH and final-retention ranking use the max-stabilized baseline by default. The report also includes a baseline-comparison plot so users can inspect both definitions before drawing conclusions.

## Main deciding metrics shown in reports

Reports include first charge/lithiation capacity, first discharge/delithiation capacity, first and last calculated CE, last capacity, retention from first capacity, retention from max-stabilized capacity, degradation slope, average CE, rolling CE, accumulated CE, and accumulated irreversible capacity.


## Single-cell vs multi-cell analysis

Use **Single cell** when the uploaded data represent one cell only. Use **Multi cells / reproducibility** when comparing two or more cells, triplicates, or repeated cells from the same material. In multi-cell mode, BatterySense AI generates an additional deviation report showing how much each cell differs from the group average.

The deviation report includes STD and RSD for first-cycle CE, first-cycle lithiation/charge capacity, first-cycle delithiation/discharge capacity, upper/max capacities, final retention, and degradation rate. Lower STD/RSD usually means better cell-to-cell reproducibility.

Cell names in plots and reports are based on the file name without extension. For example, `cell_01.csv` is shown as `cell_01`.

## AI report quality controls

BatterySense AI now uses an evidence-first report prompt. Python first calculates the cell metrics, protocol/segment summary, deviation statistics, anomaly counts, and key decision values. The AI model receives this compact evidence sheet instead of raw cycle-level data.

The report generator is instructed to use the exact detected cell names and calculated values. If a local or cloud AI model produces a generic interpretation, invents unsupported cell counts, or ignores the actual cell names, BatterySense AI retries with a stricter prompt and then falls back to the deterministic rule-based evidence report if needed.

For best local LLM results, use a stronger Ollama model and allow at least 3500 output tokens.

## Protocol segmentation update

BatterySense AI now uses a recovery-jump detector for formation/rate-test/cycling separation. This is useful for common cell-test protocols where cycle 1 is formation, cycles 2 onward are a rate test, and normal cycling starts after the capacity recovers from the final high-rate cycle.

A manual override box is also available in the sidebar. Example:

```text
cell_01: formation=1; rate_test=2-14; cycling=15-
cell_02: formation=1; rate_test=2-22; cycling=23-
cell_03: formation=1; rate_test=2-14; cycling=15-
```

Use this when the exact tester protocol is known. The detected segment range table in the app should be checked before interpreting the rate-test and cycling sections.


## Generic dataset analysis performance note

For wide overview-summary files, BatterySense AI now renders compact summary tables instead of a raw nested JSON object. This prevents Streamlit/browser slowdown and keeps PDF exports readable. Use Generic dataset analysis for screening-level overview tables; use Battery cycling analysis only for raw cycle-level files with cycle, charge/lithiation, and discharge/delithiation columns.
