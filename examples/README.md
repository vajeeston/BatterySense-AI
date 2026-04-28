# Example Reports

This folder contains example outputs generated from the uploaded battery cycling Excel files.

Files:
- `example_battery_cycling_report.html`: interactive HTML report with Plotly figures.
- `example_battery_cycling_report.pdf`: static PDF report.
- `example_cell_metrics.csv`: metrics calculated by the latest analysis module.
- `example_last_valid_metrics.csv`: comparison after excluding terminal zero-capacity points for interpretation.
- `example_anomalies.csv`: anomaly flags.
- `example_ai_context_summary.json`: summarized context used for report writing.
- `data/`: copied example input files.
- `figures/`: static PNG figures used in the PDF.

Note: The two uploaded datasets contain terminal zero-capacity/zero-CE points. These were retained and flagged as suspicious. The PDF report also includes a last-valid non-zero comparison.
