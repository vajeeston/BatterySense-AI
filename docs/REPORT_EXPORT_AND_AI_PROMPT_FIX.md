# BatterySense AI report export and AI prompt fix

This update improves the report-writing and export pipeline.

## Changes

- Removed raw evidence code blocks from the fallback scientific interpretation.
- Prevented Markdown evidence tables from appearing as broken paragraph text in HTML/PDF reports.
- Added a safer HTML Markdown renderer with support for headings, bullet lists, numbered lists, code fences, and Markdown tables.
- Added horizontally scrollable HTML tables for wide battery metrics and deviation reports.
- Fixed ReportLab PDF export failures caused by inline-code Courier font markup.
- Improved the local Ollama prompt to use the evidence sheet only and avoid JSON/code/table echoing.
- Increased the default AI output-token setting to 5000.
- Added a larger Ollama context window option (`num_ctx=8192`) for better local model performance.

## Notes

If a local LLM output still contradicts the calculated evidence, BatterySense AI will generate a deterministic evidence-based interpretation instead of publishing unsupported claims.

## Protocol segmentation update

BatterySense AI now uses a recovery-jump detector for formation/rate-test/cycling separation. This is useful for common cell-test protocols where cycle 1 is formation, cycles 2 onward are a rate test, and normal cycling starts after the capacity recovers from the final high-rate cycle.

A manual override box is also available in the sidebar. Example:

```text
cell_01: formation=1; rate_test=2-14; cycling=15-
cell_02: formation=1; rate_test=2-22; cycling=23-
cell_03: formation=1; rate_test=2-14; cycling=15-
```

Use this when the exact tester protocol is known. The detected segment range table in the app should be checked before interpreting the rate-test and cycling sections.

