# Color Export Fix

This version fixes an issue where exported HTML/PDF plots could appear black-and-white.

## Cause
Some Plotly export environments can inherit a near-black/default colorway when figures are converted to standalone HTML or static PNG images for PDF export. In the affected reports, the exported traces were assigned colors such as `#000001`, `#000002`, and `#000003`, which made all cell traces look black.

## Fix
`modules/plotting.py` now defines and forces a BatterySense color palette for every figure and every trace before export:

- line plots use explicit `color_discrete_sequence`
- bar charts use `color="cell_or_sample"`
- static PNG snapshots for PDF preserve trace colors
- Plotly template is fixed to `plotly_white`
- near-black inherited trace colors are replaced with visible colors

## Notes
After updating, regenerate the report. Existing old HTML/PDF files will remain black-and-white because their figures were already exported.
