# Reports

Automated georeferencing quality reports.

Each image processed produces:
- `<safe_name>_report.html` — fully self-contained HTML quality report (all images embedded)
- `<safe_name>_qa.json` — machine-readable QA summary

## Quality Labels

| Label | RMS Residual | Meaning |
|-------|-------------|---------|
| GOOD | < 2.0 m | Georeferencing is reliable |
| ACCEPTABLE | 2.0–5.0 m | Acceptable quality, check report |
| POOR | > 5.0 m | Problems detected, investigate before use |

Reports are generated on the result branch alongside the review artifacts.
