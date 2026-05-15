from __future__ import annotations

from typing import Any

from work.recap.r3_contract_parity.contract import _MISSING, ParityCellReport


def value_text(value: Any) -> str:
    return "__MISSING__" if value is _MISSING else repr(value)


def render_cell_report(report: ParityCellReport) -> str:
    rows = ["| axis | verdict | pattern | train | eval | note |", "|---|---|---|---|---|---|"]
    for result in report.axes:
        rows.append(
            f"| {result.axis.axis_id} | {result.verdict} | {result.pattern_id} | "
            f"`{value_text(result.train_value)}` | `{value_text(result.eval_value)}` | {result.note} |"
        )
    hits = ", ".join(report.pattern_hits) if report.pattern_hits else "none"
    return f"""# R3 Cell Parity Report — {report.cell_id}

Cite: `.reference/records/recap_a1_exclusion_record.md` keeps A.1 excluded; this report covers evidence-grade cells only.

- checkpoint: `{report.ckpt_abs_path}`
- overall_verdict: `{report.verdict}`
- openpi_homology_pattern_hits: `{hits}`

## Axis table

{chr(10).join(rows)}

## OpenPI homology

FAIL_HOT rows identify a train/eval contract-parity pattern analogous to OpenPI transform or normalization drift. PASS rows mean the static manifest agrees with local checkpoint bytes/fields.

## Runtime literal

```python
runtime_invocations = []
```
"""
