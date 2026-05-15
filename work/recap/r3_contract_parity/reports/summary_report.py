from __future__ import annotations

from work.recap.r3_contract_parity.contract import FAIL_HOT, PASS, ParityCellReport


def summary_data(reports: tuple[ParityCellReport, ...]) -> dict[str, object]:
    verdicts = {report.cell_id: report.verdict for report in reports}
    if any(v == FAIL_HOT for v in verdicts.values()):
        exit_branch = "EXIT-HOT"
    elif all(v == PASS for v in verdicts.values()):
        exit_branch = "EXIT-COLD"
    else:
        exit_branch = "EXIT-WARN"
    return {
        "cells": verdicts,
        "pattern_hits": {report.cell_id: list(report.pattern_hits) for report in reports},
        "runtime_invocations": [],
        "exit_branch": exit_branch,
    }


def render_summary_report(reports: tuple[ParityCellReport, ...]) -> str:
    data = summary_data(reports)
    rows = ["| cell | verdict | pattern_hits |", "|---|---|---|"]
    for report in reports:
        rows.append(f"| {report.cell_id} | {report.verdict} | {', '.join(report.pattern_hits) or 'none'} |")
    return f"""# R3 Contract Parity Summary

## Aggregate table

{chr(10).join(rows)}

## §S4 Aggregate runtime literal

```python
runtime_invocations = []
```

## Exit branch

`{data['exit_branch']}`
"""
