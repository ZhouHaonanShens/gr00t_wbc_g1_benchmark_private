from __future__ import annotations

from collections.abc import Sequence

from work.recap.r6_runtime_indicator_probe.contract import CellProbeReport

PHASE_B_SHA = "d699430ed7b4b1dad8b34c553e90d9cecb9ac6e7654068ffbdcee06ae5e1779d"
R3_SHA = "5ef81116b3f45742e83ce3be4d87407789871e9e392e03f98b90d595aa636599"
R5_SHA = "1960de2cc4497587ed5df3195d0c3d63984fe10449a58feb9f7dbd4f413b3fe4"


def render_summary_report(reports: Sequence[CellProbeReport]) -> str:
    rows = ["| cell_id | static_verdict | runtime_verdict | final | reaches_sink |", "|---|---|---|---|---|"]
    for report in reports:
        runtime = report.runtime.runtime_verdict if report.runtime is not None else "NOT_RUN"
        rows.append(
            f"| {report.cell_id} | `{report.static.static_verdict}` | `{runtime}` | `{report.final}` | `{str(report.static.reaches_sink).lower()}` |"
        )
    ambiguous = [r.cell_id for r in reports if r.static.static_verdict == "AMBIGUOUS"]
    branch = "R6.1_APPROVAL_REQUIRED" if ambiguous else "R6.1_SKIPPED_NO_AMBIGUOUS_STATIC_CELLS"
    return f"""# FIX_R2_A1_LOAD_05_R6_WIRING_REPORT

## Scope

R6.0 static AST/import graph trace for evidence-grade cells A.2..A.5. A.1 remains excluded by the R2 SSOT and is not traced. This run did not execute GPU, training, subprocess probes, checkpoint mutation, videos, tensor dumps, or dataset writes.

## Verdict matrix

{chr(10).join(rows)}

## Branch decision

`{branch}`

## Grounding citations

- Phase B report: `agent/artifacts/recap_substrate_recovery/fix_r2_a1_load/20260512T084019Z_phase_b/FIX_R2_A1_LOAD_02_PHASE_B_REPORT.md`, sha256 `{PHASE_B_SHA}`, anchors lines 10-12.
- R3 report: `agent/artifacts/recap_substrate_recovery/r3_contract_parity/20260512T140003Z_run/FIX_R2_A1_LOAD_03_R3_PARITY_AUDIT_REPORT.md`, sha256 `{R3_SHA}`, anchors lines 9-17.
- R5 report: `agent/artifacts/recap_substrate_recovery/r5_fidelity_audit/20260513T065300Z_run/FIX_R2_A1_LOAD_04_R5_FIDELITY_AUDIT_REPORT.md`, sha256 `{R5_SHA}`, anchors lines 13-23 and 25-31.

## Notes

All public final verdict values are emitted through `compose_final`; because no R6.1 runtime trace was approved or run in R6.0, static-WIRED cells retain final verdict `WIRED_STATIC_UNCONFIRMED_RUNTIME` by the exact synthesis table until runtime evidence is collected.
"""
