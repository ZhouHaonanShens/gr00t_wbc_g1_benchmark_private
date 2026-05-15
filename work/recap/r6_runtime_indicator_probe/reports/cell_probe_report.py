from __future__ import annotations

from work.recap.r6_runtime_indicator_probe.contract import CellProbeReport


def render_cell_report(report: CellProbeReport) -> str:
    graph = report.static
    rows = ["| src_symbol | dst_symbol | cite | via |", "|---|---|---|---|"]
    rows.extend(
        f"| `{e.src_symbol}` | `{e.dst_symbol}` | `{e.src_file}:{e.src_line}` | {e.via} |"
        for e in graph.edges
    )
    runtime = report.runtime.runtime_verdict if report.runtime is not None else "NOT_RUN"
    return "\n".join(
        [
            f"# R6 Cell Probe Report — {report.cell_id}",
            "",
            f"- static_verdict: `{graph.static_verdict}`",
            f"- runtime_verdict: `{runtime}`",
            f"- final: `{report.final}`",
            f"- reaches_sink: `{str(graph.reaches_sink).lower()}`",
            f"- notes: {graph.notes}",
            "",
            "## Wiring edges",
            "",
            *rows,
            "",
        ]
    )
