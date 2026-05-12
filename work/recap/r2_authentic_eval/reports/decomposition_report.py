"""Markdown renderer for the R2.2 decomposition (kept vs swap; pure function)."""
from __future__ import annotations

from typing import Any

from work.recap.r2_authentic_eval.ckpt_config_swap import ConfigSwapResult
from work.recap.r2_authentic_eval.eval_runner import R2CellResult


def build_decomposition_table(
    *,
    kept_cell: R2CellResult,
    swap_cell: R2CellResult,
    swap_provenance: ConfigSwapResult,
) -> dict[str, Any]:
    """Build decomposition_table.json payload (pure)."""
    return {
        "r2_decomposition_table_schema_version": "1.0.0",
        "kept_cell": {
            "abs_path": str(kept_cell.request.checkpoint.abs_path),
            "rate": kept_cell.rate,
            "wilson_ci_95": list(kept_cell.wilson_ci_95),
            "success_count": kept_cell.success_count,
            "completed_episode_total": kept_cell.completed_episode_total,
        },
        "swap_cell": {
            "abs_path": str(swap_cell.request.checkpoint.abs_path),
            "rate": swap_cell.rate,
            "wilson_ci_95": list(swap_cell.wilson_ci_95),
            "success_count": swap_cell.success_count,
            "completed_episode_total": swap_cell.completed_episode_total,
        },
        "swap_provenance": {
            "swap_dir": str(swap_provenance.swap_dir),
            "source_ckpt_root": str(swap_provenance.source_ckpt_root),
            "raw_hf_root": str(swap_provenance.raw_hf_root),
            "materialised_at_utc": swap_provenance.materialised_at_utc,
            "link_strategy": swap_provenance.link_strategy,
        },
    }


def render(
    *,
    kept_cell: R2CellResult,
    swap_cell: R2CellResult,
    swap_provenance: ConfigSwapResult,
) -> str:
    """Render the decomposition report as markdown (pure)."""
    klo, khi = kept_cell.wilson_ci_95
    slo, shi = swap_cell.wilson_ci_95
    delta = swap_cell.rate - kept_cell.rate
    out = [
        "# R2.2 Decomposition — kept vs config-swap",
        "",
        "## Side-by-side",
        "| leg | abs_path | succ/total | rate | wilson_ci_95 |",
        "|---|---|---|---|---|",
        (
            f"| kept | `{kept_cell.request.checkpoint.abs_path}` | "
            f"{kept_cell.success_count}/{kept_cell.completed_episode_total}"
            f" | {kept_cell.rate:.3f} | ({klo:.3f}, {khi:.3f}) |"
        ),
        (
            f"| swap | `{swap_cell.request.checkpoint.abs_path}` | "
            f"{swap_cell.success_count}/{swap_cell.completed_episode_total}"
            f" | {swap_cell.rate:.3f} | ({slo:.3f}, {shi:.3f}) |"
        ),
        "",
        f"- swap_minus_kept_rate: {delta:+.3f}",
        "",
        "## Swap provenance",
        f"- swap_dir: `{swap_provenance.swap_dir}`",
        f"- source_ckpt_root: `{swap_provenance.source_ckpt_root}`",
        f"- raw_hf_root: `{swap_provenance.raw_hf_root}`",
        f"- materialised_at_utc: {swap_provenance.materialised_at_utc}",
        f"- link_strategy: {swap_provenance.link_strategy}",
        "",
    ]
    return "\n".join(out) + "\n"


__all__ = ["render", "build_decomposition_table"]
