"""Markdown renderer for a single R2.1 / R2.2 cell result (pure function).

The CLI orchestrator is responsible for writing the returned string to disk;
this module performs no I/O.
"""
from __future__ import annotations

from work.recap.r2_authentic_eval.eval_runner import R2CellResult


def render(cell: R2CellResult) -> str:
    """Render an R2CellResult as a self-contained markdown document."""
    req = cell.request
    ck = req.checkpoint
    lo, hi = cell.wilson_ci_95
    nlo, nhi = cell.newcombe_delta_ci_95
    py_first_line = (cell.python_version or "").splitlines()[0] if cell.python_version else ""
    fallback_line = ""
    if cell.git_commit_sha_fallback_reason:
        fallback_line = f"- git_commit_sha_fallback_reason: {cell.git_commit_sha_fallback_reason}\n"
    return (
        f"# R2 Cell — {ck.label} {ck.abs_path.name}\n"
        f"\n"
        f"## Identity\n"
        f"- abs_path: `{ck.abs_path}`\n"
        f"- training_algo: {ck.training_algo}\n"
        f"- n_train_steps: {ck.n_train_steps}\n"
        f"- formalize_language: {ck.formalize_language}\n"
        f"- statistics_q99_matches_base: {ck.statistics_q99_matches_base}\n"
        f"- search_root: `{req.search_root}`\n"
        f"- strict_config: {req.strict_config}\n"
        f"\n"
        f"## Result\n"
        f"- success_count: {cell.success_count}\n"
        f"- completed_episode_total: {cell.completed_episode_total}\n"
        f"- rate: {cell.rate:.3f}\n"
        f"- wilson_ci_95: ({lo:.3f}, {hi:.3f})\n"
        f"- delta_vs_baseline: {cell.delta_vs_baseline:+.3f}\n"
        f"- newcombe_delta_ci_95: ({nlo:.3f}, {nhi:.3f})\n"
        f"- formal_eval_summary_status: "
        f"{cell.formal_eval_summary_json.get('status', '')}\n"
        f"\n"
        f"## Reproducibility envelope\n"
        f"- git_commit_sha: `{cell.git_commit_sha}`\n"
        f"{fallback_line}"
        f"- transformers_version: {cell.transformers_version}\n"
        f"- torch_version: {cell.torch_version}\n"
        f"- python_version: {py_first_line}\n"
        f"- gr00t_version: {cell.gr00t_version}\n"
        f"- nvidia_smi_pre_run_csv: `{cell.nvidia_smi_pre_run_csv}`\n"
        f"- protocol_sha256: `{cell.protocol_sha256}`\n"
        f"- r2_invocation_envelope_sha256: `{cell.r2_invocation_envelope_sha256}`\n"
        f"- r1_0_dir_present: {cell.r1_0_dir_present}\n"
        f"- r1_0_baseline_repro_latest_run_mtime_utc: "
        f"{cell.r1_0_baseline_repro_latest_run_mtime_utc}\n"
    )


__all__ = ["render"]
