"""CLI for regenerating R2 closure reports from frozen closure inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from work.recap.r1_repro.protocol import P0B_PROTOCOL
from work.recap.r1_repro.repro_runner import validate_baseline_pass_marker
from work.recap.r2_authentic_eval.eval_runner import AuthenticEvalRequest, R2CellResult
from work.recap.r2_authentic_eval.inventory import TrainedCheckpoint
from work.recap.r2_authentic_eval.reports import closure_report


def _checkpoint_from_json(data: dict[str, Any]) -> TrainedCheckpoint:
    """Rebuild the checkpoint record embedded in an existing cell_result.json."""
    return TrainedCheckpoint(
        label=str(data["label"]),
        abs_path=Path(data["abs_path"]),
        training_algo=str(data.get("training_algo", "")),
        base_ckpt_at_training=str(data.get("base_ckpt_at_training", "")),
        formalize_language=data.get("formalize_language"),
        statistics_q99_right_hand=tuple(float(v) for v in data["statistics_q99_right_hand"]),
        statistics_q99_matches_base=bool(data["statistics_q99_matches_base"]),
        n_train_steps=int(data["n_train_steps"]),
        training_run_dir=Path(data["training_run_dir"]),
        config_json_sha256=str(data.get("config_json_sha256", "")),
        processor_config_json_sha256=str(data.get("processor_config_json_sha256", "")),
        statistics_json_sha256=str(data.get("statistics_json_sha256", "")),
        is_valid=bool(data["is_valid"]),
        invalid_reason=str(data.get("invalid_reason", "")),
    )


def _cell_from_json(path: Path, trigger_by_abs_path: dict[str, bool]) -> R2CellResult:
    """Rebuild one R2CellResult and attach precomputed trigger status if present."""
    data = json.loads(path.read_text(encoding="utf-8"))
    request_data = dict(data["request"])
    checkpoint = _checkpoint_from_json(dict(request_data["checkpoint"]))
    request = AuthenticEvalRequest(
        checkpoint=checkpoint,
        search_root=Path(request_data["search_root"]),
        strict_config=bool(request_data["strict_config"]),
    )
    cell = R2CellResult(
        request=request,
        success_count=int(data["success_count"]),
        completed_episode_total=int(data["completed_episode_total"]),
        rate=float(data["rate"]),
        wilson_ci_95=tuple(float(v) for v in data["wilson_ci_95"]),
        delta_vs_baseline=float(data["delta_vs_baseline"]),
        newcombe_delta_ci_95=tuple(float(v) for v in data["newcombe_delta_ci_95"]),
        artifact_dir=Path(data["artifact_dir"]),
        formal_eval_summary_json=dict(data.get("formal_eval_summary_json") or {}),
        raw_repro_result=data.get("raw_repro_result"),
        ckpt_pre_run_sha256=dict(data.get("ckpt_pre_run_sha256") or {}),
        r1_0_dir_present=bool(data.get("r1_0_dir_present", False)),
        r1_0_baseline_repro_latest_run_mtime_utc=data.get(
            "r1_0_baseline_repro_latest_run_mtime_utc"
        ),
        git_commit_sha=str(data.get("git_commit_sha", "")),
        nvidia_smi_pre_run_csv=str(data.get("nvidia_smi_pre_run_csv", "")),
        transformers_version=str(data.get("transformers_version", "")),
        torch_version=str(data.get("torch_version", "")),
        python_version=str(data.get("python_version", "")),
        gr00t_version=data.get("gr00t_version"),
        protocol_sha256=str(data["protocol_sha256"]),
        r2_invocation_envelope_sha256=str(data["r2_invocation_envelope_sha256"]),
        git_commit_sha_fallback_reason=data.get("git_commit_sha_fallback_reason"),
        r2_cell_result_schema_version=str(data.get("r2_cell_result_schema_version", "")),
    )
    triggered = trigger_by_abs_path.get(str(checkpoint.abs_path))
    if triggered is not None:
        object.__setattr__(cell, "triggered_below_threshold", triggered)
    return cell


def _load_trigger_map(run_dir: Path) -> dict[str, bool]:
    """Load precomputed per-cell trigger statuses from the existing summary table."""
    summary_path = run_dir / "summary_table.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        str(row["abs_path"]): bool(row["triggered_below_threshold"])
        for row in summary.get("cells", ())
        if "abs_path" in row and "triggered_below_threshold" in row
    }


def _load_cells(run_dir: Path) -> list[R2CellResult]:
    """Load existing R2.1 cell results without rerunning evaluation."""
    trigger_by_abs_path = _load_trigger_map(run_dir)
    return [
        _cell_from_json(path, trigger_by_abs_path)
        for path in sorted(run_dir.rglob("cell_result.json"))
    ]


def _load_swap_decomposition(root: Path, trigger_by_abs_path: dict[str, bool]) -> dict[str, Any] | None:
    """Load R2.2 decomposition JSON and annotate metadata needed by the renderer."""
    table_path = root / "decomposition_table.json"
    if not table_path.exists():
        return None
    data = json.loads(table_path.read_text(encoding="utf-8"))
    data.setdefault("artifact_dir", str(root))
    rep_path = str(data.get("representative_path", ""))
    if rep_path in trigger_by_abs_path:
        data.setdefault("triggered", trigger_by_abs_path[rep_path])
    return data


def build_parser() -> argparse.ArgumentParser:
    """Build the narrow closure-report regeneration parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Existing closure_inputs.json")
    parser.add_argument("--out", required=True, help="Output r2_closure_report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Regenerate the closure report from frozen pointers in closure_inputs.json."""
    args = build_parser().parse_args(argv)
    inputs_path = Path(args.inputs)
    out_path = Path(args.out)
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    phase_e_run_dir = Path(inputs["phase_e_run_dir"])
    phase_f_root = Path(inputs["phase_f_root"])
    trigger_by_abs_path = _load_trigger_map(phase_e_run_dir)
    cells = _load_cells(phase_e_run_dir)
    baseline_marker = validate_baseline_pass_marker(P0B_PROTOCOL)
    config_delta_records = json.loads(
        Path(inputs["config_delta_inventory"]).read_text(encoding="utf-8")
    )
    swap_decomposition = _load_swap_decomposition(phase_f_root, trigger_by_abs_path)
    r2_invocation_sha = cells[0].r2_invocation_envelope_sha256 if cells else ""
    r1_0_dir_present = cells[0].r1_0_dir_present if cells else False
    r1_0_mtime = cells[0].r1_0_baseline_repro_latest_run_mtime_utc if cells else None
    markdown = closure_report.render(
        cells=cells,
        statistical_regime=dict(inputs["statistical_regime"]),
        baseline_marker=baseline_marker,
        representative_selection=dict(inputs.get("representative_selection") or {}),
        r2_invocation_envelope_sha256=r2_invocation_sha,
        r1_0_dir_present=r1_0_dir_present,
        r1_0_baseline_repro_latest_run_mtime_utc=r1_0_mtime,
        swap_decomposition=swap_decomposition,
        config_delta_records=config_delta_records,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
