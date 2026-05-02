#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import advantage
from work.recap.scripts import analyze_exact_probe_failure
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import build_uplift_schemas
from work.recap.scripts import derive_probe_stage_events


DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/apple_recap_flux_graft/diagnostic_probe_bypass"
)
DIAGNOSTIC_GAP_JSON_NAME = "diagnostic_probe_vs_screening_gap.json"

SCREENING_SCHEMA_VERSION = "flux_diagnostic_probe_bypass_screening_gap_v1"
SCREENING_ARTIFACT_KIND = "flux_diagnostic_probe_bypass_screening_gap"
SCREENING_MODE = "diagnostic_probe_bypass"
DIAGNOSTIC_SURFACE_ROUTE = "diagnostic_probe_vs_screening_gap"
DIAGNOSTIC_AUTHORITY_SCOPE = "diagnostic_probe_bypass_lane"

COMPARISON_VERDICT_ENUM: tuple[str, ...] = (
    "probe_likely_too_strict",
    "checkpoint_or_policy_likely_weak",
    "runtime_control_distortion_suspected",
    "insufficient_evidence",
)

_BASELINE_ROW_LABEL = build_uplift_schemas.LEDGER_REQUIRED_ROW_LABELS[0]
_OPTIONAL_E1_ROW_LABEL = build_uplift_schemas.LEDGER_REQUIRED_ROW_LABELS[1]
EXCLUDED_ROW_LABELS: tuple[str, ...] = tuple(
    build_uplift_schemas.LEDGER_REQUIRED_ROW_LABELS[2:]
)

ROW_SIGNAL_POSITIVE = build_uplift_schemas.ROW_SIGNAL_ENUM[0]
ROW_SIGNAL_FLAT = build_uplift_schemas.ROW_SIGNAL_ENUM[1]
ROW_SIGNAL_NEGATIVE = build_uplift_schemas.ROW_SIGNAL_ENUM[2]
ROW_SIGNAL_INCONCLUSIVE = build_uplift_schemas.ROW_SIGNAL_ENUM[3]

_STAGE_RANKS = {
    "approach_failed": 1,
    "grasp_failed": 2,
    "transport_failed": 3,
    "drop_during_transport": 3,
    "near_plate_failed": 4,
    "place_failed": 5,
    analyze_exact_probe_failure.METRIC_MISMATCH_REASON: 6,
}
_LATE_STAGE_THRESHOLD = _STAGE_RANKS["near_plate_failed"]
_WEAK_STAGE_THRESHOLD = _STAGE_RANKS["transport_failed"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_screening_probe_bypass_diagnostic.py",
        description=(
            "Materialize the diagnostic-only probe-bypass gap report. The default "
            "row set is exactly ['B0']; pass --include-e1 to extend it to exactly "
            "['B0', 'E1']. This lane is fenced off from authoritative screening and "
            "final-report consumers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory that receives the diagnostic-only "
            "diagnostic_probe_vs_screening_gap.json artifact."
        ),
    )
    parser.add_argument(
        "--include-e1",
        action="store_true",
        help="Extend the diagnostic comparison row set from ['B0'] to ['B0', 'E1'].",
    )
    _add_row_input_args(parser, row_label=_BASELINE_ROW_LABEL)
    _add_row_input_args(parser, row_label=_OPTIONAL_E1_ROW_LABEL)
    return parser


def _add_row_input_args(parser: argparse.ArgumentParser, *, row_label: str) -> None:
    lowered = str(row_label).lower()
    parser.add_argument(
        f"--{lowered}-episode-json",
        type=Path,
        default=None,
        help=f"Optional {row_label} episode-level JSON payload.",
    )
    parser.add_argument(
        f"--{lowered}-steps-jsonl",
        type=Path,
        default=None,
        help=f"Optional {row_label} step-level JSONL payload.",
    )
    parser.add_argument(
        f"--{lowered}-runtime-trace-json",
        type=Path,
        default=None,
        help=f"Optional {row_label} runtime-trace JSON payload.",
    )
    parser.add_argument(
        f"--{lowered}-drop-episode-json",
        type=Path,
        default=None,
        help=f"Optional {row_label} drop-detector episode summary JSON payload.",
    )
    parser.add_argument(
        f"--{lowered}-triplet-gate-json",
        type=Path,
        default=None,
        help=f"Optional {row_label} triplet gate JSON payload.",
    )
    parser.add_argument(
        f"--{lowered}-action-telemetry-json",
        type=Path,
        default=None,
        help=f"Optional {row_label} action-telemetry JSON payload.",
    )


def diagnostic_row_labels(*, include_e1: bool = False) -> tuple[str, ...]:
    if include_e1:
        return (_BASELINE_ROW_LABEL, _OPTIONAL_E1_ROW_LABEL)
    return (_BASELINE_ROW_LABEL,)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _signature_for_payload(payload: Mapping[str, Any]) -> str:
    signature_basis = {
        str(key): value
        for key, value in dict(payload).items()
        if key != "report_signature_sha256"
    }
    return apple_recap_execution_contract._sha256_payload(signature_basis)


def _resolve_repo_path(repo_root: Path, raw: Path | str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_output_dir(path: Path | str, *, repo_root: Path) -> Path:
    resolved = _resolve_repo_path(repo_root, path)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output-dir must be a directory path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _read_json_optional(path: Path | None, *, repo_root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = _resolve_repo_path(repo_root, path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"expected JSON object in {resolved}, got {type(payload).__name__}"
        )
    return dict(cast(Mapping[str, Any], payload))


def _read_jsonl_optional(
    path: Path | None, *, repo_root: Path
) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    resolved = _resolve_repo_path(repo_root, path)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"expected JSON object in {resolved}:{line_number}, got {type(payload).__name__}"
            )
        rows.append(dict(cast(Mapping[str, Any], payload)))
    return rows


def _row_surface_metadata(*, row_label: str) -> dict[str, Any]:
    payload = advantage.build_diagnostic_surface_metadata(
        surface_route=DIAGNOSTIC_SURFACE_ROUTE,
        authority_scope=DIAGNOSTIC_AUTHORITY_SCOPE,
        compatibility_fields=(
            "row_id",
            "comparable_to",
            "row_signal",
            "high_level_reason",
            "stage_level_reason",
            "confidence",
        ),
        surface_kind="probe_bypass_row_diagnostic",
    )
    payload.update(
        {
            "row_id": str(row_label),
            "display_label": str(row_label),
            "main_verdict_eligible": False,
            "external_reference_only": True,
            "screening_mode": SCREENING_MODE,
        }
    )
    return payload


def build_probe_evidence_row_report(
    *,
    row_label: str,
    evidence: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
    comparable_to: str | None = None,
) -> dict[str, Any]:
    evidence_payload = dict(cast(Mapping[str, Any], evidence or {}))
    episode_record = cast(
        Mapping[str, Any] | None,
        evidence_payload.get("episode_record"),
    )
    step_records = cast(
        Sequence[Mapping[str, Any]] | None,
        evidence_payload.get("step_records"),
    )
    runtime_trace = cast(
        Mapping[str, Any] | None,
        evidence_payload.get("runtime_trace"),
    )
    drop_episode_summary = cast(
        Mapping[str, Any] | None,
        evidence_payload.get("drop_episode_summary"),
    )
    triplet_gate = cast(
        Mapping[str, Any] | None,
        evidence_payload.get("triplet_gate"),
    )
    action_telemetry = cast(
        Mapping[str, Any] | None,
        evidence_payload.get("action_telemetry"),
    )
    stage_payload = derive_probe_stage_events.derive_probe_stage_events(
        episode_record=episode_record,
        step_records=step_records,
        runtime_trace=runtime_trace,
        drop_episode_summary=drop_episode_summary,
    )
    failure_analysis = analyze_exact_probe_failure.analyze_exact_probe_failure(
        episode_record=episode_record,
        step_records=step_records,
        runtime_trace=runtime_trace,
        drop_episode_summary=drop_episode_summary,
        triplet_gate=triplet_gate,
        action_telemetry=action_telemetry,
        stage_payload=stage_payload,
    )
    row_signal = _row_signal_from_analysis(failure_analysis)
    stage_rank = _stage_rank(failure_analysis.get("stage_level_reason"))
    high_level_reason = str(failure_analysis.get("high_level_reason") or "").strip()
    return {
        **dict(metadata),
        "comparable_to": comparable_to,
        "row_signal": row_signal,
        "failure_analysis": failure_analysis,
        "stage_events": stage_payload.get("stage_events", []),
        "progress_flags": stage_payload.get("progress_flags", {}),
        "geometry_metrics": stage_payload.get("geometry_metrics", {}),
        "missing_signals": list(failure_analysis.get("missing_signals", [])),
        "comparison_features": {
            "high_level_reason": high_level_reason,
            "stage_level_reason": failure_analysis.get("stage_level_reason"),
            "confidence": float(failure_analysis.get("confidence") or 0.0),
            "stage_rank": stage_rank,
            "late_stage_signal": bool(stage_rank >= _LATE_STAGE_THRESHOLD),
            "runtime_distortion_signal": bool(
                high_level_reason == analyze_exact_probe_failure.METRIC_MISMATCH_REASON
            ),
        },
        "input_presence": {
            "episode_record": isinstance(episode_record, Mapping),
            "step_records": isinstance(step_records, Sequence),
            "runtime_trace": isinstance(runtime_trace, Mapping),
            "drop_episode_summary": isinstance(drop_episode_summary, Mapping),
            "triplet_gate": isinstance(triplet_gate, Mapping),
            "action_telemetry": isinstance(action_telemetry, Mapping),
        },
    }


def _stage_rank(stage_level_reason: object) -> int:
    return int(_STAGE_RANKS.get(str(stage_level_reason or "").strip(), 0))


def _row_signal_from_analysis(failure_analysis: Mapping[str, Any]) -> str:
    high_level_reason = str(failure_analysis.get("high_level_reason") or "").strip()
    stage_rank = _stage_rank(failure_analysis.get("stage_level_reason"))
    confidence = float(failure_analysis.get("confidence") or 0.0)
    if high_level_reason == analyze_exact_probe_failure.METRIC_MISMATCH_REASON:
        return ROW_SIGNAL_POSITIVE
    if stage_rank >= _LATE_STAGE_THRESHOLD and confidence >= 0.7:
        return ROW_SIGNAL_POSITIVE
    if (
        high_level_reason == analyze_exact_probe_failure.TASK_STAGE_FAILURE_REASON
        and stage_rank > 0
        and stage_rank <= _WEAK_STAGE_THRESHOLD
    ):
        return ROW_SIGNAL_NEGATIVE
    return ROW_SIGNAL_INCONCLUSIVE


def _row_comparable_to(row_label: str) -> str | None:
    if row_label == _OPTIONAL_E1_ROW_LABEL:
        return _BASELINE_ROW_LABEL
    return None


def _build_row_report(
    *,
    row_label: str,
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return build_probe_evidence_row_report(
        row_label=row_label,
        evidence=evidence,
        metadata=_row_surface_metadata(row_label=row_label),
        comparable_to=_row_comparable_to(row_label),
    )


def validate_probe_bypass_diagnostic_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(cast(Mapping[str, Any], payload))
    if normalized.get("schema_version") != SCREENING_SCHEMA_VERSION:
        raise ValueError(
            "diagnostic_probe_bypass.schema_version mismatch: expected "
            + repr(SCREENING_SCHEMA_VERSION)
        )
    if normalized.get("artifact_kind") != SCREENING_ARTIFACT_KIND:
        raise ValueError(
            "diagnostic_probe_bypass.artifact_kind mismatch: expected "
            + repr(SCREENING_ARTIFACT_KIND)
        )
    if normalized.get("screening_mode") != SCREENING_MODE:
        raise ValueError(
            "diagnostic_probe_bypass.screening_mode mismatch: expected "
            + repr(SCREENING_MODE)
        )
    if bool(normalized.get("diagnostic_only")) is not True:
        raise ValueError("diagnostic_probe_bypass.diagnostic_only mismatch")
    if bool(normalized.get("mainline_authority")) is not False:
        raise ValueError("diagnostic_probe_bypass.mainline_authority mismatch")
    if bool(normalized.get("main_verdict_eligible")) is not False:
        raise ValueError("diagnostic_probe_bypass.main_verdict_eligible mismatch")
    if bool(normalized.get("external_reference_only")) is not True:
        raise ValueError("diagnostic_probe_bypass.external_reference_only mismatch")
    comparison_verdict = str(normalized.get("comparison_verdict") or "").strip()
    if comparison_verdict not in COMPARISON_VERDICT_ENUM:
        raise ValueError("diagnostic_probe_bypass.comparison_verdict is not recognized")
    row_labels = normalized.get("diagnostic_row_labels")
    if not isinstance(row_labels, list) or any(
        not isinstance(item, str) or not item.strip() for item in row_labels
    ):
        raise ValueError(
            "diagnostic_probe_bypass.diagnostic_row_labels must be a string list"
        )
    expected_labels = list(
        diagnostic_row_labels(include_e1=(_OPTIONAL_E1_ROW_LABEL in row_labels))
    )
    if list(row_labels) != expected_labels:
        raise ValueError(
            "diagnostic_probe_bypass row set does not match the frozen B0/E1 surface"
        )
    rows = normalized.get("rows")
    if not isinstance(rows, Mapping):
        raise ValueError("diagnostic_probe_bypass.rows must be an object")
    missing_rows = [row_label for row_label in row_labels if row_label not in rows]
    if missing_rows:
        raise ValueError(
            "diagnostic_probe_bypass.rows missing expected labels: "
            + ", ".join(missing_rows)
        )
    declared_signature = str(normalized.get("report_signature_sha256") or "").strip()
    expected_signature = _signature_for_payload(normalized)
    if declared_signature != expected_signature:
        raise ValueError(
            "diagnostic_probe_bypass.report_signature_sha256 mismatch: expected "
            + repr(expected_signature)
        )
    return normalized


def _has_runtime_distortion_signal(row_report: Mapping[str, Any]) -> bool:
    features = cast(Mapping[str, Any], row_report.get("comparison_features", {}))
    return bool(features.get("runtime_distortion_signal") is True)


def _is_weak_row_signal(row_report: Mapping[str, Any]) -> bool:
    features = cast(Mapping[str, Any], row_report.get("comparison_features", {}))
    high_level_reason = str(features.get("high_level_reason") or "").strip()
    stage_rank = int(features.get("stage_rank") or 0)
    confidence = float(features.get("confidence") or 0.0)
    return bool(
        high_level_reason == analyze_exact_probe_failure.TASK_STAGE_FAILURE_REASON
        and 0 < stage_rank <= _WEAK_STAGE_THRESHOLD
        and confidence >= 0.45
    )


def _is_late_stage_row_signal(row_report: Mapping[str, Any]) -> bool:
    features = cast(Mapping[str, Any], row_report.get("comparison_features", {}))
    stage_rank = int(features.get("stage_rank") or 0)
    confidence = float(features.get("confidence") or 0.0)
    return bool(stage_rank >= _LATE_STAGE_THRESHOLD and confidence >= 0.7)


def _has_non_task_blocker(row_report: Mapping[str, Any]) -> bool:
    features = cast(Mapping[str, Any], row_report.get("comparison_features", {}))
    high_level_reason = str(features.get("high_level_reason") or "").strip()
    return high_level_reason in {
        analyze_exact_probe_failure.PRECHECK_BLOCKER_REASON,
        analyze_exact_probe_failure.RUNTIME_BLOCKER_REASON,
    }


def determine_comparison_verdict(
    row_reports: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    normalized_rows = {
        str(row_label): dict(cast(Mapping[str, Any], payload))
        for row_label, payload in row_reports.items()
    }
    ordered_rows = list(normalized_rows.values())
    if any(_has_runtime_distortion_signal(row) for row in ordered_rows):
        return (
            "runtime_control_distortion_suspected",
            {
                "trigger": "metric_mismatch_after_success_like_behavior",
                "row_signals": {
                    row_label: row.get("row_signal")
                    for row_label, row in normalized_rows.items()
                },
            },
        )
    if any(_has_non_task_blocker(row) for row in ordered_rows):
        return (
            "insufficient_evidence",
            {
                "trigger": "non_task_blocker_present",
                "row_signals": {
                    row_label: row.get("row_signal")
                    for row_label, row in normalized_rows.items()
                },
            },
        )
    b0_row = normalized_rows.get(_BASELINE_ROW_LABEL)
    e1_row = normalized_rows.get(_OPTIONAL_E1_ROW_LABEL)
    if (
        b0_row is not None
        and e1_row is not None
        and _is_weak_row_signal(b0_row)
        and _is_late_stage_row_signal(e1_row)
    ):
        b0_rank = int(
            cast(Mapping[str, Any], b0_row.get("comparison_features", {})).get(
                "stage_rank", 0
            )
            or 0
        )
        e1_rank = int(
            cast(Mapping[str, Any], e1_row.get("comparison_features", {})).get(
                "stage_rank", 0
            )
            or 0
        )
        if e1_rank > b0_rank:
            return (
                "probe_likely_too_strict",
                {
                    "trigger": "e1_advances_beyond_b0_under_diagnostic_probe_bypass",
                    "baseline_stage_rank": b0_rank,
                    "comparison_stage_rank": e1_rank,
                },
            )
    if ordered_rows and all(_is_weak_row_signal(row) for row in ordered_rows):
        return (
            "checkpoint_or_policy_likely_weak",
            {
                "trigger": "all_available_rows_fail_before_late_stage",
                "row_signals": {
                    row_label: row.get("row_signal")
                    for row_label, row in normalized_rows.items()
                },
            },
        )
    return (
        "insufficient_evidence",
        {
            "trigger": "diagnostic_gap_not_directional",
            "row_signals": {
                row_label: row.get("row_signal")
                for row_label, row in normalized_rows.items()
            },
        },
    )


def build_probe_bypass_diagnostic_payload(
    *,
    row_evidence_by_label: Mapping[str, Mapping[str, Any] | None] | None = None,
    include_e1: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    selected_row_labels = diagnostic_row_labels(include_e1=include_e1)
    resolved_output_dir = _validate_output_dir(output_dir, repo_root=repo_root)
    normalized_evidence = {
        str(key): value for key, value in dict(row_evidence_by_label or {}).items()
    }
    row_reports = {
        row_label: _build_row_report(
            row_label=row_label,
            evidence=normalized_evidence.get(row_label),
        )
        for row_label in selected_row_labels
    }
    comparison_verdict, verdict_basis = determine_comparison_verdict(row_reports)
    payload: dict[str, Any] = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "artifact_kind": SCREENING_ARTIFACT_KIND,
        "screening_mode": SCREENING_MODE,
        "generated_at": generated_at or _now_iso(),
        "diagnostic_only": True,
        "mainline_authority": False,
        "main_verdict_eligible": False,
        "external_reference_only": True,
        "authority_status": "diagnostic_only",
        "authority_scope": DIAGNOSTIC_AUTHORITY_SCOPE,
        "surface_route": DIAGNOSTIC_SURFACE_ROUTE,
        "surface_kind": "probe_bypass_gap_summary",
        "output_dir": _repo_relative_path(repo_root, resolved_output_dir),
        "artifact_path": _repo_relative_path(
            repo_root,
            resolved_output_dir / DIAGNOSTIC_GAP_JSON_NAME,
        ),
        "diagnostic_row_labels": list(selected_row_labels),
        "excluded_mainline_rows": list(EXCLUDED_ROW_LABELS),
        "allowed_comparison_verdicts": list(COMPARISON_VERDICT_ENUM),
        "comparison_verdict": comparison_verdict,
        "comparison_basis": verdict_basis,
        "rows": row_reports,
    }
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    return payload


def materialize_probe_bypass_diagnostic(
    *,
    row_evidence_by_label: Mapping[str, Mapping[str, Any] | None] | None = None,
    include_e1: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    payload = build_probe_bypass_diagnostic_payload(
        row_evidence_by_label=row_evidence_by_label,
        include_e1=include_e1,
        output_dir=output_dir,
        repo_root=repo_root,
        generated_at=generated_at,
    )
    resolved_output_dir = _validate_output_dir(output_dir, repo_root=repo_root)
    _write_json(resolved_output_dir / DIAGNOSTIC_GAP_JSON_NAME, payload)
    return payload


def _row_evidence_from_cli(
    *,
    row_label: str,
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, Any]:
    lowered = str(row_label).lower()
    return {
        "episode_record": _read_json_optional(
            getattr(args, f"{lowered}_episode_json"),
            repo_root=repo_root,
        ),
        "step_records": _read_jsonl_optional(
            getattr(args, f"{lowered}_steps_jsonl"),
            repo_root=repo_root,
        ),
        "runtime_trace": _read_json_optional(
            getattr(args, f"{lowered}_runtime_trace_json"),
            repo_root=repo_root,
        ),
        "drop_episode_summary": _read_json_optional(
            getattr(args, f"{lowered}_drop_episode_json"),
            repo_root=repo_root,
        ),
        "triplet_gate": _read_json_optional(
            getattr(args, f"{lowered}_triplet_gate_json"),
            repo_root=repo_root,
        ),
        "action_telemetry": _read_json_optional(
            getattr(args, f"{lowered}_action_telemetry_json"),
            repo_root=repo_root,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        row_evidence_by_label = {
            row_label: _row_evidence_from_cli(
                row_label=row_label,
                args=args,
                repo_root=REPO_ROOT,
            )
            for row_label in diagnostic_row_labels(include_e1=bool(args.include_e1))
        }
        payload = materialize_probe_bypass_diagnostic(
            row_evidence_by_label=row_evidence_by_label,
            include_e1=bool(args.include_e1),
            output_dir=args.output_dir,
            repo_root=REPO_ROOT,
        )
    except (
        AttributeError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "COMPARISON_VERDICT_ENUM",
    "DEFAULT_OUTPUT_DIR",
    "DIAGNOSTIC_GAP_JSON_NAME",
    "EXCLUDED_ROW_LABELS",
    "ROW_SIGNAL_FLAT",
    "ROW_SIGNAL_INCONCLUSIVE",
    "ROW_SIGNAL_NEGATIVE",
    "ROW_SIGNAL_POSITIVE",
    "SCREENING_ARTIFACT_KIND",
    "SCREENING_MODE",
    "SCREENING_SCHEMA_VERSION",
    "build_parser",
    "build_probe_evidence_row_report",
    "build_probe_bypass_diagnostic_payload",
    "determine_comparison_verdict",
    "diagnostic_row_labels",
    "main",
    "materialize_probe_bypass_diagnostic",
    "validate_probe_bypass_diagnostic_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
