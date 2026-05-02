from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


REPORT_SCHEMA_VERSION = "exact_probe_failure_analysis_v1"
REPORT_ARTIFACT_KIND = "exact_probe_failure_analysis"
DEFAULT_OUTPUT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/exact_probe_failure_analysis.json"
)

TASK_STAGE_FAILURE_REASON = "task_stage_failure"
PRECHECK_BLOCKER_REASON = "preflight_blocker"
RUNTIME_BLOCKER_REASON = "runtime_blocker"
METRIC_MISMATCH_REASON = "metric_mismatch_after_success_like_behavior"

TASK_OUTCOME_FAILURE_REASONS = {
    "truncated_without_success",
    "terminated_without_success",
    "done_without_success",
    "outer_step_budget_exhausted",
    "episode_incomplete_without_success",
}

PRECHECK_REASON_SNIPPETS = (
    "triplet_binding_blocked",
    "checkpoint",
    "manifest",
    "preflight",
    "binding",
    "artifact_kind",
    "contract",
    "formal_eligibility",
)

RUNTIME_REASON_SNIPPETS = (
    "timeout",
    "traceback",
    "exception",
    "server",
    "controller_output unavailable",
    "controller_output_unavailable",
    "eval_failed",
    "connection",
)

METRIC_ABSORPTION_THRESHOLD = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap.scripts import derive_probe_stage_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_exact_probe_failure.py",
        description=(
            "Analyze exact-probe style FAIL artifacts into high-level blocker/task buckets "
            "plus stage-level taxonomy with fail-soft confidence and missing signal tracking."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episode-json", type=Path, default=None)
    parser.add_argument("--steps-jsonl", type=Path, default=None)
    parser.add_argument("--runtime-trace-json", type=Path, default=None)
    parser.add_argument("--drop-episode-json", type=Path, default=None)
    parser.add_argument("--triplet-gate-json", type=Path, default=None)
    parser.add_argument("--action-telemetry-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"out must be a file path, got directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"expected JSON object in {path}:{line_number}, got {type(payload).__name__}"
            )
        rows.append(dict(payload))
    return rows


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalized_failure_reason(episode_record: Mapping[str, Any] | None) -> str | None:
    if not isinstance(episode_record, Mapping):
        return None
    return _optional_string(episode_record.get("failure_reason"))


def _runtime_stage_metric(
    runtime_trace: Mapping[str, Any] | None, stage_name: str
) -> float | None:
    if not isinstance(runtime_trace, Mapping):
        return None
    stage_max = runtime_trace.get("stage_max_mean_abs_delta_over_contract_range")
    if not isinstance(stage_max, Mapping):
        return None
    return _optional_float(stage_max.get(stage_name))


def _action_chain_absorption_detected(
    action_telemetry: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(action_telemetry, Mapping):
        return False
    if bool(action_telemetry.get("controller_absorbed_upstream_difference")):
        return True
    per_group = action_telemetry.get("per_group_stats")
    if not isinstance(per_group, Mapping):
        return False
    for payload in per_group.values():
        if not isinstance(payload, Mapping):
            continue
        difference_metrics = payload.get("difference_metrics")
        if not isinstance(difference_metrics, Mapping):
            continue
        disappearance_stage = _optional_string(
            difference_metrics.get("difference_disappeared_at")
        )
        if disappearance_stage in {
            "decode",
            "relative_to_absolute",
            "controller_input",
        }:
            return True
    return False


def _preflight_blocker(
    *,
    failure_reason: str | None,
    triplet_gate: Mapping[str, Any] | None,
) -> tuple[str | None, float, str | None]:
    if isinstance(triplet_gate, Mapping):
        formal_eligibility = _optional_string(triplet_gate.get("formal_eligibility"))
        reason_code = _optional_string(triplet_gate.get("reason_code"))
        issues = triplet_gate.get("issues")
        if formal_eligibility == "BLOCK" or reason_code == "triplet_binding_blocked":
            return PRECHECK_BLOCKER_REASON, 0.98, reason_code or formal_eligibility
        if isinstance(issues, list) and issues:
            return PRECHECK_BLOCKER_REASON, 0.95, reason_code or "issues_present"
    normalized_failure_reason = (failure_reason or "").lower()
    if (
        normalized_failure_reason
        and any(
            snippet in normalized_failure_reason for snippet in PRECHECK_REASON_SNIPPETS
        )
        and normalized_failure_reason not in TASK_OUTCOME_FAILURE_REASONS
    ):
        return PRECHECK_BLOCKER_REASON, 0.9, failure_reason
    return None, 0.0, None


def _runtime_blocker(
    *,
    failure_reason: str | None,
    runtime_trace: Mapping[str, Any] | None,
) -> tuple[str | None, float, str | None]:
    normalized_failure_reason = (failure_reason or "").lower()
    if normalized_failure_reason in TASK_OUTCOME_FAILURE_REASONS:
        return None, 0.0, None
    if normalized_failure_reason and any(
        snippet in normalized_failure_reason for snippet in RUNTIME_REASON_SNIPPETS
    ):
        return RUNTIME_BLOCKER_REASON, 0.95, failure_reason
    if isinstance(runtime_trace, Mapping):
        status = _optional_string(runtime_trace.get("status"))
        if status in {"ERROR", "UNAVAILABLE", "BLOCKED"}:
            return RUNTIME_BLOCKER_REASON, 0.9, status
    return None, 0.0, None


def _success_like_behavior(
    *,
    episode_record: Mapping[str, Any] | None,
    stage_payload: Mapping[str, Any],
) -> bool:
    progress_flags = stage_payload.get("progress_flags")
    flags = dict(progress_flags) if isinstance(progress_flags, Mapping) else {}
    success_steps = 0
    if isinstance(episode_record, Mapping):
        raw_n_success_steps = episode_record.get("n_success_steps")
        if isinstance(raw_n_success_steps, int) and not isinstance(
            raw_n_success_steps, bool
        ):
            success_steps = int(raw_n_success_steps)
    return bool(
        success_steps > 0
        or flags.get("success_step_seen") is True
        or (
            flags.get("grasp_seen") is True
            and flags.get("near_plate_seen") is True
            and flags.get("place_seen") is True
        )
    )


def _metric_mismatch_detected(
    *,
    episode_record: Mapping[str, Any] | None,
    stage_payload: Mapping[str, Any],
    runtime_trace: Mapping[str, Any] | None,
    action_telemetry: Mapping[str, Any] | None,
) -> tuple[bool, float, str | None]:
    if not _success_like_behavior(
        episode_record=episode_record, stage_payload=stage_payload
    ):
        return False, 0.0, None
    prompt_or_token_distinct = False
    raw_or_decoded_distinct = False
    decoded_delta = None
    absolute_delta = None
    controller_delta = None
    if isinstance(runtime_trace, Mapping):
        upstream = runtime_trace.get("upstream_distinction")
        if isinstance(upstream, Mapping):
            prompt_or_token_distinct = bool(
                upstream.get("prompt_or_token_distinct", False)
            )
            raw_or_decoded_distinct = bool(
                upstream.get("raw_or_decoded_distinct", False)
            )
        decoded_delta = _runtime_stage_metric(runtime_trace, "decoded_action")
        absolute_delta = _runtime_stage_metric(runtime_trace, "absolute_action")
        controller_output_delta = _runtime_stage_metric(
            runtime_trace, "controller_output"
        )
        controller_input_delta = _runtime_stage_metric(
            runtime_trace, "controller_input"
        )
        controller_delta = (
            controller_output_delta
            if controller_output_delta is not None
            else controller_input_delta
        )
    runtime_absorbed = bool(
        prompt_or_token_distinct
        and raw_or_decoded_distinct
        and (
            (decoded_delta is not None and decoded_delta > METRIC_ABSORPTION_THRESHOLD)
            or (
                absolute_delta is not None
                and absolute_delta > METRIC_ABSORPTION_THRESHOLD
            )
        )
        and (
            controller_delta is None or controller_delta <= METRIC_ABSORPTION_THRESHOLD
        )
    )
    if runtime_absorbed:
        return True, 0.9, "runtime_trace_absorbed_before_controller"
    if _action_chain_absorption_detected(action_telemetry):
        return (
            True,
            0.86,
            "action_chain_difference_disappeared_before_controller_output",
        )
    return False, 0.0, None


def _stage_reason_from_fallback(
    stage_payload: Mapping[str, Any],
) -> tuple[str | None, float, str | None]:
    failure_stage_guess = stage_payload.get("normalized_failure_stage_guess")
    if not isinstance(failure_stage_guess, Mapping):
        return None, 0.0, None
    mapped = _optional_string(failure_stage_guess.get("mapped_stage_level_reason"))
    if mapped is None:
        return None, 0.0, None
    return mapped, 0.45, "failure_stage_guess"


def _stage_signal_mode(stage_payload: Mapping[str, Any]) -> str:
    stage_events = stage_payload.get("stage_events")
    if not isinstance(stage_events, list) or not stage_events:
        return "none"
    confidences = [
        float(confidence)
        for event in stage_events
        if isinstance(event, Mapping)
        for confidence in [_optional_float(event.get("confidence"))]
        if confidence is not None
    ]
    if not confidences:
        return "none"
    max_confidence = max(confidences)
    if max_confidence >= 0.9:
        return "direct"
    if max_confidence >= 0.7:
        return "geometry"
    return "heuristic"


def _stage_reason_from_progress(
    stage_payload: Mapping[str, Any],
) -> tuple[str | None, float, str | None]:
    progress_flags = stage_payload.get("progress_flags")
    geometry_metrics = stage_payload.get("geometry_metrics")
    flags = dict(progress_flags) if isinstance(progress_flags, Mapping) else {}
    metrics = dict(geometry_metrics) if isinstance(geometry_metrics, Mapping) else {}
    if not any(bool(value) for value in flags.values()):
        return None, 0.0, None

    signal_mode = _stage_signal_mode(stage_payload)
    if signal_mode == "direct":
        base_confidence = 0.9
    elif signal_mode == "geometry":
        base_confidence = 0.75
    elif signal_mode == "heuristic":
        base_confidence = 0.45
    else:
        base_confidence = 0.2

    if flags.get("drop_during_transport_seen") is True:
        return "drop_during_transport", 0.95, "drop_during_transport"
    if flags.get("approach_seen") is not True:
        return "approach_failed", base_confidence, "stage_events"
    if flags.get("grasp_seen") is not True:
        return "grasp_failed", base_confidence, "stage_events"
    if flags.get("transport_seen") is not True:
        return "transport_failed", base_confidence, "stage_events"
    if flags.get("near_plate_seen") is not True:
        improvement = _optional_float(metrics.get("plate_progress_improvement_m"))
        if improvement is not None and improvement >= 0.02:
            return "near_plate_failed", 0.75, "geometry_progress"
        return "transport_failed", base_confidence, "stage_events"
    if flags.get("place_seen") is not True:
        return "near_plate_failed", base_confidence, "stage_events"
    return "place_failed", base_confidence, "stage_events"


def analyze_exact_probe_failure(
    *,
    episode_record: Mapping[str, Any] | None = None,
    step_records: Sequence[Mapping[str, Any]] | None = None,
    runtime_trace: Mapping[str, Any] | None = None,
    drop_episode_summary: Mapping[str, Any] | None = None,
    triplet_gate: Mapping[str, Any] | None = None,
    action_telemetry: Mapping[str, Any] | None = None,
    stage_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    episode_payload = (
        dict(episode_record) if isinstance(episode_record, Mapping) else {}
    )
    stage_report = (
        dict(stage_payload)
        if isinstance(stage_payload, Mapping)
        else derive_probe_stage_events.derive_probe_stage_events(
            episode_record=episode_payload,
            step_records=step_records,
            runtime_trace=runtime_trace,
            drop_episode_summary=drop_episode_summary,
        )
    )
    failure_reason = _normalized_failure_reason(episode_payload)
    missing_signals = list(stage_report.get("missing_signals", []))

    high_level_reason, confidence, decisive_signal = _preflight_blocker(
        failure_reason=failure_reason,
        triplet_gate=triplet_gate,
    )
    stage_level_reason: str | None = None
    analysis_source = decisive_signal

    if high_level_reason is None:
        high_level_reason, confidence, decisive_signal = _runtime_blocker(
            failure_reason=failure_reason,
            runtime_trace=runtime_trace,
        )
        analysis_source = decisive_signal

    if high_level_reason is None:
        mismatch_detected, mismatch_confidence, mismatch_source = (
            _metric_mismatch_detected(
                episode_record=episode_payload,
                stage_payload=stage_report,
                runtime_trace=runtime_trace,
                action_telemetry=action_telemetry,
            )
        )
        if mismatch_detected:
            high_level_reason = METRIC_MISMATCH_REASON
            stage_level_reason = METRIC_MISMATCH_REASON
            confidence = mismatch_confidence
            analysis_source = mismatch_source

    if high_level_reason is None:
        high_level_reason = TASK_STAGE_FAILURE_REASON
        stage_level_reason, confidence, analysis_source = _stage_reason_from_progress(
            stage_report
        )
        if stage_level_reason is None:
            stage_level_reason, confidence, analysis_source = (
                _stage_reason_from_fallback(stage_report)
            )
        if stage_level_reason is None:
            stage_level_reason = "approach_failed"
            confidence = 0.2
            analysis_source = "low_signal_default"

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "high_level_reason": high_level_reason,
        "stage_level_reason": stage_level_reason,
        "confidence": float(round(confidence, 4)),
        "missing_signals": missing_signals,
        "failure_reason": failure_reason,
        "analysis_source": analysis_source,
        "normalized_failure_stage_guess": stage_report.get(
            "normalized_failure_stage_guess"
        ),
        "stage_events": stage_report.get("stage_events", []),
        "progress_flags": stage_report.get("progress_flags", {}),
        "geometry_metrics": stage_report.get("geometry_metrics", {}),
    }
    if isinstance(triplet_gate, Mapping):
        report["triplet_gate_reason_code"] = _optional_string(
            triplet_gate.get("reason_code")
        )
    if isinstance(runtime_trace, Mapping):
        report["runtime_trace_status"] = _optional_string(runtime_trace.get("status"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        episode_record = _read_json(args.episode_json) if args.episode_json else None
        step_records = _read_jsonl(args.steps_jsonl) if args.steps_jsonl else None
        runtime_trace = (
            _read_json(args.runtime_trace_json) if args.runtime_trace_json else None
        )
        drop_episode_summary = (
            _read_json(args.drop_episode_json) if args.drop_episode_json else None
        )
        triplet_gate = (
            _read_json(args.triplet_gate_json) if args.triplet_gate_json else None
        )
        action_telemetry = (
            _read_json(args.action_telemetry_json)
            if args.action_telemetry_json
            else None
        )
        payload = analyze_exact_probe_failure(
            episode_record=episode_record,
            step_records=step_records,
            runtime_trace=runtime_trace,
            drop_episode_summary=drop_episode_summary,
            triplet_gate=triplet_gate,
            action_telemetry=action_telemetry,
        )
        output_path = _validate_output_path(args.out)
        if output_path is not None:
            payload["output_path"] = str(output_path)
            _ = _write_json(output_path, payload)
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
