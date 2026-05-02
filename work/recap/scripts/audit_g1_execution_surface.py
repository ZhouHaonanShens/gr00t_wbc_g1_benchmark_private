from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPORT_SCHEMA_VERSION = "g1_execution_surface_audit_v1"
REPORT_ARTIFACT_KIND = "g1_execution_surface_audit"
DEFAULT_OUTPUT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/g1_execution_surface_audit.json"
)
WRITER_SCRIPT = "work/recap/scripts/audit_g1_execution_surface.py"

VERDICT_POLICY = "policy"
VERDICT_POSTPROCESS = "postprocess"
VERDICT_CONTROLLER_DISTORTION = "controller_distortion"
VERDICT_UNKNOWN = "unknown"
VERDICT_BLOCKED = "blocked"
VERDICT_ENUM: tuple[str, ...] = (
    VERDICT_POLICY,
    VERDICT_POSTPROCESS,
    VERDICT_CONTROLLER_DISTORTION,
    VERDICT_UNKNOWN,
    VERDICT_BLOCKED,
)

MACHINE_CHECKPOINT_ORDER = (
    "raw_action",
    "decoded_action",
    "absolute_action",
    "controller_input",
    "controller_output",
)
SUMMARY_STAGE_ORDER = (
    "preprocess",
    "predict",
    "postprocess",
    "execute",
)
DISTINCTION_EPS = 1e-6
ZERO_OUTPUT_RATE_MIN = 0.95

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap.scripts import gr00t_action_chain_telemetry
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_g1_execution_surface.py",
        description=(
            "Audit the Unitree G1 execution surface so downstream consumers can distinguish "
            "policy weakness from postprocess/controller-side distortion using existing live "
            "runtime-trace and action-chain surfaces."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--triplet-summary-json",
        type=str,
        default="",
        help=(
            "Same-checkpoint triplet summary JSON. When provided, runtime_trace and action_delta_audit "
            "are consumed from this live artifact."
        ),
    )
    _ = parser.add_argument(
        "--runtime-trace-json",
        type=str,
        default="",
        help="Optional standalone runtime_trace JSON override.",
    )
    _ = parser.add_argument(
        "--action-telemetry-json",
        type=str,
        default="",
        help=(
            "Optional action-chain telemetry JSON produced by gr00t_action_chain_telemetry.py or a "
            "compatible per_group_stats surface."
        ),
    )
    _ = parser.add_argument(
        "--action-absorption-json",
        type=str,
        default="",
        help=(
            "Optional action absorption root-cause JSON produced by gr00t_action_absorption_audit.py."
        ),
    )
    _ = parser.add_argument(
        "--eval-summary-json",
        type=str,
        default="",
        help=(
            "Optional 3D_recap_eval summary JSON. This does not drive the verdict directly; it only "
            "adds machine-readable telemetry backpointers."
        ),
    )
    _ = parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output JSON path for the execution-surface audit.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _validate_output_path(path: Path) -> Path:
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


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return int(value)


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _sorted_strings(values: set[str]) -> list[str]:
    return sorted(str(value) for value in values)


def _issue(code: str, field_path: str, message: str) -> dict[str, str]:
    return {
        "code": str(code),
        "field_path": str(field_path),
        "message": str(message),
    }


def _dedupe_issues(issues: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in issues:
        code = str(item.get("code", "unknown"))
        field_path = str(item.get("field_path", "$"))
        message = str(item.get("message", ""))
        key = (code, field_path, message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "code": code,
                "field_path": field_path,
                "message": message,
            }
        )
    return deduped


def _runtime_stage_metric(
    runtime_trace: Mapping[str, Any], stage_name: str
) -> float | None:
    stage_max = _mapping(
        runtime_trace.get("stage_max_mean_abs_delta_over_contract_range")
    )
    return _float_or_none(stage_max.get(stage_name))


def _runtime_stage_available(
    runtime_trace: Mapping[str, Any],
    stage_name: str,
) -> bool:
    if stage_name == "raw_action":
        upstream = _mapping(runtime_trace.get("upstream_distinction"))
        return any(
            key in upstream
            for key in ("raw_action_distinct", "raw_or_decoded_distinct")
        )
    return _runtime_stage_metric(runtime_trace, stage_name) is not None


def _runtime_stage_distinct(
    runtime_trace: Mapping[str, Any],
    stage_name: str,
) -> bool:
    if stage_name == "raw_action":
        upstream = _mapping(runtime_trace.get("upstream_distinction"))
        return bool(upstream.get("raw_action_distinct", False))
    metric = _runtime_stage_metric(runtime_trace, stage_name)
    return bool(metric is not None and metric > DISTINCTION_EPS)


def _resolve_runtime_terminal_surface(
    runtime_trace: Mapping[str, Any] | None,
) -> tuple[bool, str, str | None]:
    if not isinstance(runtime_trace, Mapping):
        return False, "controller_input", None
    controller_output_available = bool(
        runtime_trace.get("controller_output_available", False)
    )
    terminal_stage_used = _optional_str(runtime_trace.get("terminal_stage_used")) or (
        gr00t_same_checkpoint_triplet_eval.resolve_triplet_terminal_stage(
            controller_output_available=controller_output_available,
        )
    )
    controller_output_unavailable_reason = _optional_str(
        runtime_trace.get("controller_output_unavailable_reason")
    )
    return (
        controller_output_available,
        terminal_stage_used,
        controller_output_unavailable_reason,
    )


def _aggregate_pair_surface(
    action_delta_audit: Mapping[str, Any] | None,
    *,
    terminal_stage: str,
) -> dict[str, Any]:
    raw_pair_summaries = _mapping(
        _mapping(action_delta_audit or {}).get("mode_pair_summaries")
    )
    normalized_pair_summaries = {
        str(pair_name): cast(Mapping[str, object], pair_summary)
        for pair_name, pair_summary in raw_pair_summaries.items()
        if isinstance(pair_summary, Mapping)
    }
    execution_surface_summaries = (
        gr00t_action_chain_telemetry.build_mode_pair_execution_surface_summaries(
            normalized_pair_summaries,
            terminal_stage=terminal_stage,
        )
    )
    available_groups_by_checkpoint = {
        stage_name: set() for stage_name in MACHINE_CHECKPOINT_ORDER[:-1]
    }
    distinct_groups_by_checkpoint = {
        stage_name: set() for stage_name in MACHINE_CHECKPOINT_ORDER[:-1]
    }
    disappearance_groups = {
        "model": set(),
        "decode": set(),
        "relative_to_absolute": set(),
        terminal_stage: set(),
        "survived_to_terminal_stage": set(),
    }
    for pair_name, pair_summary in normalized_pair_summaries.items():
        available_count_by_stage = _mapping(
            pair_summary.get("available_group_count_by_stage")
        )
        difference_groups_by_stage = _mapping(
            pair_summary.get("difference_groups_by_stage")
        )
        for stage_name in MACHINE_CHECKPOINT_ORDER[:-1]:
            if _int_or_zero(available_count_by_stage.get(stage_name)) > 0:
                available_groups_by_checkpoint[stage_name].add(f"pair:{pair_name}")
            raw_groups = difference_groups_by_stage.get(stage_name)
            if isinstance(raw_groups, Sequence) and not isinstance(
                raw_groups, (str, bytes, bytearray)
            ):
                distinct_groups_by_checkpoint[stage_name].update(
                    str(item) for item in raw_groups
                )
    for pair_summary in execution_surface_summaries.values():
        disappearance_by_group = _mapping(
            pair_summary.get("difference_disappearance_groups")
        )
        for stage_name in disappearance_groups:
            raw_groups = disappearance_by_group.get(stage_name)
            if isinstance(raw_groups, Sequence) and not isinstance(
                raw_groups, (str, bytes, bytearray)
            ):
                disappearance_groups[stage_name].update(
                    str(item) for item in raw_groups
                )
    return {
        "available_groups_by_checkpoint": {
            stage_name: _sorted_strings(groups)
            for stage_name, groups in available_groups_by_checkpoint.items()
        },
        "distinct_groups_by_checkpoint": {
            stage_name: _sorted_strings(groups)
            for stage_name, groups in distinct_groups_by_checkpoint.items()
        },
        "difference_disappearance_groups": {
            stage_name: _sorted_strings(groups)
            for stage_name, groups in disappearance_groups.items()
        },
        "pair_execution_surface_summaries": execution_surface_summaries,
    }


def _group_zeroing_suspected(group_payload: Mapping[str, Any]) -> bool:
    zero_motion_flags = _mapping(group_payload.get("zero_motion_flags"))
    if bool(zero_motion_flags.get("all_zero_in_both", False)):
        return True
    stages = _mapping(group_payload.get("stages"))
    controller = _mapping(stages.get("controller_input"))
    baseline = _mapping(controller.get("baseline"))
    probe = _mapping(controller.get("probe"))
    zero_rates = [
        rate
        for rate in (
            _float_or_none(baseline.get("zero_output_rate")),
            _float_or_none(probe.get("zero_output_rate")),
        )
        if rate is not None
    ]
    return bool(zero_rates and max(zero_rates) >= ZERO_OUTPUT_RATE_MIN)


def _aggregate_action_telemetry(
    action_telemetry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    per_group_stats = _mapping(_mapping(action_telemetry or {}).get("per_group_stats"))
    available_groups_by_checkpoint = {
        stage_name: set() for stage_name in MACHINE_CHECKPOINT_ORDER[:-1]
    }
    distinct_groups_by_checkpoint = {
        stage_name: set() for stage_name in MACHINE_CHECKPOINT_ORDER[:-1]
    }
    disappearance_groups = {
        "model": set(),
        "decode": set(),
        "relative_to_absolute": set(),
        "controller_input": set(),
    }
    controller_absorbed_groups: set[str] = set()
    clip_or_saturation_groups: set[str] = set()
    zeroing_groups: set[str] = set()
    model_insensitive_groups: set[str] = set()
    for group_name, raw_group_payload in per_group_stats.items():
        if not isinstance(raw_group_payload, Mapping):
            continue
        group_payload = cast(Mapping[str, Any], raw_group_payload)
        diff_metrics = _mapping(group_payload.get("difference_metrics"))
        stage_to_metric_key = {
            "raw_action": "raw_action_l2",
            "decoded_action": "decoded_action_l2",
            "absolute_action": "absolute_action_l2",
            "controller_input": "controller_input_l2",
        }
        for stage_name, metric_key in stage_to_metric_key.items():
            metric = _float_or_none(diff_metrics.get(metric_key))
            if metric is None:
                continue
            available_groups_by_checkpoint[stage_name].add(str(group_name))
            if metric > DISTINCTION_EPS:
                distinct_groups_by_checkpoint[stage_name].add(str(group_name))
        disappearance_stage = _optional_str(
            diff_metrics.get("difference_disappeared_at")
        )
        if disappearance_stage in disappearance_groups:
            disappearance_groups[disappearance_stage].add(str(group_name))
        if bool(diff_metrics.get("controller_absorbed_upstream_difference", False)):
            controller_absorbed_groups.add(str(group_name))
        if bool(diff_metrics.get("model_insensitive", False)):
            model_insensitive_groups.add(str(group_name))
        clip_rate = _mapping(group_payload.get("clip_rate"))
        saturation_rate = _float_or_none(group_payload.get("saturation_rate")) or 0.0
        if (
            (_float_or_none(clip_rate.get("decoded_action")) or 0.0) > DISTINCTION_EPS
            or (_float_or_none(clip_rate.get("controller_input")) or 0.0)
            > DISTINCTION_EPS
            or saturation_rate > DISTINCTION_EPS
        ):
            clip_or_saturation_groups.add(str(group_name))
        if _group_zeroing_suspected(group_payload):
            zeroing_groups.add(str(group_name))
    return {
        "available_groups_by_checkpoint": {
            stage_name: _sorted_strings(groups)
            for stage_name, groups in available_groups_by_checkpoint.items()
        },
        "distinct_groups_by_checkpoint": {
            stage_name: _sorted_strings(groups)
            for stage_name, groups in distinct_groups_by_checkpoint.items()
        },
        "difference_disappearance_groups": {
            stage_name: _sorted_strings(groups)
            for stage_name, groups in disappearance_groups.items()
        },
        "controller_absorbed_groups": _sorted_strings(controller_absorbed_groups),
        "clip_or_saturation_groups": _sorted_strings(clip_or_saturation_groups),
        "zeroing_groups": _sorted_strings(zeroing_groups),
        "model_insensitive_groups": _sorted_strings(model_insensitive_groups),
    }


def _resolve_eval_saved_telemetry_surface(
    eval_summary: Mapping[str, Any] | None,
) -> dict[str, object] | None:
    if not isinstance(eval_summary, Mapping):
        return None
    module = importlib.import_module("work.recap.scripts.3D_recap_eval")
    builder = getattr(module, "build_saved_telemetry_surface")
    return cast(dict[str, object], builder(eval_summary))


def _combine_checkpoint_surface(
    *,
    runtime_trace: Mapping[str, Any] | None,
    pair_surface: Mapping[str, Any],
    action_telemetry_surface: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    checkpoint_surface: dict[str, dict[str, object]] = {}
    for stage_name in MACHINE_CHECKPOINT_ORDER:
        source_labels: list[str] = []
        available = False
        distinction_present = False
        if isinstance(runtime_trace, Mapping):
            if _runtime_stage_available(runtime_trace, stage_name):
                available = True
                source_labels.append("runtime_trace")
            if _runtime_stage_distinct(runtime_trace, stage_name):
                available = True
                distinction_present = True
        if stage_name != "controller_output":
            pair_available_groups = _mapping(
                pair_surface.get("available_groups_by_checkpoint")
            )
            pair_distinct_groups = _mapping(
                pair_surface.get("distinct_groups_by_checkpoint")
            )
            telemetry_available_groups = _mapping(
                action_telemetry_surface.get("available_groups_by_checkpoint")
            )
            telemetry_distinct_groups = _mapping(
                action_telemetry_surface.get("distinct_groups_by_checkpoint")
            )
            pair_groups = pair_available_groups.get(stage_name)
            if isinstance(pair_groups, list) and pair_groups:
                available = True
                source_labels.append("pair_action_chain")
            pair_distinct = pair_distinct_groups.get(stage_name)
            if isinstance(pair_distinct, list) and pair_distinct:
                available = True
                distinction_present = True
            telemetry_groups = telemetry_available_groups.get(stage_name)
            if isinstance(telemetry_groups, list) and telemetry_groups:
                available = True
                source_labels.append("action_telemetry")
            telemetry_distinct = telemetry_distinct_groups.get(stage_name)
            if isinstance(telemetry_distinct, list) and telemetry_distinct:
                available = True
                distinction_present = True
        checkpoint_surface[stage_name] = {
            "available": bool(available),
            "distinction_present": bool(distinction_present),
            "sources": sorted(set(source_labels)),
        }
    return checkpoint_surface


def _controller_root_cause_summary(
    action_absorption_audit: Mapping[str, Any] | None,
) -> dict[str, object]:
    payload = _mapping(action_absorption_audit)
    summary = _mapping(payload.get("summary"))
    strongest = _optional_str(summary.get("strongest_suspected_cause"))
    absorbed_dimensions = summary.get("absorbed_dimensions_union")
    return {
        "audit_status": _optional_str(payload.get("audit_status")),
        "strongest_suspected_cause": strongest,
        "absorbed_dimensions_union": [
            str(item)
            for item in absorbed_dimensions
            if isinstance(absorbed_dimensions, Sequence)
            and not isinstance(absorbed_dimensions, (str, bytes, bytearray))
        ]
        if isinstance(absorbed_dimensions, Sequence)
        and not isinstance(absorbed_dimensions, (str, bytes, bytearray))
        else [],
        "root_cause_counts": dict(summary.get("root_cause_counts", {}))
        if isinstance(summary.get("root_cause_counts"), Mapping)
        else {},
    }


def build_execution_surface_audit(
    *,
    triplet_summary: Mapping[str, Any] | None = None,
    runtime_trace: Mapping[str, Any] | None = None,
    action_telemetry: Mapping[str, Any] | None = None,
    action_absorption_audit: Mapping[str, Any] | None = None,
    eval_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    summary_payload = _mapping(triplet_summary)
    resolved_runtime_trace = (
        _mapping(summary_payload.get("runtime_trace"))
        if runtime_trace is None
        else _mapping(runtime_trace)
    )
    action_delta_audit = _mapping(summary_payload.get("action_delta_audit"))
    (
        controller_output_available,
        terminal_stage_used,
        controller_output_unavailable_reason,
    ) = _resolve_runtime_terminal_surface(resolved_runtime_trace)
    if not controller_output_available and terminal_stage_used != "controller_input":
        issues.append(
            _issue(
                "terminal_stage_drift",
                "$.terminal_stage_used",
                "missing controller_output must fail soft to controller_input rather than blocking or inventing a later stage",
            )
        )
        terminal_stage_used = "controller_input"
    pair_surface = _aggregate_pair_surface(
        action_delta_audit,
        terminal_stage=terminal_stage_used,
    )
    action_telemetry_surface = _aggregate_action_telemetry(action_telemetry)
    checkpoint_surface = _combine_checkpoint_surface(
        runtime_trace=resolved_runtime_trace,
        pair_surface=pair_surface,
        action_telemetry_surface=action_telemetry_surface,
    )
    core_checkpoints = ("decoded_action", "absolute_action", "controller_input")
    usable_core_trace = any(
        bool(
            cast(Mapping[str, object], checkpoint_surface[stage_name]).get(
                "available", False
            )
        )
        for stage_name in core_checkpoints
    )
    raw_action_distinct = bool(checkpoint_surface["raw_action"]["distinction_present"])
    decoded_distinct = bool(checkpoint_surface["decoded_action"]["distinction_present"])
    absolute_distinct = bool(
        checkpoint_surface["absolute_action"]["distinction_present"]
    )
    controller_input_distinct = bool(
        checkpoint_surface["controller_input"]["distinction_present"]
    )
    controller_output_distinct = bool(
        checkpoint_surface["controller_output"]["distinction_present"]
    )
    terminal_distinct = bool(
        checkpoint_surface[terminal_stage_used]["distinction_present"]
    )
    terminal_available = bool(checkpoint_surface[terminal_stage_used]["available"])

    pair_disappearance = _mapping(pair_surface.get("difference_disappearance_groups"))
    telemetry_disappearance = _mapping(
        action_telemetry_surface.get("difference_disappearance_groups")
    )
    controller_absorbed_groups = set(
        str(item)
        for item in cast(
            Sequence[object],
            action_telemetry_surface.get("controller_absorbed_groups", []),
        )
    )
    clip_or_saturation_groups = set(
        str(item)
        for item in cast(
            Sequence[object],
            action_telemetry_surface.get("clip_or_saturation_groups", []),
        )
    )
    zeroing_groups = set(
        str(item)
        for item in cast(
            Sequence[object], action_telemetry_surface.get("zeroing_groups", [])
        )
    )
    postprocess_groups = set(
        str(item)
        for mapping_payload in (pair_disappearance, telemetry_disappearance)
        for item in cast(
            Sequence[object], mapping_payload.get("relative_to_absolute", [])
        )
    )
    policy_groups = set(
        str(item)
        for mapping_payload in (pair_disappearance, telemetry_disappearance)
        for key in ("model", "decode")
        for item in cast(Sequence[object], mapping_payload.get(key, []))
    )
    controller_terminal_groups = set(
        str(item)
        for item in cast(
            Sequence[object], pair_disappearance.get(terminal_stage_used, [])
        )
    ) | set(
        str(item)
        for item in cast(
            Sequence[object], telemetry_disappearance.get("controller_input", [])
        )
    )

    controller_root_cause = _controller_root_cause_summary(action_absorption_audit)
    strongest_cause = _optional_str(
        controller_root_cause.get("strongest_suspected_cause")
    )
    absorbed_dimensions_union = set(
        str(item)
        for item in cast(
            Sequence[object], controller_root_cause.get("absorbed_dimensions_union", [])
        )
    )
    if strongest_cause == "relative_to_absolute_scaling":
        postprocess_groups.update(absorbed_dimensions_union)
    if strongest_cause in {"clip_or_saturation", "controller_zeroing_or_masking"}:
        controller_terminal_groups.update(absorbed_dimensions_union)

    explicit_controller_signal = bool(
        controller_terminal_groups
        or controller_absorbed_groups
        or clip_or_saturation_groups
        or zeroing_groups
        or strongest_cause in {"clip_or_saturation", "controller_zeroing_or_masking"}
        or (absolute_distinct and terminal_available and not terminal_distinct)
    )
    explicit_postprocess_signal = bool(
        postprocess_groups
        or strongest_cause == "relative_to_absolute_scaling"
        or (
            bool(checkpoint_surface["decoded_action"]["available"])
            and decoded_distinct
            and bool(checkpoint_surface["absolute_action"]["available"])
            and not absolute_distinct
        )
    )
    explicit_policy_signal = bool(
        policy_groups
        or (
            bool(checkpoint_surface["decoded_action"]["available"])
            and not decoded_distinct
        )
        or (
            bool(checkpoint_surface["raw_action"]["available"])
            and not raw_action_distinct
            and not bool(checkpoint_surface["decoded_action"]["available"])
        )
    )
    if (
        sum(
            int(value)
            for value in (
                explicit_policy_signal,
                explicit_postprocess_signal,
                explicit_controller_signal,
            )
        )
        > 1
    ):
        issues.append(
            _issue(
                "mixed_stage_signals",
                "$.classification_surface",
                "multiple stage-loss hypotheses were observed; the audit uses the deepest surviving checkpoint precedence rather than pretending the evidence is single-stage",
            )
        )
    if not usable_core_trace:
        verdict = VERDICT_BLOCKED
        reason_code = "usable_action_trace_missing"
        issues.append(
            _issue(
                "usable_action_trace_missing",
                "$.checkpoint_surface",
                "blocked because none of decoded_action / absolute_action / controller_input exposed usable evidence",
            )
        )
    elif terminal_available and terminal_distinct:
        verdict = VERDICT_UNKNOWN
        reason_code = "distinction_survives_terminal_stage"
    elif explicit_controller_signal:
        verdict = VERDICT_CONTROLLER_DISTORTION
        if strongest_cause == "clip_or_saturation" or clip_or_saturation_groups:
            reason_code = "controller_clip_or_saturation"
        elif strongest_cause == "controller_zeroing_or_masking" or zeroing_groups:
            reason_code = "controller_zeroing_or_masking"
        elif terminal_stage_used == "controller_output":
            reason_code = "difference_absorbed_at_controller_output"
        else:
            reason_code = "difference_absorbed_at_controller_input"
    elif explicit_postprocess_signal:
        verdict = VERDICT_POSTPROCESS
        if strongest_cause == "relative_to_absolute_scaling":
            reason_code = "relative_to_absolute_scaling"
        else:
            reason_code = "difference_absorbed_before_absolute_action"
    elif explicit_policy_signal:
        verdict = VERDICT_POLICY
        if (
            bool(checkpoint_surface["raw_action"]["available"])
            and not raw_action_distinct
        ):
            reason_code = "raw_action_distinction_absent"
        else:
            reason_code = "decoded_distinction_absent"
    else:
        verdict = VERDICT_UNKNOWN
        reason_code = "insufficient_surface_evidence"

    eval_saved_telemetry = _resolve_eval_saved_telemetry_surface(eval_summary)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "verdict": verdict,
        "reason_code": reason_code,
        "issues": _dedupe_issues(issues),
        "summary_stage_order": list(
            getattr(
                gr00t_same_checkpoint_triplet_eval,
                "SUMMARY_STAGE_ORDER",
                SUMMARY_STAGE_ORDER,
            )
        ),
        "machine_checkpoint_order": list(
            getattr(
                gr00t_same_checkpoint_triplet_eval,
                "MACHINE_CHECKPOINT_ORDER",
                MACHINE_CHECKPOINT_ORDER,
            )
        ),
        "controller_output_available": bool(controller_output_available),
        "controller_output_unavailable_reason": None
        if controller_output_available
        else controller_output_unavailable_reason,
        "terminal_stage_used": terminal_stage_used,
        "checkpoint_surface": checkpoint_surface,
        "classification_surface": {
            "usable_core_trace": bool(usable_core_trace),
            "raw_action_distinct": bool(raw_action_distinct),
            "decoded_action_distinct": bool(decoded_distinct),
            "absolute_action_distinct": bool(absolute_distinct),
            "controller_input_distinct": bool(controller_input_distinct),
            "controller_output_distinct": bool(controller_output_distinct),
            "terminal_stage_distinct": bool(terminal_distinct),
            "policy_groups": _sorted_strings(policy_groups),
            "postprocess_groups": _sorted_strings(postprocess_groups),
            "controller_terminal_groups": _sorted_strings(controller_terminal_groups),
        },
        "runtime_trace": {
            "status": _optional_str(resolved_runtime_trace.get("status")),
            "normalization_metric": _optional_str(
                resolved_runtime_trace.get("normalization_metric")
            ),
            "stage_max_mean_abs_delta_over_contract_range": dict(
                _mapping(
                    resolved_runtime_trace.get(
                        "stage_max_mean_abs_delta_over_contract_range"
                    )
                )
            ),
            "upstream_distinction": dict(
                _mapping(resolved_runtime_trace.get("upstream_distinction"))
            ),
        },
        "pair_action_chain": pair_surface,
        "action_telemetry": action_telemetry_surface,
        "controller_semantics": {
            "controller_absorbed_groups": _sorted_strings(controller_absorbed_groups),
            "clip_or_saturation_groups": _sorted_strings(clip_or_saturation_groups),
            "zeroing_groups": _sorted_strings(zeroing_groups),
            "action_absorption": controller_root_cause,
        },
        "eval_saved_telemetry": eval_saved_telemetry,
        "backpointer": {
            "writer_script": WRITER_SCRIPT,
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        input_paths = [
            str(args.triplet_summary_json).strip(),
            str(args.runtime_trace_json).strip(),
            str(args.action_telemetry_json).strip(),
            str(args.action_absorption_json).strip(),
            str(args.eval_summary_json).strip(),
        ]
        if not any(input_paths):
            raise ValueError(
                "at least one input surface is required; provide --triplet-summary-json or a compatible standalone JSON input"
            )
        triplet_summary = (
            _read_json(_resolve_path(str(args.triplet_summary_json)))
            if str(args.triplet_summary_json).strip()
            else None
        )
        runtime_trace = (
            _read_json(_resolve_path(str(args.runtime_trace_json)))
            if str(args.runtime_trace_json).strip()
            else None
        )
        action_telemetry = (
            _read_json(_resolve_path(str(args.action_telemetry_json)))
            if str(args.action_telemetry_json).strip()
            else None
        )
        action_absorption_audit = (
            _read_json(_resolve_path(str(args.action_absorption_json)))
            if str(args.action_absorption_json).strip()
            else None
        )
        eval_summary = (
            _read_json(_resolve_path(str(args.eval_summary_json)))
            if str(args.eval_summary_json).strip()
            else None
        )
        report = build_execution_surface_audit(
            triplet_summary=triplet_summary,
            runtime_trace=runtime_trace,
            action_telemetry=action_telemetry,
            action_absorption_audit=action_absorption_audit,
            eval_summary=eval_summary,
        )
        output_path = _validate_output_path(_resolve_path(str(args.out)))
        report["output_path"] = str(output_path)
        report["backpointer"] = {
            **cast(Mapping[str, Any], report["backpointer"]),
            "triplet_summary_json": str(_resolve_path(str(args.triplet_summary_json)))
            if str(args.triplet_summary_json).strip()
            else None,
            "runtime_trace_json": str(_resolve_path(str(args.runtime_trace_json)))
            if str(args.runtime_trace_json).strip()
            else None,
            "action_telemetry_json": str(_resolve_path(str(args.action_telemetry_json)))
            if str(args.action_telemetry_json).strip()
            else None,
            "action_absorption_json": str(
                _resolve_path(str(args.action_absorption_json))
            )
            if str(args.action_absorption_json).strip()
            else None,
            "eval_summary_json": str(_resolve_path(str(args.eval_summary_json)))
            if str(args.eval_summary_json).strip()
            else None,
        }
        _ = _write_json(output_path, report)
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
