from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_carrier_panel_gate


REPORT_SCHEMA_VERSION = "gr00t_action_absorption_root_cause_v1"
REPORT_ARTIFACT_KIND = "gr00t_action_absorption_root_cause"
ACTION_ABSORPTION_ROOT_CAUSE_JSON_NAME = "action_absorption_root_cause.json"
TARGET_STATUS = "consumed_but_absorbed_downstream"
ROOT_CAUSE_ORDER: tuple[str, ...] = (
    "clip_or_saturation",
    "controller_zeroing_or_masking",
    "relative_to_absolute_scaling",
    "absorbed_but_root_cause_unknown",
)
WRITER_SCRIPT = "work/recap/scripts/gr00t_action_absorption_audit.py"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "agent" / "artifacts" / ACTION_ABSORPTION_ROOT_CAUSE_JSON_NAME
)

ABSORBED_DIFF_EPS = 1e-9
CLIP_RATE_EPS = 1e-9
SCALE_ATTENUATION_MAX = 0.5
ZERO_OUTPUT_RATE_MIN = 0.95


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_action_absorption_audit.py",
        description=(
            "Audit why a carrier-panel distinction was consumed but absorbed downstream."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Carrier-panel or telemetry-style JSON payload to audit.",
    )
    _ = parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for the root-cause audit JSON.",
    )
    _ = parser.add_argument(
        "--normalized-threshold",
        type=float,
        default=gr00t_carrier_panel_gate.DEFAULT_NORMALIZED_THRESHOLD,
        help=(
            "Normalized threshold used when a panel-style sample needs its status derived "
            "from runtime_trace."
        ),
    )
    return parser


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _sha256(payload: Mapping[str, Any]) -> str:
    body = json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return cast(dict[str, Any], payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json_text(payload), encoding="utf-8")
    return path


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _list_or_empty(value: object) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if (
        numerator is None
        or denominator is None
        or abs(float(denominator)) <= ABSORBED_DIFF_EPS
    ):
        return None
    return float(numerator / denominator)


def _resolve_runtime_trace(sample: Mapping[str, Any]) -> Mapping[str, Any] | None:
    debug_probe = sample.get("debug_probe")
    if isinstance(debug_probe, Mapping):
        runtime_trace = debug_probe.get("runtime_trace")
        if isinstance(runtime_trace, Mapping):
            return cast(Mapping[str, Any], runtime_trace)
    runtime_trace = sample.get("runtime_trace")
    if isinstance(runtime_trace, Mapping):
        return cast(Mapping[str, Any], runtime_trace)
    return None


def _sample_status(
    sample: Mapping[str, Any],
    *,
    normalized_threshold: float,
) -> str | None:
    status = sample.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip()
    if _resolve_runtime_trace(sample) is None:
        return None
    summary = gr00t_carrier_panel_gate.build_panel_sample_summary(
        sample,
        normalized_threshold=normalized_threshold,
    )
    resolved = summary.get("status")
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    return None


def _iter_samples(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    samples_raw = payload.get("samples")
    if isinstance(samples_raw, list):
        return [cast(Mapping[str, Any], sample) for sample in samples_raw]
    return [payload]


def _extract_per_group_stats(
    sample: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    direct = sample.get("per_group_stats")
    if isinstance(direct, Mapping):
        return cast(Mapping[str, Any], direct)
    telemetry = sample.get("telemetry")
    if isinstance(telemetry, Mapping):
        nested = telemetry.get("per_group_stats")
        if isinstance(nested, Mapping):
            return cast(Mapping[str, Any], nested)
    action_telemetry = sample.get("action_telemetry")
    if isinstance(action_telemetry, Mapping):
        nested = action_telemetry.get("per_group_stats")
        if isinstance(nested, Mapping):
            return cast(Mapping[str, Any], nested)
    top_level = payload.get("per_group_stats")
    if isinstance(top_level, Mapping) and len(_iter_samples(payload)) == 1:
        return cast(Mapping[str, Any], top_level)
    return {}


def _build_stage_surface(
    *,
    value: float | None,
    metric: str,
    source: str,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "available": value is not None,
        "value": value,
        "metric": metric,
        "source": source,
        "unavailable_reason": unavailable_reason if value is None else None,
    }


def _controller_zero_output_rates(
    group_payload: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    stages = _mapping_or_empty(group_payload.get("stages"))
    controller = _mapping_or_empty(stages.get("controller_input"))
    baseline = _mapping_or_empty(controller.get("baseline"))
    probe = _mapping_or_empty(controller.get("probe"))
    return (
        _float_or_none(baseline.get("zero_output_rate")),
        _float_or_none(probe.get("zero_output_rate")),
    )


def _group_absorbed(diff_metrics: Mapping[str, Any]) -> bool:
    difference_disappeared_at = diff_metrics.get("difference_disappeared_at")
    controller_absorbed = bool(
        diff_metrics.get("controller_absorbed_upstream_difference", False)
    )
    absolute = _float_or_none(diff_metrics.get("absolute_action_l2"))
    controller = _float_or_none(diff_metrics.get("controller_input_l2"))
    return bool(
        controller_absorbed
        or difference_disappeared_at in {"relative_to_absolute", "controller_input"}
        or (
            absolute is not None
            and absolute > ABSORBED_DIFF_EPS
            and controller is not None
            and controller <= ABSORBED_DIFF_EPS
        )
    )


def _classify_group_root_cause(
    *,
    absorbed: bool,
    action_representation: str,
    clip_any: bool,
    zeroing_suspected: bool,
    absolute_over_decoded: float | None,
) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if not absorbed:
        return None, reasons
    if clip_any:
        reasons.append(
            "decoded/controller clip_rate or controller saturation_rate stayed above zero while the downstream difference was absorbed"
        )
        return "clip_or_saturation", reasons
    if zeroing_suspected:
        reasons.append(
            "controller_input zero-output evidence suggests downstream zeroing or masking swallowed the consumed difference"
        )
        return "controller_zeroing_or_masking", reasons
    if (
        action_representation.upper() == "RELATIVE"
        and absolute_over_decoded is not None
        and absolute_over_decoded <= SCALE_ATTENUATION_MAX
    ):
        reasons.append(
            "relative action delta shrank materially at the absolute-action seam before it could survive downstream"
        )
        return "relative_to_absolute_scaling", reasons
    reasons.append(
        "the sample was absorbed downstream, but the available panel/telemetry evidence does not isolate a stronger mechanism"
    )
    return "absorbed_but_root_cause_unknown", reasons


def _build_group_evidence(
    *,
    group_name: str,
    group_payload: Mapping[str, Any],
) -> dict[str, Any]:
    diff_metrics = _mapping_or_empty(group_payload.get("difference_metrics"))
    clip_rate = _mapping_or_empty(group_payload.get("clip_rate"))
    zero_motion_flags = _mapping_or_empty(group_payload.get("zero_motion_flags"))
    action_representation = str(
        group_payload.get("action_representation", "UNKNOWN")
    ).upper()

    raw_delta = _float_or_none(diff_metrics.get("raw_action_l2"))
    decoded_delta = _float_or_none(diff_metrics.get("decoded_action_l2"))
    absolute_delta = _float_or_none(diff_metrics.get("absolute_action_l2"))
    controller_delta = _float_or_none(diff_metrics.get("controller_input_l2"))
    decoded_clip_rate = _float_or_none(clip_rate.get("decoded_action")) or 0.0
    controller_clip_rate = _float_or_none(clip_rate.get("controller_input")) or 0.0
    saturation_rate = _float_or_none(group_payload.get("saturation_rate")) or 0.0
    baseline_zero_output_rate, probe_zero_output_rate = _controller_zero_output_rates(
        group_payload
    )
    max_zero_output_rate = max(
        [
            value
            for value in (baseline_zero_output_rate, probe_zero_output_rate)
            if value is not None
        ],
        default=None,
    )
    clip_any = bool(
        decoded_clip_rate > CLIP_RATE_EPS
        or controller_clip_rate > CLIP_RATE_EPS
        or saturation_rate > CLIP_RATE_EPS
    )
    zeroing_suspected = bool(
        zero_motion_flags.get("all_zero_in_both", False)
        or (
            max_zero_output_rate is not None
            and max_zero_output_rate >= ZERO_OUTPUT_RATE_MIN
        )
    )
    absolute_over_decoded = _safe_ratio(absolute_delta, decoded_delta)
    controller_over_absolute = _safe_ratio(controller_delta, absolute_delta)
    absorbed = _group_absorbed(diff_metrics)
    root_cause, reasons = _classify_group_root_cause(
        absorbed=absorbed,
        action_representation=action_representation,
        clip_any=clip_any,
        zeroing_suspected=zeroing_suspected,
        absolute_over_decoded=absolute_over_decoded,
    )
    return {
        "group": group_name,
        "action_representation": action_representation,
        "reference_state_key": group_payload.get("reference_state_key"),
        "difference_disappeared_at": diff_metrics.get("difference_disappeared_at"),
        "controller_absorbed_upstream_difference": bool(
            diff_metrics.get("controller_absorbed_upstream_difference", False)
        ),
        "absorbed": absorbed,
        "suspected_root_cause": root_cause,
        "root_cause_reasons": reasons,
        "delta_surfaces": {
            "raw": _build_stage_surface(
                value=raw_delta,
                metric="l2",
                source=f"per_group_stats.{group_name}.difference_metrics.raw_action_l2",
            ),
            "decoded": _build_stage_surface(
                value=decoded_delta,
                metric="l2",
                source=f"per_group_stats.{group_name}.difference_metrics.decoded_action_l2",
            ),
            "absolute": _build_stage_surface(
                value=absolute_delta,
                metric="l2",
                source=f"per_group_stats.{group_name}.difference_metrics.absolute_action_l2",
            ),
            "controller": _build_stage_surface(
                value=controller_delta,
                metric="l2",
                source=f"per_group_stats.{group_name}.difference_metrics.controller_input_l2",
            ),
        },
        "clip_or_saturation_flags": {
            "any_clip_or_saturation": clip_any,
            "decoded_action_clip_rate": float(decoded_clip_rate),
            "controller_input_clip_rate": float(controller_clip_rate),
            "saturation_rate": float(saturation_rate),
        },
        "relative_to_absolute_scale_factors": {
            "absolute_over_decoded": absolute_over_decoded,
            "controller_over_absolute": controller_over_absolute,
        },
        "controller_zeroing_or_masking": {
            "suspected": zeroing_suspected,
            "baseline_zero_output_rate": baseline_zero_output_rate,
            "probe_zero_output_rate": probe_zero_output_rate,
            "max_zero_output_rate": max_zero_output_rate,
            "all_zero_in_both": bool(zero_motion_flags.get("all_zero_in_both", False)),
        },
    }


def _group_stage_max(
    group_evidence: Sequence[Mapping[str, Any]],
    *,
    stage_name: str,
) -> float | None:
    values = [
        _float_or_none(
            _mapping_or_empty(
                _mapping_or_empty(group.get("delta_surfaces")).get(stage_name)
            ).get("value")
        )
        for group in group_evidence
    ]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(max(filtered))


def _sample_stage_surfaces(
    *,
    runtime_trace: Mapping[str, Any] | None,
    group_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage_max = (
        _mapping_or_empty(
            runtime_trace.get("stage_max_mean_abs_delta_over_contract_range")
        )
        if runtime_trace is not None
        else {}
    )
    controller_output = _float_or_none(stage_max.get("controller_output"))
    controller_input = _float_or_none(stage_max.get("controller_input"))
    controller_value = (
        controller_output if controller_output is not None else controller_input
    )
    controller_stage_used = (
        "controller_output" if controller_output is not None else "controller_input"
    )
    return {
        "raw": _build_stage_surface(
            value=_group_stage_max(group_evidence, stage_name="raw"),
            metric="l2",
            source="per_group_stats.*.difference_metrics.raw_action_l2",
            unavailable_reason="panel runtime_trace does not expose a raw-action numeric delta surface",
        ),
        "decoded": _build_stage_surface(
            value=_float_or_none(stage_max.get("decoded_action"))
            or _group_stage_max(group_evidence, stage_name="decoded"),
            metric=(
                "mean_abs_delta_over_contract_range"
                if _float_or_none(stage_max.get("decoded_action")) is not None
                else "l2"
            ),
            source=(
                "runtime_trace.stage_max_mean_abs_delta_over_contract_range.decoded_action"
                if _float_or_none(stage_max.get("decoded_action")) is not None
                else "per_group_stats.*.difference_metrics.decoded_action_l2"
            ),
        ),
        "absolute": _build_stage_surface(
            value=_float_or_none(stage_max.get("absolute_action"))
            or _group_stage_max(group_evidence, stage_name="absolute"),
            metric=(
                "mean_abs_delta_over_contract_range"
                if _float_or_none(stage_max.get("absolute_action")) is not None
                else "l2"
            ),
            source=(
                "runtime_trace.stage_max_mean_abs_delta_over_contract_range.absolute_action"
                if _float_or_none(stage_max.get("absolute_action")) is not None
                else "per_group_stats.*.difference_metrics.absolute_action_l2"
            ),
        ),
        "controller": {
            **_build_stage_surface(
                value=controller_value
                or _group_stage_max(group_evidence, stage_name="controller"),
                metric=(
                    "mean_abs_delta_over_contract_range"
                    if controller_value is not None
                    else "l2"
                ),
                source=(
                    f"runtime_trace.stage_max_mean_abs_delta_over_contract_range.{controller_stage_used}"
                    if controller_value is not None
                    else "per_group_stats.*.difference_metrics.controller_input_l2"
                ),
            ),
            "stage_used": controller_stage_used,
        },
    }


def _sample_root_cause_summary(
    group_evidence: Sequence[Mapping[str, Any]],
    *,
    runtime_trace: Mapping[str, Any] | None,
) -> tuple[str, dict[str, int], list[str]]:
    eligible_causes = [
        str(group.get("suspected_root_cause"))
        for group in group_evidence
        if group.get("absorbed") and isinstance(group.get("suspected_root_cause"), str)
    ]
    if eligible_causes:
        counts = Counter(eligible_causes)
        strongest = sorted(
            counts,
            key=lambda cause: (-int(counts[cause]), ROOT_CAUSE_ORDER.index(cause)),
        )[0]
        absorbed_dimensions = [
            str(group.get("group"))
            for group in group_evidence
            if bool(group.get("absorbed"))
        ]
        return (
            strongest,
            {cause: int(counts.get(cause, 0)) for cause in ROOT_CAUSE_ORDER},
            absorbed_dimensions,
        )

    stage_max = (
        _mapping_or_empty(
            runtime_trace.get("stage_max_mean_abs_delta_over_contract_range")
        )
        if runtime_trace is not None
        else {}
    )
    decoded = _float_or_none(stage_max.get("decoded_action"))
    absolute = _float_or_none(stage_max.get("absolute_action"))
    if (
        decoded is not None
        and decoded > ABSORBED_DIFF_EPS
        and absolute is not None
        and _safe_ratio(absolute, decoded) is not None
        and cast(float, _safe_ratio(absolute, decoded)) <= SCALE_ATTENUATION_MAX
    ):
        counts = {cause: 0 for cause in ROOT_CAUSE_ORDER}
        counts["relative_to_absolute_scaling"] = 1
        return "relative_to_absolute_scaling", counts, []
    counts = {cause: 0 for cause in ROOT_CAUSE_ORDER}
    counts["absorbed_but_root_cause_unknown"] = 1
    return "absorbed_but_root_cause_unknown", counts, []


def _build_sample_audit(
    sample: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    normalized_threshold: float,
) -> dict[str, Any]:
    sample_status = _sample_status(sample, normalized_threshold=normalized_threshold)
    runtime_trace = _resolve_runtime_trace(sample)
    group_stats = _extract_per_group_stats(sample, payload=payload)
    group_evidence = [
        _build_group_evidence(
            group_name=str(group_name),
            group_payload=cast(Mapping[str, Any], group_payload),
        )
        for group_name, group_payload in group_stats.items()
        if isinstance(group_payload, Mapping)
    ]
    strongest_cause, root_cause_counts, absorbed_dimensions = (
        _sample_root_cause_summary(
            group_evidence,
            runtime_trace=runtime_trace,
        )
    )
    clip_groups = [
        str(group["group"])
        for group in group_evidence
        if bool(
            _mapping_or_empty(group.get("clip_or_saturation_flags")).get(
                "any_clip_or_saturation", False
            )
        )
    ]
    masked_groups = [
        str(group["group"])
        for group in group_evidence
        if bool(
            _mapping_or_empty(group.get("controller_zeroing_or_masking")).get(
                "suspected", False
            )
        )
    ]
    per_group_scale_factors = {
        str(group["group"]): dict(
            _mapping_or_empty(group.get("relative_to_absolute_scale_factors"))
        )
        for group in group_evidence
    }
    delta_surfaces = _sample_stage_surfaces(
        runtime_trace=runtime_trace,
        group_evidence=group_evidence,
    )
    return {
        "sample_id": str(sample.get("sample_id", "")),
        "panel_slot_index": sample.get("panel_slot_index"),
        "panel_slot_name": sample.get("panel_slot_name"),
        "episode_index": sample.get("episode_index"),
        "outer_step": sample.get("outer_step"),
        "input_status": sample_status,
        "eligible_for_root_cause_audit": bool(sample_status == TARGET_STATUS),
        "delta_surfaces": delta_surfaces,
        "clip_or_saturation_flags": {
            "any_clip_or_saturation": bool(clip_groups),
            "clip_or_saturation_groups": clip_groups,
            "max_decoded_action_clip_rate": max(
                [
                    float(
                        _mapping_or_empty(group.get("clip_or_saturation_flags")).get(
                            "decoded_action_clip_rate",
                            0.0,
                        )
                    )
                    for group in group_evidence
                ],
                default=0.0,
            ),
            "max_controller_input_clip_rate": max(
                [
                    float(
                        _mapping_or_empty(group.get("clip_or_saturation_flags")).get(
                            "controller_input_clip_rate",
                            0.0,
                        )
                    )
                    for group in group_evidence
                ],
                default=0.0,
            ),
            "max_saturation_rate": max(
                [
                    float(
                        _mapping_or_empty(group.get("clip_or_saturation_flags")).get(
                            "saturation_rate",
                            0.0,
                        )
                    )
                    for group in group_evidence
                ],
                default=0.0,
            ),
        },
        "relative_to_absolute_scale_factors": {
            "sample_level": {
                "absolute_over_decoded": _safe_ratio(
                    _float_or_none(
                        _mapping_or_empty(delta_surfaces.get("absolute")).get("value")
                    ),
                    _float_or_none(
                        _mapping_or_empty(delta_surfaces.get("decoded")).get("value")
                    ),
                ),
                "controller_over_absolute": _safe_ratio(
                    _float_or_none(
                        _mapping_or_empty(delta_surfaces.get("controller")).get("value")
                    ),
                    _float_or_none(
                        _mapping_or_empty(delta_surfaces.get("absolute")).get("value")
                    ),
                ),
            },
            "by_group": per_group_scale_factors,
        },
        "controller_zeroing_or_masking": {
            "suspected": bool(masked_groups),
            "masked_groups": masked_groups,
            "max_zero_output_rate": max(
                [
                    float(
                        _mapping_or_empty(
                            group.get("controller_zeroing_or_masking")
                        ).get(
                            "max_zero_output_rate",
                            0.0,
                        )
                        or 0.0
                    )
                    for group in group_evidence
                ],
                default=0.0,
            ),
        },
        "absorbed_dimensions": absorbed_dimensions,
        "group_evidence": group_evidence,
        "root_cause_counts": root_cause_counts,
        "strongest_suspected_root_cause": strongest_cause,
    }


def build_action_absorption_audit(
    payload: Mapping[str, Any],
    *,
    normalized_threshold: float = gr00t_carrier_panel_gate.DEFAULT_NORMALIZED_THRESHOLD,
) -> dict[str, Any]:
    top_level_status = payload.get("status")
    input_status = top_level_status if isinstance(top_level_status, str) else None
    if input_status is None and len(_iter_samples(payload)) == 1:
        input_status = _sample_status(
            _iter_samples(payload)[0],
            normalized_threshold=normalized_threshold,
        )

    if input_status is None:
        audit_status = "blocked_missing_status"
        per_sample_audit: list[dict[str, Any]] = []
    elif input_status != TARGET_STATUS:
        audit_status = "skipped_non_absorbed_status"
        per_sample_audit = []
    else:
        audit_status = "ready"
        per_sample_audit = [
            _build_sample_audit(
                sample,
                payload=payload,
                normalized_threshold=normalized_threshold,
            )
            for sample in _iter_samples(payload)
            if _sample_status(sample, normalized_threshold=normalized_threshold)
            == TARGET_STATUS
        ]

    strongest_suspected_cause: str | None = None
    summary_counts = {cause: 0 for cause in ROOT_CAUSE_ORDER}
    if per_sample_audit:
        strongest_by_sample = [
            str(sample["strongest_suspected_root_cause"])
            for sample in per_sample_audit
            if isinstance(sample.get("strongest_suspected_root_cause"), str)
        ]
        counts = Counter(strongest_by_sample)
        summary_counts = {
            cause: int(counts.get(cause, 0)) for cause in ROOT_CAUSE_ORDER
        }
        if counts:
            strongest_suspected_cause = sorted(
                counts,
                key=lambda cause: (-int(counts[cause]), ROOT_CAUSE_ORDER.index(cause)),
            )[0]

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "input_status": input_status,
        "target_status": TARGET_STATUS,
        "audit_status": audit_status,
        "eligible_for_root_cause_audit": bool(audit_status == "ready"),
        "root_cause_enum": list(ROOT_CAUSE_ORDER),
        "default_output_filename": ACTION_ABSORPTION_ROOT_CAUSE_JSON_NAME,
        "per_sample_audit": per_sample_audit,
        "summary": {
            "eligible_sample_count": int(len(per_sample_audit)),
            "root_cause_counts": summary_counts,
            "strongest_suspected_cause": strongest_suspected_cause,
            "absorbed_dimensions_union": sorted(
                {
                    str(group)
                    for sample in per_sample_audit
                    for group in _list_or_empty(sample.get("absorbed_dimensions"))
                }
            ),
        },
        "backpointer": {
            "writer_script": WRITER_SCRIPT,
        },
    }
    if audit_status != "ready":
        if input_status is None:
            report["skip_reason"] = (
                "action absorption audit requires an explicit status or a panel sample runtime_trace that can derive one"
            )
        else:
            report["skip_reason"] = (
                f"action absorption audit only runs for status={TARGET_STATUS}, got {input_status}"
            )
    report["report_signature_sha256"] = _sha256(
        {
            key: value
            for key, value in report.items()
            if key != "report_signature_sha256"
        }
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        input_path = _resolve_path(str(args.input_json))
        output_path = _resolve_path(str(args.out))
        report = build_action_absorption_audit(
            _read_json(input_path),
            normalized_threshold=float(args.normalized_threshold),
        )
        report["output_path"] = str(output_path)
        report["backpointer"] = {
            **cast(Mapping[str, Any], report["backpointer"]),
            "input_json": str(input_path),
        }
        report["report_signature_sha256"] = _sha256(
            {
                key: value
                for key, value in report.items()
                if key != "report_signature_sha256"
            }
        )
        _ = _write_json(output_path, report)
        print(_canonical_json_text(report), end="")
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
