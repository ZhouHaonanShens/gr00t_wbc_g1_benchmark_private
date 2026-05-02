#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any, Protocol, cast


sys.dont_write_bytecode = True


DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/same_checkpoint_triplet"
)
DEFAULT_SUMMARY_JSON_NAME = "same_checkpoint_triplet_eval.json"
MODE_ARTIFACT_BASENAME_TEMPLATE = "same_checkpoint_triplet_{indicator_mode}.json"
DEFAULT_VARIANT = "same_checkpoint_triplet_eval"

REPORT_SCHEMA_VERSION = "gr00t_same_checkpoint_triplet_eval_v1"
REPORT_ARTIFACT_KIND = "gr00t_same_checkpoint_triplet_eval"
MODE_SCHEMA_VERSION = "gr00t_same_checkpoint_triplet_mode_surface_v1"
MODE_ARTIFACT_KIND = "gr00t_same_checkpoint_triplet_mode_surface"
FAILURE_NOTE_MARKDOWN_NAME = "same_checkpoint_triplet_failure_note.md"
DEBUG_PROBE_GATE_ROLE = "debug_probe"
NORMALIZED_DELTA_METRIC_NAME = "mean_abs_delta_over_contract_range"
CONTROLLER_OUTPUT_UNAVAILABLE_REASON = "no live controller_output stage currently exists in work/**; the runtime carrier gate remains grounded on controller_input until a minimal correct controller_output seam exists"
SUMMARY_STAGE_ORDER: tuple[str, ...] = (
    "preprocess",
    "predict",
    "postprocess",
    "execute",
)
MACHINE_CHECKPOINT_ORDER: tuple[str, ...] = (
    "raw_action",
    "decoded_action",
    "absolute_action",
    "controller_input",
    "controller_output",
)

TRIPLET_GATE_NAME = "GR00TSameCheckpointTripletBindingGate"
OK_REASON_CODE = "ok"
BLOCKED_REASON_CODE = "triplet_binding_blocked"

WRITER_SCRIPT = "work/recap/scripts/gr00t_same_checkpoint_triplet_eval.py"

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap import runtime_prompt
from work.recap import policy as recap_policy
from work.recap import run_manifest as recap_run_manifest
from work.recap import state_conditioned_bucket_a_import
from work.recap.scripts import gr00t_action_chain_telemetry
from work.recap.scripts import gr00t_checkpoint_provenance_gate
from work.recap.scripts import gr00t_eval_contract_gate


CANONICAL_TRIPLET_MODES = tuple(recap_policy.MAINLINE_RUNTIME_INDICATOR_MODES)
DEFAULT_TRIPLET_BRANCH = gr00t_action_chain_telemetry.BRANCH_UNITREE_G1


class ModeSurfaceExecutor(Protocol):
    def __call__(
        self,
        *,
        indicator_mode: str,
        prompt_text: str,
        policy_options: Mapping[str, object],
    ) -> Mapping[str, object]: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_same_checkpoint_triplet_eval.py",
        description=(
            "Build a same-checkpoint omit/positive/negative triplet artifact pack "
            "that keeps checkpoint and observation-seed surfaces fixed while making "
            "indicator_mode the primary comparison variable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--run-manifest-json",
        type=str,
        default="",
        help=(
            "Existing run manifest JSON carrying the fail-closed checkpoint/controller binding. "
            "When omitted, the triplet harness blocks instead of writing valid mode artifacts."
        ),
    )
    _ = parser.add_argument(
        "--checkpoint-loaded",
        type=str,
        default="",
        help=(
            "Checkpoint or server-loaded model reference. When omitted and --policy-server-live "
            "is used, the script will try to recover it from server provenance."
        ),
    )
    _ = parser.add_argument(
        "--prompt-raw",
        type=str,
        required=True,
        help="Canonical prompt_raw text shared by the omit/positive/negative triplet.",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where per-mode artifacts and the summary JSON will be written.",
    )
    _ = parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional explicit summary JSON path. Defaults to <output-dir>/same_checkpoint_triplet_eval.json.",
    )
    _ = parser.add_argument(
        "--observation-seed",
        type=int,
        default=0,
        help="Seed identifying the fixed observation surface used for all three modes.",
    )
    _ = parser.add_argument(
        "--observation-json",
        type=str,
        default="",
        help=(
            "Optional observation fixture JSON. Its content is hashed into the summary and can also be "
            "used for --policy-server-live mode."
        ),
    )
    _ = parser.add_argument(
        "--mode-surface-json",
        type=str,
        default="",
        help=(
            "Optional precomputed per-mode action-surface JSON keyed by omit/positive/negative. "
            "Useful when normalizing captured runtime evidence into the stable triplet schema."
        ),
    )
    _ = parser.add_argument(
        "--variant",
        type=str,
        default=DEFAULT_VARIANT,
        help="Variant label passed to runtime prompt helpers when building per-mode prompt bundles.",
    )
    _ = parser.add_argument(
        "--mode-sequence",
        nargs="+",
        default=list(CANONICAL_TRIPLET_MODES),
        help="Explicit indicator_mode order; must contain omit positive negative exactly once each.",
    )
    _ = parser.add_argument(
        "--policy-server-live",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Collect decoded action evidence live from a GR00T policy server using the same observation JSON "
            "and per-mode options['indicator_mode'] / options['seed']."
        ),
    )
    _ = parser.add_argument("--server-host", type=str, default="127.0.0.1")
    _ = parser.add_argument("--server-port", type=int, default=5555)
    _ = parser.add_argument("--server-timeout-ms", type=int, default=2000)
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return cast(dict[str, Any], dict(payload))


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return cast(Path, state_conditioned_bucket_a_import.validate_output_dir(path))


def _issue(code: str, field_path: str, message: str) -> dict[str, str]:
    return {
        "code": str(code),
        "field_path": str(field_path),
        "message": str(message),
    }


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(cast(Any, value).tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable(cast(Any, value).item())
        except Exception:
            pass
    return str(value)


def _round_float(value: float, *, digits: int = 8) -> float:
    return float(round(float(value), digits))


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int_or_default(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return int(default)
    return int(value)


def _dedupe_issue_list(
    issues: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        code = str(issue.get("code", "unknown"))
        field_path = str(issue.get("field_path", "$"))
        message = str(issue.get("message", ""))
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


def _normalize_triplet_modes(
    mode_sequence: Sequence[object] | None = None,
) -> tuple[str, ...]:
    raw_sequence = (
        list(CANONICAL_TRIPLET_MODES)
        if mode_sequence is None
        else [item for item in mode_sequence]
    )
    normalized = [
        recap_policy.validate_mainline_runtime_indicator_mode(
            item,
            field_name=f"mode_sequence[{index}]",
        )
        for index, item in enumerate(raw_sequence)
    ]
    if len(normalized) != len(CANONICAL_TRIPLET_MODES):
        expected = " ".join(CANONICAL_TRIPLET_MODES)
        raise ValueError(
            "same-checkpoint triplet requires exactly three indicator modes; expected "
            + expected
        )
    if len(set(normalized)) != len(CANONICAL_TRIPLET_MODES):
        expected = " ".join(CANONICAL_TRIPLET_MODES)
        raise ValueError(
            "mode collapse detected: mode_sequence must contain each canonical mainline mode exactly once ("
            + expected
            + ")"
        )
    canonical_modes = {str(mode) for mode in CANONICAL_TRIPLET_MODES}
    normalized_modes = set(normalized)
    if normalized_modes != canonical_modes:
        missing = sorted(canonical_modes - normalized_modes)
        extra = sorted(normalized_modes - canonical_modes)
        raise ValueError(
            "same-checkpoint triplet requires omit/positive/negative exactly once; "
            + f"missing={missing} extra={extra}"
        )
    return tuple(normalized)


def _observation_signature(
    *,
    observation_surface: object | None,
    observation_seed: int | None,
) -> str | None:
    if observation_surface is None and observation_seed is None:
        return None
    return _sha256(
        {
            "observation_seed": observation_seed,
            "observation_surface": _jsonable(observation_surface),
        }
    )


def _normalize_surface_field(
    surface: Mapping[str, object],
    *,
    value_key: str,
    unavailable_key: str,
    default_reason: str,
) -> tuple[object | None, bool, str | None]:
    if value_key in surface and surface.get(value_key) is not None:
        return _jsonable(surface.get(value_key)), True, None
    reason = surface.get(unavailable_key)
    if reason is None:
        reason = default_reason
    return None, False, str(reason)


def _normalize_group_surface_mapping(
    value: object,
    *,
    branch: str,
    field_name: str,
) -> tuple[dict[str, object] | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        return (
            None,
            f"{field_name} is present but not keyed by action group; preserving the seven-group split requires a mapping instead of {type(value).__name__}",
        )
    contract_surface = gr00t_action_chain_telemetry.build_action_chain_contract_surface(
        branch
    )
    action_order = cast(Sequence[str], contract_surface["action_group_order"])
    allowed = set(action_order)
    normalized: dict[str, object] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        if key in allowed:
            normalized[key] = item
            continue
        if key.startswith("action.") and key[len("action.") :] in allowed:
            normalized[key[len("action.") :]] = item
    if normalized:
        return normalized, None
    return (
        {},
        f"{field_name} did not expose any recognized action groups; expected keys from {list(action_order)} or their action.* aliases",
    )


def _extract_action_delta_stage_inputs(
    surface: Mapping[str, object],
    *,
    branch: str,
    raw_action_reason: str | None,
    decoded_action_reason: str | None,
    absolute_action_reason: str | None,
    controller_input_reason: str | None,
) -> tuple[dict[str, Mapping[str, object] | None], dict[str, str | None]]:
    raw_group_values, raw_group_reason = _normalize_group_surface_mapping(
        surface.get("raw_action_by_group", surface.get("raw_action_chunk")),
        branch=branch,
        field_name="raw_action_by_group/raw_action_chunk",
    )
    decoded_group_values, decoded_group_reason = _normalize_group_surface_mapping(
        surface.get("decoded_action"),
        branch=branch,
        field_name="decoded_action",
    )
    absolute_group_values, absolute_group_reason = _normalize_group_surface_mapping(
        surface.get("absolute_action"),
        branch=branch,
        field_name="absolute_action",
    )
    controller_group_values, controller_group_reason = _normalize_group_surface_mapping(
        surface.get("controller_input"),
        branch=branch,
        field_name="controller_input",
    )
    return (
        {
            "raw_action": raw_group_values,
            "decoded_action": decoded_group_values,
            "absolute_action": absolute_group_values,
            "controller_input": controller_group_values,
        },
        {
            "raw_action": raw_group_reason or raw_action_reason,
            "decoded_action": decoded_group_reason or decoded_action_reason,
            "absolute_action": absolute_group_reason or absolute_action_reason,
            "controller_input": controller_group_reason or controller_input_reason,
        },
    )


def _build_action_delta_audit_summary(
    *,
    branch: str,
    mode_payloads: Mapping[str, Mapping[str, object]],
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    contract_surface = gr00t_action_chain_telemetry.build_action_chain_contract_surface(
        branch
    )
    mode_sidecars: dict[str, Mapping[str, object]] = {}
    per_mode_sidecar_backpointers: dict[str, dict[str, object]] = {}
    for indicator_mode, payload in mode_payloads.items():
        sidecar = payload.get("action_delta_sidecar")
        if isinstance(sidecar, Mapping):
            mode_sidecars[indicator_mode] = cast(Mapping[str, object], sidecar)
            per_mode_sidecar_backpointers[indicator_mode] = {
                "artifact_path": str(payload.get("artifact_path", "")),
                "json_field": "action_delta_sidecar",
                "summary_field": "action_delta_audit",
            }

    pair_summaries = (
        gr00t_action_chain_telemetry.build_action_chain_mode_pair_summaries(
            branch,
            mode_sidecars=mode_sidecars,
        )
    )
    ready = bool(mode_sidecars)
    return {
        **contract_surface,
        "audit_status": "READY" if ready else "UNAVAILABLE",
        "unavailable_reason": None
        if ready
        else str(
            unavailable_reason
            or "action-delta audit was unavailable because no per-mode sidecars were emitted"
        ),
        "mode_pair_summary_keys": [
            pair_name
            for pair_name, _, _ in gr00t_action_chain_telemetry.MODE_PAIR_SUMMARY_SPECS
        ],
        "per_mode_sidecar_backpointers": per_mode_sidecar_backpointers,
        "mode_pair_summaries": pair_summaries,
    }


def _validate_surface_map(
    surfaces_by_mode: Mapping[str, Mapping[str, object]] | None,
    *,
    mode_sequence: Sequence[str],
) -> dict[str, Mapping[str, object]]:
    if surfaces_by_mode is None:
        return {}
    normalized: dict[str, Mapping[str, object]] = {
        recap_policy.validate_mainline_runtime_indicator_mode(
            key,
            field_name="mode_surface_json key",
        ): value
        for key, value in surfaces_by_mode.items()
    }
    expected = set(mode_sequence)
    actual = set(normalized)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "same-checkpoint triplet requires all three canonical modes in mode surfaces; "
            + f"missing={missing} extra={extra}"
        )
    return normalized


def _mode_artifact_name(indicator_mode: str) -> str:
    return MODE_ARTIFACT_BASENAME_TEMPLATE.format(indicator_mode=indicator_mode)


def _triplet_summary_section(modes: Sequence[str]) -> dict[str, object]:
    return {
        "modes_emitted": 0,
        "mode_order": list(modes),
        "per_mode_prompt_text": {},
        "per_mode_artifact_paths": {},
        "per_mode_backpointers": {},
        "per_mode_surface_availability": {},
    }


def _normalized_stage_contract_limits(
    *,
    branch: str,
    stage_name: str,
) -> tuple[dict[str, tuple[float, float]] | None, str | None]:
    if stage_name == "decoded_action":
        return (
            dict(gr00t_action_chain_telemetry.DEFAULT_DECODE_LIMITS_BY_BRANCH[branch]),
            None,
        )
    if stage_name in {"absolute_action", "controller_input"}:
        return (
            dict(
                gr00t_action_chain_telemetry.DEFAULT_CONTROLLER_LIMITS_BY_BRANCH[branch]
            ),
            None,
        )
    if stage_name == "controller_output":
        return None, CONTROLLER_OUTPUT_UNAVAILABLE_REASON
    return (
        None,
        "raw_action has no repo-local decode/controller contract range; raw-stage distinctions remain debug-only and unnormalized",
    )


def resolve_triplet_terminal_stage(
    *,
    controller_output_available: object,
) -> str:
    return (
        "controller_output" if bool(controller_output_available) else "controller_input"
    )


def _shape_weight(shape: object, *, fallback: object) -> int:
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes, bytearray)):
        dims: list[int] = []
        for item in shape:
            if isinstance(item, bool) or not isinstance(item, int):
                continue
            dims.append(int(item))
        if dims:
            return max(1, int(math.prod(dims)))
    if isinstance(fallback, bool) or not isinstance(fallback, int):
        return 1
    return max(1, int(fallback))


def _build_pair_runtime_trace_stage_summary(
    *,
    branch: str,
    pair_summary: Mapping[str, object],
    stage_name: str,
) -> dict[str, object]:
    available_group_count_by_stage = cast(
        Mapping[str, object], pair_summary.get("available_group_count_by_stage", {})
    )
    difference_group_count_by_stage = cast(
        Mapping[str, object], pair_summary.get("difference_group_count_by_stage", {})
    )
    difference_groups_by_stage = cast(
        Mapping[str, object], pair_summary.get("difference_groups_by_stage", {})
    )
    available_group_count = _int_or_default(
        available_group_count_by_stage.get(stage_name, 0),
        default=0,
    )
    difference_group_count = _int_or_default(
        difference_group_count_by_stage.get(stage_name, 0),
        default=0,
    )
    difference_groups_raw = difference_groups_by_stage.get(stage_name, [])
    difference_groups = (
        [str(item) for item in difference_groups_raw]
        if isinstance(difference_groups_raw, Sequence)
        and not isinstance(difference_groups_raw, (str, bytes, bytearray))
        else []
    )
    limit_map, unavailable_reason = _normalized_stage_contract_limits(
        branch=branch,
        stage_name=stage_name,
    )
    normalized_metric: float | None = None
    max_group_metric: float | None = None
    normalization_available_group_count = 0
    if limit_map:
        weighted_total = 0.0
        total_weight = 0
        per_group = cast(Mapping[str, object], pair_summary.get("per_group", {}))
        for group_name, group_payload in per_group.items():
            if group_name not in limit_map or not isinstance(group_payload, Mapping):
                continue
            stages = group_payload.get("stages")
            if not isinstance(stages, Mapping):
                continue
            stage_payload = stages.get(stage_name)
            if not isinstance(stage_payload, Mapping):
                continue
            if not bool(stage_payload.get("available_in_both_modes", False)):
                continue
            mean_abs_delta = _float_or_none(stage_payload.get("mean_abs_delta"))
            if mean_abs_delta is None:
                continue
            lower, upper = limit_map[group_name]
            contract_range = float(upper) - float(lower)
            if contract_range <= 0.0:
                continue
            weight = _shape_weight(
                stage_payload.get("left_shape"),
                fallback=group_payload.get("dimension", 1),
            )
            metric = float(mean_abs_delta) / contract_range
            weighted_total += metric * float(weight)
            total_weight += int(weight)
            normalization_available_group_count += 1
            max_group_metric = (
                metric
                if max_group_metric is None
                else max(float(max_group_metric), metric)
            )
        if total_weight > 0:
            normalized_metric = _round_float(weighted_total / float(total_weight))
            if max_group_metric is not None:
                max_group_metric = _round_float(max_group_metric)
    return {
        "available": bool(available_group_count > 0),
        "difference_present": bool(difference_group_count > 0),
        "available_group_count": int(available_group_count),
        "difference_group_count": int(difference_group_count),
        "difference_groups": difference_groups,
        "normalization_available": normalized_metric is not None,
        "normalization_unavailable_reason": None
        if normalized_metric is not None
        else str(unavailable_reason)
        if unavailable_reason is not None
        else "no stage/group values were simultaneously available in both modes",
        "normalization_available_group_count": int(normalization_available_group_count),
        NORMALIZED_DELTA_METRIC_NAME: normalized_metric,
        "max_group_" + NORMALIZED_DELTA_METRIC_NAME: max_group_metric,
    }


def build_triplet_runtime_trace(
    *,
    branch: str,
    mode_payloads: Mapping[str, Mapping[str, object]],
    action_delta_audit: Mapping[str, object],
) -> dict[str, object]:
    prompt_pair_distinction: dict[str, bool] = {}
    token_pair_distinction: dict[str, dict[str, object]] = {}
    pair_stage_summaries: dict[str, dict[str, object]] = {}
    stage_max_metric: dict[str, float | None] = {
        "decoded_action": None,
        "absolute_action": None,
        "controller_input": None,
        "controller_output": None,
    }
    raw_action_distinct = False
    decoded_action_distinct = False
    absolute_distinct = False
    controller_input_distinct = False
    pair_summaries = action_delta_audit.get("mode_pair_summaries")
    normalized_pair_summaries = (
        cast(Mapping[str, object], pair_summaries)
        if isinstance(pair_summaries, Mapping)
        else {}
    )
    for (
        pair_name,
        left_mode,
        right_mode,
    ) in gr00t_action_chain_telemetry.MODE_PAIR_SUMMARY_SPECS:
        left_payload = mode_payloads.get(left_mode, {})
        right_payload = mode_payloads.get(right_mode, {})
        prompt_pair_distinction[pair_name] = bool(
            str(left_payload.get("prompt_text", ""))
            != str(right_payload.get("prompt_text", ""))
        )
        left_token_available = bool(left_payload.get("token_ids_available", False))
        right_token_available = bool(right_payload.get("token_ids_available", False))
        left_tokens = left_payload.get("token_ids")
        right_tokens = right_payload.get("token_ids")
        token_pair_distinction[pair_name] = {
            "available": bool(
                left_token_available
                and right_token_available
                and left_tokens is not None
                and right_tokens is not None
            ),
            "distinct": bool(
                left_token_available
                and right_token_available
                and left_tokens is not None
                and right_tokens is not None
                and _sha256(_jsonable(left_tokens)) != _sha256(_jsonable(right_tokens))
            ),
        }
        pair_summary_raw = normalized_pair_summaries.get(pair_name)
        pair_summary = (
            cast(Mapping[str, object], pair_summary_raw)
            if isinstance(pair_summary_raw, Mapping)
            else {}
        )
        difference_group_count_by_stage = cast(
            Mapping[str, object],
            pair_summary.get("difference_group_count_by_stage", {}),
        )
        raw_action_distinct = bool(
            raw_action_distinct
            or _int_or_default(
                difference_group_count_by_stage.get("raw_action", 0),
                default=0,
            )
            > 0
        )
        per_stage: dict[str, object] = {
            stage_name: _build_pair_runtime_trace_stage_summary(
                branch=branch,
                pair_summary=pair_summary,
                stage_name=stage_name,
            )
            for stage_name in (
                "raw_action",
                "decoded_action",
                "absolute_action",
                "controller_input",
                "controller_output",
            )
        }
        decoded_action_distinct = bool(
            decoded_action_distinct
            or cast(Mapping[str, object], per_stage["decoded_action"]).get(
                "difference_present", False
            )
        )
        absolute_distinct = bool(
            absolute_distinct
            or cast(Mapping[str, object], per_stage["absolute_action"]).get(
                "difference_present", False
            )
        )
        controller_input_distinct = bool(
            controller_input_distinct
            or cast(Mapping[str, object], per_stage["controller_input"]).get(
                "difference_present", False
            )
        )
        for stage_name in stage_max_metric:
            metric = _float_or_none(
                cast(Mapping[str, object], per_stage[stage_name]).get(
                    NORMALIZED_DELTA_METRIC_NAME
                )
            )
            if metric is None:
                continue
            prior = stage_max_metric[stage_name]
            stage_max_metric[stage_name] = (
                float(metric) if prior is None else max(float(prior), float(metric))
            )
        pair_stage_summaries[pair_name] = {
            "pair_name": pair_name,
            "left_mode": left_mode,
            "right_mode": right_mode,
            "available": bool(pair_summary.get("available", False)),
            "unavailable_reason": pair_summary.get("unavailable_reason"),
            "prompt_distinct": bool(prompt_pair_distinction[pair_name]),
            "token_ids": token_pair_distinction[pair_name],
            "stages": per_stage,
        }
    token_available_in_any_pair = any(
        bool(item.get("available", False)) for item in token_pair_distinction.values()
    )
    token_distinct_in_any_pair = any(
        bool(item.get("distinct", False)) for item in token_pair_distinction.values()
    )
    prompt_distinct_in_any_pair = any(prompt_pair_distinction.values())
    controller_output_available = bool(
        stage_max_metric["controller_output"] is not None
    )
    terminal_stage_used = resolve_triplet_terminal_stage(
        controller_output_available=controller_output_available,
    )
    return {
        "trace_role": DEBUG_PROBE_GATE_ROLE,
        "main_gate_eligible": False,
        "status": str(action_delta_audit.get("audit_status", "UNAVAILABLE")),
        "normalization_metric": NORMALIZED_DELTA_METRIC_NAME,
        "summary_stage_order": list(SUMMARY_STAGE_ORDER),
        "machine_checkpoint_order": list(MACHINE_CHECKPOINT_ORDER),
        "controller_output_available": bool(controller_output_available),
        "controller_output_unavailable_reason": None
        if controller_output_available
        else CONTROLLER_OUTPUT_UNAVAILABLE_REASON,
        "terminal_stage_used": terminal_stage_used,
        "prompt_surface": {
            "per_mode_prompt_text": {
                indicator_mode: str(payload.get("prompt_text", ""))
                for indicator_mode, payload in mode_payloads.items()
            },
            "pair_distinction": prompt_pair_distinction,
            "any_pair_distinct": bool(prompt_distinct_in_any_pair),
        },
        "token_surface": {
            "pair_distinction": token_pair_distinction,
            "available_in_any_pair": bool(token_available_in_any_pair),
            "any_pair_distinct": bool(token_distinct_in_any_pair),
        },
        "pair_stage_summaries": pair_stage_summaries,
        "stage_max_mean_abs_delta_over_contract_range": {
            stage_name: None if value is None else _round_float(value)
            for stage_name, value in stage_max_metric.items()
        },
        "upstream_distinction": {
            "prompt_or_token_distinct": bool(
                prompt_distinct_in_any_pair or token_distinct_in_any_pair
            ),
            "raw_action_distinct": bool(raw_action_distinct),
            "raw_or_decoded_distinct": bool(
                raw_action_distinct or decoded_action_distinct
            ),
            "absolute_distinct": bool(absolute_distinct),
            "controller_input_distinct": bool(controller_input_distinct),
            "controller_output_distinct": False,
        },
        "backpointer": {
            "summary_field": "runtime_trace",
            "action_delta_summary_field": "action_delta_audit",
        },
    }


def _recompute_summary_signature(summary_payload: dict[str, Any]) -> None:
    summary_payload["report_signature_sha256"] = _sha256(
        {
            key: value
            for key, value in summary_payload.items()
            if key != "report_signature_sha256"
        }
    )


def _build_triplet_failure_note(summary: Mapping[str, object]) -> str:
    gate = summary.get("triplet_gate")
    gate_payload = dict(gate) if isinstance(gate, Mapping) else {}
    issues = gate_payload.get("issues")
    rendered_issues: list[str] = []
    if isinstance(issues, list):
        for item in issues:
            if isinstance(item, Mapping):
                rendered_issues.append(
                    "  - "
                    + f"`{item.get('code', 'unknown')}` @ `{item.get('field_path', '?')}`: "
                    + str(item.get("message", ""))
                )
            else:
                rendered_issues.append(f"  - `{item}`")
    if not rendered_issues:
        rendered_issues = ["  - `no blocking issues captured`"]
    resolved_binding = gate_payload.get("resolved_checkpoint_binding")
    binding_payload = (
        dict(resolved_binding) if isinstance(resolved_binding, Mapping) else {}
    )
    provenance = gate_payload.get("checkpoint_provenance")
    provenance_payload = dict(provenance) if isinstance(provenance, Mapping) else {}
    lines = [
        "# GR00T same-checkpoint triplet gate 失败说明",
        "",
        f"- formal_eligibility: `{summary.get('formal_eligibility', 'BLOCK')}`",
        f"- reason_code: `{summary.get('reason_code', BLOCKED_REASON_CODE)}`",
        f"- checkpoint_selected: `{binding_payload.get('checkpoint_selected')}`",
        f"- checkpoint_loaded: `{binding_payload.get('checkpoint_loaded')}`",
        f"- evaluation_binding.server_load_path: `{binding_payload.get('server_load_path')}`",
        f"- manifest_path: `{gate_payload.get('run_manifest_path')}`",
        f"- provenance_status: `{provenance_payload.get('loadability_status')}`",
        "- issues:",
        *rendered_issues,
        "",
        "该 triplet 运行未能证明 selected checkpoint、loaded checkpoint、run manifest 核心绑定与 controller/action contract 一致，因此本次输出不得作为可解释的 same-checkpoint 证据表面。",
        "",
    ]
    return "\n".join(lines)


def _write_failure_note(path: Path, summary: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    _ = tmp.write_text(_build_triplet_failure_note(summary), encoding="utf-8")
    _ = tmp.replace(path)
    return path


def _build_provenance_metadata_from_manifest(
    manifest_payload: Mapping[str, Any],
) -> dict[str, Any]:
    core = manifest_payload.get("core")
    normalized_core = dict(core) if isinstance(core, Mapping) else {}
    evaluation_binding = manifest_payload.get("evaluation_binding")
    normalized_binding = (
        dict(evaluation_binding) if isinstance(evaluation_binding, Mapping) else {}
    )
    base_model_path = normalized_binding.get("base_model_path")
    metadata: dict[str, Any] = {
        "comparable_run_spec": {
            "checkpoint_rule": {
                "selected_checkpoint_path": normalized_core.get("checkpoint_selected")
            },
            "stable_base": {
                "base_model": base_model_path,
            },
        },
        "evaluation_binding": normalized_binding,
        "base_model_path": base_model_path,
    }
    checkpoint_loaded = normalized_core.get("checkpoint_loaded")
    if checkpoint_loaded is not None:
        metadata["checkpoint_loaded"] = checkpoint_loaded
    return metadata


def _triplet_binding_exact_value_issues(
    *,
    normalized_manifest: Mapping[str, Any],
    declared_checkpoint_loaded: str,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    core = normalized_manifest.get("core")
    normalized_core = dict(core) if isinstance(core, Mapping) else {}

    expected_string_fields = {
        "carrier_schema_version": recap_run_manifest.TEXT_CARRIER_SCHEMA_VERSION,
        "carrier_route": recap_run_manifest.TEXT_CARRIER_ROUTE,
        "prompt_source_field": recap_run_manifest.PROMPT_SOURCE_FIELD,
        "indicator_source": recap_run_manifest.INDICATOR_SOURCE_FIELD,
    }
    for field_name, expected in expected_string_fields.items():
        value = normalized_core.get(field_name)
        if value is None:
            continue
        if str(value) != str(expected):
            issues.append(
                _issue(
                    "manifest_core_mismatch",
                    f"core.{field_name}",
                    f"core.{field_name} must equal {expected!r} for same-checkpoint triplet binding",
                )
            )

    policy_horizon = normalized_core.get("policy_horizon")
    if isinstance(policy_horizon, int) and policy_horizon != int(
        gr00t_eval_contract_gate.DEFAULT_POLICY_HORIZON_EXPECTED
    ):
        issues.append(
            _issue(
                "formal_contract_mismatch",
                "core.policy_horizon",
                "core.policy_horizon must match the frozen G1 formal policy horizon",
            )
        )

    n_action_steps = normalized_core.get("n_action_steps")
    if isinstance(n_action_steps, int) and n_action_steps != int(
        gr00t_eval_contract_gate.DEFAULT_N_ACTION_STEPS
    ):
        issues.append(
            _issue(
                "formal_contract_mismatch",
                "core.n_action_steps",
                "core.n_action_steps must match the frozen G1 formal execution window",
            )
        )

    declared_checkpoint = str(declared_checkpoint_loaded).strip()
    manifest_checkpoint = normalized_core.get("checkpoint_loaded")
    if (
        declared_checkpoint
        and isinstance(manifest_checkpoint, str)
        and manifest_checkpoint.strip()
        and not _checkpoint_reference_matches(
            declared_checkpoint, manifest_checkpoint.strip()
        )
    ):
        issues.append(
            _issue(
                "checkpoint_argument_mismatch",
                "checkpoint_loaded",
                "CLI checkpoint_loaded does not match run manifest core.checkpoint_loaded",
            )
        )
    return issues


def _checkpoint_reference_matches(left: str, right: str) -> bool:
    left_text = str(left).strip()
    right_text = str(right).strip()
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True

    def _normalized_local_checkpoint(raw: str) -> str | None:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved = path.resolve()
        if not resolved.exists():
            return None
        if resolved.is_file():
            resolved = resolved.parent
        return str(resolved)

    left_checkpoint = _normalized_local_checkpoint(left_text)
    right_checkpoint = _normalized_local_checkpoint(right_text)
    return bool(
        left_checkpoint is not None
        and right_checkpoint is not None
        and left_checkpoint == right_checkpoint
    )


def build_triplet_binding_gate(
    *,
    run_manifest_payload: Mapping[str, Any],
    run_manifest_path: Path,
    output_dir: Path,
    repo_root: Path,
    declared_checkpoint_loaded: str,
    observation_seed: int | None,
    observation_signature_sha256: str | None,
) -> dict[str, Any]:
    validation = recap_run_manifest.validate_run_manifest(
        run_manifest_payload,
        repo_root=repo_root,
    )
    normalized_manifest = dict(validation["normalized_manifest"])
    validation_issues = [
        dict(cast(Mapping[str, object], issue)) for issue in validation["issues"]
    ]
    exact_value_issues = _triplet_binding_exact_value_issues(
        normalized_manifest=normalized_manifest,
        declared_checkpoint_loaded=declared_checkpoint_loaded,
    )
    provenance_report = (
        gr00t_checkpoint_provenance_gate.build_checkpoint_provenance_report(
            metadata=_build_provenance_metadata_from_manifest(normalized_manifest),
            metadata_path=run_manifest_path,
            repo_root=repo_root,
            output_dir=output_dir,
        )
    )
    provenance_issues: list[dict[str, str]] = []
    if provenance_report["formal_eligibility"] != "ALLOW":
        for reason in cast(list[object], provenance_report.get("gate_reasons", [])):
            provenance_issues.append(
                _issue(
                    "checkpoint_provenance_blocked",
                    "checkpoint_provenance",
                    str(reason),
                )
            )
    combined_issues = _dedupe_issue_list(
        [
            *cast(list[Mapping[str, object]], validation_issues),
            *cast(list[Mapping[str, object]], exact_value_issues),
            *cast(list[Mapping[str, object]], provenance_issues),
        ]
    )
    core = normalized_manifest.get("core")
    normalized_core = dict(core) if isinstance(core, Mapping) else {}
    evaluation_binding = normalized_manifest.get("evaluation_binding")
    normalized_binding = (
        dict(evaluation_binding) if isinstance(evaluation_binding, Mapping) else {}
    )
    formal_eligibility = "ALLOW" if not combined_issues else "BLOCK"
    return {
        "gate_name": TRIPLET_GATE_NAME,
        "run_manifest_path": str(run_manifest_path),
        "status": "PASS" if formal_eligibility == "ALLOW" else "FAIL",
        "formal_eligibility": formal_eligibility,
        "reason_code": OK_REASON_CODE
        if formal_eligibility == "ALLOW"
        else BLOCKED_REASON_CODE,
        "issues": combined_issues,
        "run_manifest": {
            "schema_version": normalized_manifest.get("schema_version"),
            "artifact_kind": normalized_manifest.get("artifact_kind"),
            "core_digest": validation["core_digest"],
            "core": _jsonable(normalized_core),
            "validation_formal_eligibility": validation["formal_eligibility"],
            "validation_issues": validation_issues,
            "checkpoint_binding": _jsonable(validation["checkpoint_binding"]),
        },
        "checkpoint_provenance": {
            "status": provenance_report.get("status"),
            "formal_eligibility": provenance_report.get("formal_eligibility"),
            "reason_code": provenance_report.get("reason_code"),
            "loadability_status": provenance_report.get("loadability_status"),
            "gate_reasons": _jsonable(provenance_report.get("gate_reasons", [])),
            "is_base_fallback": provenance_report.get("is_base_fallback"),
        },
        "resolved_checkpoint_binding": {
            "checkpoint_selected": normalized_core.get("checkpoint_selected"),
            "checkpoint_loaded": normalized_core.get("checkpoint_loaded"),
            "server_load_path": normalized_binding.get("server_load_path"),
        },
        "same_observation_binding": {
            "same_observation_locked": True,
            "observation_seed": observation_seed,
            "observation_signature_sha256": observation_signature_sha256,
        },
    }


def _build_internal_triplet_binding_gate(
    *,
    run_manifest_path: Path | None,
    message: str,
    observation_seed: int | None,
    observation_signature_sha256: str | None,
) -> dict[str, Any]:
    issues = [_issue("triplet_binding_input_error", "run_manifest_json", str(message))]
    return {
        "gate_name": TRIPLET_GATE_NAME,
        "run_manifest_path": None
        if run_manifest_path is None
        else str(run_manifest_path),
        "status": "FAIL",
        "formal_eligibility": "BLOCK",
        "reason_code": BLOCKED_REASON_CODE,
        "issues": issues,
        "run_manifest": {
            "schema_version": None,
            "artifact_kind": None,
            "core_digest": None,
            "core": {},
            "validation_formal_eligibility": "BLOCK",
            "validation_issues": issues,
            "checkpoint_binding": {},
        },
        "checkpoint_provenance": {
            "status": "FAIL",
            "formal_eligibility": "BLOCK",
            "reason_code": BLOCKED_REASON_CODE,
            "loadability_status": "BLOCKED_RUN_MANIFEST_INPUT_INVALID",
            "gate_reasons": [str(message)],
            "is_base_fallback": None,
        },
        "resolved_checkpoint_binding": {
            "checkpoint_selected": None,
            "checkpoint_loaded": None,
            "server_load_path": None,
        },
        "same_observation_binding": {
            "same_observation_locked": True,
            "observation_seed": observation_seed,
            "observation_signature_sha256": observation_signature_sha256,
        },
    }


def build_blocked_same_checkpoint_triplet_bundle(
    *,
    checkpoint_loaded: str,
    output_dir: Path,
    summary_json_path: Path,
    observation_seed: int | None,
    observation_surface: object | None,
    mode_sequence: Sequence[object] | None,
    triplet_gate: Mapping[str, object],
) -> dict[str, Any]:
    modes = _normalize_triplet_modes(mode_sequence)
    observation_signature_sha256 = _observation_signature(
        observation_surface=observation_surface,
        observation_seed=observation_seed,
    )
    action_delta_audit = _build_action_delta_audit_summary(
        branch=DEFAULT_TRIPLET_BRANCH,
        mode_payloads={},
        unavailable_reason=(
            "triplet mode artifacts were blocked before additive action-delta sidecars could be emitted"
        ),
    )
    runtime_trace = build_triplet_runtime_trace(
        branch=DEFAULT_TRIPLET_BRANCH,
        mode_payloads={},
        action_delta_audit=action_delta_audit,
    )
    summary_payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "output_path": str(summary_json_path.resolve()),
        "checkpoint_loaded": checkpoint_loaded,
        "gate_role": DEBUG_PROBE_GATE_ROLE,
        "main_gate_eligible": False,
        "same_checkpoint_locked": True,
        "same_observation_locked": True,
        "observation_seed": observation_seed,
        "observation_signature_sha256": observation_signature_sha256,
        "triplet_variable": "indicator_mode",
        "numeric_advantage_main_control": False,
        "indicator_modes": list(modes),
        "prompt_source_field": recap_policy.MAINLINE_RUNTIME_PROMPT_SOURCE_FIELD,
        "carrier_route": recap_policy.MAINLINE_RUNTIME_ROUTE,
        "policy_class_name": recap_policy.MAINLINE_RUNTIME_POLICY_CLASS_NAME,
        "episodes": 0,
        "success_count": None,
        "success_rate": None,
        "episode_telemetry_jsonl": None,
        "step_telemetry_jsonl": None,
        "mode_artifacts": [],
        "summary": {
            **_triplet_summary_section(modes),
            "blocked_before_mode_artifacts": True,
        },
        "action_delta_audit": action_delta_audit,
        "runtime_trace": runtime_trace,
        "backpointer": {
            "writer_script": WRITER_SCRIPT,
            "output_dir": str(output_dir.resolve()),
        },
        "status": "FAIL",
        "formal_eligibility": "BLOCK",
        "reason_code": BLOCKED_REASON_CODE,
        "failure_note_path": None,
        "triplet_gate": _jsonable(triplet_gate),
    }
    _recompute_summary_signature(summary_payload)
    return {
        "summary": summary_payload,
        "mode_payloads": {},
    }


def _build_runtime_prompt_bundle(
    *,
    prompt_raw: str,
    indicator_mode: str,
    variant: str,
) -> runtime_prompt.PromptSurfaceBundle:
    config = runtime_prompt.resolve_runtime_indicator_config(
        requested_indicator_mode=indicator_mode,
        variant=variant,
    )
    return runtime_prompt.build_runtime_prompt_bundle(prompt_raw, config=config)


def build_mode_artifact_payload(
    *,
    branch: str,
    checkpoint_loaded: str,
    prompt_raw: str,
    indicator_mode: str,
    variant: str,
    observation_seed: int | None,
    observation_signature_sha256: str | None,
    output_dir: Path,
    summary_json_path: Path,
    mode_surface: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    normalized_mode = recap_policy.validate_mainline_runtime_indicator_mode(
        indicator_mode,
        field_name="indicator_mode",
    )
    policy_spec = recap_policy.build_runtime_policy_spec(indicator_mode=normalized_mode)
    prompt_bundle = _build_runtime_prompt_bundle(
        prompt_raw=prompt_raw,
        indicator_mode=normalized_mode,
        variant=variant,
    )
    artifact_path = (output_dir / _mode_artifact_name(normalized_mode)).resolve()
    surface = mode_surface or {}
    policy_options: dict[str, object] = {"indicator_mode": normalized_mode}
    if observation_seed is not None:
        policy_options["seed"] = int(observation_seed)

    raw_action_chunk, raw_action_available, raw_action_reason = (
        _normalize_surface_field(
            surface,
            value_key="raw_action_chunk",
            unavailable_key="raw_action_chunk_unavailable_reason",
            default_reason=(
                "raw model action chunk is not exposed on the chosen runtime path; no fabricated raw surface was emitted"
            ),
        )
    )
    decoded_action, decoded_action_available, decoded_action_reason = (
        _normalize_surface_field(
            surface,
            value_key="decoded_action",
            unavailable_key="decoded_action_unavailable_reason",
            default_reason=(
                "decoded/post-processor action was not returned by the chosen runtime path"
            ),
        )
    )
    post_transform_action, post_transform_available, post_transform_reason = (
        _normalize_surface_field(
            surface,
            value_key="post_transform_action",
            unavailable_key="post_transform_action_unavailable_reason",
            default_reason=(
                "post-transform/controller-input action surface is unavailable on the chosen runtime path"
            ),
        )
    )
    absolute_action, absolute_action_available, absolute_action_reason = (
        _normalize_surface_field(
            surface,
            value_key="absolute_action",
            unavailable_key="absolute_action_unavailable_reason",
            default_reason=(
                "absolute_action was not exposed separately on the chosen runtime path; the additive audit keeps this stage unavailable instead of reinterpreting legacy post_transform_action"
            ),
        )
    )
    controller_input, controller_input_available, controller_input_reason = (
        _normalize_surface_field(
            surface,
            value_key="controller_input",
            unavailable_key="controller_input_unavailable_reason",
            default_reason=(
                "controller_input was not exposed separately on the chosen runtime path; no fabricated controller-side surface was emitted"
            ),
        )
    )
    token_ids, token_ids_available, token_ids_reason = _normalize_surface_field(
        surface,
        value_key="token_ids",
        unavailable_key="token_ids_unavailable_reason",
        default_reason=(
            "token ids are not cheaply available from the chosen runtime path; emitted as null instead of fabricating them"
        ),
    )
    action_delta_stage_inputs, action_delta_stage_reasons = (
        _extract_action_delta_stage_inputs(
            surface,
            branch=branch,
            raw_action_reason=raw_action_reason,
            decoded_action_reason=decoded_action_reason,
            absolute_action_reason=absolute_action_reason,
            controller_input_reason=controller_input_reason,
        )
    )
    action_delta_sidecar = (
        gr00t_action_chain_telemetry.build_grouped_action_chain_sidecar(
            branch,
            stage_group_values=action_delta_stage_inputs,
            stage_unavailable_reasons=action_delta_stage_reasons,
        )
    )

    payload: dict[str, Any] = {
        "schema_version": MODE_SCHEMA_VERSION,
        "artifact_kind": MODE_ARTIFACT_KIND,
        "branch": branch,
        "checkpoint_loaded": checkpoint_loaded,
        "same_checkpoint_locked": True,
        "same_observation_locked": True,
        "observation_seed": observation_seed,
        "observation_signature_sha256": observation_signature_sha256,
        "indicator_mode": normalized_mode,
        "prompt_raw": prompt_raw,
        "prompt_text": prompt_bundle.prompt_text,
        "prompt_source_field": str(policy_spec["prompt_source_field"]),
        "prompt_source_value": prompt_raw,
        "prompt_text_surface": prompt_bundle.prompt_text_surface,
        "prompt_provenance": _jsonable(prompt_bundle.prompt_provenance),
        "carrier_route": str(policy_spec["carrier_route"]),
        "policy_class_name": str(policy_spec["policy_class_name"]),
        "indicator_source": prompt_bundle.indicator_source,
        "consumer_mode": prompt_bundle.consumer_mode,
        "fixed_indicator_mode": prompt_bundle.fixed_indicator_mode,
        "critic_checkpoint_ref": prompt_bundle.critic_checkpoint_ref,
        "policy_options": _jsonable(policy_options),
        "raw_action_chunk": raw_action_chunk,
        "raw_action_chunk_available": raw_action_available,
        "raw_action_chunk_unavailable_reason": raw_action_reason,
        "decoded_action": decoded_action,
        "decoded_action_available": decoded_action_available,
        "decoded_action_unavailable_reason": decoded_action_reason,
        "post_transform_action": post_transform_action,
        "post_transform_action_available": post_transform_available,
        "post_transform_action_unavailable_reason": post_transform_reason,
        "absolute_action": absolute_action,
        "absolute_action_available": absolute_action_available,
        "absolute_action_unavailable_reason": absolute_action_reason,
        "controller_input": controller_input,
        "controller_input_available": controller_input_available,
        "controller_input_unavailable_reason": controller_input_reason,
        "token_ids": token_ids,
        "token_ids_available": token_ids_available,
        "token_ids_unavailable_reason": token_ids_reason,
        "surface_source": surface.get("surface_source", "not_provided"),
        "surface_runtime_metadata": _jsonable(surface.get("runtime_metadata", {})),
        "action_delta_sidecar": action_delta_sidecar,
        "artifact_path": str(artifact_path),
        "backpointer": {
            "writer_script": WRITER_SCRIPT,
            "summary_json": str(summary_json_path.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
        "action_delta_sidecar_backpointer": {
            "summary_json": str(summary_json_path.resolve()),
            "summary_field": "action_delta_audit",
            "artifact_field": "action_delta_sidecar",
        },
    }
    payload["mode_signature_sha256"] = _sha256(
        {
            "branch": branch,
            "checkpoint_loaded": checkpoint_loaded,
            "indicator_mode": normalized_mode,
            "prompt_text": payload["prompt_text"],
            "raw_action_chunk": payload["raw_action_chunk"],
            "decoded_action": payload["decoded_action"],
            "post_transform_action": payload["post_transform_action"],
            "absolute_action": payload["absolute_action"],
            "controller_input": payload["controller_input"],
            "token_ids": payload["token_ids"],
        }
    )
    return payload


def build_same_checkpoint_triplet_bundle(
    *,
    branch: str = DEFAULT_TRIPLET_BRANCH,
    checkpoint_loaded: str,
    prompt_raw: str,
    output_dir: Path,
    summary_json_path: Path,
    observation_seed: int | None,
    observation_surface: object | None = None,
    variant: str = DEFAULT_VARIANT,
    mode_sequence: Sequence[object] | None = None,
    mode_surface_by_mode: Mapping[str, Mapping[str, object]] | None = None,
    mode_surface_executor: ModeSurfaceExecutor | None = None,
) -> dict[str, Any]:
    if mode_surface_by_mode is not None and mode_surface_executor is not None:
        raise ValueError(
            "Provide either mode_surface_by_mode or mode_surface_executor, not both"
        )
    modes = _normalize_triplet_modes(mode_sequence)
    normalized_surfaces = _validate_surface_map(
        mode_surface_by_mode,
        mode_sequence=modes,
    )
    observation_signature_sha256 = _observation_signature(
        observation_surface=observation_surface,
        observation_seed=observation_seed,
    )
    mode_payloads: dict[str, dict[str, Any]] = {}
    for indicator_mode in modes:
        surface = normalized_surfaces.get(indicator_mode, None)
        if surface is None and mode_surface_executor is not None:
            prompt_bundle = _build_runtime_prompt_bundle(
                prompt_raw=prompt_raw,
                indicator_mode=indicator_mode,
                variant=variant,
            )
            policy_options: dict[str, object] = {"indicator_mode": indicator_mode}
            if observation_seed is not None:
                policy_options["seed"] = int(observation_seed)
            surface = mode_surface_executor(
                indicator_mode=indicator_mode,
                prompt_text=prompt_bundle.prompt_text,
                policy_options=policy_options,
            )
        mode_payloads[indicator_mode] = build_mode_artifact_payload(
            branch=branch,
            checkpoint_loaded=checkpoint_loaded,
            prompt_raw=prompt_raw,
            indicator_mode=indicator_mode,
            variant=variant,
            observation_seed=observation_seed,
            observation_signature_sha256=observation_signature_sha256,
            output_dir=output_dir,
            summary_json_path=summary_json_path,
            mode_surface=surface,
        )

    action_delta_audit = _build_action_delta_audit_summary(
        branch=branch,
        mode_payloads=mode_payloads,
    )
    runtime_trace = build_triplet_runtime_trace(
        branch=branch,
        mode_payloads=mode_payloads,
        action_delta_audit=action_delta_audit,
    )

    summary_payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "output_path": str(summary_json_path.resolve()),
        "branch": branch,
        "checkpoint_loaded": checkpoint_loaded,
        "gate_role": DEBUG_PROBE_GATE_ROLE,
        "main_gate_eligible": False,
        "same_checkpoint_locked": True,
        "same_observation_locked": True,
        "observation_seed": observation_seed,
        "observation_signature_sha256": observation_signature_sha256,
        "triplet_variable": "indicator_mode",
        "numeric_advantage_main_control": False,
        "indicator_modes": list(modes),
        "prompt_source_field": recap_policy.MAINLINE_RUNTIME_PROMPT_SOURCE_FIELD,
        "carrier_route": recap_policy.MAINLINE_RUNTIME_ROUTE,
        "policy_class_name": recap_policy.MAINLINE_RUNTIME_POLICY_CLASS_NAME,
        "episodes": 1,
        "success_count": None,
        "success_rate": None,
        "episode_telemetry_jsonl": None,
        "step_telemetry_jsonl": None,
        "mode_artifacts": [
            {
                "indicator_mode": indicator_mode,
                "prompt_text": mode_payloads[indicator_mode]["prompt_text"],
                "artifact_path": mode_payloads[indicator_mode]["artifact_path"],
                "backpointer": _jsonable(mode_payloads[indicator_mode]["backpointer"]),
                "action_delta_sidecar_backpointer": _jsonable(
                    mode_payloads[indicator_mode]["action_delta_sidecar_backpointer"]
                ),
                "mode_signature_sha256": mode_payloads[indicator_mode][
                    "mode_signature_sha256"
                ],
            }
            for indicator_mode in modes
        ],
        "summary": {
            "modes_emitted": len(modes),
            "mode_order": list(modes),
            "per_mode_prompt_text": {
                indicator_mode: mode_payloads[indicator_mode]["prompt_text"]
                for indicator_mode in modes
            },
            "per_mode_artifact_paths": {
                indicator_mode: mode_payloads[indicator_mode]["artifact_path"]
                for indicator_mode in modes
            },
            "per_mode_backpointers": {
                indicator_mode: _jsonable(mode_payloads[indicator_mode]["backpointer"])
                for indicator_mode in modes
            },
            "per_mode_surface_availability": {
                indicator_mode: {
                    "raw_action_chunk": bool(
                        mode_payloads[indicator_mode]["raw_action_chunk_available"]
                    ),
                    "decoded_action": bool(
                        mode_payloads[indicator_mode]["decoded_action_available"]
                    ),
                    "post_transform_action": bool(
                        mode_payloads[indicator_mode]["post_transform_action_available"]
                    ),
                    "absolute_action": bool(
                        mode_payloads[indicator_mode]["absolute_action_available"]
                    ),
                    "controller_input": bool(
                        mode_payloads[indicator_mode]["controller_input_available"]
                    ),
                    "token_ids": bool(
                        mode_payloads[indicator_mode]["token_ids_available"]
                    ),
                }
                for indicator_mode in modes
            },
        },
        "action_delta_audit": action_delta_audit,
        "runtime_trace": runtime_trace,
        "backpointer": {
            "writer_script": WRITER_SCRIPT,
            "output_dir": str(output_dir.resolve()),
        },
    }
    summary_payload["report_signature_sha256"] = _sha256(
        {
            key: value
            for key, value in summary_payload.items()
            if key != "report_signature_sha256"
        }
    )
    return {
        "summary": summary_payload,
        "mode_payloads": mode_payloads,
    }


def write_same_checkpoint_triplet_bundle(
    bundle: Mapping[str, object],
) -> dict[str, Path]:
    summary = cast(Mapping[str, object], bundle["summary"])
    mode_payloads = cast(Mapping[str, Mapping[str, object]], bundle["mode_payloads"])
    summary_path = Path(str(summary["output_path"]))
    indicator_modes = summary["indicator_modes"]
    if not isinstance(indicator_modes, Sequence) or isinstance(indicator_modes, str):
        raise TypeError("summary.indicator_modes must be a sequence of mode strings")
    written_paths: dict[str, Path] = {}
    for indicator_mode in _normalize_triplet_modes(list(indicator_modes)):
        if indicator_mode not in mode_payloads:
            continue
        payload = mode_payloads[indicator_mode]
        written_paths[indicator_mode] = _write_json(
            Path(str(payload["artifact_path"])),
            payload,
        )
    written_paths["summary"] = _write_json(summary_path, summary)
    return written_paths


def _normalize_live_server_checkpoint(
    *,
    checkpoint_loaded: str,
    server_provenance: Mapping[str, object] | None,
) -> str:
    text = str(checkpoint_loaded).strip()
    if text:
        return text
    if isinstance(server_provenance, Mapping):
        for key in (
            "checkpoint_loaded",
            "policy_model_path",
            "overlay_from",
            "base_model_path",
        ):
            value = server_provenance.get(key)
            if value is None:
                continue
            candidate = str(value).strip()
            if candidate:
                return candidate
    raise ValueError(
        "checkpoint_loaded is required unless policy server provenance exposes a usable checkpoint/model path"
    )


def _normalize_policy_observation(obs: Mapping[str, object]) -> dict[str, object]:
    module = importlib.import_module("work.recap.scripts.gr00t_public_anchor_eval")
    fn = getattr(module, "_normalize_policy_observation")
    sanitized_obs = recap_policy.filter_canonical_serving_observation(
        obs,
        field_name="same_checkpoint_triplet.policy_observation",
    )
    normalized = cast(dict[str, object], fn(sanitized_obs))
    np = importlib.import_module("numpy")
    for key, value in list(normalized.items()):
        if not (str(key).startswith("video.") or str(key).endswith("_image")):
            continue
        arr = np.asarray(value, dtype=np.uint8)
        if arr.ndim == 3:
            normalized[str(key)] = arr[None, ...]
    return normalized


def _batch_policy_server_observation(obs: Mapping[str, object]) -> dict[str, object]:
    np = importlib.import_module("numpy")
    batched: dict[str, object] = {}
    for key, value in obs.items():
        text_key = str(key)
        if text_key.startswith("video."):
            arr = np.asarray(value, dtype=np.uint8)
            if arr.ndim == 3:
                arr = arr[None, None, ...]
            elif arr.ndim == 4:
                arr = arr[None, ...]
            batched[text_key] = arr
            continue
        if text_key.startswith("state."):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, None, :]
            elif arr.ndim == 2:
                arr = arr[None, ...]
            batched[text_key] = arr
            continue
        if text_key.startswith("annotation."):
            if isinstance(value, str):
                batched[text_key] = [value]
            elif isinstance(value, Sequence) and not isinstance(
                value, (bytes, bytearray)
            ):
                batched[text_key] = list(value)
            else:
                batched[text_key] = [str(value)]
            continue
        batched[text_key] = value
    return batched


def _normalize_client_host(host: str) -> str:
    module = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(module, "normalize_client_host")
    return str(fn(str(host)))


def _make_policy_client(host: str, port: int, timeout_ms: int) -> Any:
    module = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(module, "make_policy_client")
    return fn(host=str(host), port=int(port), timeout_ms=int(timeout_ms))


def _safe_ping(client: Any, timeout_ms: int) -> bool:
    module = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(module, "safe_ping")
    return bool(fn(client, int(timeout_ms)))


def _policy_server_returned_action_surface(
    *,
    action: Mapping[str, object],
    action_info: object,
    policy_options: Mapping[str, object],
    host: str,
    server_port: int,
) -> dict[str, object]:
    returned_action = _jsonable(action)
    surface_semantics = {
        "server_route": "policy_server.get_action",
        "returned_action_contract": (
            "GR00T SimPolicyWrapper returns the flat action.* mapping consumed by the "
            "simulation path. The public route does not expose the internal raw model "
            "chunk or the pre-relative-to-absolute tensor separately."
        ),
        "stage_binding": {
            "decoded_action": (
                "backward-compatible alias for the server-returned action surface"
            ),
            "absolute_action": (
                "same server-returned action surface; relative groups have already "
                "passed through the processor relative-to-absolute conversion"
            ),
            "controller_input": (
                "same server-returned action surface used as the current live "
                "controller-input seam; no additional controller_output seam exists yet"
            ),
        },
        "not_available": {
            "raw_action_chunk": "raw model action chunk is not exposed on this route",
            "controller_output": CONTROLLER_OUTPUT_UNAVAILABLE_REASON,
        },
    }
    return {
        "decoded_action": returned_action,
        "decoded_action_unavailable_reason": None,
        "post_transform_action": returned_action,
        "post_transform_action_unavailable_reason": None,
        "absolute_action": returned_action,
        "absolute_action_unavailable_reason": None,
        "controller_input": returned_action,
        "controller_input_unavailable_reason": None,
        "raw_action_chunk_unavailable_reason": (
            "policy server get_action() returns the sim-wrapper action surface only; "
            "raw model action chunk is not exposed on this route"
        ),
        "token_ids_unavailable_reason": (
            "policy server route does not expose token ids on get_action()"
        ),
        "surface_source": "policy_server.get_action",
        "surface_semantics": surface_semantics,
        "runtime_metadata": {
            "policy_options": _jsonable(policy_options),
            "action_info": _jsonable(action_info),
            "server_host": host,
            "server_port": int(server_port),
        },
    }


def collect_policy_server_mode_surfaces(
    *,
    observation: Mapping[str, object],
    observation_seed: int | None,
    mode_sequence: Sequence[object] | None,
    server_host: str,
    server_port: int,
    server_timeout_ms: int,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    normalized_obs = _batch_policy_server_observation(
        _normalize_policy_observation(observation)
    )
    host = _normalize_client_host(server_host)
    client = _make_policy_client(
        host=host,
        port=int(server_port),
        timeout_ms=int(server_timeout_ms),
    )
    if not _safe_ping(client, int(server_timeout_ms)):
        raise RuntimeError(
            f"policy server ping failed for {host}:{int(server_port)} within {int(server_timeout_ms)}ms"
        )

    server_info: Mapping[str, object] | None = None
    server_provenance: Mapping[str, object] | None = None
    try:
        info_payload = client.call_endpoint("get_server_info", requires_input=False)
        if isinstance(info_payload, Mapping):
            server_info = cast(Mapping[str, object], info_payload)
    except Exception:
        server_info = None
    try:
        provenance_payload = client.call_endpoint(
            "get_provenance", requires_input=False
        )
        if isinstance(provenance_payload, Mapping):
            server_provenance = cast(Mapping[str, object], provenance_payload)
    except Exception:
        server_provenance = None

    surfaces: dict[str, dict[str, object]] = {}
    for indicator_mode in _normalize_triplet_modes(mode_sequence):
        options: dict[str, object] = {"indicator_mode": indicator_mode}
        if observation_seed is not None:
            options["seed"] = int(observation_seed)
        client.reset(options=options)
        action, action_info = client.get_action(normalized_obs, options=options)
        surfaces[indicator_mode] = _policy_server_returned_action_surface(
            action=cast(Mapping[str, object], action),
            action_info=action_info,
            policy_options=options,
            host=host,
            server_port=int(server_port),
        )
    metadata = {
        "server_host": host,
        "server_port": int(server_port),
        "server_info": _jsonable(server_info),
        "server_provenance": _jsonable(server_provenance),
    }
    return surfaces, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = _validate_output_dir(
            _resolve_path(REPO_ROOT, str(args.output_dir))
        )
        summary_json_path = (
            _resolve_path(REPO_ROOT, str(args.summary_json))
            if str(args.summary_json).strip()
            else (output_dir / DEFAULT_SUMMARY_JSON_NAME).resolve()
        )
        if summary_json_path.exists() and summary_json_path.is_dir():
            raise ValueError(
                f"summary_json must be a file path, got directory: {summary_json_path}"
            )
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)

        observation_surface = (
            _read_json(_resolve_path(REPO_ROOT, str(args.observation_json)))
            if str(args.observation_json).strip()
            else None
        )
        observation_signature_sha256 = _observation_signature(
            observation_surface=observation_surface,
            observation_seed=int(args.observation_seed),
        )
        run_manifest_path = (
            _resolve_path(REPO_ROOT, str(args.run_manifest_json))
            if str(args.run_manifest_json).strip()
            else None
        )
        if run_manifest_path is None:
            triplet_gate = _build_internal_triplet_binding_gate(
                run_manifest_path=None,
                message=(
                    "run_manifest_json is required so same-checkpoint triplet evidence can prove checkpoint/controller binding"
                ),
                observation_seed=int(args.observation_seed),
                observation_signature_sha256=observation_signature_sha256,
            )
        else:
            try:
                run_manifest_payload = _read_json(run_manifest_path)
                triplet_gate = build_triplet_binding_gate(
                    run_manifest_payload=run_manifest_payload,
                    run_manifest_path=run_manifest_path,
                    output_dir=output_dir,
                    repo_root=REPO_ROOT,
                    declared_checkpoint_loaded=str(args.checkpoint_loaded),
                    observation_seed=int(args.observation_seed),
                    observation_signature_sha256=observation_signature_sha256,
                )
            except Exception as gate_exc:
                triplet_gate = _build_internal_triplet_binding_gate(
                    run_manifest_path=run_manifest_path,
                    message=_exception_message(gate_exc),
                    observation_seed=int(args.observation_seed),
                    observation_signature_sha256=observation_signature_sha256,
                )

        resolved_binding = triplet_gate.get("resolved_checkpoint_binding")
        resolved_binding_payload = (
            dict(resolved_binding) if isinstance(resolved_binding, Mapping) else {}
        )
        resolved_checkpoint_loaded_value = resolved_binding_payload.get(
            "checkpoint_loaded"
        )
        resolved_checkpoint_loaded = (
            str(resolved_checkpoint_loaded_value).strip()
            if resolved_checkpoint_loaded_value is not None
            else ""
        )
        if triplet_gate["formal_eligibility"] != "ALLOW":
            bundle = build_blocked_same_checkpoint_triplet_bundle(
                checkpoint_loaded=resolved_checkpoint_loaded
                or str(args.checkpoint_loaded),
                output_dir=output_dir,
                summary_json_path=summary_json_path,
                observation_seed=int(args.observation_seed),
                observation_surface=observation_surface,
                mode_sequence=cast(Sequence[object], args.mode_sequence),
                triplet_gate=cast(Mapping[str, object], triplet_gate),
            )
            summary = cast(dict[str, Any], bundle["summary"])
            failure_note_path = _write_failure_note(
                output_dir / FAILURE_NOTE_MARKDOWN_NAME,
                summary,
            )
            summary["failure_note_path"] = str(failure_note_path)
            _recompute_summary_signature(summary)
            _ = write_same_checkpoint_triplet_bundle(bundle)
            print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
            return 1

        mode_surface_by_mode = (
            cast(
                Mapping[str, Mapping[str, object]],
                _read_json(_resolve_path(REPO_ROOT, str(args.mode_surface_json))),
            )
            if str(args.mode_surface_json).strip()
            else None
        )

        runtime_metadata: dict[str, object] = {}
        if bool(args.policy_server_live):
            if mode_surface_by_mode is not None:
                raise ValueError(
                    "--policy-server-live cannot be combined with --mode-surface-json"
                )
            if observation_surface is None:
                raise ValueError(
                    "--policy-server-live requires --observation-json so the same observation surface can be replayed"
                )
            live_surfaces, runtime_metadata = collect_policy_server_mode_surfaces(
                observation=observation_surface,
                observation_seed=int(args.observation_seed),
                mode_sequence=cast(Sequence[object], args.mode_sequence),
                server_host=str(args.server_host),
                server_port=int(args.server_port),
                server_timeout_ms=int(args.server_timeout_ms),
            )
            mode_surface_by_mode = live_surfaces

        checkpoint_loaded = _normalize_live_server_checkpoint(
            checkpoint_loaded=resolved_checkpoint_loaded or str(args.checkpoint_loaded),
            server_provenance=cast(
                Mapping[str, object] | None,
                runtime_metadata.get("server_provenance") if runtime_metadata else None,
            ),
        )

        bundle = build_same_checkpoint_triplet_bundle(
            checkpoint_loaded=checkpoint_loaded,
            prompt_raw=str(args.prompt_raw),
            output_dir=output_dir,
            summary_json_path=summary_json_path,
            observation_seed=int(args.observation_seed),
            observation_surface=observation_surface,
            variant=str(args.variant),
            mode_sequence=cast(Sequence[object], args.mode_sequence),
            mode_surface_by_mode=mode_surface_by_mode,
        )
        if runtime_metadata:
            summary = cast(dict[str, Any], bundle["summary"])
            summary["runtime_metadata"] = _jsonable(runtime_metadata)
        summary = cast(dict[str, Any], bundle["summary"])
        summary["status"] = "PASS"
        summary["formal_eligibility"] = "ALLOW"
        summary["reason_code"] = OK_REASON_CODE
        summary["failure_note_path"] = None
        summary["triplet_gate"] = _jsonable(triplet_gate)
        _recompute_summary_signature(summary)
        _ = write_same_checkpoint_triplet_bundle(bundle)
        print(
            json.dumps(bundle["summary"], ensure_ascii=True, indent=2, sort_keys=True)
        )
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
