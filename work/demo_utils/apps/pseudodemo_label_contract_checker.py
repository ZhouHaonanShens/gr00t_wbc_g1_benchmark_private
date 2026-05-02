#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any, NoReturn


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_CONTRACT_PATH = Path("agent/exchange/pseudodemo_label_contract_v1.md")

SPEC_START_MARKER = "<!-- PSEUDODEMO_LABEL_CONTRACT_SPEC_START -->"
SPEC_END_MARKER = "<!-- PSEUDODEMO_LABEL_CONTRACT_SPEC_END -->"
PASS_SENTINEL = "PSEUDODEMO_LABEL_CONTRACT_PASS"
FAIL_SENTINEL = "PSEUDODEMO_LABEL_CONTRACT_FAIL"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify pseudodemo_label_contract_v1 coverage and optionally validate "
            "a normalized pseudodemo label candidate."
        )
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Markdown contract file that contains the embedded JSON spec block.",
    )
    parser.add_argument(
        "--candidate-json",
        type=str,
        default=None,
        help="Optional inline JSON candidate record that should satisfy the contract.",
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=None,
        help="Optional JSON file containing a normalized candidate record.",
    )
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="Print the parsed contract spec JSON after validation.",
    )
    return parser


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    print(FAIL_SENTINEL, file=sys.stderr)
    raise SystemExit(1)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"failed to read contract file {path}: {exc}")


def _extract_spec_json(markdown_text: str) -> dict[str, Any]:
    pattern = re.compile(
        re.escape(SPEC_START_MARKER)
        + r"\s*```json\s*(\{.*?\})\s*```\s*"
        + re.escape(SPEC_END_MARKER),
        re.DOTALL,
    )
    match = pattern.search(markdown_text)
    if match is None:
        _fail("contract markdown is missing the embedded JSON spec block")
    raw_json = match.group(1)
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        _fail(f"embedded contract spec JSON is invalid: {exc}")
    if not isinstance(payload, dict):
        _fail("embedded contract spec must be a JSON object")
    return dict(payload)


def _require_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be an object, got {type(value).__name__}")
    return dict(value)


def _require_list(
    value: Any, *, field_name: str, expected_len: int | None = None
) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field_name} must be a list, got {type(value).__name__}")
    result = list(value)
    if expected_len is not None and len(result) != int(expected_len):
        _fail(f"{field_name} must have length {expected_len}, got {len(result)}")
    return result


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        _fail(f"{field_name} must be a non-empty string")
    return normalized


def _require_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _require_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_string_list(value: Any, *, field_name: str) -> list[str]:
    result = _require_list(value, field_name=field_name)
    return [
        _require_non_empty_string(item, field_name=f"{field_name}[]") for item in result
    ]


def _check_markdown_coverage(markdown_text: str, spec: Mapping[str, Any]) -> None:
    required_fields = _as_string_list(
        spec.get("required_fields"), field_name="required_fields"
    )
    analysis_only_fields = _as_string_list(
        spec.get("analysis_only_fields"), field_name="analysis_only_fields"
    )
    invariants = _require_list(spec.get("invariants"), field_name="invariants")
    if len(required_fields) != 15:
        _fail(
            f"required_fields must contain 15 explicit names, got {len(required_fields)}"
        )
    if len(analysis_only_fields) != 6:
        _fail(
            f"analysis_only_fields must contain 6 explicit names, got {len(analysis_only_fields)}"
        )
    if len(invariants) < 5:
        _fail(f"contract must define at least 5 invariants, got {len(invariants)}")

    required_phrases = (
        "truthful teacher target",
        "reset_boundary=no_cross_episode",
        "snapshot_family ↔ source_bucket",
        "deployable / metadata / analysis-only 边界",
        "m2_label",
    )
    for phrase in required_phrases:
        if phrase not in markdown_text:
            _fail(f"contract markdown is missing required phrase: {phrase!r}")

    for field_name in required_fields + analysis_only_fields:
        if f"`{field_name}`" not in markdown_text:
            _fail(
                f"contract markdown does not mention field {field_name!r} in backticks"
            )

    for index, raw_row in enumerate(invariants, start=1):
        row = _require_dict(raw_row, field_name=f"invariants[{index - 1}]")
        _require_non_empty_string(
            row.get("id"), field_name=f"invariants[{index - 1}].id"
        )
        _require_non_empty_string(
            row.get("rule"), field_name=f"invariants[{index - 1}].rule"
        )
        _require_dict(
            row.get("violation_example"),
            field_name=f"invariants[{index - 1}].violation_example",
        )


def _validate_contract_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = _require_non_empty_string(
        spec.get("schema_version"), field_name="schema_version"
    )
    if schema_version != "pseudodemo_label_contract_v1":
        _fail(f"schema_version mismatch: {schema_version!r}")

    required_fields = _as_string_list(
        spec.get("required_fields"), field_name="required_fields"
    )
    analysis_only_fields = _as_string_list(
        spec.get("analysis_only_fields"), field_name="analysis_only_fields"
    )
    allowed_snapshot_families = _as_string_list(
        spec.get("allowed_snapshot_families"), field_name="allowed_snapshot_families"
    )
    allowed_source_buckets = _as_string_list(
        spec.get("allowed_source_buckets"), field_name="allowed_source_buckets"
    )
    family_map = _require_dict(
        spec.get("snapshot_family_to_source_bucket"),
        field_name="snapshot_family_to_source_bucket",
    )
    if sorted(family_map.keys()) != sorted(allowed_snapshot_families):
        _fail(
            "snapshot_family_to_source_bucket keys must match allowed_snapshot_families"
        )
    for family, source_bucket in family_map.items():
        if source_bucket not in allowed_source_buckets:
            _fail(
                f"snapshot_family_to_source_bucket[{family!r}] points to unknown source bucket {source_bucket!r}"
            )

    projection = _require_dict(
        spec.get("deployable_projection"), field_name="deployable_projection"
    )
    projection_fields = _as_string_list(
        projection.get("allowed_fields"),
        field_name="deployable_projection.allowed_fields",
    )
    if projection_fields != ["history_payload", "valid_mask", "phase", "mode"]:
        _fail(
            "deployable_projection.allowed_fields must freeze to ['history_payload', 'valid_mask', 'phase', 'mode']"
        )

    truth_spec = _require_dict(
        spec.get("truthful_teacher_target"), field_name="truthful_teacher_target"
    )
    if not bool(truth_spec.get("must_reference_real_teacher_rollout", False)):
        _fail(
            "truthful_teacher_target.must_reference_real_teacher_rollout must be true"
        )
    if not bool(truth_spec.get("forbid_synthetic_observation_only_backfill", False)):
        _fail(
            "truthful_teacher_target.forbid_synthetic_observation_only_backfill must be true"
        )

    m2_spec = _require_dict(spec.get("m2_label_rule"), field_name="m2_label_rule")
    if not bool(m2_spec.get("key_must_exist", False)):
        _fail("m2_label_rule.key_must_exist must be true")
    if not bool(m2_spec.get("allow_null", False)):
        _fail("m2_label_rule.allow_null must be true")

    reset_spec = _require_dict(spec.get("reset_boundary"), field_name="reset_boundary")
    if reset_spec.get("value") != "no_cross_episode":
        _fail("reset_boundary.value must be 'no_cross_episode'")
    if bool(reset_spec.get("cross_episode_history_allowed", True)):
        _fail("reset_boundary.cross_episode_history_allowed must be false")

    history_window_length = _require_int(
        spec.get("history_window_length"), field_name="history_window_length"
    )
    if history_window_length != 8:
        _fail("history_window_length is frozen at 8")

    return {
        "schema_version": schema_version,
        "required_fields": required_fields,
        "analysis_only_fields": analysis_only_fields,
        "family_map": dict(family_map),
        "allowed_source_buckets": allowed_source_buckets,
        "allowed_snapshot_families": allowed_snapshot_families,
        "history_window_length": history_window_length,
        "history_payload_required_keys": _as_string_list(
            spec.get("history_payload_required_keys"),
            field_name="history_payload_required_keys",
        ),
        "phase_vocab": _as_string_list(
            spec.get("phase_vocab"), field_name="phase_vocab"
        ),
        "mode_vocab": _as_string_list(spec.get("mode_vocab"), field_name="mode_vocab"),
        "label_kind_allowlist": _as_string_list(
            spec.get("label_kind_allowlist"), field_name="label_kind_allowlist"
        ),
        "analysis_only_container_key": _require_non_empty_string(
            spec.get("analysis_only_container_key"),
            field_name="analysis_only_container_key",
        ),
        "deployable_payload_container_key": _require_non_empty_string(
            spec.get("deployable_payload_container_key"),
            field_name="deployable_payload_container_key",
        ),
        "deployable_projection": projection,
        "truthful_teacher_target": truth_spec,
    }


def _load_candidate(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.candidate_json and args.candidate_file is not None:
        _fail("use only one of --candidate-json or --candidate-file")
    if args.candidate_json:
        payload: object
        try:
            payload = json.loads(args.candidate_json)
        except json.JSONDecodeError as exc:
            _fail(f"candidate JSON is invalid: {exc}")
        return _require_dict(payload, field_name="candidate")
    if args.candidate_file is not None:
        payload = None
        try:
            payload = json.loads(args.candidate_file.read_text(encoding="utf-8"))
        except OSError as exc:
            _fail(f"failed to read candidate file {args.candidate_file}: {exc}")
        except json.JSONDecodeError as exc:
            _fail(f"candidate file JSON is invalid: {exc}")
        return _require_dict(payload, field_name="candidate")
    return None


def _validate_history_payload(
    history_payload: object,
    valid_mask: Sequence[object],
    *,
    spec: Mapping[str, Any],
) -> None:
    history = _require_dict(history_payload, field_name="history_payload")
    expected_keys = list(spec["history_payload_required_keys"])
    observed_keys = sorted(history.keys())
    if sorted(expected_keys) != observed_keys:
        _fail(
            "history_payload must contain exactly these keys: "
            + ", ".join(expected_keys)
        )
    expected_len = int(spec["history_window_length"])
    mask = _require_list(valid_mask, field_name="valid_mask", expected_len=expected_len)
    for key in expected_keys:
        lane = _require_list(
            history.get(key),
            field_name=f"history_payload.{key}",
            expected_len=expected_len,
        )
        for index, is_valid in enumerate(mask):
            if not isinstance(is_valid, bool):
                _fail(f"valid_mask[{index}] must be a bool")
            if bool(is_valid) and lane[index] is None:
                _fail(
                    f"history_payload.{key}[{index}] must not be null when valid_mask[{index}] is true"
                )


def _validate_teacher_target(candidate: Mapping[str, Any]) -> None:
    teacher_policy_id = _require_non_empty_string(
        candidate.get("teacher_policy_id"), field_name="teacher_policy_id"
    )
    teacher_target = _require_dict(
        candidate.get("teacher_target"), field_name="teacher_target"
    )
    trace_episode_id = _require_non_empty_string(
        teacher_target.get("trace_episode_id"),
        field_name="teacher_target.trace_episode_id",
    )
    _ = trace_episode_id
    trace_t_range = _require_list(
        teacher_target.get("trace_t_range"),
        field_name="teacher_target.trace_t_range",
        expected_len=2,
    )
    start_t = _require_int(
        trace_t_range[0], field_name="teacher_target.trace_t_range[0]"
    )
    end_t = _require_int(trace_t_range[1], field_name="teacher_target.trace_t_range[1]")
    if end_t < start_t:
        _fail("teacher_target.trace_t_range must satisfy end >= start")
    producer = _require_non_empty_string(
        teacher_target.get("producer"), field_name="teacher_target.producer"
    )
    if producer != teacher_policy_id:
        _fail("teacher_policy_id must match teacher_target.producer")
    synthetic_flag = _require_bool(
        teacher_target.get("synthetic_observation_only_backfill"),
        field_name="teacher_target.synthetic_observation_only_backfill",
    )
    if synthetic_flag:
        _fail("teacher_target must not be synthetic observation-only backfill")


def _validate_analysis_only(
    candidate: Mapping[str, Any], *, spec: Mapping[str, Any]
) -> None:
    container_key = str(spec["analysis_only_container_key"])
    if container_key not in candidate:
        return
    analysis_only = _require_dict(
        candidate.get(container_key), field_name=container_key
    )
    allowed = set(spec["analysis_only_fields"])
    extra = sorted(set(analysis_only.keys()) - allowed)
    if extra:
        _fail(f"analysis_only contains unknown fields: {extra}")


def _validate_deployable_payload(
    candidate: Mapping[str, Any], *, spec: Mapping[str, Any]
) -> None:
    container_key = str(spec["deployable_payload_container_key"])
    if container_key not in candidate:
        return
    deployable_payload = _require_dict(
        candidate.get(container_key), field_name=container_key
    )
    allowed_fields = set(spec["deployable_projection"]["allowed_fields"])
    forbidden_exact = set(spec["deployable_projection"]["forbidden_exact_fields"])
    forbidden_prefixes = tuple(spec["deployable_projection"]["forbidden_prefixes"])

    for field_name in deployable_payload:
        if field_name in forbidden_exact:
            _fail(
                f"analysis-only or metadata field leaked into deployable_payload: {field_name}"
            )
        if any(str(field_name).startswith(prefix) for prefix in forbidden_prefixes):
            _fail(
                f"forbidden prefixed field leaked into deployable_payload: {field_name}"
            )
        if field_name not in allowed_fields:
            _fail(f"deployable_payload field is not allowlisted: {field_name}")

    missing = sorted(allowed_fields - set(deployable_payload.keys()))
    if missing:
        _fail(f"deployable_payload is missing frozen fields: {missing}")


def _validate_candidate(
    candidate: Mapping[str, Any], *, spec: Mapping[str, Any]
) -> dict[str, Any]:
    required_fields = list(spec["required_fields"])
    missing_fields = [
        field_name for field_name in required_fields if field_name not in candidate
    ]
    if missing_fields:
        _fail("candidate is missing required fields: " + ", ".join(missing_fields))

    _require_non_empty_string(candidate.get("snapshot_id"), field_name="snapshot_id")
    snapshot_family = _require_non_empty_string(
        candidate.get("snapshot_family"), field_name="snapshot_family"
    )
    if snapshot_family not in set(spec["allowed_snapshot_families"]):
        _fail(f"snapshot_family must be one of {spec['allowed_snapshot_families']!r}")
    source_bucket = _require_non_empty_string(
        candidate.get("source_bucket"), field_name="source_bucket"
    )
    if source_bucket not in set(spec["allowed_source_buckets"]):
        _fail(f"source_bucket must be one of {spec['allowed_source_buckets']!r}")
    expected_bucket = spec["family_map"][snapshot_family]
    if source_bucket != expected_bucket:
        _fail(
            f"snapshot_family {snapshot_family!r} must map to source_bucket {expected_bucket!r}, got {source_bucket!r}"
        )

    _require_non_empty_string(
        candidate.get("source_episode_id"), field_name="source_episode_id"
    )
    _require_int(candidate.get("source_t"), field_name="source_t")

    valid_mask = _require_list(
        candidate.get("valid_mask"),
        field_name="valid_mask",
        expected_len=int(spec["history_window_length"]),
    )
    _validate_history_payload(candidate.get("history_payload"), valid_mask, spec=spec)

    reset_boundary = _require_non_empty_string(
        candidate.get("reset_boundary"), field_name="reset_boundary"
    )
    if reset_boundary != "no_cross_episode":
        _fail("reset_boundary must equal 'no_cross_episode'")

    phase = _require_non_empty_string(candidate.get("phase"), field_name="phase")
    if phase not in set(spec["phase_vocab"]):
        _fail(f"phase must be one of {spec['phase_vocab']!r}")
    mode = _require_non_empty_string(candidate.get("mode"), field_name="mode")
    if mode not in set(spec["mode_vocab"]):
        _fail(f"mode must be one of {spec['mode_vocab']!r}")

    _validate_teacher_target(candidate)

    label_kind = _require_non_empty_string(
        candidate.get("label_kind"), field_name="label_kind"
    )
    if label_kind not in set(spec["label_kind_allowlist"]):
        _fail(f"label_kind must be one of {spec['label_kind_allowlist']!r}")

    if "m2_label" not in candidate:
        _fail("m2_label key must exist even when value is null")
    _require_non_empty_string(
        candidate.get("dataset_version"), field_name="dataset_version"
    )

    _validate_analysis_only(candidate, spec=spec)
    _validate_deployable_payload(candidate, spec=spec)

    return {
        "snapshot_family": snapshot_family,
        "source_bucket": source_bucket,
        "phase": phase,
        "mode": mode,
        "has_analysis_only": str(spec["analysis_only_container_key"]) in candidate,
        "has_deployable_payload": str(spec["deployable_payload_container_key"])
        in candidate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    markdown_text = _read_text(Path(args.contract))
    spec_json = _extract_spec_json(markdown_text)
    _check_markdown_coverage(markdown_text, spec_json)
    spec = _validate_contract_spec(spec_json)
    candidate = _load_candidate(args)

    result: dict[str, Any] = {
        "status": "PASS",
        "schema_version": str(spec["schema_version"]),
        "required_field_count": len(spec["required_fields"]),
        "analysis_only_field_count": len(spec["analysis_only_fields"]),
        "invariant_count": len(spec_json["invariants"]),
    }
    if candidate is not None:
        result["candidate"] = _validate_candidate(candidate, spec=spec)
    if args.print_spec:
        result["spec"] = spec_json

    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    print(PASS_SENTINEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
