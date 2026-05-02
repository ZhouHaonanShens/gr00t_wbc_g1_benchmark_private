from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "interface_localization_sprint"

SPLIT_AUDIT_JSON_NAME = "right_arm_vs_right_hand_split_audit.json"
SPLIT_AUDIT_SCHEMA_VERSION = "interface_localization_right_hand_split_audit_v1"
SPLIT_AUDIT_ARTIFACT_KIND = "right_arm_vs_right_hand_split_audit"

WATCH_BUCKET_ORDER: tuple[str, ...] = (
    "body_wrist_upper_limb_chain",
    "dex3_finger_hand_path",
)
TRACE_BLOCKED_STATUS = "blocked_missing_upstream"

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import interface_localization_action_roundtrip
from work.recap import interface_localization_contract
from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface_localization_right_hand_split.py",
        description=(
            "Audit the right-arm wrist versus right-hand/Dex3 split so structure-level "
            "ownership is not misread as decoder failure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, the split audit JSON is written "
            "to <artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for the generated split audit JSON.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    raw_output_dir = str(args.output_dir).strip()
    if raw_output_dir:
        return state_conditioned_bucket_a_import.validate_output_dir(
            _resolve_path(repo_root, raw_output_dir)
        )
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    return state_conditioned_bucket_a_import.validate_output_dir(
        artifact_dir / DEFAULT_OUTPUT_SUBDIR
    )


def _generation_command_for(output_dir: Path, repo_root: Path) -> str:
    display_output_dir = _relpath(repo_root, output_dir)
    return (
        "python3 work/recap/scripts/interface_localization_right_hand_split.py "
        f"--output-dir {display_output_dir}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return cast(dict[str, Any], payload)


def _read_trace_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {str(key): str(value) for key, value in row.items() if key is not None}
            )
    return rows


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _as_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _as_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, field_name=field_name)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_string_list(value: object, *, field_name: str) -> list[str]:
    return [
        _as_string(item, field_name=f"{field_name}[]")
        for item in _as_list(value, field_name=field_name)
    ]


def _load_contract_payload() -> dict[str, Any]:
    contract = interface_localization_contract.build_interface_localization_contract()
    _ = interface_localization_contract.assert_contract_matches_canonical(contract)
    return contract


def _baseline_summary(contract_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_schema_version": str(contract_payload["schema_version"]),
        "baseline_tuple": dict(contract_payload["baseline_tuple"]),
        "status_allowlist": list(contract_payload["status_ontology"]["legal_statuses"]),
        "provenance_class_allowlist": list(
            contract_payload["provenance_classes"]["class_order"]
        ),
        "mainline_task_text_field": str(
            contract_payload["advantage_contract_facts"]["mainline_task_text_field"]
        ),
    }


def _action_roundtrip_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "agent"
        / "artifacts"
        / DEFAULT_OUTPUT_SUBDIR
        / interface_localization_action_roundtrip.ACTION_ROUNDTRIP_JSON_NAME
    )


def _interface_trace_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "agent"
        / "artifacts"
        / DEFAULT_OUTPUT_SUBDIR
        / "interface_trace.csv"
    )


def _task18_pack_path(repo_root: Path) -> Path:
    return repo_root / ".sisyphus" / "evidence" / "task-18-attribution-pack.json"


def _roundtrip_bucket_payload(
    roundtrip_payload: Mapping[str, Any], *, bucket_name: str
) -> Mapping[str, Any]:
    return _as_mapping(
        _as_mapping(
            roundtrip_payload.get("watch_buckets"), field_name="watch_buckets"
        ).get(bucket_name),
        field_name=f"watch_buckets.{bucket_name}",
    )


def _trace_rows_for_bucket(
    trace_rows: Sequence[Mapping[str, str]], *, bucket_name: str
) -> list[Mapping[str, str]]:
    return [row for row in trace_rows if str(row.get("watch_bucket")) == bucket_name]


def _validate_roundtrip_split(roundtrip_payload: Mapping[str, Any]) -> None:
    actual_order = _as_string_list(
        roundtrip_payload.get("watch_bucket_order"), field_name="watch_bucket_order"
    )
    if actual_order != list(WATCH_BUCKET_ORDER):
        raise ValueError(
            "watch_bucket_order drift: expected "
            f"{list(WATCH_BUCKET_ORDER)}, got {actual_order}"
        )

    expected_action_keys = {
        "body_wrist_upper_limb_chain": "right_arm",
        "dex3_finger_hand_path": "right_hand",
    }
    expected_focus_joints = {
        "body_wrist_upper_limb_chain": [
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        "dex3_finger_hand_path": [
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
        ],
    }

    for bucket_name in WATCH_BUCKET_ORDER:
        bucket_payload = _roundtrip_bucket_payload(
            roundtrip_payload, bucket_name=bucket_name
        )
        source_mapping = _as_mapping(
            bucket_payload.get("source_mapping"),
            field_name=f"{bucket_name}.source_mapping",
        )
        action_key = _as_string(
            source_mapping.get("original_action_key"),
            field_name=f"{bucket_name}.source_mapping.original_action_key",
        )
        if action_key != expected_action_keys[bucket_name]:
            raise ValueError(
                f"{bucket_name} must stay bound to {expected_action_keys[bucket_name]}, got {action_key}"
            )
        joint_order = _as_string_list(
            source_mapping.get("joint_order"),
            field_name=f"{bucket_name}.source_mapping.joint_order",
        )
        focus_joints = expected_focus_joints[bucket_name]
        if bucket_name == "body_wrist_upper_limb_chain":
            if joint_order[-3:] != focus_joints:
                raise ValueError(
                    f"right_arm wrist split drift: expected trailing wrist joints {focus_joints}, got {joint_order[-3:]}"
                )
        else:
            if joint_order != focus_joints:
                raise ValueError(
                    f"right_hand finger split drift: expected finger-only joint order {focus_joints}, got {joint_order}"
                )


def _build_source_evidence(bucket_payload: Mapping[str, Any]) -> dict[str, Any]:
    source_mapping = _as_mapping(
        bucket_payload.get("source_mapping"), field_name="source_mapping"
    )
    action_key = _as_string(
        source_mapping.get("original_action_key"),
        field_name="source_mapping.original_action_key",
    )
    joint_order = _as_string_list(
        source_mapping.get("joint_order"), field_name="source_mapping.joint_order"
    )
    focus_joints = joint_order[-3:] if action_key == "right_arm" else joint_order
    return {
        "status": "survived",
        "provenance_class": "static",
        "action_key": action_key,
        "state_key": _as_string(
            source_mapping.get("original_state_key"),
            field_name="source_mapping.original_state_key",
        ),
        "reference_state_key": _as_optional_string(
            source_mapping.get("reference_state_key"),
            field_name="source_mapping.reference_state_key",
        ),
        "joint_order": joint_order,
        "boundary_focus_joints": list(focus_joints),
        "ownership_scope": _as_string(
            source_mapping.get("ownership_scope"),
            field_name="source_mapping.ownership_scope",
        ),
        "ownership_rule": _as_string(
            source_mapping.get("ownership_rule"),
            field_name="source_mapping.ownership_rule",
        ),
        "mapping_refs": list(
            _as_list(
                source_mapping.get("mapping_refs"),
                field_name="source_mapping.mapping_refs",
            )
        ),
    }


def _trace_surface_summary(bucket_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    observed_rows = [
        row for row in bucket_rows if str(row.get("status")) != TRACE_BLOCKED_STATUS
    ]
    blocked_rows = [
        row for row in bucket_rows if str(row.get("status")) == TRACE_BLOCKED_STATUS
    ]
    blocked_reason_by_field = {
        str(row.get("field_name")): str(row.get("blocked_reason"))
        for row in blocked_rows
        if str(row.get("field_name"))
    }
    provenance_counts = Counter(str(row.get("provenance_class")) for row in bucket_rows)
    return {
        "trace_row_count": int(len(bucket_rows)),
        "observed_field_names": sorted(
            {
                str(row.get("field_name"))
                for row in observed_rows
                if str(row.get("field_name"))
            }
        ),
        "blocked_field_names": sorted(blocked_reason_by_field.keys()),
        "blocked_reason_by_field": blocked_reason_by_field,
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "replay_live_field_names": sorted(
            {
                str(row.get("field_name"))
                for row in observed_rows
                if str(row.get("provenance_class")) == "replay_live"
                and str(row.get("field_name"))
            }
        ),
        "synthetic_field_names": sorted(
            {
                str(row.get("field_name"))
                for row in observed_rows
                if str(row.get("provenance_class")) == "synthetic"
                and str(row.get("field_name"))
            }
        ),
        "server_live_blocked_field_names": sorted(
            {
                str(row.get("field_name"))
                for row in blocked_rows
                if str(row.get("provenance_class")) == "server_live"
                and str(row.get("field_name"))
            }
        ),
    }


def _task18_finding(
    task18_payload: Mapping[str, Any], *, finding_code: str
) -> dict[str, str] | None:
    findings = _as_list(
        task18_payload.get("shared_structural_findings"),
        field_name="task18.shared_structural_findings",
    )
    for finding in findings:
        finding_mapping = _as_mapping(
            finding, field_name="task18.shared_structural_findings[]"
        )
        if (
            _as_string(
                finding_mapping.get("finding_code"),
                field_name="task18.shared_structural_findings[].finding_code",
            )
            == finding_code
        ):
            return {
                "finding_code": finding_code,
                "summary": _as_string(
                    finding_mapping.get("summary"),
                    field_name=f"task18.{finding_code}.summary",
                ),
            }
    return None


def _build_telemetry_evidence(
    bucket_name: str,
    bucket_payload: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, str]],
    task18_payload: Mapping[str, Any],
) -> dict[str, Any]:
    conclusions = _as_mapping(
        bucket_payload.get("conclusions"), field_name="conclusions"
    )
    checkpoints = _as_mapping(
        bucket_payload.get("checkpoints"), field_name="checkpoints"
    )
    task18_match: dict[str, str] | None = None
    if bucket_name == "dex3_finger_hand_path":
        task18_match = _task18_finding(
            task18_payload, finding_code="shared_model_insensitive_groups"
        )
    return {
        "action_chain_watch_bucket": {
            "status": "survived",
            "provenance_class": "synthetic",
            "watch_bucket_classification": _as_string(
                conclusions.get("watch_bucket_classification"),
                field_name="conclusions.watch_bucket_classification",
            ),
            "difference_disappeared_at": _as_optional_string(
                conclusions.get("difference_disappeared_at"),
                field_name="conclusions.difference_disappeared_at",
            ),
            "controller_absorbed_upstream_difference": _as_bool(
                conclusions.get("controller_absorbed_upstream_difference"),
                field_name="conclusions.controller_absorbed_upstream_difference",
            ),
            "model_insensitive": _as_bool(
                conclusions.get("model_insensitive"),
                field_name="conclusions.model_insensitive",
            ),
            "stage_l2_differences": {
                "raw_action": float(
                    _as_mapping(
                        checkpoints.get("produced"), field_name="checkpoints.produced"
                    )["l2_difference"]
                ),
                "absolute_action": float(
                    _as_mapping(
                        checkpoints.get("adapted"), field_name="checkpoints.adapted"
                    )["l2_difference"]
                ),
                "controller_input": float(
                    _as_mapping(
                        checkpoints.get("consumed"), field_name="checkpoints.consumed"
                    )["l2_difference"]
                ),
            },
        },
        "repo_local_trace_surfaces": {
            "status": "survived",
            "provenance_class": "replay_live",
            **_trace_surface_summary(trace_rows),
        },
        "task18_shared_finding": task18_match,
    }


def _build_upstream_source_route_state(bucket_name: str) -> dict[str, Any]:
    if bucket_name == "body_wrist_upper_limb_chain":
        return {
            "status": "survived",
            "provenance_class": "static",
            "route_scope": "right_arm wrist/body chain ownership route",
            "repo_local_telemetry_retained": True,
            "summary": (
                "Repo-local ownership evidence already binds wrist roll/pitch/yaw to "
                "right_arm, and the local telemetry stays split from right_hand."
            ),
            "blocked_reason": None,
        }
    return {
        "status": TRACE_BLOCKED_STATUS,
        "provenance_class": "static",
        "route_scope": "Dex3 or finger-level upstream source route under right_hand",
        "repo_local_telemetry_retained": True,
        "summary": (
            "Repo-local artifacts prove right_hand finger/thumb ownership and preserve local "
            "telemetry, but the current checkout does not prove a Dex3-specific upstream code "
            "route below the wrist boundary."
        ),
        "blocked_reason": (
            "Missing repo-local Dex3/finger-level upstream source-route proof; retain the "
            "right_hand telemetry evidence and mark only the source-route attribution as "
            "blocked_missing_upstream."
        ),
    }


def _bucket_summary(
    bucket_name: str,
    source_evidence: Mapping[str, Any],
    telemetry_evidence: Mapping[str, Any],
    upstream_source_route_state: Mapping[str, Any],
) -> dict[str, Any]:
    action_chain_watch_bucket = _as_mapping(
        telemetry_evidence.get("action_chain_watch_bucket"),
        field_name=f"{bucket_name}.telemetry_evidence.action_chain_watch_bucket",
    )
    return {
        "status": "survived",
        "provenance_class": "static",
        "ownership_binding": _as_string(
            source_evidence.get("action_key"),
            field_name=f"{bucket_name}.source_evidence.action_key",
        ),
        "boundary_focus_joints": list(
            _as_string_list(
                source_evidence.get("boundary_focus_joints"),
                field_name=f"{bucket_name}.source_evidence.boundary_focus_joints",
            )
        ),
        "model_insensitive": _as_bool(
            action_chain_watch_bucket.get("model_insensitive"),
            field_name=f"{bucket_name}.telemetry_evidence.action_chain_watch_bucket.model_insensitive",
        ),
        "watch_bucket_classification": _as_string(
            action_chain_watch_bucket.get("watch_bucket_classification"),
            field_name=f"{bucket_name}.telemetry_evidence.action_chain_watch_bucket.watch_bucket_classification",
        ),
        "upstream_source_route_status": _as_string(
            upstream_source_route_state.get("status"),
            field_name=f"{bucket_name}.upstream_source_route_state.status",
        ),
        "guardrail": (
            "Do not treat this bucket as part of right_hand finger ownership."
            if bucket_name == "body_wrist_upper_limb_chain"
            else "Do not upgrade right_hand model-insensitive telemetry into Dex3 runtime-route certainty."
        ),
    }


def build_right_arm_vs_right_hand_split_audit(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    roundtrip_payload: Mapping[str, Any] | None = None,
    trace_rows: Sequence[Mapping[str, str]] | None = None,
    task18_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract_payload = _load_contract_payload()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR).resolve()
    )
    generation_command = _generation_command_for(resolved_output_dir, repo_root)

    resolved_roundtrip_payload = (
        dict(roundtrip_payload)
        if roundtrip_payload is not None
        else _read_json(_action_roundtrip_path(repo_root))
    )
    resolved_trace_rows = (
        [dict(row) for row in trace_rows]
        if trace_rows is not None
        else _read_trace_rows(_interface_trace_path(repo_root))
    )
    resolved_task18_payload = (
        dict(task18_payload)
        if task18_payload is not None
        else _read_json(_task18_pack_path(repo_root))
    )

    _validate_roundtrip_split(resolved_roundtrip_payload)

    body_bucket = _roundtrip_bucket_payload(
        resolved_roundtrip_payload, bucket_name="body_wrist_upper_limb_chain"
    )
    dex_bucket = _roundtrip_bucket_payload(
        resolved_roundtrip_payload, bucket_name="dex3_finger_hand_path"
    )

    body_source_evidence = _build_source_evidence(body_bucket)
    dex_source_evidence = _build_source_evidence(dex_bucket)
    body_telemetry_evidence = _build_telemetry_evidence(
        "body_wrist_upper_limb_chain",
        body_bucket,
        _trace_rows_for_bucket(
            resolved_trace_rows, bucket_name="body_wrist_upper_limb_chain"
        ),
        resolved_task18_payload,
    )
    dex_telemetry_evidence = _build_telemetry_evidence(
        "dex3_finger_hand_path",
        dex_bucket,
        _trace_rows_for_bucket(
            resolved_trace_rows, bucket_name="dex3_finger_hand_path"
        ),
        resolved_task18_payload,
    )
    body_upstream_route = _build_upstream_source_route_state(
        "body_wrist_upper_limb_chain"
    )
    dex_upstream_route = _build_upstream_source_route_state("dex3_finger_hand_path")

    body_bucket_audit = {
        "status": "survived",
        "provenance_class": "static",
        "source_evidence": body_source_evidence,
        "telemetry_evidence": body_telemetry_evidence,
        "upstream_source_route_state": body_upstream_route,
        "audit_summary": _bucket_summary(
            "body_wrist_upper_limb_chain",
            body_source_evidence,
            body_telemetry_evidence,
            body_upstream_route,
        ),
    }
    dex_bucket_audit = {
        "status": "survived",
        "provenance_class": "static",
        "source_evidence": dex_source_evidence,
        "telemetry_evidence": dex_telemetry_evidence,
        "upstream_source_route_state": dex_upstream_route,
        "audit_summary": _bucket_summary(
            "dex3_finger_hand_path",
            dex_source_evidence,
            dex_telemetry_evidence,
            dex_upstream_route,
        ),
    }

    model_insensitive_buckets: list[str] = []
    blocked_upstream_source_route_buckets: list[str] = []
    for bucket_name, bucket_payload in (
        ("body_wrist_upper_limb_chain", body_bucket_audit),
        ("dex3_finger_hand_path", dex_bucket_audit),
    ):
        audit_summary = _as_mapping(
            bucket_payload.get("audit_summary"),
            field_name=f"{bucket_name}.audit_summary",
        )
        upstream_route_state = _as_mapping(
            bucket_payload.get("upstream_source_route_state"),
            field_name=f"{bucket_name}.upstream_source_route_state",
        )
        if _as_bool(
            audit_summary.get("model_insensitive"),
            field_name=f"{bucket_name}.audit_summary.model_insensitive",
        ):
            model_insensitive_buckets.append(bucket_name)
        if (
            _as_string(
                upstream_route_state.get("status"),
                field_name=f"{bucket_name}.upstream_source_route_state.status",
            )
            == TRACE_BLOCKED_STATUS
        ):
            blocked_upstream_source_route_buckets.append(bucket_name)

    return {
        "schema_version": SPLIT_AUDIT_SCHEMA_VERSION,
        "artifact_kind": SPLIT_AUDIT_ARTIFACT_KIND,
        "status": "survived",
        "provenance_class": "static",
        "generation_command": generation_command,
        "input_baseline_summary": _baseline_summary(contract_payload),
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_right_hand_split.py",
            "expected_task4_roundtrip_json": str(_action_roundtrip_path(repo_root)),
            "expected_task5_trace_csv": str(_interface_trace_path(repo_root)),
            "expected_task18_attribution_pack_json": str(_task18_pack_path(repo_root)),
            "pytest_command": "python3 -m pytest tests/recap/test_right_arm_vs_right_hand_split.py -q",
        },
        "source_refs": {
            "wbc_env_contract": "agent/exchange/wbc_env_io.md:77-83",
            "telemetry_term_test": "tests/recap/test_gr00t_action_chain_telemetry.py:120-142",
            "task18_shared_findings": ".sisyphus/evidence/task-18-attribution-pack.json:203-219",
            "teacher_action_chunk_helper": "work/recap/scripts/state_conditioned_snapshot_harvest.py:1676-1695",
            "controller_limit_table": "work/recap/scripts/gr00t_action_chain_telemetry.py:77-95",
            "task4_roundtrip_artifact": "agent/artifacts/interface_localization_sprint/action_chain_watchlist_split.json",
            "task5_trace_csv": "agent/artifacts/interface_localization_sprint/interface_trace.csv",
        },
        "summary": {
            "ownership_binding_by_bucket": {
                "body_wrist_upper_limb_chain": body_source_evidence["action_key"],
                "dex3_finger_hand_path": dex_source_evidence["action_key"],
            },
            "model_insensitive_buckets": model_insensitive_buckets,
            "blocked_upstream_source_route_buckets": blocked_upstream_source_route_buckets,
            "guardrail": (
                "Keep wrist ownership on right_arm and finger/thumb ownership on right_hand; "
                "right_hand model-insensitive telemetry is repo-local evidence, not Dex3 route certainty."
            ),
        },
        "body_wrist_upper_limb_chain": body_bucket_audit,
        "dex3_finger_hand_path": dex_bucket_audit,
    }


def write_artifact(*, output_dir: Path, payload: Mapping[str, Any]) -> Path:
    return state_conditioned_bucket_a_import._write_json(
        output_dir / SPLIT_AUDIT_JSON_NAME,
        payload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output_dir = resolve_output_dir(REPO_ROOT, args)
        payload = build_right_arm_vs_right_hand_split_audit(
            REPO_ROOT,
            output_dir=output_dir,
        )
        artifact_path = write_artifact(output_dir=output_dir, payload=payload)
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "output_dir": str(output_dir),
                    "split_audit_json": str(artifact_path),
                }
            ),
            end="",
        )
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
