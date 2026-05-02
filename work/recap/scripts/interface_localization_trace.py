from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "interface_localization_sprint"
DEFAULT_RUNTIME_LOG_DIR = "agent/runtime_logs/interface_localization_sprint"
DEFAULT_EVIDENCE_JSON = ".sisyphus/evidence/task-5-interface-trace.json"

INTERFACE_TRACE_CSV_NAME = "interface_trace.csv"
RESPONSE_SUMMARY_JSON_NAME = "response_summary.json"
TRACE_RUNTIME_LOG_JSON_NAME = "interface_trace_runtime_log.json"

RESPONSE_SUMMARY_SCHEMA_VERSION = "interface_localization_response_summary_v1"
RESPONSE_SUMMARY_ARTIFACT_KIND = "interface_trace_response_summary"

TRACE_CSV_FIELDNAMES: tuple[str, ...] = (
    "trace_index",
    "boundary_name",
    "status",
    "provenance_class",
    "seed",
    "condition_label",
    "watch_bucket",
    "field_name",
    "value_repr",
    "blocked_reason",
)

WATCH_BUCKET_ORDER: tuple[str, ...] = (
    "body_wrist_upper_limb_chain",
    "dex3_finger_hand_path",
)

BLOCKED_FIELD_NAMES: tuple[str, ...] = (
    "q_target",
    "q_measured",
    "q_error",
)

BODY_WATCH_VECTOR_LENGTH = 7
HAND_WATCH_VECTOR_LENGTH = 7


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import interface_localization_action_roundtrip
from work.recap import interface_localization_contract
from work.recap import interface_localization_surface_inventory
from work.recap import interface_localization_text_rewrite_map
from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_phase0_smoke
from work.recap import state_conditioned_snapshot_harvest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface_localization_trace.py",
        description=(
            "Build a deterministic minimal replay trace harness that keeps the field set "
            "small, records explicit blocked_missing_upstream rows, and writes a compact "
            "response summary JSON sidecar."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, interface_trace.csv and "
            "response_summary.json are written to <artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for the generated CSV/JSON artifacts.",
    )
    _ = parser.add_argument(
        "--runtime-log-dir",
        type=str,
        default=DEFAULT_RUNTIME_LOG_DIR,
        help="Directory for the deterministic runtime-sidecar JSON log.",
    )
    _ = parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help=(
            "Optional explicit path for response_summary.json. If empty, write it into "
            "the selected output directory."
        ),
    )
    _ = parser.add_argument(
        "--evidence-json",
        type=str,
        default=DEFAULT_EVIDENCE_JSON,
        help="Evidence JSON written after the trace CSV and summary JSON succeed.",
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


def resolve_runtime_log_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(
        _resolve_path(repo_root, str(args.runtime_log_dir))
    )


def resolve_summary_json(
    repo_root: Path,
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> Path:
    raw_summary_json = str(args.summary_json).strip()
    if raw_summary_json:
        return _resolve_path(repo_root, raw_summary_json)
    return output_dir / RESPONSE_SUMMARY_JSON_NAME


def resolve_evidence_json(repo_root: Path, args: argparse.Namespace) -> Path:
    return _resolve_path(repo_root, str(args.evidence_json))


def _generation_command_for(
    output_dir: Path,
    runtime_log_dir: Path,
    repo_root: Path,
    *,
    summary_json: Path | None = None,
) -> str:
    command_parts = [
        "python3 work/recap/scripts/interface_localization_trace.py",
        f"--output-dir {_relpath(repo_root, output_dir)}",
        f"--runtime-log-dir {_relpath(repo_root, runtime_log_dir)}",
    ]
    if (
        summary_json is not None
        and summary_json.resolve()
        != (output_dir / RESPONSE_SUMMARY_JSON_NAME).resolve()
    ):
        command_parts.append(f"--summary-json {_relpath(repo_root, summary_json)}")
    return " ".join(command_parts)


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


def _baseline_tuple_digest(contract_payload: Mapping[str, Any]) -> str:
    baseline_json = _canonical_json_text(
        dict(contract_payload["baseline_tuple"]),
    )
    return hashlib.sha256(baseline_json.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRACE_CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in TRACE_CSV_FIELDNAMES}
            )
    tmp.replace(path)
    return path


def _numpy_module() -> Any:
    import importlib

    return importlib.import_module("numpy")


def _vector_summary(value: object) -> str:
    np = _numpy_module()
    arr = np.asarray(value, dtype=float).reshape(-1)
    preview = [round(float(item), 6) for item in arr[:3].tolist()]
    payload = {
        "count": int(arr.size),
        "l2_norm": round(float(np.linalg.norm(arr)), 6),
        "min": round(float(arr.min()), 6) if arr.size else 0.0,
        "max": round(float(arr.max()), 6) if arr.size else 0.0,
        "preview": preview,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _scalar_summary(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return f"{float(value):.6f}"
    return str(value)


def _default_seed_values() -> tuple[int, ...]:
    return (0,)


def _default_condition_specs() -> tuple[dict[str, object], ...]:
    return (
        {
            "condition_label": "SEARCH_NOMINAL",
            "phase": "SEARCH",
            "mode": "NOMINAL",
            "recap_value": 0.0,
        },
        {
            "condition_label": "SEARCH_RECOVERY",
            "phase": "SEARCH",
            "mode": "RECOVERY",
            "recap_value": 1.0,
        },
    )


def _default_replay_fixture_arrays() -> dict[str, list[list[list[float]]]]:
    return {
        "action/right_arm": [
            [[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]],
            [[0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56]],
        ],
        "action/right_hand": [
            [[0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]],
            [[0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]],
        ],
    }


def _write_replay_fixture_npz(
    path: Path,
    *,
    arrays_by_key: Mapping[str, Sequence[Sequence[Sequence[float]]]] | None = None,
) -> Path:
    np = _numpy_module()
    payload = dict(arrays_by_key or _default_replay_fixture_arrays())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{key: np.asarray(value, dtype=np.float32) for key, value in payload.items()},
    )
    return path


def _build_condition_payload(condition_spec: Mapping[str, object]) -> dict[str, Any]:
    payload = state_conditioned_phase0_smoke._policy_condition_payload(
        condition_spec["phase"],
        condition_spec["mode"],
    )
    payload["recap_value"] = float(cast(Any, condition_spec["recap_value"]))
    payload["condition_label"] = str(condition_spec["condition_label"])
    return payload


def _condition_label(condition_spec: Mapping[str, object]) -> str:
    return str(condition_spec["condition_label"])


def _condition_recap_value(condition_spec: Mapping[str, object]) -> float:
    return float(cast(Any, condition_spec["recap_value"]))


def _available_row(
    *,
    trace_index: int,
    boundary_name: str,
    provenance_class: str,
    seed: int,
    condition_label: str,
    watch_bucket: str,
    field_name: str,
    value_repr: str,
) -> dict[str, object]:
    return {
        "trace_index": int(trace_index),
        "boundary_name": boundary_name,
        "status": "survived",
        "provenance_class": provenance_class,
        "seed": int(seed),
        "condition_label": condition_label,
        "watch_bucket": watch_bucket,
        "field_name": field_name,
        "value_repr": value_repr,
        "blocked_reason": "",
    }


def _blocked_row(
    *,
    trace_index: int,
    boundary_name: str,
    provenance_class: str,
    seed: int,
    condition_label: str,
    watch_bucket: str,
    field_name: str,
    blocked_reason: str,
) -> dict[str, object]:
    return {
        "trace_index": int(trace_index),
        "boundary_name": boundary_name,
        "status": "blocked_missing_upstream",
        "provenance_class": provenance_class,
        "seed": int(seed),
        "condition_label": condition_label,
        "watch_bucket": watch_bucket,
        "field_name": field_name,
        "value_repr": "blocked_missing_upstream",
        "blocked_reason": blocked_reason,
    }


def _blocked_reason_for(
    *,
    field_name: str,
    watch_bucket: str,
) -> str:
    return (
        f"current checkout exposes replay action chunks for {watch_bucket} but has no repo-local "
        f"trace surface proving {field_name} at this boundary; emit blocked_missing_upstream instead "
        "of silently omitting the field"
    )


def _build_trace_rows(
    *,
    seed_values: Sequence[int],
    condition_specs: Sequence[Mapping[str, object]],
    action_chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    np = _numpy_module()
    rows: list[dict[str, object]] = []
    trace_index = 0
    for seed in seed_values:
        for condition_index, condition_spec in enumerate(condition_specs):
            condition_payload = _build_condition_payload(condition_spec)
            action_chunk = dict(action_chunks[condition_index % len(action_chunks)])
            right_arm = np.asarray(
                action_chunk["action.right_arm"], dtype=np.float32
            ).reshape(
                BODY_WATCH_VECTOR_LENGTH,
            )
            right_hand = np.asarray(
                action_chunk["action.right_hand"],
                dtype=np.float32,
            ).reshape(HAND_WATCH_VECTOR_LENGTH)

            seed_offset = float(seed) * 0.01
            recap_value = float(condition_payload["recap_value"])
            body_q = right_arm + seed_offset + recap_value * 0.05
            body_dq = right_arm * 0.1 + recap_value * 0.01
            body_motion_ref = right_arm
            hand_motion_ref = right_hand
            upper_body_target = right_arm + recap_value * 0.02

            batched_obs = cast(
                dict[str, Any],
                state_conditioned_phase0_smoke._batch_observation_for_policy(
                    {
                        "observation.state.right_arm_q": body_q,
                        "observation.state.right_arm_dq": body_dq,
                        "annotation.human.task_description": str(
                            condition_payload["text"]
                        ),
                    }
                ),
            )
            batched_action = {
                "action.right_arm": np.expand_dims(body_motion_ref, axis=0),
                "action.right_hand": np.expand_dims(hand_motion_ref, axis=0),
            }
            unbatched_action = cast(
                dict[str, Any],
                state_conditioned_phase0_smoke._unbatch_policy_action(batched_action),
            )

            condition_label = str(condition_payload["condition_label"])
            for watch_bucket in WATCH_BUCKET_ORDER:
                trace_index += 1
                rows.append(
                    _available_row(
                        trace_index=trace_index,
                        boundary_name="collector_policy_callsite",
                        provenance_class="synthetic",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=watch_bucket,
                        field_name="condition_text",
                        value_repr=_scalar_summary(condition_payload["text"]),
                    )
                )
                trace_index += 1
                rows.append(
                    _available_row(
                        trace_index=trace_index,
                        boundary_name="collector_policy_callsite",
                        provenance_class="synthetic",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=watch_bucket,
                        field_name="recap_value",
                        value_repr=_scalar_summary(condition_payload["recap_value"]),
                    )
                )

            trace_index += 1
            rows.append(
                _available_row(
                    trace_index=trace_index,
                    boundary_name="policy_input_collation",
                    provenance_class="synthetic",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket="body_wrist_upper_limb_chain",
                    field_name="obs_body_q",
                    value_repr=_vector_summary(
                        cast(Any, batched_obs["observation.state.right_arm_q"])[0]
                    ),
                )
            )
            trace_index += 1
            rows.append(
                _available_row(
                    trace_index=trace_index,
                    boundary_name="policy_input_collation",
                    provenance_class="synthetic",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket="body_wrist_upper_limb_chain",
                    field_name="obs_body_dq",
                    value_repr=_vector_summary(
                        cast(Any, batched_obs["observation.state.right_arm_dq"])[0]
                    ),
                )
            )
            trace_index += 1
            rows.append(
                _available_row(
                    trace_index=trace_index,
                    boundary_name="action_semantics_adapter",
                    provenance_class="replay_live",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket="body_wrist_upper_limb_chain",
                    field_name="motion_ref",
                    value_repr=_vector_summary(body_motion_ref),
                )
            )
            trace_index += 1
            rows.append(
                _available_row(
                    trace_index=trace_index,
                    boundary_name="action_semantics_adapter",
                    provenance_class="synthetic",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket="body_wrist_upper_limb_chain",
                    field_name="upper_body_target",
                    value_repr=_vector_summary(upper_body_target),
                )
            )
            trace_index += 1
            rows.append(
                _available_row(
                    trace_index=trace_index,
                    boundary_name="policy_output_action",
                    provenance_class="synthetic",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket="body_wrist_upper_limb_chain",
                    field_name="raw_action_norm",
                    value_repr=_scalar_summary(
                        float(
                            np.linalg.norm(
                                np.asarray(
                                    unbatched_action["action.right_arm"], dtype=float
                                )
                            )
                        )
                    ),
                )
            )
            trace_index += 1
            rows.append(
                _available_row(
                    trace_index=trace_index,
                    boundary_name="action_semantics_adapter",
                    provenance_class="replay_live",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket="dex3_finger_hand_path",
                    field_name="motion_ref",
                    value_repr=_vector_summary(hand_motion_ref),
                )
            )
            trace_index += 1
            rows.append(
                _available_row(
                    trace_index=trace_index,
                    boundary_name="policy_output_action",
                    provenance_class="synthetic",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket="dex3_finger_hand_path",
                    field_name="raw_action_norm",
                    value_repr=_scalar_summary(
                        float(
                            np.linalg.norm(
                                np.asarray(
                                    unbatched_action["action.right_hand"], dtype=float
                                )
                            )
                        )
                    ),
                )
            )

            for watch_bucket, q_boundary_name in (
                ("body_wrist_upper_limb_chain", "body_wrist_upper_limb_chain"),
                ("dex3_finger_hand_path", "dex3_finger_hand_path"),
            ):
                trace_index += 1
                rows.append(
                    _blocked_row(
                        trace_index=trace_index,
                        boundary_name="action_semantics_adapter",
                        provenance_class="server_live",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=watch_bucket,
                        field_name="q_target",
                        blocked_reason=_blocked_reason_for(
                            field_name="q_target",
                            watch_bucket=watch_bucket,
                        ),
                    )
                )
                trace_index += 1
                rows.append(
                    _blocked_row(
                        trace_index=trace_index,
                        boundary_name=q_boundary_name,
                        provenance_class="server_live",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=watch_bucket,
                        field_name="q_measured",
                        blocked_reason=_blocked_reason_for(
                            field_name="q_measured",
                            watch_bucket=watch_bucket,
                        ),
                    )
                )
                trace_index += 1
                rows.append(
                    _blocked_row(
                        trace_index=trace_index,
                        boundary_name=q_boundary_name,
                        provenance_class="server_live",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=watch_bucket,
                        field_name="q_error",
                        blocked_reason=_blocked_reason_for(
                            field_name="q_error",
                            watch_bucket=watch_bucket,
                        ),
                    )
                )
    return rows


def build_interface_trace_payload(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    runtime_log_dir: Path | None = None,
    summary_json: Path | None = None,
    replay_fixture_path: Path | None = None,
    seed_values: Sequence[int] | None = None,
    condition_specs: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    contract_payload = _load_contract_payload()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR).resolve()
    )
    resolved_runtime_log_dir = (
        runtime_log_dir.resolve()
        if runtime_log_dir is not None
        else (repo_root / DEFAULT_RUNTIME_LOG_DIR).resolve()
    )
    resolved_summary_json = (
        summary_json.resolve()
        if summary_json is not None
        else (resolved_output_dir / RESPONSE_SUMMARY_JSON_NAME).resolve()
    )
    generation_command = _generation_command_for(
        resolved_output_dir,
        resolved_runtime_log_dir,
        repo_root,
        summary_json=resolved_summary_json,
    )

    seeds = [int(seed) for seed in (seed_values or _default_seed_values())]
    conditions = [
        dict(item) for item in (condition_specs or _default_condition_specs())
    ]
    if not seeds:
        raise ValueError("seed_values must not be empty")
    if not conditions:
        raise ValueError("condition_specs must not be empty")

    persisted_fixture = replay_fixture_path is not None
    if persisted_fixture:
        fixture_path = replay_fixture_path.resolve()
        _write_replay_fixture_npz(fixture_path)
        action_chunks = state_conditioned_snapshot_harvest._load_replay_action_chunks(
            {"source_npz_path": str(fixture_path)}
        )
    else:
        with tempfile.TemporaryDirectory(prefix="interface_trace_fixture_") as tmp_dir:
            fixture_path = Path(tmp_dir) / "interface_trace_fixture.npz"
            _write_replay_fixture_npz(fixture_path)
            action_chunks = (
                state_conditioned_snapshot_harvest._load_replay_action_chunks(
                    {"source_npz_path": str(fixture_path)}
                )
            )
    normalized_chunks = [
        state_conditioned_snapshot_harvest._normalize_replay_action_chunk_for_env(item)
        for item in action_chunks
    ]
    if len(normalized_chunks) < len(conditions):
        raise ValueError(
            "replay fixture must provide at least one normalized action chunk per condition"
        )

    trace_rows = _build_trace_rows(
        seed_values=seeds,
        condition_specs=conditions,
        action_chunks=normalized_chunks,
    )

    status_counts = Counter(str(row["status"]) for row in trace_rows)
    boundary_counts = Counter(str(row["boundary_name"]) for row in trace_rows)
    blocked_rows = [
        dict(row)
        for row in trace_rows
        if str(row["status"]) == "blocked_missing_upstream"
    ]
    response_summary = {
        "schema_version": RESPONSE_SUMMARY_SCHEMA_VERSION,
        "artifact_kind": RESPONSE_SUMMARY_ARTIFACT_KIND,
        "provenance_class": "synthetic",
        "generation_command": generation_command,
        "input_baseline_summary": _baseline_summary(contract_payload),
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_trace.py",
            "task1_contract_writer": "work/recap/scripts/interface_localization_contract.py",
            "task2_inventory_writer": "work/recap/scripts/interface_localization_surface_inventory.py",
            "task3_text_rewrite_map_writer": "work/recap/scripts/interface_localization_text_rewrite_map.py",
            "task4_action_roundtrip_writer": "work/recap/scripts/interface_localization_action_roundtrip.py",
            "expected_task1_contract_json": str(
                resolved_output_dir / interface_localization_contract.CONTRACT_JSON_NAME
            ),
            "expected_task2_inventory_json": str(
                resolved_output_dir
                / interface_localization_surface_inventory.REPLAY_SURFACE_INVENTORY_JSON_NAME
            ),
            "expected_task3_text_rewrite_map_json": str(
                resolved_output_dir
                / interface_localization_text_rewrite_map.TEXT_REWRITE_MAP_JSON_NAME
            ),
            "expected_task4_action_roundtrip_json": str(
                resolved_output_dir
                / interface_localization_action_roundtrip.ACTION_ROUNDTRIP_JSON_NAME
            ),
            "interface_trace_csv": str(resolved_output_dir / INTERFACE_TRACE_CSV_NAME),
            "runtime_log_dir": str(resolved_runtime_log_dir),
            "runtime_log_json": str(
                resolved_runtime_log_dir / TRACE_RUNTIME_LOG_JSON_NAME
            ),
            "pytest_command": "python3 -m pytest tests/recap/test_interface_trace_harness.py -q",
        },
        "baseline_tuple_digest": _baseline_tuple_digest(contract_payload),
        "trace_csv_field_order": list(TRACE_CSV_FIELDNAMES),
        "seed_order": seeds,
        "condition_order": [str(item["condition_label"]) for item in conditions],
        "watch_bucket_order": list(WATCH_BUCKET_ORDER),
        "replay_harness": {
            "source_mode": "repo_local_replay_helper_fixture",
            "replay_fixture_npz": str(fixture_path),
            "replay_fixture_persisted": bool(persisted_fixture),
            "action_chunk_count": int(len(normalized_chunks)),
            "action_chunk_keys": sorted(list(normalized_chunks[0].keys())),
            "reused_entrypoints": {
                "load_replay_action_chunks": (
                    "work/recap/scripts/state_conditioned_snapshot_harvest.py::"
                    "_load_replay_action_chunks"
                ),
                "normalize_replay_action_chunk_for_env": (
                    "work/recap/scripts/state_conditioned_snapshot_harvest.py::"
                    "_normalize_replay_action_chunk_for_env"
                ),
                "policy_condition_payload": (
                    "work/recap/scripts/state_conditioned_phase0_smoke.py::"
                    "_policy_condition_payload"
                ),
                "batch_observation_for_policy": (
                    "work/recap/scripts/state_conditioned_phase0_smoke.py::"
                    "_batch_observation_for_policy"
                ),
                "unbatch_policy_action": (
                    "work/recap/scripts/state_conditioned_phase0_smoke.py::"
                    "_unbatch_policy_action"
                ),
            },
        },
        "summary": {
            "row_count": int(len(trace_rows)),
            "blocked_row_count": int(len(blocked_rows)),
            "rows_by_status": {
                key: int(value) for key, value in sorted(status_counts.items())
            },
            "rows_by_boundary": {
                key: int(value) for key, value in sorted(boundary_counts.items())
            },
            "blocked_field_names": list(BLOCKED_FIELD_NAMES),
            "blocked_surface_count_by_field": {
                field_name: sum(
                    1 for row in blocked_rows if str(row["field_name"]) == field_name
                )
                for field_name in BLOCKED_FIELD_NAMES
            },
        },
        "responses": [
            {
                "seed": int(seed),
                "condition_label": _condition_label(condition_spec),
                "recap_value": _condition_recap_value(condition_spec),
                "blocked_fields": list(BLOCKED_FIELD_NAMES),
                "watch_bucket_status": {
                    bucket_name: "blocked_missing_upstream"
                    if any(
                        str(row["watch_bucket"]) == bucket_name
                        and str(row["status"]) == "blocked_missing_upstream"
                        and str(row["condition_label"])
                        == _condition_label(condition_spec)
                        and int(cast(Any, row["seed"])) == int(seed)
                        for row in trace_rows
                    )
                    else "survived"
                    for bucket_name in WATCH_BUCKET_ORDER
                },
            }
            for seed in seeds
            for condition_spec in conditions
        ],
        "blocked_surfaces": [
            {
                "boundary_name": str(row["boundary_name"]),
                "seed": int(cast(Any, row["seed"])),
                "condition_label": str(row["condition_label"]),
                "watch_bucket": str(row["watch_bucket"]),
                "field_name": str(row["field_name"]),
                "status": str(row["status"]),
                "provenance_class": str(row["provenance_class"]),
                "blocked_reason": str(row["blocked_reason"]),
            }
            for row in blocked_rows
        ],
    }
    runtime_log_payload = {
        "status": "PASS",
        "generation_command": generation_command,
        "trace_csv_fields": list(TRACE_CSV_FIELDNAMES),
        "row_count": int(len(trace_rows)),
        "blocked_row_count": int(len(blocked_rows)),
        "seed_order": list(seeds),
        "condition_order": [str(item["condition_label"]) for item in conditions],
        "replay_fixture_npz": str(fixture_path),
    }
    return {
        "trace_rows": trace_rows,
        "response_summary": response_summary,
        "runtime_log_payload": runtime_log_payload,
    }


def write_artifacts(
    *,
    output_dir: Path,
    runtime_log_dir: Path,
    summary_json: Path,
    evidence_json: Path,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    trace_csv_path = _write_csv(
        output_dir / INTERFACE_TRACE_CSV_NAME,
        payload["trace_rows"],
    )
    response_summary_path = _write_json(
        summary_json,
        payload["response_summary"],
    )
    runtime_log_path = _write_json(
        runtime_log_dir / TRACE_RUNTIME_LOG_JSON_NAME,
        payload["runtime_log_payload"],
    )
    evidence_payload = {
        "schema_version": "interface_localization_task5_evidence_v1",
        "artifact_kind": "interface_localization_trace_evidence",
        "provenance_class": "synthetic",
        "generation_command": str(payload["response_summary"]["generation_command"]),
        "input_baseline_summary": dict(
            payload["response_summary"]["input_baseline_summary"]
        ),
        "backpointer": {
            "interface_trace_csv": str(trace_csv_path),
            "response_summary_json": str(response_summary_path),
            "runtime_log_dir": str(runtime_log_dir),
            "runtime_log_json": str(runtime_log_path),
            "test_command": "python3 -m pytest tests/recap/test_interface_trace_harness.py -q",
        },
    }
    evidence_path = _write_json(evidence_json, evidence_payload)
    return {
        "interface_trace_csv": str(trace_csv_path),
        "response_summary_json": str(response_summary_path),
        "runtime_log_json": str(runtime_log_path),
        "evidence_json": str(evidence_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output_dir = resolve_output_dir(REPO_ROOT, args)
        runtime_log_dir = resolve_runtime_log_dir(REPO_ROOT, args)
        summary_json = resolve_summary_json(REPO_ROOT, args, output_dir=output_dir)
        evidence_json = resolve_evidence_json(REPO_ROOT, args)
        replay_fixture_path = runtime_log_dir / "interface_trace_fixture.npz"
        payload = build_interface_trace_payload(
            REPO_ROOT,
            output_dir=output_dir,
            runtime_log_dir=runtime_log_dir,
            summary_json=summary_json,
            replay_fixture_path=replay_fixture_path,
        )
        written_paths = write_artifacts(
            output_dir=output_dir,
            runtime_log_dir=runtime_log_dir,
            summary_json=summary_json,
            evidence_json=evidence_json,
            payload=payload,
        )
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "output_dir": str(output_dir),
                    **written_paths,
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
