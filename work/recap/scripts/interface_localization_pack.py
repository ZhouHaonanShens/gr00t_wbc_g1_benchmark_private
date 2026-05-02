from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "interface_localization_sprint"
DEFAULT_INPUT_SUBDIR = "interface_localization_sprint"
DEFAULT_RUNTIME_LOG_DIR = "agent/runtime_logs/interface_localization_sprint"
DEFAULT_EVIDENCE_JSON = ".sisyphus/evidence/task-9-interface-localization-pack.json"

PACK_JSON_NAME = "interface_localization_pack.json"
PACK_SCHEMA_VERSION = "interface_localization_pack_v1"
PACK_ARTIFACT_KIND = "interface_localization_final_pack"

TASK9_EVIDENCE_SCHEMA_VERSION = "sisyphus_task_evidence_v1"
TASK9_EVIDENCE_ARTIFACT_KIND = "task_9_interface_localization_pack_evidence"

TRACE_RUNTIME_LOG_JSON_NAME = "interface_trace_runtime_log.json"

TASK1_BASELINE_JSON_NAME = "baseline_tuple.json"
TASK1_CONTRACT_JSON_NAME = "interface_localization_contract.json"
TASK2_BLOCKERS_JSON_NAME = "conditional_blockers.json"
TASK2_INVENTORY_JSON_NAME = "replay_surface_inventory.json"
TASK3_TEXT_MAP_JSON_NAME = "recap_text_source_and_rewrite_map.json"
TASK4_ACTION_ROUNDTRIP_JSON_NAME = "action_chain_watchlist_split.json"
TASK5_TRACE_CSV_NAME = "interface_trace.csv"
TASK5_RESPONSE_SUMMARY_JSON_NAME = "response_summary.json"
TASK6_NUMERIC_CUSTOM_JSON_NAME = "recap_numeric_custom_path.json"
TASK7_NUMERIC_STOCK_JSON_NAME = "recap_numeric_stock_path.json"
TASK8_SPLIT_JSON_NAME = "right_arm_vs_right_hand_split_audit.json"

TASK2_EVIDENCE_JSON = ".sisyphus/evidence/task-2-surface-inventory.json"
TASK3_EVIDENCE_JSON = ".sisyphus/evidence/task-3-text-rewrite-map.json"
TASK4_EVIDENCE_JSON = ".sisyphus/evidence/task-4-action-roundtrip.json"
TASK5_EVIDENCE_JSON = ".sisyphus/evidence/task-5-interface-trace.json"
TASK6_EVIDENCE_JSON = ".sisyphus/evidence/task-6-numeric-custom-path.json"
TASK7_EVIDENCE_JSON = ".sisyphus/evidence/task-7-stock-path-audit.json"
TASK8_EVIDENCE_JSON = ".sisyphus/evidence/task-8-right-hand-split.json"
TASK18_ATTRIBUTION_PACK_JSON = ".sisyphus/evidence/task-18-attribution-pack.json"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import interface_localization_contract
from work.recap import state_conditioned_bucket_a_import
from work.recap.scripts import interface_localization_text_rewrite_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface_localization_pack.py",
        description=(
            "Synthesize the final interface-localization pack from the existing Task 1 to "
            "8 artifacts, then write a task-level evidence file with runtime-log, artifact, "
            "and evidence backpointers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, the final pack is written to "
            "<artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help=(
            "Directory that already contains the Task 1 to 8 JSON and CSV outputs. If empty, "
            "use <artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for interface_localization_pack.json.",
    )
    _ = parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional explicit output JSON path for the final pack.",
    )
    _ = parser.add_argument(
        "--runtime-log-dir",
        type=str,
        default=DEFAULT_RUNTIME_LOG_DIR,
        help="Runtime log directory that already contains Task 5 trace runtime evidence.",
    )
    _ = parser.add_argument(
        "--evidence-json",
        type=str,
        default=DEFAULT_EVIDENCE_JSON,
        help="Output path for the Task 9 evidence JSON.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _timestamp_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds")


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _validate_existing_file(path: Path, *, field_name: str) -> Path:
    if not path.exists() or not path.is_file():
        raise ValueError(f"missing required input file for {field_name}: {path}")
    return path


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _prepare_output_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _rel_repo(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = _resolve_path(REPO_ROOT, path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


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


def _as_string_list(value: object, *, field_name: str) -> list[str]:
    return [
        _as_string(item, field_name=f"{field_name}[]")
        for item in _as_list(value, field_name=field_name)
    ]


def _read_json(path: Path, *, field_name: str) -> dict[str, Any]:
    payload = json.loads(
        _validate_existing_file(path, field_name=field_name).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{field_name} must contain a JSON object")
    return cast(dict[str, Any], dict(payload))


def _path_pointer(path: Path, pointer: str) -> str:
    return f"{_rel_repo(path)}#{pointer}"


def _read_json_metadata(path: Path) -> tuple[object, object, object, str]:
    if path.suffix.lower() != ".json":
        selected_payload = {
            "path": _rel_repo(path),
            "suffix": path.suffix,
            "file_sha256": _sha256_file(path),
        }
        return None, None, None, _sha256_payload(selected_payload)
    payload = _read_json(path, field_name=str(path))
    selected_payload = {
        "artifact_kind": payload.get("artifact_kind"),
        "schema_version": payload.get("schema_version"),
        "task_code": payload.get("task_code"),
        "status": payload.get("status"),
        "provenance_class": payload.get("provenance_class"),
        "report_signature_sha256": payload.get("report_signature_sha256"),
    }
    return (
        payload.get("artifact_kind"),
        payload.get("schema_version"),
        payload.get("report_signature_sha256"),
        _sha256_payload(selected_payload),
    )


def _inventory_entry(*, artifact_id: str, task_code: str, path: Path) -> dict[str, Any]:
    artifact_kind, schema_version, report_signature_sha256, config_or_schema_digest = (
        _read_json_metadata(path)
    )
    return {
        "artifact_id": artifact_id,
        "task_code": task_code,
        "path": _rel_repo(path),
        "artifact_kind": artifact_kind,
        "schema_version": schema_version,
        "file_sha256": _sha256_file(path),
        "config_or_schema_digest": config_or_schema_digest,
        "report_signature_sha256": report_signature_sha256,
    }


def resolve_input_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    raw_input_dir = str(args.input_dir).strip()
    if raw_input_dir:
        return _resolve_path(repo_root, raw_input_dir)
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    return (artifact_dir / DEFAULT_INPUT_SUBDIR).resolve()


def resolve_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    raw_output_dir = str(args.output_dir).strip()
    if raw_output_dir:
        return _validate_output_dir(_resolve_path(repo_root, raw_output_dir))
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    return _validate_output_dir((artifact_dir / DEFAULT_OUTPUT_SUBDIR).resolve())


def resolve_output_json(
    repo_root: Path,
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> Path:
    raw_output_json = str(args.output_json).strip()
    if raw_output_json:
        return _prepare_output_file(_resolve_path(repo_root, raw_output_json))
    return _prepare_output_file((output_dir / PACK_JSON_NAME).resolve())


def resolve_runtime_log_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    return _resolve_path(repo_root, str(args.runtime_log_dir)).resolve()


def resolve_evidence_json(repo_root: Path, args: argparse.Namespace) -> Path:
    return _prepare_output_file(_resolve_path(repo_root, str(args.evidence_json)))


def _generation_command_for(
    *, output_dir: Path, output_json: Path, evidence_json: Path
) -> str:
    command_parts = [
        "python3 work/recap/scripts/interface_localization_pack.py",
        f"--output-dir {_rel_repo(output_dir) or str(output_dir)}",
    ]
    default_output_json = (output_dir / PACK_JSON_NAME).resolve()
    if output_json.resolve() != default_output_json:
        command_parts.append(
            f"--output-json {_rel_repo(output_json) or str(output_json)}"
        )
    default_evidence_json = _resolve_path(REPO_ROOT, DEFAULT_EVIDENCE_JSON)
    if evidence_json.resolve() != default_evidence_json:
        command_parts.append(
            f"--evidence-json {_rel_repo(evidence_json) or str(evidence_json)}"
        )
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
    return hashlib.sha256(
        _canonical_json_text(dict(contract_payload["baseline_tuple"])).encode("utf-8")
    ).hexdigest()


def _surface_authority_labels_from_text_payload(
    text_payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_labels = text_payload.get("surface_authority_labels")
    if isinstance(raw_labels, Mapping):
        return dict(raw_labels)
    generated_payload = (
        interface_localization_text_rewrite_map.build_text_source_and_rewrite_map(
            REPO_ROOT
        )
    )
    return dict(
        _as_mapping(
            generated_payload.get("surface_authority_labels"),
            field_name="generated_text_rewrite_map.surface_authority_labels",
        )
    )


def _required_input_paths(
    *, repo_root: Path, input_dir: Path, runtime_log_dir: Path
) -> dict[str, Path]:
    required = {
        "baseline_tuple_json": input_dir / TASK1_BASELINE_JSON_NAME,
        "contract_json": input_dir / TASK1_CONTRACT_JSON_NAME,
        "conditional_blockers_json": input_dir / TASK2_BLOCKERS_JSON_NAME,
        "replay_surface_inventory_json": input_dir / TASK2_INVENTORY_JSON_NAME,
        "text_rewrite_map_json": input_dir / TASK3_TEXT_MAP_JSON_NAME,
        "action_roundtrip_json": input_dir / TASK4_ACTION_ROUNDTRIP_JSON_NAME,
        "interface_trace_csv": input_dir / TASK5_TRACE_CSV_NAME,
        "response_summary_json": input_dir / TASK5_RESPONSE_SUMMARY_JSON_NAME,
        "numeric_custom_json": input_dir / TASK6_NUMERIC_CUSTOM_JSON_NAME,
        "numeric_stock_json": input_dir / TASK7_NUMERIC_STOCK_JSON_NAME,
        "right_hand_split_json": input_dir / TASK8_SPLIT_JSON_NAME,
        "trace_runtime_log_json": runtime_log_dir / TRACE_RUNTIME_LOG_JSON_NAME,
        "task2_evidence_json": repo_root / TASK2_EVIDENCE_JSON,
        "task3_evidence_json": repo_root / TASK3_EVIDENCE_JSON,
        "task4_evidence_json": repo_root / TASK4_EVIDENCE_JSON,
        "task5_evidence_json": repo_root / TASK5_EVIDENCE_JSON,
        "task6_evidence_json": repo_root / TASK6_EVIDENCE_JSON,
        "task7_evidence_json": repo_root / TASK7_EVIDENCE_JSON,
        "task8_evidence_json": repo_root / TASK8_EVIDENCE_JSON,
        "task18_attribution_pack_json": repo_root / TASK18_ATTRIBUTION_PACK_JSON,
    }
    for field_name, path in required.items():
        _ = _validate_existing_file(path.resolve(), field_name=field_name)
    return {key: value.resolve() for key, value in required.items()}


def _find_surface(
    payload: Mapping[str, Any], *, surface_name: str
) -> Mapping[str, Any]:
    for entry in _as_list(payload.get("surfaces"), field_name="surfaces"):
        mapping = _as_mapping(entry, field_name=f"surfaces[{surface_name}]")
        if str(mapping.get("surface_name")) == surface_name:
            return mapping
    raise KeyError(f"missing surface: {surface_name}")


def _find_blocker(
    payload: Mapping[str, Any], *, surface_name: str
) -> Mapping[str, Any]:
    for entry in _as_list(payload.get("blockers"), field_name="blockers"):
        mapping = _as_mapping(entry, field_name=f"blockers[{surface_name}]")
        if str(mapping.get("surface_name")) == surface_name:
            return mapping
    raise KeyError(f"missing blocker: {surface_name}")


def _stage_status(text_payload: Mapping[str, Any], stage_name: str) -> str:
    stages = _as_mapping(text_payload.get("stages"), field_name="stages")
    stage = _as_mapping(stages.get(stage_name), field_name=f"stages.{stage_name}")
    return _as_string(stage.get("status"), field_name=f"stages.{stage_name}.status")


def _sentinel_boundary_status(payload: Mapping[str, Any], boundary_name: str) -> str:
    sentinel_records = _as_list(
        _as_mapping(payload.get("summary"), field_name="summary").get(
            "sentinel_records"
        ),
        field_name="summary.sentinel_records",
    )
    observed = {
        _as_string(
            _as_mapping(record, field_name="summary.sentinel_records[]")
            .get("boundary_status", {})
            .get(boundary_name),
            field_name=f"summary.sentinel_records[].boundary_status.{boundary_name}",
        )
        for record in sentinel_records
    }
    if len(observed) != 1:
        raise ValueError(
            f"expected exactly one sentinel status for {boundary_name}, got {sorted(observed)!r}"
        )
    return next(iter(observed))


def _trace_blocked_fields(
    response_summary_payload: Mapping[str, Any], *, boundary_name: str
) -> list[str]:
    blocked_surfaces = _as_list(
        response_summary_payload.get("blocked_surfaces"), field_name="blocked_surfaces"
    )
    fields = [
        _as_string(
            _as_mapping(item, field_name="blocked_surfaces[]").get("field_name"),
            field_name="blocked_surfaces[].field_name",
        )
        for item in blocked_surfaces
        if str(_as_mapping(item, field_name="blocked_surfaces[]").get("boundary_name"))
        == boundary_name
    ]
    return sorted(set(fields))


def _trace_blocked_reason(
    response_summary_payload: Mapping[str, Any], *, boundary_name: str
) -> str | None:
    blocked_surfaces = _as_list(
        response_summary_payload.get("blocked_surfaces"), field_name="blocked_surfaces"
    )
    for item in blocked_surfaces:
        mapping = _as_mapping(item, field_name="blocked_surfaces[]")
        if str(mapping.get("boundary_name")) == boundary_name:
            return _as_optional_string(
                mapping.get("blocked_reason"),
                field_name=f"blocked_surfaces[{boundary_name}].blocked_reason",
            )
    return None


def _json_path_list(*paths: str) -> list[str]:
    return [path for path in paths if path]


def _final_boundary_statuses(
    *,
    contract_payload: Mapping[str, Any],
    inventory_payload: Mapping[str, Any],
    text_payload: Mapping[str, Any],
    response_summary_payload: Mapping[str, Any],
    custom_numeric_payload: Mapping[str, Any],
    stock_numeric_payload: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    input_paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    custom_surface = _find_surface(
        inventory_payload, surface_name="custom_advantage_aware_server_cli"
    )
    stock_surface = _find_surface(
        inventory_payload, surface_name="stock_mainline_server_entrypoint"
    )
    custom_blocker = _find_blocker(
        _read_json(
            input_paths["conditional_blockers_json"], field_name="conditional_blockers"
        ),
        surface_name="custom_advantage_aware_server_cli",
    )
    dex_route_state = _as_mapping(
        _as_mapping(
            split_payload.get("dex3_finger_hand_path"),
            field_name="dex3_finger_hand_path",
        ).get("upstream_source_route_state"),
        field_name="dex3_finger_hand_path.upstream_source_route_state",
    )
    body_route_state = _as_mapping(
        _as_mapping(
            split_payload.get("body_wrist_upper_limb_chain"),
            field_name="body_wrist_upper_limb_chain",
        ).get("upstream_source_route_state"),
        field_name="body_wrist_upper_limb_chain.upstream_source_route_state",
    )
    text_map_rel = _rel_repo(input_paths["text_rewrite_map_json"]) or str(
        input_paths["text_rewrite_map_json"]
    )
    response_summary_rel = _rel_repo(input_paths["response_summary_json"]) or str(
        input_paths["response_summary_json"]
    )
    custom_numeric_rel = _rel_repo(input_paths["numeric_custom_json"]) or str(
        input_paths["numeric_custom_json"]
    )
    stock_numeric_rel = _rel_repo(input_paths["numeric_stock_json"]) or str(
        input_paths["numeric_stock_json"]
    )
    split_rel = _rel_repo(input_paths["right_hand_split_json"]) or str(
        input_paths["right_hand_split_json"]
    )
    inventory_rel = _rel_repo(input_paths["replay_surface_inventory_json"]) or str(
        input_paths["replay_surface_inventory_json"]
    )
    action_roundtrip_rel = _rel_repo(input_paths["action_roundtrip_json"]) or str(
        input_paths["action_roundtrip_json"]
    )

    entries = [
        {
            "boundary_name": "prompt_raw_source",
            "surface_authority_key": "carrier_text_v1",
            "authority_status": "authority_source_field",
            "final_status": _stage_status(text_payload, "collector.prompt_raw"),
            "status_by_context": {
                "collector.prompt_raw": _stage_status(
                    text_payload, "collector.prompt_raw"
                ),
                "text_indicator_policy.prompt_raw": _stage_status(
                    text_payload, "text_indicator_policy.prompt_raw"
                ),
            },
            "summary": "carrier_text_v1 authority remains anchored to the prompt_raw source field from collector selection through the text-indicator policy. The legacy numeric_mainline_consumes label is preserved only as a compatibility alias for that authority source field.",
            "blocked_reason": None,
            "backpointer": _json_path_list(
                f"{text_map_rel}#stages.collector.prompt_raw",
                f"{text_map_rel}#stages.text_indicator_policy.prompt_raw",
                f"{text_map_rel}#summary.numeric_mainline_consumes",
            ),
        },
        {
            "boundary_name": "prompt_conditioned_write",
            "surface_authority_key": "prompt_conditioned",
            "authority_status": "legacy_non_authority",
            "final_status": _stage_status(text_payload, "collector.prompt_conditioned"),
            "status_by_context": {
                "collector.prompt_conditioned": _stage_status(
                    text_payload, "collector.prompt_conditioned"
                ),
                "labeler.prompt_conditioned": _stage_status(
                    text_payload, "labeler.prompt_conditioned"
                ),
            },
            "summary": "prompt_conditioned survives as a legacy non-authority write surface in collector and labeler paths, but it no longer narrates the active mainline after carrier_text_v1 freeze.",
            "blocked_reason": None,
            "backpointer": _json_path_list(
                f"{text_map_rel}#stages.collector.prompt_conditioned",
                f"{text_map_rel}#stages.labeler.prompt_conditioned",
            ),
        },
        {
            "boundary_name": "export_task_text_selection",
            "surface_authority_key": "carrier_text_v1",
            "authority_status": "mainline_authority",
            "final_status": _stage_status(text_payload, "export.task_text_field"),
            "status_by_context": {
                "export.task_text_field": _stage_status(
                    text_payload, "export.task_text_field"
                ),
                "export.dual_task_text": _stage_status(
                    text_payload, "export.dual_task_text"
                ),
            },
            "summary": "dataset export now defaults to carrier_text_v1 as the mainline authority carrier, while the dual-text branch keeps prompt_raw and prompt_conditioned only for legacy compatibility and later analysis.",
            "blocked_reason": None,
            "backpointer": _json_path_list(
                f"{text_map_rel}#stages.export.task_text_field",
                f"{text_map_rel}#stages.export.dual_task_text",
                f"{text_map_rel}#relationships[0]",
            ),
        },
        {
            "boundary_name": "collector_policy_callsite",
            "surface_authority_key": "numeric_advantage",
            "authority_status": "diagnostic_only",
            "final_status": "survived",
            "status_by_context": {
                "trace_replay_harness": "survived",
                "custom_adv": _sentinel_boundary_status(
                    custom_numeric_payload, "collector_policy_callsite"
                ),
                "stock_mainline": _sentinel_boundary_status(
                    stock_numeric_payload, "collector_policy_callsite"
                ),
            },
            "summary": "both numeric probes remain visible at the collector callsite, but they are diagnostic-only lanes and no longer describe the active authority carrier.",
            "blocked_reason": None,
            "backpointer": _json_path_list(
                f"{response_summary_rel}#summary.rows_by_boundary.collector_policy_callsite",
                f"{custom_numeric_rel}#summary.sentinel_records",
                f"{stock_numeric_rel}#summary.sentinel_records",
            ),
        },
        {
            "boundary_name": "server_policy_adapter",
            "surface_authority_key": "numeric_advantage",
            "authority_status": "diagnostic_only",
            "final_status": _as_string(
                custom_surface.get("status"),
                field_name="custom_advantage_aware_server_cli.status",
            ),
            "status_by_context": {
                "custom_advantage_aware_server_cli": _as_string(
                    custom_surface.get("status"),
                    field_name="custom_advantage_aware_server_cli.status",
                ),
                "stock_mainline_server_entrypoint": _as_string(
                    stock_surface.get("status"),
                    field_name="stock_mainline_server_entrypoint.status",
                ),
            },
            "summary": "the stock diagnostic probe entrypoint is present, but the custom advantage-aware diagnostic adapter is still blocked by the missing gr00t Python module.",
            "blocked_reason": _as_optional_string(
                custom_blocker.get("blocked_reason"),
                field_name="custom_advantage_aware_server_cli.blocked_reason",
            ),
            "backpointer": _json_path_list(
                f"{inventory_rel}#surfaces",
                f"{_rel_repo(input_paths['conditional_blockers_json'])}#blockers",
            ),
        },
        {
            "boundary_name": "policy_input_collation",
            "surface_authority_key": "numeric_advantage",
            "authority_status": "diagnostic_only",
            "final_status": "survived",
            "status_by_context": {
                "custom_adv": _sentinel_boundary_status(
                    custom_numeric_payload, "policy_input_collation"
                ),
                "stock_mainline": _sentinel_boundary_status(
                    stock_numeric_payload, "policy_input_collation"
                ),
            },
            "summary": "policy input collation remains visible on both diagnostic numeric lanes. The custom path shows a synthetic advantage tensor injection point, and the stock path shows runtime provenance entering the purity gate without claiming mainline authority.",
            "blocked_reason": None,
            "backpointer": _json_path_list(
                f"{custom_numeric_rel}#rows",
                f"{stock_numeric_rel}#rows",
            ),
        },
        {
            "boundary_name": "model_condition_injection",
            "surface_authority_key": "numeric_advantage",
            "authority_status": "diagnostic_only",
            "final_status": _sentinel_boundary_status(
                custom_numeric_payload, "model_condition_injection"
            ),
            "status_by_context": {
                "custom_adv": _sentinel_boundary_status(
                    custom_numeric_payload, "model_condition_injection"
                ),
                "stock_mainline": _sentinel_boundary_status(
                    stock_numeric_payload, "model_condition_injection"
                ),
            },
            "summary": "the stock diagnostic probe keeps purity and provenance through model_condition_injection, but the custom advantage-aware diagnostic path still loses runtime visibility at this boundary because gr00t is missing.",
            "blocked_reason": _trace_blocked_reason(
                custom_numeric_payload, boundary_name="model_condition_injection"
            )
            or _as_optional_string(
                _as_list(
                    custom_numeric_payload.get("blocked_surfaces"),
                    field_name="blocked_surfaces",
                )[0].get("blocked_reason"),
                field_name="blocked_surfaces[0].blocked_reason",
            ),
            "backpointer": _json_path_list(
                f"{custom_numeric_rel}#blocked_surfaces",
                f"{stock_numeric_rel}#stock_path_evidence.purity_gate",
            ),
        },
        {
            "boundary_name": "policy_output_action",
            "surface_authority_key": "numeric_advantage",
            "authority_status": "diagnostic_only",
            "final_status": _sentinel_boundary_status(
                custom_numeric_payload, "policy_output_action"
            ),
            "status_by_context": {
                "custom_adv": _sentinel_boundary_status(
                    custom_numeric_payload, "policy_output_action"
                ),
                "stock_mainline": _sentinel_boundary_status(
                    stock_numeric_payload, "policy_output_action"
                ),
            },
            "summary": "the stock diagnostic probe reaches a branch-aware next-step decision at policy_output_action, while the custom diagnostic path still cannot expose action_pred_digest because gr00t is missing.",
            "blocked_reason": _as_optional_string(
                _as_mapping(
                    _as_list(
                        custom_numeric_payload.get("blocked_surfaces"),
                        field_name="blocked_surfaces",
                    )[1],
                    field_name="blocked_surfaces[1]",
                ).get("blocked_reason"),
                field_name="blocked_surfaces[1].blocked_reason",
            ),
            "backpointer": _json_path_list(
                f"{custom_numeric_rel}#blocked_surfaces",
                f"{stock_numeric_rel}#stock_path_evidence",
            ),
        },
        {
            "boundary_name": "action_semantics_adapter",
            "final_status": "blocked_missing_upstream",
            "status_by_context": {
                "canonical_action_roundtrip": "survived",
                "repo_local_trace_q_target": "blocked_missing_upstream",
            },
            "summary": "canonical absolute action adaptation is proven, including the right_arm versus right_hand split, but repo-local trace surfaces still do not expose q_target at the semantic adapter boundary.",
            "blocked_reason": _trace_blocked_reason(
                response_summary_payload, boundary_name="action_semantics_adapter"
            ),
            "observed_fields": ["motion_ref", "upper_body_target"],
            "blocked_fields": _trace_blocked_fields(
                response_summary_payload, boundary_name="action_semantics_adapter"
            ),
            "backpointer": _json_path_list(
                f"{action_roundtrip_rel}#canonical_space",
                f"{response_summary_rel}#blocked_surfaces",
            ),
        },
        {
            "boundary_name": "body_wrist_upper_limb_chain",
            "final_status": _as_string(
                body_route_state.get("status"),
                field_name="body_wrist_upper_limb_chain.upstream_source_route_state.status",
            ),
            "status_by_context": {
                "ownership_split": _as_string(
                    _as_mapping(
                        split_payload.get("body_wrist_upper_limb_chain"),
                        field_name="body_wrist_upper_limb_chain",
                    ).get("status"),
                    field_name="body_wrist_upper_limb_chain.status",
                ),
                "upstream_source_route": _as_string(
                    body_route_state.get("status"),
                    field_name="body_wrist_upper_limb_chain.upstream_source_route_state.status",
                ),
                "trace_q_measured_and_q_error": "blocked_missing_upstream",
            },
            "summary": "right_arm keeps wrist ownership and the live difference persists through local telemetry, but repo-local trace surfaces still do not expose q_measured or q_error at this boundary.",
            "blocked_reason": _trace_blocked_reason(
                response_summary_payload, boundary_name="body_wrist_upper_limb_chain"
            ),
            "backpointer": _json_path_list(
                f"{split_rel}#body_wrist_upper_limb_chain",
                f"{response_summary_rel}#blocked_surfaces",
            ),
        },
        {
            "boundary_name": "dex3_finger_hand_path",
            "final_status": _as_string(
                dex_route_state.get("status"),
                field_name="dex3_finger_hand_path.upstream_source_route_state.status",
            ),
            "status_by_context": {
                "ownership_split": _as_string(
                    _as_mapping(
                        split_payload.get("dex3_finger_hand_path"),
                        field_name="dex3_finger_hand_path",
                    ).get("status"),
                    field_name="dex3_finger_hand_path.status",
                ),
                "telemetry": "survived",
                "upstream_source_route": _as_string(
                    dex_route_state.get("status"),
                    field_name="dex3_finger_hand_path.upstream_source_route_state.status",
                ),
            },
            "summary": "right_hand finger and thumb ownership plus model-insensitive telemetry are retained, but Dex3 or finger-level upstream source-route proof is still blocked upstream.",
            "blocked_reason": _as_optional_string(
                dex_route_state.get("blocked_reason"),
                field_name="dex3_finger_hand_path.upstream_source_route_state.blocked_reason",
            ),
            "backpointer": _json_path_list(
                f"{split_rel}#dex3_finger_hand_path",
                f"{response_summary_rel}#blocked_surfaces",
            ),
        },
    ]
    expected_order = list(contract_payload["boundary_ontology"]["boundary_order"])
    actual_order = [str(entry["boundary_name"]) for entry in entries]
    if actual_order != expected_order:
        raise ValueError(
            f"final boundary order drift: expected {expected_order}, got {actual_order}"
        )
    return entries


def _success_findings(*, input_paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    text_map_rel = _rel_repo(input_paths["text_rewrite_map_json"]) or str(
        input_paths["text_rewrite_map_json"]
    )
    stock_numeric_rel = _rel_repo(input_paths["numeric_stock_json"]) or str(
        input_paths["numeric_stock_json"]
    )
    action_roundtrip_rel = _rel_repo(input_paths["action_roundtrip_json"]) or str(
        input_paths["action_roundtrip_json"]
    )
    split_rel = _rel_repo(input_paths["right_hand_split_json"]) or str(
        input_paths["right_hand_split_json"]
    )
    response_summary_rel = _rel_repo(input_paths["response_summary_json"]) or str(
        input_paths["response_summary_json"]
    )
    runtime_log_rel = _rel_repo(input_paths["trace_runtime_log_json"]) or str(
        input_paths["trace_runtime_log_json"]
    )
    return [
        {
            "finding_code": "text_mainline_prompt_raw_retained",
            "summary": "`carrier_text_v1` 主 authority 仍锚定在 `prompt_raw` 源字段；collector 和 labeler 虽然都能写出 `prompt_conditioned`，但它现在只是 legacy non-authority lane。",
            "backpointer": _json_path_list(
                f"{text_map_rel}#summary.numeric_mainline_consumes",
                f"{text_map_rel}#relationships",
            ),
        },
        {
            "finding_code": "stock_numeric_path_survives_all_boundaries",
            "summary": "stock numeric sentinel 在 collector_policy_callsite、policy_input_collation、model_condition_injection、policy_output_action 四个边界都保持 survived，但它属于 diagnostic-only lane，不再承担主线 authority 叙述。",
            "backpointer": _json_path_list(
                f"{stock_numeric_rel}#summary.sentinel_records",
                f"{stock_numeric_rel}#stock_path_evidence",
            ),
        },
        {
            "finding_code": "right_arm_and_right_hand_split_remains_distinct",
            "summary": "right_arm 继续绑定 wrist，right_hand 继续绑定 finger/thumb。repo-local roundtrip 和 split audit 都支持这条分界，不需要重新合并成单一 right_hand bucket。",
            "backpointer": _json_path_list(
                f"{action_roundtrip_rel}#summary.explicit_split",
                f"{split_rel}#summary.ownership_binding_by_bucket",
            ),
        },
        {
            "finding_code": "trace_harness_keeps_minimal_runtime_chain",
            "summary": "Task 5 的最小 trace harness 已经把 runtime log、trace CSV、response summary 连成可回溯链，而且明确保留 blocked_missing_upstream 行而不是静默省略。",
            "backpointer": _json_path_list(
                f"{runtime_log_rel}",
                f"{response_summary_rel}#summary",
            ),
        },
    ]


def _blocker_findings(*, input_paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    blockers_rel = _rel_repo(input_paths["conditional_blockers_json"]) or str(
        input_paths["conditional_blockers_json"]
    )
    response_summary_rel = _rel_repo(input_paths["response_summary_json"]) or str(
        input_paths["response_summary_json"]
    )
    custom_numeric_rel = _rel_repo(input_paths["numeric_custom_json"]) or str(
        input_paths["numeric_custom_json"]
    )
    split_rel = _rel_repo(input_paths["right_hand_split_json"]) or str(
        input_paths["right_hand_split_json"]
    )
    return [
        {
            "finding_code": "missing_gr00t_module_blocks_custom_runtime_visibility",
            "summary": "当前最高信号 blocker 仍是缺少 gr00t Python 模块，这会同时卡住 custom advantage-aware server CLI、model_condition_injection 可见性和 policy_output_action 可见性。",
            "backpointer": _json_path_list(
                f"{blockers_rel}#blockers",
                f"{custom_numeric_rel}#blocked_surfaces",
            ),
        },
        {
            "finding_code": "trace_q_target_q_measured_q_error_still_missing",
            "summary": "repo-local trace harness 目前仍无法在两个 watch bucket 上给出 q_target、q_measured、q_error 的本地证明面，所以这些字段必须继续标成 blocked_missing_upstream。",
            "backpointer": _json_path_list(
                f"{response_summary_rel}#blocked_surfaces",
                f"{response_summary_rel}#summary.blocked_field_names",
            ),
        },
        {
            "finding_code": "dex3_source_route_remains_upstream_blocked",
            "summary": "Dex3 finger-side source route 仍然没有 repo-local upstream proof。right_hand telemetry 可以保留，但不能被升级成 Dex3 路由已证实。",
            "backpointer": _json_path_list(
                f"{split_rel}#dex3_finger_hand_path.upstream_source_route_state",
                f"{split_rel}#dex3_finger_hand_path.telemetry_evidence",
            ),
        },
    ]


def _collect_blocked_surfaces(
    *,
    conditional_blockers_payload: Mapping[str, Any],
    response_summary_payload: Mapping[str, Any],
    custom_numeric_payload: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    input_paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    blockers_rel = _rel_repo(input_paths["conditional_blockers_json"]) or str(
        input_paths["conditional_blockers_json"]
    )
    response_summary_rel = _rel_repo(input_paths["response_summary_json"]) or str(
        input_paths["response_summary_json"]
    )
    custom_numeric_rel = _rel_repo(input_paths["numeric_custom_json"]) or str(
        input_paths["numeric_custom_json"]
    )
    split_rel = _rel_repo(input_paths["right_hand_split_json"]) or str(
        input_paths["right_hand_split_json"]
    )

    for index, entry in enumerate(
        _as_list(conditional_blockers_payload.get("blockers"), field_name="blockers")
    ):
        mapping = _as_mapping(entry, field_name=f"blockers[{index}]")
        blocked.append(
            {
                "source_artifact": blockers_rel,
                "source_pointer": f"blockers[{index}]",
                "status": _as_string(
                    mapping.get("status"), field_name="blockers[].status"
                ),
                "surface_name": _as_string(
                    mapping.get("surface_name"), field_name="blockers[].surface_name"
                ),
                "boundary_name": None,
                "condition_label": None,
                "watch_bucket": None,
                "field_name": None,
                "blocked_reason": _as_optional_string(
                    mapping.get("blocked_reason"),
                    field_name="blockers[].blocked_reason",
                ),
            }
        )

    for index, entry in enumerate(
        _as_list(
            response_summary_payload.get("blocked_surfaces"),
            field_name="response_summary.blocked_surfaces",
        )
    ):
        mapping = _as_mapping(
            entry, field_name=f"response_summary.blocked_surfaces[{index}]"
        )
        blocked.append(
            {
                "source_artifact": response_summary_rel,
                "source_pointer": f"blocked_surfaces[{index}]",
                "status": _as_string(
                    mapping.get("status"),
                    field_name="response_summary.blocked_surfaces[].status",
                ),
                "surface_name": None,
                "boundary_name": _as_string(
                    mapping.get("boundary_name"),
                    field_name="response_summary.blocked_surfaces[].boundary_name",
                ),
                "condition_label": _as_string(
                    mapping.get("condition_label"),
                    field_name="response_summary.blocked_surfaces[].condition_label",
                ),
                "watch_bucket": _as_string(
                    mapping.get("watch_bucket"),
                    field_name="response_summary.blocked_surfaces[].watch_bucket",
                ),
                "field_name": _as_string(
                    mapping.get("field_name"),
                    field_name="response_summary.blocked_surfaces[].field_name",
                ),
                "blocked_reason": _as_optional_string(
                    mapping.get("blocked_reason"),
                    field_name="response_summary.blocked_surfaces[].blocked_reason",
                ),
            }
        )

    for index, entry in enumerate(
        _as_list(
            custom_numeric_payload.get("blocked_surfaces"),
            field_name="custom_numeric.blocked_surfaces",
        )
    ):
        mapping = _as_mapping(
            entry, field_name=f"custom_numeric.blocked_surfaces[{index}]"
        )
        blocked.append(
            {
                "source_artifact": custom_numeric_rel,
                "source_pointer": f"blocked_surfaces[{index}]",
                "status": _as_string(
                    mapping.get("status"),
                    field_name="custom_numeric.blocked_surfaces[].status",
                ),
                "surface_name": None,
                "boundary_name": _as_string(
                    mapping.get("boundary_name"),
                    field_name="custom_numeric.blocked_surfaces[].boundary_name",
                ),
                "condition_label": _as_string(
                    mapping.get("condition_label"),
                    field_name="custom_numeric.blocked_surfaces[].condition_label",
                ),
                "watch_bucket": _as_string(
                    mapping.get("watch_bucket"),
                    field_name="custom_numeric.blocked_surfaces[].watch_bucket",
                ),
                "field_name": _as_string(
                    mapping.get("field_name"),
                    field_name="custom_numeric.blocked_surfaces[].field_name",
                ),
                "blocked_reason": _as_optional_string(
                    mapping.get("blocked_reason"),
                    field_name="custom_numeric.blocked_surfaces[].blocked_reason",
                ),
            }
        )

    dex_route_state = _as_mapping(
        _as_mapping(
            split_payload.get("dex3_finger_hand_path"),
            field_name="dex3_finger_hand_path",
        ).get("upstream_source_route_state"),
        field_name="dex3_finger_hand_path.upstream_source_route_state",
    )
    if str(dex_route_state.get("status")) == "blocked_missing_upstream":
        blocked.append(
            {
                "source_artifact": split_rel,
                "source_pointer": "dex3_finger_hand_path.upstream_source_route_state",
                "status": "blocked_missing_upstream",
                "surface_name": "dex3_source_route",
                "boundary_name": "dex3_finger_hand_path",
                "condition_label": None,
                "watch_bucket": "dex3_finger_hand_path",
                "field_name": "upstream_source_route",
                "blocked_reason": _as_optional_string(
                    dex_route_state.get("blocked_reason"),
                    field_name="dex3_finger_hand_path.upstream_source_route_state.blocked_reason",
                ),
            }
        )
    return blocked


def _supporting_artifacts(input_paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": "task1_baseline_tuple_artifact",
            "task_code": "T1",
            "path": _rel_repo(input_paths["baseline_tuple_json"]),
            "summary": "Frozen baseline tuple for the sprint.",
        },
        {
            "artifact_id": "task1_contract_artifact",
            "task_code": "T1",
            "path": _rel_repo(input_paths["contract_json"]),
            "summary": "Frozen boundary ontology and status vocabulary.",
        },
        {
            "artifact_id": "task2_replay_surface_inventory",
            "task_code": "T2",
            "path": _rel_repo(input_paths["replay_surface_inventory_json"]),
            "summary": "Inventory of replay, stock, custom, and runtime prompt surfaces.",
        },
        {
            "artifact_id": "task2_conditional_blockers",
            "task_code": "T2",
            "path": _rel_repo(input_paths["conditional_blockers_json"]),
            "summary": "Explicit blocker list, including python_module.gr00t.",
        },
        {
            "artifact_id": "task3_text_rewrite_map",
            "task_code": "T3",
            "path": _rel_repo(input_paths["text_rewrite_map_json"]),
            "summary": "Text source, rewrite, and export selection map.",
        },
        {
            "artifact_id": "task4_action_roundtrip",
            "task_code": "T4",
            "path": _rel_repo(input_paths["action_roundtrip_json"]),
            "summary": "Canonical action roundtrip and right_arm versus right_hand split.",
        },
        {
            "artifact_id": "task5_interface_trace_csv",
            "task_code": "T5",
            "path": _rel_repo(input_paths["interface_trace_csv"]),
            "summary": "Minimal replay trace rows with explicit blocked surfaces.",
        },
        {
            "artifact_id": "task5_response_summary",
            "task_code": "T5",
            "path": _rel_repo(input_paths["response_summary_json"]),
            "summary": "Trace summary layer, including boundary counts and blocked fields.",
        },
        {
            "artifact_id": "task5_trace_runtime_log_json",
            "task_code": "T5",
            "path": _rel_repo(input_paths["trace_runtime_log_json"]),
            "summary": "Runtime log anchor for the trace harness.",
        },
        {
            "artifact_id": "task6_numeric_custom_path",
            "task_code": "T6",
            "path": _rel_repo(input_paths["numeric_custom_json"]),
            "summary": "Custom path numeric sentinel visibility summary.",
        },
        {
            "artifact_id": "task7_numeric_stock_path",
            "task_code": "T7",
            "path": _rel_repo(input_paths["numeric_stock_json"]),
            "summary": "Stock path numeric diagnostic sentinel visibility summary.",
        },
        {
            "artifact_id": "task8_right_hand_split_audit",
            "task_code": "T8",
            "path": _rel_repo(input_paths["right_hand_split_json"]),
            "summary": "Right-arm versus right-hand split audit, including Dex3 blocker state.",
        },
        {
            "artifact_id": "task18_attribution_pack_reference",
            "task_code": "T18",
            "path": _rel_repo(input_paths["task18_attribution_pack_json"]),
            "summary": "Reference pack used as the structural template for Task 9 evidence.",
        },
    ]


def build_interface_localization_pack(
    repo_root: Path,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    output_json: Path | None = None,
    runtime_log_dir: Path | None = None,
    evidence_json: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    contract_payload = _load_contract_payload()
    resolved_input_dir = (
        input_dir or (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_INPUT_SUBDIR)
    ).resolve()
    resolved_output_dir = _validate_output_dir(
        (
            output_dir or (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR)
        ).resolve()
    )
    resolved_output_json = _prepare_output_file(
        (output_json or (resolved_output_dir / PACK_JSON_NAME)).resolve()
    )
    resolved_runtime_log_dir = (
        runtime_log_dir or _resolve_path(repo_root, DEFAULT_RUNTIME_LOG_DIR)
    ).resolve()
    resolved_evidence_json = _prepare_output_file(
        (evidence_json or _resolve_path(repo_root, DEFAULT_EVIDENCE_JSON)).resolve()
    )
    input_paths = _required_input_paths(
        repo_root=repo_root,
        input_dir=resolved_input_dir,
        runtime_log_dir=resolved_runtime_log_dir,
    )

    baseline_payload = _read_json(
        input_paths["baseline_tuple_json"], field_name="baseline_tuple_json"
    )
    contract_json_payload = _read_json(
        input_paths["contract_json"], field_name="contract_json"
    )
    conditional_blockers_payload = _read_json(
        input_paths["conditional_blockers_json"], field_name="conditional_blockers_json"
    )
    inventory_payload = _read_json(
        input_paths["replay_surface_inventory_json"],
        field_name="replay_surface_inventory_json",
    )
    text_payload = _read_json(
        input_paths["text_rewrite_map_json"], field_name="text_rewrite_map_json"
    )
    response_summary_payload = _read_json(
        input_paths["response_summary_json"], field_name="response_summary_json"
    )
    custom_numeric_payload = _read_json(
        input_paths["numeric_custom_json"], field_name="numeric_custom_json"
    )
    stock_numeric_payload = _read_json(
        input_paths["numeric_stock_json"], field_name="numeric_stock_json"
    )
    split_payload = _read_json(
        input_paths["right_hand_split_json"], field_name="right_hand_split_json"
    )

    baseline_digest = _as_string(
        response_summary_payload.get("baseline_tuple_digest"),
        field_name="response_summary_json.baseline_tuple_digest",
    )
    expected_digest = _baseline_tuple_digest(contract_payload)
    if baseline_digest != expected_digest:
        raise ValueError(
            f"baseline tuple digest mismatch: expected {expected_digest}, got {baseline_digest}"
        )
    for payload_name, payload in (
        ("numeric_custom_json", custom_numeric_payload),
        ("numeric_stock_json", stock_numeric_payload),
    ):
        payload_digest = _as_string(
            payload.get("baseline_tuple_digest"),
            field_name=f"{payload_name}.baseline_tuple_digest",
        )
        if payload_digest != baseline_digest:
            raise ValueError(
                f"{payload_name} baseline digest mismatch: expected {baseline_digest}, got {payload_digest}"
            )

    blocked_surfaces = _collect_blocked_surfaces(
        conditional_blockers_payload=conditional_blockers_payload,
        response_summary_payload=response_summary_payload,
        custom_numeric_payload=custom_numeric_payload,
        split_payload=split_payload,
        input_paths=input_paths,
    )
    boundary_statuses = _final_boundary_statuses(
        contract_payload=contract_payload,
        inventory_payload=inventory_payload,
        text_payload=text_payload,
        response_summary_payload=response_summary_payload,
        custom_numeric_payload=custom_numeric_payload,
        stock_numeric_payload=stock_numeric_payload,
        split_payload=split_payload,
        input_paths=input_paths,
    )
    success_findings = _success_findings(input_paths=input_paths)
    blocker_findings = _blocker_findings(input_paths=input_paths)
    blocked_surface_count_by_source = dict(
        sorted(
            Counter(str(item["source_artifact"]) for item in blocked_surfaces).items()
        )
    )
    generated_at_value = generated_at or _timestamp_now()

    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "artifact_kind": PACK_ARTIFACT_KIND,
        "task_code": "T9",
        "generated_at": generated_at_value,
        "generation_command": _generation_command_for(
            output_dir=resolved_output_dir,
            output_json=resolved_output_json,
            evidence_json=resolved_evidence_json,
        ),
        "provenance_class": "static",
        "output_dir": _rel_repo(resolved_output_dir),
        "evidence_path": _rel_repo(resolved_evidence_json),
        "input_baseline_summary": _baseline_summary(contract_payload),
        "baseline_tuple_digest": baseline_digest,
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_pack.py",
            "runtime_log_dir": _rel_repo(resolved_runtime_log_dir),
            "runtime_log_json": _rel_repo(input_paths["trace_runtime_log_json"]),
            "interface_trace_csv": _rel_repo(input_paths["interface_trace_csv"]),
            "response_summary_json": _rel_repo(input_paths["response_summary_json"]),
            "task18_template_reference": _rel_repo(
                input_paths["task18_attribution_pack_json"]
            ),
        },
        "generated_outputs": {
            "interface_localization_pack_json": {
                "path": _rel_repo(resolved_output_json),
                "role": "final_machine_readable_localization_pack",
            },
            "task9_evidence_json": {
                "path": _rel_repo(resolved_evidence_json),
                "role": "task_level_evidence_backpointer",
            },
            "response_summary_layer": {
                "path": _rel_repo(input_paths["response_summary_json"]),
                "role": "reused_final_summary_layer",
                "updated": False,
            },
        },
        "surface_authority_labels": _surface_authority_labels_from_text_payload(
            text_payload
        ),
        "final_boundary_statuses": boundary_statuses,
        "success_findings": success_findings,
        "blocker_findings": blocker_findings,
        "blocked_surfaces": blocked_surfaces,
        "blocked_surface_summary": {
            "blocked_surface_count": len(blocked_surfaces),
            "blocked_surface_count_by_source": blocked_surface_count_by_source,
        },
        "key_supporting_artifacts": _supporting_artifacts(input_paths),
        "recommended_next_step": {
            "step_code": "unblock_gr00t_then_capture_custom_visibility_and_dex3_route_proof",
            "summary": "First restore repo-local gr00t importability so the custom advantage-aware path can be re-probed through server_policy_adapter, model_condition_injection, and policy_output_action. After that, keep the right_hand telemetry evidence but seek explicit Dex3 or finger-level upstream source-route proof below the wrist boundary.",
            "priority_blockers": [
                "python_module.gr00t",
                "dex3_finger_hand_path.upstream_source_route",
            ],
            "backpointer": _json_path_list(
                _path_pointer(input_paths["conditional_blockers_json"], "blockers"),
                _path_pointer(
                    input_paths["right_hand_split_json"],
                    "dex3_finger_hand_path.upstream_source_route_state",
                ),
            ),
        },
        "source_artifact_backpointer": {
            "baseline_tuple_json": _rel_repo(input_paths["baseline_tuple_json"]),
            "contract_json": _rel_repo(input_paths["contract_json"]),
        },
        "validation_snapshot": {
            "baseline_tuple_json_schema_version": baseline_payload.get(
                "schema_version"
            ),
            "contract_json_schema_version": contract_json_payload.get("schema_version"),
        },
    }


def build_task9_evidence_payload(
    *,
    pack_payload: Mapping[str, Any],
    pack_output_json: Path,
    evidence_json: Path,
    input_paths: Mapping[str, Path],
) -> dict[str, Any]:
    success_findings = _as_list(
        pack_payload.get("success_findings"), field_name="success_findings"
    )
    blocker_findings = _as_list(
        pack_payload.get("blocker_findings"), field_name="blocker_findings"
    )
    boundary_statuses = _as_list(
        pack_payload.get("final_boundary_statuses"),
        field_name="final_boundary_statuses",
    )
    generated_outputs = {
        "interface_localization_pack": _inventory_entry(
            artifact_id="task9_interface_localization_pack",
            task_code="T9",
            path=pack_output_json,
        ),
        "response_summary_layer": _inventory_entry(
            artifact_id="task5_response_summary_layer",
            task_code="T5",
            path=input_paths["response_summary_json"],
        ),
    }
    key_supporting_artifacts = [
        _inventory_entry(
            artifact_id="task2_surface_inventory_evidence",
            task_code="T2",
            path=input_paths["task2_evidence_json"],
        ),
        _inventory_entry(
            artifact_id="task3_text_rewrite_map_evidence",
            task_code="T3",
            path=input_paths["task3_evidence_json"],
        ),
        _inventory_entry(
            artifact_id="task4_action_roundtrip_evidence",
            task_code="T4",
            path=input_paths["task4_evidence_json"],
        ),
        _inventory_entry(
            artifact_id="task5_interface_trace_evidence",
            task_code="T5",
            path=input_paths["task5_evidence_json"],
        ),
        _inventory_entry(
            artifact_id="task6_numeric_custom_path_evidence",
            task_code="T6",
            path=input_paths["task6_evidence_json"],
        ),
        _inventory_entry(
            artifact_id="task7_stock_path_audit_evidence",
            task_code="T7",
            path=input_paths["task7_evidence_json"],
        ),
        _inventory_entry(
            artifact_id="task8_right_hand_split_evidence",
            task_code="T8",
            path=input_paths["task8_evidence_json"],
        ),
        _inventory_entry(
            artifact_id="task5_trace_runtime_log_json",
            task_code="T5",
            path=input_paths["trace_runtime_log_json"],
        ),
        _inventory_entry(
            artifact_id="task18_attribution_pack_reference",
            task_code="T18",
            path=input_paths["task18_attribution_pack_json"],
        ),
    ]
    return {
        "schema_version": TASK9_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": TASK9_EVIDENCE_ARTIFACT_KIND,
        "task_code": "T9",
        "status": "PASS",
        "generated_at": pack_payload.get("generated_at"),
        "evidence_path": _rel_repo(evidence_json),
        "output_dir": pack_payload.get("output_dir"),
        "generated_outputs": generated_outputs,
        "final_conclusions": {
            "baseline_tuple_digest": pack_payload.get("baseline_tuple_digest"),
            "final_boundary_statuses": [dict(item) for item in boundary_statuses],
            "recommended_next_step": dict(
                _as_mapping(
                    pack_payload.get("recommended_next_step"),
                    field_name="recommended_next_step",
                )
            ),
            "success_findings": [dict(item) for item in success_findings],
            "blocker_findings": [dict(item) for item in blocker_findings],
        },
        "shared_structural_findings": [
            {
                "finding_code": str(item.get("finding_code")),
                "summary": str(item.get("summary")),
            }
            for item in [*success_findings, *blocker_findings]
        ],
        "key_supporting_artifacts": key_supporting_artifacts,
    }


def write_pack_and_evidence(
    *,
    output_json: Path,
    evidence_json: Path,
    pack_payload: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    written_pack = _write_json(output_json, pack_payload)
    written_evidence = _write_json(evidence_json, evidence_payload)
    return written_pack, written_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        input_dir = resolve_input_dir(REPO_ROOT, args)
        output_dir = resolve_output_dir(REPO_ROOT, args)
        output_json = resolve_output_json(REPO_ROOT, args, output_dir=output_dir)
        runtime_log_dir = resolve_runtime_log_dir(REPO_ROOT, args)
        evidence_json = resolve_evidence_json(REPO_ROOT, args)
        input_paths = _required_input_paths(
            repo_root=REPO_ROOT,
            input_dir=input_dir,
            runtime_log_dir=runtime_log_dir,
        )
        pack_payload = build_interface_localization_pack(
            REPO_ROOT,
            input_dir=input_dir,
            output_dir=output_dir,
            output_json=output_json,
            runtime_log_dir=runtime_log_dir,
            evidence_json=evidence_json,
        )
        written_pack = _write_json(output_json, pack_payload)
        evidence_payload = build_task9_evidence_payload(
            pack_payload=pack_payload,
            pack_output_json=written_pack,
            evidence_json=evidence_json,
            input_paths=input_paths,
        )
        written_evidence = _write_json(evidence_json, evidence_payload)
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "output_json": str(written_pack),
                    "evidence_json": str(written_evidence),
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
