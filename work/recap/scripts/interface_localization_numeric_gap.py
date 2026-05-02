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


DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "interface_localization_sprint"
NUMERIC_CUSTOM_PATH_JSON_NAME = "recap_numeric_custom_path.json"
NUMERIC_STOCK_PATH_JSON_NAME = "recap_numeric_stock_path.json"

NUMERIC_CUSTOM_PATH_SCHEMA_VERSION = "interface_localization_numeric_custom_path_v1"
NUMERIC_CUSTOM_PATH_ARTIFACT_KIND = "interface_numeric_custom_path_summary"
NUMERIC_STOCK_PATH_SCHEMA_VERSION = "interface_localization_numeric_stock_path_v1"
NUMERIC_STOCK_PATH_ARTIFACT_KIND = "interface_numeric_stock_path_summary"
CUSTOM_PATH_MODE = "custom_adv"
STOCK_PATH_MODE = "stock_mainline"

STOCK_ENTRYPOINT_RELATIVE_PATH = "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"
STOCK_ENTRYPOINT_REFERENCE_LINES = (
    "work/recap/scripts/state_conditioned_phase0_smoke.py:109-110"
)
STOCK_PURITY_REFERENCE_LINES = "work/recap/scripts/3D_recap_eval.py:256-363,1358-1389"
CUSTOM_PROVENANCE_COUNTEREXAMPLE_REFERENCE_LINES = (
    "work/recap/scripts/3D_recap_run_adv_server.py:224-245"
)
TASK18_BRANCH_DECISION_JSON = ".sisyphus/evidence/task-18-attribution-pack.json"
TASK18_BRANCH_DECISION_REFERENCE_LINES = (
    ".sisyphus/evidence/task-18-attribution-pack.json:81-84"
)
DEFAULT_RECOMMENDED_NEXT_STEP_BY_BRANCH = {
    "new_embodiment": "audit_recap_injection_action_target_and_relative_action_interpretation",
    "unitree_g1": "audit_recap_injection_action_target_and_relative_action_interpretation",
}

BOUNDARY_ORDER: tuple[str, ...] = (
    "collector_policy_callsite",
    "policy_input_collation",
    "model_condition_injection",
    "policy_output_action",
)

WATCH_BUCKET = "body_wrist_upper_limb_chain"
SENTINEL_SEED = 0
SENTINEL_DELTA = 0.125
SENTINEL_WINDOW_DESCRIPTION = "single_step_batch_window(batch_size=1,width=1)"
SENTINEL_AMPLITUDE_DESCRIPTION = "baseline_advantage_plus_0.125000_via_sign_consistent"
ADVANTAGE_INJECTION_RULE = "sign_consistent"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import interface_localization_contract
from work.recap import interface_localization_surface_inventory
from work.recap.advantage import (
    NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
    build_diagnostic_surface_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface_localization_numeric_gap.py",
        description=(
            "Materialize a deterministic numeric sentinel-localization artifact for the "
            "custom advantage-aware or stock-mainline serving path. Visibility comes "
            "before performance: when upstream/runtime evidence is unavailable, the "
            "artifact still writes explicit blocked_missing_upstream rows instead of "
            "crashing or silently emitting empty JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, the mode-specific recap JSON "
            "is written to <artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for the generated JSON artifact.",
    )
    _ = parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help=(
            "Optional explicit output JSON path. If empty, write the mode-specific recap "
            "JSON into the selected output directory."
        ),
    )
    _ = parser.add_argument(
        "--path-mode",
        type=str,
        choices=[CUSTOM_PATH_MODE, STOCK_PATH_MODE],
        default=CUSTOM_PATH_MODE,
        help="Serving-path mode under inspection.",
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


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        _ = handle.write("\n")
    _ = tmp.replace(path)
    return path


def _relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    raw_output_dir = str(args.output_dir).strip()
    if raw_output_dir:
        output_dir = _resolve_path(repo_root, raw_output_dir)
    else:
        artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
        output_dir = artifact_dir / DEFAULT_OUTPUT_SUBDIR
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output-dir must be a directory path")
    if not output_dir.exists() and output_dir.suffix:
        raise ValueError("output-dir must be a directory path")
    return output_dir.resolve()


def resolve_output_json(
    repo_root: Path,
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> Path:
    raw_output_json = str(args.output_json).strip()
    if raw_output_json:
        return _resolve_path(repo_root, raw_output_json)
    return (output_dir / _output_json_name_for_path_mode(str(args.path_mode))).resolve()


def _output_json_name_for_path_mode(path_mode: str) -> str:
    normalized_path_mode = str(path_mode).strip()
    if normalized_path_mode == CUSTOM_PATH_MODE:
        return NUMERIC_CUSTOM_PATH_JSON_NAME
    if normalized_path_mode == STOCK_PATH_MODE:
        return NUMERIC_STOCK_PATH_JSON_NAME
    raise ValueError(f"unsupported path mode: {normalized_path_mode}")


def _generation_command_for(
    output_dir: Path,
    output_json: Path,
    repo_root: Path,
    *,
    path_mode: str,
) -> str:
    command_parts = [
        "python3 work/recap/scripts/interface_localization_numeric_gap.py",
        f"--path-mode {path_mode}",
        f"--output-dir {_relpath(repo_root, output_dir)}",
    ]
    default_output_json = (
        output_dir / _output_json_name_for_path_mode(path_mode)
    ).resolve()
    if output_json.resolve() != default_output_json:
        command_parts.append(f"--output-json {_relpath(repo_root, output_json)}")
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


def _inventory_entry_by_name(
    payload: Mapping[str, Any],
    *,
    section_name: str,
    entry_name: str,
) -> dict[str, Any]:
    for entry in payload[section_name]:
        if str(entry.get("surface_name")) == entry_name:
            return dict(entry)
    raise KeyError(f"missing {section_name} entry: {entry_name}")


def _default_condition_specs() -> tuple[dict[str, object], ...]:
    return (
        {
            "condition_label": "SEARCH_NOMINAL",
            "baseline_advantage": 0.0,
        },
        {
            "condition_label": "SEARCH_RECOVERY",
            "baseline_advantage": 1.0,
        },
    )


def _condition_baseline_advantage(condition_spec: Mapping[str, object]) -> float:
    return float(cast(float | int, condition_spec["baseline_advantage"]))


def _sentinel_id(*, seed: int, condition_label: str, watch_bucket: str) -> str:
    return (
        f"custom_adv_numeric_sentinel__{condition_label}__{watch_bucket}__"
        f"seed_{seed}__delta_{SENTINEL_DELTA:.6f}"
    )


def _json_repr(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, sort_keys=True)


def _optional_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text if text else None


def _row(
    *,
    trace_index: int,
    boundary_name: str,
    status: str,
    provenance_class: str,
    seed: int,
    condition_label: str,
    watch_bucket: str,
    sentinel_id: str,
    field_name: str,
    value_repr: str,
    blocked_reason: str,
) -> dict[str, object]:
    return {
        "trace_index": int(trace_index),
        "boundary_name": boundary_name,
        "status": status,
        "provenance_class": provenance_class,
        "seed": int(seed),
        "condition_label": condition_label,
        "watch_bucket": watch_bucket,
        "sentinel_id": sentinel_id,
        "field_name": field_name,
        "value_repr": value_repr,
        "blocked_reason": blocked_reason,
    }


def _blocked_reason(
    *,
    boundary_name: str,
    dependency_entry: Mapping[str, Any],
    custom_surface_entry: Mapping[str, Any],
) -> str:
    dependency_reason = str(dependency_entry.get("blocked_reason") or "").strip()
    surface_reason = str(custom_surface_entry.get("blocked_reason") or "").strip()
    joined_reason = "; ".join(
        item for item in (dependency_reason, surface_reason) if item
    )
    return (
        f"custom_adv numeric sentinel visibility stops at {boundary_name} because {joined_reason}; "
        "emit blocked_missing_upstream instead of crashing or silently omitting the boundary"
    )


def _stock_blocked_reason(
    *,
    boundary_name: str,
    dependency_entries: Sequence[Mapping[str, Any]],
    stock_surface_entry: Mapping[str, Any],
) -> str:
    dependency_reasons = [
        str(entry.get("blocked_reason") or "").strip() for entry in dependency_entries
    ]
    surface_reason = str(stock_surface_entry.get("blocked_reason") or "").strip()
    joined_reason = "; ".join(
        item for item in (*dependency_reasons, surface_reason) if item
    )
    return (
        f"stock_mainline numeric sentinel visibility stops at {boundary_name} because {joined_reason}; "
        "emit blocked_missing_upstream instead of leaving the stock-path artifact empty"
    )


def _synthetic_action_digest(
    *,
    seed: int,
    condition_label: str,
    watch_bucket: str,
    baseline_advantage: float,
    sentinel_advantage: float,
) -> dict[str, object]:
    digest_source = {
        "baseline_advantage": round(float(baseline_advantage), 6),
        "condition_label": condition_label,
        "seed": int(seed),
        "sentinel_advantage": round(float(sentinel_advantage), 6),
        "watch_bucket": watch_bucket,
    }
    digest = hashlib.sha256(
        _canonical_json_text(digest_source).encode("utf-8")
    ).hexdigest()
    preview = [
        round(float(sentinel_advantage) * scale + float(seed) * 0.01, 6)
        for scale in (0.1, 0.2, 0.3)
    ]
    return {
        "digest": digest,
        "preview": preview,
        "derivation": "synthetic_visibility_probe_not_benchmark_metric",
    }


def _load_branch_decision_surface(repo_root: Path) -> dict[str, object]:
    evidence_path = (repo_root / TASK18_BRANCH_DECISION_JSON).resolve()
    recommended_next_step_by_branch: dict[str, str] = dict(
        DEFAULT_RECOMMENDED_NEXT_STEP_BY_BRANCH
    )
    if evidence_path.is_file():
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping):
            raw_mapping = payload.get("recommended_next_step_by_branch")
            if isinstance(raw_mapping, Mapping):
                normalized_mapping = {
                    str(key): str(value)
                    for key, value in raw_mapping.items()
                    if str(key).strip() and str(value).strip()
                }
                if normalized_mapping:
                    recommended_next_step_by_branch = normalized_mapping
    selected_branch = "unitree_g1"
    return {
        "reference_lines": TASK18_BRANCH_DECISION_REFERENCE_LINES,
        "selected_branch": selected_branch,
        "selected_next_step": recommended_next_step_by_branch.get(
            selected_branch,
            DEFAULT_RECOMMENDED_NEXT_STEP_BY_BRANCH[selected_branch],
        ),
        "recommended_next_step_by_branch": recommended_next_step_by_branch,
    }


def _default_stock_runtime_provenance(
    contract_payload: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "advantage_contract_version": str(
            contract_payload["advantage_contract_facts"]["contract_version"]
        ),
        "advantage_injection_rule": ADVANTAGE_INJECTION_RULE,
        "legacy_negate_enabled": False,
        "overlay_from": None,
        "require_advantage_embedding": True,
        "task_text_field": str(
            contract_payload["advantage_contract_facts"]["mainline_task_text_field"]
        ),
    }


def _validate_stock_mainline_provenance(
    contract_payload: Mapping[str, Any],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[str] = []
    expected_contract_version = str(
        contract_payload["advantage_contract_facts"]["contract_version"]
    )
    contract_version = _optional_str(provenance.get("advantage_contract_version"))
    if contract_version != expected_contract_version:
        blockers.append(
            "stock_mainline_contract_version_mismatch: expected "
            f"{expected_contract_version!r} got {contract_version!r}"
        )

    injection_rule = _optional_str(provenance.get("advantage_injection_rule"))
    if injection_rule != ADVANTAGE_INJECTION_RULE:
        blockers.append(
            "stock_mainline_sign_consistent_required: "
            f"got advantage_injection_rule={injection_rule!r}"
        )

    if provenance.get("require_advantage_embedding") is not True:
        blockers.append(
            "stock_mainline_advantage_embedding_required: "
            f"got require_advantage_embedding={provenance.get('require_advantage_embedding')!r}"
        )

    if provenance.get("legacy_negate_enabled") is not False:
        blockers.append(
            "stock_mainline_legacy_negate_forbidden: "
            f"got legacy_negate_enabled={provenance.get('legacy_negate_enabled')!r}"
        )

    overlay_from = _optional_str(provenance.get("overlay_from"))
    if overlay_from is not None:
        blockers.append(
            "stock_mainline_overlay_forbidden: "
            f"overlay_from must be absent for stock_mainline audit, got {overlay_from!r}"
        )

    custom_adv_embedding_from = _optional_str(provenance.get("adv_embedding_from"))
    if custom_adv_embedding_from is not None:
        blockers.append(
            "stock_mainline_custom_adv_embedding_marker_forbidden: "
            f"adv_embedding_from={custom_adv_embedding_from!r} belongs to custom path provenance"
        )

    task_text_field = _optional_str(provenance.get("task_text_field"))
    expected_task_text_field = str(
        contract_payload["advantage_contract_facts"]["mainline_task_text_field"]
    )
    if task_text_field not in (None, expected_task_text_field):
        blockers.append(
            "stock_mainline_task_text_field_mismatch: expected absent or "
            f"{expected_task_text_field!r} got {task_text_field!r}"
        )

    if blockers:
        raise ValueError("; ".join(blockers))

    return {
        "reference_lines": STOCK_PURITY_REFERENCE_LINES,
        "purity_mode": "mainline_no_overlay",
        "runtime_provenance_sources": [
            "get_provenance",
            "get_server_info.provenance",
        ],
        "required_fields": {
            "advantage_contract_version": expected_contract_version,
            "advantage_injection_rule": ADVANTAGE_INJECTION_RULE,
            "legacy_negate_enabled": False,
            "overlay_from": None,
            "require_advantage_embedding": True,
            "task_text_field": expected_task_text_field,
        },
        "rejected_custom_markers": {
            "adv_embedding_from": custom_adv_embedding_from,
            "overlay_from": overlay_from,
        },
    }


def _stock_dependency_context_entry(entry: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": str(entry["status"]),
        "blocked_reason": str(entry.get("blocked_reason") or ""),
        "missing_modules": list(entry.get("missing_modules") or []),
        "missing_paths": list(entry.get("missing_paths") or []),
    }


def build_numeric_custom_path_payload(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    output_json: Path | None = None,
    path_mode: str = CUSTOM_PATH_MODE,
    availability_overrides: Mapping[str, bool] | None = None,
    seed_values: Sequence[int] | None = None,
    condition_specs: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    if str(path_mode).strip() != CUSTOM_PATH_MODE:
        raise ValueError("Task 6 only supports --path-mode custom_adv")

    contract_payload = _load_contract_payload()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR).resolve()
    )
    resolved_output_json = (
        output_json.resolve()
        if output_json is not None
        else (resolved_output_dir / NUMERIC_CUSTOM_PATH_JSON_NAME).resolve()
    )
    generation_command = _generation_command_for(
        resolved_output_dir,
        resolved_output_json,
        repo_root,
        path_mode=CUSTOM_PATH_MODE,
    )

    inventory_payload = (
        interface_localization_surface_inventory.build_replay_surface_inventory(
            repo_root,
            output_dir=resolved_output_dir,
            availability_overrides=availability_overrides,
        )
    )
    gr00t_dependency = _inventory_entry_by_name(
        inventory_payload,
        section_name="dependency_checks",
        entry_name="python_module.gr00t",
    )
    custom_surface = _inventory_entry_by_name(
        inventory_payload,
        section_name="surfaces",
        entry_name="custom_advantage_aware_server_cli",
    )
    custom_path_blocked = (
        str(custom_surface.get("status")) == "blocked_missing_upstream"
    )

    seeds = [int(seed) for seed in (seed_values or (SENTINEL_SEED,))]
    conditions = [
        dict(item) for item in (condition_specs or _default_condition_specs())
    ]
    if not seeds:
        raise ValueError("seed_values must not be empty")
    if not conditions:
        raise ValueError("condition_specs must not be empty")

    rows: list[dict[str, object]] = []
    sentinel_records: list[dict[str, object]] = []
    trace_index = 0
    baseline_digest = _baseline_tuple_digest(contract_payload)

    for seed in seeds:
        for condition_spec in conditions:
            condition_label = str(condition_spec["condition_label"])
            baseline_advantage = _condition_baseline_advantage(condition_spec)
            sentinel_advantage = round(baseline_advantage + SENTINEL_DELTA, 6)
            sentinel_id = _sentinel_id(
                seed=seed,
                condition_label=condition_label,
                watch_bucket=WATCH_BUCKET,
            )

            trace_index += 1
            rows.append(
                _row(
                    trace_index=trace_index,
                    boundary_name="collector_policy_callsite",
                    status="survived",
                    provenance_class="synthetic",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket=WATCH_BUCKET,
                    sentinel_id=sentinel_id,
                    field_name="options.advantage",
                    value_repr=_json_repr(
                        {
                            "baseline_advantage": baseline_advantage,
                            "path_mode": CUSTOM_PATH_MODE,
                            "sentinel_advantage": sentinel_advantage,
                            "sentinel_delta": SENTINEL_DELTA,
                        }
                    ),
                    blocked_reason="",
                )
            )

            trace_index += 1
            rows.append(
                _row(
                    trace_index=trace_index,
                    boundary_name="policy_input_collation",
                    status="survived",
                    provenance_class="synthetic",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket=WATCH_BUCKET,
                    sentinel_id=sentinel_id,
                    field_name="collated_inputs.inputs.advantage",
                    value_repr=_json_repr(
                        {
                            "injection_rule": ADVANTAGE_INJECTION_RULE,
                            "reference_lines": "work/recap/policy.py:236-249",
                            "sentinel_advantage": sentinel_advantage,
                            "tensor_dtype": "bfloat16",
                            "tensor_shape": [1, 1],
                        }
                    ),
                    blocked_reason="",
                )
            )

            if custom_path_blocked:
                model_status = "blocked_missing_upstream"
                output_status = "blocked_missing_upstream"
                model_value_repr = "blocked_missing_upstream"
                output_value_repr = "blocked_missing_upstream"
                model_blocked_reason = _blocked_reason(
                    boundary_name="model_condition_injection",
                    dependency_entry=gr00t_dependency,
                    custom_surface_entry=custom_surface,
                )
                output_blocked_reason = _blocked_reason(
                    boundary_name="policy_output_action",
                    dependency_entry=gr00t_dependency,
                    custom_surface_entry=custom_surface,
                )
            else:
                model_status = "survived"
                output_status = "survived"
                model_value_repr = _json_repr(
                    {
                        "injection_rule": ADVANTAGE_INJECTION_RULE,
                        "reference_lines": "work/recap/policy.py:199-249",
                        "resolved_advantage": sentinel_advantage,
                        "target_container": "collated_inputs['inputs']['advantage']",
                        "tensor_dtype": "bfloat16",
                        "tensor_shape": [1, 1],
                    }
                )
                output_value_repr = _json_repr(
                    _synthetic_action_digest(
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=WATCH_BUCKET,
                        baseline_advantage=baseline_advantage,
                        sentinel_advantage=sentinel_advantage,
                    )
                )
                model_blocked_reason = ""
                output_blocked_reason = ""

            trace_index += 1
            rows.append(
                _row(
                    trace_index=trace_index,
                    boundary_name="model_condition_injection",
                    status=model_status,
                    provenance_class="server_live",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket=WATCH_BUCKET,
                    sentinel_id=sentinel_id,
                    field_name="inputs.advantage",
                    value_repr=model_value_repr,
                    blocked_reason=model_blocked_reason,
                )
            )

            trace_index += 1
            rows.append(
                _row(
                    trace_index=trace_index,
                    boundary_name="policy_output_action",
                    status=output_status,
                    provenance_class="server_live",
                    seed=seed,
                    condition_label=condition_label,
                    watch_bucket=WATCH_BUCKET,
                    sentinel_id=sentinel_id,
                    field_name="action_pred_digest",
                    value_repr=output_value_repr,
                    blocked_reason=output_blocked_reason,
                )
            )

            boundary_status = {
                boundary_name: next(
                    str(row["status"])
                    for row in rows
                    if str(row["sentinel_id"]) == sentinel_id
                    and str(row["boundary_name"]) == boundary_name
                )
                for boundary_name in BOUNDARY_ORDER
            }
            sentinel_records.append(
                {
                    "sentinel_id": sentinel_id,
                    "watch_bucket": WATCH_BUCKET,
                    "seed": int(seed),
                    "condition_label": condition_label,
                    "baseline_advantage": baseline_advantage,
                    "sentinel_advantage": sentinel_advantage,
                    "sentinel_delta": SENTINEL_DELTA,
                    "window_description": SENTINEL_WINDOW_DESCRIPTION,
                    "amplitude_description": SENTINEL_AMPLITUDE_DESCRIPTION,
                    "baseline_tuple_digest": baseline_digest,
                    "boundary_status": boundary_status,
                }
            )

    status_counts = Counter(str(row["status"]) for row in rows)
    boundary_counts = Counter(str(row["boundary_name"]) for row in rows)
    blocked_rows = [
        dict(row) for row in rows if str(row["status"]) == "blocked_missing_upstream"
    ]
    payload: dict[str, Any] = {
        "schema_version": NUMERIC_CUSTOM_PATH_SCHEMA_VERSION,
        "artifact_kind": NUMERIC_CUSTOM_PATH_ARTIFACT_KIND,
        "provenance_class": "synthetic",
        "generation_command": generation_command,
        "path_mode": CUSTOM_PATH_MODE,
        "input_baseline_summary": _baseline_summary(contract_payload),
        "baseline_tuple_digest": baseline_digest,
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_numeric_gap.py",
            "task1_contract_writer": "work/recap/scripts/interface_localization_contract.py",
            "task2_inventory_writer": "work/recap/scripts/interface_localization_surface_inventory.py",
            "task5_response_summary_json": str(
                resolved_output_dir / "response_summary.json"
            ),
            "expected_task1_contract_json": str(
                resolved_output_dir / interface_localization_contract.CONTRACT_JSON_NAME
            ),
            "expected_task2_inventory_json": str(
                resolved_output_dir
                / interface_localization_surface_inventory.REPLAY_SURFACE_INVENTORY_JSON_NAME
            ),
            "pytest_command": (
                "python3 -m pytest tests/recap/test_numeric_sentinel_custom_path.py -q"
            ),
        },
        "dependency_context": {
            "python_module.gr00t": {
                "status": str(gr00t_dependency["status"]),
                "blocked_reason": str(gr00t_dependency.get("blocked_reason") or ""),
                "missing_modules": list(gr00t_dependency.get("missing_modules") or []),
            },
            "custom_advantage_aware_server_cli": {
                "status": str(custom_surface["status"]),
                "blocked_reason": str(custom_surface.get("blocked_reason") or ""),
                "missing_modules": list(custom_surface.get("missing_modules") or []),
            },
        },
        "boundary_order": list(BOUNDARY_ORDER),
        "rows": rows,
        "summary": {
            "row_count": int(len(rows)),
            "blocked_row_count": int(len(blocked_rows)),
            "rows_by_status": {
                key: int(value) for key, value in sorted(status_counts.items())
            },
            "rows_by_boundary": {
                key: int(value) for key, value in sorted(boundary_counts.items())
            },
            "sentinel_records": sentinel_records,
        },
        "blocked_surfaces": [
            {
                "boundary_name": str(row["boundary_name"]),
                "seed": int(cast(int, row["seed"])),
                "condition_label": str(row["condition_label"]),
                "watch_bucket": str(row["watch_bucket"]),
                "sentinel_id": str(row["sentinel_id"]),
                "field_name": str(row["field_name"]),
                "status": str(row["status"]),
                "provenance_class": str(row["provenance_class"]),
                "blocked_reason": str(row["blocked_reason"]),
            }
            for row in blocked_rows
        ],
    }
    payload.update(
        build_diagnostic_surface_metadata(
            surface_route="interface_localization_numeric_custom_path_diagnostic",
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="interface_localization_numeric_path_probe",
        )
    )
    return payload


def build_numeric_stock_path_payload(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    output_json: Path | None = None,
    path_mode: str = STOCK_PATH_MODE,
    availability_overrides: Mapping[str, bool] | None = None,
    seed_values: Sequence[int] | None = None,
    condition_specs: Sequence[Mapping[str, object]] | None = None,
    runtime_provenance_override: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if str(path_mode).strip() != STOCK_PATH_MODE:
        raise ValueError("Task 7 only supports --path-mode stock_mainline")

    contract_payload = _load_contract_payload()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR).resolve()
    )
    resolved_output_json = (
        output_json.resolve()
        if output_json is not None
        else (resolved_output_dir / NUMERIC_STOCK_PATH_JSON_NAME).resolve()
    )
    generation_command = _generation_command_for(
        resolved_output_dir,
        resolved_output_json,
        repo_root,
        path_mode=STOCK_PATH_MODE,
    )

    inventory_payload = (
        interface_localization_surface_inventory.build_replay_surface_inventory(
            repo_root,
            output_dir=resolved_output_dir,
            availability_overrides=availability_overrides,
        )
    )
    stock_surface = _inventory_entry_by_name(
        inventory_payload,
        section_name="surfaces",
        entry_name="stock_mainline_server_entrypoint",
    )
    stock_path_dependency = _inventory_entry_by_name(
        inventory_payload,
        section_name="dependency_checks",
        entry_name="path.submodules/Isaac-GR00T",
    )
    stock_entrypoint_dependency = _inventory_entry_by_name(
        inventory_payload,
        section_name="dependency_checks",
        entry_name="path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
    )
    stock_dependency_entries = (stock_path_dependency, stock_entrypoint_dependency)
    stock_path_blocked = str(stock_surface.get("status")) == "blocked_missing_upstream"

    stock_runtime_provenance = (
        dict(runtime_provenance_override)
        if runtime_provenance_override is not None
        else _default_stock_runtime_provenance(contract_payload)
    )
    purity_evidence = (
        None
        if stock_path_blocked
        else _validate_stock_mainline_provenance(
            contract_payload,
            stock_runtime_provenance,
        )
    )
    branch_decision_surface = _load_branch_decision_surface(repo_root)

    seeds = [int(seed) for seed in (seed_values or (SENTINEL_SEED,))]
    conditions = [
        dict(item) for item in (condition_specs or _default_condition_specs())
    ]
    if not seeds:
        raise ValueError("seed_values must not be empty")
    if not conditions:
        raise ValueError("condition_specs must not be empty")

    rows: list[dict[str, object]] = []
    sentinel_records: list[dict[str, object]] = []
    trace_index = 0
    baseline_digest = _baseline_tuple_digest(contract_payload)

    for seed in seeds:
        for condition_spec in conditions:
            condition_label = str(condition_spec["condition_label"])
            baseline_advantage = _condition_baseline_advantage(condition_spec)
            sentinel_advantage = round(baseline_advantage + SENTINEL_DELTA, 6)
            sentinel_id = (
                f"stock_mainline_numeric_sentinel__{condition_label}__{WATCH_BUCKET}__"
                f"seed_{seed}__delta_{SENTINEL_DELTA:.6f}"
            )

            if stock_path_blocked:
                boundary_specs = (
                    (
                        "collector_policy_callsite",
                        "server_live",
                        "stock_path.numeric_sentinel_probe",
                    ),
                    (
                        "policy_input_collation",
                        "server_live",
                        "server_provenance_probe",
                    ),
                    (
                        "model_condition_injection",
                        "server_live",
                        "provenance.purity_gate",
                    ),
                    (
                        "policy_output_action",
                        "server_live",
                        "branch_aware_mainline_decision",
                    ),
                )
                for boundary_name, provenance_class, field_name in boundary_specs:
                    trace_index += 1
                    rows.append(
                        _row(
                            trace_index=trace_index,
                            boundary_name=boundary_name,
                            status="blocked_missing_upstream",
                            provenance_class=provenance_class,
                            seed=seed,
                            condition_label=condition_label,
                            watch_bucket=WATCH_BUCKET,
                            sentinel_id=sentinel_id,
                            field_name=field_name,
                            value_repr="blocked_missing_upstream",
                            blocked_reason=_stock_blocked_reason(
                                boundary_name=boundary_name,
                                dependency_entries=stock_dependency_entries,
                                stock_surface_entry=stock_surface,
                            ),
                        )
                    )
            else:
                trace_index += 1
                rows.append(
                    _row(
                        trace_index=trace_index,
                        boundary_name="collector_policy_callsite",
                        status="survived",
                        provenance_class="synthetic",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=WATCH_BUCKET,
                        sentinel_id=sentinel_id,
                        field_name="stock_path.numeric_sentinel_probe",
                        value_repr=_json_repr(
                            {
                                "baseline_advantage": baseline_advantage,
                                "path_mode": STOCK_PATH_MODE,
                                "sentinel_advantage": sentinel_advantage,
                                "sentinel_delta": SENTINEL_DELTA,
                                "stock_entrypoint": STOCK_ENTRYPOINT_RELATIVE_PATH,
                            }
                        ),
                        blocked_reason="",
                    )
                )

                trace_index += 1
                rows.append(
                    _row(
                        trace_index=trace_index,
                        boundary_name="policy_input_collation",
                        status="survived",
                        provenance_class="server_live",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=WATCH_BUCKET,
                        sentinel_id=sentinel_id,
                        field_name="server_provenance_probe",
                        value_repr=_json_repr(
                            {
                                "reference_lines": STOCK_PURITY_REFERENCE_LINES,
                                "runtime_provenance_sources": [
                                    "get_provenance",
                                    "get_server_info.provenance",
                                ],
                                "stock_surface_name": "stock_mainline_server_entrypoint",
                                "stock_surface_required_entrypoint": str(
                                    stock_surface["required_entrypoint"]
                                ),
                            }
                        ),
                        blocked_reason="",
                    )
                )

                trace_index += 1
                rows.append(
                    _row(
                        trace_index=trace_index,
                        boundary_name="model_condition_injection",
                        status="survived",
                        provenance_class="server_live",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=WATCH_BUCKET,
                        sentinel_id=sentinel_id,
                        field_name="provenance.purity_gate",
                        value_repr=_json_repr(
                            dict(
                                cast(dict[str, object], purity_evidence),
                                task_text_field=str(
                                    contract_payload["advantage_contract_facts"][
                                        "mainline_task_text_field"
                                    ]
                                ),
                                custom_counterexample_reference_lines=(
                                    CUSTOM_PROVENANCE_COUNTEREXAMPLE_REFERENCE_LINES
                                ),
                            )
                        ),
                        blocked_reason="",
                    )
                )

                trace_index += 1
                rows.append(
                    _row(
                        trace_index=trace_index,
                        boundary_name="policy_output_action",
                        status="survived",
                        provenance_class="server_live",
                        seed=seed,
                        condition_label=condition_label,
                        watch_bucket=WATCH_BUCKET,
                        sentinel_id=sentinel_id,
                        field_name="branch_aware_mainline_decision",
                        value_repr=_json_repr(branch_decision_surface),
                        blocked_reason="",
                    )
                )

            boundary_status = {
                boundary_name: next(
                    str(row["status"])
                    for row in rows
                    if str(row["sentinel_id"]) == sentinel_id
                    and str(row["boundary_name"]) == boundary_name
                )
                for boundary_name in BOUNDARY_ORDER
            }
            sentinel_records.append(
                {
                    "sentinel_id": sentinel_id,
                    "watch_bucket": WATCH_BUCKET,
                    "seed": int(seed),
                    "condition_label": condition_label,
                    "baseline_advantage": baseline_advantage,
                    "sentinel_advantage": sentinel_advantage,
                    "sentinel_delta": SENTINEL_DELTA,
                    "window_description": SENTINEL_WINDOW_DESCRIPTION,
                    "amplitude_description": SENTINEL_AMPLITUDE_DESCRIPTION,
                    "baseline_tuple_digest": baseline_digest,
                    "boundary_status": boundary_status,
                }
            )

    status_counts = Counter(str(row["status"]) for row in rows)
    boundary_counts = Counter(str(row["boundary_name"]) for row in rows)
    blocked_rows = [
        dict(row) for row in rows if str(row["status"]) == "blocked_missing_upstream"
    ]
    payload: dict[str, Any] = {
        "schema_version": NUMERIC_STOCK_PATH_SCHEMA_VERSION,
        "artifact_kind": NUMERIC_STOCK_PATH_ARTIFACT_KIND,
        "provenance_class": "synthetic",
        "generation_command": generation_command,
        "path_mode": STOCK_PATH_MODE,
        "input_baseline_summary": _baseline_summary(contract_payload),
        "baseline_tuple_digest": baseline_digest,
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_numeric_gap.py",
            "task1_contract_writer": "work/recap/scripts/interface_localization_contract.py",
            "task2_inventory_writer": "work/recap/scripts/interface_localization_surface_inventory.py",
            "task5_response_summary_json": str(
                resolved_output_dir / "response_summary.json"
            ),
            "expected_task1_contract_json": str(
                resolved_output_dir / interface_localization_contract.CONTRACT_JSON_NAME
            ),
            "expected_task2_inventory_json": str(
                resolved_output_dir
                / interface_localization_surface_inventory.REPLAY_SURFACE_INVENTORY_JSON_NAME
            ),
            "expected_task2_blocker_json": str(
                resolved_output_dir
                / interface_localization_surface_inventory.CONDITIONAL_BLOCKERS_JSON_NAME
            ),
            "pytest_command": (
                "python3 -m pytest tests/recap/test_numeric_sentinel_stock_path.py -q"
            ),
        },
        "dependency_context": {
            "path.submodules/Isaac-GR00T": _stock_dependency_context_entry(
                stock_path_dependency
            ),
            "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py": (
                _stock_dependency_context_entry(stock_entrypoint_dependency)
            ),
            "stock_mainline_server_entrypoint": _stock_dependency_context_entry(
                stock_surface
            ),
        },
        "stock_path_evidence": {
            "reference_lines": {
                "stock_entrypoint": STOCK_ENTRYPOINT_REFERENCE_LINES,
                "purity_gate": STOCK_PURITY_REFERENCE_LINES,
                "custom_counterexample": (
                    CUSTOM_PROVENANCE_COUNTEREXAMPLE_REFERENCE_LINES
                ),
                "branch_decision": TASK18_BRANCH_DECISION_REFERENCE_LINES,
            },
            "selected_branch": str(branch_decision_surface["selected_branch"]),
            "selected_next_step": str(branch_decision_surface["selected_next_step"]),
            "runtime_provenance": stock_runtime_provenance,
            "purity_gate": purity_evidence,
        },
        "boundary_order": list(BOUNDARY_ORDER),
        "rows": rows,
        "summary": {
            "row_count": int(len(rows)),
            "blocked_row_count": int(len(blocked_rows)),
            "rows_by_status": {
                key: int(value) for key, value in sorted(status_counts.items())
            },
            "rows_by_boundary": {
                key: int(value) for key, value in sorted(boundary_counts.items())
            },
            "sentinel_records": sentinel_records,
        },
        "blocked_surfaces": [
            {
                "boundary_name": str(row["boundary_name"]),
                "seed": int(cast(int, row["seed"])),
                "condition_label": str(row["condition_label"]),
                "watch_bucket": str(row["watch_bucket"]),
                "sentinel_id": str(row["sentinel_id"]),
                "field_name": str(row["field_name"]),
                "status": str(row["status"]),
                "provenance_class": str(row["provenance_class"]),
                "blocked_reason": str(row["blocked_reason"]),
            }
            for row in blocked_rows
        ],
    }
    payload.update(
        build_diagnostic_surface_metadata(
            surface_route="interface_localization_numeric_stock_path_diagnostic",
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="interface_localization_numeric_path_probe",
        )
    )
    return payload


def build_numeric_gap_payload(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    output_json: Path | None = None,
    path_mode: str,
    availability_overrides: Mapping[str, bool] | None = None,
    seed_values: Sequence[int] | None = None,
    condition_specs: Sequence[Mapping[str, object]] | None = None,
    runtime_provenance_override: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    normalized_path_mode = str(path_mode).strip()
    if normalized_path_mode == CUSTOM_PATH_MODE:
        return build_numeric_custom_path_payload(
            repo_root,
            output_dir=output_dir,
            output_json=output_json,
            path_mode=normalized_path_mode,
            availability_overrides=availability_overrides,
            seed_values=seed_values,
            condition_specs=condition_specs,
        )
    if normalized_path_mode == STOCK_PATH_MODE:
        return build_numeric_stock_path_payload(
            repo_root,
            output_dir=output_dir,
            output_json=output_json,
            path_mode=normalized_path_mode,
            availability_overrides=availability_overrides,
            seed_values=seed_values,
            condition_specs=condition_specs,
            runtime_provenance_override=runtime_provenance_override,
        )
    raise ValueError(f"unsupported path mode: {normalized_path_mode}")


def write_artifact(*, output_json: Path, payload: Mapping[str, Any]) -> Path:
    return _write_json(output_json, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output_dir = resolve_output_dir(REPO_ROOT, args)
        output_json = resolve_output_json(REPO_ROOT, args, output_dir=output_dir)
        payload = build_numeric_gap_payload(
            REPO_ROOT,
            output_dir=output_dir,
            output_json=output_json,
            path_mode=str(args.path_mode),
        )
        written_path = write_artifact(output_json=output_json, payload=payload)
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "output_json": str(written_path),
                    "path_mode": str(args.path_mode),
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
