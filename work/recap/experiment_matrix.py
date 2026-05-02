from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, cast


from work.recap import label_policy as recap_label_policy
from work.recap import run_manifest as recap_run_manifest
from work.recap import scope_experiment as recap_scope_experiment
from work.recap import state_conditioned_bucket_a_import
from work.recap.scripts import gr00t_baseline_freeze_matrix
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_MATRIX_SCHEMA_VERSION = "gr00t_experiment_matrix_v1"
EXPERIMENT_MATRIX_ARTIFACT_KIND = "gr00t_experiment_matrix"
DEFAULT_OUTPUT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json"
)

E1_ROW_ID = "g1_e1_text_indicator_s1"
E2_ROW_ID = "g1_e2_text_indicator_s2"
E3_ROW_ID = "g1_e3_text_indicator_s2_positive_duplication"
E4_ROW_ID = "g1_e4_text_indicator_s2_task_phase_epsilon"

STANDARD_EXPERIMENT_ROW_IDS: dict[str, str] = {
    "E1": E1_ROW_ID,
    "E2": E2_ROW_ID,
    "E3": E3_ROW_ID,
    "E4": E4_ROW_ID,
}
STANDARD_COMPARE_TO_ROW_IDS: dict[str, str] = {
    "E1": gr00t_baseline_freeze_matrix.B0_BASELINE_ID,
    "E2": E1_ROW_ID,
    "E3": E2_ROW_ID,
    "E4": E3_ROW_ID,
}
STANDARD_DISPLAY_LABEL_ORDER: tuple[str, ...] = ("B0", "B1", "E1", "E2", "E3", "E4")

PRIMARY_VARIABLE_AXES: tuple[str, ...] = (
    "text_indicator_carrier",
    "scope_preset",
    "positive_duplication_policy",
    "task_phase_aware_epsilon",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _resolve_path(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw.resolve()


def _rel_repo(path: Path | str) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    resolved = _resolve_path(path)
    if resolved.exists():
        raise ValueError(
            f"experiment matrix output already exists (no-overwrite): {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return state_conditioned_bucket_a_import._write_json(resolved, payload)


def _read_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(_resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return cast(dict[str, Any], dict(payload))


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _as_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _as_path(value: object, *, field_name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(
            f"{field_name} must be a path-like string, got {type(value).__name__}"
        )
    return _resolve_path(Path(value))


def _normalize_branch_key(branch: object) -> str:
    text = _as_str(branch, field_name="branch")
    lowered = text.lower()
    if lowered == "unitree_g1":
        return "unitree_g1"
    return lowered


def _disallowed_machine_ids() -> list[str]:
    return list(
        STANDARD_DISPLAY_LABEL_ORDER
        + tuple(sorted({*STANDARD_EXPERIMENT_ROW_IDS.keys()}))
    )


def _validate_machine_row_id(*, row_id: str, display_label: str) -> None:
    if row_id in _disallowed_machine_ids() or row_id == display_label:
        raise ValueError(
            f"row_id must be namespaced and must not reuse display label {display_label!r}"
        )


def _normalize_experiment_row_spec(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    display_label = _as_str(spec.get("display_label"), field_name="display_label")
    row_id = spec.get("row_id")
    if row_id is None:
        row_id = STANDARD_EXPERIMENT_ROW_IDS.get(display_label)
    normalized_row_id = _as_str(row_id, field_name=f"{display_label}.row_id")
    _validate_machine_row_id(row_id=normalized_row_id, display_label=display_label)
    compare_to_row_id = spec.get("compare_to_row_id")
    if compare_to_row_id is None:
        compare_to_row_id = STANDARD_COMPARE_TO_ROW_IDS.get(display_label)
    normalized_compare_to = _as_str(
        compare_to_row_id,
        field_name=f"{display_label}.compare_to_row_id",
    )
    return {
        "display_label": display_label,
        "row_id": normalized_row_id,
        "compare_to_row_id": normalized_compare_to,
        "run_manifest_path": _as_path(
            spec.get("run_manifest_path"),
            field_name=f"{display_label}.run_manifest_path",
        ),
        "triplet_summary_path": _as_path(
            spec.get("triplet_summary_path"),
            field_name=f"{display_label}.triplet_summary_path",
        ),
    }


def build_experiment_row_spec(
    *,
    display_label: str,
    run_manifest_path: Path | str,
    triplet_summary_path: Path | str,
    compare_to_row_id: str | None = None,
    row_id: str | None = None,
) -> dict[str, str]:
    payload: dict[str, str] = {
        "display_label": str(display_label),
        "run_manifest_path": str(run_manifest_path),
        "triplet_summary_path": str(triplet_summary_path),
    }
    if compare_to_row_id is not None:
        payload["compare_to_row_id"] = str(compare_to_row_id)
    if row_id is not None:
        payload["row_id"] = str(row_id)
    return payload


def _baseline_axis_values() -> dict[str, Any]:
    return {
        "text_indicator_carrier": {
            "mainline_text_indicator": False,
            "carrier_schema_version": None,
            "carrier_route": None,
            "prompt_source_field": None,
            "indicator_source": None,
        },
        "scope_preset": {
            "preset_id": None,
            "coverage": None,
            "covered_trainable_component_ids": [],
            "uncovered_trainable_component_ids": [],
        },
        "positive_duplication_policy": {
            "enabled": False,
            "factor": 1,
            "distinct_factor_values": [],
        },
        "task_phase_aware_epsilon": {
            "enabled": False,
            "overall_distinct_values": [],
            "task_view": [],
            "phase_view": [],
            "task_phase_view": [],
        },
    }


def _normalize_scope_axis(scope_extension: Mapping[str, Any]) -> dict[str, Any]:
    current_eval_lane = _as_mapping(
        scope_extension.get("current_eval_lane", {}),
        field_name="scope_experiment.current_eval_lane",
    )
    return {
        "preset_id": _as_str(
            scope_extension.get("preset_id"), field_name="scope_experiment.preset_id"
        ),
        "coverage": current_eval_lane.get("coverage"),
        "covered_trainable_component_ids": list(
            current_eval_lane.get("covered_trainable_component_ids", [])
        ),
        "uncovered_trainable_component_ids": list(
            current_eval_lane.get("uncovered_trainable_component_ids", [])
        ),
    }


def _epsilon_view_signature(
    items: Sequence[object],
    *,
    key_fields: Sequence[str],
    field_name: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        mapping = _as_mapping(item, field_name=f"{field_name}[{index}]")
        epsilon_summary = _as_mapping(
            mapping.get("epsilon_summary", {}),
            field_name=f"{field_name}[{index}].epsilon_summary",
        )
        row: dict[str, Any] = {
            key: mapping.get(key) for key in key_fields if key in mapping
        }
        row["distinct_values"] = list(epsilon_summary.get("distinct_values", []))
        row["all_equal"] = bool(epsilon_summary.get("all_equal", False))
        row["constant_value"] = epsilon_summary.get("constant_value")
        normalized.append(row)
    normalized.sort(
        key=lambda item: tuple(
            str(item.get(field_name, "")) for field_name in key_fields
        )
    )
    return normalized


def _has_task_phase_aware_epsilon(signature: Mapping[str, Any]) -> bool:
    overall_values = list(signature.get("overall_distinct_values", []))
    if len(overall_values) > 1:
        return True
    for view_name in ("task_view", "phase_view", "task_phase_view"):
        for item in cast(list[Mapping[str, Any]], signature.get(view_name, [])):
            distinct_values = list(item.get("distinct_values", []))
            if len(distinct_values) > 1:
                return True
    return False


def _normalize_task_phase_epsilon_axis(
    label_policy_extension: Mapping[str, Any],
) -> dict[str, Any]:
    overall_epsilon_summary = _as_mapping(
        label_policy_extension.get("overall_epsilon_summary", {}),
        field_name="label_policy.overall_epsilon_summary",
    )
    signature = {
        "overall_distinct_values": list(
            overall_epsilon_summary.get("distinct_values", [])
        ),
        "task_view": _epsilon_view_signature(
            _as_list(
                label_policy_extension.get("task_aware_epsilon_view", []),
                field_name="label_policy.task_aware_epsilon_view",
            ),
            key_fields=("task",),
            field_name="label_policy.task_aware_epsilon_view",
        ),
        "phase_view": _epsilon_view_signature(
            _as_list(
                label_policy_extension.get("phase_aware_epsilon_view", []),
                field_name="label_policy.phase_aware_epsilon_view",
            ),
            key_fields=("phase",),
            field_name="label_policy.phase_aware_epsilon_view",
        ),
        "task_phase_view": _epsilon_view_signature(
            _as_list(
                label_policy_extension.get("task_phase_aware_epsilon_view", []),
                field_name="label_policy.task_phase_aware_epsilon_view",
            ),
            key_fields=("task", "phase"),
            field_name="label_policy.task_phase_aware_epsilon_view",
        ),
    }
    enabled = _has_task_phase_aware_epsilon(signature)
    if not enabled:
        return {
            "enabled": False,
            "overall_distinct_values": [],
            "task_view": [],
            "phase_view": [],
            "task_phase_view": [],
        }
    return {
        **signature,
        "enabled": True,
    }


def _normalize_axis_values(
    *,
    manifest: Mapping[str, Any],
    scope_extension: Mapping[str, Any],
    label_policy_extension: Mapping[str, Any],
) -> dict[str, Any]:
    core = _as_mapping(manifest.get("core", {}), field_name="run_manifest.core")
    positive_duplication_policy = _as_mapping(
        label_policy_extension.get("positive_duplication_policy", {}),
        field_name="label_policy.positive_duplication_policy",
    )
    return {
        "text_indicator_carrier": {
            "mainline_text_indicator": True,
            "carrier_schema_version": core.get("carrier_schema_version"),
            "carrier_route": core.get("carrier_route"),
            "prompt_source_field": core.get("prompt_source_field"),
            "indicator_source": core.get("indicator_source"),
        },
        "scope_preset": _normalize_scope_axis(scope_extension),
        "positive_duplication_policy": {
            "enabled": bool(positive_duplication_policy.get("enabled", False)),
            "factor": int(positive_duplication_policy.get("factor", 1)),
            "distinct_factor_values": list(
                positive_duplication_policy.get("distinct_factor_values", [])
            ),
        },
        "task_phase_aware_epsilon": _normalize_task_phase_epsilon_axis(
            label_policy_extension
        ),
    }


def _baseline_row(
    *,
    baseline_id: str,
    display_label: str,
    baseline_payload: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_entry = _as_mapping(
        _as_mapping(baseline_payload.get("baselines", {}), field_name="baselines").get(
            baseline_id
        ),
        field_name=f"baselines.{baseline_id}",
    )
    summary = dict(_as_mapping(baseline_entry.get("summary", {}), field_name="summary"))
    mainline_authority = bool(baseline_entry.get("mainline_authority", False))
    comparability_level = (
        "baseline_authority" if mainline_authority else "legacy_negative_control_only"
    )
    blockers = [] if mainline_authority else ["legacy_negative_control_only"]
    return {
        "row_id": baseline_id,
        "display_label": display_label,
        "row_kind": "baseline",
        "branch_key": baseline_entry.get("branch_key", "unitree_g1"),
        "mainline_authority": mainline_authority,
        "legacy_backpointer_only": bool(
            baseline_entry.get("legacy_backpointer_only", False)
        ),
        "compare_to_row_id": None,
        "changed_axes": [],
        "migration_only": False,
        "attribution_allowed": False,
        "comparability_level": comparability_level,
        "attribution_blockers": blockers,
        "axis_values": _baseline_axis_values(),
        "backpointers": {
            "baseline_id": baseline_id,
            "baseline_freeze_json_field": f"baselines.{baseline_id}",
        },
        "summary": summary,
    }


def _validate_baseline_freeze_payload(
    baseline_payload: Mapping[str, Any],
) -> None:
    if (
        baseline_payload.get("schema_version")
        != gr00t_baseline_freeze_matrix.REPORT_SCHEMA_VERSION
    ):
        raise ValueError("baseline freeze schema_version mismatch")
    if (
        baseline_payload.get("artifact_kind")
        != gr00t_baseline_freeze_matrix.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("baseline freeze artifact_kind mismatch")
    baselines = _as_mapping(
        baseline_payload.get("baselines", {}), field_name="baselines"
    )
    for baseline_id in (
        gr00t_baseline_freeze_matrix.B0_BASELINE_ID,
        gr00t_baseline_freeze_matrix.B1_BASELINE_ID,
    ):
        if baseline_id not in baselines:
            raise ValueError(f"baseline freeze missing required baseline {baseline_id}")


def _validate_triplet_summary_payload(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version")
        != gr00t_same_checkpoint_triplet_eval.REPORT_SCHEMA_VERSION
    ):
        raise ValueError("triplet summary schema_version mismatch")
    if (
        payload.get("artifact_kind")
        != gr00t_same_checkpoint_triplet_eval.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("triplet summary artifact_kind mismatch")
    if not isinstance(payload.get("action_delta_audit"), Mapping):
        raise ValueError("triplet summary must expose action_delta_audit")


def _experiment_row(
    *,
    spec: Mapping[str, Any],
    run_manifest_payload: Mapping[str, Any],
    triplet_summary_payload: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_spec = _normalize_experiment_row_spec(spec)
    compare_to_row_id = str(normalized_spec["compare_to_row_id"])
    if compare_to_row_id not in rows:
        raise ValueError(
            f"comparison row {compare_to_row_id!r} must already exist before building {normalized_spec['display_label']}"
        )
    validation = recap_run_manifest.validate_run_manifest(
        run_manifest_payload,
        repo_root=REPO_ROOT,
    )
    normalized_manifest = _as_mapping(
        validation.get("normalized_manifest", {}),
        field_name="normalized_manifest",
    )
    extensions = _as_mapping(
        normalized_manifest.get("extensions", {}),
        field_name="run_manifest.extensions",
    )
    scope_extension = extensions.get(
        recap_scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY
    )
    if not isinstance(scope_extension, Mapping):
        raise ValueError(
            f"{normalized_spec['display_label']} requires extensions.{recap_scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY}"
        )
    label_policy_extension = extensions.get(
        recap_label_policy.LABEL_POLICY_EXTENSION_KEY
    )
    if not isinstance(label_policy_extension, Mapping):
        raise ValueError(
            f"{normalized_spec['display_label']} requires extensions.{recap_label_policy.LABEL_POLICY_EXTENSION_KEY}"
        )
    _validate_triplet_summary_payload(triplet_summary_payload)
    normalized_scope_extension = (
        recap_scope_experiment.normalize_scope_experiment_payload(
            scope_extension,
            field_name=f"{normalized_spec['display_label']}.scope_experiment",
        )
    )
    normalized_label_policy_extension = (
        recap_label_policy.normalize_label_policy_payload(
            label_policy_extension,
            field_name=f"{normalized_spec['display_label']}.label_policy",
        )
    )
    axis_values = _normalize_axis_values(
        manifest=normalized_manifest,
        scope_extension=normalized_scope_extension,
        label_policy_extension=normalized_label_policy_extension,
    )
    compare_row = _as_mapping(rows.get(compare_to_row_id), field_name=compare_to_row_id)
    reference_axis_values = _as_mapping(
        compare_row.get("axis_values", {}),
        field_name=f"{compare_to_row_id}.axis_values",
    )
    changed_axes = [
        axis_name
        for axis_name in PRIMARY_VARIABLE_AXES
        if deepcopy(axis_values.get(axis_name))
        != deepcopy(reference_axis_values.get(axis_name))
    ]

    attribution_blockers: list[str] = []
    if validation.get("formal_eligibility") != "ALLOW":
        attribution_blockers.append("run_manifest_formal_eligibility_blocked")
    if triplet_summary_payload.get("formal_eligibility") == "BLOCK":
        attribution_blockers.append("triplet_summary_formal_eligibility_blocked")
    triplet_gate = triplet_summary_payload.get("triplet_gate")
    if (
        isinstance(triplet_gate, Mapping)
        and triplet_gate.get("formal_eligibility") == "BLOCK"
    ):
        attribution_blockers.append("triplet_binding_gate_blocked")
    action_delta_audit = _as_mapping(
        triplet_summary_payload.get("action_delta_audit", {}),
        field_name="triplet_summary.action_delta_audit",
    )
    if action_delta_audit.get("audit_status") != "READY":
        attribution_blockers.append("action_delta_audit_not_ready")
    if not bool(compare_row.get("mainline_authority", False)):
        attribution_blockers.append("reference_row_not_mainline_authority")
    if not changed_axes:
        attribution_blockers.append("no_primary_axis_change_detected")
    if len(changed_axes) > 1:
        attribution_blockers.append("multiple_primary_axes_changed")

    migration_only = len(changed_axes) > 1
    scope_axis = _as_mapping(
        axis_values.get("scope_preset", {}), field_name="scope_preset"
    )
    coverage = scope_axis.get("coverage")
    if attribution_blockers and not (
        migration_only and attribution_blockers == ["multiple_primary_axes_changed"]
    ):
        comparability_level = "blocked"
    elif migration_only:
        comparability_level = "migration_only"
    elif coverage == "partial_action_head_only":
        comparability_level = "partial_action_head_only"
    else:
        comparability_level = "full"

    attribution_allowed = len(changed_axes) == 1 and not {
        "run_manifest_formal_eligibility_blocked",
        "triplet_summary_formal_eligibility_blocked",
        "triplet_binding_gate_blocked",
        "action_delta_audit_not_ready",
        "reference_row_not_mainline_authority",
        "no_primary_axis_change_detected",
        "multiple_primary_axes_changed",
    }.intersection(attribution_blockers)

    backpointers = {
        "run_manifest_path": _rel_repo(
            cast(Path, normalized_spec["run_manifest_path"])
        ),
        "scope_experiment": {
            "json_field": f"extensions.{recap_scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY}",
            "preset_id": normalized_scope_extension.get("preset_id"),
            "schema_version": normalized_scope_extension.get("schema_version"),
        },
        "label_policy": {
            "json_field": f"extensions.{recap_label_policy.LABEL_POLICY_EXTENSION_KEY}",
            "artifact_kind": normalized_label_policy_extension.get("artifact_kind"),
            "schema_version": normalized_label_policy_extension.get("schema_version"),
            "positive_duplication_enabled": axis_values["positive_duplication_policy"][
                "enabled"
            ],
            "task_phase_aware_epsilon_enabled": axis_values["task_phase_aware_epsilon"][
                "enabled"
            ],
        },
        "triplet_summary": {
            "path": _rel_repo(cast(Path, normalized_spec["triplet_summary_path"])),
            "artifact_kind": triplet_summary_payload.get("artifact_kind"),
            "schema_version": triplet_summary_payload.get("schema_version"),
            "report_signature_sha256": triplet_summary_payload.get(
                "report_signature_sha256"
            ),
            "same_checkpoint_locked": triplet_summary_payload.get(
                "same_checkpoint_locked"
            ),
        },
        "action_delta_audit": {
            "path": _rel_repo(cast(Path, normalized_spec["triplet_summary_path"])),
            "json_field": "action_delta_audit",
            "audit_status": action_delta_audit.get("audit_status"),
        },
    }

    core = _as_mapping(
        normalized_manifest.get("core", {}), field_name="run_manifest.core"
    )
    branch_key = _normalize_branch_key(core.get("branch"))
    return {
        "row_id": normalized_spec["row_id"],
        "display_label": normalized_spec["display_label"],
        "row_kind": "experiment",
        "branch_key": branch_key,
        "mainline_authority": True,
        "compare_to_row_id": compare_to_row_id,
        "changed_axes": changed_axes,
        "migration_only": migration_only,
        "attribution_allowed": attribution_allowed,
        "comparability_level": comparability_level,
        "attribution_blockers": attribution_blockers,
        "axis_values": axis_values,
        "backpointers": backpointers,
        "summary": {
            "scope_preset_id": axis_values["scope_preset"]["preset_id"],
            "scope_eval_coverage": axis_values["scope_preset"]["coverage"],
            "positive_duplication_enabled": axis_values["positive_duplication_policy"][
                "enabled"
            ],
            "task_phase_aware_epsilon_enabled": axis_values["task_phase_aware_epsilon"][
                "enabled"
            ],
            "run_manifest_core_digest": validation.get("core_digest"),
            "triplet_formal_eligibility": triplet_summary_payload.get(
                "formal_eligibility", "ALLOW"
            ),
            "triplet_gate_formal_eligibility": triplet_gate.get("formal_eligibility")
            if isinstance(triplet_gate, Mapping)
            else None,
            "action_delta_audit_status": action_delta_audit.get("audit_status"),
        },
    }


def materialize_experiment_matrix(
    *,
    baseline_freeze_payload: Mapping[str, Any],
    experiment_row_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_baseline_freeze_payload(baseline_freeze_payload)

    rows: dict[str, dict[str, Any]] = {}
    display_rows: list[dict[str, str]] = []

    b0_row = _baseline_row(
        baseline_id=gr00t_baseline_freeze_matrix.B0_BASELINE_ID,
        display_label=gr00t_baseline_freeze_matrix.DISPLAY_LABEL_B0,
        baseline_payload=baseline_freeze_payload,
    )
    b1_row = _baseline_row(
        baseline_id=gr00t_baseline_freeze_matrix.B1_BASELINE_ID,
        display_label=gr00t_baseline_freeze_matrix.DISPLAY_LABEL_B1,
        baseline_payload=baseline_freeze_payload,
    )
    for row in (b0_row, b1_row):
        rows[str(row["row_id"])] = row
        display_rows.append(
            {
                "display_label": str(row["display_label"]),
                "row_id": str(row["row_id"]),
            }
        )

    normalized_specs = [
        _normalize_experiment_row_spec(spec) for spec in experiment_row_specs
    ]
    seen_row_ids = set(rows)
    seen_display_labels = {row["display_label"] for row in display_rows}
    for spec in normalized_specs:
        row_id = str(spec["row_id"])
        display_label = str(spec["display_label"])
        if row_id in seen_row_ids:
            raise ValueError(f"duplicate experiment row_id {row_id!r}")
        if display_label in seen_display_labels:
            raise ValueError(f"duplicate display_label {display_label!r}")
        run_manifest_payload = _read_json(cast(Path, spec["run_manifest_path"]))
        triplet_summary_payload = _read_json(cast(Path, spec["triplet_summary_path"]))
        row = _experiment_row(
            spec=spec,
            run_manifest_payload=run_manifest_payload,
            triplet_summary_payload=triplet_summary_payload,
            rows=rows,
        )
        rows[row_id] = row
        display_rows.append({"display_label": display_label, "row_id": row_id})
        seen_row_ids.add(row_id)
        seen_display_labels.add(display_label)

    ordered_display_labels = [
        label for label in STANDARD_DISPLAY_LABEL_ORDER if label in seen_display_labels
    ]
    ordered_display_labels.extend(
        row["display_label"]
        for row in display_rows
        if row["display_label"] not in ordered_display_labels
    )
    display_row_map = {row["display_label"]: row for row in display_rows}
    ordered_display_rows = [display_row_map[label] for label in ordered_display_labels]

    row_id_order = [
        display_row_map[label]["row_id"] for label in ordered_display_labels
    ]
    return {
        "schema_version": EXPERIMENT_MATRIX_SCHEMA_VERSION,
        "artifact_kind": EXPERIMENT_MATRIX_ARTIFACT_KIND,
        "generated_at": _now_iso(),
        "machine_id_policy": {
            "display_labels_are_not_machine_ids": True,
            "namespaced_machine_ids_required": True,
            "disallowed_machine_ids": _disallowed_machine_ids(),
            "standard_experiment_row_ids": dict(STANDARD_EXPERIMENT_ROW_IDS),
        },
        "attribution_policy": {
            "reference_mode": "compare_to_row_id",
            "primary_variable_axes": list(PRIMARY_VARIABLE_AXES),
            "extension_only_diffs_count_as_real_changes": True,
            "multi_variable_rows_are_migration_only": True,
            "partial_scope_coverage_levels": ["partial_action_head_only"],
        },
        "display_rows": ordered_display_rows,
        "row_id_order": row_id_order,
        "rows": rows,
        "report_signature_sha256": _sha256_payload(
            {
                "schema_version": EXPERIMENT_MATRIX_SCHEMA_VERSION,
                "artifact_kind": EXPERIMENT_MATRIX_ARTIFACT_KIND,
                "display_rows": ordered_display_rows,
                "row_id_order": row_id_order,
                "rows": rows,
            }
        ),
    }


def write_experiment_matrix(
    *,
    output_path: Path | str,
    baseline_freeze_payload: Mapping[str, Any],
    experiment_row_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = materialize_experiment_matrix(
        baseline_freeze_payload=baseline_freeze_payload,
        experiment_row_specs=experiment_row_specs,
    )
    written = _write_json(_resolve_path(output_path), payload)
    return {
        "output_path": str(written),
        "row_count": len(payload["row_id_order"]),
        "display_labels": [row["display_label"] for row in payload["display_rows"]],
    }


def write_experiment_matrix_from_paths(
    *,
    baseline_freeze_json: Path | str,
    experiment_spec_json: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    baseline_payload = _read_json(baseline_freeze_json)
    experiment_spec_payload = _read_json(experiment_spec_json)
    experiment_rows = _as_list(
        experiment_spec_payload.get("experiment_rows", []),
        field_name="experiment_rows",
    )
    return write_experiment_matrix(
        output_path=output_path,
        baseline_freeze_payload=baseline_payload,
        experiment_row_specs=[
            _as_mapping(item, field_name=f"experiment_rows[{index}]")
            for index, item in enumerate(experiment_rows)
        ],
    )


__all__ = [
    "DEFAULT_OUTPUT",
    "E1_ROW_ID",
    "E2_ROW_ID",
    "E3_ROW_ID",
    "E4_ROW_ID",
    "EXPERIMENT_MATRIX_ARTIFACT_KIND",
    "EXPERIMENT_MATRIX_SCHEMA_VERSION",
    "PRIMARY_VARIABLE_AXES",
    "STANDARD_COMPARE_TO_ROW_IDS",
    "STANDARD_DISPLAY_LABEL_ORDER",
    "STANDARD_EXPERIMENT_ROW_IDS",
    "build_experiment_row_spec",
    "materialize_experiment_matrix",
    "write_experiment_matrix",
    "write_experiment_matrix_from_paths",
]
