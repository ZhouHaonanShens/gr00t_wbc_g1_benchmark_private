from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_MODALITY_CONFIG_PATH = Path("work/configs/new_embodiment/modality_config.json")
DEFAULT_BRANCH_MANIFEST_PATH = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/branch_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/controller_audit_new_embodiment.json"
)

REPORT_SCHEMA_VERSION = "gr00t_controller_audit_new_embodiment_v1"
REPORT_ARTIFACT_KIND = "gr00t_controller_audit_new_embodiment"
BRANCH_MANIFEST_SCHEMA_VERSION = "gr00t_new_embodiment_branch_manifest_v1"
BRANCH_MANIFEST_ARTIFACT_KIND = "gr00t_new_embodiment_branch_manifest"
FAILURE_NOTE_MARKDOWN_NAME = "controller_audit_new_embodiment_failure_note.md"

EXPECTED_BRANCH_TAG = "NEW_EMBODIMENT"
EXPECTED_SERVER_MODALITY_CONFIG_KEYS = ["action", "language", "state", "video"]
EXPECTED_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
}
EXPECTED_ACTION_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "waist": 3,
    "base_height_command": 1,
    "navigate_command": 3,
}

REQUIRED_MANIFEST_PROVENANCE_FIELDS: tuple[str, ...] = (
    "modality_config_path",
    "modality_config_fingerprint_sha256",
    "normalization_source",
    "controller_provenance",
    "dataset_provenance",
    "relative_action_policy",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_controller_audit_new_embodiment.py",
        description=(
            "Freeze and audit the repo-local NEW_EMBODIMENT branch contract so later "
            "tasks consume one explicit modality config, one explicit branch manifest, "
            "and one explicit non-public-comparable controller provenance surface."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--modality-config-path",
        type=Path,
        default=DEFAULT_MODALITY_CONFIG_PATH,
        help="Frozen repo-local JSON modality config for NEW_EMBODIMENT branch audit.",
    )
    _ = parser.add_argument(
        "--branch-manifest-path",
        type=Path,
        default=DEFAULT_BRANCH_MANIFEST_PATH,
        help=(
            "Canonical NEW_EMBODIMENT branch manifest path. When missing, the script "
            "materializes a frozen manifest from the modality config. When present, it "
            "must already contain complete provenance."
        ),
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Machine-readable NEW_EMBODIMENT audit JSON output path.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _validate_json_output_path(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f"{arg_name} must be a file path, got existing directory: {resolved}"
        )
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"{arg_name} must be a file path: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _rel_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object in {path}, got {type(payload).__name__}"
        )
    return dict(payload)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_of_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _as_list_of_strings(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(
                f"{field_name}[{index}] must be a string, got {type(item).__name__}"
            )
        normalized = item.strip()
        if not normalized:
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
        result.append(normalized)
    return result


def _as_list_of_ints(value: object, *, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"{field_name}[{index}] must be an int, got {type(item).__name__}"
            )
        result.append(int(item))
    return result


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _optional_non_empty_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field_name=field_name)


def _dict_of_positive_ints(value: object, *, field_name: str) -> dict[str, int]:
    mapping = _as_mapping(value, field_name=field_name)
    result: dict[str, int] = {}
    for key, raw in mapping.items():
        normalized_key = _non_empty_string(key, field_name=f"{field_name}.key")
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(
                f"{field_name}.{normalized_key} must be an int, got {type(raw).__name__}"
            )
        int_value = int(raw)
        if int_value <= 0:
            raise ValueError(f"{field_name}.{normalized_key} must be > 0")
        result[normalized_key] = int_value
    return result


def load_modality_contract(modality_config_path: Path) -> dict[str, Any]:
    payload = _read_json(modality_config_path)
    branch_tag = _non_empty_string(
        payload.get("branch_tag"), field_name="modality_config.branch_tag"
    )
    embodiment_tag = _non_empty_string(
        payload.get("embodiment_tag"), field_name="modality_config.embodiment_tag"
    )
    if branch_tag != EXPECTED_BRANCH_TAG or embodiment_tag != EXPECTED_BRANCH_TAG:
        raise ValueError(
            "modality_config branch_tag/embodiment_tag must both equal NEW_EMBODIMENT"
        )

    intended_usage = _as_mapping(
        payload.get("intended_usage"), field_name="modality_config.intended_usage"
    )
    modalities = _as_mapping(
        payload.get("modalities"), field_name="modality_config.modalities"
    )
    required_modalities = tuple(EXPECTED_SERVER_MODALITY_CONFIG_KEYS)
    missing_modalities = [key for key in required_modalities if key not in modalities]
    if missing_modalities:
        raise ValueError(
            f"modality_config.modalities missing required keys: {missing_modalities}"
        )

    state = _as_mapping(
        modalities["state"], field_name="modality_config.modalities.state"
    )
    action = _as_mapping(
        modalities["action"], field_name="modality_config.modalities.action"
    )
    video = _as_mapping(
        modalities["video"], field_name="modality_config.modalities.video"
    )
    language = _as_mapping(
        modalities["language"], field_name="modality_config.modalities.language"
    )

    state_order_expected = _as_list_of_strings(
        state.get("modality_keys"),
        field_name="modality_config.modalities.state.modality_keys",
    )
    action_order_expected = _as_list_of_strings(
        action.get("modality_keys"),
        field_name="modality_config.modalities.action.modality_keys",
    )
    state_dims_expected = _dict_of_positive_ints(
        state.get("dims_by_key"),
        field_name="modality_config.modalities.state.dims_by_key",
    )
    action_dims_expected = _dict_of_positive_ints(
        action.get("dims_by_key"),
        field_name="modality_config.modalities.action.dims_by_key",
    )
    state_horizon = len(
        _as_list_of_ints(
            state.get("delta_indices"),
            field_name="modality_config.modalities.state.delta_indices",
        )
    )
    action_horizon = len(
        _as_list_of_ints(
            action.get("delta_indices"),
            field_name="modality_config.modalities.action.delta_indices",
        )
    )
    _ = _as_list_of_ints(
        video.get("delta_indices"),
        field_name="modality_config.modalities.video.delta_indices",
    )
    _ = _as_list_of_strings(
        video.get("modality_keys"),
        field_name="modality_config.modalities.video.modality_keys",
    )
    _ = _as_list_of_ints(
        language.get("delta_indices"),
        field_name="modality_config.modalities.language.delta_indices",
    )
    _ = _as_list_of_strings(
        language.get("modality_keys"),
        field_name="modality_config.modalities.language.modality_keys",
    )

    action_configs_raw = action.get("action_configs")
    if not isinstance(action_configs_raw, list):
        raise TypeError(
            "modality_config.modalities.action.action_configs must be a list"
        )
    if len(action_configs_raw) != len(action_order_expected):
        raise ValueError(
            "modality_config.modalities.action.action_configs must match action modality_keys length"
        )

    relative_action_keys: list[str] = []
    absolute_action_keys: list[str] = []
    reference_state_keys: dict[str, str] = {}
    action_representations: dict[str, str] = {}
    for index, expected_key in enumerate(action_order_expected):
        raw_config = _as_mapping(
            action_configs_raw[index],
            field_name=f"modality_config.modalities.action.action_configs[{index}]",
        )
        action_key = _non_empty_string(
            raw_config.get("key"),
            field_name=f"modality_config.modalities.action.action_configs[{index}].key",
        )
        if action_key != expected_key:
            raise ValueError(
                "modality_config.modalities.action.action_configs order must match action modality_keys"
            )
        rep = _non_empty_string(
            raw_config.get("rep"),
            field_name=f"modality_config.modalities.action.action_configs[{index}].rep",
        )
        if rep not in {"RELATIVE", "ABSOLUTE"}:
            raise ValueError(
                f"unsupported action representation for {action_key}: {rep}"
            )
        action_representations[action_key] = rep
        if rep == "RELATIVE":
            relative_action_keys.append(action_key)
            reference_state_keys[action_key] = (
                _optional_non_empty_string(
                    raw_config.get("state_key"),
                    field_name=(
                        f"modality_config.modalities.action.action_configs[{index}].state_key"
                    ),
                )
                or action_key
            )
        else:
            absolute_action_keys.append(action_key)

    return {
        "payload": payload,
        "branch_tag": branch_tag,
        "embodiment_tag": embodiment_tag,
        "intended_usage": dict(intended_usage),
        "state_order_expected": state_order_expected,
        "action_order_expected": action_order_expected,
        "state_dims_expected": state_dims_expected,
        "action_dims_expected": action_dims_expected,
        "state_horizon_expected": state_horizon,
        "policy_horizon_expected": action_horizon,
        "relative_action_keys": relative_action_keys,
        "absolute_action_keys": absolute_action_keys,
        "relative_reference_state_keys": reference_state_keys,
        "action_representations": action_representations,
        "payload_fingerprint_sha256": _sha256_of_payload(payload),
    }


def build_branch_manifest_payload(
    *, modality_config_path: Path, modality_contract: Mapping[str, Any]
) -> dict[str, Any]:
    rel_config_path = _rel_repo(modality_config_path)
    return {
        "schema_version": BRANCH_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": BRANCH_MANIFEST_ARTIFACT_KIND,
        "branch_tag": EXPECTED_BRANCH_TAG,
        "embodiment_tag": EXPECTED_BRANCH_TAG,
        "public_anchor_comparable": False,
        "unitree_equivalence_reference": "informational_only",
        "modality_config_path": rel_config_path,
        "modality_config_fingerprint_sha256": str(
            modality_contract["payload_fingerprint_sha256"]
        ),
        "modality_config_usage_scope": {
            "frozen_file_role": "canonical_repo_local_new_embodiment_branch_contract",
            "replay_side_json_contract": True,
            "training_side_python_registration_required": True,
            "direct_launch_finetune_compatible": False,
            "format_mismatch_note": str(
                cast(Mapping[str, Any], modality_contract["intended_usage"]).get(
                    "format_mismatch_note", ""
                )
            ),
        },
        "server_contract_expectations": {
            "branch_tag": EXPECTED_BRANCH_TAG,
            "server_modality_config_keys": list(EXPECTED_SERVER_MODALITY_CONFIG_KEYS),
            "run_gr00t_server_default_embodiment_tag": EXPECTED_BRANCH_TAG,
        },
        "state_order_expected": list(modality_contract["state_order_expected"]),
        "state_dims_expected": dict(modality_contract["state_dims_expected"]),
        "action_order_expected": list(modality_contract["action_order_expected"]),
        "action_dims_expected": dict(modality_contract["action_dims_expected"]),
        "relative_action_policy": {
            "enabled_when_use_relative_action": bool(
                modality_contract["relative_action_keys"]
            ),
            "relative_action_keys": list(modality_contract["relative_action_keys"]),
            "absolute_action_keys": list(modality_contract["absolute_action_keys"]),
            "action_representation_by_key": dict(
                modality_contract["action_representations"]
            ),
            "reference_state_keys": dict(
                modality_contract["relative_reference_state_keys"]
            ),
            "policy_horizon_expected": int(
                modality_contract["policy_horizon_expected"]
            ),
            "unitree_equivalence_reference": "informational_only",
        },
        "normalization_source": {
            "owner": "NEW_EMBODIMENT branch-local statistics",
            "policy": "branch_specific_stats_required_no_cross_branch_reuse",
            "cross_branch_reuse_allowed": False,
            "source_kind": "repo_local_branch_contract",
            "shipped_stats_artifact_path": None,
            "provenance_complete": True,
            "note": (
                "This branch contract records normalization ownership explicitly but does "
                "not claim a public UNITREE_G1-compatible stats file. Later tasks must "
                "record the concrete stats artifact they use."
            ),
        },
        "controller_provenance": {
            "controller_family": "custom_non_official_whole_body_controller",
            "controller_name": "NEW_EMBODIMENT_branch_controller_contract",
            "data_collection_stack": "different_whole_body_controller_than_GR00T_WholeBodyControl",
            "public_benchmark_equivalent": False,
            "provenance_complete": True,
            "provenance_rule": (
                "Official docs route different whole-body controllers to NEW_EMBODIMENT, "
                "so this branch stays isolated from the public UNITREE_G1 benchmark line."
            ),
            "source_refs": [
                "submodules/Isaac-GR00T/examples/GR00T-WholeBodyControl/README.md:96-102",
                "submodules/Isaac-GR00T/README.md:210-224",
            ],
        },
        "dataset_provenance": {
            "dataset_lineage": "branch_local_custom_embodiment_dataset_contract",
            "admission_scope": "NEW_EMBODIMENT_only",
            "collected_with_gr00t_wholebodycontrol_repo": False,
            "public_anchor_comparable": False,
            "requires_branch_specific_normalization": True,
            "provenance_complete": True,
        },
        "required_provenance_fields": list(REQUIRED_MANIFEST_PROVENANCE_FIELDS),
        "formal_branch_eligibility": "ALLOW",
        "formal_branch_blockers": [],
    }


def _missing_or_incomplete(
    manifest: Mapping[str, Any],
    *,
    field_name: str,
    expected_mapping_keys: tuple[str, ...] | None = None,
) -> str | None:
    if field_name not in manifest:
        return f"missing_branch_manifest.{field_name}"
    value = manifest[field_name]
    if expected_mapping_keys is None:
        if value is None:
            return f"missing_branch_manifest.{field_name}"
        if isinstance(value, str) and not value.strip():
            return f"missing_branch_manifest.{field_name}"
        return None
    if not isinstance(value, Mapping):
        return f"invalid_branch_manifest.{field_name}"
    for key in expected_mapping_keys:
        if key not in value:
            return f"missing_branch_manifest.{field_name}.{key}"
        item = value[key]
        if item is None:
            return f"missing_branch_manifest.{field_name}.{key}"
        if isinstance(item, str) and not item.strip():
            return f"missing_branch_manifest.{field_name}.{key}"
    return None


def build_audit_report(
    *,
    modality_config_path: Path,
    branch_manifest_path: Path,
    branch_manifest_created: bool,
) -> dict[str, Any]:
    modality_contract = load_modality_contract(modality_config_path)
    manifest = _read_json(branch_manifest_path)

    mismatch_fields: list[str] = []
    reason_codes: list[str] = []
    for field_name in REQUIRED_MANIFEST_PROVENANCE_FIELDS:
        required_keys: tuple[str, ...] | None = None
        if field_name == "normalization_source":
            required_keys = ("owner", "policy", "cross_branch_reuse_allowed")
        elif field_name == "controller_provenance":
            required_keys = (
                "controller_family",
                "controller_name",
                "data_collection_stack",
            )
        elif field_name == "dataset_provenance":
            required_keys = ("dataset_lineage", "admission_scope")
        elif field_name == "relative_action_policy":
            required_keys = ("relative_action_keys", "absolute_action_keys")
        blocker = _missing_or_incomplete(
            manifest,
            field_name=field_name,
            expected_mapping_keys=required_keys,
        )
        if blocker is not None:
            mismatch_fields.append(field_name)
            reason_codes.append(blocker)

    manifest_branch_tag = manifest.get("branch_tag")
    if manifest_branch_tag != EXPECTED_BRANCH_TAG:
        mismatch_fields.append("branch_tag")
        reason_codes.append("branch_tag_mismatch")

    manifest_public_anchor = manifest.get("public_anchor_comparable")
    if manifest_public_anchor is not False:
        mismatch_fields.append("public_anchor_comparable")
        reason_codes.append("public_anchor_comparable_must_be_false")

    manifest_unitree_reference = manifest.get("unitree_equivalence_reference")
    if manifest_unitree_reference != "informational_only":
        mismatch_fields.append("unitree_equivalence_reference")
        reason_codes.append("unitree_equivalence_reference_must_be_informational_only")

    expected_rel_config_path = _rel_repo(modality_config_path)
    if manifest.get("modality_config_path") != expected_rel_config_path:
        mismatch_fields.append("modality_config_path")
        reason_codes.append("modality_config_path_manifest_mismatch")

    expected_fingerprint = str(modality_contract["payload_fingerprint_sha256"])
    if manifest.get("modality_config_fingerprint_sha256") != expected_fingerprint:
        mismatch_fields.append("modality_config_fingerprint_sha256")
        reason_codes.append("modality_config_fingerprint_mismatch")

    if (
        manifest.get("state_order_expected")
        != modality_contract["state_order_expected"]
    ):
        mismatch_fields.append("state_order_expected")
        reason_codes.append("state_order_manifest_mismatch")
    if (
        manifest.get("action_order_expected")
        != modality_contract["action_order_expected"]
    ):
        mismatch_fields.append("action_order_expected")
        reason_codes.append("action_order_manifest_mismatch")

    if manifest.get("state_dims_expected") != modality_contract["state_dims_expected"]:
        mismatch_fields.append("state_dims_expected")
        reason_codes.append("state_dims_manifest_mismatch")
    if (
        manifest.get("action_dims_expected")
        != modality_contract["action_dims_expected"]
    ):
        mismatch_fields.append("action_dims_expected")
        reason_codes.append("action_dims_manifest_mismatch")

    expected_relative_action_policy = {
        "enabled_when_use_relative_action": bool(
            modality_contract["relative_action_keys"]
        ),
        "relative_action_keys": list(modality_contract["relative_action_keys"]),
        "absolute_action_keys": list(modality_contract["absolute_action_keys"]),
        "action_representation_by_key": dict(
            modality_contract["action_representations"]
        ),
        "reference_state_keys": dict(
            modality_contract["relative_reference_state_keys"]
        ),
        "policy_horizon_expected": int(modality_contract["policy_horizon_expected"]),
        "unitree_equivalence_reference": "informational_only",
    }
    manifest_relative_action_policy = manifest.get("relative_action_policy")
    if isinstance(manifest_relative_action_policy, Mapping):
        for key, expected_value in expected_relative_action_policy.items():
            if manifest_relative_action_policy.get(key) != expected_value:
                mismatch_fields.append("relative_action_policy")
                reason_codes.append(f"relative_action_policy.{key}_mismatch")
    elif "relative_action_policy" in manifest:
        mismatch_fields.append("relative_action_policy")
        reason_codes.append("invalid_branch_manifest.relative_action_policy")

    normalization_source = manifest.get("normalization_source")
    if isinstance(normalization_source, Mapping):
        if normalization_source.get("cross_branch_reuse_allowed") is not False:
            mismatch_fields.append("normalization_source")
            reason_codes.append(
                "normalization_source.cross_branch_reuse_allowed_must_be_false"
            )
    controller_provenance = manifest.get("controller_provenance")
    if isinstance(controller_provenance, Mapping):
        if controller_provenance.get("public_benchmark_equivalent") is not False:
            mismatch_fields.append("controller_provenance")
            reason_codes.append(
                "controller_provenance.public_benchmark_equivalent_must_be_false"
            )
    dataset_provenance = manifest.get("dataset_provenance")
    if isinstance(dataset_provenance, Mapping):
        if dataset_provenance.get("public_anchor_comparable") is not False:
            mismatch_fields.append("dataset_provenance")
            reason_codes.append(
                "dataset_provenance.public_anchor_comparable_must_be_false"
            )

    normalized_mismatch_fields = sorted(set(mismatch_fields))
    normalized_reason_codes = sorted(set(reason_codes))
    formal_branch_eligibility = "ALLOW" if not normalized_reason_codes else "BLOCK"
    reason_code = normalized_reason_codes[0] if normalized_reason_codes else "OK"

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "branch_tag": EXPECTED_BRANCH_TAG,
        "embodiment_tag": EXPECTED_BRANCH_TAG,
        "modality_config_path": expected_rel_config_path,
        "modality_config_fingerprint_sha256": expected_fingerprint,
        "branch_manifest_path": _rel_repo(branch_manifest_path),
        "branch_manifest_created": branch_manifest_created,
        "branch_manifest_schema_version": manifest.get("schema_version"),
        "server_modality_config_keys_expected": list(
            EXPECTED_SERVER_MODALITY_CONFIG_KEYS
        ),
        "state_order_expected": list(modality_contract["state_order_expected"]),
        "action_order_expected": list(modality_contract["action_order_expected"]),
        "state_dims_expected": dict(modality_contract["state_dims_expected"]),
        "action_dims_expected": dict(modality_contract["action_dims_expected"]),
        "state_horizon_expected": int(modality_contract["state_horizon_expected"]),
        "policy_horizon_expected": int(modality_contract["policy_horizon_expected"]),
        "relative_action_policy": expected_relative_action_policy,
        "normalization_source": dict(normalization_source)
        if isinstance(normalization_source, Mapping)
        else None,
        "controller_provenance": dict(controller_provenance)
        if isinstance(controller_provenance, Mapping)
        else None,
        "dataset_provenance": dict(dataset_provenance)
        if isinstance(dataset_provenance, Mapping)
        else None,
        "public_anchor_comparable": False,
        "unitree_equivalence_reference": "informational_only",
        "equivalent_to_official_unitree_g1": False,
        "formal_branch_eligibility": formal_branch_eligibility,
        "reason_code": reason_code,
        "reason_codes": normalized_reason_codes,
        "mismatch_fields": normalized_mismatch_fields,
        "formal_branch_blockers": normalized_reason_codes,
        "branch_manifest_required_fields": list(REQUIRED_MANIFEST_PROVENANCE_FIELDS),
        "config_usage_scope": dict(modality_contract["intended_usage"]),
        "validation_summary": {
            "public_anchor_comparable_expected": False,
            "same_as_public_unitree_protocol": False,
            "same_config_fingerprint_as_manifest": not any(
                code == "modality_config_fingerprint_mismatch"
                for code in normalized_reason_codes
            ),
            "normalization_source_recorded": isinstance(normalization_source, Mapping),
            "controller_provenance_recorded": isinstance(
                controller_provenance, Mapping
            ),
        },
        "source_refs": {
            "official_branch_split_rule": "submodules/Isaac-GR00T/examples/GR00T-WholeBodyControl/README.md:96-102",
            "new_embodiment_cli_context": "submodules/Isaac-GR00T/README.md:210-224",
            "replay_server_modality_config_path": "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py:23-35,73-88",
            "public_anchor_contract": "agent/exchange/gr00t_policy_io.md:125-157",
            "branch_protected_surface_patterns": "agent/run/state_conditioned_contract_gate.py:53-80",
            "selected_digest_manifest_patterns": "agent/run/state_conditioned_wave_freeze_manifest.py:58-176",
            "unitree_reference_pattern": "agent/run/gr00t_controller_audit_unitree_g1.py:408-567",
        },
    }


def build_failure_note(report: Mapping[str, Any], *, output_path: Path) -> str:
    lines = [
        "# NEW_EMBODIMENT controller audit failure note",
        "",
        f"- output: `{_rel_repo(output_path)}`",
        f"- branch_manifest_path: `{report.get('branch_manifest_path', 'unknown')}`",
        f"- modality_config_path: `{report.get('modality_config_path', 'unknown')}`",
        f"- formal_branch_eligibility: `{report.get('formal_branch_eligibility')}`",
        f"- reason_code: `{report.get('reason_code')}`",
        f"- mismatch_fields: `{json.dumps(report.get('mismatch_fields', []), ensure_ascii=True)}`",
        "",
        "## Formal branch blockers",
        "",
        "```json",
        json.dumps(
            report.get("formal_branch_blockers", []), ensure_ascii=True, indent=2
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        modality_config_path = _validate_existing_file(
            cast(Path, args.modality_config_path),
            arg_name="modality-config-path",
        )
        branch_manifest_path = _validate_json_output_path(
            cast(Path, args.branch_manifest_path),
            arg_name="branch-manifest-path",
        )
        output_path = _validate_json_output_path(
            cast(Path, args.output), arg_name="output"
        )

        modality_contract = load_modality_contract(modality_config_path)
        branch_manifest_created = False
        if not branch_manifest_path.exists():
            manifest = build_branch_manifest_payload(
                modality_config_path=modality_config_path,
                modality_contract=modality_contract,
            )
            _ = _write_json(branch_manifest_path, manifest)
            branch_manifest_created = True

        report = build_audit_report(
            modality_config_path=modality_config_path,
            branch_manifest_path=branch_manifest_path,
            branch_manifest_created=branch_manifest_created,
        )
        _ = _write_json(output_path, report)

        failure_note_path = output_path.with_name(FAILURE_NOTE_MARKDOWN_NAME)
        if str(report["formal_branch_eligibility"]) != "ALLOW":
            _ = _write_text(
                failure_note_path, build_failure_note(report, output_path=output_path)
            )
        elif failure_note_path.exists():
            failure_note_path.unlink()

        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
