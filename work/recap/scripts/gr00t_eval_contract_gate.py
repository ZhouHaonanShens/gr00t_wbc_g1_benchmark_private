from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/gr00t_eval_contract/freeze")
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_MAX_EPISODE_STEPS = 1440
DEFAULT_N_ACTION_STEPS = 20
DEFAULT_POLICY_HORIZON_EXPECTED = 30
DEFAULT_N_EPISODES = 10
DEFAULT_N_ENVS = 5
DEFAULT_PROMPT_TEMPLATE_ID = "gr00t_g1_pnp_apple_to_plate_public_eval_v1"
DEFAULT_BRANCH_COMPARABILITY_TAG = "unitree_g1_public_anchor_vs_new_embodiment_split_v1"
DEFAULT_PUBLIC_REFERENCE_MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_RELATIVE_ACTION_KEYS: tuple[str, ...] = ("left_arm", "right_arm")
DEFAULT_ABSOLUTE_ACTION_KEYS: tuple[str, ...] = (
    "left_hand",
    "right_hand",
    "waist",
    "base_height_command",
    "navigate_command",
)
DEFAULT_ACTION_REPRESENTATION_BY_KEY: dict[str, str] = {
    "left_arm": "RELATIVE",
    "right_arm": "RELATIVE",
    "left_hand": "ABSOLUTE",
    "right_hand": "ABSOLUTE",
    "waist": "ABSOLUTE",
    "base_height_command": "ABSOLUTE",
    "navigate_command": "ABSOLUTE",
}
DEFAULT_RELATIVE_REFERENCE_STATE_KEYS: dict[str, str] = {
    "left_arm": "left_arm",
    "right_arm": "right_arm",
}
DEFAULT_SCENE_POOL_IDENTIFIER = (
    "repo_local::LMPnPAppleToPlateDC_public_eval_scene_pool_v1"
)
DEFAULT_FORMAL_SEED_VALUES: tuple[int, ...] = tuple(range(20000, 20010))

DEFAULT_WRAPPER_PARAMETERS: dict[str, object] = {
    "multistep_wrapper": {
        "enabled": True,
        "execution_mode": "sequential_first_n_steps",
        "n_action_steps": DEFAULT_N_ACTION_STEPS,
    },
    "sim_policy_wrapper": {
        "action_key_prefix": "action.",
        "enabled": True,
        "observation_layout": "flat_keys",
        "use_sim_policy_wrapper": True,
    },
    "timebase": {
        "control_frequency_hz": 50,
        "sim_frequency_hz": 200,
        "sim_steps_per_control_step": 4,
    },
}

DEFAULT_CAMERA_CONFIG: dict[str, object] = {
    "image_observation_keys": ["ego_view_image", "tpp_view_image"],
    "layout": "B,T,H,W,C",
    "video_observation_keys": ["ego_view", "tpp_view"],
    "view_count": 2,
}

FREEZE_JSON_NAME = "eval_contract_freeze.json"
COMPARABILITY_REPORT_JSON_NAME = "comparability_report.json"
FAILURE_NOTE_MARKDOWN_NAME = "comparability_failure_note.md"

FREEZE_SCHEMA_VERSION = "gr00t_eval_contract_freeze_v1"
REPORT_SCHEMA_VERSION = "gr00t_eval_comparability_report_v1"
FREEZE_ARTIFACT_KIND = "gr00t_eval_contract_freeze"
REPORT_ARTIFACT_KIND = "gr00t_eval_comparability_report"
CONTRACT_GATE_NAME = "GR00TEvalContractGate"
ALLOWED_CHANGE_POLICY = "BLOCK_ALL_FORMAL_DRIFT"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


PROTECTED_FIELD_PATHS: tuple[str, ...] = (
    "env_name",
    "wrapper_parameters",
    "camera_config",
    "max_episode_steps",
    "n_action_steps",
    "policy_horizon_expected",
    "action_semantics",
    "scene_pool_identifier",
    "prompt_template_id",
    "seed_manifest",
    "normalization_policy",
    "branch_comparability_tag",
    "checkpoint_provenance_schema",
    "server_contract.embodiment_tag",
    "server_contract.use_sim_policy_wrapper",
    "embodiment_branches.UNITREE_G1.public_anchor_comparable",
    "embodiment_branches.NEW_EMBODIMENT.public_anchor_comparable",
)

BRANCH_PROTECTED_FIELD_PATHS: tuple[str, ...] = (
    "branch_comparability_tag",
    "embodiment_branches.UNITREE_G1.public_anchor_comparable",
    "embodiment_branches.NEW_EMBODIMENT.public_anchor_comparable",
)

SEED_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "python",
    "numpy",
    "torch",
    "env",
    "rollout_episode_order",
)


MISSING = object()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_eval_contract_gate.py",
        description=(
            "Freeze the public G1 evaluation contract into deterministic JSON artifacts "
            "and compare candidate payloads against the formal comparability surface."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives eval_contract_freeze.json and comparability_report.json.",
    )
    _ = parser.add_argument(
        "--candidate-json",
        type=Path,
        default=None,
        help=(
            "Optional candidate freeze JSON to validate against the canonical Task 1 "
            "formal eval contract. When omitted, the canonical payload validates itself."
        ),
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"candidate-json does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return dict(
        _as_mapping(
            payload,
            field_name="candidate_json",
        )
    )


def _deep_get(payload: Mapping[str, Any], field_path: str) -> object:
    current: object = payload
    for key in field_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return MISSING
        current = current[key]
    return current


def _value_for_report(value: object) -> object:
    if value is MISSING:
        return {"missing": True}
    return value


def _selected_digest_payload(
    payload: Mapping[str, Any], selectors: Sequence[str]
) -> Mapping[str, Any]:
    selected: dict[str, Any] = {}
    for selector in selectors:
        value = _deep_get(payload, str(selector))
        if value is not MISSING:
            selected[str(selector)] = value
    return selected


def _contract_digest(payload: Mapping[str, Any], selectors: Sequence[str]) -> str:
    selected_payload = _selected_digest_payload(payload, selectors)
    return _sha256_bytes(_canonical_json_bytes(selected_payload))


def _build_action_semantics() -> dict[str, Any]:
    return {
        "policy_horizon_expected": int(DEFAULT_POLICY_HORIZON_EXPECTED),
        "n_action_steps": int(DEFAULT_N_ACTION_STEPS),
        "relative_action_keys": list(DEFAULT_RELATIVE_ACTION_KEYS),
        "absolute_action_keys": list(DEFAULT_ABSOLUTE_ACTION_KEYS),
        "action_representation_by_key": dict(DEFAULT_ACTION_REPRESENTATION_BY_KEY),
        "relative_to_absolute_rule": {
            "enabled_for_relative_action_keys": True,
            "reference_state_timestep": "last",
            "reference_state_keys": dict(DEFAULT_RELATIVE_REFERENCE_STATE_KEYS),
        },
        "must_not_conflate_horizon_and_execution": True,
        "repo_local_formalization": {
            "field_names_are_repo_local": True,
            "upstream_policy_horizon_authority": "action.delta_indices",
            "upstream_execution_steps_authority": "rollout --n_action_steps",
            "note": (
                "This repo freezes local contract field names for comparability; "
                "they summarize upstream semantics but are not upstream official JSON field names."
            ),
        },
    }


def _compare_field_paths(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    field_paths: Sequence[str],
    reason: str,
) -> list[dict[str, Any]]:
    drifts: list[dict[str, Any]] = []
    for field_path in field_paths:
        expected_value = _deep_get(expected, field_path)
        actual_value = _deep_get(actual, field_path)
        if expected_value != actual_value:
            drifts.append(
                {
                    "field_path": field_path,
                    "reason": reason,
                    "expected": _value_for_report(expected_value),
                    "actual": _value_for_report(actual_value),
                }
            )
    return drifts


def _build_check_result(
    name: str,
    *,
    error: str | None = None,
    offending_field_paths: Sequence[str] | None = None,
    drifts: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    passed = error is None
    result: dict[str, Any] = {
        "name": name,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }
    if offending_field_paths is not None:
        result["offending_field_paths"] = [str(path) for path in offending_field_paths]
    if drifts is not None:
        result["drifts"] = [dict(drift) for drift in drifts]
    if extra is not None:
        result.update(dict(extra))
    if error is not None:
        result["error"] = error
    return result


def build_eval_contract_freeze() -> dict[str, Any]:
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "artifact_kind": FREEZE_ARTIFACT_KIND,
        "contract_gate": CONTRACT_GATE_NAME,
        "env_name": DEFAULT_ENV_NAME,
        "wrapper_parameters": dict(DEFAULT_WRAPPER_PARAMETERS),
        "camera_config": dict(DEFAULT_CAMERA_CONFIG),
        "max_episode_steps": int(DEFAULT_MAX_EPISODE_STEPS),
        "n_action_steps": int(DEFAULT_N_ACTION_STEPS),
        "policy_horizon_expected": int(DEFAULT_POLICY_HORIZON_EXPECTED),
        "action_semantics": _build_action_semantics(),
        "scene_pool_identifier": DEFAULT_SCENE_POOL_IDENTIFIER,
        "prompt_template_id": DEFAULT_PROMPT_TEMPLATE_ID,
        "public_eval_client_defaults": {
            "n_episodes": int(DEFAULT_N_EPISODES),
            "n_envs": int(DEFAULT_N_ENVS),
        },
        "server_contract": {
            "embodiment_tag": "UNITREE_G1",
            "use_sim_policy_wrapper": True,
            "public_reference_model_path": DEFAULT_PUBLIC_REFERENCE_MODEL_PATH,
        },
        "seed_manifest": {
            "schema": "explicit_eval_seed_manifest_v1",
            "same_manifest_required": True,
            "published_seed_values_in_upstream_readme": False,
            "required_fields": list(SEED_MANIFEST_REQUIRED_FIELDS),
            "seed_values": [int(seed) for seed in DEFAULT_FORMAL_SEED_VALUES],
            "seed_values_origin": "repo_local_formal_protocol",
            "comparison_rule": (
                "numeric comparability requires the identical seed manifest across compared runs"
            ),
        },
        "normalization_policy": {
            "policy_id": "branch_scoped_normalization_policy_v1",
            "same_policy_required": True,
            "cross_branch_reuse_allowed": False,
            "delta_indices_bound_to_stats": True,
            "public_anchor_owner": "pre_registered_UNITREE_G1_modality_and_stats_contract",
            "new_embodiment_owner": (
                "custom_modality_config_and_explicit_local_stats_provenance"
            ),
        },
        "branch_comparability_tag": DEFAULT_BRANCH_COMPARABILITY_TAG,
        "embodiment_branches": {
            "UNITREE_G1": {
                "public_anchor_comparable": True,
                "comparability_scope": "official_public_anchor",
                "controller_boundary": "GR00T-WholeBodyControl protocol only",
            },
            "NEW_EMBODIMENT": {
                "public_anchor_comparable": False,
                "comparability_scope": "branch_only",
                "controller_boundary": "different or custom controller path",
            },
        },
        "checkpoint_provenance_schema": {
            "schema_id": "gr00t_eval_checkpoint_provenance_v1",
            "required_server_fields": [
                "model_path",
                "embodiment_tag",
                "use_sim_policy_wrapper",
            ],
            "allowed_model_path_kinds": [
                "huggingface_model",
                "local_checkpoint_dir",
            ],
            "required_embodiment_tag_for_public_anchor": "UNITREE_G1",
            "required_use_sim_policy_wrapper": True,
            "public_reference_model_path": DEFAULT_PUBLIC_REFERENCE_MODEL_PATH,
        },
        "horizon_semantics": {
            "policy_horizon_expected_meaning": (
                "server get_action chunk horizon defined by the current action delta_indices"
            ),
            "n_action_steps_meaning": (
                "client-side rollout execution window per outer-loop policy call"
            ),
            "must_not_be_conflated": True,
        },
        "allowed_change_surface": {
            "policy": ALLOWED_CHANGE_POLICY,
            "human_readable_rule": (
                "Formal public-anchor comparability blocks drift in env, wrapper, horizon, "
                "camera, scene pool, prompt, seed manifest, branch tag, normalization "
                "policy, and checkpoint provenance schema."
            ),
            "protected_field_paths": list(PROTECTED_FIELD_PATHS),
        },
        "sources": {
            "public_anchor_protocol": (
                "submodules/Isaac-GR00T/examples/GR00T-WholeBodyControl/README.md"
            ),
            "policy_io_contract": "agent/exchange/gr00t_policy_io.md",
            "wbc_env_contract": "agent/exchange/wbc_env_io.md",
        },
    }


def validate_eval_contract_freeze(payload: Mapping[str, Any]) -> dict[str, Any]:
    freeze = dict(_as_mapping(payload, field_name="freeze_payload"))
    if freeze.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError(
            f"freeze_payload.schema_version must be {FREEZE_SCHEMA_VERSION}"
        )
    if freeze.get("artifact_kind") != FREEZE_ARTIFACT_KIND:
        raise ValueError(f"freeze_payload.artifact_kind must be {FREEZE_ARTIFACT_KIND}")
    if freeze.get("contract_gate") != CONTRACT_GATE_NAME:
        raise ValueError(f"freeze_payload.contract_gate must be {CONTRACT_GATE_NAME}")
    if (
        _as_non_empty_string(
            freeze.get("env_name"), field_name="freeze_payload.env_name"
        )
        != DEFAULT_ENV_NAME
    ):
        raise ValueError(f"freeze_payload.env_name must be {DEFAULT_ENV_NAME}")
    wrapper_parameters = dict(
        _as_mapping(
            freeze.get("wrapper_parameters"),
            field_name="freeze_payload.wrapper_parameters",
        )
    )
    if wrapper_parameters != DEFAULT_WRAPPER_PARAMETERS:
        raise ValueError(
            "freeze_payload.wrapper_parameters must match formal wrapper freeze"
        )
    camera_config = dict(
        _as_mapping(
            freeze.get("camera_config"),
            field_name="freeze_payload.camera_config",
        )
    )
    if camera_config != DEFAULT_CAMERA_CONFIG:
        raise ValueError("freeze_payload.camera_config must match formal camera freeze")
    if (
        _as_int(
            freeze.get("max_episode_steps"),
            field_name="freeze_payload.max_episode_steps",
        )
        != DEFAULT_MAX_EPISODE_STEPS
    ):
        raise ValueError(
            f"freeze_payload.max_episode_steps must be {DEFAULT_MAX_EPISODE_STEPS}"
        )
    if (
        _as_int(
            freeze.get("n_action_steps"), field_name="freeze_payload.n_action_steps"
        )
        != DEFAULT_N_ACTION_STEPS
    ):
        raise ValueError(
            f"freeze_payload.n_action_steps must be {DEFAULT_N_ACTION_STEPS}"
        )
    if (
        _as_int(
            freeze.get("policy_horizon_expected"),
            field_name="freeze_payload.policy_horizon_expected",
        )
        != DEFAULT_POLICY_HORIZON_EXPECTED
    ):
        raise ValueError(
            "freeze_payload.policy_horizon_expected must be "
            f"{DEFAULT_POLICY_HORIZON_EXPECTED}"
        )
    action_semantics = dict(
        _as_mapping(
            freeze.get("action_semantics"),
            field_name="freeze_payload.action_semantics",
        )
    )
    if action_semantics != _build_action_semantics():
        raise ValueError(
            "freeze_payload.action_semantics must match formal action semantics freeze"
        )
    if (
        _as_non_empty_string(
            freeze.get("scene_pool_identifier"),
            field_name="freeze_payload.scene_pool_identifier",
        )
        != DEFAULT_SCENE_POOL_IDENTIFIER
    ):
        raise ValueError(
            "freeze_payload.scene_pool_identifier must be "
            f"{DEFAULT_SCENE_POOL_IDENTIFIER}"
        )
    if (
        _as_non_empty_string(
            freeze.get("prompt_template_id"),
            field_name="freeze_payload.prompt_template_id",
        )
        != DEFAULT_PROMPT_TEMPLATE_ID
    ):
        raise ValueError(
            f"freeze_payload.prompt_template_id must be {DEFAULT_PROMPT_TEMPLATE_ID}"
        )

    client_defaults = dict(
        _as_mapping(
            freeze.get("public_eval_client_defaults"),
            field_name="freeze_payload.public_eval_client_defaults",
        )
    )
    if (
        _as_int(
            client_defaults.get("n_episodes"),
            field_name="freeze_payload.public_eval_client_defaults.n_episodes",
        )
        != DEFAULT_N_EPISODES
    ):
        raise ValueError(
            f"freeze_payload.public_eval_client_defaults.n_episodes must be {DEFAULT_N_EPISODES}"
        )
    if (
        _as_int(
            client_defaults.get("n_envs"),
            field_name="freeze_payload.public_eval_client_defaults.n_envs",
        )
        != DEFAULT_N_ENVS
    ):
        raise ValueError(
            f"freeze_payload.public_eval_client_defaults.n_envs must be {DEFAULT_N_ENVS}"
        )

    server_contract = dict(
        _as_mapping(
            freeze.get("server_contract"), field_name="freeze_payload.server_contract"
        )
    )
    if (
        _as_non_empty_string(
            server_contract.get("embodiment_tag"),
            field_name="freeze_payload.server_contract.embodiment_tag",
        )
        != "UNITREE_G1"
    ):
        raise ValueError(
            "freeze_payload.server_contract.embodiment_tag must be UNITREE_G1"
        )
    if not _as_bool(
        server_contract.get("use_sim_policy_wrapper"),
        field_name="freeze_payload.server_contract.use_sim_policy_wrapper",
    ):
        raise ValueError(
            "freeze_payload.server_contract.use_sim_policy_wrapper must be true"
        )
    if (
        _as_non_empty_string(
            server_contract.get("public_reference_model_path"),
            field_name="freeze_payload.server_contract.public_reference_model_path",
        )
        != DEFAULT_PUBLIC_REFERENCE_MODEL_PATH
    ):
        raise ValueError(
            "freeze_payload.server_contract.public_reference_model_path must be "
            f"{DEFAULT_PUBLIC_REFERENCE_MODEL_PATH}"
        )

    seed_manifest = dict(
        _as_mapping(
            freeze.get("seed_manifest"), field_name="freeze_payload.seed_manifest"
        )
    )
    if (
        _as_non_empty_string(
            seed_manifest.get("schema"),
            field_name="freeze_payload.seed_manifest.schema",
        )
        != "explicit_eval_seed_manifest_v1"
    ):
        raise ValueError(
            "freeze_payload.seed_manifest.schema must be explicit_eval_seed_manifest_v1"
        )
    if not _as_bool(
        seed_manifest.get("same_manifest_required"),
        field_name="freeze_payload.seed_manifest.same_manifest_required",
    ):
        raise ValueError(
            "freeze_payload.seed_manifest.same_manifest_required must be true"
        )
    if (
        tuple(
            _as_list(
                seed_manifest.get("required_fields"),
                field_name="freeze_payload.seed_manifest.required_fields",
            )
        )
        != SEED_MANIFEST_REQUIRED_FIELDS
    ):
        raise ValueError(
            "freeze_payload.seed_manifest.required_fields mismatch for explicit eval seed manifest"
        )
    if (
        tuple(
            _as_list(
                seed_manifest.get("seed_values"),
                field_name="freeze_payload.seed_manifest.seed_values",
            )
        )
        != DEFAULT_FORMAL_SEED_VALUES
    ):
        raise ValueError(
            "freeze_payload.seed_manifest.seed_values must match the repo-local formal seed list"
        )
    if (
        _as_non_empty_string(
            seed_manifest.get("seed_values_origin"),
            field_name="freeze_payload.seed_manifest.seed_values_origin",
        )
        != "repo_local_formal_protocol"
    ):
        raise ValueError(
            "freeze_payload.seed_manifest.seed_values_origin must be repo_local_formal_protocol"
        )

    normalization_policy = dict(
        _as_mapping(
            freeze.get("normalization_policy"),
            field_name="freeze_payload.normalization_policy",
        )
    )
    if (
        _as_non_empty_string(
            normalization_policy.get("policy_id"),
            field_name="freeze_payload.normalization_policy.policy_id",
        )
        != "branch_scoped_normalization_policy_v1"
    ):
        raise ValueError(
            "freeze_payload.normalization_policy.policy_id must be branch_scoped_normalization_policy_v1"
        )
    if _as_bool(
        normalization_policy.get("cross_branch_reuse_allowed"),
        field_name="freeze_payload.normalization_policy.cross_branch_reuse_allowed",
    ):
        raise ValueError(
            "freeze_payload.normalization_policy.cross_branch_reuse_allowed must be false"
        )
    if not _as_bool(
        normalization_policy.get("delta_indices_bound_to_stats"),
        field_name="freeze_payload.normalization_policy.delta_indices_bound_to_stats",
    ):
        raise ValueError(
            "freeze_payload.normalization_policy.delta_indices_bound_to_stats must be true"
        )

    if (
        _as_non_empty_string(
            freeze.get("branch_comparability_tag"),
            field_name="freeze_payload.branch_comparability_tag",
        )
        != DEFAULT_BRANCH_COMPARABILITY_TAG
    ):
        raise ValueError(
            "freeze_payload.branch_comparability_tag must be "
            f"{DEFAULT_BRANCH_COMPARABILITY_TAG}"
        )

    embodiment_branches = dict(
        _as_mapping(
            freeze.get("embodiment_branches"),
            field_name="freeze_payload.embodiment_branches",
        )
    )
    unitree_g1 = dict(
        _as_mapping(
            embodiment_branches.get("UNITREE_G1"),
            field_name="freeze_payload.embodiment_branches.UNITREE_G1",
        )
    )
    new_embodiment = dict(
        _as_mapping(
            embodiment_branches.get("NEW_EMBODIMENT"),
            field_name="freeze_payload.embodiment_branches.NEW_EMBODIMENT",
        )
    )
    if not _as_bool(
        unitree_g1.get("public_anchor_comparable"),
        field_name=(
            "freeze_payload.embodiment_branches.UNITREE_G1.public_anchor_comparable"
        ),
    ):
        raise ValueError(
            "freeze_payload.embodiment_branches.UNITREE_G1.public_anchor_comparable must be true"
        )
    if _as_bool(
        new_embodiment.get("public_anchor_comparable"),
        field_name=(
            "freeze_payload.embodiment_branches.NEW_EMBODIMENT.public_anchor_comparable"
        ),
    ):
        raise ValueError(
            "freeze_payload.embodiment_branches.NEW_EMBODIMENT.public_anchor_comparable must be false"
        )

    provenance_schema = dict(
        _as_mapping(
            freeze.get("checkpoint_provenance_schema"),
            field_name="freeze_payload.checkpoint_provenance_schema",
        )
    )
    if (
        _as_non_empty_string(
            provenance_schema.get("schema_id"),
            field_name="freeze_payload.checkpoint_provenance_schema.schema_id",
        )
        != "gr00t_eval_checkpoint_provenance_v1"
    ):
        raise ValueError(
            "freeze_payload.checkpoint_provenance_schema.schema_id must be "
            "gr00t_eval_checkpoint_provenance_v1"
        )
    if tuple(
        _as_list(
            provenance_schema.get("required_server_fields"),
            field_name=(
                "freeze_payload.checkpoint_provenance_schema.required_server_fields"
            ),
        )
    ) != ("model_path", "embodiment_tag", "use_sim_policy_wrapper"):
        raise ValueError(
            "freeze_payload.checkpoint_provenance_schema.required_server_fields mismatch"
        )
    if not _as_bool(
        provenance_schema.get("required_use_sim_policy_wrapper"),
        field_name=(
            "freeze_payload.checkpoint_provenance_schema.required_use_sim_policy_wrapper"
        ),
    ):
        raise ValueError(
            "freeze_payload.checkpoint_provenance_schema.required_use_sim_policy_wrapper must be true"
        )

    allowed_change_surface = dict(
        _as_mapping(
            freeze.get("allowed_change_surface"),
            field_name="freeze_payload.allowed_change_surface",
        )
    )
    if (
        _as_non_empty_string(
            allowed_change_surface.get("policy"),
            field_name="freeze_payload.allowed_change_surface.policy",
        )
        != ALLOWED_CHANGE_POLICY
    ):
        raise ValueError(
            f"freeze_payload.allowed_change_surface.policy must be {ALLOWED_CHANGE_POLICY}"
        )
    if (
        tuple(
            _as_list(
                allowed_change_surface.get("protected_field_paths"),
                field_name="freeze_payload.allowed_change_surface.protected_field_paths",
            )
        )
        != PROTECTED_FIELD_PATHS
    ):
        raise ValueError(
            "freeze_payload.allowed_change_surface.protected_field_paths mismatch"
        )

    digest = _contract_digest(freeze, PROTECTED_FIELD_PATHS)
    return {
        "protected_field_count": int(len(PROTECTED_FIELD_PATHS)),
        "contract_digest": digest,
        "branch_tags": ["UNITREE_G1", "NEW_EMBODIMENT"],
    }


def build_comparability_report(
    freeze_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    canonical_freeze = dict(_as_mapping(freeze_payload, field_name="freeze_payload"))
    candidate = (
        canonical_freeze
        if candidate_payload is None
        else dict(_as_mapping(candidate_payload, field_name="candidate_payload"))
    )

    try:
        validate_eval_contract_freeze(canonical_freeze)
    except (TypeError, ValueError) as exc:
        checks["freeze_payload_schema"] = _build_check_result(
            "freeze_payload_schema", error=_exception_message(exc)
        )
    else:
        checks["freeze_payload_schema"] = _build_check_result(
            "freeze_payload_schema",
            extra={
                "protected_field_count": int(len(PROTECTED_FIELD_PATHS)),
                "contract_digest": _contract_digest(
                    canonical_freeze, PROTECTED_FIELD_PATHS
                ),
            },
        )

    protected_drifts: list[dict[str, Any]] = []
    if checks["freeze_payload_schema"]["passed"]:
        protected_drifts = _compare_field_paths(
            canonical_freeze,
            candidate,
            field_paths=PROTECTED_FIELD_PATHS,
            reason="formal eval contract drift is forbidden",
        )
        if protected_drifts:
            protected_field_paths = [
                str(drift["field_path"]) for drift in protected_drifts
            ]
            checks["protected_field_freeze"] = _build_check_result(
                "protected_field_freeze",
                error="protected_field_freeze drift: "
                + ", ".join(protected_field_paths),
                offending_field_paths=protected_field_paths,
                drifts=protected_drifts,
                extra={
                    "protected_field_count": int(len(PROTECTED_FIELD_PATHS)),
                    "drift_count": int(len(protected_drifts)),
                },
            )
        else:
            checks["protected_field_freeze"] = _build_check_result(
                "protected_field_freeze",
                offending_field_paths=[],
                drifts=[],
                extra={
                    "protected_field_count": int(len(PROTECTED_FIELD_PATHS)),
                    "drift_count": 0,
                },
            )
    else:
        checks["protected_field_freeze"] = _build_check_result(
            "protected_field_freeze",
            error="freeze_payload_schema must pass before protected field comparison",
        )

    branch_drifts = _compare_field_paths(
        canonical_freeze,
        candidate,
        field_paths=BRANCH_PROTECTED_FIELD_PATHS,
        reason="branch comparability drift is forbidden",
    )
    if branch_drifts:
        branch_field_paths = [str(drift["field_path"]) for drift in branch_drifts]
        checks["branch_comparability_rules"] = _build_check_result(
            "branch_comparability_rules",
            error="branch_comparability_rules drift: " + ", ".join(branch_field_paths),
            offending_field_paths=branch_field_paths,
            drifts=branch_drifts,
            extra={
                "required_branch_tags": ["UNITREE_G1", "NEW_EMBODIMENT"],
                "unitree_g1_public_anchor_comparable_expected": True,
                "new_embodiment_public_anchor_comparable_expected": False,
            },
        )
    else:
        checks["branch_comparability_rules"] = _build_check_result(
            "branch_comparability_rules",
            offending_field_paths=[],
            drifts=[],
            extra={
                "required_branch_tags": ["UNITREE_G1", "NEW_EMBODIMENT"],
                "unitree_g1_public_anchor_comparable_expected": True,
                "new_embodiment_public_anchor_comparable_expected": False,
            },
        )

    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "contract_gate": {
            "name": CONTRACT_GATE_NAME,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        },
        "freeze_schema_version": FREEZE_SCHEMA_VERSION,
        "change_policy": ALLOWED_CHANGE_POLICY,
        "digest_basis": {
            "mode": "selected_fields",
            "selectors": list(PROTECTED_FIELD_PATHS),
        },
        "canonical_contract_digest": _contract_digest(
            canonical_freeze, PROTECTED_FIELD_PATHS
        ),
        "candidate_contract_digest": _contract_digest(candidate, PROTECTED_FIELD_PATHS),
        "counts": {
            "protected_field_count": int(len(PROTECTED_FIELD_PATHS)),
            "drift_count": int(len(protected_drifts)),
        },
        "checks": checks,
    }


def _build_failure_note(report: Mapping[str, Any]) -> str:
    checks = dict(_as_mapping(report.get("checks"), field_name="report.checks"))
    drift_check = dict(
        _as_mapping(
            checks.get("protected_field_freeze"),
            field_name="report.checks.protected_field_freeze",
        )
    )
    offending_paths = [
        str(path) for path in list(drift_check.get("offending_field_paths", []))
    ]
    lines = [
        "# GR00T eval contract gate failure note",
        "",
        f"- gate: `{CONTRACT_GATE_NAME}`",
        f"- status: `{report.get('contract_gate', {}).get('status', 'FAIL')}`",
        f"- drift_count: `{len(offending_paths)}`",
        "- offending_field_paths:",
    ]
    if offending_paths:
        lines.extend(f"  - `{field_path}`" for field_path in offending_paths)
    else:
        lines.append("  - none captured")
    lines.extend(
        [
            "",
            "请修复上述 formal contract drift 后再把该候选结果与公开 UNITREE_G1 anchor 做数值比较。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_failure_note(path: Path, report: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_build_failure_note(report), encoding="utf-8")
    tmp.replace(path)
    return path


def materialize_eval_contract_gate(
    *, output_dir: Path, candidate_json: Path | None = None
) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    freeze_payload = build_eval_contract_freeze()
    candidate_payload = (
        freeze_payload if candidate_json is None else _read_json(candidate_json)
    )
    report_payload = build_comparability_report(freeze_payload, candidate_payload)

    freeze_path = _write_json(resolved_output_dir / FREEZE_JSON_NAME, freeze_payload)
    report_path = _write_json(
        resolved_output_dir / COMPARABILITY_REPORT_JSON_NAME,
        report_payload,
    )
    contract_gate = dict(
        _as_mapping(
            report_payload.get("contract_gate"),
            field_name="report_payload.contract_gate",
        )
    )
    report_passed = bool(contract_gate.get("passed", False))
    failure_note_path: str | None = None
    if not report_passed:
        failure_note = _write_failure_note(
            resolved_output_dir / FAILURE_NOTE_MARKDOWN_NAME,
            report_payload,
        )
        failure_note_path = str(failure_note)
    else:
        stale_failure_note = resolved_output_dir / FAILURE_NOTE_MARKDOWN_NAME
        if stale_failure_note.exists():
            stale_failure_note.unlink()

    return {
        "eval_contract_freeze_path": str(freeze_path),
        "comparability_report_path": str(report_path),
        "failure_note_path": failure_note_path,
        "passed": report_passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        result = materialize_eval_contract_gate(
            output_dir=args.output_dir,
            candidate_json=args.candidate_json,
        )
    except SystemExit:
        raise
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1

    if not bool(result["passed"]):
        print(
            "comparability gate failed; see comparability_report.json for offending field paths",
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = [
    "ALLOWED_CHANGE_POLICY",
    "BRANCH_PROTECTED_FIELD_PATHS",
    "COMPARABILITY_REPORT_JSON_NAME",
    "CONTRACT_GATE_NAME",
    "DEFAULT_BRANCH_COMPARABILITY_TAG",
    "DEFAULT_ENV_NAME",
    "DEFAULT_MAX_EPISODE_STEPS",
    "DEFAULT_N_ACTION_STEPS",
    "DEFAULT_POLICY_HORIZON_EXPECTED",
    "DEFAULT_PROMPT_TEMPLATE_ID",
    "FAILURE_NOTE_MARKDOWN_NAME",
    "FREEZE_ARTIFACT_KIND",
    "FREEZE_JSON_NAME",
    "FREEZE_SCHEMA_VERSION",
    "PROTECTED_FIELD_PATHS",
    "REPORT_ARTIFACT_KIND",
    "REPORT_SCHEMA_VERSION",
    "build_comparability_report",
    "build_eval_contract_freeze",
    "build_parser",
    "main",
    "materialize_eval_contract_gate",
    "validate_eval_contract_freeze",
]


if __name__ == "__main__":
    raise SystemExit(main())
