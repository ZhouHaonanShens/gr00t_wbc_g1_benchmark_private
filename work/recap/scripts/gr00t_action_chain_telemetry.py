from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

REPORT_SCHEMA_VERSION = "gr00t_action_chain_telemetry_v1"
REPORT_ARTIFACT_KIND = "gr00t_action_chain_telemetry"

BRANCH_UNITREE_G1 = "UNITREE_G1"
BRANCH_NEW_EMBODIMENT = "NEW_EMBODIMENT"
ALL_BRANCHES = (BRANCH_UNITREE_G1, BRANCH_NEW_EMBODIMENT)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import gr00t_controller_audit_new_embodiment
from work.recap import state_conditioned_bucket_a_import


DEFAULT_OUTPUTS: dict[str, Path] = {
    BRANCH_UNITREE_G1: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "unitree_g1"
    / "action_chain_telemetry_unitree_g1.json",
    BRANCH_NEW_EMBODIMENT: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "new_embodiment"
    / "action_chain_telemetry_new_embodiment.json",
}

DEFAULT_DECODE_LIMITS_BY_BRANCH: dict[str, dict[str, tuple[float, float]]] = {
    BRANCH_UNITREE_G1: {
        "left_arm": (-0.30, 0.60),
        "right_arm": (-0.35, 0.55),
        "left_hand": (-1.00, 1.00),
        "right_hand": (-1.00, 1.00),
        "waist": (-0.40, 0.40),
        "navigate_command": (-0.60, 0.60),
        "base_height_command": (0.60, 0.90),
    },
    BRANCH_NEW_EMBODIMENT: {
        "left_arm": (-0.25, 0.55),
        "right_arm": (-0.30, 0.60),
        "left_hand": (-1.00, 1.00),
        "right_hand": (-1.00, 1.00),
        "waist": (-0.35, 0.35),
        "navigate_command": (-0.50, 0.50),
        "base_height_command": (0.58, 0.92),
    },
}

DEFAULT_CONTROLLER_LIMITS_BY_BRANCH: dict[str, dict[str, tuple[float, float]]] = {
    BRANCH_UNITREE_G1: {
        "left_arm": (-0.90, 0.90),
        "right_arm": (-1.20, 1.20),
        "left_hand": (-0.85, 0.85),
        "right_hand": (-0.10, 0.10),
        "waist": (-0.35, 0.35),
        "navigate_command": (-0.50, 0.50),
        "base_height_command": (0.65, 0.82),
    },
    BRANCH_NEW_EMBODIMENT: {
        "left_arm": (-0.88, 0.88),
        "right_arm": (-1.10, 1.10),
        "left_hand": (-0.80, 0.80),
        "right_hand": (-0.10, 0.10),
        "waist": (-0.32, 0.32),
        "navigate_command": (-0.45, 0.45),
        "base_height_command": (0.63, 0.84),
    },
}

ABSORBED_DIFF_EPS = 1e-6
ZERO_OUTPUT_EPS = 1e-6
CONTROLLER_SATURATION_ATOL = 1e-6

ACTION_CHAIN_STAGE_NAMES: tuple[str, ...] = (
    "raw_action",
    "decoded_action",
    "absolute_action",
    "controller_input",
)
MODE_PAIR_SUMMARY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("positive_vs_negative", "positive", "negative"),
    ("positive_vs_omit", "positive", "omit"),
    ("negative_vs_omit", "negative", "omit"),
)

JsonDict = dict[str, object]
FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


class BranchContract(TypedDict):
    branch: str
    embodiment_tag: str
    public_anchor_comparable: bool
    action_order: list[str]
    action_dims: dict[str, int]
    state_order: list[str]
    state_dims: dict[str, int]
    relative_action_keys: list[str]
    absolute_action_keys: list[str]
    reference_state_keys: dict[str, str]
    action_representation_by_key: dict[str, str]
    policy_horizon_expected: int
    n_action_steps_expected: int | None
    source_artifacts: dict[str, str]
    controller_provenance: Mapping[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_action_chain_telemetry.py",
        description=(
            "Generate four-point action-chain telemetry for GR00T branch contracts: "
            "raw model action -> decoded/denormalized -> relative-to-absolute -> "
            "final controller input."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--branch",
        choices=list(ALL_BRANCHES),
        required=True,
        help="Frozen branch contract to audit.",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to branch-specific action_chain_telemetry_*.json.",
    )
    return parser


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object in {path}, got {type(payload).__name__}"
        )
    return cast(JsonDict, dict(payload))


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return cast(Mapping[str, object], value)


def _as_string_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(
                f"{field_name}[{index}] must be a string, got {type(item).__name__}"
            )
        result.append(item)
    return result


def _as_int_mapping(value: object, *, field_name: str) -> dict[str, int]:
    mapping = _as_mapping(value, field_name=field_name)
    result: dict[str, int] = {}
    for key, raw in mapping.items():
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise TypeError(
                f"{field_name}.{key} must be an int, got {type(raw).__name__}"
            )
        result[str(key)] = int(raw)
    return result


def _as_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f"output must be a file path, got existing directory: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _rel_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def default_output_path_for_branch(branch: str) -> Path:
    if branch not in DEFAULT_OUTPUTS:
        raise KeyError(f"unsupported branch for default output: {branch}")
    return DEFAULT_OUTPUTS[branch]


def build_execution_surface_contract(contract: BranchContract) -> dict[str, object]:
    return {
        "policy_horizon_expected": int(contract["policy_horizon_expected"]),
        "n_action_steps_expected": contract["n_action_steps_expected"],
        "relative_action_keys": list(contract["relative_action_keys"]),
        "absolute_action_keys": list(contract["absolute_action_keys"]),
        "action_representation_by_key": dict(contract["action_representation_by_key"]),
        "relative_to_absolute_rule": {
            "enabled_for_relative_action_keys": bool(contract["relative_action_keys"]),
            "reference_state_timestep": "last",
            "reference_state_keys": dict(contract["reference_state_keys"]),
        },
        "must_not_conflate_horizon_and_execution": True,
        "repo_local_formalization": {
            "field_names_are_repo_local": True,
            "upstream_policy_horizon_authority": "action.delta_indices",
            "upstream_execution_steps_authority": "rollout --n_action_steps",
            "note": (
                "Telemetry re-states repo-local contract field names for drift detection; "
                "they summarize upstream semantics but are not upstream official JSON field names."
            ),
        },
    }


def load_branch_contract(branch: str) -> BranchContract:
    if branch == BRANCH_UNITREE_G1:
        audit_path = (
            REPO_ROOT
            / "agent"
            / "artifacts"
            / "gr00t_anchor_controller_recap"
            / "unitree_g1"
            / "controller_audit_unitree_g1.json"
        )
        payload = _read_json(audit_path)
        relative_to_absolute = _as_mapping(
            payload["relative_to_absolute_processor"],
            field_name="relative_to_absolute_processor",
        )
        return {
            "branch": BRANCH_UNITREE_G1,
            "embodiment_tag": str(payload["embodiment_tag"]),
            "public_anchor_comparable": True,
            "action_order": _as_string_list(
                payload["action_order_expected"], field_name="action_order_expected"
            ),
            "action_dims": _as_int_mapping(
                payload["action_dims_expected"], field_name="action_dims_expected"
            ),
            "state_order": _as_string_list(
                payload["state_order_expected"], field_name="state_order_expected"
            ),
            "state_dims": _as_int_mapping(
                payload["state_dims_expected"], field_name="state_dims_expected"
            ),
            "relative_action_keys": _as_string_list(
                payload["relative_action_keys"], field_name="relative_action_keys"
            ),
            "absolute_action_keys": _as_string_list(
                payload["absolute_action_keys"], field_name="absolute_action_keys"
            ),
            "reference_state_keys": {
                str(key): str(value)
                for key, value in _as_mapping(
                    relative_to_absolute["reference_state_keys"],
                    field_name="relative_to_absolute_processor.reference_state_keys",
                ).items()
            },
            "action_representation_by_key": {
                str(key): str(value).upper()
                for key, value in _as_mapping(
                    payload["action_representation_by_key"],
                    field_name="action_representation_by_key",
                ).items()
            },
            "policy_horizon_expected": _as_int(
                payload["policy_horizon_expected"], field_name="policy_horizon_expected"
            ),
            "n_action_steps_expected": _as_int(
                payload["n_action_steps_expected"], field_name="n_action_steps_expected"
            ),
            "source_artifacts": {
                "controller_audit": _rel_repo(audit_path),
                "task_evidence": ".sisyphus/evidence/task-5-unitree-audit.json",
            },
            "controller_provenance": cast(
                Mapping[str, Any], payload["controller_provenance"]
            ),
        }

    if branch == BRANCH_NEW_EMBODIMENT:
        audit_path = (
            REPO_ROOT
            / "agent"
            / "artifacts"
            / "gr00t_anchor_controller_recap"
            / "new_embodiment"
            / "controller_audit_new_embodiment.json"
        )
        manifest_path = (
            REPO_ROOT
            / "agent"
            / "artifacts"
            / "gr00t_anchor_controller_recap"
            / "new_embodiment"
            / "branch_manifest.json"
        )
        modality_path = (
            REPO_ROOT / "work" / "configs" / "new_embodiment" / "modality_config.json"
        )
        audit_payload = _read_json(audit_path)
        modality_contract = (
            gr00t_controller_audit_new_embodiment.load_modality_contract(modality_path)
        )
        relative_policy = _as_mapping(
            audit_payload["relative_action_policy"],
            field_name="relative_action_policy",
        )
        return {
            "branch": BRANCH_NEW_EMBODIMENT,
            "embodiment_tag": str(audit_payload["embodiment_tag"]),
            "public_anchor_comparable": False,
            "action_order": _as_string_list(
                modality_contract["action_order_expected"],
                field_name="modality_contract.action_order_expected",
            ),
            "action_dims": _as_int_mapping(
                modality_contract["action_dims_expected"],
                field_name="modality_contract.action_dims_expected",
            ),
            "state_order": _as_string_list(
                modality_contract["state_order_expected"],
                field_name="modality_contract.state_order_expected",
            ),
            "state_dims": _as_int_mapping(
                modality_contract["state_dims_expected"],
                field_name="modality_contract.state_dims_expected",
            ),
            "relative_action_keys": [
                str(item)
                for item in _as_string_list(
                    relative_policy["relative_action_keys"],
                    field_name="relative_action_policy.relative_action_keys",
                )
            ],
            "absolute_action_keys": [
                str(item)
                for item in _as_string_list(
                    relative_policy["absolute_action_keys"],
                    field_name="relative_action_policy.absolute_action_keys",
                )
            ],
            "reference_state_keys": {
                str(key): str(value)
                for key, value in _as_mapping(
                    relative_policy["reference_state_keys"],
                    field_name="relative_action_policy.reference_state_keys",
                ).items()
            },
            "action_representation_by_key": {
                str(key): str(value).upper()
                for key, value in _as_mapping(
                    relative_policy["action_representation_by_key"],
                    field_name="relative_action_policy.action_representation_by_key",
                ).items()
            },
            "policy_horizon_expected": _as_int(
                audit_payload["policy_horizon_expected"],
                field_name="policy_horizon_expected",
            ),
            "n_action_steps_expected": None,
            "source_artifacts": {
                "controller_audit": _rel_repo(audit_path),
                "branch_manifest": _rel_repo(manifest_path),
                "modality_config": _rel_repo(modality_path),
                "task_evidence": ".sisyphus/evidence/task-6-new-embodiment-audit.json",
            },
            "controller_provenance": cast(
                Mapping[str, Any], audit_payload["controller_provenance"]
            ),
        }

    raise ValueError(f"unsupported branch: {branch}")


def _to_serializable_array(arr: FloatArray) -> list[list[float]]:
    rows = np.asarray(arr, dtype=np.float32).tolist()
    return [
        [float(v) for v in cast(list[float], row)]
        for row in cast(list[list[float]], rows)
    ]


def build_action_chain_contract_surface(branch: str) -> dict[str, Any]:
    contract = load_branch_contract(branch)
    return {
        "branch": contract["branch"],
        "embodiment_tag": contract["embodiment_tag"],
        "canonical_stage_names": list(ACTION_CHAIN_STAGE_NAMES),
        "action_group_order": list(contract["action_order"]),
        "action_dims": dict(contract["action_dims"]),
        "relative_action_keys": list(contract["relative_action_keys"]),
        "absolute_action_keys": list(contract["absolute_action_keys"]),
        "action_representation_by_key": dict(contract["action_representation_by_key"]),
        "execution_surface_contract": build_execution_surface_contract(contract),
    }


def _serialize_group_stage_value(
    value: object, *, dims: int, field_name: str
) -> tuple[list[list[float]], list[int]]:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        if dims != 1:
            raise ValueError(
                f"{field_name} scalar input only supported for 1-D groups, got dims={dims}"
            )
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if arr.shape[0] != dims:
            raise ValueError(
                f"{field_name} expected 1-D shape {(dims,)}, got {tuple(arr.shape)}"
            )
        arr = arr.reshape(1, dims)
    elif arr.ndim == 2:
        if arr.shape[1] != dims:
            raise ValueError(
                f"{field_name} expected trailing dimension {dims}, got {tuple(arr.shape)}"
            )
    elif arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"{field_name} expected singleton batch dimension for rank-3 (B,T,D) surface, got {tuple(arr.shape)}"
            )
        if arr.shape[2] != dims:
            raise ValueError(
                f"{field_name} expected trailing dimension {dims}, got {tuple(arr.shape)}"
            )
        arr = arr[0]
    else:
        raise ValueError(
            f"{field_name} must be a scalar, 1-D, 2-D, or singleton-batch 3-D (B,T,D) array-like surface, got rank {arr.ndim}"
        )
    normalized = np.asarray(arr, dtype=np.float32)
    return _to_serializable_array(normalized), [int(v) for v in normalized.shape]


def build_grouped_action_chain_sidecar(
    branch: str,
    *,
    stage_group_values: Mapping[str, Mapping[str, object] | None],
    stage_unavailable_reasons: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    contract = load_branch_contract(branch)
    unavailable_reasons = dict(stage_unavailable_reasons or {})
    per_group_stage_surfaces: dict[str, Any] = {}
    stage_group_coverage: dict[str, Any] = {}
    for stage_name in ACTION_CHAIN_STAGE_NAMES:
        stage_group_coverage[stage_name] = {
            "available_group_count": 0,
            "unavailable_group_count": int(len(contract["action_order"])),
            "fully_available": False,
        }

    for key in contract["action_order"]:
        group_payload: dict[str, Any] = {
            "group": key,
            "dimension": int(contract["action_dims"][key]),
            "action_representation": str(
                contract["action_representation_by_key"][key]
            ).upper(),
            "reference_state_key": contract["reference_state_keys"].get(key),
            "stages": {},
        }
        dims = int(contract["action_dims"][key])
        for stage_name in ACTION_CHAIN_STAGE_NAMES:
            stage_entry: dict[str, Any]
            raw_stage_values = stage_group_values.get(stage_name)
            stage_values: dict[str, object] | None
            if raw_stage_values is None:
                stage_values = None
            else:
                stage_values = {}
                for raw_group_name, raw_group_value in raw_stage_values.items():
                    group_name = str(raw_group_name)
                    if group_name in contract["action_order"]:
                        stage_values[group_name] = raw_group_value
                        continue
                    if group_name.startswith("action."):
                        alias = group_name[len("action.") :]
                        if alias in contract["action_order"]:
                            stage_values[alias] = raw_group_value
            if stage_values is None:
                reason = unavailable_reasons.get(stage_name) or (
                    f"{stage_name} surface was not provided on this path"
                )
                stage_entry = {
                    "available": False,
                    "values": None,
                    "shape": None,
                    "unavailable_reason": str(reason),
                }
            elif key not in stage_values:
                stage_entry = {
                    "available": False,
                    "values": None,
                    "shape": None,
                    "unavailable_reason": (
                        f"{stage_name} surface did not expose group {key!r}; preserving the seven-group split without fabricating values"
                    ),
                }
            else:
                try:
                    serialized, shape = _serialize_group_stage_value(
                        stage_values[key],
                        dims=dims,
                        field_name=f"{stage_name}.{key}",
                    )
                except Exception as exc:
                    stage_entry = {
                        "available": False,
                        "values": None,
                        "shape": None,
                        "unavailable_reason": (
                            f"{stage_name} surface for group {key!r} was rejected: {_exception_message(exc)}"
                        ),
                    }
                else:
                    stage_group_coverage[stage_name]["available_group_count"] += 1
                    stage_entry = {
                        "available": True,
                        "values": serialized,
                        "shape": shape,
                        "unavailable_reason": None,
                    }
            group_payload["stages"][stage_name] = stage_entry
        per_group_stage_surfaces[key] = group_payload

    total_groups = int(len(contract["action_order"]))
    for stage_name in ACTION_CHAIN_STAGE_NAMES:
        available_group_count = int(
            stage_group_coverage[stage_name]["available_group_count"]
        )
        stage_group_coverage[stage_name]["unavailable_group_count"] = (
            total_groups - available_group_count
        )
        stage_group_coverage[stage_name]["fully_available"] = (
            available_group_count == total_groups
        )

    return {
        **build_action_chain_contract_surface(branch),
        "per_group_stage_surfaces": per_group_stage_surfaces,
        "stage_group_coverage": stage_group_coverage,
    }


def _build_unavailable_pair_summary(
    branch: str,
    *,
    pair_name: str,
    left_mode: str,
    right_mode: str,
    unavailable_reason: str,
) -> dict[str, Any]:
    contract_surface = build_action_chain_contract_surface(branch)
    return {
        **contract_surface,
        "pair_name": pair_name,
        "left_mode": left_mode,
        "right_mode": right_mode,
        "available": False,
        "unavailable_reason": unavailable_reason,
        "available_group_count_by_stage": {
            stage_name: 0 for stage_name in ACTION_CHAIN_STAGE_NAMES
        },
        "difference_group_count_by_stage": {
            stage_name: 0 for stage_name in ACTION_CHAIN_STAGE_NAMES
        },
        "difference_groups_by_stage": {
            stage_name: [] for stage_name in ACTION_CHAIN_STAGE_NAMES
        },
        "per_group": {},
    }


def _pair_stage_delta_summary(
    *,
    left_stage: Mapping[str, object],
    right_stage: Mapping[str, object],
    left_mode: str,
    right_mode: str,
) -> dict[str, Any]:
    left_available = bool(left_stage.get("available", False))
    right_available = bool(right_stage.get("available", False))
    summary: dict[str, Any] = {
        "available_in_both_modes": bool(left_available and right_available),
        "left_mode": left_mode,
        "right_mode": right_mode,
        "left_available": left_available,
        "right_available": right_available,
        "left_unavailable_reason": left_stage.get("unavailable_reason"),
        "right_unavailable_reason": right_stage.get("unavailable_reason"),
        "left_shape": left_stage.get("shape"),
        "right_shape": right_stage.get("shape"),
        "difference_present": False,
        "l2_delta": None,
        "max_abs_delta": None,
        "mean_abs_delta": None,
    }
    if not left_available or not right_available:
        return summary

    left_values = np.asarray(left_stage.get("values"), dtype=np.float32)
    right_values = np.asarray(right_stage.get("values"), dtype=np.float32)
    if left_values.shape != right_values.shape:
        summary["available_in_both_modes"] = False
        summary["shape_mismatch"] = {
            "left_shape": [int(v) for v in left_values.shape],
            "right_shape": [int(v) for v in right_values.shape],
        }
        return summary

    diff = left_values - right_values
    abs_diff = np.abs(diff)
    l2_delta = _round_float(float(np.linalg.norm(diff.reshape(-1), ord=2)))
    max_abs_delta = (
        _round_float(float(np.max(abs_diff))) if int(abs_diff.size) > 0 else 0.0
    )
    mean_abs_delta = (
        _round_float(float(np.mean(abs_diff))) if int(abs_diff.size) > 0 else 0.0
    )
    summary["difference_present"] = bool(l2_delta > ABSORBED_DIFF_EPS)
    summary["l2_delta"] = l2_delta
    summary["max_abs_delta"] = max_abs_delta
    summary["mean_abs_delta"] = mean_abs_delta
    return summary


def build_action_chain_mode_pair_summary(
    branch: str,
    *,
    pair_name: str,
    left_mode: str,
    right_mode: str,
    left_sidecar: Mapping[str, object],
    right_sidecar: Mapping[str, object],
) -> dict[str, Any]:
    contract = load_branch_contract(branch)
    left_groups = cast(
        Mapping[str, Mapping[str, object]], left_sidecar["per_group_stage_surfaces"]
    )
    right_groups = cast(
        Mapping[str, Mapping[str, object]], right_sidecar["per_group_stage_surfaces"]
    )

    available_group_count_by_stage = {
        stage_name: 0 for stage_name in ACTION_CHAIN_STAGE_NAMES
    }
    difference_group_count_by_stage = {
        stage_name: 0 for stage_name in ACTION_CHAIN_STAGE_NAMES
    }
    difference_groups_by_stage: dict[str, list[str]] = {
        stage_name: [] for stage_name in ACTION_CHAIN_STAGE_NAMES
    }
    per_group: dict[str, Any] = {}

    for key in contract["action_order"]:
        left_group = cast(Mapping[str, object], left_groups[key])
        right_group = cast(Mapping[str, object], right_groups[key])
        left_stages = cast(Mapping[str, Mapping[str, object]], left_group["stages"])
        right_stages = cast(Mapping[str, Mapping[str, object]], right_group["stages"])
        group_summary: dict[str, Any] = {
            "group": key,
            "dimension": int(contract["action_dims"][key]),
            "action_representation": str(
                contract["action_representation_by_key"][key]
            ).upper(),
            "reference_state_key": contract["reference_state_keys"].get(key),
            "stages": {},
        }
        for stage_name in ACTION_CHAIN_STAGE_NAMES:
            stage_summary = _pair_stage_delta_summary(
                left_stage=left_stages[stage_name],
                right_stage=right_stages[stage_name],
                left_mode=left_mode,
                right_mode=right_mode,
            )
            group_summary["stages"][stage_name] = stage_summary
            if bool(stage_summary["available_in_both_modes"]):
                available_group_count_by_stage[stage_name] += 1
            if bool(stage_summary["difference_present"]):
                difference_group_count_by_stage[stage_name] += 1
                difference_groups_by_stage[stage_name].append(key)
        per_group[key] = group_summary

    return {
        **build_action_chain_contract_surface(branch),
        "pair_name": pair_name,
        "left_mode": left_mode,
        "right_mode": right_mode,
        "available": True,
        "unavailable_reason": None,
        "available_group_count_by_stage": available_group_count_by_stage,
        "difference_group_count_by_stage": difference_group_count_by_stage,
        "difference_groups_by_stage": difference_groups_by_stage,
        "per_group": per_group,
    }


def build_action_chain_mode_pair_summaries(
    branch: str,
    *,
    mode_sidecars: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    pair_summaries: dict[str, Any] = {}
    for pair_name, left_mode, right_mode in MODE_PAIR_SUMMARY_SPECS:
        if left_mode not in mode_sidecars or right_mode not in mode_sidecars:
            pair_summaries[pair_name] = _build_unavailable_pair_summary(
                branch,
                pair_name=pair_name,
                left_mode=left_mode,
                right_mode=right_mode,
                unavailable_reason=(
                    f"mode sidecars must include both {left_mode!r} and {right_mode!r} to compute {pair_name}"
                ),
            )
            continue
        pair_summaries[pair_name] = build_action_chain_mode_pair_summary(
            branch,
            pair_name=pair_name,
            left_mode=left_mode,
            right_mode=right_mode,
            left_sidecar=mode_sidecars[left_mode],
            right_sidecar=mode_sidecars[right_mode],
        )
    return pair_summaries


def build_pair_execution_surface_summary(
    pair_summary: Mapping[str, object],
    *,
    terminal_stage: str = "controller_input",
) -> dict[str, Any]:
    if terminal_stage not in ACTION_CHAIN_STAGE_NAMES:
        raise ValueError(
            f"terminal_stage must be one of {list(ACTION_CHAIN_STAGE_NAMES)}, got {terminal_stage!r}"
        )
    raw_per_group = pair_summary.get("per_group")
    per_group = (
        cast(Mapping[str, Mapping[str, object]], raw_per_group)
        if isinstance(raw_per_group, Mapping)
        else {}
    )
    disappearance_counts = {
        "model": 0,
        "decode": 0,
        "relative_to_absolute": 0,
        terminal_stage: 0,
        "survived_to_terminal_stage": 0,
    }
    disappearance_groups: dict[str, list[str]] = {
        key: [] for key in disappearance_counts
    }
    summarized_groups: dict[str, Any] = {}
    for group_name, group_payload in per_group.items():
        stages = cast(
            Mapping[str, Mapping[str, object]], group_payload.get("stages", {})
        )
        checkpoint_distinction = {
            stage_name: bool(
                cast(Mapping[str, object], stages.get(stage_name, {})).get(
                    "difference_present", False
                )
            )
            for stage_name in ACTION_CHAIN_STAGE_NAMES
        }
        deepest_distinct_checkpoint = None
        for stage_name in ACTION_CHAIN_STAGE_NAMES:
            if checkpoint_distinction[stage_name]:
                deepest_distinct_checkpoint = stage_name
        if deepest_distinct_checkpoint is None:
            disappearance_stage = "model"
        elif not checkpoint_distinction["decoded_action"]:
            disappearance_stage = "decode"
        elif not checkpoint_distinction["absolute_action"]:
            disappearance_stage = "relative_to_absolute"
        elif not checkpoint_distinction[terminal_stage]:
            disappearance_stage = terminal_stage
        else:
            disappearance_stage = None
        disappearance_key = (
            "survived_to_terminal_stage"
            if disappearance_stage is None
            else str(disappearance_stage)
        )
        disappearance_counts[disappearance_key] += 1
        disappearance_groups[disappearance_key].append(str(group_name))
        summarized_groups[str(group_name)] = {
            "group": str(group_name),
            "action_representation": str(
                group_payload.get("action_representation", "UNKNOWN")
            ).upper(),
            "reference_state_key": group_payload.get("reference_state_key"),
            "difference_present_by_checkpoint": checkpoint_distinction,
            "deepest_distinct_checkpoint": deepest_distinct_checkpoint,
            "difference_disappeared_at": disappearance_stage,
        }
    return {
        "pair_name": str(pair_summary.get("pair_name", "")),
        "available": bool(pair_summary.get("available", False)),
        "terminal_stage_used": terminal_stage,
        "canonical_stage_names": list(ACTION_CHAIN_STAGE_NAMES),
        "difference_disappearance_counts": disappearance_counts,
        "difference_disappearance_groups": disappearance_groups,
        "per_group": summarized_groups,
    }


def build_mode_pair_execution_surface_summaries(
    pair_summaries: Mapping[str, Mapping[str, object]],
    *,
    terminal_stage: str = "controller_input",
) -> dict[str, Any]:
    return {
        str(pair_name): build_pair_execution_surface_summary(
            pair_summary,
            terminal_stage=terminal_stage,
        )
        for pair_name, pair_summary in pair_summaries.items()
    }


def _vector(values: Sequence[float], *, dims: int) -> FloatArray:
    arr: FloatArray = np.asarray(list(values), dtype=np.float32)
    if arr.shape != (dims,):
        raise ValueError(f"expected vector shape {(dims,)}, got {tuple(arr.shape)}")
    return arr


def _make_time_series(
    base: FloatArray, *, amplitude: float, horizon: int
) -> FloatArray:
    t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
    dim_scale = (np.arange(base.shape[0], dtype=np.float32) + 1.0)[None, :] * amplitude
    return np.asarray(base[None, :] + t * dim_scale, dtype=np.float32)


def _build_reference_state(contract: BranchContract) -> dict[str, FloatArray]:
    state_dims = contract["state_dims"]
    return {
        "left_leg": np.linspace(-0.2, 0.3, state_dims["left_leg"], dtype=np.float32)[
            None, :
        ],
        "right_leg": np.linspace(
            0.25, -0.15, state_dims["right_leg"], dtype=np.float32
        )[None, :],
        "waist": np.asarray([[0.10, -0.05, 0.04]], dtype=np.float32),
        "left_arm": np.asarray(
            [[0.55, 0.20, 0.10, -0.10, 0.15, 0.05, -0.05]], dtype=np.float32
        ),
        "right_arm": np.asarray(
            [[-0.40, 0.10, -0.20, 0.15, -0.10, 0.08, 0.03]], dtype=np.float32
        ),
        "left_hand": np.asarray(
            [[0.15, 0.08, 0.05, -0.04, 0.02, 0.01, -0.02]], dtype=np.float32
        ),
        "right_hand": np.zeros((1, state_dims["right_hand"]), dtype=np.float32),
    }


def _build_probe_pair(contract: BranchContract) -> dict[str, object]:
    horizon = int(contract["policy_horizon_expected"])
    action_dims = contract["action_dims"]

    baseline: dict[str, FloatArray] = {
        "left_arm": _make_time_series(
            _vector(
                [0.55, 0.20, 0.10, -0.10, 0.15, 0.05, -0.05],
                dims=action_dims["left_arm"],
            ),
            amplitude=0.002,
            horizon=horizon,
        ),
        "right_arm": _make_time_series(
            _vector(
                [-0.25, 0.15, -0.05, 0.10, -0.02, 0.04, 0.00],
                dims=action_dims["right_arm"],
            ),
            amplitude=0.003,
            horizon=horizon,
        ),
        "left_hand": _make_time_series(
            _vector(
                [0.10, -0.15, 0.20, -0.05, 0.03, -0.02, 0.01],
                dims=action_dims["left_hand"],
            ),
            amplitude=0.001,
            horizon=horizon,
        ),
        "right_hand": np.zeros((horizon, action_dims["right_hand"]), dtype=np.float32),
        "waist": _make_time_series(
            _vector([0.10, -0.05, 0.08], dims=action_dims["waist"]),
            amplitude=0.002,
            horizon=horizon,
        ),
        "navigate_command": _make_time_series(
            _vector([0.10, 0.00, 0.05], dims=action_dims["navigate_command"]),
            amplitude=0.002,
            horizon=horizon,
        ),
        "base_height_command": _make_time_series(
            _vector([0.20], dims=action_dims["base_height_command"]),
            amplitude=0.0,
            horizon=horizon,
        ),
    }

    probe = {key: np.array(value, copy=True) for key, value in baseline.items()}
    probe["left_arm"][:, 0] = baseline["left_arm"][:, 0] + 0.40
    probe["right_arm"][:, 1] = baseline["right_arm"][:, 1] + 0.10
    probe["left_hand"][:, 2] = baseline["left_hand"][:, 2] + 0.95
    probe["waist"][:, 0] = baseline["waist"][:, 0] + 0.12
    probe["navigate_command"][:, 0] = baseline["navigate_command"][:, 0] + 0.30
    probe["base_height_command"][:, 0] = baseline["base_height_command"][:, 0] + 0.15

    return {
        "state": _build_reference_state(contract),
        "raw_action": {
            "baseline": baseline,
            "probe": probe,
        },
    }


def _decode_params(contract: BranchContract) -> dict[str, dict[str, FloatArray]]:
    params: dict[str, dict[str, FloatArray]] = {}
    for key in contract["action_order"]:
        dims = int(contract["action_dims"][key])
        lower, upper = DEFAULT_DECODE_LIMITS_BY_BRANCH[contract["branch"]][key]
        params[key] = {
            "min": np.full((dims,), lower, dtype=np.float32),
            "max": np.full((dims,), upper, dtype=np.float32),
        }
    return params


def _controller_limits(contract: BranchContract) -> dict[str, dict[str, FloatArray]]:
    params: dict[str, dict[str, FloatArray]] = {}
    for key in contract["action_order"]:
        dims = int(contract["action_dims"][key])
        lower, upper = DEFAULT_CONTROLLER_LIMITS_BY_BRANCH[contract["branch"]][key]
        params[key] = {
            "low": np.full((dims,), lower, dtype=np.float32),
            "high": np.full((dims,), upper, dtype=np.float32),
        }
    return params


def _unnormalize_minmax(
    raw_action: FloatArray, params: Mapping[str, FloatArray]
) -> tuple[FloatArray, BoolArray]:
    clipped = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
    decoded = (clipped + 1.0) / 2.0 * (params["max"] - params["min"]) + params["min"]
    clip_mask = np.abs(np.asarray(raw_action, dtype=np.float32)) > 1.0
    return np.asarray(decoded, dtype=np.float32), np.asarray(clip_mask, dtype=bool)


def _relative_to_absolute(
    contract: BranchContract,
    *,
    key: str,
    decoded_action: FloatArray,
    state: Mapping[str, FloatArray],
) -> FloatArray:
    arr = np.asarray(decoded_action, dtype=np.float32)
    if key not in contract["relative_action_keys"]:
        return arr
    reference_state_key = contract["reference_state_keys"][key]
    reference_state = np.asarray(state[reference_state_key], dtype=np.float32)
    if reference_state.ndim != 2 or reference_state.shape[0] < 1:
        raise ValueError(
            f"expected reference state shape (T_state, D) for {reference_state_key}, got {reference_state.shape}"
        )
    reference_last = reference_state[-1]
    return np.asarray(arr + reference_last[None, :], dtype=np.float32)


def _controller_clip(
    absolute_action: FloatArray,
    *,
    limits: Mapping[str, FloatArray],
) -> tuple[FloatArray, BoolArray, BoolArray]:
    arr = np.asarray(absolute_action, dtype=np.float32)
    clipped = np.clip(arr, limits["low"], limits["high"])
    clip_mask = np.abs(clipped - arr) > CONTROLLER_SATURATION_ATOL
    saturation_mask = np.logical_and(
        np.isfinite(clipped),
        np.logical_or(
            np.isclose(clipped, limits["low"], atol=CONTROLLER_SATURATION_ATOL),
            np.isclose(clipped, limits["high"], atol=CONTROLLER_SATURATION_ATOL),
        ),
    )
    return (
        np.asarray(clipped, dtype=np.float32),
        np.asarray(clip_mask, dtype=bool),
        np.asarray(saturation_mask, dtype=bool),
    )


def _fraction(mask: BoolArray) -> float:
    total = int(mask.size)
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(mask) / total)


def _round_float(value: float, *, digits: int = 8) -> float:
    return float(round(float(value), digits))


def _stage_stats(
    values: FloatArray,
    *,
    clip_mask: BoolArray | None = None,
    saturation_mask: BoolArray | None = None,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    abs_arr = np.abs(arr)
    zero_fraction = _fraction(abs_arr <= ZERO_OUTPUT_EPS)
    clip_rate = (
        _fraction(np.asarray(clip_mask, dtype=bool)) if clip_mask is not None else 0.0
    )
    saturation_rate = (
        _fraction(np.asarray(saturation_mask, dtype=bool))
        if saturation_mask is not None
        else 0.0
    )
    return {
        "shape": [int(v) for v in arr.shape],
        "mean": _round_float(float(np.mean(arr))) if int(arr.size) > 0 else 0.0,
        "variance": _round_float(float(np.var(arr))) if int(arr.size) > 0 else 0.0,
        "max_abs": _round_float(float(np.max(abs_arr))) if int(arr.size) > 0 else 0.0,
        "clip_rate": _round_float(clip_rate),
        "zero_output_rate": _round_float(zero_fraction),
        "saturation_rate": _round_float(saturation_rate),
    }


def _l2_norm(a: FloatArray, b: FloatArray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return _round_float(float(np.linalg.norm(diff.reshape(-1), ord=2)))


def _zero_motion_flag(values: FloatArray) -> bool:
    return bool(np.max(np.abs(np.asarray(values, dtype=np.float32))) <= ZERO_OUTPUT_EPS)


def _build_group_telemetry(
    contract: BranchContract,
    *,
    key: str,
    state: Mapping[str, FloatArray],
    raw_baseline: FloatArray,
    raw_probe: FloatArray,
    decode_params: Mapping[str, FloatArray],
    controller_limits: Mapping[str, FloatArray],
) -> dict[str, Any]:
    decoded_baseline, decode_clip_baseline = _unnormalize_minmax(
        raw_baseline, decode_params
    )
    decoded_probe, decode_clip_probe = _unnormalize_minmax(raw_probe, decode_params)

    absolute_baseline = _relative_to_absolute(
        contract,
        key=key,
        decoded_action=decoded_baseline,
        state=state,
    )
    absolute_probe = _relative_to_absolute(
        contract,
        key=key,
        decoded_action=decoded_probe,
        state=state,
    )

    controller_baseline, controller_clip_baseline, controller_sat_baseline = (
        _controller_clip(
            absolute_baseline,
            limits=controller_limits,
        )
    )
    controller_probe, controller_clip_probe, controller_sat_probe = _controller_clip(
        absolute_probe,
        limits=controller_limits,
    )

    raw_diff_l2 = _l2_norm(raw_baseline, raw_probe)
    decoded_diff_l2 = _l2_norm(decoded_baseline, decoded_probe)
    absolute_diff_l2 = _l2_norm(absolute_baseline, absolute_probe)
    controller_diff_l2 = _l2_norm(controller_baseline, controller_probe)

    model_insensitive = bool(raw_diff_l2 <= ABSORBED_DIFF_EPS)
    controller_absorbed = bool(
        absolute_diff_l2 > ABSORBED_DIFF_EPS and controller_diff_l2 <= ABSORBED_DIFF_EPS
    )
    if model_insensitive:
        disappearance_stage = "model"
    elif decoded_diff_l2 <= ABSORBED_DIFF_EPS:
        disappearance_stage = "decode"
    elif absolute_diff_l2 <= ABSORBED_DIFF_EPS:
        disappearance_stage = "relative_to_absolute"
    elif controller_diff_l2 <= ABSORBED_DIFF_EPS:
        disappearance_stage = "controller_input"
    else:
        disappearance_stage = None

    return {
        "group": key,
        "dimension": int(contract["action_dims"][key]),
        "action_representation": str(
            contract["action_representation_by_key"][key]
        ).upper(),
        "reference_state_key": contract["reference_state_keys"].get(key),
        "raw_action": {
            "baseline": _to_serializable_array(raw_baseline),
            "probe": _to_serializable_array(raw_probe),
        },
        "decoded_action": {
            "baseline": _to_serializable_array(decoded_baseline),
            "probe": _to_serializable_array(decoded_probe),
        },
        "absolute_action": {
            "baseline": _to_serializable_array(absolute_baseline),
            "probe": _to_serializable_array(absolute_probe),
        },
        "controller_input": {
            "baseline": _to_serializable_array(controller_baseline),
            "probe": _to_serializable_array(controller_probe),
        },
        "stages": {
            "raw_action": {
                "baseline": _stage_stats(raw_baseline),
                "probe": _stage_stats(raw_probe),
            },
            "decoded_action": {
                "baseline": _stage_stats(
                    decoded_baseline, clip_mask=decode_clip_baseline
                ),
                "probe": _stage_stats(decoded_probe, clip_mask=decode_clip_probe),
            },
            "absolute_action": {
                "baseline": _stage_stats(absolute_baseline),
                "probe": _stage_stats(absolute_probe),
            },
            "controller_input": {
                "baseline": _stage_stats(
                    controller_baseline,
                    clip_mask=controller_clip_baseline,
                    saturation_mask=controller_sat_baseline,
                ),
                "probe": _stage_stats(
                    controller_probe,
                    clip_mask=controller_clip_probe,
                    saturation_mask=controller_sat_probe,
                ),
            },
        },
        "difference_metrics": {
            "raw_action_l2": raw_diff_l2,
            "decoded_action_l2": decoded_diff_l2,
            "absolute_action_l2": absolute_diff_l2,
            "controller_input_l2": controller_diff_l2,
            "difference_disappeared_at": disappearance_stage,
            "model_insensitive": model_insensitive,
            "controller_absorbed_upstream_difference": controller_absorbed,
        },
        "clip_rate": {
            "decoded_action": _round_float(
                max(_fraction(decode_clip_baseline), _fraction(decode_clip_probe))
            ),
            "controller_input": _round_float(
                max(
                    _fraction(controller_clip_baseline),
                    _fraction(controller_clip_probe),
                )
            ),
        },
        "saturation_rate": _round_float(
            max(_fraction(controller_sat_baseline), _fraction(controller_sat_probe))
        ),
        "zero_motion_flags": {
            "baseline_controller_input_all_zero": _zero_motion_flag(
                controller_baseline
            ),
            "probe_controller_input_all_zero": _zero_motion_flag(controller_probe),
            "all_zero_in_both": bool(
                _zero_motion_flag(controller_baseline)
                and _zero_motion_flag(controller_probe)
            ),
        },
    }


def build_telemetry_report(branch: str) -> dict[str, Any]:
    contract = load_branch_contract(branch)
    probe_pair = _build_probe_pair(contract)
    state = cast(Mapping[str, FloatArray], probe_pair["state"])
    raw_pair = cast(Mapping[str, Mapping[str, FloatArray]], probe_pair["raw_action"])
    decode_params = _decode_params(contract)
    controller_limits = _controller_limits(contract)

    per_group_stats: dict[str, Any] = {}
    raw_action: dict[str, Any] = {"baseline": {}, "probe": {}}
    decoded_action: dict[str, Any] = {"baseline": {}, "probe": {}}
    absolute_action: dict[str, Any] = {"baseline": {}, "probe": {}}
    controller_input: dict[str, Any] = {"baseline": {}, "probe": {}}

    decode_clip_by_group: dict[str, float] = {}
    controller_clip_by_group: dict[str, float] = {}
    saturation_by_group: dict[str, float] = {}
    zero_groups_baseline: list[str] = []
    zero_groups_probe: list[str] = []
    absorbed_groups: list[str] = []
    model_insensitive_groups: list[str] = []

    for key in contract["action_order"]:
        group_payload = _build_group_telemetry(
            contract,
            key=key,
            state=state,
            raw_baseline=np.asarray(raw_pair["baseline"][key], dtype=np.float32),
            raw_probe=np.asarray(raw_pair["probe"][key], dtype=np.float32),
            decode_params=decode_params[key],
            controller_limits=controller_limits[key],
        )
        per_group_stats[key] = group_payload
        raw_action["baseline"][key] = group_payload["raw_action"]["baseline"]
        raw_action["probe"][key] = group_payload["raw_action"]["probe"]
        decoded_action["baseline"][key] = group_payload["decoded_action"]["baseline"]
        decoded_action["probe"][key] = group_payload["decoded_action"]["probe"]
        absolute_action["baseline"][key] = group_payload["absolute_action"]["baseline"]
        absolute_action["probe"][key] = group_payload["absolute_action"]["probe"]
        controller_input["baseline"][key] = group_payload["controller_input"][
            "baseline"
        ]
        controller_input["probe"][key] = group_payload["controller_input"]["probe"]

        decode_clip_by_group[key] = float(group_payload["clip_rate"]["decoded_action"])
        controller_clip_by_group[key] = float(
            group_payload["clip_rate"]["controller_input"]
        )
        saturation_by_group[key] = float(group_payload["saturation_rate"])
        if bool(
            group_payload["zero_motion_flags"]["baseline_controller_input_all_zero"]
        ):
            zero_groups_baseline.append(key)
        if bool(group_payload["zero_motion_flags"]["probe_controller_input_all_zero"]):
            zero_groups_probe.append(key)
        diff_metrics = cast(Mapping[str, object], group_payload["difference_metrics"])
        if bool(diff_metrics["controller_absorbed_upstream_difference"]):
            absorbed_groups.append(key)
        if bool(diff_metrics["model_insensitive"]):
            model_insensitive_groups.append(key)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "branch": contract["branch"],
        "embodiment_tag": contract["embodiment_tag"],
        "public_anchor_comparable": bool(contract["public_anchor_comparable"]),
        "action_order": list(contract["action_order"]),
        "action_dims": dict(contract["action_dims"]),
        "state_order": list(contract["state_order"]),
        "state_dims": dict(contract["state_dims"]),
        "relative_action_keys": list(contract["relative_action_keys"]),
        "absolute_action_keys": list(contract["absolute_action_keys"]),
        "policy_horizon_expected": int(contract["policy_horizon_expected"]),
        "n_action_steps_expected": contract["n_action_steps_expected"],
        "execution_surface_contract": build_execution_surface_contract(contract),
        "source_artifacts": dict(contract["source_artifacts"]),
        "controller_provenance": dict(contract["controller_provenance"]),
        "state_reference": {
            key: _to_serializable_array(np.asarray(value, dtype=np.float32))
            for key, value in state.items()
        },
        "raw_action": raw_action,
        "decoded_action": decoded_action,
        "absolute_action": absolute_action,
        "controller_input": controller_input,
        "per_group_stats": per_group_stats,
        "clip_rate": {
            "decoded_action_overall": _round_float(
                float(np.mean(list(decode_clip_by_group.values())))
                if decode_clip_by_group
                else 0.0
            ),
            "controller_input_overall": _round_float(
                float(np.mean(list(controller_clip_by_group.values())))
                if controller_clip_by_group
                else 0.0
            ),
            "by_group": {
                key: {
                    "decoded_action": _round_float(decode_clip_by_group[key]),
                    "controller_input": _round_float(controller_clip_by_group[key]),
                }
                for key in contract["action_order"]
            },
        },
        "saturation_rate": {
            "overall": _round_float(
                float(np.mean(list(saturation_by_group.values())))
                if saturation_by_group
                else 0.0
            ),
            "by_group": {
                key: _round_float(saturation_by_group[key])
                for key in contract["action_order"]
            },
        },
        "zero_motion_flags": {
            "baseline_controller_input_zero_groups": zero_groups_baseline,
            "probe_controller_input_zero_groups": zero_groups_probe,
            "all_zero_in_both_groups": [
                key
                for key in contract["action_order"]
                if key in zero_groups_baseline and key in zero_groups_probe
            ],
        },
        "controller_absorbed_upstream_difference": bool(absorbed_groups),
        "controller_absorbed_groups": absorbed_groups,
        "model_insensitive_groups": model_insensitive_groups,
        "telemetry_notes": {
            "paired_probe_style": "same fixed reference state with baseline/probe action pair",
            "decode_rule": "minmax unnormalize with clipping to [-1, 1] before denormalization",
            "relative_to_absolute_rule": "relative joint actions add the last timestep of the reference state",
            "controller_input_rule": "absolute action clipped to branch-local controller diagnostic limits for saturation analysis",
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = _validate_output_path(
        args.output
        if args.output is not None
        else default_output_path_for_branch(str(args.branch))
    )
    report = build_telemetry_report(str(args.branch))
    report["output_path"] = _rel_repo(output_path)
    _ = _write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
