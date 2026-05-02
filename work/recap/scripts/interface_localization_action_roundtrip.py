from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "interface_localization_sprint"

ACTION_ROUNDTRIP_JSON_NAME = "action_chain_watchlist_split.json"
ACTION_ROUNDTRIP_SCHEMA_VERSION = "interface_localization_action_roundtrip_v1"
ACTION_ROUNDTRIP_ARTIFACT_KIND = "action_chain_watchlist_split"

CANONICAL_SPACE_NAME = "unitree_g1_absolute_joint_order_canonical_space_v1"
CANONICAL_TRANSFORM_NAME = "raw_to_decoded_to_absolute_then_controller_input_v1"

WATCH_BUCKET_ORDER: tuple[str, ...] = (
    "body_wrist_upper_limb_chain",
    "dex3_finger_hand_path",
)

RIGHT_ARM_JOINT_ORDER: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

RIGHT_HAND_JOINT_ORDER: tuple[str, ...] = (
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import interface_localization_contract
from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface_localization_action_roundtrip.py",
        description=(
            "Freeze a canonical-space action roundtrip watchlist that keeps the "
            "right-arm wrist chain distinct from the Dex3 finger-hand path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, the roundtrip JSON is written "
            "to <artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for the generated roundtrip JSON.",
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


def _relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _generation_command_for(output_dir: Path, repo_root: Path) -> str:
    display_output_dir = _relpath(repo_root, output_dir)
    return (
        "python3 work/recap/scripts/interface_localization_action_roundtrip.py "
        f"--output-dir {display_output_dir}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return cast(dict[str, Any], payload)


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


def _controller_audit_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "agent"
        / "artifacts"
        / "gr00t_anchor_controller_recap"
        / "unitree_g1"
        / "controller_audit_unitree_g1.json"
    )


def _action_telemetry_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "agent"
        / "artifacts"
        / "gr00t_anchor_controller_recap"
        / "unitree_g1"
        / "action_chain_telemetry_unitree_g1.json"
    )


def _reference(
    *,
    source_name: str,
    source_ref: str,
    evidence_role: str,
) -> dict[str, str]:
    return {
        "source_name": source_name,
        "source_ref": source_ref,
        "evidence_role": evidence_role,
    }


def default_watch_bucket_specs() -> dict[str, dict[str, Any]]:
    return {
        "body_wrist_upper_limb_chain": {
            "bucket_name": "body_wrist_upper_limb_chain",
            "status": "survived",
            "provenance_class": "static",
            "bucket_kind": "body_chain",
            "body_side": "right",
            "source_mapping": {
                "original_action_key": "right_arm",
                "original_state_key": "right_arm",
                "reference_state_key": "right_arm",
                "original_group_keys": ["action.right_arm", "state.right_arm"],
                "action_representation": "RELATIVE",
                "canonical_representation": "ABSOLUTE",
                "joint_order": list(RIGHT_ARM_JOINT_ORDER),
                "ownership_scope": "right upper limb and wrist chain up to the wrist boundary",
                "ownership_rule": (
                    "right_arm owns shoulder, elbow, and wrist roll/pitch/yaw joints; "
                    "these wrist joints must stay in body_wrist_upper_limb_chain and must "
                    "not be merged into right_hand."
                ),
                "mapping_refs": [
                    _reference(
                        source_name="controller_audit_source",
                        source_ref="work/recap/scripts/gr00t_controller_audit_unitree_g1.py:348-398",
                        evidence_role="controller provenance and timebase freeze the Unitree G1 rollout/controller contract",
                    ),
                    _reference(
                        source_name="policy_io_contract",
                        source_ref="agent/exchange/gr00t_policy_io.md:149-157",
                        evidence_role="right_arm belongs to the RELATIVE action family and uses the last reference state timestep",
                    ),
                    _reference(
                        source_name="wbc_joint_order_contract",
                        source_ref="agent/exchange/wbc_env_io.md:77-83",
                        evidence_role="right_arm joint order explicitly includes right_wrist_roll/pitch/yaw joints",
                    ),
                ],
            },
        },
        "dex3_finger_hand_path": {
            "bucket_name": "dex3_finger_hand_path",
            "status": "survived",
            "provenance_class": "static",
            "bucket_kind": "dex3_hand_path",
            "body_side": "right",
            "source_mapping": {
                "original_action_key": "right_hand",
                "original_state_key": "right_hand",
                "reference_state_key": None,
                "original_group_keys": ["action.right_hand", "state.right_hand"],
                "action_representation": "ABSOLUTE",
                "canonical_representation": "ABSOLUTE",
                "joint_order": list(RIGHT_HAND_JOINT_ORDER),
                "ownership_scope": "right-hand Dex3 finger and thumb actuation below the wrist boundary",
                "ownership_rule": (
                    "right_hand owns finger and thumb joints only; finger semantics stay in "
                    "dex3_finger_hand_path and must not absorb any wrist joints from right_arm."
                ),
                "mapping_refs": [
                    _reference(
                        source_name="controller_audit_source",
                        source_ref="work/recap/scripts/gr00t_controller_audit_unitree_g1.py:348-398",
                        evidence_role="controller provenance and timebase freeze the Unitree G1 rollout/controller contract",
                    ),
                    _reference(
                        source_name="policy_io_contract",
                        source_ref="agent/exchange/gr00t_policy_io.md:149-157",
                        evidence_role="right_hand belongs to the ABSOLUTE action family and therefore should not use a relative reference state",
                    ),
                    _reference(
                        source_name="wbc_joint_order_contract",
                        source_ref="agent/exchange/wbc_env_io.md:77-83",
                        evidence_role="right_hand joint order only contains finger/thumb joints, keeping the Dex3 path separate from wrist joints",
                    ),
                ],
            },
        },
    }


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


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _stage_snapshot(
    group_payload: Mapping[str, Any], *, stage_name: str
) -> dict[str, Any]:
    stage_payload = _as_mapping(group_payload.get(stage_name), field_name=stage_name)
    baseline = _as_list(
        stage_payload.get("baseline"), field_name=f"{stage_name}.baseline"
    )
    probe = _as_list(stage_payload.get("probe"), field_name=f"{stage_name}.probe")
    horizon = len(baseline)
    dimension = (
        len(_as_list(baseline[0], field_name=f"{stage_name}.baseline[0]"))
        if baseline
        else 0
    )
    return {
        "shape": {
            "policy_horizon": int(horizon),
            "dimension": int(dimension),
        },
        "baseline_first_timestep": baseline[0] if baseline else [],
        "probe_first_timestep": probe[0] if probe else [],
    }


def _difference_metrics(group_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(
        group_payload.get("difference_metrics"),
        field_name="group_payload.difference_metrics",
    )


def _validate_bucket_specs(
    bucket_specs: Mapping[str, Mapping[str, Any]],
    *,
    controller_audit_payload: Mapping[str, Any],
) -> None:
    actual_bucket_names = list(bucket_specs.keys())
    if actual_bucket_names != list(WATCH_BUCKET_ORDER):
        raise ValueError(
            "watch buckets must preserve order "
            f"{list(WATCH_BUCKET_ORDER)}, got {actual_bucket_names}"
        )

    action_repr_by_key = {
        str(key): _as_string(
            value, field_name=f"action_representation_by_key.{key}"
        ).upper()
        for key, value in _as_mapping(
            controller_audit_payload.get("action_representation_by_key"),
            field_name="controller_audit.action_representation_by_key",
        ).items()
    }
    action_dims_expected = {
        str(key): int(_as_number(value, field_name=f"action_dims_expected.{key}"))
        for key, value in _as_mapping(
            controller_audit_payload.get("action_dims_expected"),
            field_name="controller_audit.action_dims_expected",
        ).items()
    }
    relative_action_keys = set(
        _as_string_list(
            controller_audit_payload.get("relative_action_keys"),
            field_name="controller_audit.relative_action_keys",
        )
    )
    absolute_action_keys = set(
        _as_string_list(
            controller_audit_payload.get("absolute_action_keys"),
            field_name="controller_audit.absolute_action_keys",
        )
    )
    reference_state_keys = {
        str(key): _as_string(value, field_name=f"reference_state_keys.{key}")
        for key, value in _as_mapping(
            _as_mapping(
                controller_audit_payload.get("relative_to_absolute_processor"),
                field_name="controller_audit.relative_to_absolute_processor",
            ).get("reference_state_keys"),
            field_name="controller_audit.relative_to_absolute_processor.reference_state_keys",
        ).items()
    }

    expected_action_keys = {
        "body_wrist_upper_limb_chain": "right_arm",
        "dex3_finger_hand_path": "right_hand",
    }
    expected_representations = {
        "body_wrist_upper_limb_chain": "RELATIVE",
        "dex3_finger_hand_path": "ABSOLUTE",
    }
    expected_joint_orders = {
        "body_wrist_upper_limb_chain": list(RIGHT_ARM_JOINT_ORDER),
        "dex3_finger_hand_path": list(RIGHT_HAND_JOINT_ORDER),
    }

    seen_action_keys: set[str] = set()
    for bucket_name in WATCH_BUCKET_ORDER:
        spec = _as_mapping(bucket_specs[bucket_name], field_name=bucket_name)
        source_mapping = _as_mapping(
            spec.get("source_mapping"), field_name=f"{bucket_name}.source_mapping"
        )
        action_key = _as_string(
            source_mapping.get("original_action_key"),
            field_name=f"{bucket_name}.source_mapping.original_action_key",
        )
        source_state_key = _as_string(
            source_mapping.get("original_state_key"),
            field_name=f"{bucket_name}.source_mapping.original_state_key",
        )
        reference_state_key = _as_optional_string(
            source_mapping.get("reference_state_key"),
            field_name=f"{bucket_name}.source_mapping.reference_state_key",
        )
        action_representation = _as_string(
            source_mapping.get("action_representation"),
            field_name=f"{bucket_name}.source_mapping.action_representation",
        ).upper()
        joint_order = _as_string_list(
            source_mapping.get("joint_order"),
            field_name=f"{bucket_name}.source_mapping.joint_order",
        )

        if action_key in seen_action_keys:
            raise ValueError(
                "watch buckets must stay split: multiple buckets cannot consume the same "
                f"original action key {action_key!r}"
            )
        seen_action_keys.add(action_key)

        expected_action_key = expected_action_keys[bucket_name]
        if action_key != expected_action_key:
            raise ValueError(
                f"{bucket_name} must stay bound to {expected_action_key}, got {action_key}"
            )
        if source_state_key != expected_action_key:
            raise ValueError(
                f"{bucket_name} must track state.{expected_action_key}, got state.{source_state_key}"
            )

        expected_representation = expected_representations[bucket_name]
        if action_representation != expected_representation:
            raise ValueError(
                f"relative/absolute semantics drift for {bucket_name}: expected "
                f"{expected_representation}, got {action_representation}"
            )

        audit_representation = action_repr_by_key.get(action_key)
        if audit_representation != action_representation:
            raise ValueError(
                f"relative/absolute semantics mismatch for {bucket_name}: controller audit "
                f"says {action_key} is {audit_representation}, spec says {action_representation}"
            )

        expected_joint_order = expected_joint_orders[bucket_name]
        if joint_order != expected_joint_order:
            raise ValueError(
                f"joint order drift for {bucket_name}: expected {expected_joint_order}, "
                f"got {joint_order}"
            )

        expected_dim = action_dims_expected[action_key]
        if len(joint_order) != expected_dim:
            raise ValueError(
                f"joint order dimension mismatch for {bucket_name}: expected {expected_dim}, "
                f"got {len(joint_order)}"
            )

        if bucket_name == "body_wrist_upper_limb_chain":
            if action_key not in relative_action_keys:
                raise ValueError(
                    f"{bucket_name} expects a relative action key, but {action_key} is not in relative_action_keys"
                )
            if reference_state_key != reference_state_keys.get(action_key):
                raise ValueError(
                    f"relative/absolute reference-state mismatch for {bucket_name}: expected "
                    f"{reference_state_keys.get(action_key)!r}, got {reference_state_key!r}"
                )
            if not any("wrist" in joint for joint in joint_order):
                raise ValueError(
                    f"joint order drift for {bucket_name}: right wrist joints disappeared from {joint_order}"
                )
        else:
            if action_key not in absolute_action_keys:
                raise ValueError(
                    f"{bucket_name} expects an absolute action key, but {action_key} is not in absolute_action_keys"
                )
            if reference_state_key is not None:
                raise ValueError(
                    f"{bucket_name} must not carry a relative reference_state_key, got {reference_state_key!r}"
                )
            if any("wrist" in joint for joint in joint_order):
                raise ValueError(
                    f"split-bucket mistake for {bucket_name}: wrist joints leaked into Dex3 hand ownership"
                )
            if not all(joint.startswith("right_hand_") for joint in joint_order):
                raise ValueError(
                    f"joint order drift for {bucket_name}: expected right_hand_* joints, got {joint_order}"
                )


def _bucket_watch_classification(
    *,
    controller_absorbed_upstream_difference: bool,
    model_insensitive: bool,
) -> str:
    if model_insensitive:
        return "model_insensitive"
    if controller_absorbed_upstream_difference:
        return "controller_absorbed_upstream_difference"
    return "live_difference_persists"


def _checkpoint_entry(
    *,
    checkpoint_name: str,
    stage_name: str,
    boundary_name: str,
    status: str,
    provenance_class: str,
    space_name: str,
    representation: str,
    l2_difference: float,
    snapshot: Mapping[str, Any],
    notes: Sequence[str],
    transform_assumption: str,
) -> dict[str, Any]:
    return {
        "checkpoint_name": checkpoint_name,
        "stage_name": stage_name,
        "boundary_name": boundary_name,
        "status": status,
        "provenance_class": provenance_class,
        "space_name": space_name,
        "representation": representation,
        "transform_assumption": transform_assumption,
        "l2_difference": float(l2_difference),
        "shape": dict(
            _as_mapping(snapshot.get("shape"), field_name=f"{checkpoint_name}.shape")
        ),
        "baseline_first_timestep": list(
            _as_list(
                snapshot.get("baseline_first_timestep"),
                field_name=f"{checkpoint_name}.baseline_first_timestep",
            )
        ),
        "probe_first_timestep": list(
            _as_list(
                snapshot.get("probe_first_timestep"),
                field_name=f"{checkpoint_name}.probe_first_timestep",
            )
        ),
        "notes": list(notes),
    }


def _build_bucket_payload(
    *,
    bucket_name: str,
    bucket_spec: Mapping[str, Any],
    telemetry_payload: Mapping[str, Any],
) -> dict[str, Any]:
    source_mapping = _as_mapping(
        bucket_spec.get("source_mapping"), field_name=f"{bucket_name}.source_mapping"
    )
    action_key = _as_string(
        source_mapping.get("original_action_key"),
        field_name=f"{bucket_name}.source_mapping.original_action_key",
    )
    per_group_stats = _as_mapping(
        telemetry_payload.get("per_group_stats"),
        field_name="action_telemetry.per_group_stats",
    )
    group_payload = _as_mapping(
        per_group_stats.get(action_key),
        field_name=f"action_telemetry.per_group_stats.{action_key}",
    )
    diff_metrics = _difference_metrics(group_payload)
    action_representation = _as_string(
        source_mapping.get("action_representation"),
        field_name=f"{bucket_name}.source_mapping.action_representation",
    ).upper()
    canonical_representation = _as_string(
        source_mapping.get("canonical_representation"),
        field_name=f"{bucket_name}.source_mapping.canonical_representation",
    ).upper()

    raw_snapshot = _stage_snapshot(group_payload, stage_name="raw_action")
    absolute_snapshot = _stage_snapshot(group_payload, stage_name="absolute_action")
    controller_snapshot = _stage_snapshot(group_payload, stage_name="controller_input")

    consumed_difference_disappeared_at = _as_optional_string(
        diff_metrics.get("difference_disappeared_at"),
        field_name=f"{bucket_name}.difference_disappeared_at",
    )
    controller_absorbed = _as_bool(
        diff_metrics.get("controller_absorbed_upstream_difference"),
        field_name=f"{bucket_name}.controller_absorbed_upstream_difference",
    )
    model_insensitive = _as_bool(
        diff_metrics.get("model_insensitive"),
        field_name=f"{bucket_name}.model_insensitive",
    )

    if action_representation == "RELATIVE":
        adapted_status = "mutated"
        adapted_transform = "relative_action_adds_last_reference_state"
        adapted_notes = [
            "This bucket becomes canonical by adding decoded right_arm deltas to the last state.right_arm timestep.",
            "The canonical absolute space keeps wrist joints inside right_arm rather than collapsing them into right_hand.",
        ]
    else:
        adapted_status = "survived"
        adapted_transform = "absolute_action_identity_passthrough"
        adapted_notes = [
            "This bucket is already absolute before the canonical comparison stage.",
            "The canonical absolute space keeps finger/thumb joints below the wrist boundary and does not borrow right_arm wrist joints.",
        ]

    checkpoints = {
        "produced": _checkpoint_entry(
            checkpoint_name="produced",
            stage_name="raw_action",
            boundary_name="policy_output_action",
            status="survived",
            provenance_class="synthetic",
            space_name="policy_output_raw_action_space",
            representation=action_representation,
            l2_difference=_as_number(
                diff_metrics.get("raw_action_l2"),
                field_name=f"{bucket_name}.raw_action_l2",
            ),
            snapshot=raw_snapshot,
            notes=[
                f"The original watch bucket is produced at action.{action_key}.",
                "This checkpoint preserves the model-emitted group before any canonical absolute adaptation.",
            ],
            transform_assumption="raw_model_output_before_decode",
        ),
        "adapted": _checkpoint_entry(
            checkpoint_name="adapted",
            stage_name="absolute_action",
            boundary_name="action_semantics_adapter",
            status=adapted_status,
            provenance_class="synthetic",
            space_name=CANONICAL_SPACE_NAME,
            representation=canonical_representation,
            l2_difference=_as_number(
                diff_metrics.get("absolute_action_l2"),
                field_name=f"{bucket_name}.absolute_action_l2",
            ),
            snapshot=absolute_snapshot,
            notes=adapted_notes,
            transform_assumption=adapted_transform,
        ),
        "consumed": _checkpoint_entry(
            checkpoint_name="consumed",
            stage_name="controller_input",
            boundary_name=bucket_name,
            status="survived",
            provenance_class="synthetic",
            space_name="controller_input_diagnostic_limit_space",
            representation=canonical_representation,
            l2_difference=_as_number(
                diff_metrics.get("controller_input_l2"),
                field_name=f"{bucket_name}.controller_input_l2",
            ),
            snapshot=controller_snapshot,
            notes=[
                "This checkpoint shows what the WBC controller finally consumes after diagnostic clipping.",
                "difference_disappeared_at/controller_absorbed_upstream_difference/model_insensitive are copied from the action-chain telemetry contract for this exact action key.",
            ],
            transform_assumption="controller_input_after_diagnostic_limits",
        ),
    }

    conclusions = {
        "difference_disappeared_at": consumed_difference_disappeared_at,
        "controller_absorbed_upstream_difference": controller_absorbed,
        "model_insensitive": model_insensitive,
        "watch_bucket_classification": _bucket_watch_classification(
            controller_absorbed_upstream_difference=controller_absorbed,
            model_insensitive=model_insensitive,
        ),
        "explanation": (
            "The body/wrist bucket still carries an upstream difference until controller_input."
            if controller_absorbed
            else "The Dex3 finger-hand bucket shows no upstream delta at model output, so the bucket is model-insensitive in this probe."
            if model_insensitive
            else "The bucket retains a live difference through the controller input stage."
        ),
    }

    return {
        "bucket_name": _as_string(
            bucket_spec.get("bucket_name"), field_name=f"{bucket_name}.bucket_name"
        ),
        "status": _as_string(
            bucket_spec.get("status"), field_name=f"{bucket_name}.status"
        ),
        "provenance_class": _as_string(
            bucket_spec.get("provenance_class"),
            field_name=f"{bucket_name}.provenance_class",
        ),
        "bucket_kind": _as_string(
            bucket_spec.get("bucket_kind"), field_name=f"{bucket_name}.bucket_kind"
        ),
        "body_side": _as_string(
            bucket_spec.get("body_side"), field_name=f"{bucket_name}.body_side"
        ),
        "source_mapping": dict(source_mapping),
        "checkpoints": checkpoints,
        "conclusions": conclusions,
    }


def build_action_semantics_roundtrip(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    controller_audit_payload: Mapping[str, Any] | None = None,
    action_telemetry_payload: Mapping[str, Any] | None = None,
    watch_bucket_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    contract_payload = _load_contract_payload()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR).resolve()
    )
    generation_command = _generation_command_for(resolved_output_dir, repo_root)

    controller_payload = (
        dict(controller_audit_payload)
        if controller_audit_payload is not None
        else _read_json(_controller_audit_path(repo_root))
    )
    telemetry_payload = (
        dict(action_telemetry_payload)
        if action_telemetry_payload is not None
        else _read_json(_action_telemetry_path(repo_root))
    )
    bucket_specs = (
        copy.deepcopy(default_watch_bucket_specs())
        if watch_bucket_specs is None
        else copy.deepcopy(dict(watch_bucket_specs))
    )

    _validate_bucket_specs(bucket_specs, controller_audit_payload=controller_payload)

    watch_buckets: dict[str, Any] = {}
    controller_absorbed_watch_buckets: list[str] = []
    model_insensitive_watch_buckets: list[str] = []
    difference_disappeared_at_by_bucket: dict[str, Any] = {}
    watch_bucket_classification_by_bucket: dict[str, str] = {}
    for bucket_name in WATCH_BUCKET_ORDER:
        bucket_payload = _build_bucket_payload(
            bucket_name=bucket_name,
            bucket_spec=_as_mapping(bucket_specs[bucket_name], field_name=bucket_name),
            telemetry_payload=telemetry_payload,
        )
        watch_buckets[bucket_name] = bucket_payload
        conclusions = _as_mapping(
            bucket_payload.get("conclusions"),
            field_name=f"watch_buckets.{bucket_name}.conclusions",
        )
        if _as_bool(
            conclusions.get("controller_absorbed_upstream_difference"),
            field_name=f"watch_buckets.{bucket_name}.controller_absorbed_upstream_difference",
        ):
            controller_absorbed_watch_buckets.append(bucket_name)
        if _as_bool(
            conclusions.get("model_insensitive"),
            field_name=f"watch_buckets.{bucket_name}.model_insensitive",
        ):
            model_insensitive_watch_buckets.append(bucket_name)
        watch_bucket_classification_by_bucket[bucket_name] = _as_string(
            conclusions.get("watch_bucket_classification"),
            field_name=f"watch_buckets.{bucket_name}.watch_bucket_classification",
        )
        difference_disappeared_at_by_bucket[bucket_name] = conclusions.get(
            "difference_disappeared_at"
        )

    canonical_space = {
        "canonical_space_name": CANONICAL_SPACE_NAME,
        "status": "survived",
        "provenance_class": "static",
        "comparison_stage": "absolute_action",
        "transform_assumption_name": CANONICAL_TRANSFORM_NAME,
        "space_summary": (
            "Canonical comparison happens in absolute joint-order space after decode and relative-to-absolute adaptation, then controller_input is treated as the downstream consumed checkpoint."
        ),
        "stage_chain": [
            "raw_action",
            "decoded_action",
            "absolute_action",
            "controller_input",
        ],
        "relative_action_interpretation": {
            "representation_owner_keys": ["left_arm", "right_arm"],
            "reference_state_timestep": "last",
            "reference_state_keys": {
                str(key): value
                for key, value in _as_mapping(
                    _as_mapping(
                        controller_payload.get("relative_to_absolute_processor"),
                        field_name="controller_audit.relative_to_absolute_processor",
                    ).get("reference_state_keys"),
                    field_name="controller_audit.relative_to_absolute_processor.reference_state_keys",
                ).items()
            },
            "interpretation_rule": (
                "Relative decoded joint deltas are added to the last timestep of the matching state group before cross-bucket comparison."
            ),
            "source_refs": [
                "agent/exchange/gr00t_policy_io.md:149-157",
                "work/recap/scripts/gr00t_action_chain_telemetry.py:223-287",
            ],
        },
        "absolute_action_interpretation": {
            "representation_owner_keys": _as_string_list(
                controller_payload.get("absolute_action_keys"),
                field_name="controller_audit.absolute_action_keys",
            ),
            "interpretation_rule": (
                "Absolute decoded actions already live in canonical semantics, so absolute_action is an identity semantic stage for those groups."
            ),
            "source_refs": [
                "agent/exchange/gr00t_policy_io.md:149-157",
                "work/recap/scripts/gr00t_action_chain_telemetry.py:223-287",
            ],
        },
        "wrist_vs_finger_ownership": {
            "body_wrist_upper_limb_chain": {
                "action_key": "right_arm",
                "joint_order": list(RIGHT_ARM_JOINT_ORDER),
                "ownership_summary": (
                    "Wrist roll/pitch/yaw stay inside right_arm and therefore belong to body_wrist_upper_limb_chain."
                ),
            },
            "dex3_finger_hand_path": {
                "action_key": "right_hand",
                "joint_order": list(RIGHT_HAND_JOINT_ORDER),
                "ownership_summary": (
                    "Finger/thumb joints stay inside right_hand and therefore belong to dex3_finger_hand_path."
                ),
            },
            "must_not_merge": True,
            "source_refs": [
                "agent/exchange/wbc_env_io.md:77-83",
                "tests/recap/test_gr00t_action_chain_telemetry.py:81-142",
            ],
        },
    }

    return {
        "schema_version": ACTION_ROUNDTRIP_SCHEMA_VERSION,
        "artifact_kind": ACTION_ROUNDTRIP_ARTIFACT_KIND,
        "provenance_class": "static",
        "generation_command": generation_command,
        "input_baseline_summary": _baseline_summary(contract_payload),
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_action_roundtrip.py",
            "task1_contract_writer": "work/recap/scripts/interface_localization_contract.py",
            "expected_task1_contract_json": str(
                resolved_output_dir / interface_localization_contract.CONTRACT_JSON_NAME
            ),
            "expected_task2_inventory_json": str(
                resolved_output_dir / "replay_surface_inventory.json"
            ),
            "expected_task3_text_rewrite_map_json": str(
                resolved_output_dir / "recap_text_source_and_rewrite_map.json"
            ),
            "expected_controller_audit_json": str(_controller_audit_path(repo_root)),
            "expected_action_chain_telemetry_json": str(
                _action_telemetry_path(repo_root)
            ),
            "pytest_command": "python3 -m pytest tests/recap/test_action_semantics_roundtrip.py -q",
        },
        "source_refs": {
            "telemetry_contract_script": "work/recap/scripts/gr00t_action_chain_telemetry.py:23-29,56-96,223-287",
            "controller_audit_script": "work/recap/scripts/gr00t_controller_audit_unitree_g1.py:348-398",
            "policy_io_contract": "agent/exchange/gr00t_policy_io.md:149-157",
            "wbc_env_contract": "agent/exchange/wbc_env_io.md:77-83",
            "telemetry_term_tests": "tests/recap/test_gr00t_action_chain_telemetry.py:81-142",
        },
        "canonical_space": canonical_space,
        "watch_bucket_order": list(WATCH_BUCKET_ORDER),
        "watch_buckets": watch_buckets,
        "summary": {
            "controller_absorbed_watch_buckets": controller_absorbed_watch_buckets,
            "model_insensitive_watch_buckets": model_insensitive_watch_buckets,
            "difference_disappeared_at_by_bucket": difference_disappeared_at_by_bucket,
            "watch_bucket_classification_by_bucket": watch_bucket_classification_by_bucket,
            "explicit_split": {
                "body_wrist_upper_limb_chain": "right_arm",
                "dex3_finger_hand_path": "right_hand",
                "must_not_collapse_back_to_right_hand": True,
            },
        },
    }


def write_artifact(*, output_dir: Path, payload: Mapping[str, Any]) -> Path:
    return state_conditioned_bucket_a_import._write_json(
        output_dir / ACTION_ROUNDTRIP_JSON_NAME,
        payload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output_dir = resolve_output_dir(REPO_ROOT, args)
        payload = build_action_semantics_roundtrip(REPO_ROOT, output_dir=output_dir)
        artifact_path = write_artifact(output_dir=output_dir, payload=payload)
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "output_dir": str(output_dir),
                    "action_roundtrip_json": str(artifact_path),
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
