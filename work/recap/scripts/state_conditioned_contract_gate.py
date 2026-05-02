from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_materialization/freeze")
DEFAULT_STABLE_BASE_CHECKPOINT_KIND = "model_path"
DEFAULT_STABLE_BASE_CHECKPOINT_VALUE = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_BASELINE_DATASET_KIND = "dataset_dir"

FREEZE_JSON_NAME = "state_conditioned_freeze.json"
CONTRACT_GATE_REPORT_JSON_NAME = "contract_gate_report.json"
PHASE_MODE_FSM_JSON_NAME = "phase_mode_fsm.json"

FREEZE_SCHEMA_VERSION = "g1_state_conditioned_contract_freeze_v2"
FSM_SCHEMA_VERSION = "g1_state_conditioned_phase_mode_fsm_v2"
REPORT_SCHEMA_VERSION = "g1_state_conditioned_contract_gate_report_v2"
CONTRACT_GATE_NAME = "ContractGate"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import text_indicator
from work.recap.lerobot_export import dataset_export as lerobot_v2_export


PHASE_VOCAB: tuple[str, ...] = tuple(
    state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
)
MODE_VOCAB: tuple[str, ...] = tuple(
    state_conditioned_bucket_a_import.STATE_CONDITIONED_MODES
)
HISTORY_K = int(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K)
HISTORY_STRIDE = int(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_STRIDE)
RESET_BOUNDARY = str(state_conditioned_bucket_a_import.STATE_CONDITIONED_RESET_BOUNDARY)

POLICY_TEXT_ALLOWLIST: tuple[str, ...] = ("phase", "mode")
POLICY_CONDITION_TEXT_TEMPLATE = "[PolicyCondition-v1]\nPHASE={phase}\nMODE={mode}"
EXPERIMENT_SPLIT_ALLOWLIST: tuple[str, ...] = ("devtrain", "devbench")
TEACHER_PROVENANCE_FIELD_NAMES: tuple[str, ...] = (
    "teacher_producer",
    "teacher_version",
    "teacher_trigger_reason",
    "teacher_trigger_success_rate",
    "teacher_trigger_threshold",
)

DEPLOYABLE_HISTORY_ALLOWLIST: tuple[str, ...] = (
    "history_k",
    "history_stride",
    "history_valid_mask",
    "history_t_std_indices",
    "history_t_raw_indices",
    "history_timestamp_s",
    "anchor_mujoco_state_ref",
    "prehistory_window",
    "deployable.previous_action_history",
    "deployable.proprio_history",
    "deployable.short_visual_history_refs",
)
DEPLOYABLE_OBSERVATION_ALLOWLIST: tuple[str, ...] = (
    "policy_condition.phase",
    "policy_condition.mode",
    *DEPLOYABLE_HISTORY_ALLOWLIST,
)
ANALYSIS_ONLY_FIELDS: tuple[str, ...] = tuple(
    lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_FIELD_NAMES
)
DEPLOYABLE_DENYLIST_PREFIXES: tuple[str, ...] = (
    "privileged.",
    *tuple(lerobot_v2_export.TEACHER_ONLY_FIELD_PREFIXES),
)
DEPLOYABLE_DENYLIST_EXACT_NAMES: tuple[str, ...] = (
    *tuple(lerobot_v2_export.DEPLOYABLE_FIELD_LEAKAGE_EXACT_NAMES),
    *TEACHER_PROVENANCE_FIELD_NAMES,
)
HISTORY_VALIDATION_ONLY_FIELDS: tuple[str, ...] = (
    "anchor_episode_id",
    "history_episode_ids",
    "reset_boundary",
)
MAINLINE_TRAINING_TEXT_FIELD = text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
POLICY_METADATA_FIELD_NAMES: tuple[str, ...] = (
    "policy_condition.phase",
    "policy_condition.mode",
    "policy_condition_text",
)
LEGACY_NON_AUTHORITY_FIELD_NAMES: tuple[str, ...] = (
    "prompt_conditioned",
    "advantage_input",
    "dual_task_text",
)
LEGACY_TEXT_NON_AUTHORITY_FIELD_NAMES: tuple[str, ...] = (
    "prompt_conditioned",
    "dual_task_text",
)
DIAGNOSTIC_ONLY_FIELD_NAMES: tuple[str, ...] = ("advantage_input",)

PHASE_RULES: dict[str, str] = {
    "SEARCH": "apple not visible",
    "APPROACH": "apple visible and hand_to_apple_distance_m > 0.08",
    "GRASP": "apple visible and hand_to_apple_distance_m <= 0.08 and not in_hand",
    "VERIFY_HOLD": "first 8 contiguous policy steps after in_hand becomes true",
    "TRANSPORT": "contiguous_in_hand_policy_steps >= 8 and apple_to_plate_xy_distance_m > 0.12",
    "PLACE": "in_hand and apple_to_plate_xy_distance_m <= 0.12",
}
ALLOWED_PHASE_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "SEARCH": ("SEARCH", "APPROACH"),
    "APPROACH": ("SEARCH", "APPROACH", "GRASP"),
    "GRASP": ("SEARCH", "APPROACH", "GRASP", "VERIFY_HOLD"),
    "VERIFY_HOLD": ("SEARCH", "GRASP", "VERIFY_HOLD", "TRANSPORT"),
    "TRANSPORT": (
        "SEARCH",
        "APPROACH",
        "GRASP",
        "VERIFY_HOLD",
        "TRANSPORT",
        "PLACE",
    ),
    "PLACE": (
        "SEARCH",
        "APPROACH",
        "GRASP",
        "VERIFY_HOLD",
        "TRANSPORT",
        "PLACE",
    ),
}
MODE_RULES: dict[str, dict[str, object]] = {
    "NOMINAL": {
        "entry_condition": "default mode while recovery is inactive",
        "can_transition_to": ["NOMINAL", "RECOVERY"],
    },
    "RECOVERY": {
        "entry_condition": "family-triggered off-nominal or teacher fallback",
        "exit_condition": "family-specific local success satisfied",
        "can_transition_to": ["RECOVERY", "NOMINAL"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the Task 1 state-conditioned contract gate and materialize "
            "machine-readable JSON artifacts for the frozen history-aware schema."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives the frozen contract JSON artifacts.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


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


def _as_list(
    value: object,
    *,
    field_name: str,
    expected_len: int | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    result = list(value)
    if expected_len is not None and len(result) != int(expected_len):
        raise ValueError(
            f"{field_name} must have length {expected_len}, got {len(result)}"
        )
    return result


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _normalize_phase(value: object, *, field_name: str) -> str:
    normalized = _as_non_empty_string(value, field_name=field_name).upper()
    if normalized not in PHASE_VOCAB:
        raise ValueError(f"{field_name} must be one of {PHASE_VOCAB!r}")
    return normalized


def _normalize_mode(value: object, *, field_name: str) -> str:
    normalized = _as_non_empty_string(value, field_name=field_name).upper()
    if normalized not in MODE_VOCAB:
        raise ValueError(f"{field_name} must be one of {MODE_VOCAB!r}")
    return normalized


def build_policy_condition_text(phase: object, mode: object) -> str:
    normalized_phase = _normalize_phase(phase, field_name="phase")
    normalized_mode = _normalize_mode(mode, field_name="mode")
    return state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
        normalized_phase,
        normalized_mode,
    )


def validate_policy_condition_text(
    *,
    phase: object,
    mode: object,
    policy_condition_text: object,
) -> dict[str, Any]:
    normalized_phase = _normalize_phase(
        phase, field_name="deployable_observation.policy_condition.phase"
    )
    normalized_mode = _normalize_mode(
        mode, field_name="deployable_observation.policy_condition.mode"
    )
    normalized_text = _as_non_empty_string(
        policy_condition_text,
        field_name="policy_condition_text",
    )
    expected_text = build_policy_condition_text(normalized_phase, normalized_mode)
    if normalized_text != expected_text:
        raise ValueError(
            "policy_condition_text must match the canonical [phase,mode]-only template"
        )
    return {
        "phase": normalized_phase,
        "mode": normalized_mode,
        "policy_condition_text": normalized_text,
        "authority_status": "metadata_only",
        "mainline_authority": False,
        "diagnostic_only": False,
        "metadata_fields": list(POLICY_METADATA_FIELD_NAMES),
        "allowlisted_tokens": list(POLICY_TEXT_ALLOWLIST),
    }


def build_carrier_text_v1(prompt_raw: object, indicator_I: object) -> str:
    indicator_mode = text_indicator.indicator_mode_from_indicator_value(
        indicator_I,
        field_name="indicator_I",
    )
    return text_indicator.build_canonical_text_indicator(prompt_raw, indicator_mode)


def validate_authoritative_training_text(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(_as_mapping(candidate, field_name="candidate"))
    prompt_raw = text_indicator.require_prompt_raw(
        payload.get("prompt_raw"),
        field_name="prompt_raw",
    )
    indicator_mode = text_indicator.indicator_mode_from_indicator_value(
        payload.get("indicator_I"),
        field_name="indicator_I",
    )
    carrier_text_v1 = _as_non_empty_string(
        payload.get(MAINLINE_TRAINING_TEXT_FIELD),
        field_name=MAINLINE_TRAINING_TEXT_FIELD,
    )
    expected = build_carrier_text_v1(prompt_raw, payload.get("indicator_I"))
    if carrier_text_v1 != expected:
        raise ValueError(
            "carrier_text_v1 must match the canonical prompt_raw + indicator_I text-indicator carrier"
        )
    prompt_conditioned = payload.get("prompt_conditioned")
    if isinstance(prompt_conditioned, str) and prompt_conditioned.strip():
        if prompt_conditioned != carrier_text_v1:
            raise ValueError(
                "prompt_conditioned must not override carrier_text_v1 authority"
            )
    if payload.get("advantage_input") is not None:
        raise ValueError(
            "advantage_input must remain out of the mainline carrier_text_v1 authority channel"
        )
    if bool(payload.get("dual_task_text")):
        raise ValueError(
            "dual_task_text authority must remain disabled for the carrier_text_v1 mainline"
        )
    return {
        "carrier_field": MAINLINE_TRAINING_TEXT_FIELD,
        "authority_name": text_indicator.RECAP_TEXT_INDICATOR_AUTHORITY_NAME,
        "schema_version": text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION,
        "authority_status": "mainline_authority",
        "mainline_authority": True,
        "diagnostic_only": False,
        "indicator_mode": indicator_mode,
        "carrier_text_v1": carrier_text_v1,
        "prompt_raw": prompt_raw,
        "policy_metadata_fields": list(POLICY_METADATA_FIELD_NAMES),
        "legacy_non_authority_fields": list(LEGACY_NON_AUTHORITY_FIELD_NAMES),
        "legacy_text_non_authority_fields": list(LEGACY_TEXT_NON_AUTHORITY_FIELD_NAMES),
        "diagnostic_only_fields": list(DIAGNOSTIC_ONLY_FIELD_NAMES),
    }


def _field_leakage_reason(field_name: str) -> str | None:
    if field_name in ANALYSIS_ONLY_FIELDS:
        return f"analysis-only field leaked into deployable observation: {field_name!r}"
    if field_name in DEPLOYABLE_DENYLIST_EXACT_NAMES:
        return f"forbidden field leaked into deployable observation: {field_name!r}"
    for prefix in DEPLOYABLE_DENYLIST_PREFIXES:
        if field_name.startswith(prefix):
            return (
                f"forbidden prefix leaked into deployable observation: {field_name!r}"
            )
    return None


def validate_deployable_observation(
    deployable_observation: Mapping[str, Any],
) -> dict[str, object]:
    observation = dict(
        _as_mapping(deployable_observation, field_name="deployable_observation")
    )
    observed_keys = sorted(observation.keys())
    required_keys = set(DEPLOYABLE_OBSERVATION_ALLOWLIST)
    missing_keys = sorted(required_keys - set(observed_keys))
    if missing_keys:
        raise ValueError(
            "deployable observation is missing frozen fields: "
            + ", ".join(missing_keys)
        )

    extra_keys = sorted(set(observed_keys) - required_keys)
    if extra_keys:
        extra = extra_keys[0]
        leakage_reason = _field_leakage_reason(extra)
        if leakage_reason is not None:
            raise ValueError(leakage_reason)
        raise ValueError(f"deployable observation field not allowlisted: {extra!r}")

    for field_name in observed_keys:
        leakage_reason = _field_leakage_reason(field_name)
        if leakage_reason is not None:
            raise ValueError(leakage_reason)

    phase = _normalize_phase(
        observation["policy_condition.phase"],
        field_name="deployable_observation.policy_condition.phase",
    )
    mode = _normalize_mode(
        observation["policy_condition.mode"],
        field_name="deployable_observation.policy_condition.mode",
    )
    if (
        _as_int(observation["history_k"], field_name="deployable_observation.history_k")
        != HISTORY_K
    ):
        raise ValueError(f"history_k is frozen at {HISTORY_K}")
    if (
        _as_int(
            observation["history_stride"],
            field_name="deployable_observation.history_stride",
        )
        != HISTORY_STRIDE
    ):
        raise ValueError(f"history_stride is frozen at {HISTORY_STRIDE}")

    _as_list(
        observation["history_valid_mask"],
        field_name="deployable_observation.history_valid_mask",
        expected_len=HISTORY_K,
    )
    _as_list(
        observation["history_t_std_indices"],
        field_name="deployable_observation.history_t_std_indices",
        expected_len=HISTORY_K,
    )
    _as_list(
        observation["history_t_raw_indices"],
        field_name="deployable_observation.history_t_raw_indices",
        expected_len=HISTORY_K,
    )
    _as_list(
        observation["history_timestamp_s"],
        field_name="deployable_observation.history_timestamp_s",
        expected_len=HISTORY_K,
    )
    _as_non_empty_string(
        observation["anchor_mujoco_state_ref"],
        field_name="deployable_observation.anchor_mujoco_state_ref",
    )
    _as_list(
        observation["prehistory_window"],
        field_name="deployable_observation.prehistory_window",
        expected_len=HISTORY_K,
    )
    _as_list(
        observation["deployable.previous_action_history"],
        field_name="deployable_observation.deployable.previous_action_history",
        expected_len=HISTORY_K,
    )
    _as_list(
        observation["deployable.proprio_history"],
        field_name="deployable_observation.deployable.proprio_history",
        expected_len=HISTORY_K,
    )
    _as_list(
        observation["deployable.short_visual_history_refs"],
        field_name="deployable_observation.deployable.short_visual_history_refs",
        expected_len=HISTORY_K,
    )
    return {
        "phase": phase,
        "mode": mode,
        "observed_field_count": len(observed_keys),
    }


def _parse_anchor_step(anchor_ref: str) -> int:
    last_component = anchor_ref.rsplit("/", 1)[-1]
    try:
        return int(last_component)
    except ValueError as exc:
        raise ValueError(
            "anchor_mujoco_state_ref must end with an integer step index"
        ) from exc


def validate_history_contract(
    *,
    deployable_observation: Mapping[str, Any],
    history_context: Mapping[str, Any],
) -> dict[str, object]:
    observation = dict(
        _as_mapping(deployable_observation, field_name="deployable_observation")
    )
    context = dict(_as_mapping(history_context, field_name="history_context"))

    anchor_episode_id = _as_non_empty_string(
        context.get("anchor_episode_id"),
        field_name="history_context.anchor_episode_id",
    )
    history_episode_ids = _as_list(
        context.get("history_episode_ids"),
        field_name="history_context.history_episode_ids",
        expected_len=HISTORY_K,
    )
    reset_boundary = _as_non_empty_string(
        context.get("reset_boundary"),
        field_name="history_context.reset_boundary",
    )

    try:
        base_result = state_conditioned_bucket_a_import.validate_state_conditioned_history_contract(
            anchor_episode_id=anchor_episode_id,
            history_episode_ids=history_episode_ids,
            history_valid_mask=observation.get("history_valid_mask"),
            anchor_mujoco_state_ref=observation.get("anchor_mujoco_state_ref"),
            prehistory_window=observation.get("prehistory_window"),
            history_k=observation.get("history_k"),
            history_stride=observation.get("history_stride"),
            reset_boundary=reset_boundary,
        )
    except ValueError as exc:
        if "history episode mismatch" in str(exc) or "cross-episode history" in str(
            exc
        ):
            raise ValueError("cross-episode history is forbidden") from exc
        raise

    history_valid_mask = [
        _as_bool(value, field_name=f"history_valid_mask[{index}]")
        for index, value in enumerate(
            _as_list(
                observation.get("history_valid_mask"),
                field_name="deployable_observation.history_valid_mask",
                expected_len=HISTORY_K,
            )
        )
    ]
    history_t_std_indices = [
        _as_int(
            value, field_name=f"deployable_observation.history_t_std_indices[{index}]"
        )
        for index, value in enumerate(
            _as_list(
                observation.get("history_t_std_indices"),
                field_name="deployable_observation.history_t_std_indices",
                expected_len=HISTORY_K,
            )
        )
    ]
    history_t_raw_indices = [
        _as_int(
            value, field_name=f"deployable_observation.history_t_raw_indices[{index}]"
        )
        for index, value in enumerate(
            _as_list(
                observation.get("history_t_raw_indices"),
                field_name="deployable_observation.history_t_raw_indices",
                expected_len=HISTORY_K,
            )
        )
    ]
    history_timestamp_s = [
        _as_number(
            value, field_name=f"deployable_observation.history_timestamp_s[{index}]"
        )
        for index, value in enumerate(
            _as_list(
                observation.get("history_timestamp_s"),
                field_name="deployable_observation.history_timestamp_s",
                expected_len=HISTORY_K,
            )
        )
    ]
    prehistory_window = _as_list(
        observation.get("prehistory_window"),
        field_name="deployable_observation.prehistory_window",
        expected_len=HISTORY_K,
    )
    anchor_step = _parse_anchor_step(str(base_result["anchor_mujoco_state_ref"]))

    last_valid_timestamp: float | None = None
    for index, is_valid in enumerate(history_valid_mask):
        row = dict(
            _as_mapping(
                prehistory_window[index],
                field_name=f"deployable_observation.prehistory_window[{index}]",
            )
        )
        row_t_std = _as_int(
            row.get("t_std"),
            field_name=f"deployable_observation.prehistory_window[{index}].t_std",
        )
        if not is_valid:
            continue
        if str(history_episode_ids[index]) != anchor_episode_id:
            raise ValueError("cross-episode history is forbidden at reset boundary")
        if row_t_std > anchor_step or history_t_std_indices[index] > anchor_step:
            raise ValueError("future timestamp history is forbidden")
        if history_t_raw_indices[index] > anchor_step:
            raise ValueError("future timestamp history is forbidden")
        if history_t_std_indices[index] != row_t_std:
            raise ValueError("history_t_std_indices must align with prehistory_window")
        if history_t_raw_indices[index] != row_t_std:
            raise ValueError("history_t_raw_indices must align with prehistory_window")
        if (
            last_valid_timestamp is not None
            and history_timestamp_s[index] < last_valid_timestamp
        ):
            raise ValueError(
                "history_timestamp_s must be non-decreasing for valid slots"
            )
        last_valid_timestamp = history_timestamp_s[index]

    return {
        "anchor_episode_id": anchor_episode_id,
        "anchor_step": anchor_step,
        "history_k": HISTORY_K,
        "history_stride": HISTORY_STRIDE,
        "reset_boundary": RESET_BOUNDARY,
    }


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, object]:
    values = dict(_as_mapping(metadata, field_name="metadata"))
    checkpoint_kind = _as_non_empty_string(
        values.get("stable_base_checkpoint_kind"),
        field_name="metadata.stable_base_checkpoint_kind",
    )
    checkpoint_value = _as_non_empty_string(
        values.get("stable_base_checkpoint_value"),
        field_name="metadata.stable_base_checkpoint_value",
    )
    if checkpoint_kind != DEFAULT_STABLE_BASE_CHECKPOINT_KIND:
        raise ValueError(
            "stable_base_checkpoint_kind is frozen at "
            + repr(DEFAULT_STABLE_BASE_CHECKPOINT_KIND)
        )
    if checkpoint_value != DEFAULT_STABLE_BASE_CHECKPOINT_VALUE:
        raise ValueError(
            "stable_base_checkpoint_value is frozen at "
            + repr(DEFAULT_STABLE_BASE_CHECKPOINT_VALUE)
        )

    experiment_split = _as_non_empty_string(
        values.get("experiment_split"),
        field_name="metadata.experiment_split",
    )
    if experiment_split not in EXPERIMENT_SPLIT_ALLOWLIST:
        raise ValueError(
            "metadata.experiment_split must be one of "
            + repr(EXPERIMENT_SPLIT_ALLOWLIST)
        )

    for field_name in TEACHER_PROVENANCE_FIELD_NAMES[:3]:
        _as_non_empty_string(
            values.get(field_name),
            field_name=f"metadata.{field_name}",
        )
    success_rate = _as_number(
        values.get("teacher_trigger_success_rate"),
        field_name="metadata.teacher_trigger_success_rate",
    )
    threshold = _as_number(
        values.get("teacher_trigger_threshold"),
        field_name="metadata.teacher_trigger_threshold",
    )
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("teacher_trigger_success_rate must be within [0.0, 1.0]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("teacher_trigger_threshold must be within [0.0, 1.0]")

    return {
        "experiment_split": experiment_split,
        "stable_base_checkpoint_kind": checkpoint_kind,
        "stable_base_checkpoint_value": checkpoint_value,
    }


def validate_contract_candidate(candidate: Mapping[str, Any]) -> dict[str, object]:
    payload = dict(_as_mapping(candidate, field_name="candidate"))
    deployable_observation = _as_mapping(
        payload.get("deployable_observation"),
        field_name="candidate.deployable_observation",
    )
    history_context = _as_mapping(
        payload.get("history_context"),
        field_name="candidate.history_context",
    )
    metadata = _as_mapping(
        payload.get("metadata"),
        field_name="candidate.metadata",
    )
    deployable_result = validate_deployable_observation(deployable_observation)
    carrier_result = validate_authoritative_training_text(payload)
    policy_text_result = validate_policy_condition_text(
        phase=deployable_observation["policy_condition.phase"],
        mode=deployable_observation["policy_condition.mode"],
        policy_condition_text=payload.get("policy_condition_text"),
    )
    history_result = validate_history_contract(
        deployable_observation=deployable_observation,
        history_context=history_context,
    )
    metadata_result = validate_metadata(metadata)
    return {
        "deployable_observation": deployable_result,
        "mainline_training_text": carrier_result,
        "policy_text": policy_text_result,
        "history_contract": history_result,
        "metadata": metadata_result,
    }


def build_reference_contract_example() -> dict[str, object]:
    anchor_episode_id = "episode_001"
    anchor_step = HISTORY_K - 1
    prompt_raw = "pick up the apple and place it on the plate"
    indicator_I = 1
    history_valid_mask = [True] * HISTORY_K
    prehistory_window = [
        {
            "episode_id": anchor_episode_id,
            "t_std": index,
            "mujoco_state_ref": f"mujoco://{anchor_episode_id}/{index}",
        }
        for index in range(HISTORY_K)
    ]
    deployable_observation = {
        "policy_condition.phase": "VERIFY_HOLD",
        "policy_condition.mode": "NOMINAL",
        "history_k": HISTORY_K,
        "history_stride": HISTORY_STRIDE,
        "history_valid_mask": history_valid_mask,
        "history_t_std_indices": list(range(HISTORY_K)),
        "history_t_raw_indices": list(range(HISTORY_K)),
        "history_timestamp_s": [
            round(0.05 * float(index), 3) for index in range(HISTORY_K)
        ],
        "anchor_mujoco_state_ref": f"mujoco://{anchor_episode_id}/{anchor_step}",
        "prehistory_window": prehistory_window,
        "deployable.previous_action_history": [
            [round(0.1 * float(index), 3), round(0.2 * float(index), 3)]
            for index in range(HISTORY_K)
        ],
        "deployable.proprio_history": [
            [round(0.01 * float(index), 3), round(0.02 * float(index), 3), 1.0]
            for index in range(HISTORY_K)
        ],
        "deployable.short_visual_history_refs": [
            f"video://{anchor_episode_id}/{index}" for index in range(HISTORY_K)
        ],
    }
    return {
        "deployable_observation": deployable_observation,
        "prompt_raw": prompt_raw,
        "indicator_I": indicator_I,
        MAINLINE_TRAINING_TEXT_FIELD: build_carrier_text_v1(prompt_raw, indicator_I),
        "policy_condition_text": build_policy_condition_text(
            deployable_observation["policy_condition.phase"],
            deployable_observation["policy_condition.mode"],
        ),
        "history_context": {
            "anchor_episode_id": anchor_episode_id,
            "history_episode_ids": [anchor_episode_id] * HISTORY_K,
            "reset_boundary": RESET_BOUNDARY,
        },
        "metadata": {
            "stable_base_checkpoint_kind": DEFAULT_STABLE_BASE_CHECKPOINT_KIND,
            "stable_base_checkpoint_value": DEFAULT_STABLE_BASE_CHECKPOINT_VALUE,
            "experiment_split": "devtrain",
            "teacher_producer": "oracle_sidecar",
            "teacher_version": "v1",
            "teacher_trigger_reason": "family_triggered_off_nominal",
            "teacher_trigger_success_rate": 0.92,
            "teacher_trigger_threshold": 0.9,
        },
    }


def build_state_conditioned_freeze() -> dict[str, object]:
    baseline_dataset_value = str(
        Path(state_conditioned_bucket_a_import.DEFAULT_SOURCE).resolve()
    )
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_contract_freeze",
        "contract_gate": CONTRACT_GATE_NAME,
        "baseline_dataset": {
            "kind": DEFAULT_BASELINE_DATASET_KIND,
            "value": baseline_dataset_value,
        },
        "stable_base_checkpoint": {
            "kind": DEFAULT_STABLE_BASE_CHECKPOINT_KIND,
            "value": DEFAULT_STABLE_BASE_CHECKPOINT_VALUE,
        },
        "phase_vocab": list(PHASE_VOCAB),
        "mode_vocab": list(MODE_VOCAB),
        "phase_index_by_value": {
            phase: index for index, phase in enumerate(PHASE_VOCAB)
        },
        "mode_index_by_value": {mode: index for index, mode in enumerate(MODE_VOCAB)},
        "history_contract": {
            "history_k": HISTORY_K,
            "history_stride": HISTORY_STRIDE,
            "history_valid_mask_required": True,
            "snapshot_window_kind": "anchor_mujoco_state_ref + prehistory window",
            "reset_boundary": RESET_BOUNDARY,
            "cross_episode_history_allowed": False,
            "validation_only_fields": list(HISTORY_VALIDATION_ONLY_FIELDS),
        },
        "experiment_split_allowlist": list(EXPERIMENT_SPLIT_ALLOWLIST),
        "deployable_history_allowlist": list(DEPLOYABLE_HISTORY_ALLOWLIST),
        "deployable_observation_allowlist": list(DEPLOYABLE_OBSERVATION_ALLOWLIST),
        "deployable_denylist": {
            "exact_names": list(DEPLOYABLE_DENYLIST_EXACT_NAMES),
            "prefixes": list(DEPLOYABLE_DENYLIST_PREFIXES),
        },
        "analysis_only_fields": list(ANALYSIS_ONLY_FIELDS),
        "teacher_provenance_contract": {
            "required_fields": list(TEACHER_PROVENANCE_FIELD_NAMES),
            "scope": "teacher-only",
        },
        "mainline_training_text": {
            "field": MAINLINE_TRAINING_TEXT_FIELD,
            "schema_version": text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION,
            "authority_name": text_indicator.RECAP_TEXT_INDICATOR_AUTHORITY_NAME,
            "authority_status": "mainline_authority",
            "mainline_authority": True,
            "diagnostic_only": False,
            "source_prompt_field": text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
            "indicator_source_field": "indicator_I",
            "policy_metadata_fields": list(POLICY_METADATA_FIELD_NAMES),
            "legacy_non_authority_fields": list(LEGACY_NON_AUTHORITY_FIELD_NAMES),
            "legacy_text_non_authority_fields": list(
                LEGACY_TEXT_NON_AUTHORITY_FIELD_NAMES
            ),
            "diagnostic_only_fields": list(DIAGNOSTIC_ONLY_FIELD_NAMES),
        },
        "policy_text_allowlist": list(POLICY_TEXT_ALLOWLIST),
        "policy_condition_text_template": POLICY_CONDITION_TEXT_TEMPLATE,
        "policy_text_surface": {
            "field": "policy_condition_text",
            "authority_status": "metadata_only",
            "mainline_authority": False,
            "diagnostic_only": False,
            "metadata_fields": list(POLICY_METADATA_FIELD_NAMES),
            "allowlist": list(POLICY_TEXT_ALLOWLIST),
            "template": POLICY_CONDITION_TEXT_TEMPLATE,
        },
        "reference_contract_example": build_reference_contract_example(),
    }


def validate_freeze_payload(payload: Mapping[str, Any]) -> dict[str, object]:
    freeze = dict(_as_mapping(payload, field_name="freeze_payload"))
    baseline_dataset = dict(
        _as_mapping(
            freeze.get("baseline_dataset"), field_name="freeze_payload.baseline_dataset"
        )
    )
    if tuple(freeze.get("phase_vocab", ())) != PHASE_VOCAB:
        raise ValueError(
            "phase_vocab does not match frozen 6-phase canonical vocabulary"
        )
    if tuple(freeze.get("mode_vocab", ())) != MODE_VOCAB:
        raise ValueError("mode_vocab does not match frozen 2-mode canonical vocabulary")
    if (
        _as_int(
            dict(
                _as_mapping(
                    freeze.get("history_contract"),
                    field_name="freeze_payload.history_contract",
                )
            ).get("history_k"),
            field_name="freeze_payload.history_contract.history_k",
        )
        != HISTORY_K
    ):
        raise ValueError(f"history_contract.history_k must be {HISTORY_K}")
    if baseline_dataset.get("kind") != DEFAULT_BASELINE_DATASET_KIND:
        raise ValueError("baseline_dataset.kind must freeze a single dataset_dir")
    _as_non_empty_string(
        baseline_dataset.get("value"),
        field_name="freeze_payload.baseline_dataset.value",
    )
    checkpoint = dict(
        _as_mapping(
            freeze.get("stable_base_checkpoint"),
            field_name="freeze_payload.stable_base_checkpoint",
        )
    )
    if checkpoint.get("kind") != DEFAULT_STABLE_BASE_CHECKPOINT_KIND:
        raise ValueError("stable_base_checkpoint.kind mismatch")
    if checkpoint.get("value") != DEFAULT_STABLE_BASE_CHECKPOINT_VALUE:
        raise ValueError("stable_base_checkpoint.value mismatch")
    allowlist = tuple(freeze.get("deployable_history_allowlist", ()))
    if allowlist != DEPLOYABLE_HISTORY_ALLOWLIST:
        raise ValueError("deployable_history_allowlist mismatch")
    denylist = dict(
        _as_mapping(
            freeze.get("deployable_denylist"),
            field_name="freeze_payload.deployable_denylist",
        )
    )
    prefixes = tuple(denylist.get("prefixes", ()))
    if "history." in prefixes:
        raise ValueError("history. must not appear in deployable_denylist.prefixes")
    if tuple(freeze.get("analysis_only_fields", ())) != ANALYSIS_ONLY_FIELDS:
        raise ValueError("analysis_only_fields mismatch")
    mainline_training_text = dict(
        _as_mapping(
            freeze.get("mainline_training_text"),
            field_name="freeze_payload.mainline_training_text",
        )
    )
    if mainline_training_text.get("field") != MAINLINE_TRAINING_TEXT_FIELD:
        raise ValueError("mainline_training_text.field must point to carrier_text_v1")
    if (
        mainline_training_text.get("schema_version")
        != text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION
    ):
        raise ValueError("mainline_training_text.schema_version mismatch")
    if (
        mainline_training_text.get("authority_name")
        != text_indicator.RECAP_TEXT_INDICATOR_AUTHORITY_NAME
    ):
        raise ValueError("mainline_training_text.authority_name mismatch")
    if mainline_training_text.get("authority_status") != "mainline_authority":
        raise ValueError("mainline_training_text.authority_status mismatch")
    if mainline_training_text.get("mainline_authority") is not True:
        raise ValueError("mainline_training_text.mainline_authority mismatch")
    if mainline_training_text.get("diagnostic_only") is not False:
        raise ValueError("mainline_training_text.diagnostic_only mismatch")
    if (
        mainline_training_text.get("source_prompt_field")
        != text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD
    ):
        raise ValueError("mainline_training_text.source_prompt_field mismatch")
    if mainline_training_text.get("indicator_source_field") != "indicator_I":
        raise ValueError("mainline_training_text.indicator_source_field mismatch")
    if (
        tuple(mainline_training_text.get("policy_metadata_fields", ()))
        != POLICY_METADATA_FIELD_NAMES
    ):
        raise ValueError("mainline_training_text.policy_metadata_fields mismatch")
    if (
        tuple(mainline_training_text.get("legacy_non_authority_fields", ()))
        != LEGACY_NON_AUTHORITY_FIELD_NAMES
    ):
        raise ValueError("mainline_training_text.legacy_non_authority_fields mismatch")
    if (
        tuple(mainline_training_text.get("legacy_text_non_authority_fields", ()))
        != LEGACY_TEXT_NON_AUTHORITY_FIELD_NAMES
    ):
        raise ValueError(
            "mainline_training_text.legacy_text_non_authority_fields mismatch"
        )
    if (
        tuple(mainline_training_text.get("diagnostic_only_fields", ()))
        != DIAGNOSTIC_ONLY_FIELD_NAMES
    ):
        raise ValueError("mainline_training_text.diagnostic_only_fields mismatch")
    if tuple(freeze.get("policy_text_allowlist", ())) != POLICY_TEXT_ALLOWLIST:
        raise ValueError("policy_text_allowlist must be ['phase', 'mode']")
    policy_text_surface = dict(
        _as_mapping(
            freeze.get("policy_text_surface"),
            field_name="freeze_payload.policy_text_surface",
        )
    )
    if policy_text_surface.get("field") != "policy_condition_text":
        raise ValueError("policy_text_surface.field mismatch")
    if policy_text_surface.get("authority_status") != "metadata_only":
        raise ValueError("policy_text_surface.authority_status mismatch")
    if policy_text_surface.get("mainline_authority") is not False:
        raise ValueError("policy_text_surface.mainline_authority mismatch")
    if policy_text_surface.get("diagnostic_only") is not False:
        raise ValueError("policy_text_surface.diagnostic_only mismatch")
    if (
        tuple(policy_text_surface.get("metadata_fields", ()))
        != POLICY_METADATA_FIELD_NAMES
    ):
        raise ValueError("policy_text_surface.metadata_fields mismatch")
    if tuple(policy_text_surface.get("allowlist", ())) != POLICY_TEXT_ALLOWLIST:
        raise ValueError("policy_text_surface.allowlist mismatch")
    if policy_text_surface.get("template") != POLICY_CONDITION_TEXT_TEMPLATE:
        raise ValueError("policy_text_surface.template mismatch")
    validate_contract_candidate(
        _as_mapping(
            freeze.get("reference_contract_example"),
            field_name="freeze_payload.reference_contract_example",
        )
    )
    return {
        "baseline_dataset_kind": str(baseline_dataset["kind"]),
        "history_k": HISTORY_K,
        "deployable_history_allowlist_count": len(allowlist),
    }


def build_phase_mode_fsm() -> dict[str, object]:
    return {
        "schema_version": FSM_SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_phase_mode_fsm",
        "phases": [
            {
                "name": phase,
                "predicate": PHASE_RULES[phase],
            }
            for phase in PHASE_VOCAB
        ],
        "modes": [
            {
                "name": mode,
                **dict(MODE_RULES[mode]),
            }
            for mode in MODE_VOCAB
        ],
        "allowed_phase_transitions": {
            phase: list(ALLOWED_PHASE_TRANSITIONS[phase]) for phase in PHASE_VOCAB
        },
        "allowed_mode_transitions": {
            mode: list(
                _as_list(
                    MODE_RULES[mode]["can_transition_to"],
                    field_name=f"MODE_RULES[{mode}].can_transition_to",
                )
            )
            for mode in MODE_VOCAB
        },
        "thresholds": {
            "hand_to_apple_distance_m_for_grasp": 0.08,
            "apple_to_plate_xy_distance_m_for_place": 0.12,
            "verify_hold_contiguous_policy_steps": 8,
        },
        "reset_clear_semantics": {
            "reset_boundary": RESET_BOUNDARY,
            "phase_on_reset": "SEARCH",
            "mode_on_reset": "NOMINAL",
            "history_valid_mask_required": True,
            "cross_episode_history_allowed": False,
            "clear_fields_on_reset": [
                "history_valid_mask",
                "anchor_mujoco_state_ref",
                "prehistory_window",
                "deployable.previous_action_history",
                "deployable.proprio_history",
                "deployable.short_visual_history_refs",
            ],
            "verify_hold_counter_reset": True,
        },
    }


def validate_phase_mode_fsm(payload: Mapping[str, Any]) -> dict[str, object]:
    fsm = dict(_as_mapping(payload, field_name="phase_mode_fsm"))
    phase_names = tuple(
        _as_non_empty_string(row.get("name"), field_name="phase_mode_fsm.phases[].name")
        for row in _as_list(
            fsm.get("phases"),
            field_name="phase_mode_fsm.phases",
            expected_len=len(PHASE_VOCAB),
        )
    )
    mode_names = tuple(
        _as_non_empty_string(row.get("name"), field_name="phase_mode_fsm.modes[].name")
        for row in _as_list(
            fsm.get("modes"),
            field_name="phase_mode_fsm.modes",
            expected_len=len(MODE_VOCAB),
        )
    )
    if phase_names != PHASE_VOCAB:
        raise ValueError("phase_mode_fsm phases do not match canonical vocabulary")
    if mode_names != MODE_VOCAB:
        raise ValueError("phase_mode_fsm modes do not match canonical vocabulary")
    transitions = dict(
        _as_mapping(
            fsm.get("allowed_phase_transitions"),
            field_name="phase_mode_fsm.allowed_phase_transitions",
        )
    )
    for phase in PHASE_VOCAB:
        if tuple(transitions.get(phase, ())) != ALLOWED_PHASE_TRANSITIONS[phase]:
            raise ValueError(f"allowed transitions mismatch for phase {phase}")
    reset_semantics = dict(
        _as_mapping(
            fsm.get("reset_clear_semantics"),
            field_name="phase_mode_fsm.reset_clear_semantics",
        )
    )
    if reset_semantics.get("reset_boundary") != RESET_BOUNDARY:
        raise ValueError("reset_clear_semantics.reset_boundary mismatch")
    if reset_semantics.get("phase_on_reset") != "SEARCH":
        raise ValueError("reset_clear_semantics.phase_on_reset must be SEARCH")
    if reset_semantics.get("mode_on_reset") != "NOMINAL":
        raise ValueError("reset_clear_semantics.mode_on_reset must be NOMINAL")
    if not bool(reset_semantics.get("history_valid_mask_required", False)):
        raise ValueError(
            "reset_clear_semantics.history_valid_mask_required must be true"
        )
    return {
        "phase_count": len(phase_names),
        "mode_count": len(mode_names),
        "reset_boundary": RESET_BOUNDARY,
    }


def _build_check_result(name: str, error: str | None = None) -> dict[str, object]:
    passed = error is None
    result: dict[str, object] = {
        "name": name,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }
    if error is not None:
        result["error"] = error
    return result


def build_contract_gate_report(
    freeze_payload: Mapping[str, Any],
    phase_mode_fsm: Mapping[str, Any],
) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}

    for name, validator, value in (
        ("freeze_payload", validate_freeze_payload, freeze_payload),
        ("phase_mode_fsm", validate_phase_mode_fsm, phase_mode_fsm),
        (
            "reference_contract_example",
            validate_contract_candidate,
            build_reference_contract_example(),
        ),
    ):
        try:
            validator(value)
        except (TypeError, ValueError) as exc:
            checks[name] = _build_check_result(name, _exception_message(exc))
        else:
            checks[name] = _build_check_result(name)

    passed = all(bool(result["passed"]) for result in checks.values())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_contract_gate_report",
        "contract_gate": {
            "name": CONTRACT_GATE_NAME,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
        },
        "freeze_schema_version": FREEZE_SCHEMA_VERSION,
        "fsm_schema_version": FSM_SCHEMA_VERSION,
        "checks": checks,
    }


def materialize_contract_gate(output_dir: Path) -> dict[str, str]:
    resolved_output_dir = _validate_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    freeze_payload = build_state_conditioned_freeze()
    phase_mode_fsm = build_phase_mode_fsm()
    report_payload = build_contract_gate_report(freeze_payload, phase_mode_fsm)

    freeze_path = _write_json(resolved_output_dir / FREEZE_JSON_NAME, freeze_payload)
    report_path = _write_json(
        resolved_output_dir / CONTRACT_GATE_REPORT_JSON_NAME,
        report_payload,
    )
    fsm_path = _write_json(
        resolved_output_dir / PHASE_MODE_FSM_JSON_NAME, phase_mode_fsm
    )

    return {
        "state_conditioned_freeze_path": str(freeze_path),
        "contract_gate_report_path": str(report_path),
        "phase_mode_fsm_path": str(fsm_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        materialize_contract_gate(args.output_dir)
    except SystemExit:
        raise
    except (OSError, TypeError, ValueError) as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
