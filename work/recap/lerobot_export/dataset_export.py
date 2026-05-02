from __future__ import annotations

import json
import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from work.openpi.recap.prompt_builder import (
    CONDITIONING_MODE as EXPORTER_CONDITIONING_MODE,
)
from work.openpi.recap.prompt_builder import (
    PHASE1_PROMPT_ROUTE as EXPORTER_PROMPT_ROUTE,
)
from work.recap.text_indicator import (
    RECAP_TEXT_INDICATOR_CARRIER_FIELD as EXPORTER_CARRIER_ROUTE,
)
from work.recap.text_indicator import (
    RECAP_TEXT_INDICATOR_CARRIER_FIELD as EXPORTER_MAINLINE_TASK_TEXT_FIELD,
)
from work.recap.text_indicator import (
    RECAP_TEXT_INDICATOR_SCHEMA_VERSION as EXPORTER_CARRIER_SCHEMA_VERSION,
)
from work.recap.text_indicator import (
    RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD as EXPORTER_PROMPT_SOURCE_FIELD,
)
from work.recap.text_indicator import require_authoritative_carrier_text_v1
from work.recap.text_indicator import indicator_mode_from_indicator_value
from work.recap.text_indicator import normalize_indicator_mode
from work.recap.lerobot_export.contract_io import as_int as _as_int
from work.recap.lerobot_export.contract_io import read_json_object as _read_json_dict
from work.recap.lerobot_export.contract_io import (
    read_jsonl_objects as _read_jsonl_dicts,
)
from work.recap.lerobot_export.contract_io import write_json_object as _write_json
from work.recap.lerobot_export.contract_io import write_jsonl_objects as _write_jsonl

LEROBOT_V2_OUTPUT_ROOT_REL = Path("agent") / "artifacts" / "lerobot_datasets"

WBC_STATE_GROUP_KEY = "wbc_state"
WBC_ACTION_GROUP_KEY = "wbc_action"
DEPLOYABLE_HISTORY_GROUP_KEY = "deployable_history"
PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY = "privileged_analysis_only"
TEACHER_ONLY_GROUP_KEY = "teacher_only"

STATE_GROUP_NAME = f"state.{WBC_STATE_GROUP_KEY}"
ACTION_GROUP_NAME = f"action.{WBC_ACTION_GROUP_KEY}"

LANGUAGE_ANNOTATION_KEY = "annotation.human.action.task_description"
LANGUAGE_ANNOTATION_KEY_ALIAS = "annotation.human.task_description"

DEPLOYABLE_HISTORY_FIELD_NAMES: tuple[str, ...] = (
    "history_k",
    "history_stride",
    "history_valid_mask",
    "history_t_std_indices",
    "history_t_raw_indices",
    "history_timestamp_s",
    "deployable.previous_action_history",
    "deployable.proprio_history",
    "deployable.short_visual_history_refs",
)

PRIVILEGED_ANALYSIS_ONLY_FIELD_NAMES: tuple[str, ...] = (
    "privileged.apple_pose_world",
    "privileged.hand_to_apple_rel_pose",
    "privileged.apple_to_plate_rel_pose",
    "privileged.contact_flag",
    "privileged.apple_in_hand",
    "privileged.apple_visible",
    "privileged.last_seen_dt",
    "privileged.last_in_hand_dt",
    "semantic_state",
    "memory_commit_mask",
    "memory_commit_cause",
    "recovery_entry_step",
    "recovery_exit_step",
    "summary_template",
)

TEACHER_ONLY_FIELD_PREFIXES: tuple[str, ...] = (
    "teacher.",
    "oracle.",
    "hindsight.",
    "future.",
)

DEPLOYABLE_FIELD_LEAKAGE_PREFIXES: tuple[str, ...] = (
    "privileged.",
    *TEACHER_ONLY_FIELD_PREFIXES,
)

DEPLOYABLE_FIELD_LEAKAGE_EXACT_NAMES: tuple[str, ...] = (
    "semantic_state",
    "memory_commit_mask",
    "memory_commit_cause",
    "recovery_entry_step",
    "recovery_exit_step",
    "summary_template",
)

META_DIRNAME = "meta"
DATA_DIRNAME = "data"

META_INFO_JSON = "info.json"
META_EPISODES_JSONL = "episodes.jsonl"
META_TASKS_JSONL = "tasks.jsonl"
META_MODALITY_JSON = "modality.json"
META_STATS_JSON = "stats.json"

DATA_CHUNK_DIRNAME = "chunk-000"

EXPORTER_INDICATOR_MODE_FIELD = "indicator_mode"
EXPORTER_INDICATOR_SOURCE_FIELD = "indicator_source"
EXPORTER_INDICATOR_VALUE_SOURCE_FIELD = "recap_m2.indicator_I"


__all__ = [
    "LEROBOT_V2_OUTPUT_ROOT_REL",
    "WBC_STATE_GROUP_KEY",
    "WBC_ACTION_GROUP_KEY",
    "DEPLOYABLE_HISTORY_GROUP_KEY",
    "PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY",
    "TEACHER_ONLY_GROUP_KEY",
    "STATE_GROUP_NAME",
    "ACTION_GROUP_NAME",
    "LANGUAGE_ANNOTATION_KEY",
    "DEPLOYABLE_HISTORY_FIELD_NAMES",
    "PRIVILEGED_ANALYSIS_ONLY_FIELD_NAMES",
    "TEACHER_ONLY_FIELD_PREFIXES",
    "META_DIRNAME",
    "DATA_DIRNAME",
    "META_INFO_JSON",
    "META_EPISODES_JSONL",
    "META_TASKS_JSONL",
    "META_MODALITY_JSON",
    "META_STATS_JSON",
    "DATA_CHUNK_DIRNAME",
    "EXPORTER_MAINLINE_TASK_TEXT_FIELD",
    "EXPORTER_CARRIER_SCHEMA_VERSION",
    "EXPORTER_CARRIER_ROUTE",
    "EXPORTER_PROMPT_SOURCE_FIELD",
    "EXPORTER_PROMPT_ROUTE",
    "EXPORTER_CONDITIONING_MODE",
    "EXPORTER_INDICATOR_MODE_FIELD",
    "EXPORTER_INDICATOR_SOURCE_FIELD",
    "EXPORTER_INDICATOR_VALUE_SOURCE_FIELD",
    "build_state_conditioned_field_groups",
    "validate_state_conditioned_field_groups",
    "resolve_lerobot_v2_dataset_dir",
    "validate_export_config",
    "STATE_KEY_ORDER_LOCK",
    "ACTION_KEY_ORDER_LOCK",
    "LeRobotV2ExportResult",
    "export_recap_to_lerobot_v2",
]


_ITER_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_iter_tag(iter_tag: str) -> None:
    if not isinstance(iter_tag, str) or not iter_tag:
        raise ValueError(f"iter_tag must be a non-empty str, got {iter_tag!r}")
    if not _ITER_TAG_RE.match(iter_tag):
        raise ValueError(
            "iter_tag must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ "
            f"(got {iter_tag!r})"
        )
    if "/" in iter_tag or "\\" in iter_tag:
        raise ValueError(f"iter_tag must not contain path separators: {iter_tag!r}")
    if iter_tag in (".", ".."):
        raise ValueError(f"iter_tag must not be '.' or '..': {iter_tag!r}")


def resolve_lerobot_v2_dataset_dir(
    iter_tag: str,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    _validate_iter_tag(iter_tag)

    root = _repo_root_from_here() if repo_root is None else Path(repo_root)
    root = root.resolve()
    return root / LEROBOT_V2_OUTPUT_ROOT_REL / iter_tag


def validate_export_config(
    *,
    iter_tag: str,
    output_dataset_dir: str | Path,
    state_group_name: str = STATE_GROUP_NAME,
    action_group_name: str = ACTION_GROUP_NAME,
    language_annotation_key: str = LANGUAGE_ANNOTATION_KEY,
    repo_root: str | Path | None = None,
) -> Path:
    _validate_iter_tag(iter_tag)

    if state_group_name != STATE_GROUP_NAME:
        raise ValueError(
            f"state_group_name is locked to {STATE_GROUP_NAME!r} (got {state_group_name!r})"
        )
    if action_group_name != ACTION_GROUP_NAME:
        raise ValueError(
            f"action_group_name is locked to {ACTION_GROUP_NAME!r} (got {action_group_name!r})"
        )
    if language_annotation_key != LANGUAGE_ANNOTATION_KEY:
        raise ValueError(
            f"language_annotation_key is locked to {LANGUAGE_ANNOTATION_KEY!r} "
            f"(got {language_annotation_key!r})"
        )

    expected = resolve_lerobot_v2_dataset_dir(iter_tag, repo_root=repo_root)
    got = Path(output_dataset_dir).resolve()
    if got != expected:
        raise ValueError(
            "output_dataset_dir is locked; expected "
            f"{expected.as_posix()!r} but got {got.as_posix()!r}"
        )

    return expected


# Deterministic schema locks for NPZ concatenation order.
#
# Rationale: this exporter flattens multiple WBC state/action groups into a single
# 1D vector each (LeRobot v2 columns `observation.state` and `action`). The vector
# dimension ordering must be stable across runs and iterations, otherwise the
# loader/trainer will silently learn the wrong mapping.
STATE_KEY_ORDER_LOCK: list[str] = [
    "state/left_arm",
    "state/left_hand",
    "state/left_leg",
    "state/right_arm",
    "state/right_hand",
    "state/right_leg",
    "state/waist",
]

ACTION_KEY_ORDER_LOCK: list[str] = [
    "action/base_height_command",
    "action/left_arm",
    "action/left_hand",
    "action/navigate_command",
    "action/right_arm",
    "action/right_hand",
    "action/waist",
]


@dataclass(frozen=True)
class LeRobotV2ExportResult:
    output_dataset_dir: Path
    total_episodes: int
    total_frames: int
    total_tasks: int
    state_dim: int
    action_dim: int


def _pick_task_text(label: dict[str, Any], *, field: str) -> str:
    val = label.get(field)
    if field == EXPORTER_MAINLINE_TASK_TEXT_FIELD:
        if not isinstance(val, str) or not val.strip():
            episode_id = label.get("episode_id")
            t_i = label.get("t")
            raise ValueError(
                "Missing authoritative carrier_text_v1 for mainline single-text export; "
                "exporter will not fall back to prompt_conditioned or prompt_raw "
                f"(episode_id={episode_id!r} t={t_i!r})"
            )
        indicator_mode = _label_indicator_mode(label)
        if indicator_mode is None:
            episode_id = label.get("episode_id")
            t_i = label.get("t")
            raise ValueError(
                "Missing indicator_I/indicator_mode for mainline carrier_text_v1 authority validation "
                f"(episode_id={episode_id!r} t={t_i!r})"
            )
        return require_authoritative_carrier_text_v1(
            val,
            prompt_raw=label.get(EXPORTER_PROMPT_SOURCE_FIELD),
            indicator_mode=indicator_mode,
        )
    if isinstance(val, str) and val.strip():
        return val
    # Fallbacks for robustness.
    for k in ("prompt_conditioned", "prompt_raw"):
        v = label.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _non_empty_str(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    if not raw.strip():
        return None
    return raw


def _label_string(label: dict[str, Any], *candidate_keys: str) -> str | None:
    for key in candidate_keys:
        value = _non_empty_str(label.get(key))
        if value is not None:
            return value
    return None


def _label_indicator_mode(label: dict[str, Any]) -> str | None:
    explicit_mode = _label_string(
        label,
        EXPORTER_INDICATOR_MODE_FIELD,
        f"recap_m2.{EXPORTER_INDICATOR_MODE_FIELD}",
    )
    if explicit_mode is not None:
        return normalize_indicator_mode(
            explicit_mode,
            field_name=EXPORTER_INDICATOR_MODE_FIELD,
        )
    if "indicator_I" in label:
        return indicator_mode_from_indicator_value(
            label.get("indicator_I"),
            field_name="indicator_I",
        )
    return None


def _build_mainline_text_provenance(
    labels: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    indicator_modes: set[str] = set()
    indicator_sources: set[str] = set()
    for label in labels:
        indicator_mode = _label_indicator_mode(label)
        if indicator_mode is not None:
            indicator_modes.add(indicator_mode)
        indicator_source = _label_string(
            label,
            EXPORTER_INDICATOR_SOURCE_FIELD,
            f"recap_m2.{EXPORTER_INDICATOR_SOURCE_FIELD}",
        )
        if indicator_source is not None:
            indicator_sources.add(indicator_source)

    provenance: dict[str, Any] = {
        "carrier_schema_version": EXPORTER_CARRIER_SCHEMA_VERSION,
        "carrier_route": EXPORTER_CARRIER_ROUTE,
        "prompt_source_field": EXPORTER_PROMPT_SOURCE_FIELD,
        "prompt_route": EXPORTER_PROMPT_ROUTE,
        "conditioning_mode": EXPORTER_CONDITIONING_MODE,
        "indicator_mode_field": EXPORTER_INDICATOR_MODE_FIELD,
        "indicator_source_field": EXPORTER_INDICATOR_SOURCE_FIELD,
        "indicator_mode_source_field": EXPORTER_INDICATOR_VALUE_SOURCE_FIELD,
    }
    if indicator_modes:
        provenance["indicator_mode_values"] = sorted(indicator_modes)
        if len(indicator_modes) == 1:
            provenance["indicator_mode"] = next(iter(indicator_modes))
    if indicator_sources:
        provenance["indicator_source_values"] = sorted(indicator_sources)
        if len(indicator_sources) == 1:
            provenance["indicator_source"] = next(iter(indicator_sources))
    return provenance


def _stable_mix50_pick_raw(*, episode_id: str, t_i: int) -> bool:
    key = f"{episode_id}:{int(t_i)}".encode("utf-8")
    return int(zlib.crc32(key) % 2) == 0


def _feature_names_for_group(
    keys: list[str], dims_by_key: dict[str, int], *, group: str
) -> list[str]:
    names: list[str] = []
    for k in keys:
        d = int(dims_by_key[k])
        base = k.split("/", 1)[1] if "/" in k else k
        for i in range(d):
            names.append(f"{group}.{base}:{i}")
    return names


def _build_group_offsets(
    keys: list[str], dims_by_key: dict[str, int]
) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    cur = 0
    for k in keys:
        d = int(dims_by_key[k])
        base = k.split("/", 1)[1] if "/" in k else k
        out[str(base)] = (int(cur), int(cur + d))
        cur += d
    return out


def _normalize_field_names(field_names: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for field_name in field_names:
        if not isinstance(field_name, str):
            raise TypeError(
                "state-conditioned field names must be strings, got "
                + type(field_name).__name__
            )
        candidate = field_name.strip()
        if not candidate:
            raise ValueError("state-conditioned field names must be non-empty")
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def validate_state_conditioned_field_groups(
    field_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    required_keys = {
        DEPLOYABLE_HISTORY_GROUP_KEY,
        PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY,
        TEACHER_ONLY_GROUP_KEY,
    }
    missing = sorted(required_keys - set(field_groups.keys()))
    if missing:
        raise ValueError(f"state-conditioned field groups missing keys: {missing}")

    normalized = {
        group_name: _normalize_field_names(field_names)
        for group_name, field_names in field_groups.items()
    }

    deployable_fields = normalized[DEPLOYABLE_HISTORY_GROUP_KEY]
    allowed_deployable = set(DEPLOYABLE_HISTORY_FIELD_NAMES)
    for field_name in deployable_fields:
        if field_name not in allowed_deployable:
            raise ValueError(
                "unexpected field in deployable_history: "
                f"{field_name!r}; only the frozen history-aware deployable fields are allowed"
            )
        if field_name in DEPLOYABLE_FIELD_LEAKAGE_EXACT_NAMES or any(
            field_name.startswith(prefix)
            for prefix in DEPLOYABLE_FIELD_LEAKAGE_PREFIXES
        ):
            raise ValueError(
                f"analysis-only or teacher-only field leaked into deployable_history: {field_name!r}"
            )

    privileged_fields = normalized[PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY]
    for field_name in privileged_fields:
        if field_name in DEPLOYABLE_FIELD_LEAKAGE_EXACT_NAMES:
            continue
        if field_name.startswith("privileged."):
            continue
        raise ValueError(
            "privileged_analysis_only accepts only privileged.* or frozen analysis-only fields; got "
            + repr(field_name)
        )

    teacher_fields = normalized[TEACHER_ONLY_GROUP_KEY]
    for field_name in teacher_fields:
        if not any(
            field_name.startswith(prefix) for prefix in TEACHER_ONLY_FIELD_PREFIXES
        ):
            raise ValueError(
                "teacher_only accepts only teacher./oracle./hindsight./future. fields; got "
                + repr(field_name)
            )

    overlap_checks = (
        (DEPLOYABLE_HISTORY_GROUP_KEY, PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY),
        (DEPLOYABLE_HISTORY_GROUP_KEY, TEACHER_ONLY_GROUP_KEY),
        (PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY, TEACHER_ONLY_GROUP_KEY),
    )
    for left_name, right_name in overlap_checks:
        overlap = sorted(set(normalized[left_name]) & set(normalized[right_name]))
        if overlap:
            raise ValueError(
                f"state-conditioned field groups overlap between {left_name} and {right_name}: {overlap}"
            )
    return normalized


def build_state_conditioned_field_groups(
    observed_field_names: Iterable[str] = (),
) -> dict[str, list[str]]:
    observed = _normalize_field_names(observed_field_names)
    teacher_only_fields = [
        field_name
        for field_name in observed
        if any(field_name.startswith(prefix) for prefix in TEACHER_ONLY_FIELD_PREFIXES)
    ]
    return validate_state_conditioned_field_groups(
        {
            DEPLOYABLE_HISTORY_GROUP_KEY: list(DEPLOYABLE_HISTORY_FIELD_NAMES),
            PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY: list(
                PRIVILEGED_ANALYSIS_ONLY_FIELD_NAMES
            ),
            TEACHER_ONLY_GROUP_KEY: teacher_only_fields,
        }
    )


def export_recap_to_lerobot_v2(
    *,
    iter_tag: str,
    repo_root: str | Path | None = None,
    input_recap_dataset_dir: str | Path | None = None,
    output_dataset_dir: str | Path | None = None,
    max_episodes: int | None = None,
    task_text_field: str = EXPORTER_MAINLINE_TASK_TEXT_FIELD,
    dual_task_text: bool = False,
    fps: float = 30.0,
    chunk_size: int = 1000,
    include_m2_label_columns: bool = True,
    overwrite_existing: bool = False,
) -> LeRobotV2ExportResult:
    """Export a RECAP (M1+M2) dataset to GR00T-flavored LeRobot v2 format.

    Inputs (read-only):
    - M1: `agent/artifacts/recap_datasets/<iter_tag>/episodes.jsonl`, `transitions.jsonl`, `arrays/*.npz`
    - M2: `agent/artifacts/recap_datasets/<iter_tag>/m2_labels/labels.jsonl`

    Output (write-only):
    - `agent/artifacts/lerobot_datasets/<iter_tag>/`.

    Notes:
    - Import-time is stdlib-only; third-party deps are imported at runtime.
    - State/action are flattened using locked key orders: `STATE_KEY_ORDER_LOCK` and `ACTION_KEY_ORDER_LOCK`.
    """

    root = _repo_root_from_here() if repo_root is None else Path(repo_root)
    root = root.resolve()

    if input_recap_dataset_dir is None:
        input_dir = root / "agent" / "artifacts" / "recap_datasets" / iter_tag
    else:
        input_dir = Path(input_recap_dataset_dir)
        if not input_dir.is_absolute():
            input_dir = (root / input_dir).resolve()
        else:
            input_dir = input_dir.resolve()

    expected_output_dir = resolve_lerobot_v2_dataset_dir(iter_tag, repo_root=root)
    out_dir = (
        expected_output_dir if output_dataset_dir is None else Path(output_dataset_dir)
    )
    out_dir = validate_export_config(
        iter_tag=iter_tag,
        output_dataset_dir=out_dir,
        repo_root=root,
    )

    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError(f"max_episodes must be positive or None, got {max_episodes!r}")
    if float(fps) <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    if not isinstance(task_text_field, str) or not task_text_field:
        raise ValueError(
            f"task_text_field must be a non-empty str, got {task_text_field!r}"
        )
    if not isinstance(dual_task_text, bool):
        raise ValueError(f"dual_task_text must be a bool, got {dual_task_text!r}")

    if out_dir.exists():
        if not bool(overwrite_existing):
            raise FileExistsError(f"Refusing to overwrite existing dataset dir: {out_dir}")
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()

    try:
        from work.recap.dataset_reader import read_m1_dataset
    except Exception as e:
        raise RuntimeError(f"Failed to import work.recap.dataset_reader: {e}") from e

    try:
        from work.recap.advantage import (
            ADVANTAGE_CONTRACT_VERSION,
            ADVANTAGE_INPUT_COLUMN,
            ADVANTAGE_RAW_COLUMN,
            ADVANTAGE_RETURN_COLUMN,
            ADVANTAGE_SCALE_RULE,
            ADVANTAGE_VALUE_COLUMN,
            LEGACY_ADVANTAGE_CONTRACT_VERSION,
            extract_advantage_contract,
            normalize_advantage_to_input,
            validate_advantage_input_value,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import work.recap.advantage: {e}") from e

    # Runtime imports (non-stdlib).
    try:
        import importlib

        np = importlib.import_module("numpy")
        pd = importlib.import_module("pandas")
    except Exception as e:
        raise RuntimeError(
            f"export_recap_to_lerobot_v2 requires numpy+pandas: {e}"
        ) from e

    m1 = read_m1_dataset(input_dir, check_npz_keys=True)
    episodes_raw = m1.get("episodes")
    transitions_by_episode_raw = m1.get("transitions_by_episode")
    if not isinstance(episodes_raw, list) or not isinstance(
        transitions_by_episode_raw, dict
    ):
        raise ValueError("Invalid M1 dataset object from read_m1_dataset")

    episodes: list[dict[str, Any]] = [
        ep
        for ep in episodes_raw
        if isinstance(ep, dict) and isinstance(ep.get("episode_id"), str)
    ]
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]
    if not episodes:
        raise ValueError(
            f"No episodes to export (iter_tag={iter_tag!r}, input_dir={input_dir})"
        )

    transitions_by_episode: dict[str, list[dict[str, Any]]] = {}
    for eid, trs in transitions_by_episode_raw.items():
        if not isinstance(eid, str) or not isinstance(trs, list):
            continue
        transitions_by_episode[eid] = [tr for tr in trs if isinstance(tr, dict)]

    labels_path = input_dir / "m2_labels" / "labels.jsonl"
    labels_raw = _read_jsonl_dicts(labels_path)
    labels_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for idx, lab in enumerate(labels_raw, start=1):
        episode_id = lab.get("episode_id")
        t = lab.get("t")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(
                f"Invalid label episode_id in {labels_path} record#{idx}: {episode_id!r}"
            )
        t_i = _as_int(t, context=f"labels.jsonl record#{idx} episode_id={episode_id}")
        k = (episode_id, t_i)
        if k in labels_by_key:
            raise ValueError(f"Duplicate label for (episode_id,t)={k} in {labels_path}")
        labels_by_key[k] = lab

    advantage_contract: dict[str, Any] | None = None
    if include_m2_label_columns:
        advantage_contract_path_candidates = [
            input_dir / "continuous_advantage_contract.json",
            input_dir / "m2_labels" / "continuous_advantage_contract.json",
            input_dir / "info.json",
            input_dir / "m2_labels" / "info.json",
        ]
        for contract_path in advantage_contract_path_candidates:
            if not contract_path.exists():
                continue
            raw_contract_obj = _read_json_dict(contract_path)
            if contract_path.name == "info.json" or isinstance(
                raw_contract_obj.get("recap_advantage_input_contract"), dict
            ):
                contract_obj = extract_advantage_contract(raw_contract_obj)
            else:
                contract_obj = dict(raw_contract_obj)
                contract_version = contract_obj.get(
                    "contract_version"
                ) or contract_obj.get("version")
                if contract_version not in {
                    ADVANTAGE_CONTRACT_VERSION,
                    LEGACY_ADVANTAGE_CONTRACT_VERSION,
                }:
                    raise ValueError(
                        "Unsupported advantage contract version in "
                        f"{contract_path}: {contract_version!r}"
                    )
                model_advantage_column = contract_obj.get(
                    "model_advantage_column"
                ) or contract_obj.get("model_facing_column")
                if model_advantage_column != ADVANTAGE_INPUT_COLUMN:
                    raise ValueError(
                        "Unexpected model-facing advantage column in "
                        f"{contract_path}: expected {ADVANTAGE_INPUT_COLUMN!r} got "
                        f"{model_advantage_column!r}"
                    )
            advantage_contract = dict(contract_obj)
            break

    advantage_input_label_key = ADVANTAGE_INPUT_COLUMN.split(".", 1)[1]
    advantage_raw_label_key = ADVANTAGE_RAW_COLUMN.split(".", 1)[1]
    advantage_scale_rule = (
        None if advantage_contract is None else advantage_contract.get("scale_rule")
    )
    advantage_positive_scale = (
        None if advantage_contract is None else advantage_contract.get("positive_scale")
    )
    advantage_negative_scale_abs = (
        None
        if advantage_contract is None
        else advantage_contract.get("negative_scale_abs")
    )
    advantage_shared_scale = (
        None
        if advantage_contract is None
        else advantage_contract.get("p95_abs_advantage")
    )
    if include_m2_label_columns and advantage_scale_rule not in {
        None,
        ADVANTAGE_SCALE_RULE,
    }:
        if (
            advantage_positive_scale is not None
            or advantage_negative_scale_abs is not None
        ):
            raise ValueError(
                "Unsupported sign-aware advantage scale_rule: "
                f"expected {ADVANTAGE_SCALE_RULE!r} got {advantage_scale_rule!r}"
            )
    total_transitions_all = 0
    for trs in transitions_by_episode.values():
        total_transitions_all += int(len(trs))
    if int(total_transitions_all) != int(len(labels_raw)):
        raise ValueError(
            f"M1/M2 count mismatch: transitions.jsonl has {total_transitions_all} records but labels.jsonl has {len(labels_raw)} records"
        )

    # Pre-scan tasks in deterministic order.
    task_texts_set: set[str] = set()
    task_texts_in_order: list[str] = []
    for ep in episodes:
        episode_id = ep.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            continue
        for tr in transitions_by_episode.get(episode_id, []):
            t_i = _as_int(tr.get("t"), context=f"transition episode_id={episode_id}")
            label = labels_by_key.get((episode_id, t_i))
            if label is None:
                raise ValueError(
                    f"Missing M2 label for episode_id={episode_id} t={t_i}"
                )
            if not dual_task_text:
                text = _pick_task_text(label, field=task_text_field)
                if not text:
                    raise ValueError(
                        f"Empty task text (task_text_field={task_text_field!r}) for episode_id={episode_id} t={t_i}"
                    )
                if text not in task_texts_set:
                    task_texts_set.add(text)
                    task_texts_in_order.append(text)
            else:
                raw = label.get("prompt_raw")
                conditioned = label.get("prompt_conditioned")
                raw_s = raw if isinstance(raw, str) and raw else ""
                conditioned_s = (
                    conditioned if isinstance(conditioned, str) and conditioned else ""
                )
                if not raw_s and not conditioned_s:
                    raise ValueError(
                        f"Empty task text (prompt_raw/prompt_conditioned) for episode_id={episode_id} t={t_i}"
                    )
                for text in (raw_s, conditioned_s):
                    if not text:
                        continue
                    if text not in task_texts_set:
                        task_texts_set.add(text)
                        task_texts_in_order.append(text)

    # Stable task index assignment.
    tasks_sorted = sorted(task_texts_in_order)
    task_to_index = {t: i for i, t in enumerate(tasks_sorted)}

    meta_dir = out_dir / META_DIRNAME
    data_dir = out_dir / DATA_DIRNAME
    meta_dir.mkdir(parents=True, exist_ok=False)
    data_dir.mkdir(parents=True, exist_ok=False)

    all_state_rows: list[Any] = []
    all_action_rows: list[Any] = []
    episodes_meta_out: list[dict[str, Any]] = []

    state_dim: int | None = None
    action_dim: int | None = None
    dims_state_lock: dict[str, int] | None = None
    dims_action_lock: dict[str, int] | None = None
    total_frames = 0

    for episode_index, ep in enumerate(episodes):
        episode_id = ep.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"Invalid episode_id in episodes.jsonl: {episode_id!r}")

        trs = transitions_by_episode.get(episode_id, [])
        if not trs:
            raise ValueError(f"No transitions for episode_id={episode_id}")

        npz_path_val = ep.get("npz_path")
        if not isinstance(npz_path_val, str) or not npz_path_val:
            raise ValueError(f"Missing npz_path for episode_id={episode_id}")
        npz_path = Path(npz_path_val)
        npz_path = npz_path if npz_path.is_absolute() else (input_dir / npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        with np.load(npz_path, allow_pickle=False) as data:
            keys = list(getattr(data, "files", []))
            state_keys = sorted(
                [k for k in keys if isinstance(k, str) and k.startswith("state/")]
            )
            action_keys = sorted(
                [k for k in keys if isinstance(k, str) and k.startswith("action/")]
            )
            other_keys = [
                k
                for k in keys
                if isinstance(k, str)
                and (k not in state_keys)
                and (k not in action_keys)
            ]
            if other_keys:
                other_preview = ", ".join(sorted(other_keys)[:5])
                raise ValueError(
                    f"episode_id={episode_id} NPZ contains unexpected keys: {other_preview} (file={npz_path})"
                )
            if state_keys != STATE_KEY_ORDER_LOCK:
                raise ValueError(
                    f"episode_id={episode_id} state key order mismatch: expected {STATE_KEY_ORDER_LOCK} but got {state_keys}"
                )
            if action_keys != ACTION_KEY_ORDER_LOCK:
                raise ValueError(
                    f"episode_id={episode_id} action key order mismatch: expected {ACTION_KEY_ORDER_LOCK} but got {action_keys}"
                )

            # Build per-key dims and validate shapes.
            n_policy_steps = len(trs)
            dims_state: dict[str, int] = {}
            dims_action: dict[str, int] = {}
            t_action_from_npz: int | None = None

            state_parts: list[Any] = []
            action_parts: list[Any] = []

            for k in state_keys:
                arr = np.asarray(data[k])
                if arr.ndim != 4:
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} expected ndim=4, got shape={arr.shape}"
                    )
                if int(arr.shape[0]) != int(n_policy_steps):
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} n_policy_steps mismatch: transitions={n_policy_steps} but npz has {arr.shape[0]}"
                    )
                if int(arr.shape[1]) != 1 or int(arr.shape[2]) != 1:
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} expected shape[1:3]=(1,1), got shape={arr.shape}"
                    )
                d = int(arr.shape[3])
                dims_state[k] = d
                state_parts.append(arr[:, 0, 0, :].astype(np.float32, copy=False))

            for k in action_keys:
                arr = np.asarray(data[k])
                if arr.ndim != 4:
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} expected ndim=4, got shape={arr.shape}"
                    )
                if int(arr.shape[0]) != int(n_policy_steps):
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} n_policy_steps mismatch: transitions={n_policy_steps} but npz has {arr.shape[0]}"
                    )
                if int(arr.shape[1]) != 1:
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} expected shape[1]=1, got shape={arr.shape}"
                    )
                t_action = int(arr.shape[2])
                if t_action_from_npz is None:
                    t_action_from_npz = t_action
                elif int(t_action_from_npz) != int(t_action):
                    raise ValueError(
                        f"episode_id={episode_id} inconsistent T_action across keys: expected {t_action_from_npz} but {k!r} has {t_action}"
                    )
                d = int(arr.shape[3])
                dims_action[k] = d
                action_parts.append(arr[:, 0, :, :].astype(np.float32, copy=False))

            if t_action_from_npz is None:
                raise ValueError(f"episode_id={episode_id} missing action keys in npz")

            state_by_step = np.concatenate(state_parts, axis=-1)
            action_by_step = np.concatenate(action_parts, axis=-1)
            # state_by_step: (n_policy_steps, state_dim)
            # action_by_step: (n_policy_steps, T_action, action_dim)

            if state_dim is None:
                state_dim = int(state_by_step.shape[-1])
            elif int(state_dim) != int(state_by_step.shape[-1]):
                raise ValueError(
                    f"state_dim mismatch across episodes: expected {state_dim} got {state_by_step.shape[-1]} (episode_id={episode_id})"
                )

            if action_dim is None:
                action_dim = int(action_by_step.shape[-1])
            elif int(action_dim) != int(action_by_step.shape[-1]):
                raise ValueError(
                    f"action_dim mismatch across episodes: expected {action_dim} got {action_by_step.shape[-1]} (episode_id={episode_id})"
                )

            if dims_state_lock is None:
                dims_state_lock = dict(dims_state)
            elif dims_state_lock != dims_state:
                raise ValueError(
                    f"state per-key dims mismatch across episodes (episode_id={episode_id}): expected {dims_state_lock} got {dims_state}"
                )

            if dims_action_lock is None:
                dims_action_lock = dict(dims_action)
            elif dims_action_lock != dims_action:
                raise ValueError(
                    f"action per-key dims mismatch across episodes (episode_id={episode_id}): expected {dims_action_lock} got {dims_action}"
                )

            # Episode expansion: each policy step expands into n_action_steps_executed inner steps.
            rows_state: list[Any] = []
            rows_action: list[Any] = []
            rows_timestamp: list[float] = []
            rows_episode_index: list[int] = []
            rows_index: list[int] = []
            rows_task_index: list[int] = []
            rows_labels: dict[str, list[Any]] = {}

            step_frame_idx = 0
            for step_idx, tr in enumerate(trs):
                t_i = _as_int(
                    tr.get("t"), context=f"transition episode_id={episode_id}"
                )
                if t_i != step_idx:
                    raise ValueError(
                        f"episode_id={episode_id} invalid transition order: expected t={step_idx} but got t={t_i}"
                    )

                n_exec = _as_int(
                    tr.get("n_action_steps_executed"),
                    context=f"transition episode_id={episode_id} t={t_i}",
                )
                if n_exec <= 0:
                    raise ValueError(
                        f"episode_id={episode_id} t={t_i} invalid n_action_steps_executed={n_exec}"
                    )
                if n_exec > int(t_action_from_npz):
                    raise ValueError(
                        f"episode_id={episode_id} t={t_i} n_action_steps_executed={n_exec} exceeds T_action={t_action_from_npz}"
                    )

                t_action_from_tr = tr.get("T_action", tr.get("n_action_steps_config"))
                if t_action_from_tr is not None:
                    t_tr = _as_int(
                        t_action_from_tr,
                        context=f"transition episode_id={episode_id} t={t_i}",
                    )
                    if int(t_tr) != int(t_action_from_npz):
                        raise ValueError(
                            f"episode_id={episode_id} t={t_i} T_action mismatch: transitions={t_tr} npz={t_action_from_npz}"
                        )

                label = labels_by_key.get((episode_id, t_i))
                if label is None:
                    raise ValueError(
                        f"Missing M2 label for episode_id={episode_id} t={t_i}"
                    )
                if not dual_task_text:
                    task_text = _pick_task_text(label, field=task_text_field)
                    if not task_text:
                        raise ValueError(
                            f"Empty task text (task_text_field={task_text_field!r}) for episode_id={episode_id} t={t_i}"
                        )
                else:
                    raw = label.get("prompt_raw")
                    conditioned = label.get("prompt_conditioned")
                    raw_s = raw if isinstance(raw, str) and raw else ""
                    conditioned_s = (
                        conditioned
                        if isinstance(conditioned, str) and conditioned
                        else ""
                    )
                    want_raw = _stable_mix50_pick_raw(
                        episode_id=str(episode_id), t_i=int(t_i)
                    )
                    task_text = raw_s if want_raw else conditioned_s
                    if not task_text:
                        task_text = conditioned_s if want_raw else raw_s
                    if not task_text:
                        raise ValueError(
                            f"Empty task text (prompt_raw/prompt_conditioned) for episode_id={episode_id} t={t_i}"
                        )
                task_index = int(task_to_index[task_text])

                cur_state = state_by_step[step_idx]
                cur_actions = action_by_step[step_idx, :n_exec, :]
                if cur_actions.shape != (n_exec, int(action_dim)):
                    raise ValueError(
                        f"episode_id={episode_id} t={t_i} action slice has wrong shape: {cur_actions.shape}"
                    )

                advantage_input_value: float | None = None
                if include_m2_label_columns:
                    existing_advantage_input = label.get(advantage_input_label_key)
                    raw_advantage_value = label.get(advantage_raw_label_key)
                    label_positive_scale = label.get("positive_scale")
                    label_negative_scale_abs = label.get("negative_scale_abs")
                    label_shared_scale = label.get("p95_abs_advantage")
                    positive_scale_to_use = (
                        advantage_positive_scale
                        if advantage_positive_scale is not None
                        else label_positive_scale
                    )
                    negative_scale_to_use = (
                        advantage_negative_scale_abs
                        if advantage_negative_scale_abs is not None
                        else label_negative_scale_abs
                    )
                    shared_scale_to_use = (
                        advantage_shared_scale
                        if advantage_shared_scale is not None
                        else label_shared_scale
                    )
                    if raw_advantage_value is None:
                        if existing_advantage_input is None:
                            raise ValueError(
                                f"Missing label field {advantage_raw_label_key!r} for episode_id={episode_id} t={t_i}"
                            )
                        advantage_input_value = validate_advantage_input_value(
                            existing_advantage_input,
                            context=(
                                f"labels.jsonl episode_id={episode_id} t={t_i} "
                                f"{advantage_input_label_key}"
                            ),
                        )
                    else:
                        raw_advantage_value_f: float | None
                        try:
                            raw_advantage_value_f = float(raw_advantage_value)
                        except (TypeError, ValueError):
                            raw_advantage_value_f = None
                        if (
                            positive_scale_to_use is None
                            and negative_scale_to_use is None
                            and shared_scale_to_use is None
                        ):
                            if existing_advantage_input is None:
                                if (
                                    raw_advantage_value_f is not None
                                    and abs(raw_advantage_value_f) <= 1e-12
                                ):
                                    advantage_input_value = validate_advantage_input_value(
                                        0.0,
                                        context=(
                                            f"zero-advantage fallback episode_id={episode_id} "
                                            f"t={t_i}"
                                        ),
                                    )
                                else:
                                    raise ValueError(
                                        "Missing sign-aware scales for "
                                        f"episode_id={episode_id} t={t_i}; checked contract "
                                        "and label fields positive_scale / negative_scale_abs / "
                                        "p95_abs_advantage"
                                    )
                            else:
                                advantage_input_value = validate_advantage_input_value(
                                    existing_advantage_input,
                                    context=(
                                        f"labels.jsonl episode_id={episode_id} t={t_i} "
                                        f"{advantage_input_label_key}"
                                    ),
                                )
                        else:
                            advantage_input_value = validate_advantage_input_value(
                                normalize_advantage_to_input(
                                    raw_advantage_value,
                                    p95_abs_advantage=shared_scale_to_use,
                                    positive_scale=positive_scale_to_use,
                                    negative_scale_abs=negative_scale_to_use,
                                ),
                                context=(
                                    f"derived advantage_input episode_id={episode_id} "
                                    f"t={t_i}"
                                ),
                            )
                    if existing_advantage_input is not None:
                        existing_advantage_input_value = validate_advantage_input_value(
                            existing_advantage_input,
                            context=(
                                f"labels.jsonl episode_id={episode_id} t={t_i} "
                                f"{advantage_input_label_key}"
                            ),
                        )
                        if (
                            abs(existing_advantage_input_value - advantage_input_value)
                            > 1e-6
                        ):
                            raise ValueError(
                                "Advantage input mismatch for "
                                f"episode_id={episode_id} t={t_i}: existing "
                                f"{existing_advantage_input_value} vs derived "
                                f"{advantage_input_value}"
                            )

                for inner in range(n_exec):
                    rows_state.append(cur_state)
                    rows_action.append(cur_actions[inner])
                    rows_timestamp.append(float(step_frame_idx) / float(fps))
                    rows_episode_index.append(int(episode_index))
                    rows_index.append(int(step_frame_idx))
                    rows_task_index.append(int(task_index))

                    if include_m2_label_columns:
                        assert advantage_input_value is not None
                        rows_labels.setdefault(ADVANTAGE_INPUT_COLUMN, []).append(
                            float(advantage_input_value)
                        )
                        for lk, lv in label.items():
                            if lk in (
                                "schema_version",
                                "code_version",
                                "iter_tag",
                                "episode_id",
                                advantage_input_label_key,
                            ):
                                continue
                            col = f"recap_m2.{lk}"
                            rows_labels.setdefault(col, []).append(lv)

                    step_frame_idx += 1

            df_dict: dict[str, Any] = {
                "observation.state": rows_state,
                "action": rows_action,
                "timestamp": rows_timestamp,
                "episode_index": rows_episode_index,
                "index": rows_index,
                LANGUAGE_ANNOTATION_KEY: rows_task_index,
            }
            for col, vals in rows_labels.items():
                df_dict[col] = vals

            df = pd.DataFrame(df_dict)

            chunk_idx = int(episode_index) // int(chunk_size)
            chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = chunk_dir / f"episode_{int(episode_index):06d}.parquet"
            df.to_parquet(parquet_path, engine="pyarrow", index=False)

            all_state_rows.append(np.stack(rows_state, axis=0))
            all_action_rows.append(np.stack(rows_action, axis=0))

            # Episode meta.
            episode_tasks = sorted(
                {
                    tasks_sorted[int(i)]
                    for i in set(rows_task_index)
                    if int(i) >= 0 and int(i) < len(tasks_sorted)
                }
            )
            episodes_meta_out.append(
                {
                    "episode_index": int(episode_index),
                    "tasks": episode_tasks,
                    "length": int(len(df)),
                    "recap.episode_id": episode_id,
                }
            )
            total_frames += int(len(df))

    assert state_dim is not None and action_dim is not None
    assert dims_state_lock is not None and dims_action_lock is not None

    # Meta: tasks.jsonl
    tasks_records = [
        {"task_index": int(i), "task": str(t)} for i, t in enumerate(tasks_sorted)
    ]
    _write_jsonl(meta_dir / META_TASKS_JSONL, tasks_records)

    # Meta: episodes.jsonl
    _write_jsonl(meta_dir / META_EPISODES_JSONL, episodes_meta_out)

    # Meta: modality.json
    state_offsets = _build_group_offsets(STATE_KEY_ORDER_LOCK, dims_state_lock)
    action_offsets = _build_group_offsets(ACTION_KEY_ORDER_LOCK, dims_action_lock)

    state_modality: dict[str, dict[str, object]] = {
        WBC_STATE_GROUP_KEY: {"start": 0, "end": int(state_dim)}
    }
    for name, (st, ed) in state_offsets.items():
        state_modality[str(name)] = {
            "start": int(st),
            "end": int(ed),
            "original_key": "observation.state",
        }

    action_modality: dict[str, dict[str, object]] = {
        WBC_ACTION_GROUP_KEY: {"start": 0, "end": int(action_dim)}
    }
    for name, (st, ed) in action_offsets.items():
        action_modality[str(name)] = {
            "start": int(st),
            "end": int(ed),
            "original_key": "action",
        }

    modality_obj = {
        "state": state_modality,
        "action": action_modality,
        "annotation": {
            "human.action.task_description": {"original_key": LANGUAGE_ANNOTATION_KEY},
            "human.task_description": {"original_key": LANGUAGE_ANNOTATION_KEY},
        },
    }
    _write_json(meta_dir / META_MODALITY_JSON, modality_obj)

    # Meta: stats.json (computed from exported vectors).
    state_mat = np.concatenate(all_state_rows, axis=0).astype(np.float32, copy=False)
    action_mat = np.concatenate(all_action_rows, axis=0).astype(np.float32, copy=False)
    if state_mat.ndim != 2 or int(state_mat.shape[1]) != int(state_dim):
        raise ValueError(f"Invalid exported state_mat shape: {state_mat.shape}")
    if action_mat.ndim != 2 or int(action_mat.shape[1]) != int(action_dim):
        raise ValueError(f"Invalid exported action_mat shape: {action_mat.shape}")

    def _stats_for(mat: Any) -> dict[str, list[float]]:
        q01, q99 = np.quantile(mat, [0.01, 0.99], axis=0)
        return {
            "mean": [float(x) for x in mat.mean(axis=0)],
            "std": [float(x) for x in mat.std(axis=0)],
            "min": [float(x) for x in mat.min(axis=0)],
            "max": [float(x) for x in mat.max(axis=0)],
            "q01": [float(x) for x in q01],
            "q99": [float(x) for x in q99],
        }

    stats_obj = {
        "observation.state": _stats_for(state_mat),
        "action": _stats_for(action_mat),
    }
    _write_json(meta_dir / META_STATS_JSON, stats_obj)

    # Meta: info.json
    # Feature names are informational only; keep them stable and reproducible.
    assert dims_state_lock is not None and dims_action_lock is not None
    state_names = _feature_names_for_group(
        STATE_KEY_ORDER_LOCK, dims_state_lock, group="wbc_state"
    )
    action_names = _feature_names_for_group(
        ACTION_KEY_ORDER_LOCK, dims_action_lock, group="wbc_action"
    )
    if len(state_names) != int(state_dim):
        raise ValueError(
            f"state feature names length mismatch: expected {state_dim} got {len(state_names)}"
        )
    if len(action_names) != int(action_dim):
        raise ValueError(
            f"action feature names length mismatch: expected {action_dim} got {len(action_names)}"
        )
    info_obj = {
        "codebase_version": "recap-m3",
        "robot_type": "unitree_g1_wbc",
        "total_episodes": int(len(episodes_meta_out)),
        "total_frames": int(total_frames),
        "total_tasks": int(len(tasks_sorted)),
        "chunks_size": int(chunk_size),
        "fps": float(fps),
        "task_text_field": str(task_text_field),
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [int(action_dim)],
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [int(state_dim)],
                "names": state_names,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            LANGUAGE_ANNOTATION_KEY: {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            LANGUAGE_ANNOTATION_KEY_ALIAS: {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
        "total_chunks": int((len(episodes_meta_out) - 1) // int(chunk_size) + 1),
        "total_videos": 0,
        "field_groups": {
            "state": [WBC_STATE_GROUP_KEY],
            "action": [WBC_ACTION_GROUP_KEY],
            "state_conditioned_sidecar": build_state_conditioned_field_groups(),
        },
    }
    if include_m2_label_columns:
        info_features = info_obj.get("features")
        if not isinstance(info_features, dict):
            raise ValueError("info.json features payload must be a dict")
        info_features.update(
            {
                ADVANTAGE_RETURN_COLUMN: {"dtype": "float32", "shape": [1], "names": None},
                ADVANTAGE_VALUE_COLUMN: {"dtype": "float32", "shape": [1], "names": None},
                ADVANTAGE_RAW_COLUMN: {"dtype": "float32", "shape": [1], "names": None},
                ADVANTAGE_INPUT_COLUMN: {"dtype": "float32", "shape": [1], "names": None},
                "recap_m2.epsilon_l": {"dtype": "float32", "shape": [1], "names": None},
                "recap_m2.indicator_I": {"dtype": "int64", "shape": [1], "names": None},
                "recap_m2.t": {"dtype": "int64", "shape": [1], "names": None},
            }
        )
        if advantage_contract is not None:
            info_obj["recap_advantage_input_contract"] = dict(advantage_contract)
    if not dual_task_text and task_text_field == EXPORTER_MAINLINE_TASK_TEXT_FIELD:
        info_obj.update(_build_mainline_text_provenance(labels_raw))
    if dual_task_text:
        info_obj["recap_export.dual_task_text"] = True
        info_obj["task_text_mode"] = "mix50"
    _write_json(meta_dir / META_INFO_JSON, info_obj)

    return LeRobotV2ExportResult(
        output_dataset_dir=out_dir,
        total_episodes=int(len(episodes_meta_out)),
        total_frames=int(total_frames),
        total_tasks=int(len(tasks_sorted)),
        state_dim=int(state_dim),
        action_dim=int(action_dim),
    )
