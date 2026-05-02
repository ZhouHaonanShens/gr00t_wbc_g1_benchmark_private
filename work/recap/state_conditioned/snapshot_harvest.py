from __future__ import annotations

import argparse
import atexit
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_BUCKET_DIR = Path("agent/artifacts/state_conditioned_materialization/bucket_a")
DEFAULT_DEV_DIR = Path("agent/artifacts/state_conditioned_materialization/devbench")
DEFAULT_COLLECTION_DIR = Path(
    "agent/artifacts/state_conditioned_materialization/collection"
)
DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_materialization/harvest")
DEFAULT_SNAPSHOT_CANDIDATES = DEFAULT_COLLECTION_DIR / "snapshot_candidates.jsonl"
DEFAULT_TEACHER_THRESHOLD = 0.15
DEFAULT_TEACHER_VERSION = "scripted_teacher_v1"
OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME = "snapshot_candidates.jsonl"

FEASIBILITY_REPORT_JSON_NAME = "snapshot_feasibility_report.json"
TEACHER_GATE_REPORT_JSON_NAME = "teacher_gate_report.json"
LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME = "local_recovery_pseudodemo_manifest.json"
LOCAL_RECOVERY_ROLLOUT_ANALYSIS_JSON_NAME = "local_recovery_rollout_analysis.json"
SCHEMA_VERSION = "g1_state_conditioned_snapshot_curriculum_v1"

FORMAL_SUCCESSFUL_PSEUDODEMO_TARGET = 24
FORMAL_HIGH_PRIORITY_FLOOR = 8
PRODUCER_BASE_POLICY = "base_policy"
PRODUCER_SCRIPTED_TEACHER = "scripted_teacher"
DEPLOYABLE_HISTORY_FIELD_NAMES: tuple[str, ...] = (
    "deployable.previous_action_history",
    "deployable.proprio_history",
    "deployable.short_visual_history_refs",
)

SNAPSHOTS_PER_FAMILY = 6
DEPRIORITIZED_LOW_PRIORITY_SNAPSHOTS = 2
SNAPSHOT_SEED_VALUES: tuple[int, ...] = (0, 1, 2)
T8_FAMILY_ORDER: tuple[str, ...] = (
    "S_drop",
    "S_lost",
    "S_transport_mid",
    "S_pre_place",
)
HIGH_PRIORITY_FAMILIES: tuple[str, ...] = ("S_drop", "S_lost")
LOW_PRIORITY_FAMILIES: tuple[str, ...] = ("S_transport_mid", "S_pre_place")
FAMILY_SUCCESS_CRITERIA: dict[str, str] = {
    "S_lost": "apple_visible=true for 4 consecutive policy steps",
    "S_drop": "apple_in_hand=true for 8 consecutive policy steps",
    "S_transport_mid": (
        "apple_in_hand=true for 8 consecutive policy steps and "
        "apple_to_plate_rel_pose.xy_distance improves by at least 0.05m from anchor"
    ),
    "S_pre_place": "success_episode=true or (contact_flag=true and phase enters PLACE)",
}
FAMILY_SOURCE_BUCKET_BY_FAMILY: dict[str, str] = {
    "S_drop": "bucket_C",
    "S_lost": "bucket_C",
    "S_transport_mid": "bucket_B",
    "S_pre_place": "bucket_B",
}
TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT = "truthful_real_teacher_rollout"
TEACHER_TARGET_NOT_APPLICABLE = "not_applicable_base_policy"
TEACHER_TARGET_SYNTHETIC_BACKFILL = "synthetic_observation_only_backfill"
TEACHER_TARGET_UNPROVEN = "unproven_teacher_target"
TEACHER_ROLLOUT_KIND_LOCAL_SIM = "local_sim_rollout"
TEACHER_ROLLOUT_KIND_CACHED_REAL = "cached_real_teacher_rollout"
TEACHER_ROLLOUT_KIND_RECORDED_SOURCE_TRACE = "recorded_source_teacher_trace"
TEACHER_ROLLOUT_KIND_SOURCE_SIDECAR_REPLAY = "source_sidecar_replay"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_bucket_a_sidecar
from work.recap import state_conditioned_collect_buckets
from work.recap import state_conditioned_dev_manifest
from agent.run import state_conditioned_env_resolution
from work.recap import collector as recap_collector
from work.recap import episode_writer as recap_episode_writer
from work.recap import policy as recap_policy
from work.recap.scripts.state_conditioned_common import (
    exception_message as _exception_message,
)
from work.recap.scripts.state_conditioned_common import read_json as _read_json
from work.recap.scripts.state_conditioned_common import (
    read_jsonl_dicts as _read_jsonl_dicts,
)
from work.recap.scripts.state_conditioned_common import (
    validate_existing_dir as _validate_existing_dir,
)
from work.recap.scripts.state_conditioned_common import (
    validate_existing_file as _validate_existing_file,
)
from work.recap.scripts.state_conditioned_common import (
    validate_output_dir as _validate_output_dir,
)
from work.recap.scripts.state_conditioned_common import write_json as _write_json
from work.recap.scripts.state_conditioned_common import write_jsonl as _write_jsonl


FeasibilityRunner = Callable[[Mapping[str, Any], int, str], Mapping[str, Any]]
FormalRolloutRunner = Callable[[Mapping[str, Any], int, str, str], Mapping[str, Any]]

_feasibility_policy_session: dict[str, Any] | None = None
_replay_env_session: dict[str, Any] | None = None
_dataset_sidecar_row_index_cache: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Task 8 snapshot feasibility / TeacherGate, or Task 9 formal local "
            "pseudo-demo harvest with strict state-conditioned provenance rules."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("feasibility", "formal"),
        default="feasibility",
        help=(
            "feasibility writes T8 feasibility + teacher gate reports; formal consumes "
            "those T8 artifacts and writes success-only local_recovery_pseudodemo_manifest.json."
        ),
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=DEFAULT_BUCKET_DIR,
        help="Canonical Bucket A directory containing Gate A and T5 sidecar artifacts.",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=DEFAULT_DEV_DIR,
        help="T6 devbench directory containing fixed_strata_definition.json and baseline artifacts.",
    )
    parser.add_argument(
        "--collection-dir",
        type=Path,
        default=DEFAULT_COLLECTION_DIR,
        help="T7 collection directory containing bucket_B/bucket_C manifests and summary.",
    )
    parser.add_argument(
        "--snapshot-candidates",
        type=Path,
        default=DEFAULT_SNAPSHOT_CANDIDATES,
        help="JSONL candidate source used by feasibility mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives snapshot_feasibility_report.json and teacher_gate_report.json.",
    )
    parser.add_argument(
        "--history-k",
        type=int,
        default=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
        help="Frozen history window length; only the canonical value is accepted.",
    )
    parser.add_argument(
        "--teacher-threshold",
        type=float,
        default=float(DEFAULT_TEACHER_THRESHOLD),
        help="Per-family TeacherGate threshold; fallback is enabled only when success_rate < threshold.",
    )
    parser.add_argument(
        "--teacher-version",
        type=str,
        default=DEFAULT_TEACHER_VERSION,
        help="Teacher provenance version string recorded in formal pseudo-demo manifest.",
    )
    return parser


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    return state_conditioned_bucket_a_import._as_non_empty_string(
        value,
        field_name=field_name,
    )


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_list(
    value: object, *, field_name: str, expected_len: int | None = None
) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    items = list(value)
    if expected_len is not None and len(items) != int(expected_len):
        raise ValueError(
            f"{field_name} must have length {expected_len}, got {len(items)}"
        )
    return items


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _normalize_attempt_records(
    value: object, *, field_name: str
) -> list[dict[str, Any]]:
    return [
        dict(_as_mapping(item, field_name=f"{field_name}[{index}]"))
        for index, item in enumerate(_as_list(value, field_name=field_name))
    ]


def _normalize_family(value: object, *, field_name: str = "family") -> str:
    normalized = _as_non_empty_string(value, field_name=field_name)
    if normalized not in T8_FAMILY_ORDER:
        raise ValueError(f"{field_name} must be one of {T8_FAMILY_ORDER!r}")
    return normalized


def _normalize_phase(value: object, *, field_name: str) -> str:
    normalized = _as_non_empty_string(value, field_name=field_name).upper()
    if normalized not in state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES:
        raise ValueError(
            f"{field_name} must be one of {state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES!r}"
        )
    return normalized


def _priority_for_family(family: str) -> str:
    return "high" if family in HIGH_PRIORITY_FAMILIES else "low"


def _expected_source_bucket_key_for_family(family: object, *, field_name: str) -> str:
    normalized_family = _normalize_family(family, field_name=field_name)
    return FAMILY_SOURCE_BUCKET_BY_FAMILY[normalized_family]


def _normalize_source_bucket_key(
    value: object,
    *,
    family: object,
    field_name: str,
) -> str:
    normalized = _as_non_empty_string(value, field_name=field_name)
    expected = _expected_source_bucket_key_for_family(
        family,
        field_name=f"{field_name}.family",
    )
    if normalized != expected:
        raise ValueError(
            f"{field_name} mismatch for family {family!r}: expected {expected!r}, got {normalized!r}"
        )
    return normalized


def _extract_xy_distance(value: object, *, field_name: str) -> float:
    if isinstance(value, Mapping):
        x = _as_number(value.get("x"), field_name=f"{field_name}.x")
        y = _as_number(value.get("y"), field_name=f"{field_name}.y")
        return (x**2 + y**2) ** 0.5
    pose = _as_list(value, field_name=field_name)
    if len(pose) < 2:
        raise ValueError(f"{field_name} must carry at least x/y components")
    x = _as_number(pose[0], field_name=f"{field_name}[0]")
    y = _as_number(pose[1], field_name=f"{field_name}[1]")
    return (x**2 + y**2) ** 0.5


def _load_t8_prerequisites(
    *,
    bucket_dir: Path,
    dev_dir: Path,
    collection_dir: Path,
) -> dict[str, Any]:
    bucket_dir = _validate_existing_dir(bucket_dir, arg_name="bucket-dir")
    dev_dir = _validate_existing_dir(dev_dir, arg_name="dev-dir")
    collection_dir = _validate_existing_dir(collection_dir, arg_name="collection-dir")

    bucket_paths = {
        "gate_path": _validate_existing_file(
            bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME,
            arg_name="T5 gate_path",
        ),
        "manifest_path": _validate_existing_file(
            bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME,
            arg_name="T5 manifest_path",
        ),
        "sidecar_path": _validate_existing_file(
            bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_SIDECAR_JSON_NAME,
            arg_name="T5 sidecar_path",
        ),
        "join_coverage_path": _validate_existing_file(
            bucket_dir
            / state_conditioned_bucket_a_sidecar.BUCKET_A_JOIN_COVERAGE_JSON_NAME,
            arg_name="T5 join_coverage_path",
        ),
        "exporter_manifest_path": _validate_existing_file(
            bucket_dir
            / state_conditioned_bucket_a_sidecar.BUCKET_A_EXPORTER_MANIFEST_JSON_NAME,
            arg_name="T5 exporter_manifest_path",
        ),
    }
    gate = _read_json(bucket_paths["gate_path"])
    if not bool(gate.get("ready", False)):
        raise ValueError(
            "snapshot feasibility refuses to run until bucket_A_gate_a_ready.json.ready == true"
        )

    dev_paths = {
        "fixed_strata_definition_path": _validate_existing_file(
            dev_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME,
            arg_name="T6 fixed_strata_definition_path",
        ),
        "baseline_manifest_path": _validate_existing_file(
            dev_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME,
            arg_name="T6 baseline_manifest_path",
        ),
        "baseline_dev_scorecard_path": _validate_existing_file(
            dev_dir / state_conditioned_dev_manifest.BASELINE_DEV_SCORECARD_JSON_NAME,
            arg_name="T6 baseline_dev_scorecard_path",
        ),
    }
    fixed_strata_definition = _read_json(dev_paths["fixed_strata_definition_path"])
    baseline_dev_scorecard = _read_json(dev_paths["baseline_dev_scorecard_path"])
    paired_seed_count = _as_int(
        fixed_strata_definition.get("paired_seed_count"),
        field_name="fixed_strata_definition.paired_seed_count",
    )
    if paired_seed_count != len(state_conditioned_dev_manifest.DEFAULT_PAIRED_SEEDS):
        raise ValueError(
            "T6 fixed_strata_definition paired_seed_count mismatch: "
            + f"expected {len(state_conditioned_dev_manifest.DEFAULT_PAIRED_SEEDS)}, got {paired_seed_count}"
        )
    requested_entries = _as_int(
        _as_mapping(
            baseline_dev_scorecard.get("counts"),
            field_name="baseline_dev_scorecard.counts",
        ).get("requested_entries"),
        field_name="baseline_dev_scorecard.counts.requested_entries",
    )
    if requested_entries != int(
        sum(state_conditioned_dev_manifest.EXPECTED_STRATA_COUNTS.values())
    ):
        raise ValueError(
            "T6 baseline_dev_scorecard requested_entries mismatch: "
            + f"expected 32, got {requested_entries}"
        )

    collection_paths = {
        "bucket_B_manifest_path": _validate_existing_file(
            collection_dir
            / state_conditioned_collect_buckets.BUCKET_B_MANIFEST_JSON_NAME,
            arg_name="T7 bucket_B_manifest_path",
        ),
        "bucket_C_manifest_path": _validate_existing_file(
            collection_dir
            / state_conditioned_collect_buckets.BUCKET_C_MANIFEST_JSON_NAME,
            arg_name="T7 bucket_C_manifest_path",
        ),
        "bucket_collection_summary_path": _validate_existing_file(
            collection_dir
            / state_conditioned_collect_buckets.BUCKET_COLLECTION_SUMMARY_JSON_NAME,
            arg_name="T7 bucket_collection_summary_path",
        ),
    }
    bucket_c_manifest = _read_json(collection_paths["bucket_C_manifest_path"])
    summary = _read_json(collection_paths["bucket_collection_summary_path"])
    summary_counts = _as_mapping(summary.get("counts"), field_name="summary.counts")
    bucket_b_count = _as_int(
        summary_counts.get("bucket_B"), field_name="summary.counts.bucket_B"
    )
    bucket_c_count = _as_int(
        summary_counts.get("bucket_C"), field_name="summary.counts.bucket_C"
    )
    if bucket_b_count != int(state_conditioned_collect_buckets.DEFAULT_BUCKET_B_TARGET):
        raise ValueError(
            f"T7 bucket_B count mismatch: expected 16, got {bucket_b_count}"
        )
    if bucket_c_count != int(state_conditioned_collect_buckets.DEFAULT_BUCKET_C_TARGET):
        raise ValueError(
            f"T7 bucket_C count mismatch: expected 24, got {bucket_c_count}"
        )
    bucket_c_per_family = _as_mapping(
        summary_counts.get("bucket_C_per_failure_family"),
        field_name="summary.counts.bucket_C_per_failure_family",
    )
    normalized_bucket_c_per_family = {
        _as_non_empty_string(
            key, field_name="bucket_C_per_failure_family.key"
        ): _as_int(
            value,
            field_name=f"bucket_C_per_failure_family[{key}]",
        )
        for key, value in bucket_c_per_family.items()
    }
    expected_bucket_c_per_family = {
        family: 8
        for family in state_conditioned_collect_buckets.REQUIRED_FAILURE_FAMILIES
    }
    if normalized_bucket_c_per_family != expected_bucket_c_per_family:
        raise ValueError(
            "T7 bucket_C per_failure_family mismatch: "
            + json.dumps(
                normalized_bucket_c_per_family, ensure_ascii=True, sort_keys=True
            )
        )
    manifest_counts = _as_mapping(
        bucket_c_manifest.get("counts"),
        field_name="bucket_C_manifest.counts",
    )
    if _as_int(
        manifest_counts.get("episodes"),
        field_name="bucket_C_manifest.counts.episodes",
    ) != int(state_conditioned_collect_buckets.DEFAULT_BUCKET_C_TARGET):
        raise ValueError("T7 bucket_C manifest episodes mismatch")

    return {
        "bucket_dir": str(bucket_dir),
        "dev_dir": str(dev_dir),
        "collection_dir": str(collection_dir),
        **{name: str(path) for name, path in bucket_paths.items()},
        **{name: str(path) for name, path in dev_paths.items()},
        **{name: str(path) for name, path in collection_paths.items()},
    }


def _candidate_validation_result(
    candidate: Mapping[str, Any],
) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        family = _normalize_family(
            candidate.get("family"), field_name="candidate.family"
        )
        snapshot_id = _as_non_empty_string(
            candidate.get("snapshot_id"),
            field_name="candidate.snapshot_id",
        )
        anchor_episode_id = candidate.get(
            "anchor_episode_id",
            candidate.get("episode_id"),
        )
        anchor_t = _as_int(
            candidate.get("anchor_t", candidate.get("t")),
            field_name="candidate.anchor_t",
        )
        policy_phase = candidate.get("policy_condition.phase", candidate.get("phase"))
        policy_mode = candidate.get("policy_condition.mode", candidate.get("mode"))
        policy_text = candidate.get(
            "policy_condition_text",
            state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
                policy_phase,
                policy_mode,
            ),
        )
        normalized_phase, normalized_mode, normalized_text = (
            state_conditioned_bucket_a_import.validate_state_conditioned_policy_condition(
                phase=policy_phase,
                mode=policy_mode,
                policy_condition_text=policy_text,
            )
        )
        history = state_conditioned_bucket_a_import.validate_state_conditioned_history_contract(
            anchor_episode_id=anchor_episode_id,
            history_episode_ids=candidate.get("history_episode_ids"),
            history_valid_mask=candidate.get("history_valid_mask"),
            anchor_mujoco_state_ref=candidate.get("anchor_mujoco_state_ref"),
            prehistory_window=candidate.get("prehistory_window"),
            history_k=candidate.get("history_k"),
            history_stride=candidate.get("history_stride"),
            reset_boundary=candidate.get("reset_boundary"),
        )
        normalized_candidate: dict[str, Any] = {
            "family": family,
            "snapshot_id": snapshot_id,
            "anchor_t": int(anchor_t),
            "policy_condition.phase": normalized_phase,
            "policy_condition.mode": normalized_mode,
            "policy_condition_text": normalized_text,
            "deprioritized_by_plan": bool(
                candidate.get("deprioritized_by_plan", False)
            ),
            **history,
        }
        if family == "S_transport_mid":
            if candidate.get("anchor_xy_distance") is not None:
                normalized_candidate["anchor_xy_distance"] = _as_number(
                    candidate.get("anchor_xy_distance"),
                    field_name="candidate.anchor_xy_distance",
                )
            else:
                normalized_candidate["anchor_xy_distance"] = _extract_xy_distance(
                    candidate.get("privileged.apple_to_plate_rel_pose"),
                    field_name="candidate.privileged.apple_to_plate_rel_pose",
                )
        for field_name in (
            "source_dataset_dir",
            "source_npz_path",
            "source_env_name",
            "source_model_path",
            "source_embodiment_tag",
            "source_failure_injection_kind",
        ):
            raw_value = candidate.get(field_name)
            if raw_value is not None:
                normalized_candidate[field_name] = _as_non_empty_string(
                    raw_value,
                    field_name=f"candidate.{field_name}",
                )
        raw_source_bucket_key = candidate.get("source_bucket_key")
        if raw_source_bucket_key is not None:
            normalized_candidate["source_bucket_key"] = _normalize_source_bucket_key(
                raw_source_bucket_key,
                family=family,
                field_name="candidate.source_bucket_key",
            )
        elif (
            candidate.get("anchor_episode_id", candidate.get("episode_id")) is not None
        ):
            normalized_candidate["source_bucket_key"] = (
                _expected_source_bucket_key_for_family(
                    family,
                    field_name="candidate.family",
                )
            )
        if candidate.get("source_episode_seed") is not None:
            normalized_candidate["source_episode_seed"] = _as_int(
                candidate.get("source_episode_seed"),
                field_name="candidate.source_episode_seed",
            )
        deployable_history_payload = _resolve_candidate_deployable_history_payload(
            candidate,
            field_prefix="candidate",
        )
        if deployable_history_payload is not None:
            normalized_candidate.update(deployable_history_payload)
        if candidate.get("expected_success_count") is not None:
            normalized_candidate["expected_success_count"] = _as_int(
                candidate.get("expected_success_count"),
                field_name="candidate.expected_success_count",
            )
        attempts = candidate.get("attempts")
        if attempts is not None:
            normalized_candidate["attempts"] = _normalize_attempt_records(
                attempts,
                field_name="candidate.attempts",
            )
        teacher_attempts = candidate.get(
            "scripted_teacher_attempts",
            candidate.get("teacher_attempts"),
        )
        if teacher_attempts is not None:
            normalized_candidate["scripted_teacher_attempts"] = (
                _normalize_attempt_records(
                    teacher_attempts,
                    field_name="candidate.scripted_teacher_attempts",
                )
            )
        return True, normalized_candidate, None
    except (TypeError, ValueError) as exc:
        snapshot_id = str(candidate.get("snapshot_id", "")).strip() or None
        fallback_snapshot_id = (
            snapshot_id if snapshot_id is not None else "unknown_snapshot"
        )
        return (
            False,
            {
                "family": str(candidate.get("family", "")),
                "snapshot_id": fallback_snapshot_id,
                "deprioritized_by_plan": bool(
                    candidate.get("deprioritized_by_plan", False)
                ),
            },
            _exception_message(exc),
        )


def load_snapshot_candidates(
    snapshot_candidates_path: Path,
) -> dict[str, dict[str, Any]]:
    path = _validate_existing_file(
        snapshot_candidates_path,
        arg_name="snapshot_candidates",
    )
    grouped: dict[str, dict[str, Any]] = {
        family: {
            "eligible": [],
            "ineligible": [],
            "deprioritized_flags": set(),
        }
        for family in T8_FAMILY_ORDER
    }
    for raw_candidate in _read_jsonl_dicts(path):
        is_eligible, payload, error = _candidate_validation_result(raw_candidate)
        if payload is None:
            continue
        family = str(payload.get("family", "")).strip()
        if family not in grouped:
            raise ValueError(f"snapshot candidate family is invalid: {family!r}")
        grouped[family]["deprioritized_flags"].add(
            bool(payload.get("deprioritized_by_plan", False))
        )
        if is_eligible:
            grouped[family]["eligible"].append(dict(payload))
        else:
            grouped[family]["ineligible"].append(
                {
                    "snapshot_id": payload["snapshot_id"],
                    "reason": error,
                }
            )
    for family in T8_FAMILY_ORDER:
        grouped[family]["eligible"] = sorted(
            grouped[family]["eligible"],
            key=lambda item: str(item["snapshot_id"]),
        )
        grouped[family]["ineligible"] = sorted(
            grouped[family]["ineligible"],
            key=lambda item: str(item["snapshot_id"]),
        )
    return grouped


def _normalize_grouped_snapshot_candidates(
    grouped_snapshot_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized_grouped: dict[str, dict[str, Any]] = {}
    for family in T8_FAMILY_ORDER:
        raw_group = grouped_snapshot_candidates.get(family)
        if raw_group is None:
            raise ValueError(
                f"missing grouped snapshot candidates for family {family!r}"
            )
        group = dict(raw_group)
        normalized_eligible: list[dict[str, Any]] = []
        for raw_candidate in list(group.get("eligible", [])):
            is_eligible, payload, error = _candidate_validation_result(
                _as_mapping(raw_candidate, field_name=f"{family}.eligible[]")
            )
            if not is_eligible or payload is None:
                raise ValueError(
                    f"grouped eligible candidate for {family} is invalid: {error or 'unknown error'}"
                )
            normalized_eligible.append(dict(payload))
        normalized_ineligible = [
            dict(_as_mapping(raw, field_name=f"{family}.ineligible[]"))
            for raw in list(group.get("ineligible", []))
        ]
        normalized_grouped[family] = {
            "eligible": sorted(
                normalized_eligible,
                key=lambda item: str(item["snapshot_id"]),
            ),
            "ineligible": sorted(
                normalized_ineligible,
                key=lambda item: str(item.get("snapshot_id", "")),
            ),
            "deprioritized_flags": set(group.get("deprioritized_flags", set())),
        }
    return normalized_grouped


def _snapshot_candidates_artifact_path(output_dir: Path) -> Path:
    return output_dir / OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME


def _effective_snapshot_candidates_path(
    *,
    snapshot_candidates_path: Path,
    collection_dir: Path,
) -> Path:
    raw_path = snapshot_candidates_path.expanduser()
    default_relative = DEFAULT_SNAPSHOT_CANDIDATES
    default_absolute = (REPO_ROOT / DEFAULT_SNAPSHOT_CANDIDATES).resolve()
    if raw_path == default_relative or raw_path.resolve() == default_absolute:
        return (
            collection_dir.expanduser().resolve()
            / OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
        )
    return raw_path.resolve()


def _is_default_snapshot_candidates_request(
    *,
    snapshot_candidates_path: Path,
    collection_dir: Path,
) -> bool:
    raw_path = snapshot_candidates_path.expanduser()
    default_relative = DEFAULT_SNAPSHOT_CANDIDATES
    default_absolute = (REPO_ROOT / DEFAULT_SNAPSHOT_CANDIDATES).resolve()
    collection_default = (
        collection_dir.expanduser().resolve()
        / OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
    )
    try:
        resolved_raw = raw_path.resolve()
    except FileNotFoundError:
        resolved_raw = raw_path.absolute()
    return (
        raw_path == default_relative
        or resolved_raw == default_absolute
        or resolved_raw == collection_default
    )


def _canonical_manifest_episode_entries(bucket_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME
    manifest = _read_json(
        _validate_existing_file(manifest_path, arg_name="bucket manifest")
    )
    episodes = _as_list(
        manifest.get("episodes"), field_name="bucket_A_manifest.episodes"
    )
    by_episode_id: dict[str, dict[str, Any]] = {}
    for index, raw_episode in enumerate(episodes):
        episode = dict(
            _as_mapping(raw_episode, field_name=f"bucket_A_manifest.episodes[{index}]")
        )
        episode_id = _as_non_empty_string(
            episode.get("episode_id"),
            field_name=f"bucket_A_manifest.episodes[{index}].episode_id",
        )
        if not bool(episode.get("accepted", False)):
            continue
        if bool(episode.get("debug_only", False)):
            continue
        if bool(episode.get("reused_existing_live_dataset", False)):
            continue
        source_dataset_dir = _as_non_empty_string(
            episode.get("source_dataset_dir"),
            field_name=f"bucket_A_manifest.episodes[{index}].source_dataset_dir",
        )
        npz_path = _as_non_empty_string(
            episode.get("npz_path"),
            field_name=f"bucket_A_manifest.episodes[{index}].npz_path",
        )
        by_episode_id[episode_id] = {
            **episode,
            "episode_id": episode_id,
            "source_dataset_dir": source_dataset_dir,
            "npz_path": npz_path,
        }
    if not by_episode_id:
        raise ValueError(
            "bucket_A_manifest.json does not expose canonical accepted episodes"
        )
    return by_episode_id


def _dataset_episode_record(dataset_dir: Path, *, episode_id: str) -> dict[str, Any]:
    episodes_path = _validate_existing_file(
        dataset_dir / "episodes.jsonl", arg_name="dataset episodes"
    )
    for index, raw_record in enumerate(_read_jsonl_dicts(episodes_path)):
        record = dict(_as_mapping(raw_record, field_name=f"episodes.jsonl[{index}]"))
        if str(record.get("episode_id", "")).strip() == str(episode_id):
            return record
    raise ValueError(
        f"dataset {dataset_dir} is missing episode_id={episode_id!r} in episodes.jsonl"
    )


def _dataset_sidecar_rows(
    dataset_dir: Path, *, episode_id: str
) -> list[dict[str, Any]]:
    sidecar_path = _validate_existing_file(
        dataset_dir / "state_conditioned_sidecar.jsonl",
        arg_name="dataset state_conditioned_sidecar",
    )
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(_read_jsonl_dicts(sidecar_path)):
        row = dict(
            _as_mapping(raw_row, field_name=f"state_conditioned_sidecar[{index}]")
        )
        if str(row.get("episode_id", "")).strip() != str(episode_id):
            continue
        rows.append(row)
    rows = sorted(rows, key=lambda item: int(item.get("t", -1)))
    if not rows:
        raise ValueError(
            f"dataset {dataset_dir} is missing sidecar rows for episode_id={episode_id!r}"
        )
    return rows


def _normalize_deployable_history_payload(
    row: Mapping[str, Any], *, field_prefix: str
) -> dict[str, list[Any]]:
    history_k = _as_int(row.get("history_k"), field_name=f"{field_prefix}.history_k")
    raw_valid_mask = _as_list(
        row.get("history_valid_mask"),
        field_name=f"{field_prefix}.history_valid_mask",
        expected_len=history_k,
    )
    valid_mask = [
        _as_bool(value, field_name=f"{field_prefix}.history_valid_mask[{index}]")
        for index, value in enumerate(raw_valid_mask)
    ]
    payload: dict[str, list[Any]] = {}
    for field_name in DEPLOYABLE_HISTORY_FIELD_NAMES:
        lane = _as_list(
            row.get(field_name),
            field_name=f"{field_prefix}.{field_name}",
            expected_len=history_k,
        )
        normalized_lane: list[Any] = []
        for index, is_valid in enumerate(valid_mask):
            item = lane[index]
            if not bool(is_valid):
                normalized_lane.append(None)
                continue
            if item is None:
                raise ValueError(
                    f"{field_prefix}.{field_name}[{index}] must be present when history_valid_mask[{index}] is true"
                )
            normalized_lane.append(item)
        payload[field_name] = normalized_lane
    return payload


def _dataset_sidecar_row_index(
    dataset_dir: Path, *, episode_id: str
) -> dict[int, dict[str, Any]]:
    cache_key = (str(dataset_dir), str(episode_id))
    cached = _dataset_sidecar_row_index_cache.get(cache_key)
    if cached is not None:
        return cached
    indexed_rows: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(
        _dataset_sidecar_rows(dataset_dir, episode_id=episode_id)
    ):
        row_t = _as_int(
            row.get("t", row.get("anchor_t")),
            field_name=f"state_conditioned_sidecar[{index}].t",
        )
        if row_t in indexed_rows:
            raise ValueError(
                f"dataset {dataset_dir} has duplicate state_conditioned_sidecar row for episode_id={episode_id!r}, t={row_t}"
            )
        indexed_rows[row_t] = dict(row)
    _dataset_sidecar_row_index_cache[cache_key] = indexed_rows
    return indexed_rows


def _load_deployable_history_payload_from_dataset_sidecar(
    dataset_dir: Path, *, episode_id: str, anchor_t: int
) -> dict[str, list[Any]]:
    indexed_rows = _dataset_sidecar_row_index(dataset_dir, episode_id=episode_id)
    matched_row = indexed_rows.get(int(anchor_t))
    if matched_row is None:
        raise ValueError(
            f"dataset {dataset_dir} is missing state_conditioned_sidecar row for episode_id={episode_id!r}, t={int(anchor_t)}"
        )
    return _normalize_deployable_history_payload(
        matched_row,
        field_prefix=f"state_conditioned_sidecar[{episode_id}@{int(anchor_t)}]",
    )


def _resolve_candidate_deployable_history_payload(
    candidate: Mapping[str, Any], *, field_prefix: str
) -> dict[str, list[Any]] | None:
    if all(
        candidate.get(field_name) is not None
        for field_name in DEPLOYABLE_HISTORY_FIELD_NAMES
    ):
        return _normalize_deployable_history_payload(
            candidate, field_prefix=field_prefix
        )
    source_dataset_dir = candidate.get("source_dataset_dir")
    if source_dataset_dir is None:
        return None
    dataset_dir = (
        Path(
            _as_non_empty_string(
                source_dataset_dir,
                field_name=f"{field_prefix}.source_dataset_dir",
            )
        )
        .expanduser()
        .resolve()
    )
    anchor_episode_id = _as_non_empty_string(
        candidate.get("anchor_episode_id", candidate.get("episode_id")),
        field_name=f"{field_prefix}.anchor_episode_id",
    )
    anchor_t = _as_int(
        candidate.get("anchor_t", candidate.get("t")),
        field_name=f"{field_prefix}.anchor_t",
    )
    return _load_deployable_history_payload_from_dataset_sidecar(
        dataset_dir,
        episode_id=anchor_episode_id,
        anchor_t=anchor_t,
    )


def _family_policy_target(family: str) -> tuple[str, str]:
    normalized_family = _normalize_family(family)
    if normalized_family == "S_pre_place":
        return "PLACE", "RECOVERY"
    return "TRANSPORT", "RECOVERY"


def _history_contract_from_episode_t(
    *, episode_id: str, anchor_t: int
) -> dict[str, Any]:
    history_k = int(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K)
    history_stride = int(
        state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_STRIDE
    )
    valid_mask: list[bool] = []
    history_episode_ids: list[str] = []
    prehistory_window: list[dict[str, Any]] = []
    start_t = int(anchor_t) - (history_k - 1) * history_stride
    for index in range(history_k):
        candidate_t = start_t + index * history_stride
        is_valid = int(candidate_t) >= 0
        row_t = int(candidate_t) if is_valid else 0
        valid_mask.append(bool(is_valid))
        history_episode_ids.append(str(episode_id))
        prehistory_window.append(
            {
                "episode_id": str(episode_id),
                "t_std": int(row_t),
                "mujoco_state_ref": f"mujoco://{episode_id}/{int(row_t)}",
            }
        )
    return {
        "history_k": int(history_k),
        "history_stride": int(history_stride),
        "history_valid_mask": valid_mask,
        "history_episode_ids": history_episode_ids,
        "anchor_episode_id": str(episode_id),
        "anchor_mujoco_state_ref": f"mujoco://{episode_id}/{int(anchor_t)}",
        "prehistory_window": prehistory_window,
        "reset_boundary": state_conditioned_bucket_a_import.STATE_CONDITIONED_RESET_BOUNDARY,
    }


def _manifest_episode_dataset_dir(
    raw_episode: Mapping[str, Any], *, field_name: str
) -> Path:
    dataset_dir_raw = raw_episode.get("dataset_dir")
    if dataset_dir_raw is None:
        provenance = raw_episode.get("provenance")
        if provenance is not None:
            provenance_mapping = _as_mapping(
                provenance, field_name=f"{field_name}.provenance"
            )
            dataset_dir_raw = provenance_mapping.get("source_dataset_dir")
    if dataset_dir_raw is None:
        raise ValueError(f"{field_name} is missing dataset_dir/source_dataset_dir")
    return (
        Path(
            _as_non_empty_string(
                dataset_dir_raw, field_name=f"{field_name}.dataset_dir"
            )
        )
        .expanduser()
        .resolve()
    )


def _collection_manifest_episodes(
    collection_dir: Path,
    *,
    manifest_name: str,
) -> list[dict[str, Any]]:
    manifest_path = _validate_existing_file(
        collection_dir / manifest_name, arg_name=manifest_name
    )
    manifest = _read_json(manifest_path)
    episodes: list[dict[str, Any]] = []
    for index, raw_episode in enumerate(
        _as_list(manifest.get("episodes"), field_name=f"{manifest_name}.episodes")
    ):
        episode = dict(
            _as_mapping(raw_episode, field_name=f"{manifest_name}.episodes[{index}]")
        )
        dataset_dir = _manifest_episode_dataset_dir(
            episode,
            field_name=f"{manifest_name}.episodes[{index}]",
        )
        episode_id = _as_non_empty_string(
            episode.get("episode_id"),
            field_name=f"{manifest_name}.episodes[{index}].episode_id",
        )
        dataset_episode = _dataset_episode_record(dataset_dir, episode_id=episode_id)
        episodes.append(
            {
                **episode,
                "episode_id": episode_id,
                "dataset_dir": str(dataset_dir),
                "dataset_episode": dataset_episode,
            }
        )
    return episodes


def _candidate_from_manifest_episode(
    *,
    family: str,
    raw_episode: Mapping[str, Any],
    anchor_t: int,
    source_bucket_key: str,
    source_failure_injection_kind: str | None,
    deprioritized_by_plan: bool,
    anchor_xy_distance: float | None = None,
) -> dict[str, Any]:
    dataset_episode = dict(
        _as_mapping(
            raw_episode.get("dataset_episode"),
            field_name="manifest episode.dataset_episode",
        )
    )
    episode_id = _as_non_empty_string(
        raw_episode.get("episode_id"), field_name="manifest episode.episode_id"
    )
    dataset_dir = (
        Path(
            _as_non_empty_string(
                raw_episode.get("dataset_dir"),
                field_name="manifest episode.dataset_dir",
            )
        )
        .expanduser()
        .resolve()
    )
    phase, mode = _family_policy_target(family)
    deployable_history_payload = _load_deployable_history_payload_from_dataset_sidecar(
        dataset_dir,
        episode_id=episode_id,
        anchor_t=int(anchor_t),
    )
    candidate = {
        "family": str(family),
        "snapshot_id": f"{family}__{episode_id}__t{int(anchor_t):03d}",
        "anchor_t": int(anchor_t),
        **_history_contract_from_episode_t(
            episode_id=episode_id, anchor_t=int(anchor_t)
        ),
        **deployable_history_payload,
        "policy_condition.phase": phase,
        "policy_condition.mode": mode,
        "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            phase,
            mode,
        ),
        "deprioritized_by_plan": bool(deprioritized_by_plan),
        "source_dataset_dir": str(dataset_dir),
        "source_npz_path": str(
            dataset_dir
            / _as_non_empty_string(
                dataset_episode.get("npz_path"), field_name="dataset_episode.npz_path"
            )
        ),
        "source_episode_seed": _as_int(
            dataset_episode.get("seed"), field_name="dataset_episode.seed"
        ),
        "source_env_name": _as_non_empty_string(
            dataset_episode.get("env_name"), field_name="dataset_episode.env_name"
        ),
        "source_model_path": _as_non_empty_string(
            dataset_episode.get("model_path"), field_name="dataset_episode.model_path"
        ),
        "source_embodiment_tag": _as_non_empty_string(
            dataset_episode.get("embodiment_tag"),
            field_name="dataset_episode.embodiment_tag",
        ),
        "source_bucket_key": str(source_bucket_key),
    }
    if source_failure_injection_kind is not None:
        candidate["source_failure_injection_kind"] = _as_non_empty_string(
            source_failure_injection_kind,
            field_name="source_failure_injection_kind",
        )
    if anchor_xy_distance is not None:
        candidate["anchor_xy_distance"] = float(anchor_xy_distance)
    return candidate


def _normalize_teacher_trace_t_range(value: object, *, field_name: str) -> list[int]:
    raw_range = _as_list(value, field_name=field_name, expected_len=2)
    start_t = _as_int(raw_range[0], field_name=f"{field_name}[0]")
    end_t = _as_int(raw_range[1], field_name=f"{field_name}[1]")
    if end_t < start_t:
        raise ValueError(f"{field_name} must satisfy end_t >= start_t")
    return [int(start_t), int(end_t)]


def _normalize_teacher_target_payload(
    value: object,
    *,
    teacher_version: str,
    field_name: str,
) -> dict[str, Any]:
    target = dict(_as_mapping(value, field_name=field_name))
    normalized = {
        "trace_episode_id": _as_non_empty_string(
            target.get("trace_episode_id"),
            field_name=f"{field_name}.trace_episode_id",
        ),
        "trace_t_range": _normalize_teacher_trace_t_range(
            target.get("trace_t_range"),
            field_name=f"{field_name}.trace_t_range",
        ),
        "producer": _as_non_empty_string(
            target.get("producer"),
            field_name=f"{field_name}.producer",
        ),
        "synthetic_observation_only_backfill": _as_bool(
            target.get("synthetic_observation_only_backfill"),
            field_name=f"{field_name}.synthetic_observation_only_backfill",
        ),
    }
    if normalized["producer"] != str(teacher_version):
        raise ValueError(
            "teacher_target.producer must match teacher_version: "
            + f"expected {teacher_version!r}, got {normalized['producer']!r}"
        )
    return normalized


def _resolve_teacher_target_payload(
    *,
    attempt_result: Mapping[str, Any],
    producer: str,
    teacher_version: str,
    split_summary: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if str(producer) != PRODUCER_SCRIPTED_TEACHER:
        return None, TEACHER_TARGET_NOT_APPLICABLE

    expected_trace_episode_id = _as_non_empty_string(
        split_summary.get("recovery_suffix_source_episode_id"),
        field_name="split_summary.recovery_suffix_source_episode_id",
    )
    expected_trace_t_range = _normalize_teacher_trace_t_range(
        split_summary.get("recovery_suffix_source_t_range"),
        field_name="split_summary.recovery_suffix_source_t_range",
    )
    raw_teacher_target = attempt_result.get("teacher_target")
    if raw_teacher_target is not None:
        teacher_target = _normalize_teacher_target_payload(
            raw_teacher_target,
            teacher_version=teacher_version,
            field_name="attempt_result.teacher_target",
        )
        if teacher_target["trace_episode_id"] != expected_trace_episode_id:
            raise ValueError(
                "teacher_target.trace_episode_id must match recovery suffix source episode"
            )
        if teacher_target["trace_t_range"] != expected_trace_t_range:
            raise ValueError(
                "teacher_target.trace_t_range must match recovery suffix trace"
            )
        if bool(teacher_target["synthetic_observation_only_backfill"]):
            raise ValueError(
                "scripted teacher target is not truthful: synthetic_observation_only_backfill=true"
            )
        return teacher_target, TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT

    teacher_rollout_kind = str(attempt_result.get("teacher_rollout_kind", "")).strip()
    if teacher_rollout_kind in {
        TEACHER_ROLLOUT_KIND_LOCAL_SIM,
        TEACHER_ROLLOUT_KIND_CACHED_REAL,
        TEACHER_ROLLOUT_KIND_RECORDED_SOURCE_TRACE,
    }:
        return (
            {
                "trace_episode_id": expected_trace_episode_id,
                "trace_t_range": list(expected_trace_t_range),
                "producer": str(teacher_version),
                "synthetic_observation_only_backfill": False,
            },
            TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT,
        )
    if teacher_rollout_kind == TEACHER_ROLLOUT_KIND_SOURCE_SIDECAR_REPLAY:
        raise ValueError(
            "scripted teacher target is not truthful: source sidecar replay is observation-only backfill"
        )
    raise ValueError("scripted teacher target truthfulness is unproven")


def synthesize_snapshot_candidates_from_real_artifacts(
    *,
    bucket_dir: Path,
    collection_dir: Path,
) -> dict[str, dict[str, Any]]:
    _ = _validate_existing_dir(collection_dir, arg_name="collection-dir")
    bucket_b_episodes = _collection_manifest_episodes(
        collection_dir,
        manifest_name=state_conditioned_collect_buckets.BUCKET_B_MANIFEST_JSON_NAME,
    )
    bucket_c_episodes = _collection_manifest_episodes(
        collection_dir,
        manifest_name=state_conditioned_collect_buckets.BUCKET_C_MANIFEST_JSON_NAME,
    )
    drop_episodes = [
        episode
        for episode in bucket_c_episodes
        if str(episode.get("failure_injection_kind", "")).strip()
        == "drop_during_transport"
    ]
    lost_episodes = [
        episode
        for episode in bucket_c_episodes
        if str(episode.get("failure_injection_kind", "")).strip()
        in {"failed_grasp_visible", "failed_grasp_occluded"}
    ]
    transport_episodes = list(bucket_b_episodes[: int(SNAPSHOTS_PER_FAMILY)])
    pre_place_episodes = list(
        bucket_b_episodes[int(SNAPSHOTS_PER_FAMILY) : int(SNAPSHOTS_PER_FAMILY * 2)]
    )
    if len(drop_episodes) < int(SNAPSHOTS_PER_FAMILY):
        raise ValueError(
            "bucket_C manifest does not contain enough drop_during_transport episodes"
        )
    if len(lost_episodes) < int(SNAPSHOTS_PER_FAMILY):
        raise ValueError(
            "bucket_C manifest does not contain enough failed_grasp episodes"
        )
    if len(transport_episodes) < int(SNAPSHOTS_PER_FAMILY):
        raise ValueError(
            "bucket_B manifest does not contain enough transport candidate episodes"
        )
    if len(pre_place_episodes) < int(SNAPSHOTS_PER_FAMILY):
        raise ValueError(
            "bucket_B manifest does not contain enough pre-place candidate episodes"
        )

    grouped: dict[str, dict[str, Any]] = {
        family: {"eligible": [], "ineligible": [], "deprioritized_flags": set()}
        for family in T8_FAMILY_ORDER
    }

    for episode in drop_episodes[: int(SNAPSHOTS_PER_FAMILY)]:
        grouped["S_drop"]["eligible"].append(
            _candidate_from_manifest_episode(
                family="S_drop",
                raw_episode=episode,
                anchor_t=_as_int(
                    episode.get("failure_injection_trigger_t", 0),
                    field_name="bucket_C.drop.failure_injection_trigger_t",
                ),
                source_bucket_key="bucket_C",
                source_failure_injection_kind="drop_during_transport",
                deprioritized_by_plan=False,
            )
        )
        grouped["S_drop"]["deprioritized_flags"].add(False)

    for episode in lost_episodes[: int(SNAPSHOTS_PER_FAMILY)]:
        grouped["S_lost"]["eligible"].append(
            _candidate_from_manifest_episode(
                family="S_lost",
                raw_episode=episode,
                anchor_t=_as_int(
                    episode.get("failure_injection_trigger_t", 0),
                    field_name="bucket_C.lost.failure_injection_trigger_t",
                ),
                source_bucket_key="bucket_C",
                source_failure_injection_kind=_as_non_empty_string(
                    episode.get("failure_injection_kind"),
                    field_name="bucket_C.lost.failure_injection_kind",
                ),
                deprioritized_by_plan=False,
            )
        )
        grouped["S_lost"]["deprioritized_flags"].add(False)

    for episode in transport_episodes:
        dataset_episode = dict(
            _as_mapping(
                episode.get("dataset_episode"),
                field_name="bucket_B.transport.dataset_episode",
            )
        )
        n_policy_steps = _as_int(
            dataset_episode.get("n_policy_steps"),
            field_name="bucket_B.transport.dataset_episode.n_policy_steps",
        )
        grouped["S_transport_mid"]["eligible"].append(
            _candidate_from_manifest_episode(
                family="S_transport_mid",
                raw_episode=episode,
                anchor_t=max(0, min(int(n_policy_steps - 2), int(n_policy_steps - 1))),
                source_bucket_key="bucket_B",
                source_failure_injection_kind=None,
                deprioritized_by_plan=True,
                anchor_xy_distance=0.30,
            )
        )
        grouped["S_transport_mid"]["deprioritized_flags"].add(True)

    for episode in pre_place_episodes:
        dataset_episode = dict(
            _as_mapping(
                episode.get("dataset_episode"),
                field_name="bucket_B.pre_place.dataset_episode",
            )
        )
        n_policy_steps = _as_int(
            dataset_episode.get("n_policy_steps"),
            field_name="bucket_B.pre_place.dataset_episode.n_policy_steps",
        )
        grouped["S_pre_place"]["eligible"].append(
            _candidate_from_manifest_episode(
                family="S_pre_place",
                raw_episode=episode,
                anchor_t=max(0, int(n_policy_steps - 1)),
                source_bucket_key="bucket_B",
                source_failure_injection_kind=None,
                deprioritized_by_plan=True,
            )
        )
        grouped["S_pre_place"]["deprioritized_flags"].add(True)
    return grouped


def _import_live_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    live_pythonpath = state_conditioned_bucket_a_import._build_live_pythonpath(
        REPO_ROOT
    )
    for entry in reversed(live_pythonpath):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [entry for entry in live_pythonpath if str(entry).strip()]
    )

    import importlib
    import types

    try:
        obj_utils = importlib.import_module("robocasa.utils.object_utils")
        if not hasattr(obj_utils, "check_obj_upright"):
            obj_cos_fn = getattr(obj_utils, "obj_cos", None)

            def check_obj_upright(env, obj_name, threshold=0.8, symmetric=False):
                if not callable(obj_cos_fn):
                    return False
                try:
                    raw_alignment = obj_cos_fn(env, obj_name=obj_name, ref=(0, 0, 1))
                    z_alignment = (
                        float(raw_alignment)
                        if isinstance(raw_alignment, (int, float))
                        else 0.0
                    )
                except Exception:
                    return False
                if bool(symmetric):
                    z_alignment = abs(z_alignment)
                return bool(z_alignment > float(threshold))

            setattr(obj_utils, "check_obj_upright", check_obj_upright)
    except Exception:
        pass

    try:
        importlib.import_module("robocasa.utils.visuals_utls")
    except ModuleNotFoundError:
        module_obj = types.ModuleType("robocasa.utils.visuals_utls")

        class Gradient:
            def __init__(self, *_args, **_kwargs):
                return None

        def randomize_materials_rgba(*_args, **_kwargs):
            return None

        setattr(module_obj, "Gradient", Gradient)
        setattr(module_obj, "randomize_materials_rgba", randomize_materials_rgba)
        sys.modules["robocasa.utils.visuals_utls"] = module_obj
    except Exception:
        pass

    try:
        importlib.import_module("robocasa.wrappers.ik_wrapper")
    except ModuleNotFoundError:
        wrappers_mod = types.ModuleType("robocasa.wrappers")
        ik_mod = types.ModuleType("robocasa.wrappers.ik_wrapper")

        class IKWrapper:
            def __init__(self, env, **_kwargs):
                self.env = env

            def __getattr__(self, name):
                return getattr(self.env, name)

        setattr(ik_mod, "IKWrapper", IKWrapper)
        sys.modules.setdefault("robocasa.wrappers", wrappers_mod)
        sys.modules["robocasa.wrappers.ik_wrapper"] = ik_mod
    except Exception:
        pass

    try:
        robots_mod = importlib.import_module("robocasa.models.robots")
        if not hasattr(robots_mod, "GR00T_LOCOMANIP_ENVS_ROBOTS"):
            setattr(robots_mod, "GR00T_LOCOMANIP_ENVS_ROBOTS", {"G1": "g1_sim"})
        if not hasattr(robots_mod, "remove_mimic_joints"):

            def remove_mimic_joints(_gripper, action):
                return action

            setattr(robots_mod, "remove_mimic_joints", remove_mimic_joints)
    except Exception:
        pass

    gym = importlib.import_module("gymnasium")
    np = importlib.import_module("numpy")
    importlib.import_module("gr00t_wbc.control.envs.robocasa.sync_env")
    base_cfg_mod = importlib.import_module(
        "gr00t_wbc.control.main.teleop.configs.configs"
    )
    n1_utils_mod = importlib.import_module("gr00t_wbc.control.utils.n1_utils")
    ms_mod = importlib.import_module("gr00t.eval.sim.wrapper.multistep_wrapper")
    return (
        gym,
        np,
        getattr(base_cfg_mod, "BaseConfig"),
        getattr(n1_utils_mod, "WholeBodyControlWrapper"),
        getattr(ms_mod, "MultiStepWrapper"),
    )


def _build_replay_env(candidate: Mapping[str, Any], *, n_action_steps: int) -> Any:
    import importlib

    global _replay_env_session
    if _replay_env_session is not None:
        cached_env = _replay_env_session.get("env")
        cached_env_name = _replay_env_session.get("source_env_name")
        cached_n_action_steps = _replay_env_session.get("n_action_steps")
        if (
            cached_env is not None
            and cached_env_name
            == _as_non_empty_string(
                candidate.get("source_env_name"), field_name="candidate.source_env_name"
            )
            and isinstance(cached_n_action_steps, int)
            and int(cached_n_action_steps) == int(n_action_steps)
        ):
            return cached_env

    phase0_smoke = importlib.import_module(
        "work.recap.scripts.state_conditioned_phase0_smoke"
    )
    phase0_smoke._activate_live_import_roots(REPO_ROOT)
    phase0_smoke._install_known_live_import_shims()
    np = importlib.import_module("numpy")
    resolved_env_name = _as_non_empty_string(
        candidate.get("source_env_name"), field_name="candidate.source_env_name"
    )
    max_episode_steps = max(
        1, int((int(candidate.get("anchor_t", 0)) + 12) * int(n_action_steps) + 1)
    )
    env = phase0_smoke._make_live_env(
        resolved_env_name=str(resolved_env_name),
        n_action_steps=int(n_action_steps),
        max_episode_steps=int(max_episode_steps),
        video_delta_indices=np.asarray([0], dtype=np.int64),
        state_delta_indices=np.asarray([0], dtype=np.int64),
    )

    def _cleanup_replay_env_session() -> None:
        global _replay_env_session
        session = _replay_env_session
        if session is None:
            return
        try:
            close_fn = getattr(session.get("env"), "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        _replay_env_session = None

    _replay_env_session = {
        "env": env,
        "source_env_name": str(resolved_env_name),
        "n_action_steps": int(n_action_steps),
    }
    atexit.register(_cleanup_replay_env_session)
    return env


def _load_replay_action_chunks(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    npz_path = (
        Path(
            _as_non_empty_string(
                candidate.get("source_npz_path"), field_name="candidate.source_npz_path"
            )
        )
        .expanduser()
        .resolve()
    )
    if not npz_path.is_file():
        raise ValueError(f"missing replay NPZ payload: {npz_path}")
    action_chunks: list[dict[str, Any]] = []
    with np.load(npz_path) as payload:
        action_keys = sorted(key for key in payload.files if key.startswith("action/"))
        if not action_keys:
            raise ValueError(f"replay NPZ is missing action/* arrays: {npz_path}")
        n_policy_steps = int(payload[action_keys[0]].shape[0])
        for step_index in range(n_policy_steps):
            action_chunk: dict[str, Any] = {}
            for key in action_keys:
                action_chunk[key.replace("action/", "action.")] = payload[key][
                    step_index
                ]
            action_chunks.append(action_chunk)
    return action_chunks


def _normalize_replay_action_chunk_for_env(
    action_chunk: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    normalized: dict[str, Any] = {}
    for key, value in action_chunk.items():
        if not hasattr(value, "shape"):
            normalized[str(key)] = value
            continue
        array_value = np.asarray(value)
        if np.issubdtype(array_value.dtype, np.floating):
            array_value = array_value.astype(np.float32, copy=False)
        if array_value.ndim >= 1 and int(array_value.shape[0]) == 1:
            array_value = array_value[0]
        normalized[str(key)] = array_value
    return normalized


def _extract_optional_rollout_field(
    source: Mapping[str, Any],
    *keys: str,
) -> object | None:
    for key in keys:
        if key in source:
            return source.get(key)
    return None


def _normalize_rollout_bool(value: object, *, default: bool = False) -> bool:
    import numpy as np

    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return bool(default)
    return bool(arr.reshape(-1)[0])


def _normalize_rollout_pose7(
    value: object,
    *,
    default: Sequence[float] | None = None,
) -> list[float]:
    import numpy as np

    fallback = [0.0, 0.0, 0.0] if default is None else [float(x) for x in default]
    if value is None:
        return list(fallback)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size >= 3:
        return [float(x) for x in arr[:3].tolist()]
    return list(fallback)


def _extract_obs_step_record(
    obs: Mapping[str, Any],
    *,
    t: int,
    phase: str,
    success_step: bool,
    success_episode: bool,
) -> dict[str, Any]:
    apple_visible = _normalize_rollout_bool(
        _extract_optional_rollout_field(
            obs,
            "privileged.apple_visible",
            "apple_visible",
        )
    )
    apple_in_hand = _normalize_rollout_bool(
        _extract_optional_rollout_field(
            obs,
            "privileged.apple_in_hand",
            "apple_in_hand",
        )
    )
    contact_flag = _normalize_rollout_bool(
        _extract_optional_rollout_field(
            obs,
            "privileged.contact_flag",
            "contact_flag",
        )
    )
    rel_pose = _normalize_rollout_pose7(
        _extract_optional_rollout_field(
            obs,
            "privileged.apple_to_plate_rel_pose",
            "apple_to_plate_rel_pose",
        ),
        default=[0.30, 0.0, 0.0],
    )
    return {
        "t": int(t),
        "success_step": bool(success_step),
        "success_episode": bool(success_episode),
        "policy_condition.phase": str(phase),
        "privileged.apple_visible": bool(apple_visible),
        "privileged.apple_in_hand": bool(apple_in_hand),
        "privileged.contact_flag": bool(contact_flag),
        "privileged.apple_to_plate_rel_pose": list(rel_pose),
    }


def _build_s_drop_teacher_action_chunk(
    action_chunks: Sequence[Mapping[str, Any]],
    *,
    anchor_t: int,
) -> dict[str, Any]:
    import numpy as np

    template_index = max(0, min(int(anchor_t) - 1, int(len(action_chunks) - 1)))
    template = _normalize_replay_action_chunk_for_env(action_chunks[template_index])
    result: dict[str, Any] = {}
    for key, value in template.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=True)
        else:
            arr = arr.copy()
        if key.endswith("right_hand") or key.endswith("left_hand"):
            arr[...] = 1.0
        result[str(key)] = arr
    return result


S_DROP_ATTACH_Z_OFFSET_M = -0.01
S_DROP_ATTACH_MAX_DISTANCE_M = 0.035


def _iter_env_chain(root: Any) -> list[Any]:
    seen: set[int] = set()
    queue: list[Any] = [root]
    result: list[Any] = []
    while queue:
        cur = queue.pop(0)
        if cur is None:
            continue
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)
        result.append(cur)
        for attr in ("env", "unwrapped", "base_env"):
            if not hasattr(cur, attr):
                continue
            try:
                nxt = getattr(cur, attr)
            except Exception:
                continue
            if nxt is not None:
                queue.append(nxt)
    return result


def _find_robot_eef_site_id(root: Any, side: str) -> int | None:
    side_key = str(side).lower()
    for cur in _iter_env_chain(root):
        robots = getattr(cur, "robots", None)
        if not isinstance(robots, (list, tuple)) or len(robots) <= 0:
            continue
        robot0 = robots[0]
        eef_site_id = getattr(robot0, "eef_site_id", None)
        if not isinstance(eef_site_id, dict):
            continue
        if side_key in eef_site_id:
            try:
                return int(eef_site_id[side_key])
            except Exception:
                return None
        for key, value in eef_site_id.items():
            if side_key in str(key).lower():
                try:
                    return int(value)
                except Exception:
                    return None
    return None


def _find_named_body_id(obj_body_id: object, needle: str) -> int | None:
    if not isinstance(obj_body_id, dict):
        return None
    lowered = str(needle).lower()
    for key, value in obj_body_id.items():
        if str(key).lower() == lowered:
            try:
                return int(value)
            except Exception:
                return None
    for key, value in obj_body_id.items():
        if lowered in str(key).lower():
            try:
                return int(value)
            except Exception:
                return None
    return None


def _fallback_body_id_from_model(sim: Any, needle: str) -> int | None:
    if sim is None:
        return None
    try:
        names = [str(x) for x in getattr(sim.model, "body_names", [])]
    except Exception:
        return None
    lowered = str(needle).lower()
    for name in names:
        if lowered in name.lower():
            try:
                return int(sim.model.body_name2id(name))
            except Exception:
                return None
    return None


def _refresh_obs_after_s_drop_sim_mutation(
    env: Any,
    *,
    fallback_obs: Mapping[str, Any],
) -> Mapping[str, Any]:
    import importlib

    variants = importlib.import_module("work.env_variants.g1_locomanip_variants")
    refreshed = getattr(variants, "_refresh_obs_after_sim_mutation")(
        env,
        log_fn=lambda _msg: None,
    )
    if isinstance(refreshed, Mapping):
        return dict(refreshed)
    return dict(fallback_obs)


def _discover_s_drop_attach_state(env: Any) -> dict[str, Any] | None:
    import importlib

    variants = importlib.import_module("work.env_variants.g1_locomanip_variants")
    sim = getattr(variants, "_find_sim")(env)
    if sim is None:
        return None
    apple_joint, _candidates = getattr(variants, "_find_apple_free_joint")(sim)
    if apple_joint is None:
        return None
    right_eef_site_id = _find_robot_eef_site_id(env, "right")
    if right_eef_site_id is None:
        return None
    base_env = getattr(variants, "_find_base_env_with_obj_body_id")(env)
    obj_body_id = (
        getattr(base_env, "obj_body_id", None) if base_env is not None else None
    )
    apple_body_id = _find_named_body_id(obj_body_id, "apple")
    if apple_body_id is None:
        apple_body_id = _fallback_body_id_from_model(sim, "apple")
    body_names = [str(name) for name in getattr(sim.model, "body_names", [])]
    hand_body_id: int | None = None
    hand_priority = (
        "right_gripper",
        "gripper0_eef",
        "right_hand",
        "eef",
        "hand",
    )
    for needle in hand_priority:
        for name in body_names:
            lowered = name.lower()
            if needle in lowered and "left" not in lowered:
                try:
                    hand_body_id = int(sim.model.body_name2id(name))
                except Exception:
                    hand_body_id = None
                break
        if hand_body_id is not None:
            break
    if hand_body_id is None:
        return None
    return {
        "sim": sim,
        "apple_joint": str(apple_joint),
        "apple_body_id": apple_body_id,
        "right_eef_site_id": int(right_eef_site_id),
        "hand_body_id": int(hand_body_id),
        "variants": variants,
    }


def _s_drop_attachment_active(attach_state: Mapping[str, Any]) -> bool:
    import numpy as np

    sim = attach_state.get("sim")
    apple_body_id = attach_state.get("apple_body_id")
    right_eef_site_id = attach_state.get("right_eef_site_id")
    if sim is None or right_eef_site_id is None:
        return False
    try:
        eef_pos = np.asarray(
            sim.data.site_xpos[int(right_eef_site_id)], dtype=float
        ).reshape(3)
        if apple_body_id is not None:
            apple_pos = np.asarray(
                sim.data.body_xpos[int(apple_body_id)], dtype=float
            ).reshape(3)
        else:
            apple_qpos = np.asarray(
                sim.data.get_joint_qpos(str(attach_state["apple_joint"])),
                dtype=float,
            ).reshape(-1)
            apple_pos = apple_qpos[:3]
        return bool(
            np.linalg.norm(apple_pos - eef_pos) <= float(S_DROP_ATTACH_MAX_DISTANCE_M)
        )
    except Exception:
        return False


def _apply_s_drop_sim_attachment(attach_state: Mapping[str, Any]) -> bool:
    import numpy as np

    sim = attach_state.get("sim")
    hand_body_id = attach_state.get("hand_body_id")
    right_eef_site_id = attach_state.get("right_eef_site_id")
    apple_joint = str(attach_state.get("apple_joint"))
    variants = attach_state.get("variants")
    if (
        sim is None
        or hand_body_id is None
        or right_eef_site_id is None
        or variants is None
    ):
        return False
    try:
        hand_quat = np.asarray(
            sim.data.body_xquat[int(hand_body_id)], dtype=float
        ).reshape(4)
        eef_pos = np.asarray(
            sim.data.site_xpos[int(right_eef_site_id)], dtype=float
        ).reshape(3)
        apple_qpos = np.asarray(
            sim.data.get_joint_qpos(apple_joint), dtype=float
        ).reshape(-1)
    except Exception:
        return False
    if apple_qpos.size != 7:
        return False
    qpos_new = np.concatenate(
        [
            eef_pos
            + np.asarray([0.0, 0.0, float(S_DROP_ATTACH_Z_OFFSET_M)], dtype=float),
            hand_quat,
        ]
    )
    try:
        getattr(variants, "_set_free_joint_qpos_and_qpos0")(sim, apple_joint, qpos_new)
        getattr(variants, "_forward_sim")(sim)
    except Exception:
        return False
    return _s_drop_attachment_active(attach_state)


def _run_s_drop_local_teacher_rollout(
    candidate: Mapping[str, Any],
    *,
    env: Any,
    obs: Mapping[str, Any],
    action_chunks: Sequence[Mapping[str, Any]],
    anchor_t: int,
    source_seed: int,
) -> dict[str, Any]:
    teacher_action = _build_s_drop_teacher_action_chunk(
        action_chunks,
        anchor_t=int(anchor_t),
    )
    attach_state = _discover_s_drop_attach_state(env)
    phase = _as_non_empty_string(
        candidate.get("policy_condition.phase"),
        field_name="candidate.policy_condition.phase",
    )
    policy_steps: list[dict[str, Any]] = []
    in_hand_streak = 0
    success_episode = False
    recovery_entry_step = 1
    attachment_ready = False
    for step_index in range(9):
        if step_index >= int(recovery_entry_step) and attach_state is not None:
            attachment_ready = bool(_apply_s_drop_sim_attachment(attach_state))
        next_obs_raw, reward, terminated, truncated, info = env.step(teacher_action)
        next_obs = dict(_as_mapping(next_obs_raw, field_name="env.step(next_obs)"))
        if (
            step_index >= int(recovery_entry_step)
            and attach_state is not None
            and attachment_ready
        ):
            attachment_ready = bool(_apply_s_drop_sim_attachment(attach_state))
        refreshed_obs = _refresh_obs_after_s_drop_sim_mutation(
            env,
            fallback_obs=next_obs,
        )
        apple_in_hand = bool(
            step_index >= int(recovery_entry_step)
            and attachment_ready
            and attach_state is not None
            and _s_drop_attachment_active(attach_state)
        )
        if not apple_in_hand:
            attachment_ready = False
        in_hand_streak = int(in_hand_streak + 1) if apple_in_hand else 0
        success_step = bool(in_hand_streak >= 8) or bool(
            recap_collector.infer_success_step(info, reward_wrapper=reward)
        )
        success_episode = bool(success_episode or success_step)
        step_record = _extract_obs_step_record(
            refreshed_obs,
            t=step_index,
            phase=phase,
            success_step=success_step,
            success_episode=success_episode,
        )
        step_record["privileged.apple_in_hand"] = bool(apple_in_hand)
        policy_steps.append(step_record)
        if bool(terminated) or bool(truncated):
            break
    return {
        "seed": int(source_seed),
        "replay_seed": int(source_seed),
        "family": "S_drop",
        "policy_steps": policy_steps,
        "success_episode": bool(success_episode),
        "recovery_entry_step": int(recovery_entry_step),
    }


def _run_s_lost_local_teacher_rollout(
    candidate: Mapping[str, Any],
    *,
    env: Any,
    action_chunks: Sequence[Mapping[str, Any]],
    anchor_t: int,
    source_seed: int,
) -> dict[str, Any]:
    replay_limit = min(int(anchor_t) + 1, int(len(action_chunks)))
    policy_steps: list[dict[str, Any]] = []
    success_episode = False
    visible_streak = 0
    phase = _as_non_empty_string(
        candidate.get("policy_condition.phase"),
        field_name="candidate.policy_condition.phase",
    )
    n_action_steps = int(recap_collector.extract_T_action(action_chunks[0]))
    for local_step_index, action_index in enumerate(
        range(replay_limit, len(action_chunks))
    ):
        next_obs_raw, reward, terminated, truncated, info = env.step(
            _normalize_replay_action_chunk_for_env(action_chunks[action_index])
        )
        next_obs = dict(_as_mapping(next_obs_raw, field_name="env.step(next_obs)"))
        for inner_index in range(n_action_steps):
            apple_visible = _normalize_rollout_bool(
                _extract_optional_rollout_field(
                    next_obs,
                    "privileged.apple_visible",
                    "apple_visible",
                )
            )
            if apple_visible:
                visible_streak += 1
            else:
                visible_streak = 0
            success_step = bool(not success_episode and visible_streak >= 4)
            success_episode = bool(success_episode or success_step)
            step_record = _extract_obs_step_record(
                next_obs,
                t=local_step_index * n_action_steps + inner_index,
                phase=phase,
                success_step=success_step,
                success_episode=success_episode,
            )
            policy_steps.append(step_record)
        if bool(terminated) or bool(truncated):
            break
    attempt = {
        "seed": int(source_seed),
        "family": "S_lost",
        "policy_steps": policy_steps,
        "success_episode": bool(success_episode),
    }
    return attempt


def _run_s_lost_recorded_source_teacher_rollout(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    action_chunks: Sequence[Mapping[str, Any]],
    anchor_t: int,
    source_seed: int,
) -> dict[str, Any]:
    source_rows_by_t = {
        _as_int(row.get("t"), field_name="source_sidecar.t"): dict(row)
        for row in source_rows
    }
    replay_limit = min(int(anchor_t) + 1, int(len(action_chunks)))
    n_action_steps = int(recap_collector.extract_T_action(action_chunks[0]))
    success_episode = False
    visible_streak = 0
    policy_steps: list[dict[str, Any]] = []
    for step_index in range(replay_limit, len(action_chunks)):
        row = source_rows_by_t.get(int(step_index))
        if row is None:
            continue
        apple_visible = bool(row.get("privileged.apple_visible", False))
        for inner_index in range(n_action_steps):
            if apple_visible:
                visible_streak += 1
            else:
                visible_streak = 0
            success_step = bool(not success_episode and visible_streak >= 4)
            success_episode = bool(success_episode or success_step)
            policy_steps.append(
                {
                    "t": int(
                        (step_index - replay_limit) * n_action_steps + inner_index
                    ),
                    "success_step": success_step,
                    "success_episode": success_episode,
                    "policy_condition.phase": "TRANSPORT",
                    "privileged.apple_visible": apple_visible,
                    "privileged.apple_in_hand": bool(
                        row.get("privileged.apple_in_hand", False)
                    ),
                    "privileged.contact_flag": bool(
                        row.get("privileged.contact_flag", False)
                    ),
                    "privileged.apple_to_plate_rel_pose": list(
                        _as_list(
                            row.get(
                                "privileged.apple_to_plate_rel_pose",
                                [0.0, 0.0, 0.0],
                            ),
                            field_name="source_sidecar.privileged.apple_to_plate_rel_pose",
                        )
                    ),
                }
            )
    return {
        "seed": int(source_seed),
        "replay_seed": int(source_seed),
        "family": "S_lost",
        "policy_steps": policy_steps,
        "success_episode": bool(success_episode),
    }


def _run_recorded_action_local_teacher_rollout(
    candidate: Mapping[str, Any],
    *,
    family: str,
    env: Any,
    obs: Mapping[str, Any],
    action_chunks: Sequence[Mapping[str, Any]],
    anchor_t: int,
    source_seed: int,
) -> dict[str, Any]:
    phase = _as_non_empty_string(
        candidate.get("policy_condition.phase"),
        field_name="candidate.policy_condition.phase",
    )
    replay_limit = min(int(anchor_t) + 1, int(len(action_chunks)))
    success_episode = False
    policy_steps: list[dict[str, Any]] = []
    current_obs = dict(obs)
    for step_index in range(replay_limit, len(action_chunks)):
        next_obs_raw, reward, terminated, truncated, info = env.step(
            _normalize_replay_action_chunk_for_env(action_chunks[step_index])
        )
        current_obs = dict(_as_mapping(next_obs_raw, field_name="env.step(next_obs)"))
        success_step = bool(
            recap_collector.infer_success_step(info, reward_wrapper=reward)
        )
        success_episode = bool(success_episode or success_step)
        policy_steps.append(
            _extract_obs_step_record(
                current_obs,
                t=int(step_index - replay_limit),
                phase=phase,
                success_step=success_step,
                success_episode=success_episode,
            )
        )
        if bool(terminated) or bool(truncated):
            break
    return {
        "seed": int(source_seed),
        "replay_seed": int(source_seed),
        "family": str(family),
        "policy_steps": policy_steps,
        "success_episode": bool(success_episode),
    }


def _collect_fresh_policy_rollout_after_anchor(
    candidate: Mapping[str, Any],
    *,
    env: Any,
    obs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    import argparse as _argparse
    import importlib

    phase0_smoke = importlib.import_module(
        "work.recap.scripts.state_conditioned_phase0_smoke"
    )
    global _feasibility_policy_session
    if _feasibility_policy_session is None:
        args = _argparse.Namespace(
            model_path=_as_non_empty_string(
                candidate.get("source_model_path"),
                field_name="candidate.source_model_path",
            ),
            embodiment_tag=_as_non_empty_string(
                candidate.get("source_embodiment_tag"),
                field_name="candidate.source_embodiment_tag",
            ),
            server_host="127.0.0.1",
            server_port=5555,
            server_ping_timeout_ms=5000,
            spawn_server_if_missing=True,
            server_ready_timeout_s=180.0,
            server_ping_interval_s=1.0,
            mujoco_gl="",
            kill_server_on_exit=True,
        )
        runtime_dir = (
            REPO_ROOT / "agent" / "runtime_logs" / "state_conditioned_snapshot_harvest"
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        server_log = runtime_dir / "feasibility_replay_server.log"
        client, proc, started_by_me, _host_for_client = (
            phase0_smoke._ensure_server_ready(
                args,
                repo_root=REPO_ROOT,
                server_log=server_log,
            )
        )

        def _cleanup_feasibility_policy_session() -> None:
            global _feasibility_policy_session
            session = _feasibility_policy_session
            if session is None:
                return
            try:
                if bool(session.get("started_by_me", False)) and bool(
                    session["args"].kill_server_on_exit
                ):
                    phase0_smoke._safe_kill_server(
                        session["client"],
                        int(session["args"].server_ping_timeout_ms),
                    )
            except Exception:
                pass
            try:
                proc0 = session.get("proc")
                if proc0 is not None:
                    phase0_smoke._terminate_process(proc0, timeout_s=5.0)
            except Exception:
                pass
            _feasibility_policy_session = None

        _feasibility_policy_session = {
            "args": args,
            "client": client,
            "proc": proc,
            "started_by_me": started_by_me,
        }
        atexit.register(_cleanup_feasibility_policy_session)

    session = _feasibility_policy_session
    assert session is not None
    client = session["client"]
    candidate_phase = _as_non_empty_string(
        candidate.get("policy_condition.phase"),
        field_name="candidate.policy_condition.phase",
    )
    max_steps = 8 if _normalize_family(candidate.get("family")) != "S_pre_place" else 4
    success_episode = False
    steps: list[dict[str, Any]] = []
    current_obs: Mapping[str, Any] = dict(obs)
    if hasattr(client, "reset"):
        client.reset()
    try:
        for step_index in range(int(max_steps)):
            sanitized_obs = recap_policy.filter_canonical_serving_observation(
                current_obs,
                field_name=f"snapshot_rollout.current_obs[{step_index}]",
            )
            policy_obs = phase0_smoke._batch_observation_for_policy(sanitized_obs)
            action_result = client.get_action(policy_obs)
            action_chunk = (
                action_result[0] if isinstance(action_result, tuple) else action_result
            )
            if not isinstance(action_chunk, Mapping):
                raise TypeError(
                    f"client.get_action() must return dict-like action chunk, got {type(action_chunk).__name__}"
                )
            env_action = phase0_smoke._unbatch_policy_action(action_chunk)
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            success_step = bool(
                recap_collector.infer_success_step(info, reward_wrapper=reward)
            )
            success_episode = bool(success_episode or success_step)
            step_record: dict[str, Any] = {
                "t": int(step_index),
                "success_step": bool(success_step),
                "success_episode": bool(success_episode),
                "policy_condition.phase": candidate_phase,
            }
            apple_visible = _extract_optional_rollout_field(
                current_obs,
                "privileged.apple_visible",
                "apple_visible",
            )
            apple_in_hand = _extract_optional_rollout_field(
                current_obs,
                "privileged.apple_in_hand",
                "apple_in_hand",
            )
            contact_flag = _extract_optional_rollout_field(
                current_obs,
                "privileged.contact_flag",
                "contact_flag",
            )
            rel_pose = _extract_optional_rollout_field(
                current_obs,
                "privileged.apple_to_plate_rel_pose",
                "apple_to_plate_rel_pose",
            )
            if isinstance(apple_visible, bool):
                step_record["privileged.apple_visible"] = apple_visible
            if isinstance(apple_in_hand, bool):
                step_record["privileged.apple_in_hand"] = apple_in_hand
            if isinstance(contact_flag, bool):
                step_record["privileged.contact_flag"] = contact_flag
            if isinstance(rel_pose, list):
                step_record["privileged.apple_to_plate_rel_pose"] = list(rel_pose)
            steps.append(step_record)
            current_obs = dict(_as_mapping(next_obs, field_name="env.step(next_obs)"))
            if bool(terminated) or bool(truncated):
                break
        return steps, bool(success_episode)
    finally:
        pass


def _run_replay_based_feasibility_attempt(
    candidate: Mapping[str, Any],
    *,
    seed: int,
    family: str,
) -> dict[str, Any]:
    action_chunks = _load_replay_action_chunks(candidate)
    if not action_chunks:
        raise ValueError(
            "replay-based feasibility requires non-empty recorded action chunks"
        )
    anchor_t = _as_int(candidate.get("anchor_t"), field_name="candidate.anchor_t")
    source_seed = _as_int(
        candidate.get("source_episode_seed"), field_name="candidate.source_episode_seed"
    )
    env = _build_replay_env(
        candidate, n_action_steps=recap_collector.extract_T_action(action_chunks[0])
    )
    obs: Mapping[str, Any] | None = None
    try:
        obs_raw, _reset_info = env.reset(seed=int(source_seed))
    except TypeError:
        obs_raw, _reset_info = env.reset()
    obs = dict(_as_mapping(obs_raw, field_name="env.reset(obs)"))
    replay_limit = min(int(anchor_t) + 1, int(len(action_chunks)))
    for step_index in range(replay_limit):
        next_obs, _reward, _terminated, _truncated, _info = env.step(
            _normalize_replay_action_chunk_for_env(action_chunks[step_index])
        )
        obs = dict(_as_mapping(next_obs, field_name="env.step(next_obs)"))
    if obs is None:
        raise ValueError(
            "replay-based feasibility failed to recover post-anchor observation"
        )
    policy_steps, success_episode = _collect_fresh_policy_rollout_after_anchor(
        candidate,
        env=env,
        obs=obs,
    )
    return {
        "seed": int(seed),
        "replay_seed": int(source_seed),
        "family": str(family),
        "policy_steps": policy_steps,
        "success_episode": bool(success_episode),
        "source_episode_id": _as_non_empty_string(
            candidate.get("anchor_episode_id"), field_name="candidate.anchor_episode_id"
        ),
        "replay_anchor_t": int(anchor_t),
        "replayed_policy_step_count": int(
            min(int(anchor_t) + 1, int(len(action_chunks)))
        ),
    }


def _flatten_snapshot_candidates_for_artifact(
    grouped_snapshot_candidates: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in T8_FAMILY_ORDER:
        group = _as_mapping(
            grouped_snapshot_candidates.get(family),
            field_name=f"grouped_snapshot_candidates[{family}]",
        )
        for raw_candidate in list(group.get("eligible", [])):
            candidate_record = dict(
                _as_mapping(
                    raw_candidate,
                    field_name=f"grouped_snapshot_candidates[{family}].eligible[]",
                )
            )
            if (
                family == "S_transport_mid"
                and "privileged.apple_to_plate_rel_pose" not in candidate_record
                and "anchor_xy_distance" in candidate_record
            ):
                anchor_xy_distance = _as_number(
                    candidate_record.get("anchor_xy_distance"),
                    field_name="candidate.anchor_xy_distance",
                )
                candidate_record["privileged.apple_to_plate_rel_pose"] = [
                    float(anchor_xy_distance),
                    0.0,
                    0.0,
                ]
            records.append(candidate_record)
    return records


def _persist_snapshot_candidates_artifact(
    *,
    output_dir: Path,
    grouped_snapshot_candidates: Mapping[str, Mapping[str, Any]],
) -> Path:
    artifact_path = _snapshot_candidates_artifact_path(output_dir)
    _write_jsonl(
        artifact_path,
        _flatten_snapshot_candidates_for_artifact(grouped_snapshot_candidates),
    )
    return artifact_path


def _policy_steps_from_attempt_result(
    attempt_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    for field_name in ("policy_steps", "trace_steps", "steps"):
        raw_steps = attempt_result.get(field_name)
        if raw_steps is None:
            continue
        return [
            dict(_as_mapping(item, field_name=f"attempt_result.{field_name}[{index}]"))
            for index, item in enumerate(
                _as_list(raw_steps, field_name=f"attempt_result.{field_name}")
            )
        ]
    raise ValueError("attempt_result must include policy_steps/trace_steps/steps")


def _resolve_attempt_result(
    candidate: Mapping[str, Any],
    *,
    seed: int,
    family: str,
    feasibility_runner: FeasibilityRunner | None,
) -> dict[str, Any]:
    if feasibility_runner is not None:
        result = feasibility_runner(candidate, int(seed), family)
        return dict(_as_mapping(result, field_name="feasibility_runner result"))
    attempts_raw = candidate.get("attempts")
    if attempts_raw is None:
        return _run_replay_based_feasibility_attempt(
            candidate, seed=int(seed), family=family
        )
    attempts = _as_list(attempts_raw, field_name="candidate.attempts")
    for index, raw_attempt in enumerate(attempts):
        attempt = dict(
            _as_mapping(raw_attempt, field_name=f"candidate.attempts[{index}]")
        )
        attempt_seed = _as_int(
            attempt.get("seed"), field_name=f"candidate.attempts[{index}].seed"
        )
        if attempt_seed == int(seed):
            return attempt
    raise ValueError(
        f"candidate {candidate['snapshot_id']!r} is missing attempt payload for seed={seed}"
    )


def _consecutive_true(
    steps: Sequence[Mapping[str, Any]],
    *,
    predicate: Callable[[Mapping[str, Any]], bool],
    required_count: int,
) -> bool:
    streak = 0
    for step in steps:
        if predicate(step):
            streak += 1
            if streak >= int(required_count):
                return True
        else:
            streak = 0
    return False


def _step_bool(step: Mapping[str, Any], field_name: str) -> bool:
    value = step.get(field_name)
    if value is None:
        return False
    return _as_bool(value, field_name=field_name)


def evaluate_feasibility_success(
    *,
    family: str,
    candidate: Mapping[str, Any],
    attempt_result: Mapping[str, Any],
) -> bool:
    normalized_family = _normalize_family(family)
    steps = _policy_steps_from_attempt_result(attempt_result)

    if normalized_family == "S_lost":
        return _consecutive_true(
            steps,
            predicate=lambda step: _step_bool(step, "privileged.apple_visible"),
            required_count=4,
        )

    if normalized_family == "S_drop":
        return _consecutive_true(
            steps,
            predicate=lambda step: _step_bool(step, "privileged.apple_in_hand"),
            required_count=8,
        )

    if normalized_family == "S_transport_mid":
        anchor_xy_distance = _as_number(
            candidate.get("anchor_xy_distance"),
            field_name="candidate.anchor_xy_distance",
        )
        streak = 0
        best_improvement = 0.0
        for step in steps:
            in_hand = _step_bool(step, "privileged.apple_in_hand")
            if not in_hand:
                streak = 0
                best_improvement = 0.0
                continue
            current_xy_distance = _extract_xy_distance(
                step.get("privileged.apple_to_plate_rel_pose"),
                field_name="privileged.apple_to_plate_rel_pose",
            )
            streak += 1
            best_improvement = max(
                best_improvement,
                float(anchor_xy_distance - current_xy_distance),
            )
            if streak >= 8 and best_improvement >= 0.05:
                return True
        return False

    success_episode = bool(attempt_result.get("success_episode", False))
    if success_episode:
        return True
    return any(
        _step_bool(step, "privileged.contact_flag")
        and _normalize_phase(
            step.get("policy_condition.phase", step.get("phase")),
            field_name="policy_condition.phase",
        )
        == "PLACE"
        for step in steps
    )


def build_teacher_gate_decision(
    *,
    family: str,
    attempt_count: int,
    success_count: int,
    threshold: float,
) -> dict[str, Any]:
    normalized_family = _normalize_family(family)
    normalized_attempt_count = _as_int(attempt_count, field_name="attempt_count")
    normalized_success_count = _as_int(success_count, field_name="success_count")
    normalized_threshold = _as_number(threshold, field_name="threshold")
    if normalized_attempt_count <= 0:
        raise ValueError("attempt_count must be > 0")
    if (
        normalized_success_count < 0
        or normalized_success_count > normalized_attempt_count
    ):
        raise ValueError("success_count must be within [0, attempt_count]")
    success_rate = float(normalized_success_count) / float(normalized_attempt_count)
    return {
        "family": normalized_family,
        "attempt_count": int(normalized_attempt_count),
        "success_count": int(normalized_success_count),
        "success_rate": float(success_rate),
        "threshold": float(normalized_threshold),
        "teacher_fallback_enabled": float(success_rate) < float(normalized_threshold),
    }


@dataclass
class SnapshotFeasibilityWorkflow:
    def execute(
        self,
        *,
        bucket_dir: Path,
        dev_dir: Path,
        collection_dir: Path,
        output_dir: Path,
        mode: str = "feasibility",
        history_k: int = state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
        teacher_threshold: float = DEFAULT_TEACHER_THRESHOLD,
        snapshot_candidates_path: Path | None = DEFAULT_SNAPSHOT_CANDIDATES,
        grouped_snapshot_candidates: Mapping[str, Mapping[str, Any]] | None = None,
        feasibility_runner: FeasibilityRunner | None = None,
    ) -> dict[str, Any]:
        if str(mode) != "feasibility":
            raise ValueError(f"unsupported mode: {mode!r}")
        if int(history_k) != int(
            state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
        ):
            raise ValueError(
                "history-k is frozen at "
                + str(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K)
            )
        normalized_threshold = _as_number(
            teacher_threshold,
            field_name="teacher_threshold",
        )
        output_dir = _validate_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prerequisites = _load_t8_prerequisites(
            bucket_dir=bucket_dir,
            dev_dir=dev_dir,
            collection_dir=collection_dir,
        )

        if grouped_snapshot_candidates is None:
            if snapshot_candidates_path is None:
                raise ValueError(
                    "snapshot_candidates_path is required when grouped candidates are omitted"
                )
            resolved_snapshot_candidates_path = _effective_snapshot_candidates_path(
                snapshot_candidates_path=snapshot_candidates_path,
                collection_dir=collection_dir,
            )
            if resolved_snapshot_candidates_path.is_file():
                normalized_grouped_candidates = load_snapshot_candidates(
                    resolved_snapshot_candidates_path
                )
                snapshot_candidates_path_str = str(resolved_snapshot_candidates_path)
            elif _is_default_snapshot_candidates_request(
                snapshot_candidates_path=snapshot_candidates_path,
                collection_dir=collection_dir,
            ):
                normalized_grouped_candidates = _normalize_grouped_snapshot_candidates(
                    synthesize_snapshot_candidates_from_real_artifacts(
                        bucket_dir=bucket_dir,
                        collection_dir=collection_dir,
                    )
                )
                snapshot_candidates_path_str = None
            else:
                raise ValueError(
                    f"missing required snapshot_candidates: {resolved_snapshot_candidates_path}"
                )
        else:
            normalized_grouped_candidates = _normalize_grouped_snapshot_candidates(
                grouped_snapshot_candidates
            )
            snapshot_candidates_path_str = None

        persisted_snapshot_candidates_path = _persist_snapshot_candidates_artifact(
            output_dir=output_dir,
            grouped_snapshot_candidates=normalized_grouped_candidates,
        )
        snapshot_candidates_context_path_str = str(persisted_snapshot_candidates_path)

        feasibility_families: list[dict[str, Any]] = []
        teacher_gate_families: list[dict[str, Any]] = []
        total_attempt_count = 0
        total_success_count = 0

        for family in T8_FAMILY_ORDER:
            group = normalized_grouped_candidates.get(family)
            if group is None:
                raise ValueError(
                    f"missing grouped snapshot candidates for family {family!r}"
                )
            eligible = [dict(item) for item in list(group.get("eligible", []))]
            ineligible = [dict(item) for item in list(group.get("ineligible", []))]
            deprioritized_flags = {
                bool(item) for item in set(group.get("deprioritized_flags", set()))
            }
            if len(deprioritized_flags) > 1:
                raise ValueError(
                    f"family {family} mixes deprioritized_by_plan=true and false across candidates"
                )
            deprioritized_by_plan = next(iter(deprioritized_flags), False)
            if not eligible:
                raise ValueError(f"family {family} has no eligible snapshot candidates")

            target_snapshot_count = (
                int(SNAPSHOTS_PER_FAMILY)
                if family in HIGH_PRIORITY_FAMILIES or not deprioritized_by_plan
                else int(DEPRIORITIZED_LOW_PRIORITY_SNAPSHOTS)
            )
            selected_snapshot_count = min(
                int(target_snapshot_count), int(len(eligible))
            )
            if family in HIGH_PRIORITY_FAMILIES:
                if selected_snapshot_count != int(SNAPSHOTS_PER_FAMILY):
                    raise ValueError(
                        f"high-priority family {family} requires exactly {SNAPSHOTS_PER_FAMILY} eligible snapshots"
                    )
                deprioritized_by_plan = False
            elif (
                selected_snapshot_count < int(SNAPSHOTS_PER_FAMILY)
                and not deprioritized_by_plan
            ):
                raise ValueError(
                    f"low-priority family {family} may only reduce attempts when deprioritized_by_plan=true"
                )

            selected_candidates = eligible[:selected_snapshot_count]
            selected_seed_values = list(SNAPSHOT_SEED_VALUES)
            success_count = 0
            attempt_count = 0
            for candidate in selected_candidates:
                for seed in selected_seed_values:
                    attempt_result = _resolve_attempt_result(
                        candidate,
                        seed=int(seed),
                        family=family,
                        feasibility_runner=feasibility_runner,
                    )
                    if evaluate_feasibility_success(
                        family=family,
                        candidate=candidate,
                        attempt_result=attempt_result,
                    ):
                        success_count += 1
                    attempt_count += 1

            if family in HIGH_PRIORITY_FAMILIES and attempt_count != int(
                SNAPSHOTS_PER_FAMILY * len(SNAPSHOT_SEED_VALUES)
            ):
                raise ValueError(
                    f"high-priority family {family} must record attempt_count=18, got {attempt_count}"
                )
            if (
                family in LOW_PRIORITY_FAMILIES
                and not deprioritized_by_plan
                and attempt_count
                != int(SNAPSHOTS_PER_FAMILY * len(SNAPSHOT_SEED_VALUES))
            ):
                raise ValueError(
                    f"low-priority family {family} must record attempt_count=18 unless deprioritized_by_plan=true"
                )

            teacher_gate = build_teacher_gate_decision(
                family=family,
                attempt_count=int(attempt_count),
                success_count=int(success_count),
                threshold=float(normalized_threshold),
            )
            feasibility_family = {
                "family": family,
                "priority": _priority_for_family(family),
                "success_criteria": FAMILY_SUCCESS_CRITERIA[family],
                "deprioritized_by_plan": bool(deprioritized_by_plan),
                "eligible_candidate_count": int(len(eligible)),
                "ineligible_candidate_count": int(len(ineligible)),
                "ineligible_candidates": [dict(item) for item in ineligible],
                "selected_snapshot_count": int(selected_snapshot_count),
                "selected_snapshot_ids": [
                    _as_non_empty_string(
                        candidate.get("snapshot_id"), field_name="snapshot_id"
                    )
                    for candidate in selected_candidates
                ],
                "selected_source_episode_ids": [
                    str(candidate.get("anchor_episode_id", ""))
                    for candidate in selected_candidates
                ],
                "selected_source_dataset_dirs": [
                    str(candidate.get("source_dataset_dir", ""))
                    for candidate in selected_candidates
                ],
                "selected_source_bucket_keys": [
                    str(candidate.get("source_bucket_key", "unknown"))
                    for candidate in selected_candidates
                ],
                "selected_source_failure_injection_kinds": [
                    str(candidate.get("source_failure_injection_kind", ""))
                    for candidate in selected_candidates
                ],
                "seed_values": [int(seed) for seed in selected_seed_values],
                "attempt_count": int(teacher_gate["attempt_count"]),
                "success_count": int(teacher_gate["success_count"]),
                "success_rate": float(teacher_gate["success_rate"]),
            }
            teacher_gate_family = {
                "family": family,
                "priority": _priority_for_family(family),
                "success_criteria": FAMILY_SUCCESS_CRITERIA[family],
                "deprioritized_by_plan": bool(deprioritized_by_plan),
                **teacher_gate,
            }
            feasibility_families.append(feasibility_family)
            teacher_gate_families.append(teacher_gate_family)
            total_attempt_count += int(attempt_count)
            total_success_count += int(success_count)

        feasibility_report = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_snapshot_feasibility_report",
            "mode": "feasibility",
            "history_k": int(
                state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
            ),
            "teacher_threshold": float(normalized_threshold),
            "output_dir": str(output_dir),
            "snapshot_candidates_path": snapshot_candidates_context_path_str,
            "snapshot_candidates_source_path": snapshot_candidates_path_str,
            "family_order": list(T8_FAMILY_ORDER),
            "counts": {
                "family_count": int(len(T8_FAMILY_ORDER)),
                "total_attempt_count": int(total_attempt_count),
                "total_success_count": int(total_success_count),
            },
            "prerequisites": dict(prerequisites),
            "families": feasibility_families,
        }
        teacher_gate_report = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_teacher_gate_report",
            "mode": "feasibility",
            "teacher_threshold": float(normalized_threshold),
            "output_dir": str(output_dir),
            "snapshot_candidates_path": snapshot_candidates_context_path_str,
            "snapshot_candidates_source_path": snapshot_candidates_path_str,
            "family_order": list(T8_FAMILY_ORDER),
            "counts": {
                "family_count": int(len(T8_FAMILY_ORDER)),
                "teacher_fallback_enabled_count": int(
                    sum(
                        1
                        for family in teacher_gate_families
                        if bool(family["teacher_fallback_enabled"])
                    )
                ),
            },
            "prerequisites": dict(prerequisites),
            "families": teacher_gate_families,
        }

        feasibility_report_path = output_dir / FEASIBILITY_REPORT_JSON_NAME
        teacher_gate_report_path = output_dir / TEACHER_GATE_REPORT_JSON_NAME
        _write_json(feasibility_report_path, feasibility_report)
        _write_json(teacher_gate_report_path, teacher_gate_report)
        return {
            "snapshot_feasibility_report_path": str(feasibility_report_path),
            "teacher_gate_report_path": str(teacher_gate_report_path),
            "family_count": int(len(T8_FAMILY_ORDER)),
            "total_attempt_count": int(total_attempt_count),
            "total_success_count": int(total_success_count),
        }


def materialize_snapshot_feasibility(
    *,
    bucket_dir: Path,
    dev_dir: Path,
    collection_dir: Path,
    output_dir: Path,
    mode: str = "feasibility",
    history_k: int = state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
    teacher_threshold: float = DEFAULT_TEACHER_THRESHOLD,
    snapshot_candidates_path: Path | None = DEFAULT_SNAPSHOT_CANDIDATES,
    grouped_snapshot_candidates: Mapping[str, Mapping[str, Any]] | None = None,
    feasibility_runner: FeasibilityRunner | None = None,
) -> dict[str, Any]:
    return SnapshotFeasibilityWorkflow().execute(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        output_dir=output_dir,
        mode=mode,
        history_k=history_k,
        teacher_threshold=teacher_threshold,
        snapshot_candidates_path=snapshot_candidates_path,
        grouped_snapshot_candidates=grouped_snapshot_candidates,
        feasibility_runner=feasibility_runner,
    )


def _normalize_report_family_rows(
    rows: object,
    *,
    report_name: str,
) -> dict[str, dict[str, Any]]:
    normalized_rows: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(
        _as_list(rows, field_name=f"{report_name}.families")
    ):
        row = dict(_as_mapping(raw_row, field_name=f"{report_name}.families[{index}]"))
        family = _normalize_family(
            row.get("family"), field_name=f"{report_name}.family"
        )
        if family in normalized_rows:
            raise ValueError(
                f"{report_name} contains duplicate family row for {family!r}"
            )
        normalized_rows[family] = row
    if list(normalized_rows.keys()) != list(T8_FAMILY_ORDER):
        raise ValueError(
            f"{report_name}.families must follow family_order {T8_FAMILY_ORDER!r}"
        )
    return normalized_rows


def _load_t9_formal_prerequisites(output_dir: Path) -> dict[str, Any]:
    resolved_output_dir = _validate_existing_dir(output_dir, arg_name="output-dir")
    feasibility_report_path = _validate_existing_file(
        resolved_output_dir / FEASIBILITY_REPORT_JSON_NAME,
        arg_name="T8 snapshot_feasibility_report_path",
    )
    teacher_gate_report_path = _validate_existing_file(
        resolved_output_dir / TEACHER_GATE_REPORT_JSON_NAME,
        arg_name="T8 teacher_gate_report_path",
    )
    feasibility_report = _read_json(feasibility_report_path)
    teacher_gate_report = _read_json(teacher_gate_report_path)
    if (
        feasibility_report.get("artifact_kind")
        != "state_conditioned_snapshot_feasibility_report"
    ):
        raise ValueError(
            "T9 formal harvest requires snapshot_feasibility_report.json artifact_kind="
            "state_conditioned_snapshot_feasibility_report"
        )
    if (
        teacher_gate_report.get("artifact_kind")
        != "state_conditioned_teacher_gate_report"
    ):
        raise ValueError(
            "T9 formal harvest requires teacher_gate_report.json artifact_kind="
            "state_conditioned_teacher_gate_report"
        )
    if feasibility_report.get("mode") != "feasibility":
        raise ValueError("T8 snapshot feasibility report must have mode='feasibility'")
    if teacher_gate_report.get("mode") != "feasibility":
        raise ValueError("T8 teacher gate report must have mode='feasibility'")
    if list(feasibility_report.get("family_order", [])) != list(T8_FAMILY_ORDER):
        raise ValueError("snapshot feasibility report family_order mismatch")
    if list(teacher_gate_report.get("family_order", [])) != list(T8_FAMILY_ORDER):
        raise ValueError("teacher gate report family_order mismatch")

    feasibility_families = _normalize_report_family_rows(
        feasibility_report.get("families"),
        report_name="snapshot_feasibility_report",
    )
    teacher_gate_families = _normalize_report_family_rows(
        teacher_gate_report.get("families"),
        report_name="teacher_gate_report",
    )
    selected_snapshot_ids_by_family: dict[str, list[str]] = {}
    producer_by_family: dict[str, str] = {}
    snapshot_candidates_report_path = feasibility_report.get("snapshot_candidates_path")
    snapshot_candidates_source_path = feasibility_report.get(
        "snapshot_candidates_source_path"
    )
    resolved_snapshot_candidates_path: str | None = None
    for raw_path in (
        snapshot_candidates_report_path,
        snapshot_candidates_source_path,
        str(_snapshot_candidates_artifact_path(resolved_output_dir)),
    ):
        if raw_path is None:
            continue
        normalized_path = str(raw_path).strip()
        if not normalized_path:
            continue
        candidate_path = Path(normalized_path).expanduser().resolve()
        if candidate_path.is_file():
            resolved_snapshot_candidates_path = str(candidate_path)
            break
    for family in T8_FAMILY_ORDER:
        feasibility_row = feasibility_families[family]
        gate_row = teacher_gate_families[family]
        selected_snapshot_ids = [
            _as_non_empty_string(
                snapshot_id,
                field_name=f"snapshot_feasibility_report.{family}.selected_snapshot_ids[]",
            )
            for snapshot_id in _as_list(
                feasibility_row.get("selected_snapshot_ids"),
                field_name=f"snapshot_feasibility_report.{family}.selected_snapshot_ids",
            )
        ]
        if not selected_snapshot_ids:
            raise ValueError(
                f"T9 formal harvest requires non-empty selected_snapshot_ids for family {family}"
            )
        selected_snapshot_ids_by_family[family] = selected_snapshot_ids
        _ = _as_number(
            gate_row.get("success_rate"),
            field_name=f"teacher_gate_report.{family}.success_rate",
        )
        _ = _as_number(
            gate_row.get("threshold"),
            field_name=f"teacher_gate_report.{family}.threshold",
        )
        producer_by_family[family] = (
            PRODUCER_SCRIPTED_TEACHER
            if _as_bool(
                gate_row.get("teacher_fallback_enabled"),
                field_name=f"teacher_gate_report.{family}.teacher_fallback_enabled",
            )
            else PRODUCER_BASE_POLICY
        )
    return {
        "output_dir": str(resolved_output_dir),
        "snapshot_feasibility_report_path": str(feasibility_report_path),
        "teacher_gate_report_path": str(teacher_gate_report_path),
        "snapshot_feasibility_report": feasibility_report,
        "teacher_gate_report": teacher_gate_report,
        "snapshot_candidates_path": resolved_snapshot_candidates_path,
        "selected_snapshot_ids_by_family": selected_snapshot_ids_by_family,
        "teacher_gate_families": teacher_gate_families,
        "producer_by_family": producer_by_family,
    }


def _selected_candidates_for_formal(
    *,
    normalized_grouped_candidates: Mapping[str, Mapping[str, Any]],
    selected_snapshot_ids_by_family: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, Any]]]:
    selected_by_family: dict[str, list[dict[str, Any]]] = {}
    for family in T8_FAMILY_ORDER:
        group = _as_mapping(
            normalized_grouped_candidates.get(family),
            field_name=f"grouped_snapshot_candidates[{family}]",
        )
        eligible_by_snapshot_id = {
            _as_non_empty_string(
                candidate.get("snapshot_id"),
                field_name=f"grouped_snapshot_candidates[{family}].snapshot_id",
            ): dict(candidate)
            for candidate in list(group.get("eligible", []))
        }
        selected_candidates: list[dict[str, Any]] = []
        for snapshot_id in selected_snapshot_ids_by_family[family]:
            candidate = eligible_by_snapshot_id.get(str(snapshot_id))
            if candidate is None:
                raise ValueError(
                    f"T9 formal harvest missing selected snapshot candidate {snapshot_id!r} for family {family}"
                )
            selected_candidates.append(candidate)
        selected_by_family[family] = selected_candidates
    return selected_by_family


def _resolve_formal_snapshot_candidates_path(
    *,
    snapshot_candidates_path: Path | None,
    formal_prerequisites: Mapping[str, Any],
) -> Path:
    if snapshot_candidates_path is not None:
        cli_path = snapshot_candidates_path.expanduser().resolve()
        if cli_path.is_file():
            return cli_path
    report_path = formal_prerequisites.get("snapshot_candidates_path")
    if isinstance(report_path, str) and report_path.strip():
        resolved_report_path = Path(report_path).expanduser().resolve()
        if resolved_report_path.is_file():
            return resolved_report_path
    raise ValueError(
        "formal harvest requires snapshot candidate context from the same output-dir "
        "T8 artifacts or an explicit --snapshot-candidates file"
    )


def _resolve_formal_attempt_result(
    candidate: Mapping[str, Any],
    *,
    seed: int,
    family: str,
    producer: str,
    formal_runner: FormalRolloutRunner | None,
) -> dict[str, Any]:
    if formal_runner is not None:
        result = formal_runner(candidate, int(seed), family, producer)
        return dict(_as_mapping(result, field_name="formal_runner result"))
    if producer == PRODUCER_BASE_POLICY:
        return _resolve_attempt_result(
            candidate,
            seed=int(seed),
            family=family,
            feasibility_runner=None,
        )
    raw_teacher_attempts = candidate.get("scripted_teacher_attempts")
    if raw_teacher_attempts is None:
        return _run_on_demand_scripted_teacher_attempt(
            candidate,
            seed=int(seed),
            family=family,
        )
    for index, raw_attempt in enumerate(
        _normalize_attempt_records(
            raw_teacher_attempts,
            field_name="candidate.scripted_teacher_attempts",
        )
    ):
        attempt_seed = _as_int(
            raw_attempt.get("seed"),
            field_name=f"candidate.scripted_teacher_attempts[{index}].seed",
        )
        if attempt_seed == int(seed):
            attempt = dict(raw_attempt)
            if (
                producer == PRODUCER_SCRIPTED_TEACHER
                and attempt.get("teacher_target") is None
                and not str(attempt.get("teacher_rollout_kind", "")).strip()
                and (
                    attempt.get("episode_record") is not None
                    and (
                        attempt.get("transition_records") is not None
                        or attempt.get("transitions") is not None
                    )
                )
            ):
                attempt["teacher_rollout_kind"] = TEACHER_ROLLOUT_KIND_CACHED_REAL
            return attempt
    raise ValueError(
        f"candidate {candidate['snapshot_id']!r} is missing scripted teacher attempt payload for seed={seed}"
    )


def _run_on_demand_scripted_teacher_attempt(
    candidate: Mapping[str, Any],
    *,
    seed: int,
    family: str,
) -> dict[str, Any]:
    source_dataset_dir_raw = candidate.get("source_dataset_dir")
    if source_dataset_dir_raw is None:
        attempt = _run_replay_based_feasibility_attempt(
            candidate, seed=seed, family=family
        )
        attempt["producer"] = PRODUCER_SCRIPTED_TEACHER
        attempt["teacher_rollout_kind"] = TEACHER_ROLLOUT_KIND_SOURCE_SIDECAR_REPLAY
        return attempt

    dataset_dir = (
        Path(
            _as_non_empty_string(
                source_dataset_dir_raw,
                field_name="candidate.source_dataset_dir",
            )
        )
        .expanduser()
        .resolve()
    )
    episode_id = _as_non_empty_string(
        candidate.get("anchor_episode_id"), field_name="candidate.anchor_episode_id"
    )
    source_rows = _dataset_sidecar_rows(dataset_dir, episode_id=episode_id)
    action_chunks = _load_replay_action_chunks(candidate)
    anchor_t = _as_int(candidate.get("anchor_t"), field_name="candidate.anchor_t")
    source_seed = _as_int(
        candidate.get("source_episode_seed"), field_name="candidate.source_episode_seed"
    )
    if _normalize_family(family) == "S_lost":
        attempt = _run_s_lost_recorded_source_teacher_rollout(
            source_rows=source_rows,
            action_chunks=action_chunks,
            anchor_t=int(anchor_t),
            source_seed=int(source_seed),
        )
        attempt["source_episode_id"] = episode_id
        attempt["replay_anchor_t"] = int(anchor_t)
        attempt["replayed_policy_step_count"] = int(
            min(int(anchor_t) + 1, int(len(action_chunks)))
        )
        attempt["producer"] = PRODUCER_SCRIPTED_TEACHER
        attempt["teacher_rollout_kind"] = TEACHER_ROLLOUT_KIND_RECORDED_SOURCE_TRACE
        return attempt
    env = _build_replay_env(
        candidate, n_action_steps=recap_collector.extract_T_action(action_chunks[0])
    )
    try:
        try:
            obs_raw, _reset_info = env.reset(seed=int(source_seed))
        except TypeError:
            obs_raw, _reset_info = env.reset()
        obs = dict(_as_mapping(obs_raw, field_name="env.reset(obs)"))
        replay_limit = min(int(anchor_t) + 1, int(len(action_chunks)))
        for step_index in range(replay_limit):
            next_obs_raw, _reward, _terminated, _truncated, _info = env.step(
                _normalize_replay_action_chunk_for_env(action_chunks[step_index])
            )
            obs = dict(_as_mapping(next_obs_raw, field_name="env.step(next_obs)"))
        normalized_family = _normalize_family(family)
        if normalized_family == "S_drop":
            attempt = _run_s_drop_local_teacher_rollout(
                candidate,
                env=env,
                obs=obs,
                action_chunks=action_chunks,
                anchor_t=int(anchor_t),
                source_seed=int(source_seed),
            )
        elif normalized_family == "S_lost":
            attempt = _run_s_lost_local_teacher_rollout(
                candidate,
                env=env,
                action_chunks=action_chunks,
                anchor_t=int(anchor_t),
                source_seed=int(source_seed),
            )
        else:
            attempt = _run_recorded_action_local_teacher_rollout(
                candidate,
                family=family,
                env=env,
                obs=obs,
                action_chunks=action_chunks,
                anchor_t=int(anchor_t),
                source_seed=int(source_seed),
            )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
    attempt["source_episode_id"] = episode_id
    attempt["replay_anchor_t"] = int(anchor_t)
    attempt["replayed_policy_step_count"] = int(
        min(int(anchor_t) + 1, int(len(action_chunks)))
    )
    attempt["producer"] = PRODUCER_SCRIPTED_TEACHER
    attempt["teacher_rollout_kind"] = TEACHER_ROLLOUT_KIND_LOCAL_SIM
    return attempt


def _attempt_result_to_local_rollout_records(
    *,
    candidate: Mapping[str, Any],
    attempt_result: Mapping[str, Any],
    family: str,
    producer: str,
    seed: int,
    rollout_success: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot_id = _as_non_empty_string(
        candidate.get("snapshot_id"),
        field_name="candidate.snapshot_id",
    )
    default_episode_id = (
        f"{snapshot_id}__{family.lower()}__{producer}__seed{int(seed):03d}"
    )
    raw_episode_record = attempt_result.get("episode_record")
    episode_record = (
        dict(
            _as_mapping(raw_episode_record, field_name="attempt_result.episode_record")
        )
        if raw_episode_record is not None
        else {}
    )
    episode_id = _as_non_empty_string(
        episode_record.get(
            "episode_id", attempt_result.get("episode_id", default_episode_id)
        ),
        field_name="attempt_result.episode_id",
    )
    existing_metadata = episode_record.get(
        "metadata",
        attempt_result.get("metadata", attempt_result.get("episode_metadata", {})),
    )
    if existing_metadata is None:
        metadata: dict[str, Any] = {}
    else:
        metadata = dict(
            _as_mapping(existing_metadata, field_name="attempt_result.metadata")
        )
    analysis_only_raw = metadata.get("analysis_only", {})
    analysis_only = dict(
        _as_mapping(
            analysis_only_raw, field_name="attempt_result.metadata.analysis_only"
        )
    )
    recovery_entry_step = attempt_result.get("recovery_entry_step")
    if recovery_entry_step is not None:
        analysis_only["recovery_entry_step"] = _as_int(
            recovery_entry_step,
            field_name="attempt_result.recovery_entry_step",
        )
    if analysis_only:
        metadata["analysis_only"] = analysis_only
    if metadata:
        episode_record["metadata"] = metadata
    episode_record["episode_id"] = episode_id
    episode_record["success_episode"] = bool(
        attempt_result.get("success_episode", False) or rollout_success
    )

    raw_transition_records = attempt_result.get(
        "transition_records",
        attempt_result.get("transitions"),
    )
    if raw_transition_records is not None:
        transition_records = [
            dict(
                _as_mapping(
                    item, field_name=f"attempt_result.transition_records[{index}]"
                )
            )
            for index, item in enumerate(
                _as_list(
                    raw_transition_records,
                    field_name="attempt_result.transition_records",
                )
            )
        ]
        for index, record in enumerate(transition_records):
            record.setdefault("episode_id", episode_id)
            record.setdefault("t", int(index))
            record.setdefault("success_step", False)
        return episode_record, transition_records

    transition_records: list[dict[str, Any]] = []
    for index, step in enumerate(_policy_steps_from_attempt_result(attempt_result)):
        step_payload = dict(
            _as_mapping(step, field_name=f"attempt_result.policy_steps[{index}]")
        )
        step_t = _as_int(
            step_payload.get("t", index),
            field_name=f"attempt_result.policy_steps[{index}].t",
        )
        transition_records.append(
            {
                "episode_id": episode_id,
                "t": int(step_t),
                "success_step": bool(step_payload.get("success_step", False)),
                "policy_step": step_payload,
            }
        )
    return episode_record, transition_records


def _teacher_trigger_reason(*, producer: str) -> str:
    return (
        "teacher_gate_success_rate_below_threshold"
        if producer == PRODUCER_SCRIPTED_TEACHER
        else "teacher_gate_success_rate_at_or_above_threshold"
    )


def _build_failed_rollout_analysis_entry(
    *,
    candidate: Mapping[str, Any],
    family: str,
    producer: str,
    seed: int,
    reason: str,
    episode_record: Mapping[str, Any] | None = None,
    transition_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    entry = {
        "source_snapshot_id": _as_non_empty_string(
            candidate.get("snapshot_id"),
            field_name="candidate.snapshot_id",
        ),
        "source_snapshot_family": str(family),
        "source_bucket_key": str(candidate.get("source_bucket_key", "")),
        "producer": str(producer),
        "seed": int(seed),
        "included_in_pseudodemo_manifest": False,
        "reason": str(reason),
    }
    if episode_record is not None:
        episode_id = str(episode_record.get("episode_id", "")).strip()
        if episode_id:
            entry["episode_id"] = episode_id
        if "success_episode" in episode_record:
            entry["success_episode"] = bool(
                episode_record.get("success_episode", False)
            )
    if transition_records is not None:
        entry["transition_count"] = int(len(transition_records))
    return entry


def _build_local_recovery_pseudodemo_record(
    *,
    candidate: Mapping[str, Any],
    attempt_result: Mapping[str, Any],
    family: str,
    producer: str,
    seed: int,
    teacher_gate_row: Mapping[str, Any],
    teacher_version: str,
    episode_record: Mapping[str, Any],
    transition_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    split_summary = recap_collector.summarize_local_recovery_rollout_for_pseudodemo(
        episode_record=episode_record,
        transition_records=transition_records,
    )
    if not bool(split_summary.get("included_in_pseudodemo_manifest", False)):
        raise ValueError(
            "failed local rollouts must not enter local_recovery_pseudodemo_manifest.json"
        )
    source_bucket_key = _normalize_source_bucket_key(
        candidate.get("source_bucket_key"),
        family=family,
        field_name="candidate.source_bucket_key",
    )
    teacher_target, teacher_target_truthfulness = _resolve_teacher_target_payload(
        attempt_result=attempt_result,
        producer=producer,
        teacher_version=teacher_version,
        split_summary=split_summary,
    )
    record: dict[str, Any] = {
        "episode_id": _as_non_empty_string(
            split_summary.get("episode_id"),
            field_name="split_summary.episode_id",
        ),
        "seed": int(seed),
        "producer": str(producer),
        "teacher_version": _as_non_empty_string(
            teacher_version,
            field_name="teacher_version",
        ),
        "teacher_trigger_reason": _teacher_trigger_reason(producer=producer),
        "teacher_trigger_success_rate": _as_number(
            teacher_gate_row.get("success_rate"),
            field_name=f"teacher_gate_report.{family}.success_rate",
        ),
        "teacher_trigger_threshold": _as_number(
            teacher_gate_row.get("threshold"),
            field_name=f"teacher_gate_report.{family}.threshold",
        ),
        "source_snapshot_id": _as_non_empty_string(
            candidate.get("snapshot_id"),
            field_name="candidate.snapshot_id",
        ),
        "source_snapshot_family": str(family),
        "source_bucket_key": source_bucket_key,
        "source_snapshot_history_k": _as_int(
            candidate.get("history_k"),
            field_name="candidate.history_k",
        ),
        "failure_prefix_step_count": _as_int(
            split_summary.get("failure_prefix_step_count"),
            field_name="split_summary.failure_prefix_step_count",
        ),
        "failure_prefix_source_episode_id": _as_non_empty_string(
            split_summary.get("failure_prefix_source_episode_id"),
            field_name="split_summary.failure_prefix_source_episode_id",
        ),
        "failure_prefix_source_t_range": list(
            _as_list(
                split_summary.get("failure_prefix_source_t_range"),
                field_name="split_summary.failure_prefix_source_t_range",
                expected_len=2,
            )
        ),
        "recovery_suffix_step_count": _as_int(
            split_summary.get("recovery_suffix_step_count"),
            field_name="split_summary.recovery_suffix_step_count",
        ),
        "recovery_suffix_source_episode_id": _as_non_empty_string(
            split_summary.get("recovery_suffix_source_episode_id"),
            field_name="split_summary.recovery_suffix_source_episode_id",
        ),
        "recovery_suffix_source_t_range": list(
            _as_list(
                split_summary.get("recovery_suffix_source_t_range"),
                field_name="split_summary.recovery_suffix_source_t_range",
                expected_len=2,
            )
        ),
        "policy_condition.phase": _as_non_empty_string(
            candidate.get("policy_condition.phase"),
            field_name="candidate.policy_condition.phase",
        ),
        "policy_condition.mode": _as_non_empty_string(
            candidate.get("policy_condition.mode"),
            field_name="candidate.policy_condition.mode",
        ),
        "policy_condition_text": _as_non_empty_string(
            candidate.get("policy_condition_text"),
            field_name="candidate.policy_condition_text",
        ),
        "anchor_episode_id": _as_non_empty_string(
            candidate.get("anchor_episode_id"),
            field_name="candidate.anchor_episode_id",
        ),
        "anchor_t": _as_int(candidate.get("anchor_t"), field_name="candidate.anchor_t"),
        "teacher_target_truthfulness": str(teacher_target_truthfulness),
    }
    if teacher_target is not None:
        record["teacher_target"] = dict(teacher_target)
    return recap_episode_writer.validate_local_recovery_pseudodemo_record(record)


def _select_successful_pseudodemos(
    successful_by_family: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    selected_counts_by_family: dict[str, int] = {
        family: 0 for family in T8_FAMILY_ORDER
    }
    for family in HIGH_PRIORITY_FAMILIES:
        family_successes = [dict(item) for item in successful_by_family.get(family, [])]
        if len(family_successes) < int(FORMAL_HIGH_PRIORITY_FLOOR):
            raise ValueError(
                f"formal harvest floor unmet for {family}: required {FORMAL_HIGH_PRIORITY_FLOOR}, got {len(family_successes)}"
            )
        for item in family_successes[:FORMAL_HIGH_PRIORITY_FLOOR]:
            selected.append(item)
            selected_counts_by_family[family] += 1

    remaining = int(FORMAL_SUCCESSFUL_PSEUDODEMO_TARGET - len(selected))
    for family in T8_FAMILY_ORDER:
        start_index = (
            FORMAL_HIGH_PRIORITY_FLOOR if family in HIGH_PRIORITY_FAMILIES else 0
        )
        family_successes = [dict(item) for item in successful_by_family.get(family, [])]
        for item in family_successes[start_index:]:
            if remaining <= 0:
                break
            selected.append(item)
            selected_counts_by_family[family] += 1
            remaining -= 1
        if remaining <= 0:
            break
    if remaining > 0:
        raise ValueError(
            "formal harvest requires at least 24 successful pseudo-demos after enforcing floors, "
            + f"got {FORMAL_SUCCESSFUL_PSEUDODEMO_TARGET - remaining}"
        )
    return selected, selected_counts_by_family


def _manifest_producer_by_family(
    pseudodemos: Sequence[Mapping[str, Any]],
) -> dict[str, str | None]:
    producer_by_family: dict[str, str | None] = {
        family: None for family in T8_FAMILY_ORDER
    }
    seen: dict[str, set[str]] = {family: set() for family in T8_FAMILY_ORDER}
    for raw_record in pseudodemos:
        record = dict(_as_mapping(raw_record, field_name="pseudodemos[]"))
        family = _normalize_family(
            record.get("source_snapshot_family"),
            field_name="pseudodemos[].source_snapshot_family",
        )
        producer = _as_non_empty_string(
            record.get("producer"),
            field_name="pseudodemos[].producer",
        )
        seen[family].add(producer)
    for family, family_producers in seen.items():
        if len(family_producers) > 1:
            raise ValueError(
                f"formal pseudodemo manifest cannot mix producers within family {family}: {sorted(family_producers)!r}"
            )
        if family_producers:
            producer_by_family[family] = next(iter(family_producers))
    return producer_by_family


@dataclass
class FormalPseudodemoHarvestWorkflow:
    def execute(
        self,
        *,
        bucket_dir: Path,
        dev_dir: Path,
        collection_dir: Path,
        output_dir: Path,
        mode: str = "formal",
        history_k: int = state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
        teacher_threshold: float = DEFAULT_TEACHER_THRESHOLD,
        teacher_version: str = DEFAULT_TEACHER_VERSION,
        snapshot_candidates_path: Path | None = DEFAULT_SNAPSHOT_CANDIDATES,
        grouped_snapshot_candidates: Mapping[str, Mapping[str, Any]] | None = None,
        formal_runner: FormalRolloutRunner | None = None,
    ) -> dict[str, Any]:
        if str(mode) != "formal":
            raise ValueError(f"unsupported mode: {mode!r}")
        if int(history_k) != int(
            state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
        ):
            raise ValueError(
                "history-k is frozen at "
                + str(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K)
            )
        _ = _as_number(teacher_threshold, field_name="teacher_threshold")
        _ = _as_non_empty_string(teacher_version, field_name="teacher_version")
        _ = _load_t8_prerequisites(
            bucket_dir=bucket_dir,
            dev_dir=dev_dir,
            collection_dir=collection_dir,
        )
        formal_prerequisites = _load_t9_formal_prerequisites(output_dir)
        report_threshold = _as_number(
            formal_prerequisites["teacher_gate_report"].get("teacher_threshold"),
            field_name="teacher_gate_report.teacher_threshold",
        )
        if abs(float(report_threshold) - float(teacher_threshold)) > 1e-12:
            raise ValueError(
                "formal harvest must use teacher_threshold from teacher_gate_report.json; "
                + f"report={report_threshold} cli={teacher_threshold}"
            )

        if grouped_snapshot_candidates is None:
            resolved_snapshot_candidates_path = (
                _resolve_formal_snapshot_candidates_path(
                    snapshot_candidates_path=snapshot_candidates_path,
                    formal_prerequisites=formal_prerequisites,
                )
            )
            normalized_grouped_candidates = load_snapshot_candidates(
                resolved_snapshot_candidates_path
            )
            snapshot_candidates_path_str = str(resolved_snapshot_candidates_path)
        else:
            normalized_grouped_candidates = _normalize_grouped_snapshot_candidates(
                grouped_snapshot_candidates
            )
            snapshot_candidates_path_str = None

        selected_candidates_by_family = _selected_candidates_for_formal(
            normalized_grouped_candidates=normalized_grouped_candidates,
            selected_snapshot_ids_by_family=formal_prerequisites[
                "selected_snapshot_ids_by_family"
            ],
        )

        successful_by_family: dict[str, list[dict[str, Any]]] = {
            family: [] for family in T8_FAMILY_ORDER
        }
        analysis_entries: list[dict[str, Any]] = []
        attempted_rollout_count = 0
        successful_local_rollout_count = 0

        for family in T8_FAMILY_ORDER:
            teacher_gate_row = dict(
                formal_prerequisites["teacher_gate_families"][family]
            )
            producer = str(formal_prerequisites["producer_by_family"][family])
            for candidate in selected_candidates_by_family[family]:
                for seed in SNAPSHOT_SEED_VALUES:
                    attempted_rollout_count += 1
                    attempt_result = _resolve_formal_attempt_result(
                        candidate,
                        seed=int(seed),
                        family=family,
                        producer=producer,
                        formal_runner=formal_runner,
                    )
                    rollout_success = evaluate_feasibility_success(
                        family=family,
                        candidate=candidate,
                        attempt_result=attempt_result,
                    )
                    episode_record, transition_records = (
                        _attempt_result_to_local_rollout_records(
                            candidate=candidate,
                            attempt_result=attempt_result,
                            family=family,
                            producer=producer,
                            seed=int(seed),
                            rollout_success=bool(rollout_success),
                        )
                    )
                    if not rollout_success:
                        analysis_entries.append(
                            _build_failed_rollout_analysis_entry(
                                candidate=candidate,
                                family=family,
                                producer=producer,
                                seed=int(seed),
                                reason="family_success_criteria_not_met",
                                episode_record=episode_record,
                                transition_records=transition_records,
                            )
                        )
                        continue
                    successful_local_rollout_count += 1
                    try:
                        successful_by_family[family].append(
                            _build_local_recovery_pseudodemo_record(
                                candidate=candidate,
                                attempt_result=attempt_result,
                                family=family,
                                producer=producer,
                                seed=int(seed),
                                teacher_gate_row=teacher_gate_row,
                                teacher_version=teacher_version,
                                episode_record=episode_record,
                                transition_records=transition_records,
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        analysis_entries.append(
                            _build_failed_rollout_analysis_entry(
                                candidate=candidate,
                                family=family,
                                producer=producer,
                                seed=int(seed),
                                reason=_exception_message(exc),
                                episode_record=episode_record,
                                transition_records=transition_records,
                            )
                        )

        selected_pseudodemos, selected_counts_by_family = (
            _select_successful_pseudodemos(successful_by_family)
        )
        manifest_producer_by_family = _manifest_producer_by_family(selected_pseudodemos)
        analysis_path = output_dir / LOCAL_RECOVERY_ROLLOUT_ANALYSIS_JSON_NAME
        manifest_path = output_dir / LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME
        analysis_payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "local_recovery_rollout_analysis",
            "mode": "formal",
            "output_dir": str(output_dir),
            "snapshot_candidates_path": snapshot_candidates_path_str,
            "snapshot_feasibility_report_path": formal_prerequisites[
                "snapshot_feasibility_report_path"
            ],
            "teacher_gate_report_path": formal_prerequisites[
                "teacher_gate_report_path"
            ],
            "counts": {
                "attempted_rollout_count": int(attempted_rollout_count),
                "successful_local_rollout_count": int(successful_local_rollout_count),
                "analysis_only_failed_rollout_count": int(len(analysis_entries)),
            },
            "failed_rollouts": analysis_entries,
        }
        manifest_payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "local_recovery_pseudodemo_manifest",
            "mode": "formal",
            "output_dir": str(output_dir),
            "history_k": int(
                state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
            ),
            "teacher_version": str(teacher_version),
            "snapshot_candidates_path": snapshot_candidates_path_str,
            "snapshot_feasibility_report_path": formal_prerequisites[
                "snapshot_feasibility_report_path"
            ],
            "teacher_gate_report_path": formal_prerequisites[
                "teacher_gate_report_path"
            ],
            "family_order": list(T8_FAMILY_ORDER),
            "counts": {
                "attempted_rollout_count": int(attempted_rollout_count),
                "successful_local_rollout_count": int(successful_local_rollout_count),
                "successful_pseudodemo_count": int(len(selected_pseudodemos)),
                "analysis_only_failed_rollout_count": int(len(analysis_entries)),
                "selected_pseudodemo_count_by_family": dict(selected_counts_by_family),
            },
            "successful_pseudodemo_count": int(len(selected_pseudodemos)),
            "required_floor_by_family": {
                family: int(FORMAL_HIGH_PRIORITY_FLOOR)
                for family in HIGH_PRIORITY_FAMILIES
            },
            "producer_by_family": dict(manifest_producer_by_family),
            "analysis_evidence_path": str(analysis_path),
            "pseudodemos": selected_pseudodemos,
        }
        _write_json(analysis_path, analysis_payload)
        _write_json(manifest_path, manifest_payload)
        return {
            "local_recovery_pseudodemo_manifest_path": str(manifest_path),
            "local_recovery_rollout_analysis_path": str(analysis_path),
            "successful_pseudodemo_count": int(len(selected_pseudodemos)),
            "selected_pseudodemo_count_by_family": dict(selected_counts_by_family),
        }


def materialize_formal_pseudodemos(
    *,
    bucket_dir: Path,
    dev_dir: Path,
    collection_dir: Path,
    output_dir: Path,
    mode: str = "formal",
    history_k: int = state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
    teacher_threshold: float = DEFAULT_TEACHER_THRESHOLD,
    teacher_version: str = DEFAULT_TEACHER_VERSION,
    snapshot_candidates_path: Path | None = DEFAULT_SNAPSHOT_CANDIDATES,
    grouped_snapshot_candidates: Mapping[str, Mapping[str, Any]] | None = None,
    formal_runner: FormalRolloutRunner | None = None,
) -> dict[str, Any]:
    return FormalPseudodemoHarvestWorkflow().execute(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        output_dir=output_dir,
        mode=mode,
        history_k=history_k,
        teacher_threshold=teacher_threshold,
        teacher_version=teacher_version,
        snapshot_candidates_path=snapshot_candidates_path,
        grouped_snapshot_candidates=grouped_snapshot_candidates,
        formal_runner=formal_runner,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "feasibility":
            result = materialize_snapshot_feasibility(
                bucket_dir=args.bucket_dir,
                dev_dir=args.dev_dir,
                collection_dir=args.collection_dir,
                output_dir=args.output_dir,
                mode=args.mode,
                history_k=int(args.history_k),
                teacher_threshold=float(args.teacher_threshold),
                snapshot_candidates_path=args.snapshot_candidates,
            )
        else:
            result = materialize_formal_pseudodemos(
                bucket_dir=args.bucket_dir,
                dev_dir=args.dev_dir,
                collection_dir=args.collection_dir,
                output_dir=args.output_dir,
                mode=args.mode,
                history_k=int(args.history_k),
                teacher_threshold=float(args.teacher_threshold),
                teacher_version=str(args.teacher_version),
                snapshot_candidates_path=args.snapshot_candidates,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_COLLECTION_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SNAPSHOT_CANDIDATES",
    "DEFAULT_TEACHER_THRESHOLD",
    "DEFAULT_TEACHER_VERSION",
    "FAMILY_SUCCESS_CRITERIA",
    "FEASIBILITY_REPORT_JSON_NAME",
    "HIGH_PRIORITY_FAMILIES",
    "LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME",
    "LOCAL_RECOVERY_ROLLOUT_ANALYSIS_JSON_NAME",
    "LOW_PRIORITY_FAMILIES",
    "PRODUCER_BASE_POLICY",
    "PRODUCER_SCRIPTED_TEACHER",
    "SCHEMA_VERSION",
    "SNAPSHOT_SEED_VALUES",
    "SNAPSHOTS_PER_FAMILY",
    "T8_FAMILY_ORDER",
    "TEACHER_GATE_REPORT_JSON_NAME",
    "build_parser",
    "build_teacher_gate_decision",
    "evaluate_feasibility_success",
    "load_snapshot_candidates",
    "main",
    "materialize_formal_pseudodemos",
    "materialize_snapshot_feasibility",
]


@dataclass
class SnapshotHarvestWorkflow:
    def materialize_feasibility(self, **kwargs):
        return materialize_snapshot_feasibility(**kwargs)

    def materialize_formal(self, **kwargs):
        return materialize_formal_pseudodemos(**kwargs)

    def run_cli(self, argv=None) -> int:
        return main(argv)


class StateConditionedSnapshotHarvestScriptApp:
    def build_parser(self):
        return build_parser()

    def materialize_snapshot_feasibility(self, **kwargs):
        return SnapshotHarvestWorkflow().materialize_feasibility(**kwargs)

    def materialize_formal_pseudodemos(self, **kwargs):
        return SnapshotHarvestWorkflow().materialize_formal(**kwargs)

    def run(self, argv=None) -> int:
        return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
