#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_HARVEST_DIR = Path("agent/artifacts/state_conditioned_materialization/harvest")
DEFAULT_TRAINING_DIR = Path(
    "agent/artifacts/state_conditioned_materialization/training"
)
DEFAULT_EVAL_DIR = Path("agent/artifacts/state_conditioned_materialization/eval")
DEFAULT_OUTPUT = Path(
    "agent/artifacts/state_conditioned_materialization/freeze/wave_freeze_manifest.json"
)

SCHEMA_VERSION = "state_conditioned_wave_freeze_manifest_v1"
IMMUTABLE_ARCHIVE_DIRNAME = "immutable_archive"
IMMUTABLE_ARCHIVE_SUBDIR = "wave_manifest"
ALLOWED_CHANGE_POLICY = "ONLY_LABELS_AND_DATA_VERSION_CHANGES"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_build_training_set
from work.recap import state_conditioned_oracle_eval
from work.recap import state_conditioned_snapshot_harvest
from work.recap import state_conditioned_train


OWNER_CODE_MAP = {
    "T8": "work/recap/scripts/state_conditioned_snapshot_harvest.py",
    "T9": "work/recap/scripts/state_conditioned_snapshot_harvest.py",
    "T10": "work/recap/scripts/state_conditioned_build_training_set.py",
    "T11": "work/recap/scripts/state_conditioned_train.py",
    "T12": "work/recap/scripts/state_conditioned_oracle_eval.py",
}

JSON_DIGEST_SELECTORS: dict[str, tuple[str, ...]] = {
    "T8.snapshot_feasibility_report_json": (
        "artifact_kind",
        "schema_version",
        "family_order",
        "history_k",
        "teacher_threshold",
        "counts",
    ),
    "T8.teacher_gate_report_json": (
        "artifact_kind",
        "schema_version",
        "family_order",
        "teacher_threshold",
        "counts",
    ),
    "T9.local_recovery_rollout_analysis_json": (),
    "T9.local_recovery_pseudodemo_manifest_json": (
        "artifact_kind",
        "schema_version",
        "family_order",
        "history_k",
        "teacher_version",
        "required_floor_by_family",
        "counts",
    ),
    "T10.state_conditioned_sft_stats_json": (
        "artifact_kind",
        "schema_version",
        "deployable_observation_allowlist",
        "recovery_oversample_factor_min",
        "recovery_oversample_factor_max",
        "views",
        "counts",
    ),
    "T10.equal_data_fairness_audit_json": (
        "artifact_kind",
        "schema_version",
        "overall_pass",
        "comparisons",
    ),
    "T10.conditioning_channel_liveness_json": (
        "artifact_kind",
        "schema_version",
        "overall_pass",
        "differing_only_fields",
        "c1_distinct_phase_values",
        "c1_distinct_mode_values",
        "counts",
    ),
    "T10.dev_only_promotion_gate_json": (
        "artifact_kind",
        "schema_version",
        "promotion_allowed",
        "checks",
        "failure_reasons",
    ),
    "T10.lerobot_dataset_meta_info_json": (),
    "T11.run_metadata_c0_json": (
        "artifact_kind",
        "schema_version",
        "variant_key",
        "comparable_run_spec.dataset_fingerprint",
        "comparable_run_spec.conditioning_enabled",
        "comparable_run_spec.null_phase_mode_token_enabled",
        "comparable_run_spec.training_budget",
        "comparable_run_spec.optimizer_schedule",
        "comparable_run_spec.sampling",
        "comparable_run_spec.stable_base",
    ),
    "T11.run_metadata_c1_json": (
        "artifact_kind",
        "schema_version",
        "variant_key",
        "comparable_run_spec.dataset_fingerprint",
        "comparable_run_spec.conditioning_enabled",
        "comparable_run_spec.null_phase_mode_token_enabled",
        "comparable_run_spec.training_budget",
        "comparable_run_spec.optimizer_schedule",
        "comparable_run_spec.sampling",
        "comparable_run_spec.stable_base",
    ),
    "T11.delegate_summary_c0_json": (
        "wrapper_status",
        "delegate_mode",
        "wrapper",
        "effective_config",
        "upstream_returncode",
    ),
    "T11.delegate_summary_c1_json": (
        "wrapper_status",
        "delegate_mode",
        "wrapper",
        "effective_config",
        "upstream_returncode",
    ),
    "T11.state_conditioned_training_fairness_diff_whitelist_json": (
        "artifact_kind",
        "schema_version",
        "status",
        "allowed_difference_paths",
        "observed_difference_paths",
        "same_dataset_fingerprint",
        "same_equal_data_fairness_audit_path",
    ),
    "T11.selected_checkpoint_c0_config_json": (),
    "T11.selected_checkpoint_c0_model_index_json": (),
    "T11.selected_checkpoint_c1_config_json": (),
    "T11.selected_checkpoint_c1_model_index_json": (),
    "T12.oracle_conditioned_dev_scorecard_json": (
        "artifact_kind",
        "line_order",
        "line_labels",
        "comparable_metric_names",
    ),
    "T12.oracle_gate_decision_json": (),
    "T12.recovery_benchmark_summary_json": (),
    "T12.result_split_decision_json": (),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="state_conditioned_wave_freeze_manifest.py",
        description=(
            "Freeze the current T8/T9/T10/T11/T12 state-conditioned wave into a "
            "deterministic manifest plus immutable archive copies."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--harvest-dir", type=Path, default=DEFAULT_HARVEST_DIR)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_dir(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"{arg_name} directory does not exist: {resolved}")
    return resolved


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"output path must be a file, got directory: {resolved}")
    if resolved.name in {"", ".", ".."}:
        raise ValueError(f"output path must name a JSON file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                value = json.loads(stripped)
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"JSONL record must be an object in {path}, got {type(value).__name__}"
                    )
                records.append(dict(value))
    return records


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    return path


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _schema_digest_for_json(payload: Mapping[str, Any]) -> str:
    schema_payload: dict[str, object] = {}
    artifact_kind = payload.get("artifact_kind")
    schema_version = payload.get("schema_version")
    if artifact_kind is not None:
        schema_payload["artifact_kind"] = artifact_kind
    if schema_version is not None:
        schema_payload["schema_version"] = schema_version
    if not schema_payload:
        schema_payload = {
            "top_level_keys": sorted(str(key) for key in payload.keys()),
        }
    return _sha256_bytes(_canonical_json_bytes(schema_payload))


def _deep_get(payload: Mapping[str, Any], dotted_path: str) -> object | None:
    current: object = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _selected_digest_payload(
    payload: Mapping[str, Any], selectors: Sequence[str]
) -> Mapping[str, object] | object:
    if not selectors:
        return payload
    selected: dict[str, object] = {}
    for selector in selectors:
        value = _deep_get(payload, str(selector))
        if value is not None:
            selected[str(selector)] = value
    return selected if selected else payload


def _jsonl_contract_digest(
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    union_keys = sorted({str(key) for record in records for key in record.keys()})
    primary_id_key = next(
        (
            candidate
            for candidate in ("sample_id", "snapshot_id", "episode_id")
            if any(candidate in record for record in records)
        ),
        None,
    )
    primary_ids = [
        str(record.get(primary_id_key, ""))
        for record in records
        if primary_id_key is not None and primary_id_key in record
    ]
    training_view_counts = Counter(
        str(record.get("training_view"))
        for record in records
        if record.get("training_view") is not None
    )
    contract_payload = {
        "line_count": int(len(records)),
        "union_keys": union_keys,
        "primary_id_key": primary_id_key,
        "primary_id_sequence_sha256": None
        if primary_id_key is None
        else _sha256_bytes("\n".join(primary_ids).encode("utf-8")),
        "training_view_counts": {
            key: int(value) for key, value in sorted(training_view_counts.items())
        },
    }
    return _sha256_bytes(_canonical_json_bytes(contract_payload)), contract_payload


def _normalize_existing_path(raw_value: object, *, field_name: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    candidate = Path(raw_value.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise ValueError(f"{field_name} does not exist: {candidate}")
    return candidate


def _archive_copy(source_path: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, archive_path)
    return archive_path


def _selected_checkpoint_dir(
    run_metadata: Mapping[str, Any], *, field_name: str
) -> Path:
    raw_value = _deep_get(
        run_metadata, "comparable_run_spec.checkpoint_rule.selected_checkpoint_path"
    )
    checkpoint_path = _normalize_existing_path(raw_value, field_name=field_name)
    if checkpoint_path.is_file():
        checkpoint_path = checkpoint_path.parent
    if not checkpoint_path.is_dir():
        raise ValueError(f"{field_name} must resolve to a checkpoint directory")
    return checkpoint_path


def _json_artifact_entry(
    *,
    artifact_id: str,
    stage: str,
    label: str,
    owner_script: str,
    source_path: Path,
    archive_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(source_path)
    selected_payload = _selected_digest_payload(
        payload,
        JSON_DIGEST_SELECTORS.get(artifact_id, ()),
    )
    archived_path = _archive_copy(source_path, archive_path)
    entry = {
        "artifact_id": artifact_id,
        "stage": stage,
        "label": label,
        "content_type": "json",
        "owner_script": owner_script,
        "path": _repo_relative(source_path),
        "resolved_path": str(source_path),
        "sha256": _sha256_file(source_path),
        "schema_digest": _schema_digest_for_json(payload),
        "config_or_schema_digest": _sha256_bytes(
            _canonical_json_bytes(selected_payload)
        ),
        "digest_basis": {
            "mode": "selected_fields"
            if JSON_DIGEST_SELECTORS.get(artifact_id, ())
            else "whole_json_canonical_payload",
            "selectors": list(JSON_DIGEST_SELECTORS.get(artifact_id, ())),
        },
        "artifact_kind": payload.get("artifact_kind"),
        "schema_version": payload.get("schema_version"),
        "immutable_archive_path": _repo_relative(archived_path),
    }
    return entry, payload


def _jsonl_artifact_entry(
    *,
    artifact_id: str,
    stage: str,
    label: str,
    owner_script: str,
    source_path: Path,
    archive_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _read_jsonl(source_path)
    contract_digest, contract_payload = _jsonl_contract_digest(records)
    archived_path = _archive_copy(source_path, archive_path)
    entry = {
        "artifact_id": artifact_id,
        "stage": stage,
        "label": label,
        "content_type": "jsonl",
        "owner_script": owner_script,
        "path": _repo_relative(source_path),
        "resolved_path": str(source_path),
        "sha256": _sha256_file(source_path),
        "schema_digest": contract_digest,
        "config_or_schema_digest": contract_digest,
        "digest_basis": {
            "mode": "jsonl_contract",
            "contract": contract_payload,
        },
        "line_count": int(len(records)),
        "immutable_archive_path": _repo_relative(archived_path),
    }
    return entry, records


def _collect_entries(
    *,
    harvest_dir: Path,
    training_dir: Path,
    eval_dir: Path,
    archive_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    artifacts_by_stage: dict[str, list[str]] = {
        "T8": [],
        "T9": [],
        "T10": [],
        "T11": [],
        "T12": [],
    }

    snapshot_candidates_path = _validate_existing_file(
        harvest_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME,
        arg_name="T8 snapshot_candidates.jsonl",
    )
    snapshot_feasibility_path = _validate_existing_file(
        harvest_dir / state_conditioned_snapshot_harvest.FEASIBILITY_REPORT_JSON_NAME,
        arg_name="T8 snapshot_feasibility_report.json",
    )
    teacher_gate_path = _validate_existing_file(
        harvest_dir / state_conditioned_snapshot_harvest.TEACHER_GATE_REPORT_JSON_NAME,
        arg_name="T8 teacher_gate_report.json",
    )
    rollout_analysis_path = _validate_existing_file(
        harvest_dir
        / state_conditioned_snapshot_harvest.LOCAL_RECOVERY_ROLLOUT_ANALYSIS_JSON_NAME,
        arg_name="T9 local_recovery_rollout_analysis.json",
    )
    pseudodemo_manifest_path = _validate_existing_file(
        harvest_dir
        / state_conditioned_snapshot_harvest.LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME,
        arg_name="T9 local_recovery_pseudodemo_manifest.json",
    )

    sft_labels_path = _validate_existing_file(
        training_dir
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_LABELS_JSONL_NAME,
        arg_name="T10 state_conditioned_sft_labels.jsonl",
    )
    sft_stats_path = _validate_existing_file(
        training_dir
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_STATS_JSON_NAME,
        arg_name="T10 state_conditioned_sft_stats.json",
    )
    fairness_audit_path = _validate_existing_file(
        training_dir
        / state_conditioned_build_training_set.EQUAL_DATA_FAIRNESS_AUDIT_JSON_NAME,
        arg_name="T10 equal_data_fairness_audit.json",
    )
    conditioning_liveness_path = _validate_existing_file(
        training_dir
        / state_conditioned_build_training_set.CONDITIONING_CHANNEL_LIVENESS_JSON_NAME,
        arg_name="T10 conditioning_channel_liveness.json",
    )
    promotion_gate_path = _validate_existing_file(
        training_dir
        / state_conditioned_build_training_set.DEV_ONLY_PROMOTION_GATE_JSON_NAME,
        arg_name="T10 dev_only_promotion_gate.json",
    )
    sft_stats = _read_json(sft_stats_path)
    lerobot_meta_info_path = _validate_existing_file(
        _normalize_existing_path(
            sft_stats.get("lerobot_dataset_path"),
            field_name="T10 state_conditioned_sft_stats.json.lerobot_dataset_path",
        )
        / "meta"
        / "info.json",
        arg_name="T10 lerobot_training_dataset/meta/info.json",
    )

    run_metadata_c0_path = _validate_existing_file(
        training_dir / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT["c0"],
        arg_name="T11 run_metadata_C0_equal_data_control.json",
    )
    run_metadata_c1_path = _validate_existing_file(
        training_dir / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT["c1"],
        arg_name="T11 run_metadata_C1_phase_mode.json",
    )
    delegate_summary_c0_path = _validate_existing_file(
        training_dir
        / state_conditioned_train.DELEGATE_SUMMARY_BASENAME_BY_VARIANT["c0"],
        arg_name="T11 delegate_summary_C0_equal_data_control.json",
    )
    delegate_summary_c1_path = _validate_existing_file(
        training_dir
        / state_conditioned_train.DELEGATE_SUMMARY_BASENAME_BY_VARIANT["c1"],
        arg_name="T11 delegate_summary_C1_phase_mode.json",
    )
    diff_whitelist_path = _validate_existing_file(
        training_dir / state_conditioned_train.DIFF_WHITELIST_JSON_NAME,
        arg_name="T11 state_conditioned_training_fairness_diff_whitelist.json",
    )
    run_metadata_c0 = _read_json(run_metadata_c0_path)
    run_metadata_c1 = _read_json(run_metadata_c1_path)
    checkpoint_c0_dir = _selected_checkpoint_dir(
        run_metadata_c0,
        field_name=(
            "T11 run_metadata_C0_equal_data_control.json.comparable_run_spec."
            "checkpoint_rule.selected_checkpoint_path"
        ),
    )
    checkpoint_c1_dir = _selected_checkpoint_dir(
        run_metadata_c1,
        field_name=(
            "T11 run_metadata_C1_phase_mode.json.comparable_run_spec."
            "checkpoint_rule.selected_checkpoint_path"
        ),
    )
    checkpoint_c0_config_path = _validate_existing_file(
        checkpoint_c0_dir / "config.json",
        arg_name="T11 selected C0 checkpoint config.json",
    )
    checkpoint_c1_config_path = _validate_existing_file(
        checkpoint_c1_dir / "config.json",
        arg_name="T11 selected C1 checkpoint config.json",
    )
    checkpoint_c0_asset_path = state_conditioned_train._selected_checkpoint_asset(
        checkpoint_c0_dir
    )
    checkpoint_c1_asset_path = state_conditioned_train._selected_checkpoint_asset(
        checkpoint_c1_dir
    )
    if checkpoint_c0_asset_path is None:
        raise ValueError(
            f"missing required T11 selected C0 checkpoint asset: {checkpoint_c0_dir}"
        )
    if checkpoint_c1_asset_path is None:
        raise ValueError(
            f"missing required T11 selected C1 checkpoint asset: {checkpoint_c1_dir}"
        )
    checkpoint_c0_asset_path = _validate_existing_file(
        checkpoint_c0_asset_path,
        arg_name="T11 selected C0 checkpoint asset",
    )
    checkpoint_c1_asset_path = _validate_existing_file(
        checkpoint_c1_asset_path,
        arg_name="T11 selected C1 checkpoint asset",
    )

    scorecard_path = _validate_existing_file(
        eval_dir
        / state_conditioned_oracle_eval.ORACLE_CONDITIONED_DEV_SCORECARD_JSON_NAME,
        arg_name="T12 oracle_conditioned_dev_scorecard.json",
    )
    oracle_gate_path = _validate_existing_file(
        eval_dir / state_conditioned_oracle_eval.ORACLE_GATE_DECISION_JSON_NAME,
        arg_name="T12 oracle_gate_decision.json",
    )
    recovery_summary_path = _validate_existing_file(
        eval_dir / state_conditioned_oracle_eval.RECOVERY_BENCHMARK_SUMMARY_JSON_NAME,
        arg_name="T12 recovery_benchmark_summary.json",
    )
    result_split_path = _validate_existing_file(
        eval_dir / state_conditioned_oracle_eval.RESULT_SPLIT_DECISION_JSON_NAME,
        arg_name="T12 result_split_decision.json",
    )

    specs: list[dict[str, Any]] = [
        {
            "artifact_id": "T8.snapshot_candidates_jsonl",
            "stage": "T8",
            "label": "snapshot_candidates",
            "content_type": "jsonl",
            "owner_script": OWNER_CODE_MAP["T8"],
            "source_path": snapshot_candidates_path,
            "archive_path": archive_root / "t8" / snapshot_candidates_path.name,
        },
        {
            "artifact_id": "T8.snapshot_feasibility_report_json",
            "stage": "T8",
            "label": "snapshot_feasibility_report",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T8"],
            "source_path": snapshot_feasibility_path,
            "archive_path": archive_root / "t8" / snapshot_feasibility_path.name,
        },
        {
            "artifact_id": "T8.teacher_gate_report_json",
            "stage": "T8",
            "label": "teacher_gate_report",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T8"],
            "source_path": teacher_gate_path,
            "archive_path": archive_root / "t8" / teacher_gate_path.name,
        },
        {
            "artifact_id": "T9.local_recovery_rollout_analysis_json",
            "stage": "T9",
            "label": "local_recovery_rollout_analysis",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T9"],
            "source_path": rollout_analysis_path,
            "archive_path": archive_root / "t9" / rollout_analysis_path.name,
        },
        {
            "artifact_id": "T9.local_recovery_pseudodemo_manifest_json",
            "stage": "T9",
            "label": "local_recovery_pseudodemo_manifest",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T9"],
            "source_path": pseudodemo_manifest_path,
            "archive_path": archive_root / "t9" / pseudodemo_manifest_path.name,
        },
        {
            "artifact_id": "T10.state_conditioned_sft_labels_jsonl",
            "stage": "T10",
            "label": "state_conditioned_sft_labels",
            "content_type": "jsonl",
            "owner_script": OWNER_CODE_MAP["T10"],
            "source_path": sft_labels_path,
            "archive_path": archive_root / "t10" / sft_labels_path.name,
        },
        {
            "artifact_id": "T10.state_conditioned_sft_stats_json",
            "stage": "T10",
            "label": "state_conditioned_sft_stats",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T10"],
            "source_path": sft_stats_path,
            "archive_path": archive_root / "t10" / sft_stats_path.name,
        },
        {
            "artifact_id": "T10.equal_data_fairness_audit_json",
            "stage": "T10",
            "label": "equal_data_fairness_audit",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T10"],
            "source_path": fairness_audit_path,
            "archive_path": archive_root / "t10" / fairness_audit_path.name,
        },
        {
            "artifact_id": "T10.conditioning_channel_liveness_json",
            "stage": "T10",
            "label": "conditioning_channel_liveness",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T10"],
            "source_path": conditioning_liveness_path,
            "archive_path": archive_root / "t10" / conditioning_liveness_path.name,
        },
        {
            "artifact_id": "T10.dev_only_promotion_gate_json",
            "stage": "T10",
            "label": "dev_only_promotion_gate",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T10"],
            "source_path": promotion_gate_path,
            "archive_path": archive_root / "t10" / promotion_gate_path.name,
        },
        {
            "artifact_id": "T10.lerobot_dataset_meta_info_json",
            "stage": "T10",
            "label": "lerobot_dataset_meta_info",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T10"],
            "source_path": lerobot_meta_info_path,
            "archive_path": archive_root / "t10" / "lerobot_meta_info.json",
        },
        {
            "artifact_id": "T11.run_metadata_c0_json",
            "stage": "T11",
            "label": "run_metadata_c0",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": run_metadata_c0_path,
            "archive_path": archive_root / "t11" / run_metadata_c0_path.name,
        },
        {
            "artifact_id": "T11.run_metadata_c1_json",
            "stage": "T11",
            "label": "run_metadata_c1",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": run_metadata_c1_path,
            "archive_path": archive_root / "t11" / run_metadata_c1_path.name,
        },
        {
            "artifact_id": "T11.delegate_summary_c0_json",
            "stage": "T11",
            "label": "delegate_summary_c0",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": delegate_summary_c0_path,
            "archive_path": archive_root / "t11" / delegate_summary_c0_path.name,
        },
        {
            "artifact_id": "T11.delegate_summary_c1_json",
            "stage": "T11",
            "label": "delegate_summary_c1",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": delegate_summary_c1_path,
            "archive_path": archive_root / "t11" / delegate_summary_c1_path.name,
        },
        {
            "artifact_id": "T11.state_conditioned_training_fairness_diff_whitelist_json",
            "stage": "T11",
            "label": "state_conditioned_training_fairness_diff_whitelist",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": diff_whitelist_path,
            "archive_path": archive_root / "t11" / diff_whitelist_path.name,
        },
        {
            "artifact_id": "T11.selected_checkpoint_c0_config_json",
            "stage": "T11",
            "label": "selected_checkpoint_c0_config",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": checkpoint_c0_config_path,
            "archive_path": archive_root / "t11" / "selected_checkpoint_c0_config.json",
        },
        {
            "artifact_id": "T11.selected_checkpoint_c0_model_index_json",
            "stage": "T11",
            "label": "selected_checkpoint_c0_model_index",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": checkpoint_c0_asset_path,
            "archive_path": archive_root
            / "t11"
            / "selected_checkpoint_c0_model_index.json",
        },
        {
            "artifact_id": "T11.selected_checkpoint_c1_config_json",
            "stage": "T11",
            "label": "selected_checkpoint_c1_config",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": checkpoint_c1_config_path,
            "archive_path": archive_root / "t11" / "selected_checkpoint_c1_config.json",
        },
        {
            "artifact_id": "T11.selected_checkpoint_c1_model_index_json",
            "stage": "T11",
            "label": "selected_checkpoint_c1_model_index",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T11"],
            "source_path": checkpoint_c1_asset_path,
            "archive_path": archive_root
            / "t11"
            / "selected_checkpoint_c1_model_index.json",
        },
        {
            "artifact_id": "T12.oracle_conditioned_dev_scorecard_json",
            "stage": "T12",
            "label": "oracle_conditioned_dev_scorecard",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T12"],
            "source_path": scorecard_path,
            "archive_path": archive_root / "t12" / scorecard_path.name,
        },
        {
            "artifact_id": "T12.oracle_gate_decision_json",
            "stage": "T12",
            "label": "oracle_gate_decision",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T12"],
            "source_path": oracle_gate_path,
            "archive_path": archive_root / "t12" / oracle_gate_path.name,
        },
        {
            "artifact_id": "T12.recovery_benchmark_summary_json",
            "stage": "T12",
            "label": "recovery_benchmark_summary",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T12"],
            "source_path": recovery_summary_path,
            "archive_path": archive_root / "t12" / recovery_summary_path.name,
        },
        {
            "artifact_id": "T12.result_split_decision_json",
            "stage": "T12",
            "label": "result_split_decision",
            "content_type": "json",
            "owner_script": OWNER_CODE_MAP["T12"],
            "source_path": result_split_path,
            "archive_path": archive_root / "t12" / result_split_path.name,
        },
    ]

    captured_payloads: dict[str, Any] = {}
    for spec in specs:
        if spec["content_type"] == "json":
            entry, payload = _json_artifact_entry(
                artifact_id=str(spec["artifact_id"]),
                stage=str(spec["stage"]),
                label=str(spec["label"]),
                owner_script=str(spec["owner_script"]),
                source_path=Path(spec["source_path"]),
                archive_path=Path(spec["archive_path"]),
            )
        else:
            entry, payload = _jsonl_artifact_entry(
                artifact_id=str(spec["artifact_id"]),
                stage=str(spec["stage"]),
                label=str(spec["label"]),
                owner_script=str(spec["owner_script"]),
                source_path=Path(spec["source_path"]),
                archive_path=Path(spec["archive_path"]),
            )
        artifacts.append(entry)
        artifacts_by_stage[str(spec["stage"])].append(str(spec["artifact_id"]))
        captured_payloads[str(spec["artifact_id"])] = payload

    return artifacts, artifacts_by_stage, captured_payloads


def materialize_state_conditioned_wave_freeze_manifest(
    *,
    harvest_dir: Path,
    training_dir: Path,
    eval_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    resolved_harvest_dir = _validate_existing_dir(harvest_dir, arg_name="harvest-dir")
    resolved_training_dir = _validate_existing_dir(
        training_dir, arg_name="training-dir"
    )
    resolved_eval_dir = _validate_existing_dir(eval_dir, arg_name="eval-dir")
    resolved_output_path = _validate_output_path(output_path)

    archive_root = (
        resolved_output_path.parent
        / IMMUTABLE_ARCHIVE_DIRNAME
        / IMMUTABLE_ARCHIVE_SUBDIR
    )
    if archive_root.exists():
        shutil.rmtree(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)

    artifacts, artifacts_by_stage, captured_payloads = _collect_entries(
        harvest_dir=resolved_harvest_dir,
        training_dir=resolved_training_dir,
        eval_dir=resolved_eval_dir,
        archive_root=archive_root,
    )

    artifact_ids = {entry["artifact_id"] for entry in artifacts}
    fairness_json_ids = [
        "T10.equal_data_fairness_audit_json",
        "T10.conditioning_channel_liveness_json",
        "T10.dev_only_promotion_gate_json",
        "T11.state_conditioned_training_fairness_diff_whitelist_json",
        "T12.oracle_conditioned_dev_scorecard_json",
    ]
    t12_decision_json_ids = [
        "T12.oracle_conditioned_dev_scorecard_json",
        "T12.oracle_gate_decision_json",
        "T12.recovery_benchmark_summary_json",
        "T12.result_split_decision_json",
    ]
    if not set(fairness_json_ids).issubset(artifact_ids):
        raise RuntimeError("baseline/C0/C1 fairness artifact list is incomplete")
    if not set(t12_decision_json_ids).issubset(artifact_ids):
        raise RuntimeError("T12 decision artifact list is incomplete")

    teacher_gate_report = dict(captured_payloads["T8.teacher_gate_report_json"])
    oracle_gate_decision = dict(captured_payloads["T12.oracle_gate_decision_json"])
    result_split_decision = dict(captured_payloads["T12.result_split_decision_json"])
    oracle_scorecard = dict(
        captured_payloads["T12.oracle_conditioned_dev_scorecard_json"]
    )
    c0_run_metadata = dict(captured_payloads["T11.run_metadata_c0_json"])
    c1_run_metadata = dict(captured_payloads["T11.run_metadata_c1_json"])
    labels_entry = next(
        entry
        for entry in artifacts
        if entry["artifact_id"] == "T10.state_conditioned_sft_labels_jsonl"
    )
    lerobot_info_entry = next(
        entry
        for entry in artifacts
        if entry["artifact_id"] == "T10.lerobot_dataset_meta_info_json"
    )

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_wave_freeze_manifest",
        "wave_label": "freeze_current_t8_t12_wave",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "owner_code_map": dict(OWNER_CODE_MAP),
        "source_roots": {
            "harvest_dir": _repo_relative(resolved_harvest_dir),
            "training_dir": _repo_relative(resolved_training_dir),
            "eval_dir": _repo_relative(resolved_eval_dir),
        },
        "allowed_change_surface": {
            "policy": ALLOWED_CHANGE_POLICY,
            "human_readable_rule": (
                "Freeze result forbids changes to T8/T9/T10/T11/T12 decisions, fairness "
                "artifacts, owner scripts, checkpoint configs, and conditioning contracts; "
                "the only downstream variation surface is a new labels/data version."
            ),
            "allowed_artifact_ids": [
                "T10.state_conditioned_sft_labels_jsonl",
                "T10.lerobot_dataset_meta_info_json",
            ],
            "allowed_path_prefixes": [
                "agent/artifacts/state_conditioned_materialization/training/state_conditioned_sft_labels.jsonl",
                "agent/artifacts/state_conditioned_materialization/training/lerobot_training_dataset/",
            ],
            "current_mutable_surface_snapshot": {
                "state_conditioned_sft_labels_sha256": str(labels_entry["sha256"]),
                "state_conditioned_sft_labels_contract_digest": str(
                    labels_entry["config_or_schema_digest"]
                ),
                "lerobot_dataset_meta_info_sha256": str(lerobot_info_entry["sha256"]),
                "lerobot_dataset_meta_info_contract_digest": str(
                    lerobot_info_entry["config_or_schema_digest"]
                ),
                "dataset_fingerprint_c0": _deep_get(
                    c0_run_metadata,
                    "comparable_run_spec.dataset_fingerprint",
                ),
                "dataset_fingerprint_c1": _deep_get(
                    c1_run_metadata,
                    "comparable_run_spec.dataset_fingerprint",
                ),
            },
            "forbidden_change_examples": [
                "teacher gate thresholds or selected snapshot families",
                "formal pseudodemo teacher provenance or floor counts",
                "equal_data_fairness_audit / conditioning_channel_liveness / diff whitelist contents",
                "T12 oracle gate / result split decisions",
                "selected checkpoint config.json or retained model index JSON",
            ],
        },
        "artifacts": artifacts,
        "artifacts_by_stage": {
            stage: list(ids) for stage, ids in sorted(artifacts_by_stage.items())
        },
        "baseline_c0_c1_fairness_jsons": fairness_json_ids,
        "t12_decision_jsons": t12_decision_json_ids,
        "current_wave_snapshot": {
            "teacher_gate_all_family_success_rates_zero": all(
                float(family.get("success_rate", -1.0)) == 0.0
                for family in list(teacher_gate_report.get("families", []))
            ),
            "teacher_gate_fallback_enabled_count": int(
                teacher_gate_report.get("counts", {}).get(
                    "teacher_fallback_enabled_count", 0
                )
            ),
            "oracle_gate_status": oracle_gate_decision.get("gate_status"),
            "oracle_gate_passed": bool(oracle_gate_decision.get("gate_passed", False)),
            "oracle_gate_next_step_if_blocked": oracle_gate_decision.get(
                "next_step_if_blocked"
            ),
            "result_split_next_step": result_split_decision.get("next_step"),
            "result_split_branch_reason": result_split_decision.get("branch_reason"),
            "result_split_oracle_uplift_clearly_established": bool(
                result_split_decision.get("oracle_uplift_clearly_established", False)
            ),
            "result_split_metric_snapshot": dict(
                result_split_decision.get("metric_snapshot", {})
            ),
            "oracle_scorecard_line_order": list(oracle_scorecard.get("line_order", [])),
        },
        "archive": {
            "immutable_archive_root": _repo_relative(archive_root),
            "entry_count": int(len(artifacts)),
        },
        "counts": {
            "artifact_count": int(len(artifacts)),
            "stage_count": 5,
            "fairness_json_count": int(len(fairness_json_ids)),
            "t12_decision_json_count": int(len(t12_decision_json_ids)),
        },
    }
    _write_json(resolved_output_path, manifest_payload)
    return {
        "output_path": str(resolved_output_path),
        "immutable_archive_root": str(archive_root),
        "artifact_count": int(len(artifacts)),
        "allowed_change_policy": ALLOWED_CHANGE_POLICY,
        "result_split_next_step": result_split_decision.get("next_step"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_state_conditioned_wave_freeze_manifest(
            harvest_dir=args.harvest_dir,
            training_dir=args.training_dir,
            eval_dir=args.eval_dir,
            output_path=args.output,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ALLOWED_CHANGE_POLICY",
    "DEFAULT_EVAL_DIR",
    "DEFAULT_HARVEST_DIR",
    "DEFAULT_OUTPUT",
    "DEFAULT_TRAINING_DIR",
    "IMMUTABLE_ARCHIVE_DIRNAME",
    "IMMUTABLE_ARCHIVE_SUBDIR",
    "SCHEMA_VERSION",
    "build_parser",
    "main",
    "materialize_state_conditioned_wave_freeze_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
