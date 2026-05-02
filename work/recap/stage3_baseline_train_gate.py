from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts.gr00t_checkpoint_provenance_gate import (
    CHECKPOINT_PROVENANCE_REPORT_JSON_NAME,
)
from work.recap.scripts.gr00t_run_manifest_gate import RUN_MANIFEST_JSON_NAME
from work.recap.scripts.gr00t_run_manifest_gate import RUN_MANIFEST_REPORT_JSON_NAME
from work.recap.scripts.state_conditioned_common import read_json
from work.recap.scripts.state_conditioned_common import write_json
from work.recap.stage3_collect_checkpoint_binding import (
    CHECKPOINT_PROVENANCE_GATE_JSON_NAME,
)
from work.recap.stage3_collect_checkpoint_binding import (
    DEFAULT_STAGE3_ITERATION_MANIFEST_REL,
)
from work.recap.stage3_collect_checkpoint_binding import (
    HISTORICAL_SUCCESS_RATE_THRESHOLD,
)
from work.recap.stage3_collect_checkpoint_binding import OFFICIAL_TASK_ANCHOR_MODEL
from work.recap.stage3_collect_checkpoint_binding import RUN_MANIFEST_GATE_JSON_NAME
from work.recap.stage3_collect_checkpoint_binding import (
    STAGE3_ITERATION_MANIFEST_V3,
)
from work.recap.stage3_collect_checkpoint_binding import (
    SUCCESS_GATE_THRESHOLD_COUNT,
)
from work.recap.stage3_collect_checkpoint_binding import (
    _load_training_python_contract_from_manifest,
)
from work.recap.finetune_full import resolve_full_update_authority_output_dir
from work.recap.stage3_collect_checkpoint_binding import _manifest_env_name
from work.recap.stage3_collect_checkpoint_binding import (
    _build_candidate_provenance_metadata,
)
from work.recap.stage3_collect_checkpoint_binding import (
    _build_checkpoint_provenance_gate_payload,
)
from work.recap.stage3_collect_checkpoint_binding import (
    _build_preliminary_run_manifest,
)
from work.recap.stage3_collect_checkpoint_binding import (
    _build_run_manifest_gate_payload,
)
from work.recap.stage3_contract_precondition_gate import (
    _inspect_checkpoint_weight_map_features,
)
from work.demo_utils import paths as demo_paths


DEFAULT_SINGLE_GPU_SMOKE_VERDICT_REL = Path(
    "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/green_smoke_single_gpu_verdict.json"
)
DEFAULT_FORMAL_OUTPUT_DIR_REL = Path(
    "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run"
)
DEFAULT_RUNTIME_LOG_DIR_REL = Path("agent/runtime_logs/stage3_baseline_train_gate")
DEFAULT_ATTEMPT_DIRNAME = "baseline_train_attempt_001"
DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_PORT = 49647
DEFAULT_TRAIN_LOG_NAME = "baseline_train_attempt.log"
DEFAULT_EVAL_LOG_NAME = "baseline_prelim_eval.log"
DEFAULT_CKPT_GATE_LOG_NAME = "checkpoint_provenance_gate.log"
DEFAULT_RUN_MANIFEST_GATE_LOG_NAME = "run_manifest_gate.log"
DEFAULT_FINETUNE_SUMMARY_JSON_NAME = "baseline_train_finetune_summary.json"
DEFAULT_PRELIM_EVAL_SUMMARY_JSON_NAME = "baseline_train_prelim_eval_summary.json"
DEFAULT_PROVENANCE_INPUT_JSON_NAME = "baseline_train_checkpoint_provenance_input.json"
DEFAULT_PRELIM_RUN_MANIFEST_INPUT_JSON_NAME = (
    "baseline_train_prelim_run_manifest_input.json"
)
DEFAULT_CKPT_GATE_RAW_DIRNAME = "checkpoint_provenance_gate_raw"
DEFAULT_RUN_MANIFEST_GATE_RAW_DIRNAME = "run_manifest_gate_raw"
DEFAULT_PREFLIGHT_JSON_NAME = "rtx_pro_6000_blackwell_max_q_preflight.json"
LEGACY_PREFLIGHT_JSON_NAMES = ("a6000_preflight.json",)
DEFAULT_SUPERSEDED_OUTPUTS_JSON_NAME = "superseded_outputs.json"
DEFAULT_DELEGATE_RUNTIME_REPAIR_LOG_DIR_REL = Path(
    "agent/runtime_logs/stage3_delegate_runtime_repair"
)

LIVE_LAUNCH_FAMILY = "single_gpu_v1"
SINGLE_GPU_SMOKE_VERDICT_SCHEMA_VERSION = "stage3_single_gpu_smoke_verdict_v1"
SINGLE_GPU_SMOKE_VERDICT_UNLOCK_SIGNAL = "single_gpu_formal_baseline_allowed"
HISTORICAL_DDP_SMOKE_VERDICT_SCHEMA_VERSION = "task10_green_smoke_verdict_v1"
MEANINGFUL_NUM_GPUS = 1
MEANINGFUL_GLOBAL_BATCH_SIZE = 4
MEANINGFUL_GRADIENT_ACCUMULATION_STEPS = 4
MEANINGFUL_PER_DEVICE_BATCH_SIZE = 1
MEANINGFUL_EFFECTIVE_UPDATE_BATCH = 4
MEANINGFUL_DATALOADER_NUM_WORKERS = 0
MEANINGFUL_MAX_STEPS = 200
MEANINGFUL_SAVE_STEPS = 50
MEANINGFUL_SAVE_TOTAL_LIMIT = 1
MEANINGFUL_LEARNING_RATE = 1e-5
MEANINGFUL_TUNE_PROJECTOR = True
MEANINGFUL_TUNE_DIFFUSION_MODEL = True
MIN_DISK_FREE_GIB = 50.0
MIN_GPU_FREE_MEMORY_MIB = 8192
EXPECTED_CUDA_ARCH = "sm_120"
EXPECTED_GPU_NAME_SUBSTRING = "rtx pro 6000 blackwell max-q"
MIN_EXPECTED_GPU_TOTAL_MEMORY_MIB = 90000
EXPECTED_HARDWARE_PROFILE = "rtx_pro_6000_blackwell_max_q_96g_x2_subset"
LEGACY_HARDWARE_PROFILES = (
    "a6000_96g_x4",
    "rtx_pro_6000_blackwell_max_q_96g_x4",
)
PREFLIGHT_V1_SCHEMA_VERSION = "stage3_rtx_pro_6000_blackwell_max_q_preflight_v1"
LEGACY_PREFLIGHT_V1_SCHEMA_VERSION = "stage3_a6000_preflight_v1"
PREFLIGHT_SCHEMA_VERSION = "stage3_rtx_pro_6000_blackwell_max_q_preflight_v2"
LEGACY_PREFLIGHT_SCHEMA_VERSION = "stage3_a6000_preflight_v2"
PREFLIGHT_ARTIFACT_KIND = "stage3_rtx_pro_6000_blackwell_max_q_preflight"
LEGACY_PREFLIGHT_ARTIFACT_KIND = "stage3_a6000_preflight"
SUPERSEDED_OUTPUTS_SCHEMA_VERSION = "stage3_superseded_outputs_v1"
SUPERSEDED_OUTPUTS_ARTIFACT_KIND = "stage3_superseded_outputs"

ATTEMPT_SCHEMA_VERSION = "stage3_collect_policy_baseline_train_attempt_v1"
HARD_BLOCK_SCHEMA_VERSION = "stage3_collect_policy_hard_block_v1"
PROVENANCE_SCHEMA_VERSION = "stage3_collect_policy_ckpt_provenance_v1"
HARD_BLOCKER_BASELINE_TRAINED_PRELIM_NOT_VIABLE = "baseline_trained_prelim_not_viable"
HARD_BLOCKER_RTX_PRO_6000_BLACKWELL_MAX_Q_PREFLIGHT_FAILED = (
    "rtx_pro_6000_blackwell_max_q_preflight_failed"
)
LEGACY_HARD_BLOCKER_A6000_PREFLIGHT_FAILED = "a6000_preflight_failed"
HARD_BLOCKER_BASELINE_TRAIN_EXECUTION_BLOCKED = "baseline_train_execution_blocked"
HARD_BLOCKER_BASELINE_TRAIN_CONTRACT_MISMATCH = "baseline_train_contract_mismatch"
NEXT_ACTION_USER_ESCALATION_REQUIRED = "user_escalation_required"
NEXT_ACTION_WAIT_FOR_RTX_PRO_6000_BLACKWELL_MAX_Q_X2_SUBSET = (
    "wait_for_rtx_pro_6000_blackwell_max_q_x2_subset"
)
LEGACY_NEXT_ACTION_WAIT_FOR_A6000_X4 = "wait_for_a6000_x4"
HARD_BLOCK_SUBFAMILY_HARDWARE_PROFILE_MISMATCH = "hardware_profile_mismatch"
HARD_BLOCK_SUBFAMILY_DELEGATE_RUNTIME_UNHEALTHY = "delegate_runtime_unhealthy"
PREFLIGHT_MANIFEST_KEY = "rtx_pro_6000_blackwell_max_q_preflight"
LEGACY_PREFLIGHT_MANIFEST_KEY = "a6000_preflight"
PREFLIGHT_PATH_FIELD = "rtx_pro_6000_blackwell_max_q_preflight_path"
LEGACY_PREFLIGHT_PATH_FIELD = "a6000_preflight_path"
REASON_GPU_NAME_NOT_EXPECTED = "gpu_name_not_rtx_pro_6000_blackwell_max_q"
LEGACY_REASON_GPU_NAME_NOT_A6000 = "gpu_name_not_a6000"
REASON_GPU_TOTAL_MEMORY_BELOW_EXPECTED_CLASS = (
    "gpu_total_memory_below_rtx_pro_6000_blackwell_max_q_class"
)
LEGACY_REASON_GPU_TOTAL_MEMORY_BELOW_A6000_CLASS = "gpu_total_memory_below_a6000_class"

DECISION_BASELINE_TRAIN_REQUIRED = "baseline_train_required"
DECISION_BASELINE_TRAINED = "baseline_trained"
DECISION_ITERATION_HARD_BLOCK = "iteration_hard_block"

ATTEMPT_STATE_STARTED = "started"
ATTEMPT_STATE_FINISHED = "finished"
STATUS_CONTINUE = "continue"
STATUS_EXECUTION_HARD_BLOCK = "execution_hard_block"
STATUS_INCONCLUSIVE_CONTRACT_MISMATCH = "inconclusive_contract_mismatch"


@dataclass(frozen=True)
class BaselineTrainPrereq:
    smoke_summary_path: Path
    trainability_gate_path: Path
    dataset_path: Path
    max_steps: int
    save_steps: int
    save_total_limit: int
    global_batch_size: int
    gradient_accumulation_steps: int
    dataloader_num_workers: int
    learning_rate: float | None
    num_gpus: int | None


@dataclass(frozen=True)
class AttemptPaths:
    attempt_dir: Path
    finetune_output_dir: Path
    finetune_summary_path: Path
    prelim_eval_summary_path: Path
    provenance_input_path: Path
    prelim_run_manifest_input_path: Path
    checkpoint_gate_raw_dir: Path
    checkpoint_gate_raw_report_path: Path
    run_manifest_gate_raw_dir: Path
    run_manifest_gate_raw_report_path: Path
    run_manifest_gate_raw_manifest_path: Path
    train_log_path: Path
    eval_log_path: Path
    checkpoint_gate_log_path: Path
    run_manifest_gate_log_path: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = demo_paths.remap_legacy_project_root(path)
    return path.resolve()


def _preflight_schema_versions() -> set[str]:
    return {
        PREFLIGHT_V1_SCHEMA_VERSION,
        LEGACY_PREFLIGHT_V1_SCHEMA_VERSION,
        PREFLIGHT_SCHEMA_VERSION,
        LEGACY_PREFLIGHT_SCHEMA_VERSION,
    }


def _preflight_artifact_kinds() -> set[str]:
    return {PREFLIGHT_ARTIFACT_KIND, LEGACY_PREFLIGHT_ARTIFACT_KIND}


def _legacy_preflight_paths(iteration_artifact_root: Path) -> list[Path]:
    return [iteration_artifact_root / name for name in LEGACY_PREFLIGHT_JSON_NAMES]


def _existing_preflight_path(preflight_path: Path) -> Path | None:
    if preflight_path.is_file():
        return preflight_path
    for candidate in _legacy_preflight_paths(preflight_path.parent):
        if candidate.is_file():
            return candidate
    return None


def _repo_relative_path(repo_root: Path, path: Path | str) -> str:
    resolved = _resolve_path(repo_root, path)
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_superseded_outputs_path(iteration_artifact_root: Path) -> Path:
    return iteration_artifact_root / DEFAULT_SUPERSEDED_OUTPUTS_JSON_NAME


def _repair_summary_candidates(runtime_log_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in runtime_log_dir.glob("stage3_delegate_runtime_repair_*.json")
        ],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def _load_delegate_runtime_repair_evidence(
    *,
    repo_root: Path,
    training_python_contract: Mapping[str, str],
) -> dict[str, Any]:
    runtime_log_dir = _resolve_path(
        repo_root, DEFAULT_DELEGATE_RUNTIME_REPAIR_LOG_DIR_REL
    )
    candidates = _repair_summary_candidates(runtime_log_dir)
    if not candidates:
        raise FileNotFoundError(
            f"delegate runtime repair summary not found under {runtime_log_dir}"
        )
    summary_path = candidates[0]
    summary_payload = _read_json_object(summary_path)
    artifacts = _require_mapping(summary_payload, field_name="artifacts")
    declared_summary_path = _resolve_path(
        repo_root,
        _require_string(artifacts, field_name="summary_json"),
    )
    if declared_summary_path != summary_path.resolve():
        raise ValueError(
            "delegate runtime repair summary artifact binding mismatch: "
            f"declared {declared_summary_path}, actual {summary_path.resolve()}"
        )
    log_path = _resolve_path(
        repo_root,
        _require_string(artifacts, field_name="session_log"),
    )
    if not log_path.is_file():
        raise FileNotFoundError(f"delegate runtime repair log missing: {log_path}")

    manifest_delegate = str(training_python_contract["delegate_runtime_python"])
    summary_delegate = _require_string(
        summary_payload,
        field_name="delegate_runtime_python",
    )
    if summary_delegate != manifest_delegate:
        raise ValueError(
            "delegate runtime repair summary drifted from manifest delegate_runtime_python: "
            f"expected {manifest_delegate}, got {summary_delegate}"
        )

    health = _require_mapping(summary_payload, field_name="final_health")
    final_probe = _require_mapping(summary_payload, field_name="final_probe")
    probe_command = final_probe.get("command")
    if not isinstance(probe_command, list) or not probe_command:
        raise ValueError("delegate runtime repair summary missing final_probe.command")
    if str(probe_command[0]) != manifest_delegate:
        raise ValueError(
            "delegate runtime repair summary probe argv[0] drifted from manifest "
            f"delegate_runtime_python: expected {manifest_delegate}, got {probe_command[0]}"
        )
    probe_payload = _require_mapping(final_probe, field_name="payload")
    if str(probe_payload.get("python_executable") or "").strip() != manifest_delegate:
        raise ValueError(
            "delegate runtime repair summary payload.python_executable drifted from manifest "
            f"delegate_runtime_python: expected {manifest_delegate}, got {probe_payload.get('python_executable')}"
        )

    return {
        "summary_path": summary_path.resolve(),
        "summary_rel": _repo_relative_path(repo_root, summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "log_path": log_path.resolve(),
        "log_rel": _repo_relative_path(repo_root, log_path),
        "log_sha256": _sha256_file(log_path),
        "summary": summary_payload,
        "health": health,
        "final_probe": final_probe,
        "status": str(summary_payload.get("status") or "").strip(),
        "pass": bool(
            summary_payload.get("exit_code") == 0
            and health.get("healthy") is True
            and summary_delegate == manifest_delegate
            and str(probe_command[0]) == manifest_delegate
            and str(probe_payload.get("python_executable") or "").strip()
            == manifest_delegate
        ),
        "reason_codes": [
            reason
            for reason in [
                None
                if summary_payload.get("exit_code") == 0
                else "repair_summary_exit_code_nonzero",
                None
                if health.get("healthy") is True
                else "repair_summary_final_health_unhealthy",
            ]
            if reason is not None
        ],
    }


def _build_superseded_outputs_payload(
    *,
    repo_root: Path,
    superseded_outputs_path: Path,
    preflight_path: Path,
    existing_preflight_path: Path | None,
    existing_preflight_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if (
        existing_preflight_path is not None
        and existing_preflight_payload is not None
        and (
            existing_preflight_path.resolve() != preflight_path.resolve()
            or str(existing_preflight_payload.get("schema_version") or "").strip()
            in _preflight_schema_versions()
            or str(existing_preflight_payload.get("artifact_kind") or "").strip()
            in _preflight_artifact_kinds()
        )
    ):
        existing_schema = str(
            existing_preflight_payload.get("schema_version") or ""
        ).strip()
        entries.append(
            {
                "artifact_kind": PREFLIGHT_ARTIFACT_KIND,
                "superseded_path": _repo_relative_path(
                    repo_root, existing_preflight_path
                ),
                "superseded_schema_version": existing_schema,
                "superseded_sha256": _sha256_file(existing_preflight_path),
                "superseded_reason_code": "hardware_profile_authority_renamed",
                "replacement_path": _repo_relative_path(repo_root, preflight_path),
                "replacement_schema_version": PREFLIGHT_SCHEMA_VERSION,
            }
        )
    return {
        "schema_version": SUPERSEDED_OUTPUTS_SCHEMA_VERSION,
        "artifact_kind": SUPERSEDED_OUTPUTS_ARTIFACT_KIND,
        "generated_at": _now_iso(),
        "artifact_path": _repo_relative_path(repo_root, superseded_outputs_path),
        "superseded_outputs": entries,
    }


def _validate_preflight_authority(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    superseded_outputs_path: Path,
    superseded_outputs_payload: Mapping[str, Any],
    preflight_payload: Mapping[str, Any],
    existing_preflight_path: Path | None,
    existing_preflight_payload: Mapping[str, Any] | None,
    existing_preflight_sha256: str | None = None,
) -> None:
    if (
        str(preflight_payload.get("schema_version") or "").strip()
        != PREFLIGHT_SCHEMA_VERSION
    ):
        raise ValueError("preflight authority schema_version mismatch")
    if (
        str(preflight_payload.get("artifact_kind") or "").strip()
        != PREFLIGHT_ARTIFACT_KIND
    ):
        raise ValueError("preflight authority artifact_kind mismatch")
    declared_manifest_sha = str(preflight_payload.get("manifest_sha256") or "").strip()
    actual_manifest_sha = _sha256_file(manifest_path)
    if declared_manifest_sha != actual_manifest_sha:
        raise ValueError(
            "preflight authority manifest sha256 binding mismatch: "
            f"expected {declared_manifest_sha}, actual {actual_manifest_sha}"
        )

    evidence = _require_mapping(preflight_payload, field_name="evidence")
    summary_path = _resolve_path(
        repo_root,
        _require_string(evidence, field_name="delegate_runtime_repair_summary_path"),
    )
    summary_sha = _require_string(
        evidence,
        field_name="delegate_runtime_repair_summary_sha256",
    )
    actual_summary_sha = _sha256_file(summary_path)
    if summary_sha != actual_summary_sha:
        raise ValueError(
            "delegate runtime repair summary sha256 binding mismatch: "
            f"expected {summary_sha}, actual {actual_summary_sha}"
        )
    log_path = _resolve_path(
        repo_root,
        _require_string(evidence, field_name="delegate_runtime_repair_log"),
    )
    log_sha = _require_string(
        evidence,
        field_name="delegate_runtime_repair_log_sha256",
    )
    actual_log_sha = _sha256_file(log_path)
    if log_sha != actual_log_sha:
        raise ValueError(
            "delegate runtime repair log sha256 binding mismatch: "
            f"expected {log_sha}, actual {actual_log_sha}"
        )

    existing_schema = (
        None
        if existing_preflight_payload is None
        else str(existing_preflight_payload.get("schema_version") or "").strip()
    )
    if existing_schema in {
        PREFLIGHT_V1_SCHEMA_VERSION,
        LEGACY_PREFLIGHT_V1_SCHEMA_VERSION,
    }:
        persisted_superseded = _read_json_object(superseded_outputs_path)
        recorded_entries = persisted_superseded.get("superseded_outputs")
        if not isinstance(recorded_entries, list):
            raise ValueError("superseded_outputs must be a list")
        expected_source_path = (
            existing_preflight_path
            if existing_preflight_path is not None
            else manifest_path.parent / LEGACY_PREFLIGHT_JSON_NAMES[0]
        )
        expected_path = _repo_relative_path(repo_root, expected_source_path)
        expected_sha = str(existing_preflight_sha256 or "").strip()
        if not expected_sha:
            raise ValueError(
                "missing existing_preflight_sha256 for v1 supersede validation"
            )
        matched = False
        for entry in recorded_entries:
            if not isinstance(entry, Mapping):
                continue
            if (
                str(entry.get("superseded_path") or "").strip() == expected_path
                and str(entry.get("superseded_schema_version") or "").strip()
                == existing_schema
                and str(entry.get("superseded_sha256") or "").strip() == expected_sha
            ):
                matched = True
                break
        if not matched:
            raise ValueError(
                "preflight v1 authority must be superseded before trusting v2"
            )

    delegate_runtime_health = _require_mapping(
        preflight_payload,
        field_name="delegate_runtime_health",
    )
    if bool(delegate_runtime_health.get("argv0_matches_manifest")) is not True:
        raise ValueError(
            "delegate runtime preflight probe argv[0] drifted from manifest"
        )
    preflight_delegate = str(
        demo_paths.abspath_preserve_symlink(
            str(preflight_payload.get("delegate_runtime_python_manifest") or "")
        )
    ).strip()
    manifest_delegate = str(
        demo_paths.abspath_preserve_symlink(
            str(manifest_payload.get("delegate_runtime_python") or "")
        )
    ).strip()
    if preflight_delegate != manifest_delegate:
        raise ValueError(
            "delegate_runtime_python_manifest drifted from iteration manifest"
        )

    _ = superseded_outputs_payload


def _build_observed_hardware_profile(gpus: list[dict[str, Any]]) -> str:
    if not gpus:
        return "no_gpu_detected"
    first = gpus[0]
    name = str(first.get("name") or "unknown_gpu").strip().lower().replace(" ", "_")
    total_mib = int(_coerce_int(first.get("memory_total_mib")) or 0)
    approx_gib = max(int(round(float(total_mib) / 1024.0)), 0)
    return f"{name}_{approx_gib}g_x{len(gpus)}"


def _require_mapping(payload: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return dict(value)


def _require_string(payload: Mapping[str, Any], *, field_name: str) -> str:
    text = str(payload.get(field_name) or "").strip()
    if not text:
        raise ValueError(f"missing non-empty {field_name}")
    return text


def _require_int(payload: Mapping[str, Any], *, field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _optional_int(payload: Mapping[str, Any], *, field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an int when present, got {type(value).__name__}"
        )
    return int(value)


def _optional_float(payload: Mapping[str, Any], *, field_name: str) -> float | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be a float when present, got {type(value).__name__}"
        )
    return float(value)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _single_gpu_smoke_authority_mappings(
    verdict: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = [verdict]
    for field_name in (
        "launch_policy",
        "launch_contract",
        "effective_config",
        "delegate_summary",
        "summary",
        "training_summary",
    ):
        nested = verdict.get(field_name)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    return candidates


def _lookup_single_gpu_smoke_summary_path(verdict: Mapping[str, Any]) -> str | None:
    for candidate in _single_gpu_smoke_authority_mappings(verdict):
        summary_path = str(candidate.get("summary_path") or "").strip()
        if summary_path:
            return summary_path
    return None


def _lookup_prereq_dataset_path(smoke_summary: Mapping[str, Any]) -> str | None:
    dataset_path = str(smoke_summary.get("dataset_path") or "").strip()
    if dataset_path:
        return dataset_path
    delegate_summary = smoke_summary.get("delegate_summary")
    if isinstance(delegate_summary, Mapping):
        dataset_path = str(delegate_summary.get("dataset_path") or "").strip()
        if dataset_path:
            return dataset_path
    return None


def _load_prereq(
    repo_root: Path,
    *,
    single_gpu_smoke_verdict_path: Path,
) -> BaselineTrainPrereq:
    trainability_gate_path = _resolve_path(repo_root, single_gpu_smoke_verdict_path)
    smoke_verdict = _load_single_gpu_smoke_verdict(trainability_gate_path)
    smoke_summary_rel = _lookup_single_gpu_smoke_summary_path(smoke_verdict)
    if not smoke_summary_rel:
        raise ValueError(
            "single_gpu_smoke_verdict missing launch_policy.summary_path for live prereq resolution"
        )
    smoke_summary_path = _resolve_path(repo_root, smoke_summary_rel)
    smoke_summary = _read_json_object(smoke_summary_path)
    dataset_path_raw = _lookup_prereq_dataset_path(smoke_summary)
    if not dataset_path_raw:
        raise ValueError(
            "single_gpu smoke summary missing non-empty dataset_path for live prereq resolution"
        )
    dataset_path = _resolve_path(repo_root, dataset_path_raw)
    return BaselineTrainPrereq(
        smoke_summary_path=smoke_summary_path,
        trainability_gate_path=trainability_gate_path,
        dataset_path=dataset_path,
        max_steps=MEANINGFUL_MAX_STEPS,
        save_steps=MEANINGFUL_SAVE_STEPS,
        save_total_limit=MEANINGFUL_SAVE_TOTAL_LIMIT,
        global_batch_size=MEANINGFUL_GLOBAL_BATCH_SIZE,
        gradient_accumulation_steps=MEANINGFUL_GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=MEANINGFUL_DATALOADER_NUM_WORKERS,
        learning_rate=MEANINGFUL_LEARNING_RATE,
        num_gpus=MEANINGFUL_NUM_GPUS,
    )


def _build_live_authority_placeholder_prereq(
    repo_root: Path, *, single_gpu_smoke_verdict_path: Path
) -> BaselineTrainPrereq:
    authority_path = _resolve_path(repo_root, single_gpu_smoke_verdict_path)
    return BaselineTrainPrereq(
        smoke_summary_path=authority_path,
        trainability_gate_path=authority_path,
        dataset_path=authority_path.parent,
        max_steps=MEANINGFUL_MAX_STEPS,
        save_steps=MEANINGFUL_SAVE_STEPS,
        save_total_limit=MEANINGFUL_SAVE_TOTAL_LIMIT,
        global_batch_size=MEANINGFUL_GLOBAL_BATCH_SIZE,
        gradient_accumulation_steps=MEANINGFUL_GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=MEANINGFUL_DATALOADER_NUM_WORKERS,
        learning_rate=MEANINGFUL_LEARNING_RATE,
        num_gpus=MEANINGFUL_NUM_GPUS,
    )


def _authority_candidate_mappings(
    verdict: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = [verdict]
    for field_name in (
        "launch_policy",
        "launch_contract",
        "effective_config",
        "delegate_summary",
        "summary",
        "training_summary",
    ):
        nested = verdict.get(field_name)
        if isinstance(nested, Mapping):
            candidates.append(nested)
            nested_effective = nested.get("effective_config")
            if isinstance(nested_effective, Mapping):
                candidates.append(nested_effective)
    return candidates


def _lookup_launch_authority_value(
    verdict: Mapping[str, Any], field_name: str
) -> Any | None:
    for candidate in _authority_candidate_mappings(verdict):
        if field_name in candidate:
            value = candidate.get(field_name)
            if value is not None:
                return value
    return None


def _lookup_geometry_check_actual(
    verdict: Mapping[str, Any], field_name: str
) -> Any | None:
    geometry_checks = verdict.get("geometry_checks")
    if not isinstance(geometry_checks, Mapping):
        return None
    entry = geometry_checks.get(field_name)
    if not isinstance(entry, Mapping):
        return None
    actual = entry.get("actual")
    if actual is not None:
        return actual
    return entry.get(field_name)


def _load_single_gpu_smoke_verdict(path: Path) -> dict[str, Any]:
    verdict = _read_json_object(path)
    schema_version = str(verdict.get("schema_version") or "").strip()
    if schema_version == HISTORICAL_DDP_SMOKE_VERDICT_SCHEMA_VERSION:
        raise ValueError(
            "historical diagnostic verdict task10_green_smoke_verdict_v1 cannot be used as live single_gpu_v1 authority"
        )
    return verdict


def _validate_single_gpu_launch_authority(
    verdict: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    schema_version = str(verdict.get("schema_version") or "").strip()
    if schema_version != SINGLE_GPU_SMOKE_VERDICT_SCHEMA_VERSION:
        failures.append("single_gpu_smoke_verdict_schema_version_mismatch")
    if bool(verdict.get("pass")) is not True:
        failures.append("single_gpu_smoke_verdict_not_green")
    unlock_signal = str(verdict.get("unlock_signal") or "").strip()
    if unlock_signal != SINGLE_GPU_SMOKE_VERDICT_UNLOCK_SIGNAL:
        failures.append("single_gpu_smoke_verdict_unlock_signal_mismatch")

    expected_ints = {
        "num_gpus": int(MEANINGFUL_NUM_GPUS),
        "global_batch_size": int(MEANINGFUL_GLOBAL_BATCH_SIZE),
        "gradient_accumulation_steps": int(MEANINGFUL_GRADIENT_ACCUMULATION_STEPS),
        "per_device_batch_size": int(MEANINGFUL_PER_DEVICE_BATCH_SIZE),
        "effective_update_batch": int(MEANINGFUL_EFFECTIVE_UPDATE_BATCH),
    }
    for field_name, expected in expected_ints.items():
        observed = _coerce_int(
            _lookup_launch_authority_value(verdict, field_name)
            if field_name == "num_gpus"
            else _lookup_geometry_check_actual(verdict, field_name)
            or _lookup_launch_authority_value(verdict, field_name)
        )
        if observed is None:
            failures.append(f"single_gpu_launch_authority_{field_name}_missing")
            continue
        if int(observed) != int(expected):
            failures.append(f"single_gpu_launch_authority_{field_name}_mismatch")

    launch_family = str(
        _lookup_launch_authority_value(verdict, "launch_family")
        or _lookup_launch_authority_value(verdict, "live_launch_family")
        or (LIVE_LAUNCH_FAMILY if schema_version == SINGLE_GPU_SMOKE_VERDICT_SCHEMA_VERSION else "")
    ).strip()
    if launch_family != LIVE_LAUNCH_FAMILY:
        failures.append("single_gpu_launch_authority_launch_family_mismatch")
    use_ddp = _lookup_launch_authority_value(verdict, "use_ddp")
    if use_ddp is None:
        failures.append("single_gpu_launch_authority_use_ddp_missing")
    elif bool(use_ddp):
        failures.append("single_gpu_launch_authority_use_ddp_mismatch")
    return not failures, failures


def _attempt_paths(
    repo_root: Path,
    *,
    iteration_artifact_root: Path,
    formal_output_dir: Path,
) -> AttemptPaths:
    runtime_log_root = _resolve_path(repo_root, DEFAULT_RUNTIME_LOG_DIR_REL)
    attempt_dir = iteration_artifact_root / DEFAULT_ATTEMPT_DIRNAME
    checkpoint_gate_raw_dir = attempt_dir / DEFAULT_CKPT_GATE_RAW_DIRNAME
    run_manifest_gate_raw_dir = attempt_dir / DEFAULT_RUN_MANIFEST_GATE_RAW_DIRNAME
    return AttemptPaths(
        attempt_dir=attempt_dir,
        finetune_output_dir=formal_output_dir,
        finetune_summary_path=attempt_dir / DEFAULT_FINETUNE_SUMMARY_JSON_NAME,
        prelim_eval_summary_path=attempt_dir / DEFAULT_PRELIM_EVAL_SUMMARY_JSON_NAME,
        provenance_input_path=attempt_dir / DEFAULT_PROVENANCE_INPUT_JSON_NAME,
        prelim_run_manifest_input_path=(
            attempt_dir / DEFAULT_PRELIM_RUN_MANIFEST_INPUT_JSON_NAME
        ),
        checkpoint_gate_raw_dir=checkpoint_gate_raw_dir,
        checkpoint_gate_raw_report_path=(
            checkpoint_gate_raw_dir / CHECKPOINT_PROVENANCE_REPORT_JSON_NAME
        ),
        run_manifest_gate_raw_dir=run_manifest_gate_raw_dir,
        run_manifest_gate_raw_report_path=(
            run_manifest_gate_raw_dir / RUN_MANIFEST_REPORT_JSON_NAME
        ),
        run_manifest_gate_raw_manifest_path=(
            run_manifest_gate_raw_dir / RUN_MANIFEST_JSON_NAME
        ),
        train_log_path=runtime_log_root / DEFAULT_TRAIN_LOG_NAME,
        eval_log_path=runtime_log_root / DEFAULT_EVAL_LOG_NAME,
        checkpoint_gate_log_path=runtime_log_root / DEFAULT_CKPT_GATE_LOG_NAME,
        run_manifest_gate_log_path=runtime_log_root
        / DEFAULT_RUN_MANIFEST_GATE_LOG_NAME,
    )


def _run_logged_command(*, cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"\n[exit_code] {proc.returncode}\n")
        handle.flush()
    return int(proc.returncode)


def _run_json_command(*, cmd: list[str], cwd: Path) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout = str(proc.stdout or "").strip()
    try:
        raw_payload = json.loads(stdout) if stdout else {}
        payload: dict[str, Any] = (
            dict(raw_payload)
            if isinstance(raw_payload, Mapping)
            else {"raw_payload": raw_payload}
        )
    except json.JSONDecodeError:
        payload = {
            "raw_stdout": stdout,
            "raw_stderr": str(proc.stderr or "").strip(),
        }
    normalized: dict[str, Any] = dict(payload)
    normalized.setdefault("raw_stdout", stdout)
    normalized.setdefault("raw_stderr", str(proc.stderr or "").strip())
    normalized["returncode"] = int(proc.returncode)
    normalized["cmd"] = [str(part) for part in cmd]
    return int(proc.returncode), normalized


def _default_preflight_path(iteration_artifact_root: Path) -> Path:
    return iteration_artifact_root / DEFAULT_PREFLIGHT_JSON_NAME


def _collect_duplicate_processes(*, current_pid: int) -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,args"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [
            {
                "pid": None,
                "reason": "ps_failed",
                "stderr": str(proc.stderr or "").strip() or None,
            }
        ]
    patterns = (
        "30b_stage3_baseline_train_gate.py",
        "work/recap/finetune_full.py",
        "34_recap_finetune_repro.py",
        "launch_finetune.py",
        "45d_vlm_critic_eval_smoke.py",
        "3D_recap_run_adv_server.py",
    )
    shell_programs = {"bash", "sh", "zsh", "dash", "fish"}
    duplicates: list[dict[str, Any]] = []
    for raw_line in str(proc.stdout or "").splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        pid_text, _, args_text = line.partition(" ")
        pid_text = pid_text.strip()
        args_text = args_text.strip()
        if not pid_text.isdigit() or not args_text:
            continue
        pid = int(pid_text)
        if pid == int(current_pid):
            continue
        try:
            argv = shlex.split(args_text)
        except ValueError:
            argv = [args_text]
        program_name = Path(argv[0]).name if argv else ""
        is_shell_wrapper = program_name in shell_programs and "-c" in argv[1:]
        matched = [pattern for pattern in patterns if pattern in args_text]
        if matched and not is_shell_wrapper:
            duplicates.append(
                {
                    "pid": pid,
                    "matched_patterns": matched,
                    "args": args_text,
                    "program": program_name or None,
                }
            )
    return duplicates


def _query_gpu_inventory() -> tuple[int, dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    result: dict[str, Any] = {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "raw_stdout": str(proc.stdout or "").strip(),
        "raw_stderr": str(proc.stderr or "").strip(),
        "gpus": [],
    }
    if proc.returncode != 0:
        return int(proc.returncode), result
    gpus: list[dict[str, Any]] = []
    for line in str(proc.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index_text, name, total_text, free_text = parts[:4]
        try:
            gpus.append(
                {
                    "index": int(index_text),
                    "name": str(name),
                    "memory_total_mib": int(total_text),
                    "memory_free_mib": int(free_text),
                }
            )
        except ValueError:
            continue
    result["gpus"] = gpus
    return int(proc.returncode), result


def _build_delegate_probe_code() -> str:
    return (
        "import json\n"
        "payload = {}\n"
        "try:\n"
        "    import torch\n"
        "    payload['torch_import_ok'] = True\n"
        "    payload['torch_cuda_available'] = bool(torch.cuda.is_available())\n"
        "    payload['torch_cuda_arch_list'] = list(torch.cuda.get_arch_list())\n"
        "except Exception as exc:\n"
        "    payload['torch_import_ok'] = False\n"
        "    payload['torch_error'] = f'{type(exc).__name__}: {exc}'\n"
        "    payload['torch_cuda_available'] = False\n"
        "    payload['torch_cuda_arch_list'] = []\n"
        "try:\n"
        "    from transformers.utils import is_flash_attn_2_available\n"
        "    payload['flash_attn_2_available'] = bool(is_flash_attn_2_available())\n"
        "except Exception as exc:\n"
        "    payload['flash_attn_2_available'] = False\n"
        "    payload['transformers_error'] = f'{type(exc).__name__}: {exc}'\n"
        "try:\n"
        "    import flash_attn_2_cuda\n"
        "    payload['flash_attn_2_cuda_import_ok'] = True\n"
        "except Exception as exc:\n"
        "    payload['flash_attn_2_cuda_import_ok'] = False\n"
        "    payload['flash_attn_2_cuda_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, ensure_ascii=True, sort_keys=True))\n"
    )


def _run_rtx_pro_6000_blackwell_max_q_preflight(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    preflight_path: Path,
    training_python_contract: Mapping[str, str],
) -> dict[str, Any]:
    existing_preflight_path = _existing_preflight_path(preflight_path)
    existing_preflight_payload = _read_json_object_if_exists(existing_preflight_path)
    existing_preflight_sha256 = (
        _sha256_file(existing_preflight_path)
        if existing_preflight_path is not None and existing_preflight_path.is_file()
        else None
    )
    superseded_outputs_path = _default_superseded_outputs_path(preflight_path.parent)
    superseded_outputs_payload = _build_superseded_outputs_payload(
        repo_root=repo_root,
        superseded_outputs_path=superseded_outputs_path,
        preflight_path=preflight_path,
        existing_preflight_path=existing_preflight_path,
        existing_preflight_payload=existing_preflight_payload,
    )
    write_json(superseded_outputs_path, superseded_outputs_payload)

    duplicate_processes = _collect_duplicate_processes(current_pid=os.getpid())
    disk_usage = shutil.disk_usage(repo_root)
    disk_free_gib = float(disk_usage.free) / float(1024**3)
    gpu_query_rc, gpu_query = _query_gpu_inventory()
    repair_evidence = _load_delegate_runtime_repair_evidence(
        repo_root=repo_root,
        training_python_contract=training_python_contract,
    )
    delegate_probe_cmd = [
        str(training_python_contract["delegate_runtime_python"]),
        "-c",
        _build_delegate_probe_code(),
    ]
    delegate_probe_rc, delegate_probe = _run_json_command(
        cmd=delegate_probe_cmd,
        cwd=repo_root,
    )
    gpus = [
        dict(item)
        for item in list(gpu_query.get("gpus", []))
        if isinstance(item, Mapping)
    ]
    eligible_gpu_free = [
        int(_coerce_int(item.get("memory_free_mib")) or 0)
        for item in gpus
        if int(_coerce_int(item.get("memory_free_mib")) or 0)
        >= int(MIN_GPU_FREE_MEMORY_MIB)
    ]
    live_arch_list = list(delegate_probe.get("torch_cuda_arch_list", []))

    delegate_runtime_health_reason_codes: list[str] = []
    if int(delegate_probe_rc) != 0:
        delegate_runtime_health_reason_codes.append("delegate_probe_returncode_nonzero")
    if EXPECTED_CUDA_ARCH not in live_arch_list:
        delegate_runtime_health_reason_codes.append("delegate_torch_missing_sm_120")
    if not bool(delegate_probe.get("torch_cuda_available")):
        delegate_runtime_health_reason_codes.append("delegate_torch_cuda_unavailable")
    if not bool(delegate_probe.get("flash_attn_2_available")):
        delegate_runtime_health_reason_codes.append("delegate_flash_attn_2_unavailable")
    if not bool(delegate_probe.get("flash_attn_2_cuda_import_ok")):
        delegate_runtime_health_reason_codes.append(
            "delegate_flash_attn_2_cuda_import_failed"
        )
    delegate_probe_argv0 = str(delegate_probe_cmd[0])
    if delegate_probe_argv0 != str(training_python_contract["delegate_runtime_python"]):
        delegate_runtime_health_reason_codes.append(
            "delegate_probe_argv0_manifest_mismatch"
        )
    if not bool(repair_evidence.get("pass")):
        delegate_runtime_health_reason_codes.extend(
            list(repair_evidence.get("reason_codes") or [])
        )

    hardware_profile_reason_codes: list[str] = []
    gpu_names = [str(item.get("name") or "").strip() for item in gpus]
    if gpu_query_rc != 0:
        hardware_profile_reason_codes.append("nvidia_smi_query_failed")
    if len(gpus) < int(MEANINGFUL_NUM_GPUS):
        hardware_profile_reason_codes.append("gpu_count_below_2")
    if len(eligible_gpu_free) < int(MEANINGFUL_NUM_GPUS):
        hardware_profile_reason_codes.append("per_process_gpu_memory_below_8_gib")
    if gpus and not all(
        EXPECTED_GPU_NAME_SUBSTRING in name.lower() for name in gpu_names
    ):
        hardware_profile_reason_codes.append(REASON_GPU_NAME_NOT_EXPECTED)
    if gpus and not all(
        int(_coerce_int(item.get("memory_total_mib")) or 0)
        >= int(MIN_EXPECTED_GPU_TOTAL_MEMORY_MIB)
        for item in gpus
    ):
        hardware_profile_reason_codes.append(
            REASON_GPU_TOTAL_MEMORY_BELOW_EXPECTED_CLASS
        )

    blockers: list[str] = []
    blockers.extend(delegate_runtime_health_reason_codes)
    blockers.extend(hardware_profile_reason_codes)
    if duplicate_processes:
        blockers.append("duplicate_stage3_processes_detected")
    if disk_free_gib < float(MIN_DISK_FREE_GIB):
        blockers.append("disk_free_below_50_gib")

    delegate_runtime_health_pass = not delegate_runtime_health_reason_codes
    hardware_profile_match_pass = not hardware_profile_reason_codes
    operational_readiness_reason_codes: list[str] = []
    if duplicate_processes:
        operational_readiness_reason_codes.append("duplicate_stage3_processes_detected")
    if disk_free_gib < float(MIN_DISK_FREE_GIB):
        operational_readiness_reason_codes.append("disk_free_below_50_gib")

    if delegate_runtime_health_pass and not hardware_profile_match_pass:
        hard_block_subfamily = HARD_BLOCK_SUBFAMILY_HARDWARE_PROFILE_MISMATCH
        next_action = NEXT_ACTION_WAIT_FOR_RTX_PRO_6000_BLACKWELL_MAX_Q_X2_SUBSET
    elif not delegate_runtime_health_pass:
        hard_block_subfamily = HARD_BLOCK_SUBFAMILY_DELEGATE_RUNTIME_UNHEALTHY
        next_action = NEXT_ACTION_USER_ESCALATION_REQUIRED
    else:
        hard_block_subfamily = None
        next_action = NEXT_ACTION_USER_ESCALATION_REQUIRED

    manifest_hardware_profile = str(
        manifest_payload.get("hardware_profile") or ""
    ).strip()
    if manifest_hardware_profile in LEGACY_HARDWARE_PROFILES:
        manifest_hardware_profile = EXPECTED_HARDWARE_PROFILE

    payload = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "artifact_kind": PREFLIGHT_ARTIFACT_KIND,
        "generated_at": _now_iso(),
        "manifest_path": _repo_relative_path(repo_root, manifest_path),
        "artifact_path": _repo_relative_path(repo_root, preflight_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "hardware_profile": manifest_hardware_profile,
        "expected_hardware_profile": EXPECTED_HARDWARE_PROFILE,
        "delegate_runtime_python_manifest": str(
            training_python_contract["delegate_runtime_python"]
        ),
        "delegate_runtime_python_realpath": str(
            Path(str(training_python_contract["delegate_runtime_python"])).resolve()
        ),
        "orchestrator_python_manifest": str(
            training_python_contract["orchestrator_python"]
        ),
        "orchestrator_python_realpath": str(
            Path(str(training_python_contract["orchestrator_python"])).resolve()
        ),
        "repair_tool_status": str(repair_evidence.get("status") or ""),
        "pass": not blockers,
        "status": STATUS_CONTINUE if not blockers else STATUS_EXECUTION_HARD_BLOCK,
        "reason_codes": list(dict.fromkeys(blockers)),
        "hard_block_subfamily": hard_block_subfamily,
        "next_action": next_action,
        "delegate_runtime_health": {
            "pass": delegate_runtime_health_pass,
            "reason_codes": list(dict.fromkeys(delegate_runtime_health_reason_codes)),
            "argv0_matches_manifest": delegate_probe_argv0
            == str(training_python_contract["delegate_runtime_python"]),
            "probe_returncode": int(delegate_probe_rc),
            "torch_import_ok": bool(delegate_probe.get("torch_import_ok")),
            "torch_cuda_available": bool(delegate_probe.get("torch_cuda_available")),
            "torch_cuda_arch_list": live_arch_list,
            "torch_error": delegate_probe.get("torch_error"),
            "flash_attn_2_available": bool(
                delegate_probe.get("flash_attn_2_available")
            ),
            "transformers_error": delegate_probe.get("transformers_error"),
            "flash_attn_2_cuda_import_ok": bool(
                delegate_probe.get("flash_attn_2_cuda_import_ok")
            ),
            "flash_attn_2_cuda_error": delegate_probe.get("flash_attn_2_cuda_error"),
            "repair_summary_status": repair_evidence.get("status"),
            "repair_summary_pass": bool(repair_evidence.get("pass")),
            "repair_summary_reason_codes": list(
                repair_evidence.get("reason_codes") or []
            ),
            "repair_summary_checked_at": repair_evidence["summary"].get("checked_at"),
            "repair_summary_probe_python_executable": repair_evidence["final_probe"][
                "payload"
            ].get("python_executable"),
        },
        "hardware_profile_match": {
            "pass": hardware_profile_match_pass,
            "reason_codes": list(dict.fromkeys(hardware_profile_reason_codes)),
            "expected_profile": EXPECTED_HARDWARE_PROFILE,
            "manifest_profile": manifest_hardware_profile,
            "observed_profile": _build_observed_hardware_profile(gpus),
            "gpu_query_returncode": int(gpu_query_rc),
            "required_gpu_count": int(MEANINGFUL_NUM_GPUS),
            "required_free_mib": int(MIN_GPU_FREE_MEMORY_MIB),
            "observed_gpu_count": int(len(gpus)),
            "observed_gpu_names": gpu_names,
            "gpus": gpus,
        },
        "operational_readiness": {
            "pass": not operational_readiness_reason_codes,
            "reason_codes": operational_readiness_reason_codes,
            "duplicate_processes": {
                "pass": not duplicate_processes,
                "duplicates": duplicate_processes,
            },
            "disk_free_gib": {
                "pass": disk_free_gib >= float(MIN_DISK_FREE_GIB),
                "observed_gib": disk_free_gib,
                "required_gib": float(MIN_DISK_FREE_GIB),
            },
        },
        "evidence": {
            "superseded_outputs_path": _repo_relative_path(
                repo_root, superseded_outputs_path
            ),
            "delegate_runtime_repair_summary_path": repair_evidence["summary_rel"],
            "delegate_runtime_repair_summary_sha256": repair_evidence["summary_sha256"],
            "delegate_runtime_repair_log": repair_evidence["log_rel"],
            "delegate_runtime_repair_log_sha256": repair_evidence["log_sha256"],
        },
        "probe_commands": {
            "delegate_runtime_probe": delegate_probe_cmd,
            "nvidia_smi_query": gpu_query.get("cmd"),
            "ps_query": ["ps", "-eo", "pid,args"],
        },
    }

    _validate_preflight_authority(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        superseded_outputs_path=superseded_outputs_path,
        superseded_outputs_payload=superseded_outputs_payload,
        preflight_payload=payload,
        existing_preflight_path=existing_preflight_path,
        existing_preflight_payload=existing_preflight_payload,
        existing_preflight_sha256=existing_preflight_sha256,
    )
    write_json(preflight_path, payload)
    return payload


def _preflight_manifest_entry(
    *, repo_root: Path, preflight_path: Path, preflight_payload: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "json_path": _repo_relative_path(repo_root, preflight_path),
        "schema_version": str(preflight_payload.get("schema_version") or "").strip(),
        "pass": bool(preflight_payload.get("pass")),
        "status": str(preflight_payload.get("status") or "").strip(),
        "reason_codes": list(preflight_payload.get("reason_codes") or []),
        "checked_at": str(preflight_payload.get("generated_at") or "").strip(),
        "hard_block_subfamily": str(
            preflight_payload.get("hard_block_subfamily") or ""
        ).strip()
        or None,
        "next_action": str(preflight_payload.get("next_action") or "").strip() or None,
        "repair_tool_status": str(
            preflight_payload.get("repair_tool_status") or ""
        ).strip()
        or None,
    }


def _build_attempt_record(
    *,
    repo_root: Path,
    prereq: BaselineTrainPrereq,
    attempt_paths: AttemptPaths,
    started_at: str,
    state: str,
    prelaunch_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_number": 1,
        "attempt_limit": 1,
        "single_attempt_only": True,
        "state": str(state),
        "started_at": str(started_at),
        "official_task_anchor_model": OFFICIAL_TASK_ANCHOR_MODEL,
        "prereq_smoke_summary_path": _repo_relative_path(
            repo_root, prereq.smoke_summary_path
        ),
        "trainability_gate_path": _repo_relative_path(
            repo_root, prereq.trainability_gate_path
        ),
        "dataset_path": _repo_relative_path(repo_root, prereq.dataset_path),
        "finetune_output_dir": _repo_relative_path(
            repo_root, attempt_paths.finetune_output_dir
        ),
        "finetune_summary_path": _repo_relative_path(
            repo_root, attempt_paths.finetune_summary_path
        ),
        "prelim_eval_summary_path": _repo_relative_path(
            repo_root, attempt_paths.prelim_eval_summary_path
        ),
        "checkpoint_provenance_input_path": _repo_relative_path(
            repo_root, attempt_paths.provenance_input_path
        ),
        "prelim_run_manifest_input_path": _repo_relative_path(
            repo_root, attempt_paths.prelim_run_manifest_input_path
        ),
        "checkpoint_provenance_gate_report_path": _repo_relative_path(
            repo_root, attempt_paths.checkpoint_gate_raw_report_path
        ),
        "run_manifest_gate_report_path": _repo_relative_path(
            repo_root, attempt_paths.run_manifest_gate_raw_report_path
        ),
    }
    if prelaunch_summary is not None:
        payload["prelaunch_summary"] = dict(prelaunch_summary)
    return payload


def _write_attempt_started_manifest(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    prereq: BaselineTrainPrereq,
    attempt_paths: AttemptPaths,
    started_at: str,
    prelaunch_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(manifest_payload)
    updated["collect_policy_ckpt_baseline_train_attempt"] = _build_attempt_record(
        repo_root=repo_root,
        prereq=prereq,
        attempt_paths=attempt_paths,
        started_at=started_at,
        state=ATTEMPT_STATE_STARTED,
        prelaunch_summary=prelaunch_summary,
    )
    write_json(manifest_path, updated)
    return updated


def _authority_refs_from_manifest(
    manifest_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    provenance = manifest_payload.get("collect_policy_ckpt_provenance")
    if not isinstance(provenance, Mapping):
        return []
    authority_refs = provenance.get("authority_refs")
    if not isinstance(authority_refs, list):
        return []
    return [dict(item) for item in authority_refs if isinstance(item, Mapping)]


def _build_dataset_fingerprint(
    *,
    manifest_payload: Mapping[str, Any],
    prereq: BaselineTrainPrereq,
) -> str:
    formal_iter_tag = str(manifest_payload.get("formal_iter_tag") or "stage3_iteration")
    return f"stage3_baseline_train::{formal_iter_tag}::{prereq.dataset_path.name}::attempt_001"


def _build_finetune_cmd(
    *,
    repo_root: Path,
    manifest_payload: Mapping[str, Any],
    prereq: BaselineTrainPrereq,
    attempt_paths: AttemptPaths,
    training_python_contract: Mapping[str, str],
) -> list[str]:
    cmd = [
        str(training_python_contract["orchestrator_python"]),
        str(_resolve_path(repo_root, "work/recap/finetune_full.py")),
        "--dataset-path",
        str(prereq.dataset_path),
        "--output-dir",
        str(attempt_paths.finetune_output_dir),
        "--runtime-log-dir",
        str(attempt_paths.train_log_path.parent),
        "--summary-json",
        str(attempt_paths.finetune_summary_path),
        "--base-model",
        OFFICIAL_TASK_ANCHOR_MODEL,
        "--embodiment-tag",
        "UNITREE_G1",
        "--max-steps",
        str(prereq.max_steps),
        "--save-steps",
        str(prereq.save_steps),
        "--save-total-limit",
        "1",
        "--global-batch-size",
        str(prereq.global_batch_size),
        "--gradient-accumulation-steps",
        str(prereq.gradient_accumulation_steps),
        "--dataloader-num-workers",
        str(prereq.dataloader_num_workers),
        "--tune-projector",
        "--tune-diffusion-model",
        "--no-use-wandb",
        "--python",
        str(training_python_contract["delegate_runtime_python"]),
    ]
    if prereq.learning_rate is not None:
        cmd.extend(["--learning-rate", str(prereq.learning_rate)])
    if prereq.num_gpus is not None:
        cmd.extend(["--num-gpus", str(prereq.num_gpus)])
    return cmd


def _baseline_default_adv_init_compatible(
    checkpoint_features: Mapping[str, Any],
) -> bool:
    return bool(checkpoint_features.get("baseline_like_path")) and not bool(
        checkpoint_features.get("has_advantage_embedding_pair")
    )


def _build_eval_cmd(
    *,
    repo_root: Path,
    manifest_payload: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_features: Mapping[str, Any],
    attempt_paths: AttemptPaths,
    training_python_contract: Mapping[str, str],
) -> list[str]:
    env_name = _manifest_env_name(manifest_payload)
    cmd = [
        str(training_python_contract["orchestrator_python"]),
        str(
            _resolve_path(repo_root, "work/recap/scripts/45d_vlm_critic_eval_smoke.py")
        ),
        "--model-path",
        str(checkpoint_path),
        "--env-name",
        str(env_name),
        "--summary-json",
        str(attempt_paths.prelim_eval_summary_path),
        "--runtime-log-dir",
        str(attempt_paths.eval_log_path.parent),
        "--artifact-dir",
        str(attempt_paths.attempt_dir / "prelim_eval_artifacts"),
        "--telemetry-dir",
        str(attempt_paths.attempt_dir / "prelim_eval_telemetry"),
        "--n-episodes",
        str(DEFAULT_EVAL_EPISODES),
        "--advantage",
        "None",
        "--eval-label",
        "stage3_baseline_train_prelim",
        "--port",
        str(DEFAULT_EVAL_PORT),
        "--python",
        str(training_python_contract["orchestrator_python"]),
        "--main-repo-root",
        str(repo_root),
    ]
    if not _baseline_default_adv_init_compatible(checkpoint_features):
        cmd.extend(["--base-model-path", OFFICIAL_TASK_ANCHOR_MODEL])
    return cmd


def _mirror_formal_root_artifacts(
    *,
    formal_output_dir: Path,
    checkpoint_path: Path | None,
    checkpoint_gate_payload: Mapping[str, Any] | None = None,
    run_manifest_gate_payload: Mapping[str, Any] | None = None,
) -> None:
    formal_output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is not None:
        checkpoint_trainer_state_path = checkpoint_path / "trainer_state.json"
        if checkpoint_trainer_state_path.is_file():
            write_json(
                formal_output_dir / "trainer_state.json",
                _read_json_object(checkpoint_trainer_state_path),
            )
    if checkpoint_gate_payload:
        write_json(
            formal_output_dir / CHECKPOINT_PROVENANCE_GATE_JSON_NAME,
            dict(checkpoint_gate_payload),
        )
    if run_manifest_gate_payload:
        write_json(
            formal_output_dir / RUN_MANIFEST_GATE_JSON_NAME,
            dict(run_manifest_gate_payload),
        )


def _read_json_object_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return _read_json_object(path)


def _read_eval_summary(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    success_rate = payload.get("success_rate")
    if isinstance(success_rate, bool) or not isinstance(success_rate, (int, float)):
        raise TypeError("prelim eval summary missing numeric success_rate")
    return payload


def _build_gate_inputs(
    *,
    repo_root: Path,
    manifest_payload: Mapping[str, Any],
    prereq: BaselineTrainPrereq,
    attempt_paths: AttemptPaths,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_fingerprint = _build_dataset_fingerprint(
        manifest_payload=manifest_payload,
        prereq=prereq,
    )
    provenance_input = _build_candidate_provenance_metadata(
        checkpoint_path=str(checkpoint_path),
        base_model_path=OFFICIAL_TASK_ANCHOR_MODEL,
        eval_uses_finetuned=True,
    )
    prelim_run_manifest = _build_preliminary_run_manifest(
        checkpoint_path=str(checkpoint_path),
        base_model_path=OFFICIAL_TASK_ANCHOR_MODEL,
        eval_uses_finetuned=True,
        dataset_fingerprint=dataset_fingerprint,
        env_name=_manifest_env_name(manifest_payload),
    )
    write_json(attempt_paths.provenance_input_path, provenance_input)
    write_json(attempt_paths.prelim_run_manifest_input_path, prelim_run_manifest)
    return provenance_input, prelim_run_manifest


def _build_checkpoint_gate_cmd(
    *,
    repo_root: Path,
    attempt_paths: AttemptPaths,
    training_python_contract: Mapping[str, str],
) -> list[str]:
    return [
        str(training_python_contract["orchestrator_python"]),
        str(
            _resolve_path(
                repo_root,
                "work/recap/scripts/gr00t_checkpoint_provenance_gate.py",
            )
        ),
        "--metadata",
        str(attempt_paths.provenance_input_path),
        "--output-dir",
        str(attempt_paths.checkpoint_gate_raw_dir),
    ]


def _build_run_manifest_gate_cmd(
    *,
    repo_root: Path,
    attempt_paths: AttemptPaths,
    training_python_contract: Mapping[str, str],
) -> list[str]:
    return [
        str(training_python_contract["orchestrator_python"]),
        str(_resolve_path(repo_root, "work/recap/scripts/gr00t_run_manifest_gate.py")),
        "--manifest-json",
        str(attempt_paths.prelim_run_manifest_input_path),
        "--output-dir",
        str(attempt_paths.run_manifest_gate_raw_dir),
    ]


def _normalize_checkpoint_report_for_payload(
    *, checkpoint_report: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    normalized = dict(checkpoint_report)
    if "is_base_fallback" not in normalized:
        normalized["is_base_fallback"] = not bool(
            normalized.get("formal_eligibility") == "ALLOW" and checkpoint_path
        )
    return normalized


def _candidate_eval_from_gate_reports(
    *,
    checkpoint_path: Path,
    eval_summary: Mapping[str, Any],
    authority_refs: list[dict[str, Any]],
    final_decision: str,
    checkpoint_report: Mapping[str, Any],
    prelim_run_manifest: Mapping[str, Any],
    run_manifest_report: Mapping[str, Any],
) -> dict[str, Any]:
    success_rate = float(eval_summary.get("success_rate") or 0.0)
    success_count = int(eval_summary.get("success_count") or 0)
    normalized_checkpoint_report = _normalize_checkpoint_report_for_payload(
        checkpoint_report=checkpoint_report,
        checkpoint_path=checkpoint_path,
    )
    return {
        "candidate_id": DEFAULT_ATTEMPT_DIRNAME,
        "candidate_tier": "baseline_train",
        "decision_on_select": final_decision,
        "checkpoint_path": str(checkpoint_path),
        "success_rate": success_rate,
        "success_count": success_count,
        "explicit_checkpoint_identity_present": True,
        "checkpoint_provenance": normalized_checkpoint_report,
        "run_manifest_validation": {
            "formal_eligibility": run_manifest_report.get("formal_eligibility"),
            "core_digest": run_manifest_report.get("core_digest"),
            "normalized_manifest": dict(prelim_run_manifest),
            "issues": list(run_manifest_report.get("issues", [])),
            "checkpoint_binding": dict(
                run_manifest_report.get("checkpoint_binding", {})
            ),
        },
        "authority_refs": authority_refs,
    }


def _build_hard_block_metadata(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    prereq: BaselineTrainPrereq,
    attempt_paths: AttemptPaths,
    finetune_returncode: int,
    eval_returncode: int,
    checkpoint_gate_returncode: int,
    run_manifest_gate_returncode: int,
    status_family: str,
    hard_blocker: str,
    reason_codes: list[str],
    preflight_path: Path,
    checkpoint_path: Path | None = None,
    eval_summary: Mapping[str, Any] | None = None,
    checkpoint_gate_payload: Mapping[str, Any] | None = None,
    run_manifest_gate_payload: Mapping[str, Any] | None = None,
    threshold: float | None = None,
    attempt_budget_consumed: bool = True,
    next_action: str = NEXT_ACTION_USER_ESCALATION_REQUIRED,
    hard_block_subfamily: str | None = None,
    repair_tool_status: str | None = None,
    prelaunch_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eval_summary = dict(eval_summary or {})
    checkpoint_gate_payload = dict(checkpoint_gate_payload or {})
    run_manifest_gate_payload = dict(run_manifest_gate_payload or {})
    success_rate = float(eval_summary.get("success_rate") or 0.0)
    eval_wrapper_status = str(eval_summary.get("wrapper_status") or "").strip()
    completed_episodes = int(eval_summary.get("episodes") or 0)
    artifact_paths: dict[str, Any] = {
        "manifest_path": _repo_relative_path(repo_root, manifest_path),
        PREFLIGHT_PATH_FIELD: _repo_relative_path(repo_root, preflight_path),
        "prereq_smoke_summary_path": _repo_relative_path(
            repo_root, prereq.smoke_summary_path
        ),
        "checkpoint_provenance_gate_path": _repo_relative_path(
            repo_root,
            manifest_path.parent / CHECKPOINT_PROVENANCE_GATE_JSON_NAME,
        ),
        "run_manifest_gate_path": _repo_relative_path(
            repo_root,
            manifest_path.parent / RUN_MANIFEST_GATE_JSON_NAME,
        ),
    }
    if attempt_budget_consumed:
        artifact_paths["finetune_summary_path"] = _repo_relative_path(
            repo_root, attempt_paths.finetune_summary_path
        )
        artifact_paths["prelim_eval_summary_path"] = _repo_relative_path(
            repo_root, attempt_paths.prelim_eval_summary_path
        )
    payload = {
        "schema_version": HARD_BLOCK_SCHEMA_VERSION,
        "status_family": str(status_family),
        "hard_blocker": str(hard_blocker),
        "hard_block_subfamily": None
        if not hard_block_subfamily
        else str(hard_block_subfamily),
        "next_action": str(next_action),
        "hard_blocked_at": _now_iso(),
        "reason_codes": list(reason_codes),
        "thresholds": {
            "prelim_success_rate": None if threshold is None else float(threshold),
            "prelim_success_count_hint": int(SUCCESS_GATE_THRESHOLD_COUNT),
        },
        "observed": {
            "checkpoint_path": None
            if checkpoint_path is None
            else _repo_relative_path(repo_root, checkpoint_path),
            "prelim_success_rate": success_rate,
            "prelim_success_count": int(eval_summary.get("success_count") or 0),
            "prelim_episodes": completed_episodes,
            "prelim_requested_episodes": int(
                eval_summary.get("requested_episodes") or DEFAULT_EVAL_EPISODES
            ),
            "prelim_eval_wrapper_status": eval_wrapper_status or None,
            "finetune_returncode": int(finetune_returncode),
            "prelim_eval_returncode": int(eval_returncode),
            "checkpoint_provenance_gate_returncode": int(checkpoint_gate_returncode),
            "run_manifest_gate_returncode": int(run_manifest_gate_returncode),
            "checkpoint_provenance_gate_pass": bool(
                checkpoint_gate_payload.get("pass")
            ),
            "run_manifest_gate_pass": bool(run_manifest_gate_payload.get("pass")),
        },
        "artifact_paths": artifact_paths,
        "attempt_guard": {
            "single_attempt_only": True,
            "attempt_record_field": "collect_policy_ckpt_baseline_train_attempt",
        },
        "attempt_budget_consumed": bool(attempt_budget_consumed),
        "t3_start_forbidden": bool(attempt_budget_consumed),
        "env_name": _manifest_env_name(manifest_payload),
        "official_task_anchor_model": OFFICIAL_TASK_ANCHOR_MODEL,
        "repair_tool_status": None
        if not repair_tool_status
        else str(repair_tool_status),
    }
    if prelaunch_summary is not None:
        payload["prelaunch_summary"] = dict(prelaunch_summary)
    return payload


def _build_collect_policy_ckpt_provenance(
    *,
    repo_root: Path,
    manifest_payload: Mapping[str, Any],
    prereq: BaselineTrainPrereq,
    attempt_paths: AttemptPaths,
    checkpoint_path: Path,
    eval_summary: Mapping[str, Any],
    checkpoint_gate_payload: Mapping[str, Any],
    run_manifest_gate_payload: Mapping[str, Any],
    final_decision: str,
    threshold: float,
) -> dict[str, Any]:
    authority_refs = _authority_refs_from_manifest(manifest_payload)
    success_rate = float(eval_summary.get("success_rate") or 0.0)
    success_count = int(eval_summary.get("success_count") or 0)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "selected_at": _now_iso(),
        "selected_candidate_id": DEFAULT_ATTEMPT_DIRNAME,
        "selected_candidate_tier": "baseline_train",
        "decision_on_select": final_decision,
        "selected_checkpoint_path": (
            str(checkpoint_path)
            if final_decision == DECISION_BASELINE_TRAINED
            else None
        ),
        "success_rate": success_rate,
        "success_count": success_count,
        "success_rate_source": (
            f"{_repo_relative_path(repo_root, attempt_paths.prelim_eval_summary_path)}:success_rate"
        ),
        "success_count_source": (
            f"{_repo_relative_path(repo_root, attempt_paths.prelim_eval_summary_path)}:success_count"
        ),
        "explicit_checkpoint_identity_present": True,
        "success_rate_threshold_pass": success_rate >= float(threshold),
        "success_count_threshold_pass": success_count
        >= int(SUCCESS_GATE_THRESHOLD_COUNT),
        "checkpoint_provenance_gate": {
            "path": _repo_relative_path(
                repo_root,
                attempt_paths.attempt_dir.parent / CHECKPOINT_PROVENANCE_GATE_JSON_NAME,
            ),
            "pass": checkpoint_gate_payload.get("pass"),
            "formal_eligibility": checkpoint_gate_payload.get("formal_eligibility"),
            "reason_code": checkpoint_gate_payload.get("reason_code"),
        },
        "run_manifest_gate": {
            "path": _repo_relative_path(
                repo_root,
                attempt_paths.attempt_dir.parent / RUN_MANIFEST_GATE_JSON_NAME,
            ),
            "pass": run_manifest_gate_payload.get("pass"),
            "formal_eligibility": run_manifest_gate_payload.get("formal_eligibility"),
            "reason_code": run_manifest_gate_payload.get("reason_code"),
        },
        "authority_refs": authority_refs,
        "candidate_notes": [
            "single baseline-train attempt mandated by T0c",
            f"trained from official task anchor {OFFICIAL_TASK_ANCHOR_MODEL}",
            (
                "prereq smoke dataset sourced from "
                + _repo_relative_path(repo_root, prereq.smoke_summary_path)
            ),
        ],
        "baseline_train_attempt": {
            "attempt_number": 1,
            "single_attempt_only": True,
            "attempt_dir": _repo_relative_path(repo_root, attempt_paths.attempt_dir),
            "dataset_path": _repo_relative_path(repo_root, prereq.dataset_path),
            "finetune_summary_path": _repo_relative_path(
                repo_root, attempt_paths.finetune_summary_path
            ),
            "prelim_eval_summary_path": _repo_relative_path(
                repo_root, attempt_paths.prelim_eval_summary_path
            ),
            "checkpoint_provenance_input_path": _repo_relative_path(
                repo_root, attempt_paths.provenance_input_path
            ),
            "prelim_run_manifest_input_path": _repo_relative_path(
                repo_root, attempt_paths.prelim_run_manifest_input_path
            ),
        },
        "viable": bool(
            final_decision == DECISION_BASELINE_TRAINED
            and checkpoint_gate_payload.get("pass")
            and run_manifest_gate_payload.get("pass")
        ),
    }


def _upgrade_manifest_payload(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    preflight_path: Path,
    preflight_payload: Mapping[str, Any] | None,
    workflow_status: str,
    collect_policy_ckpt_decision: str,
    collect_policy_ckpt_path: str | None = None,
    collect_policy_ckpt_provenance: Mapping[str, Any] | None = None,
    attempt_record: Mapping[str, Any] | None = None,
    hard_block_metadata: Mapping[str, Any] | None,
    eval_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    upgraded = dict(manifest_payload)
    upgraded["schema_version"] = STAGE3_ITERATION_MANIFEST_V3
    upgraded["collect_policy_ckpt_decision"] = str(collect_policy_ckpt_decision)
    upgraded["collect_policy_ckpt_path"] = collect_policy_ckpt_path
    upgraded["collect_policy_ckpt_t3_status"] = str(workflow_status)
    upgraded.pop(LEGACY_PREFLIGHT_MANIFEST_KEY, None)
    upgraded[PREFLIGHT_MANIFEST_KEY] = _preflight_manifest_entry(
        repo_root=repo_root,
        preflight_path=preflight_path,
        preflight_payload=preflight_payload or {},
    )
    upgraded["next_action"] = (
        None
        if hard_block_metadata is None
        else str(hard_block_metadata.get("next_action") or "").strip() or None
    )
    upgraded["collect_policy_ckpt_selected_at"] = _now_iso()
    if collect_policy_ckpt_provenance is not None:
        upgraded["collect_policy_ckpt_provenance"] = dict(
            collect_policy_ckpt_provenance
        )
    if attempt_record is not None:
        upgraded["collect_policy_ckpt_baseline_train_attempt"] = dict(attempt_record)
    if eval_summary is not None:
        upgraded["collect_policy_ckpt_prelim_success_rate"] = float(
            eval_summary.get("success_rate") or 0.0
        )
        upgraded["collect_policy_ckpt_prelim_success_count"] = int(
            eval_summary.get("success_count") or 0
        )
        upgraded["collect_policy_ckpt_prelim_episodes"] = int(
            eval_summary.get("episodes") or 0
        )
    upgraded["success_gate_threshold_count"] = int(SUCCESS_GATE_THRESHOLD_COUNT)
    upgraded["historical_success_rate_threshold"] = float(
        upgraded.get("historical_success_rate_threshold")
        or HISTORICAL_SUCCESS_RATE_THRESHOLD
    )
    upgraded["official_task_anchor_model"] = OFFICIAL_TASK_ANCHOR_MODEL
    upgraded["hardware_profile"] = str(
        upgraded.get("hardware_profile") or EXPECTED_HARDWARE_PROFILE
    )
    if upgraded["hardware_profile"] in LEGACY_HARDWARE_PROFILES:
        upgraded["hardware_profile"] = EXPECTED_HARDWARE_PROFILE
    if hard_block_metadata is None:
        upgraded["collect_policy_ckpt_hard_block"] = None
    else:
        upgraded["collect_policy_ckpt_hard_block"] = dict(hard_block_metadata)
    return upgraded


def _validate_finetune_summary_config(
    finetune_summary: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    effective_config = finetune_summary.get("effective_config")
    if not isinstance(effective_config, Mapping):
        return False, ["finetune_effective_config_missing"]
    checks = {
        "max_steps": int(MEANINGFUL_MAX_STEPS),
        "save_steps": int(MEANINGFUL_SAVE_STEPS),
        "save_total_limit": int(MEANINGFUL_SAVE_TOTAL_LIMIT),
        "global_batch_size": int(MEANINGFUL_GLOBAL_BATCH_SIZE),
        "gradient_accumulation_steps": int(MEANINGFUL_GRADIENT_ACCUMULATION_STEPS),
        "per_device_batch_size": int(MEANINGFUL_PER_DEVICE_BATCH_SIZE),
        "effective_update_batch": int(MEANINGFUL_EFFECTIVE_UPDATE_BATCH),
        "dataloader_num_workers": int(MEANINGFUL_DATALOADER_NUM_WORKERS),
        "num_gpus": int(MEANINGFUL_NUM_GPUS),
    }
    failures: list[str] = []
    for field_name, expected in checks.items():
        if int(_coerce_int(effective_config.get(field_name)) or 0) != int(expected):
            failures.append(f"effective_config_{field_name}_mismatch")
    learning_rate = _coerce_float(effective_config.get("learning_rate"))
    if float(learning_rate or 0.0) != float(MEANINGFUL_LEARNING_RATE):
        failures.append("effective_config_learning_rate_mismatch")
    if bool(effective_config.get("tune_projector")) is not bool(
        MEANINGFUL_TUNE_PROJECTOR
    ):
        failures.append("effective_config_tune_projector_mismatch")
    if bool(effective_config.get("tune_diffusion_model")) is not bool(
        MEANINGFUL_TUNE_DIFFUSION_MODEL
    ):
        failures.append("effective_config_tune_diffusion_model_mismatch")
    return not failures, failures


def _checkpoint_count_ok(output_dir: Path) -> tuple[bool, int]:
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    return len(checkpoints) == 1, int(len(checkpoints))


def run_stage3_baseline_train_gate(
    *,
    repo_root: Path,
    manifest_path: Path | None = None,
    preflight_only: bool = False,
    launch_family: str = LIVE_LAUNCH_FAMILY,
    single_gpu_smoke_verdict_path: Path | str | None = None,
    formal_output_dir: Path | str | None = None,
) -> dict[str, Any]:
    resolved_repo_root = Path(repo_root).resolve()
    resolved_manifest_path = (
        _resolve_path(resolved_repo_root, manifest_path)
        if manifest_path is not None
        else _resolve_path(resolved_repo_root, DEFAULT_STAGE3_ITERATION_MANIFEST_REL)
    )
    manifest_payload = _read_json_object(resolved_manifest_path)
    decision = str(manifest_payload.get("collect_policy_ckpt_decision") or "").strip()
    if decision != DECISION_BASELINE_TRAIN_REQUIRED:
        return {
            "manifest_path": str(resolved_manifest_path),
            "status": "skipped_by_condition",
            "exit_code": 0,
            "collect_policy_ckpt_decision": decision,
            "reason": (
                "iteration manifest decision is not baseline_train_required; "
                "T0c performs no side effects"
            ),
        }

    prior_attempt = manifest_payload.get("collect_policy_ckpt_baseline_train_attempt")
    repair_started_attempt = False
    if isinstance(prior_attempt, Mapping):
        repair_started_attempt = (
            str(prior_attempt.get("state") or "").strip() == ATTEMPT_STATE_STARTED
        )
    if isinstance(prior_attempt, Mapping) and not repair_started_attempt:
        return {
            "manifest_path": str(resolved_manifest_path),
            "status": "blocked_second_attempt",
            "exit_code": 1,
            "collect_policy_ckpt_decision": decision,
            "reason": "baseline-train attempt state already recorded; refusing second attempt",
        }

    training_python_contract = _load_training_python_contract_from_manifest(
        resolved_repo_root,
        manifest_payload,
        manifest_path=resolved_manifest_path,
    )
    manifest_payload = dict(manifest_payload)
    manifest_payload["orchestrator_python"] = str(
        training_python_contract["orchestrator_python"]
    )
    manifest_payload["delegate_runtime_python"] = str(
        training_python_contract["delegate_runtime_python"]
    )
    if str(manifest_payload.get("hardware_profile") or "") in LEGACY_HARDWARE_PROFILES:
        manifest_payload["hardware_profile"] = EXPECTED_HARDWARE_PROFILE
    iteration_artifact_root = resolved_manifest_path.parent.resolve()
    resolved_formal_output_dir = resolve_full_update_authority_output_dir(
        resolved_repo_root,
        DEFAULT_FORMAL_OUTPUT_DIR_REL if formal_output_dir is None else formal_output_dir,
        require_v2_authority=False,
    )
    resolved_launch_family = str(launch_family or "").strip() or LIVE_LAUNCH_FAMILY
    resolved_single_gpu_smoke_verdict_path = _resolve_path(
        resolved_repo_root,
        DEFAULT_SINGLE_GPU_SMOKE_VERDICT_REL
        if single_gpu_smoke_verdict_path is None
        else single_gpu_smoke_verdict_path,
    )
    preflight_path = _default_preflight_path(iteration_artifact_root)
    preflight_payload = _run_rtx_pro_6000_blackwell_max_q_preflight(
        repo_root=resolved_repo_root,
        manifest_path=resolved_manifest_path,
        manifest_payload=manifest_payload,
        preflight_path=preflight_path,
        training_python_contract=training_python_contract,
    )
    if not bool(preflight_payload.get("pass")):
        hard_block_metadata = _build_hard_block_metadata(
            repo_root=resolved_repo_root,
            manifest_path=resolved_manifest_path,
            manifest_payload=manifest_payload,
            prereq=_load_prereq(
                resolved_repo_root,
                single_gpu_smoke_verdict_path=resolved_single_gpu_smoke_verdict_path,
            ),
            attempt_paths=_attempt_paths(
                resolved_repo_root,
                iteration_artifact_root=iteration_artifact_root,
                formal_output_dir=resolved_formal_output_dir,
            ),
            finetune_returncode=1,
            eval_returncode=1,
            checkpoint_gate_returncode=1,
            run_manifest_gate_returncode=1,
            status_family=STATUS_EXECUTION_HARD_BLOCK,
            hard_blocker=HARD_BLOCKER_RTX_PRO_6000_BLACKWELL_MAX_Q_PREFLIGHT_FAILED,
            reason_codes=list(preflight_payload.get("reason_codes") or []),
            preflight_path=preflight_path,
            checkpoint_path=None,
            eval_summary=None,
            checkpoint_gate_payload=None,
            run_manifest_gate_payload=None,
            threshold=None,
            attempt_budget_consumed=False,
            next_action=str(
                preflight_payload.get("next_action")
                or NEXT_ACTION_USER_ESCALATION_REQUIRED
            ),
            hard_block_subfamily=str(
                preflight_payload.get("hard_block_subfamily") or ""
            ).strip()
            or None,
            repair_tool_status=str(
                preflight_payload.get("repair_tool_status") or ""
            ).strip()
            or None,
        )
        final_manifest = _upgrade_manifest_payload(
            repo_root=resolved_repo_root,
            manifest_path=resolved_manifest_path,
            manifest_payload=manifest_payload,
            preflight_path=preflight_path,
            preflight_payload=preflight_payload,
            workflow_status=STATUS_EXECUTION_HARD_BLOCK,
            collect_policy_ckpt_decision=decision,
            collect_policy_ckpt_path=None,
            collect_policy_ckpt_provenance=None,
            attempt_record=None,
            hard_block_metadata=hard_block_metadata,
            eval_summary=None,
        )
        write_json(resolved_manifest_path, final_manifest)
        return {
            "manifest_path": str(resolved_manifest_path),
            "status": STATUS_EXECUTION_HARD_BLOCK,
            "exit_code": 1,
            "collect_policy_ckpt_decision": decision,
            "reason_codes": list(preflight_payload.get("reason_codes") or []),
            "hard_block_subfamily": preflight_payload.get("hard_block_subfamily"),
            "next_action": preflight_payload.get("next_action"),
            PREFLIGHT_PATH_FIELD: str(preflight_path),
        }

    if preflight_only:
        manifest_after_preflight = _upgrade_manifest_payload(
            repo_root=resolved_repo_root,
            manifest_path=resolved_manifest_path,
            manifest_payload=manifest_payload,
            preflight_path=preflight_path,
            preflight_payload=preflight_payload,
            workflow_status=STATUS_CONTINUE,
            collect_policy_ckpt_decision=decision,
            collect_policy_ckpt_path=manifest_payload.get("collect_policy_ckpt_path"),
            collect_policy_ckpt_provenance=(
                manifest_payload.get("collect_policy_ckpt_provenance")
                if isinstance(
                    manifest_payload.get("collect_policy_ckpt_provenance"), Mapping
                )
                else None
            ),
            attempt_record=None,
            hard_block_metadata=None,
            eval_summary=None,
        )
        write_json(resolved_manifest_path, manifest_after_preflight)
        return {
            "manifest_path": str(resolved_manifest_path),
            "status": "preflight_ready",
            "exit_code": 0,
            "collect_policy_ckpt_decision": decision,
            "reason_codes": list(preflight_payload.get("reason_codes") or []),
            PREFLIGHT_PATH_FIELD: str(preflight_path),
            "attempt_budget_consumed": False,
        }

    attempt_paths = _attempt_paths(
        resolved_repo_root,
        iteration_artifact_root=iteration_artifact_root,
        formal_output_dir=resolved_formal_output_dir,
    )
    prelaunch_summary: dict[str, Any] = {
        "launch_family": resolved_launch_family,
        "formal_output_dir": _repo_relative_path(
            resolved_repo_root, resolved_formal_output_dir
        ),
        "single_gpu_smoke_verdict_path": _repo_relative_path(
            resolved_repo_root, resolved_single_gpu_smoke_verdict_path
        ),
        "single_gpu_launch_authority_pass": False,
        "single_gpu_launch_authority_reason_codes": [],
    }
    prelaunch_reason_codes: list[str] = []
    if resolved_launch_family != LIVE_LAUNCH_FAMILY:
        prelaunch_reason_codes.append("launch_family_not_single_gpu_v1")
    else:
        try:
            smoke_verdict = _load_single_gpu_smoke_verdict(
                resolved_single_gpu_smoke_verdict_path
            )
        except FileNotFoundError:
            prelaunch_reason_codes.append("single_gpu_smoke_verdict_missing")
        except ValueError as exc:
            if HISTORICAL_DDP_SMOKE_VERDICT_SCHEMA_VERSION in str(exc):
                prelaunch_reason_codes.append(
                    "historical_task10_ddp_verdict_rejected_as_live_authority"
                )
            else:
                prelaunch_reason_codes.append(
                    "single_gpu_smoke_verdict_live_authority_rejected"
                )
        else:
            prelaunch_summary["single_gpu_smoke_verdict_schema_version"] = str(
                smoke_verdict.get("schema_version") or ""
            ).strip() or None
            prelaunch_summary["single_gpu_smoke_verdict_unlock_signal"] = str(
                smoke_verdict.get("unlock_signal") or ""
            ).strip() or None
            authority_ok, authority_failures = _validate_single_gpu_launch_authority(
                smoke_verdict
            )
            prelaunch_reason_codes.extend(authority_failures)
            prelaunch_summary["single_gpu_launch_authority_pass"] = bool(authority_ok)
    prelaunch_summary["single_gpu_launch_authority_reason_codes"] = list(
        dict.fromkeys(prelaunch_reason_codes)
    )
    if prelaunch_reason_codes:
        hard_block_metadata = _build_hard_block_metadata(
            repo_root=resolved_repo_root,
            manifest_path=resolved_manifest_path,
            manifest_payload=manifest_payload,
            prereq=_build_live_authority_placeholder_prereq(
                resolved_repo_root,
                single_gpu_smoke_verdict_path=resolved_single_gpu_smoke_verdict_path,
            ),
            attempt_paths=attempt_paths,
            finetune_returncode=1,
            eval_returncode=1,
            checkpoint_gate_returncode=1,
            run_manifest_gate_returncode=1,
            status_family=STATUS_INCONCLUSIVE_CONTRACT_MISMATCH,
            hard_blocker=HARD_BLOCKER_BASELINE_TRAIN_CONTRACT_MISMATCH,
            reason_codes=list(dict.fromkeys(prelaunch_reason_codes)),
            preflight_path=preflight_path,
            checkpoint_path=None,
            eval_summary=None,
            checkpoint_gate_payload=None,
            run_manifest_gate_payload=None,
            threshold=None,
            attempt_budget_consumed=False,
            prelaunch_summary=prelaunch_summary,
        )
        final_manifest = _upgrade_manifest_payload(
            repo_root=resolved_repo_root,
            manifest_path=resolved_manifest_path,
            manifest_payload=manifest_payload,
            preflight_path=preflight_path,
            preflight_payload=preflight_payload,
            workflow_status=STATUS_INCONCLUSIVE_CONTRACT_MISMATCH,
            collect_policy_ckpt_decision=decision,
            collect_policy_ckpt_path=manifest_payload.get("collect_policy_ckpt_path"),
            collect_policy_ckpt_provenance=(
                manifest_payload.get("collect_policy_ckpt_provenance")
                if isinstance(
                    manifest_payload.get("collect_policy_ckpt_provenance"), Mapping
                )
                else None
            ),
            attempt_record=None,
            hard_block_metadata=hard_block_metadata,
            eval_summary=None,
        )
        write_json(resolved_manifest_path, final_manifest)
        return {
            "manifest_path": str(resolved_manifest_path),
            "status": STATUS_INCONCLUSIVE_CONTRACT_MISMATCH,
            "exit_code": 1,
            "collect_policy_ckpt_decision": decision,
            "launch_family": resolved_launch_family,
            "single_gpu_smoke_verdict_path": str(
                resolved_single_gpu_smoke_verdict_path
            ),
            "single_gpu_launch_authority_pass": False,
            "reason_codes": list(dict.fromkeys(prelaunch_reason_codes)),
        }
    prereq = _load_prereq(
        resolved_repo_root,
        single_gpu_smoke_verdict_path=resolved_single_gpu_smoke_verdict_path,
    )
    checkpoint_gate_path = (
        iteration_artifact_root / CHECKPOINT_PROVENANCE_GATE_JSON_NAME
    )
    run_manifest_gate_path = iteration_artifact_root / RUN_MANIFEST_GATE_JSON_NAME
    threshold = float(
        manifest_payload.get("historical_success_rate_threshold")
        or HISTORICAL_SUCCESS_RATE_THRESHOLD
    )
    manifest_after_preflight = _upgrade_manifest_payload(
        repo_root=resolved_repo_root,
        manifest_path=resolved_manifest_path,
        manifest_payload=manifest_payload,
        preflight_path=preflight_path,
        preflight_payload=preflight_payload,
        workflow_status=STATUS_CONTINUE,
        collect_policy_ckpt_decision=decision,
        collect_policy_ckpt_path=manifest_payload.get("collect_policy_ckpt_path"),
        collect_policy_ckpt_provenance=(
            manifest_payload.get("collect_policy_ckpt_provenance")
            if isinstance(
                manifest_payload.get("collect_policy_ckpt_provenance"), Mapping
            )
            else None
        ),
        attempt_record=None,
        hard_block_metadata=None,
        eval_summary=None,
    )
    write_json(resolved_manifest_path, manifest_after_preflight)
    started_at = (
        str(prior_attempt.get("started_at") or "").strip()
        if repair_started_attempt and isinstance(prior_attempt, Mapping)
        else ""
    ) or _now_iso()
    manifest_with_attempt = _write_attempt_started_manifest(
        repo_root=resolved_repo_root,
        manifest_path=resolved_manifest_path,
        manifest_payload=manifest_after_preflight,
        prereq=prereq,
        attempt_paths=attempt_paths,
        started_at=started_at,
        prelaunch_summary=prelaunch_summary,
    )

    finetune_cmd = _build_finetune_cmd(
        repo_root=resolved_repo_root,
        manifest_payload=manifest_with_attempt,
        prereq=prereq,
        attempt_paths=attempt_paths,
        training_python_contract=training_python_contract,
    )
    finetune_returncode = _run_logged_command(
        cmd=finetune_cmd,
        cwd=resolved_repo_root,
        log_path=attempt_paths.train_log_path,
    )
    finetune_summary = (
        _read_json_object_if_exists(attempt_paths.finetune_summary_path) or {}
    )
    checkpoint_path_raw = str(
        finetune_summary.get("selected_checkpoint_path") or ""
    ).strip()
    checkpoint_path = (
        _resolve_path(resolved_repo_root, checkpoint_path_raw)
        if checkpoint_path_raw
        else None
    )
    config_ok, config_failures = _validate_finetune_summary_config(finetune_summary)
    checkpoint_count_ok, checkpoint_count = _checkpoint_count_ok(
        attempt_paths.finetune_output_dir
    )
    checkpoint_features = (
        _inspect_checkpoint_weight_map_features(
            repo_root=resolved_repo_root,
            manifest_payload={
                **manifest_with_attempt,
                "collect_policy_ckpt_decision": DECISION_BASELINE_TRAINED,
            },
            checkpoint_path=checkpoint_path,
            checkpoint_source_field="finetune_summary.selected_checkpoint_path",
        )
        if checkpoint_path is not None
        else {}
    )

    failure_reason_codes: list[str] = []
    failure_status = STATUS_CONTINUE
    failure_blocker = HARD_BLOCKER_BASELINE_TRAIN_CONTRACT_MISMATCH
    eval_returncode = 1
    checkpoint_gate_returncode = 1
    run_manifest_gate_returncode = 1
    eval_summary: dict[str, Any] = {}
    prelim_run_manifest: dict[str, Any] = {}
    checkpoint_gate_report: dict[str, Any] = {}
    run_manifest_gate_report: dict[str, Any] = {}
    checkpoint_gate_payload: dict[str, Any] = {}
    run_manifest_gate_payload: dict[str, Any] = {}
    final_decision = DECISION_BASELINE_TRAIN_REQUIRED

    if finetune_returncode != 0:
        failure_status = STATUS_EXECUTION_HARD_BLOCK
        failure_blocker = HARD_BLOCKER_BASELINE_TRAIN_EXECUTION_BLOCKED
        failure_reason_codes.append("baseline_train_returncode_nonzero")
    if not checkpoint_path_raw:
        failure_status = STATUS_EXECUTION_HARD_BLOCK
        failure_blocker = HARD_BLOCKER_BASELINE_TRAIN_EXECUTION_BLOCKED
        failure_reason_codes.append("selected_checkpoint_path_missing")
    if checkpoint_path is not None and not checkpoint_path.exists():
        failure_status = STATUS_EXECUTION_HARD_BLOCK
        failure_blocker = HARD_BLOCKER_BASELINE_TRAIN_EXECUTION_BLOCKED
        failure_reason_codes.append("selected_checkpoint_path_missing_on_disk")
    if not config_ok:
        failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
        failure_reason_codes.extend(config_failures)
    if not checkpoint_count_ok:
        failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
        failure_reason_codes.append("checkpoint_retention_not_single")
    if checkpoint_path is not None and not bool(
        checkpoint_features.get("has_advantage_embedding_pair")
    ) and not _baseline_default_adv_init_compatible(checkpoint_features):
        failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
        failure_reason_codes.append("checkpoint_missing_advantage_embedding_weights")

    if not failure_reason_codes and checkpoint_path is not None:
        eval_cmd = _build_eval_cmd(
            repo_root=resolved_repo_root,
            manifest_payload=manifest_with_attempt,
            checkpoint_path=checkpoint_path,
            checkpoint_features=checkpoint_features,
            attempt_paths=attempt_paths,
            training_python_contract=training_python_contract,
        )
        eval_returncode = _run_logged_command(
            cmd=eval_cmd,
            cwd=resolved_repo_root,
            log_path=attempt_paths.eval_log_path,
        )
        eval_summary = (
            _read_json_object_if_exists(attempt_paths.prelim_eval_summary_path) or {}
        )
        success_rate_raw = eval_summary.get("success_rate")
        if isinstance(success_rate_raw, bool) or not isinstance(
            success_rate_raw, (int, float)
        ):
            failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
            failure_reason_codes.append(
                "prelim_eval_summary_missing_numeric_success_rate"
            )
        else:
            _, prelim_run_manifest = _build_gate_inputs(
                repo_root=resolved_repo_root,
                manifest_payload=manifest_with_attempt,
                prereq=prereq,
                attempt_paths=attempt_paths,
                checkpoint_path=checkpoint_path,
            )

            checkpoint_gate_cmd = _build_checkpoint_gate_cmd(
                repo_root=resolved_repo_root,
                attempt_paths=attempt_paths,
                training_python_contract=training_python_contract,
            )
            checkpoint_gate_returncode = _run_logged_command(
                cmd=checkpoint_gate_cmd,
                cwd=resolved_repo_root,
                log_path=attempt_paths.checkpoint_gate_log_path,
            )
            checkpoint_gate_report = (
                _read_json_object_if_exists(
                    attempt_paths.checkpoint_gate_raw_report_path
                )
                or {}
            )

            run_manifest_gate_cmd = _build_run_manifest_gate_cmd(
                repo_root=resolved_repo_root,
                attempt_paths=attempt_paths,
                training_python_contract=training_python_contract,
            )
            run_manifest_gate_returncode = _run_logged_command(
                cmd=run_manifest_gate_cmd,
                cwd=resolved_repo_root,
                log_path=attempt_paths.run_manifest_gate_log_path,
            )
            run_manifest_gate_report = (
                _read_json_object_if_exists(
                    attempt_paths.run_manifest_gate_raw_report_path
                )
                or {}
            )

            candidate_eval = _candidate_eval_from_gate_reports(
                checkpoint_path=checkpoint_path,
                eval_summary=eval_summary,
                authority_refs=_authority_refs_from_manifest(manifest_with_attempt),
                final_decision=DECISION_BASELINE_TRAINED,
                checkpoint_report=checkpoint_gate_report,
                prelim_run_manifest=prelim_run_manifest,
                run_manifest_report=run_manifest_gate_report,
            )
            checkpoint_gate_payload = _build_checkpoint_provenance_gate_payload(
                repo_root=resolved_repo_root,
                gate_path=checkpoint_gate_path,
                candidate_eval=candidate_eval,
            )
            write_json(checkpoint_gate_path, checkpoint_gate_payload)
            run_manifest_gate_payload = _build_run_manifest_gate_payload(
                repo_root=resolved_repo_root,
                gate_path=run_manifest_gate_path,
                candidate_eval=candidate_eval,
            )
            write_json(run_manifest_gate_path, run_manifest_gate_payload)
            _mirror_formal_root_artifacts(
                formal_output_dir=attempt_paths.finetune_output_dir,
                checkpoint_path=checkpoint_path,
                checkpoint_gate_payload=checkpoint_gate_payload,
                run_manifest_gate_payload=run_manifest_gate_payload,
            )

            success_rate = float(success_rate_raw)
            eval_wrapper_ok = (
                str(eval_summary.get("wrapper_status") or "").strip() == "ok"
            )
            eval_episodes_ok = int(eval_summary.get("episodes") or 0) >= int(
                DEFAULT_EVAL_EPISODES
            )
            gates_pass = bool(
                checkpoint_gate_payload.get("pass")
                and run_manifest_gate_payload.get("pass")
            )
            if eval_returncode != 0:
                failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
                failure_reason_codes.append("prelim_eval_returncode_nonzero")
            if not eval_wrapper_ok:
                failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
                failure_reason_codes.append("prelim_eval_wrapper_blocked")
            if not eval_episodes_ok:
                failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
                failure_reason_codes.append(
                    "prelim_eval_completed_episodes_below_minimum"
                )
            if not bool(checkpoint_gate_payload.get("pass")):
                failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
                failure_reason_codes.append("checkpoint_provenance_gate_blocked")
            if not bool(run_manifest_gate_payload.get("pass")):
                failure_status = STATUS_INCONCLUSIVE_CONTRACT_MISMATCH
                failure_reason_codes.append("run_manifest_gate_blocked")
            if (
                gates_pass
                and eval_wrapper_ok
                and eval_episodes_ok
            ):
                final_decision = DECISION_BASELINE_TRAINED

    finished_attempt_record = _build_attempt_record(
        repo_root=resolved_repo_root,
        prereq=prereq,
        attempt_paths=attempt_paths,
        started_at=started_at,
        state=ATTEMPT_STATE_FINISHED,
        prelaunch_summary=prelaunch_summary,
    )
    finished_attempt_record.update(
        {
            "finished_at": _now_iso(),
            "final_decision": final_decision,
            "workflow_status": (
                STATUS_CONTINUE
                if final_decision == DECISION_BASELINE_TRAINED
                else failure_status
            ),
            "finetune_returncode": int(finetune_returncode),
            "prelim_eval_returncode": int(eval_returncode),
            "checkpoint_provenance_gate_returncode": int(checkpoint_gate_returncode),
            "run_manifest_gate_returncode": int(run_manifest_gate_returncode),
            "selected_checkpoint_path": None
            if checkpoint_path is None
            else _repo_relative_path(resolved_repo_root, checkpoint_path),
            "prelim_success_rate": float(eval_summary.get("success_rate") or 0.0),
            "prelim_success_count": int(eval_summary.get("success_count") or 0),
            "prelim_episodes": int(eval_summary.get("episodes") or 0),
            "prelim_requested_episodes": int(
                eval_summary.get("requested_episodes") or DEFAULT_EVAL_EPISODES
            ),
            "prelim_eval_wrapper_status": str(
                eval_summary.get("wrapper_status") or ""
            ).strip()
            or None,
            "checkpoint_provenance_gate_pass": bool(
                checkpoint_gate_payload.get("pass")
            ),
            "run_manifest_gate_pass": bool(run_manifest_gate_payload.get("pass")),
            "checkpoint_weight_map_features": dict(checkpoint_features),
            "meaningful_config_pass": bool(config_ok),
            "meaningful_config_reason_codes": list(config_failures),
            "retained_checkpoint_count": int(checkpoint_count),
            "attempt_budget_consumed": True,
        }
    )

    hard_block_metadata = None
    workflow_status = STATUS_CONTINUE
    if final_decision != DECISION_BASELINE_TRAINED:
        workflow_status = failure_status
        hard_block_metadata = _build_hard_block_metadata(
            repo_root=resolved_repo_root,
            manifest_path=resolved_manifest_path,
            manifest_payload=manifest_with_attempt,
            prereq=prereq,
            attempt_paths=attempt_paths,
            finetune_returncode=finetune_returncode,
            eval_returncode=eval_returncode,
            checkpoint_gate_returncode=checkpoint_gate_returncode,
            run_manifest_gate_returncode=run_manifest_gate_returncode,
            status_family=failure_status,
            hard_blocker=(
                failure_blocker
                if failure_status == STATUS_EXECUTION_HARD_BLOCK
                else HARD_BLOCKER_BASELINE_TRAIN_CONTRACT_MISMATCH
            ),
            reason_codes=list(dict.fromkeys(failure_reason_codes)),
            preflight_path=preflight_path,
            checkpoint_path=checkpoint_path,
            eval_summary=eval_summary,
            checkpoint_gate_payload=checkpoint_gate_payload,
            run_manifest_gate_payload=run_manifest_gate_payload,
            threshold=threshold,
            attempt_budget_consumed=True,
            prelaunch_summary=prelaunch_summary,
        )

    collect_policy_ckpt_provenance = None
    if (
        checkpoint_path is not None
        and checkpoint_gate_payload
        and run_manifest_gate_payload
    ):
        collect_policy_ckpt_provenance = _build_collect_policy_ckpt_provenance(
            repo_root=resolved_repo_root,
            manifest_payload=manifest_with_attempt,
            prereq=prereq,
            attempt_paths=attempt_paths,
            checkpoint_path=checkpoint_path,
            eval_summary=eval_summary,
            checkpoint_gate_payload=checkpoint_gate_payload,
            run_manifest_gate_payload=run_manifest_gate_payload,
            final_decision=final_decision,
            threshold=threshold,
        )

    final_manifest = _upgrade_manifest_payload(
        repo_root=resolved_repo_root,
        manifest_path=resolved_manifest_path,
        manifest_payload=manifest_with_attempt,
        preflight_path=preflight_path,
        preflight_payload=preflight_payload,
        workflow_status=workflow_status,
        collect_policy_ckpt_decision=final_decision,
        collect_policy_ckpt_path=(
            str(checkpoint_path)
            if final_decision == DECISION_BASELINE_TRAINED
            and checkpoint_path is not None
            else None
        ),
        collect_policy_ckpt_provenance=collect_policy_ckpt_provenance,
        attempt_record=finished_attempt_record,
        hard_block_metadata=hard_block_metadata,
        eval_summary=eval_summary if eval_summary else None,
    )
    write_json(resolved_manifest_path, final_manifest)

    return {
        "manifest_path": str(resolved_manifest_path),
        "status": "ok"
        if final_decision == DECISION_BASELINE_TRAINED
        else workflow_status,
        "exit_code": 0 if final_decision == DECISION_BASELINE_TRAINED else 1,
        "collect_policy_ckpt_decision": final_decision,
        "collect_policy_ckpt_path": final_manifest.get("collect_policy_ckpt_path"),
        "collect_policy_ckpt_prelim_success_rate": float(
            final_manifest.get("collect_policy_ckpt_prelim_success_rate") or 0.0
        ),
        "collect_policy_ckpt_prelim_success_count": int(
            final_manifest.get("collect_policy_ckpt_prelim_success_count") or 0
        ),
        "threshold": threshold,
        "finetune_returncode": int(finetune_returncode),
        "prelim_eval_returncode": int(eval_returncode),
        "checkpoint_provenance_gate_pass": bool(checkpoint_gate_payload.get("pass")),
        "run_manifest_gate_pass": bool(run_manifest_gate_payload.get("pass")),
        PREFLIGHT_PATH_FIELD: str(preflight_path),
        "attempt_dir": str(attempt_paths.attempt_dir),
        "finetune_summary_path": str(attempt_paths.finetune_summary_path),
        "prelim_eval_summary_path": str(attempt_paths.prelim_eval_summary_path),
        "checkpoint_provenance_gate_path": str(checkpoint_gate_path),
        "run_manifest_gate_path": str(run_manifest_gate_path),
        "official_task_anchor_model": OFFICIAL_TASK_ANCHOR_MODEL,
        "reason_codes": list(dict.fromkeys(failure_reason_codes)),
    }


__all__ = [
    "DECISION_BASELINE_TRAINED",
    "DECISION_BASELINE_TRAIN_REQUIRED",
    "DECISION_ITERATION_HARD_BLOCK",
    "run_stage3_baseline_train_gate",
]
