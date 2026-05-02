from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from work.openpi.pipelines.recap.blind_calibration_runtime import (
    atomic_json_write,
    read_json_object,
    repo_rel,
    sha256_file,
    utc_now,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "stage1_v22_full_training_eval_iter9_20260426T_nextZ"
ITER8_RUN_ID = "stage1_v22_blind_calibration_iter8_20260426T_nextZ"
HASH_LOCK_SCHEMA_VERSION = "v22_preregistration_hash_lock_v1"
TRAIN_MANIFEST_SCHEMA_VERSION = "v22_variant_train_manifest_v1"
TRAIN_PRECONDITION_SCHEMA_VERSION = "v22_variant_train_precondition_v1"
CHECKPOINT_PROVENANCE_SCHEMA_VERSION = "v22_variant_checkpoint_provenance_v1"
VARIANT_AUTHORITY_SCHEMA_VERSION = "variant_authority_manifest_v22_v1"
ITER5_R2_SHA256 = "0f4a1b74152a8e4f88c7259b0033e0f262257a1222e1a6fb5436f084547e7e69"
ITER5_R4_SHA256 = "c08da923e96c6d2d6f1f6b2522219eee7f3ab5b6851f880e7502b3fadd7af965"
PAIRING_TASK_KEY = "all_tasks_round_robin_episode_index_modulo_10"
FORMAL_VARIANTS = ("A", "B", "C", "X")
ALLOWED_CUDA_VISIBLE_DEVICES_BY_VARIANT = {
    # v2.5 policy expansion (2026-04-27): GPU3 also allowed (only GPU0 forbidden by iter9 hook)
    "B": frozenset({"1", "2", "3"}),
    "C": frozenset({"1", "2", "3"}),
    "X": frozenset({"1", "2", "3"}),
}


@dataclass(frozen=True)
class LockedProtocol:
    suite: str
    budget: float
    cell_id: str
    step_cap: int
    max_steps_full: int
    tasks: tuple[str, ...]
    n_per_variant: int
    variants: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "budget": self.budget,
            "cell_id": self.cell_id,
            "step_cap": self.step_cap,
            "max_steps_full": self.max_steps_full,
            "tasks": list(self.tasks),
            "n_per_variant": self.n_per_variant,
            "variants": list(self.variants),
        }


@dataclass(frozen=True)
class V22HashLock:
    path: Path
    sha256: str
    schema_version: str
    run_id: str
    selected_protocol: LockedProtocol
    selected_using_c_results: bool
    selected_using_x_results: bool
    iter5_r2_sha256: str
    iter5_r4_sha256: str
    raw: Mapping[str, object]


@dataclass(frozen=True)
class VariantTrainingSpec:
    variant: str
    variant_id: str
    role: str
    requires_alpha_trace: bool
    requires_shuffle_manifest: bool
    requires_control_absence_attestation: bool


@dataclass(frozen=True)
class TrainingRequest:
    variant: str
    variant_id: str
    prereg_hash_lock: Path
    output_dir: Path
    runtime_log_dir: Path
    warm_start_checkpoint: Path
    enable_r2_phase_threshold_switching: bool
    enable_r4_alpha_dual_loss: bool
    emit_loss_decomposition: bool
    emit_threshold_trace: bool
    emit_alpha_dual_trace: bool
    emit_gradient_attestation: bool
    emit_shuffle_manifest: bool
    emit_deterministic_shuffle_provenance: bool
    emit_control_signal_absence_attestation: bool
    emit_sha256sums: bool
    no_sudo: bool
    cuda_visible_devices: str


@dataclass(frozen=True)
class VariantAuthorityEvaluation:
    formal_eval_allowed: bool
    reasons: tuple[str, ...]


VARIANT_SPECS: dict[str, VariantTrainingSpec] = {
    "B": VariantTrainingSpec(
        variant="B",
        variant_id="control_no_recap_shuffled_adversarial_relabel",
        role="B_control",
        requires_alpha_trace=False,
        requires_shuffle_manifest=False,
        requires_control_absence_attestation=True,
    ),
    "C": VariantTrainingSpec(
        variant="C",
        variant_id="main_recap_method",
        role="C_recap",
        requires_alpha_trace=True,
        requires_shuffle_manifest=False,
        requires_control_absence_attestation=False,
    ),
    "X": VariantTrainingSpec(
        variant="X",
        variant_id="recap_variant_shuffle_diag",
        role="X_shuffle_diag",
        requires_alpha_trace=True,
        requires_shuffle_manifest=True,
        requires_control_absence_attestation=False,
    ),
}
ALLOWED_VARIANT_IDS = {
    "A_stock_pi0_libero",
    *(spec.variant_id for spec in VARIANT_SPECS.values()),
}


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value))


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _has_legacy_media_root(path: Path | str) -> bool:
    text = str(path)
    return text.startswith("/media/") or "/media/howard/Data/" in text


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            rel = child.relative_to(path).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(f"missing path for sha256: {path}")


def load_v22_hash_lock(path: str | Path) -> V22HashLock:
    resolved = resolve_repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"BLOCK_HASH_LOCK_MISSING:{resolved}")
    payload = read_json_object(resolved)
    selected = _mapping(payload.get("selected_protocol"), context="selected_protocol")
    protocol = LockedProtocol(
        suite=str(selected.get("suite") or ""),
        budget=float(selected.get("budget") or 0.0),
        cell_id=str(selected.get("cell_id") or ""),
        step_cap=int(selected.get("step_cap") or 0),
        max_steps_full=int(selected.get("max_steps_full") or 0),
        tasks=_string_tuple(selected.get("tasks")),
        n_per_variant=int(payload.get("n_per_variant") or 0),
        variants=_string_tuple(payload.get("variants")),
    )
    return V22HashLock(
        path=resolved,
        sha256=sha256_file(resolved),
        schema_version=str(payload.get("schema_version") or ""),
        run_id=str(payload.get("run_id") or ""),
        selected_protocol=protocol,
        selected_using_c_results=bool(payload.get("selected_using_c_results")),
        selected_using_x_results=bool(payload.get("selected_using_x_results")),
        iter5_r2_sha256=str(payload.get("iter5_r2_sha256") or ""),
        iter5_r4_sha256=str(payload.get("iter5_r4_sha256") or ""),
        raw=payload,
    )


def validate_hash_lock(lock: V22HashLock) -> tuple[str, ...]:
    reasons: list[str] = []
    protocol = lock.selected_protocol
    if lock.schema_version != HASH_LOCK_SCHEMA_VERSION:
        reasons.append("BLOCK_HASH_LOCK_SCHEMA")
    if lock.run_id != ITER8_RUN_ID:
        reasons.append("BLOCK_HASH_LOCK_RUN_ID")
    if lock.selected_using_c_results or lock.selected_using_x_results:
        reasons.append("BLOCK_C_X_SELECTION_LEAKAGE")
    if protocol.suite != "libero_spatial":
        reasons.append("BLOCK_HASH_LOCK_SUITE")
    if protocol.budget != 0.5:
        reasons.append("BLOCK_HASH_LOCK_BUDGET")
    if protocol.step_cap != 110 or protocol.max_steps_full != 220:
        reasons.append("BLOCK_HASH_LOCK_STEP_BUDGET")
    if protocol.n_per_variant != 192:
        reasons.append("BLOCK_HASH_LOCK_N_PER_VARIANT")
    if protocol.variants != FORMAL_VARIANTS:
        reasons.append("BLOCK_HASH_LOCK_VARIANTS")
    if protocol.tasks != (PAIRING_TASK_KEY,):
        reasons.append("BLOCK_HASH_LOCK_TASKS")
    if lock.iter5_r2_sha256 != ITER5_R2_SHA256:
        reasons.append("BLOCK_R2_CLOSURE_SHA")
    if lock.iter5_r4_sha256 != ITER5_R4_SHA256:
        reasons.append("BLOCK_R4_CLOSURE_SHA")
    return tuple(reasons)


def validate_warm_start_checkpoint(path: Path) -> tuple[str, ...]:
    reasons: list[str] = []
    if _has_legacy_media_root(path):
        reasons.append("BLOCK_WARM_START_LEGACY_MEDIA_ROOT")
    text = str(path)
    if "2026_04_02_03" in text or "iter3" in text:
        reasons.append("BLOCK_WARM_START_REJECTED_LINEAGE")
    if not path.exists():
        reasons.append("BLOCK_WARM_START_MISSING")
    return tuple(reasons)


def validate_training_request(
    request: TrainingRequest,
    lock: V22HashLock,
) -> tuple[str, ...]:
    reasons: list[str] = list(validate_hash_lock(lock))
    spec = VARIANT_SPECS.get(request.variant)
    if spec is None:
        reasons.append("BLOCK_VARIANT_UNSUPPORTED")
    elif request.variant_id != spec.variant_id:
        reasons.append("BLOCK_VARIANT_ID_MISMATCH")
    if not request.no_sudo:
        reasons.append("BLOCK_NO_SUDO_REQUIRED")
    allowed_devices = ALLOWED_CUDA_VISIBLE_DEVICES_BY_VARIANT.get(
        request.variant,
        frozenset({"1"}),
    )
    if request.cuda_visible_devices not in allowed_devices:
        reasons.append("BLOCK_CUDA_VISIBLE_DEVICES_BOUNDARY")
    reasons.extend(validate_warm_start_checkpoint(request.warm_start_checkpoint))
    if spec is not None:
        if spec.requires_alpha_trace and not request.emit_alpha_dual_trace:
            reasons.append("BLOCK_ALPHA_TRACE_REQUIRED")
        if spec.requires_shuffle_manifest and not request.emit_shuffle_manifest:
            reasons.append("BLOCK_SHUFFLE_MANIFEST_REQUIRED")
        if spec.requires_control_absence_attestation and (
            not request.emit_control_signal_absence_attestation
        ):
            reasons.append("BLOCK_CONTROL_SIGNAL_ABSENCE_ATTESTATION_REQUIRED")
    return tuple(dict.fromkeys(reasons))


def build_precondition_check(
    request: TrainingRequest,
    lock: V22HashLock,
) -> dict[str, object]:
    reasons = validate_training_request(request, lock)
    return {
        "schema_version": TRAIN_PRECONDITION_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "status": "PASS" if not reasons else "BLOCK",
        "blocking_reasons": list(reasons),
        "variant": request.variant,
        "variant_id": request.variant_id,
        "hash_lock_present": request.prereg_hash_lock.is_file(),
        "hash_lock_sha256": lock.sha256,
        "hash_lock_contract_ok": not validate_hash_lock(lock),
        "warm_start_checkpoint": repo_rel(REPO_ROOT, request.warm_start_checkpoint),
        "warm_start_no_legacy_media": not _has_legacy_media_root(
            request.warm_start_checkpoint
        ),
        "cuda_visible_devices": request.cuda_visible_devices,
        "no_sudo": request.no_sudo,
        "r2_closure_sha256": lock.iter5_r2_sha256,
        "r4_closure_sha256": lock.iter5_r4_sha256,
    }


def build_training_run_manifest(
    request: TrainingRequest,
    lock: V22HashLock,
    precondition: Mapping[str, object],
) -> dict[str, object]:
    spec = VARIANT_SPECS[request.variant]
    return {
        "schema_version": TRAIN_MANIFEST_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "variant": request.variant,
        "variant_id": request.variant_id,
        "role": spec.role,
        "status": "PASS" if precondition.get("status") == "PASS" else "BLOCK",
        "hash_lock_path": repo_rel(REPO_ROOT, lock.path),
        "hash_lock_sha256": lock.sha256,
        "locked_protocol": lock.selected_protocol.as_json(),
        "warm_start_checkpoint": repo_rel(REPO_ROOT, request.warm_start_checkpoint),
        "provenance_no_legacy_media": not _has_legacy_media_root(
            request.warm_start_checkpoint
        ),
        "trained_after_r2_r4_closed": True,
        "iter5_r2_sha256": lock.iter5_r2_sha256,
        "iter5_r4_sha256": lock.iter5_r4_sha256,
        "emit_flags": {
            "loss_decomposition": request.emit_loss_decomposition,
            "threshold_trace": request.emit_threshold_trace,
            "alpha_dual_trace": request.emit_alpha_dual_trace,
            "gradient_attestation": request.emit_gradient_attestation,
            "shuffle_manifest": request.emit_shuffle_manifest,
            "deterministic_shuffle_provenance": (
                request.emit_deterministic_shuffle_provenance
            ),
            "control_signal_absence_attestation": (
                request.emit_control_signal_absence_attestation
            ),
            "sha256sums": request.emit_sha256sums,
        },
        "r2_phase_threshold_switching_enabled": (
            request.enable_r2_phase_threshold_switching
        ),
        "r4_alpha_dual_loss_enabled": request.enable_r4_alpha_dual_loss,
        "cuda_visible_devices": request.cuda_visible_devices,
        "no_sudo": request.no_sudo,
    }


def build_checkpoint_provenance(
    request: TrainingRequest,
    lock: V22HashLock,
    checkpoint_path: Path,
) -> dict[str, object]:
    spec = VARIANT_SPECS[request.variant]
    return {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "variant": request.variant,
        "variant_id": request.variant_id,
        "role": spec.role,
        "checkpoint_path": repo_rel(REPO_ROOT, checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "hash_lock_sha256": lock.sha256,
        "source_warm_start_checkpoint": repo_rel(REPO_ROOT, request.warm_start_checkpoint),
        "trained_after_r2_r4_closed": True,
        "no_legacy_media_root": not _has_legacy_media_root(request.warm_start_checkpoint),
    }


def _artifact_path(payload: Mapping[str, object], key: str) -> Path:
    raw = str(payload.get(key) or "")
    return resolve_repo_path(raw)


def evaluate_variant_authority_manifest(
    payload: Mapping[str, object],
) -> VariantAuthorityEvaluation:
    reasons: list[str] = []
    if payload.get("schema_version") != VARIANT_AUTHORITY_SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    hash_lock_path = _artifact_path(payload, "hash_lock_path")
    if not hash_lock_path.is_file():
        reasons.append("hash_lock_path_missing")
        lock: V22HashLock | None = None
    else:
        lock = load_v22_hash_lock(hash_lock_path)
        if payload.get("hash_lock_sha256") != lock.sha256:
            reasons.append("hash_lock_sha256_mismatch")
        if payload.get("locked_protocol") != lock.selected_protocol.as_json():
            reasons.append("locked_protocol_mismatch")
    if not _artifact_path(payload, "no_c_x_leakage_attestation_path").is_file():
        reasons.append("no_c_x_leakage_attestation_missing")

    rows = payload.get("variants")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        reasons.append("variants_missing")
        rows = ()
    seen_ids: set[str] = set()
    for item in rows:
        if not isinstance(item, Mapping):
            reasons.append("variant_row_not_object")
            continue
        variant_id = str(item.get("variant_id") or "")
        role = str(item.get("role") or "")
        seen_ids.add(variant_id)
        if variant_id not in ALLOWED_VARIANT_IDS:
            reasons.append(f"variant_id_unexpected:{variant_id}")
        checkpoint_path = _artifact_path(item, "checkpoint_path")
        train_manifest_path = _artifact_path(item, "train_manifest_path")
        if not checkpoint_path.exists():
            reasons.append(f"checkpoint_path_missing:{variant_id}")
        elif item.get("checkpoint_sha256") != sha256_path(checkpoint_path):
            reasons.append(f"checkpoint_sha256_mismatch:{variant_id}")
        if not train_manifest_path.is_file():
            reasons.append(f"train_manifest_path_missing:{variant_id}")
        elif item.get("train_manifest_sha256") != sha256_file(train_manifest_path):
            reasons.append(f"train_manifest_sha256_mismatch:{variant_id}")
        if item.get("no_legacy_media_root") is not True:
            reasons.append(f"legacy_media_root:{variant_id}")
        if item.get("sha256sums_present") is not True:
            reasons.append(f"sha256sums_missing:{variant_id}")
        trained_after = item.get("trained_after_r2_r4_closed")
        if variant_id == "A_stock_pi0_libero":
            if trained_after != "not_applicable":
                reasons.append("a_stock_trained_after_sentinel_mismatch")
        elif trained_after is not True:
            reasons.append(f"trained_after_r2_r4_closed_missing:{variant_id}")
        if role == "C_recap" and not (
            item.get("loss_decomposition_present")
            and item.get("threshold_switch_trace_present")
            and item.get("alpha_dual_loss_trace_present")
        ):
            reasons.append("c_trace_predicates_missing")
        if role == "X_shuffle_diag" and not (
            item.get("loss_decomposition_present")
            and item.get("threshold_switch_trace_present")
            and item.get("alpha_dual_loss_trace_present")
            and item.get("shuffle_manifest_present")
        ):
            reasons.append("x_trace_predicates_missing")
        if role == "B_control" and not (
            item.get("loss_decomposition_present")
            and item.get("threshold_switch_trace_present")
            and item.get("control_signal_absence_attestation_present")
        ):
            reasons.append("b_trace_predicates_missing")
    missing_ids = ALLOWED_VARIANT_IDS - seen_ids
    reasons.extend(f"variant_missing:{variant_id}" for variant_id in sorted(missing_ids))
    return VariantAuthorityEvaluation(
        formal_eval_allowed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def build_variant_authority_manifest(
    *,
    hash_lock: V22HashLock,
    no_c_x_leakage_attestation_path: Path,
    variants: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": VARIANT_AUTHORITY_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "hash_lock_path": repo_rel(REPO_ROOT, hash_lock.path),
        "hash_lock_sha256": hash_lock.sha256,
        "locked_protocol": hash_lock.selected_protocol.as_json(),
        "iter5_r2_sha256": hash_lock.iter5_r2_sha256,
        "iter5_r4_sha256": hash_lock.iter5_r4_sha256,
        "no_c_x_leakage_attestation_path": repo_rel(
            REPO_ROOT,
            no_c_x_leakage_attestation_path,
        ),
        "variants": [dict(row) for row in variants],
        "formal_eval_allowed": False,
        "formal_eval_allowed_rationale": [],
        "sha256self": "0" * 64,
    }
    evaluation = evaluate_variant_authority_manifest(payload)
    payload["formal_eval_allowed"] = evaluation.formal_eval_allowed
    payload["formal_eval_allowed_rationale"] = (
        [
            "all per-variant predicates PASS",
            "hash_lock_sha256 matches iter8 carryforward",
            "locked_protocol unchanged",
            "no C/X selection leakage attestation present",
        ]
        if evaluation.formal_eval_allowed
        else list(evaluation.reasons)
    )
    payload["sha256self"] = compute_manifest_self_hash(payload)
    return payload


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def compute_manifest_self_hash(payload: Mapping[str, object]) -> str:
    canonical = dict(payload)
    canonical["sha256self"] = "0" * 64
    return hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()


def verify_manifest_self_hash(payload: Mapping[str, object]) -> bool:
    expected = str(payload.get("sha256self") or "")
    return expected == compute_manifest_self_hash(payload)


def write_variant_authority_manifest(path: Path, payload: Mapping[str, object]) -> None:
    materialized = dict(payload)
    materialized["sha256self"] = compute_manifest_self_hash(materialized)
    atomic_json_write(path, materialized)


__all__ = [
    "ALLOWED_VARIANT_IDS",
    "CHECKPOINT_PROVENANCE_SCHEMA_VERSION",
    "FORMAL_VARIANTS",
    "HASH_LOCK_SCHEMA_VERSION",
    "ITER5_R2_SHA256",
    "ITER5_R4_SHA256",
    "PAIRING_TASK_KEY",
    "REPO_ROOT",
    "RUN_ID",
    "TRAIN_MANIFEST_SCHEMA_VERSION",
    "TRAIN_PRECONDITION_SCHEMA_VERSION",
    "VARIANT_AUTHORITY_SCHEMA_VERSION",
    "VARIANT_SPECS",
    "LockedProtocol",
    "TrainingRequest",
    "V22HashLock",
    "VariantAuthorityEvaluation",
    "VariantTrainingSpec",
    "build_checkpoint_provenance",
    "build_precondition_check",
    "build_training_run_manifest",
    "build_variant_authority_manifest",
    "compute_manifest_self_hash",
    "evaluate_variant_authority_manifest",
    "load_v22_hash_lock",
    "resolve_repo_path",
    "sha256_path",
    "validate_hash_lock",
    "validate_training_request",
    "validate_warm_start_checkpoint",
    "verify_manifest_self_hash",
    "write_variant_authority_manifest",
]
