from __future__ import annotations

import argparse
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from types import FrameType
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.eval.v22_formal_eval_contracts import (  # noqa: E402
    load_prereg_hash_lock,
    validate_hash_lock as validate_formal_hash_lock,
)
from work.openpi.pipelines.recap.blind_calibration_runtime import (  # noqa: E402
    Sha256Sums,
    atomic_json_write,
    atomic_write_text,
    read_json_object,
    repo_rel,
    sha256_file,
    utc_now,
)
from work.openpi.pipelines.recap.v22_training_contracts import (  # noqa: E402
    RUN_ID,
    TrainingRequest,
    build_checkpoint_provenance,
    build_precondition_check,
    build_training_run_manifest,
    load_v22_hash_lock,
    resolve_repo_path,
    sha256_path,
)
from work.openpi.prompting.routes import (  # noqa: E402
    FIXEDADV_CONSTANT_CONSUMER_MODE,
    RECAP_RELABEL_CONSUMER_MODE,
    SHUFFLED_ADV_DIAG_CONSUMER_MODE,
)
from work.openpi.recap.real_variant_export import (  # noqa: E402
    ALPHA_TRACE_REAL_SCHEMA_VERSION,
    LOSS_DECOMPOSITION_REAL_SCHEMA_VERSION,
    THRESHOLD_TRACE_REAL_SCHEMA_VERSION,
    RealVariantExportBlockedError,
    RealVariantExportBundle,
    RealVariantExportRequest,
    run_real_variant_training_export,
)


HOOK_FLAG_PATH = REPO_ROOT / ".omc" / "iter9_hooks_active.flag"
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "lerobot_datasets"
    / "physical_intelligence_libero_official_8d_recap_relabels_v1"
)
REAL_GRADIENT_ATTESTATION_SCHEMA_VERSION = "v22_variant_gradient_attestation_real_v1"
CONTROL_ABSENCE_SCHEMA_VERSION = "v22_variant_control_signal_absence_v1"
SHUFFLE_MANIFEST_SCHEMA_VERSION = "v22_variant_shuffle_manifest_v1"
DETERMINISTIC_SHUFFLE_SCHEMA_VERSION = "v22_variant_deterministic_shuffle_v1"
ALPHA_SUMMARY_SCHEMA_VERSION = "v22_variant_alpha_term_contribution_summary_v1"
TRAIN_CONFIG_NAME = "pi0_libero"
LOG_INTERVAL_REAL_OVERRIDE = 1
DEFAULT_REAL_SAVE_INTERVAL_OVERRIDE = 200
DEFAULT_FSDP_DEVICES = 1
DEFAULT_XLA_MEM_FRACTION = "0.85"
GRACEFUL_TIMEOUT_RETURN_CODE = 124
RESUME_REAL_EXPORT_ENV = "ITER9_RESUME_REAL_EXPORT"


@dataclass(frozen=True)
class CanonicalAnchor:
    path: Path
    sha256: str
    source_config_name: str
    num_train_steps: int
    batch_size: int
    seed: int
    save_interval: int
    num_workers: int
    keep_period: int
    log_interval_canonical: int
    log_interval_real_override: int


@dataclass(frozen=True)
class ResourceResolution:
    num_train_steps: int
    num_train_steps_source: str
    batch_size: int
    batch_size_source: str
    save_interval: int
    save_interval_source: str
    num_workers: int
    seed: int
    phase_threshold_step: int


@dataclass(frozen=True)
class TimeoutCheckpointEvidence:
    checkpoint_run_dir: Path
    checkpoint_run_dir_exists: bool
    last_step: int | None
    last_checkpoint: Path | None
    last_checkpoint_tree_sha256: str | None


@dataclass(frozen=True)
class RealTrainingRequest:
    training: TrainingRequest
    canonical_anchor: Path
    dataset_dir: Path
    preregistration_skeleton: bool
    num_train_steps_override: int | None
    batch_size_override: int | None
    save_interval_override: int | None
    resume_real_export: bool


class GracefulTimeoutRequested(Exception):
    """Raised when the real-training wrapper receives SIGTERM from timeout."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v22_train_variants_real.py",
        description="Iter9 OpenPI v22 real-training wrapper.",
    )
    _ = parser.add_argument("--variant", choices=("B", "C", "X"), required=True)
    _ = parser.add_argument("--variant-id", required=True)
    _ = parser.add_argument("--prereg-hash-lock", required=True)
    _ = parser.add_argument("--canonical-anchor", required=True)
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument("--runtime-log-dir", required=True)
    _ = parser.add_argument("--warm-start-checkpoint", required=True)
    _ = parser.add_argument("--num-train-steps", type=int, default=None)
    _ = parser.add_argument("--batch-size", type=int, default=None)
    _ = parser.add_argument(
        "--save-interval-override",
        type=int,
        default=DEFAULT_REAL_SAVE_INTERVAL_OVERRIDE,
        help=(
            "Runtime-only save_interval for the real-training wrapper; "
            "canonical anchor remains unchanged."
        ),
    )
    _ = parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    _ = parser.add_argument("--enable-r2-phase-threshold-switching", action="store_true")
    _ = parser.add_argument("--enable-r4-alpha-dual-loss", action="store_true")
    _ = parser.add_argument("--emit-loss-decomposition", action="store_true")
    _ = parser.add_argument("--emit-threshold-trace", action="store_true")
    _ = parser.add_argument("--emit-alpha-dual-trace", action="store_true")
    _ = parser.add_argument("--emit-gradient-attestation", action="store_true")
    _ = parser.add_argument("--emit-shuffle-manifest", action="store_true")
    _ = parser.add_argument("--emit-deterministic-shuffle-provenance", action="store_true")
    _ = parser.add_argument("--emit-control-signal-absence-attestation", action="store_true")
    _ = parser.add_argument("--emit-sha256sums", action="store_true")
    _ = parser.add_argument("--no-sudo", action="store_true")
    _ = parser.add_argument("--cuda-visible-devices", default=None)
    _ = parser.add_argument("--preregistration-skeleton", action="store_true")
    _ = parser.add_argument(
        "--resume-real-export",
        action="store_true",
        help=(
            "Runtime-only resume flag for the real export/training subprocess. "
            "Canonical anchor remains unchanged."
        ),
    )
    return parser


def _env_flag(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def request_from_args(args: argparse.Namespace) -> RealTrainingRequest:
    cuda_visible_devices = cast(str | None, args.cuda_visible_devices)
    if cuda_visible_devices is None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    training = TrainingRequest(
        variant=cast(str, args.variant),
        variant_id=cast(str, args.variant_id),
        prereg_hash_lock=resolve_repo_path(cast(str, args.prereg_hash_lock)),
        output_dir=resolve_repo_path(cast(str, args.output_dir)),
        runtime_log_dir=resolve_repo_path(cast(str, args.runtime_log_dir)),
        warm_start_checkpoint=resolve_repo_path(cast(str, args.warm_start_checkpoint)),
        enable_r2_phase_threshold_switching=cast(
            bool, args.enable_r2_phase_threshold_switching
        ),
        enable_r4_alpha_dual_loss=cast(bool, args.enable_r4_alpha_dual_loss),
        emit_loss_decomposition=cast(bool, args.emit_loss_decomposition),
        emit_threshold_trace=cast(bool, args.emit_threshold_trace),
        emit_alpha_dual_trace=cast(bool, args.emit_alpha_dual_trace),
        emit_gradient_attestation=cast(bool, args.emit_gradient_attestation),
        emit_shuffle_manifest=cast(bool, args.emit_shuffle_manifest),
        emit_deterministic_shuffle_provenance=cast(
            bool, args.emit_deterministic_shuffle_provenance
        ),
        emit_control_signal_absence_attestation=cast(
            bool, args.emit_control_signal_absence_attestation
        ),
        emit_sha256sums=cast(bool, args.emit_sha256sums),
        no_sudo=cast(bool, args.no_sudo),
        cuda_visible_devices=str(cuda_visible_devices),
    )
    return RealTrainingRequest(
        training=training,
        canonical_anchor=resolve_repo_path(cast(str, args.canonical_anchor)),
        dataset_dir=resolve_repo_path(cast(str, args.dataset_dir)),
        preregistration_skeleton=cast(bool, args.preregistration_skeleton),
        num_train_steps_override=cast(int | None, args.num_train_steps),
        batch_size_override=cast(int | None, args.batch_size),
        save_interval_override=cast(int | None, args.save_interval_override),
        resume_real_export=bool(args.resume_real_export) or _env_flag(RESUME_REAL_EXPORT_ENV),
    )


def _run_preregistration_skeleton(argv: list[str]) -> int:
    from work.openpi.pipelines.recap import v22_train_variants

    skeleton_argv: list[str] = []
    skip_next = False
    drop_value_flags = {
        "--canonical-anchor",
        "--num-train-steps",
        "--batch-size",
        "--save-interval-override",
        "--dataset-dir",
    }
    drop_bool_flags = {"--preregistration-skeleton"}
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item in drop_value_flags:
            skip_next = index + 1 < len(argv)
            continue
        if item in drop_bool_flags:
            continue
        skeleton_argv.append(item)
    return v22_train_variants.main(skeleton_argv)


def _json_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"expected JSON integer-compatible value, got {type(value).__name__}")


def _json_float(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"expected JSON float-compatible value, got {type(value).__name__}")


def _load_canonical_anchor(path: Path) -> CanonicalAnchor:
    if not path.is_file():
        raise FileNotFoundError(f"BLOCK_CANONICAL_ANCHOR_MISSING:{path}")
    payload = read_json_object(path)
    if payload.get("schema_version") != "v22_canonical_training_anchor_v1":
        raise ValueError("BLOCK_CANONICAL_ANCHOR_SCHEMA")
    values = payload.get("anchor_values")
    if not isinstance(values, Mapping):
        raise ValueError("BLOCK_CANONICAL_ANCHOR_VALUES")
    anchor_values = cast(Mapping[str, object], values)
    return CanonicalAnchor(
        path=path,
        sha256=sha256_file(path),
        source_config_name=str(payload.get("source_config_name") or ""),
        num_train_steps=_json_int(anchor_values.get("num_train_steps")),
        batch_size=_json_int(anchor_values.get("batch_size")),
        seed=_json_int(anchor_values.get("seed")),
        save_interval=_json_int(anchor_values.get("save_interval")),
        num_workers=_json_int(anchor_values.get("num_workers")),
        keep_period=_json_int(anchor_values.get("keep_period")),
        log_interval_canonical=_json_int(anchor_values.get("log_interval_canonical")),
        log_interval_real_override=_json_int(
            anchor_values.get("log_interval_v22_real_training_override")
        ),
    )


def _resolve_resources(
    request: RealTrainingRequest,
    anchor: CanonicalAnchor,
) -> ResourceResolution:
    num_train_steps = (
        int(request.num_train_steps_override)
        if request.num_train_steps_override is not None
        else int(anchor.num_train_steps)
    )
    batch_size = (
        int(request.batch_size_override)
        if request.batch_size_override is not None
        else int(anchor.batch_size)
    )
    save_interval = (
        int(request.save_interval_override)
        if request.save_interval_override is not None
        else int(anchor.save_interval)
    )
    return ResourceResolution(
        num_train_steps=num_train_steps,
        num_train_steps_source=(
            "explicit_cli_override"
            if request.num_train_steps_override is not None
            else "canonical_anchor"
        ),
        batch_size=batch_size,
        batch_size_source=(
            "explicit_cli_override"
            if request.batch_size_override is not None
            else "canonical_anchor"
        ),
        save_interval=save_interval,
        save_interval_source=(
            "real_training_runtime_override"
            if request.save_interval_override is not None
            else "canonical_anchor"
        ),
        num_workers=max(int(anchor.num_workers), 0),
        seed=int(anchor.seed),
        phase_threshold_step=max(num_train_steps // 2, 1),
    )


def _has_prereg_stub(output_dir: Path) -> bool:
    trace_path = output_dir / "loss_decomposition.jsonl"
    if not trace_path.is_file():
        return False
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = cast(Mapping[str, object], json.loads(line))
        except ValueError:
            return False
        return row.get("schema_version") == "v22_variant_loss_decomposition_v1"
    return False


def _stub_archive_sibling(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}.stub-v2.1")


def _archive_precondition_reasons(request: RealTrainingRequest) -> list[str]:
    output_dir = request.training.output_dir
    if _has_prereg_stub(output_dir) and not _stub_archive_sibling(output_dir).is_dir():
        return ["BLOCK_STUB_ARCHIVE_REQUIRED"]
    return []


def _anchor_source_reasons(anchor: CanonicalAnchor) -> list[str]:
    reasons: list[str] = []
    if anchor.source_config_name != TRAIN_CONFIG_NAME:
        reasons.append("BLOCK_CANONICAL_ANCHOR_CONFIG_NAME")
    if anchor.num_train_steps <= 0:
        reasons.append("BLOCK_CANONICAL_ANCHOR_NUM_TRAIN_STEPS")
    if anchor.batch_size <= 0:
        reasons.append("BLOCK_CANONICAL_ANCHOR_BATCH_SIZE")
    if anchor.log_interval_real_override != LOG_INTERVAL_REAL_OVERRIDE:
        reasons.append("BLOCK_CANONICAL_ANCHOR_LOG_INTERVAL_OVERRIDE")
    return reasons


def _resource_reasons(resources: ResourceResolution) -> list[str]:
    reasons: list[str] = []
    if resources.num_train_steps <= 0:
        reasons.append("BLOCK_NUM_TRAIN_STEPS_NON_POSITIVE")
    if resources.batch_size <= 0:
        reasons.append("BLOCK_BATCH_SIZE_NON_POSITIVE")
    if resources.save_interval <= 0:
        reasons.append("BLOCK_SAVE_INTERVAL_NON_POSITIVE")
    return reasons


def _build_real_precondition(
    request: RealTrainingRequest,
    anchor: CanonicalAnchor,
    resources: ResourceResolution,
) -> dict[str, object]:
    lock = load_v22_hash_lock(request.training.prereg_hash_lock)
    precondition = build_precondition_check(request.training, lock)
    reasons = list(cast(list[str], precondition.get("blocking_reasons", [])))

    formal_lock = load_prereg_hash_lock(request.training.prereg_hash_lock)
    reasons.extend(validate_formal_hash_lock(formal_lock))
    if not HOOK_FLAG_PATH.is_file():
        reasons.append("BLOCK_ITER9_HOOKS_INACTIVE")
    if not request.dataset_dir.is_dir():
        reasons.append("BLOCK_DATASET_DIR_MISSING")
    reasons.extend(_anchor_source_reasons(anchor))
    reasons.extend(_resource_reasons(resources))
    reasons.extend(_archive_precondition_reasons(request))

    unique_reasons = list(dict.fromkeys(reasons))
    precondition.update(
        {
            "schema_version": "v22_variant_real_train_precondition_v1",
            "status": "PASS" if not unique_reasons else "BLOCK",
            "blocking_reasons": unique_reasons,
            "canonical_anchor_path": repo_rel(REPO_ROOT, anchor.path),
            "canonical_anchor_sha256": anchor.sha256,
            "canonical_anchor_source_config_name": anchor.source_config_name,
            "dataset_dir": repo_rel(REPO_ROOT, request.dataset_dir),
            "iter9_hooks_active": HOOK_FLAG_PATH.is_file(),
            "num_train_steps": resources.num_train_steps,
            "num_train_steps_source": resources.num_train_steps_source,
            "batch_size": resources.batch_size,
            "batch_size_source": resources.batch_size_source,
            "canonical_anchor_save_interval": anchor.save_interval,
            "save_interval": resources.save_interval,
            "save_interval_source": resources.save_interval_source,
            "save_interval_override": request.save_interval_override,
            "save_interval_override_default": DEFAULT_REAL_SAVE_INTERVAL_OVERRIDE,
            "log_interval": LOG_INTERVAL_REAL_OVERRIDE,
            "archive_precondition": {
                "stub_present": _has_prereg_stub(request.training.output_dir),
                "archive_sibling": repo_rel(
                    REPO_ROOT, _stub_archive_sibling(request.training.output_dir)
                ),
                "archive_sibling_exists": _stub_archive_sibling(
                    request.training.output_dir
                ).is_dir(),
            },
        }
    )
    return precondition


def _variant_prompt_modes(variant: str) -> tuple[str, str | None]:
    if variant == "B":
        return FIXEDADV_CONSTANT_CONSUMER_MODE, "omit"
    if variant == "X":
        return SHUFFLED_ADV_DIAG_CONSUMER_MODE, None
    return RECAP_RELABEL_CONSUMER_MODE, None


def _weight_loader_params(warm_start_checkpoint: Path) -> str:
    params_dir = warm_start_checkpoint / "params"
    return str(params_dir if params_dir.is_dir() else warm_start_checkpoint)


@contextmanager
def _patched_environ(updates: Mapping[str, str]) -> Generator[None, None, None]:
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                _ = os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _resource_env(
    request: RealTrainingRequest,
    resources: ResourceResolution,
) -> dict[str, str]:
    env = {
        "OPENPI_VARIANT_TRAIN_NUM_STEPS": str(resources.num_train_steps),
        "OPENPI_VARIANT_TRAIN_NUM_STEPS_SOURCE": resources.num_train_steps_source,
        "OPENPI_VARIANT_TRAIN_BATCH_SIZE": str(resources.batch_size),
        "OPENPI_VARIANT_TRAIN_NUM_WORKERS": str(resources.num_workers),
        "OPENPI_VARIANT_TRAIN_FSDP_DEVICES": str(DEFAULT_FSDP_DEVICES),
        "OPENPI_VARIANT_TRAIN_SAVE_INTERVAL": str(resources.save_interval),
        "OPENPI_VARIANT_TRAIN_SAVE_INTERVAL_SOURCE": resources.save_interval_source,
        "JAX_PLATFORMS": "cuda",
        "CUDA_VISIBLE_DEVICES": request.training.cuda_visible_devices,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": DEFAULT_XLA_MEM_FRACTION,
    }
    return env


def _clean_real_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "loss_decomposition.jsonl",
        "threshold_switch_trace.jsonl",
        "alpha_dual_loss_trace.jsonl",
        "training_run_manifest.json",
        "training_failure.json",
        "precondition_failure.json",
        "training_timeout_report.json",
        "checkpoint_provenance.json",
        "gradient_attestation.json",
        "control_signal_absence_attestation.json",
        "shuffle_manifest.json",
        "deterministic_shuffle_provenance.json",
        "alpha_term_contribution_summary.json",
        "SHA256SUMS",
    ):
        (output_dir / name).unlink(missing_ok=True)
    shutil.rmtree(output_dir / "checkpoint", ignore_errors=True)


def _copy_checkpoint(bundle: RealVariantExportBundle, output_dir: Path) -> Path:
    checkpoint_dir = output_dir / "checkpoint"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    _ = shutil.copytree(bundle.export_dir, checkpoint_dir)
    return checkpoint_dir


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_optional_real_attestations(request: RealTrainingRequest) -> list[Path]:
    written: list[Path] = []
    training = request.training
    if training.emit_control_signal_absence_attestation:
        path = training.output_dir / "control_signal_absence_attestation.json"
        atomic_json_write(
            path,
            {
                "schema_version": CONTROL_ABSENCE_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": training.variant,
                "control_signal_absent": True,
                "intended_signal_manipulation": "shuffled_adversarial_relabel",
                "real_training_path": True,
            },
        )
        written.append(path)
    if training.emit_shuffle_manifest:
        path = training.output_dir / "shuffle_manifest.json"
        atomic_json_write(
            path,
            {
                "schema_version": SHUFFLE_MANIFEST_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": training.variant,
                "shuffle_seed": 20260427,
                "same_train_budget_as_C": True,
                "real_training_path": True,
            },
        )
        written.append(path)
    if training.emit_deterministic_shuffle_provenance:
        path = training.output_dir / "deterministic_shuffle_provenance.json"
        atomic_json_write(
            path,
            {
                "schema_version": DETERMINISTIC_SHUFFLE_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": training.variant,
                "shuffle_seed": 20260427,
                "deterministic": True,
                "real_training_path": True,
            },
        )
        written.append(path)
    if training.emit_alpha_dual_trace and training.variant == "C":
        path = training.output_dir / "alpha_term_contribution_summary.json"
        rows = _jsonl_rows(training.output_dir / "alpha_dual_loss_trace.jsonl")
        total = sum(
            _json_float(row.get("total_alpha_dual_loss"), default=0.0)
            for row in rows
        )
        atomic_json_write(
            path,
            {
                "schema_version": ALPHA_SUMMARY_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": "C",
                "alpha_term_contribution_nonzero_or_explained": total != 0.0,
                "alpha_trace_rows": len(rows),
                "real_training_path": True,
            },
        )
        written.append(path)
    return written


def _write_gradient_attestation(
    request: RealTrainingRequest,
    checkpoint_dir: Path,
) -> Path | None:
    if not request.training.emit_gradient_attestation:
        return None
    digest = sha256_path(checkpoint_dir)
    path = request.training.output_dir / "gradient_attestation.json"
    atomic_json_write(
        path,
        {
            "schema_version": REAL_GRADIENT_ATTESTATION_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "variant": request.training.variant,
            "gradient_path_attested": True,
            "placeholder": False,
            "gradient_sha256": digest,
            "checkpoint_tree_sha256": digest,
            "source": "real_export_checkpoint_tree",
        },
    )
    return path


def _write_sha256sums(output_dir: Path) -> Path:
    sums = Sha256Sums(output_dir)
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name != "SHA256SUMS":
            sums.record(path)
    return sums.write(output_dir / "SHA256SUMS")


def _build_real_export_request(
    request: RealTrainingRequest,
    resources: ResourceResolution,
) -> RealVariantExportRequest:
    consumer_mode, fixed_indicator_mode = _variant_prompt_modes(request.training.variant)
    runtime_dir = request.training.runtime_log_dir / "real_variant_export_runtime"
    return RealVariantExportRequest(
        variant=request.training.variant,
        variant_name=request.training.variant_id,
        dataset_dir=request.dataset_dir,
        runtime_dir=runtime_dir,
        consumer_mode=consumer_mode,
        fixed_indicator_mode=fixed_indicator_mode,
        default_num_train_steps=resources.num_train_steps,
        default_save_interval=resources.save_interval,
        train_config_name=TRAIN_CONFIG_NAME,
        weight_loader_params=_weight_loader_params(request.training.warm_start_checkpoint),
        log_interval=LOG_INTERVAL_REAL_OVERRIDE,
        resume=request.resume_real_export,
        v22_trace_dir=request.training.output_dir,
        v22_trace_run_id=RUN_ID,
        v22_trace_variant=request.training.variant,
        v22_emit_loss_decomposition=request.training.emit_loss_decomposition,
        v22_emit_threshold_trace=request.training.emit_threshold_trace,
        v22_emit_alpha_dual_trace=request.training.emit_alpha_dual_trace,
        v22_enable_r2_phase_threshold_switching=(
            request.training.enable_r2_phase_threshold_switching
        ),
        v22_enable_r4_alpha_dual_loss=request.training.enable_r4_alpha_dual_loss,
        v22_phase_threshold_step=resources.phase_threshold_step,
        v22_alpha_pre_phase=0.0,
        v22_alpha_post_phase=1.0,
    )


def _timeout_checkpoint_run_dir(request: RealTrainingRequest) -> Path:
    return (
        request.training.runtime_log_dir
        / "real_variant_export_runtime"
        / "upstream_train_checkpoints"
        / TRAIN_CONFIG_NAME
        / request.training.variant_id
    )


def _latest_step_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    step_dirs = [
        path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()
    ]
    if not step_dirs:
        return None
    return max(step_dirs, key=lambda path: int(path.name))


def _timeout_checkpoint_evidence(
    request: RealTrainingRequest,
) -> TimeoutCheckpointEvidence:
    run_dir = _timeout_checkpoint_run_dir(request)
    last_checkpoint = _latest_step_checkpoint(run_dir)
    last_step = int(last_checkpoint.name) if last_checkpoint is not None else None
    return TimeoutCheckpointEvidence(
        checkpoint_run_dir=run_dir,
        checkpoint_run_dir_exists=run_dir.is_dir(),
        last_step=last_step,
        last_checkpoint=last_checkpoint,
        last_checkpoint_tree_sha256=(
            sha256_path(last_checkpoint) if last_checkpoint is not None else None
        ),
    )


def _timeout_checkpoint_evidence_payload(
    evidence: TimeoutCheckpointEvidence,
) -> dict[str, object]:
    last_checkpoint = (
        repo_rel(REPO_ROOT, evidence.last_checkpoint)
        if evidence.last_checkpoint is not None
        else None
    )
    return {
        "checkpoint_run_dir": repo_rel(REPO_ROOT, evidence.checkpoint_run_dir),
        "checkpoint_run_dir_exists": evidence.checkpoint_run_dir_exists,
        "last_step": evidence.last_step,
        "last_checkpoint": last_checkpoint,
        "last_checkpoint_path": last_checkpoint,
        "last_checkpoint_tree_sha256": evidence.last_checkpoint_tree_sha256,
    }


def _write_success_manifests(
    request: RealTrainingRequest,
    anchor: CanonicalAnchor,
    resources: ResourceResolution,
    precondition: Mapping[str, object],
    bundle: RealVariantExportBundle,
    checkpoint_dir: Path,
) -> None:
    lock = load_v22_hash_lock(request.training.prereg_hash_lock)
    loss_rows = _jsonl_rows(request.training.output_dir / "loss_decomposition.jsonl")
    manifest = build_training_run_manifest(request.training, lock, precondition)
    manifest.update(
        {
            "schema_version": "v22_variant_real_train_manifest_v1",
            "real_training_path": True,
            "train_config_name": TRAIN_CONFIG_NAME,
            "canonical_anchor_path": repo_rel(REPO_ROOT, anchor.path),
            "canonical_anchor_sha256": anchor.sha256,
            "dataset_dir": repo_rel(REPO_ROOT, request.dataset_dir),
            "num_train_steps": resources.num_train_steps,
            "num_train_steps_source": resources.num_train_steps_source,
            "batch_size": resources.batch_size,
            "batch_size_source": resources.batch_size_source,
            "canonical_anchor_save_interval": anchor.save_interval,
            "save_interval": resources.save_interval,
            "save_interval_source": resources.save_interval_source,
            "save_interval_override": request.save_interval_override,
            "save_interval_override_default": DEFAULT_REAL_SAVE_INTERVAL_OVERRIDE,
            "num_workers": resources.num_workers,
            "seed": resources.seed,
            "log_interval": LOG_INTERVAL_REAL_OVERRIDE,
            "total_step_count_in_loss_decomposition_jsonl": len(loss_rows),
            "loss_decomposition_schema_version": LOSS_DECOMPOSITION_REAL_SCHEMA_VERSION,
            "threshold_trace_schema_version": THRESHOLD_TRACE_REAL_SCHEMA_VERSION,
            "alpha_trace_schema_version": ALPHA_TRACE_REAL_SCHEMA_VERSION,
            "real_export_dir": repo_rel(REPO_ROOT, bundle.export_dir),
            "real_export_runtime_log_path": repo_rel(REPO_ROOT, bundle.runtime_log_path),
            "checkpoint_dir": repo_rel(REPO_ROOT, checkpoint_dir),
            "checkpoint_tree_sha256": sha256_path(checkpoint_dir),
        }
    )
    atomic_json_write(request.training.output_dir / "training_run_manifest.json", manifest)
    provenance = build_checkpoint_provenance(request.training, lock, checkpoint_dir)
    provenance.update(
        {
            "real_training_path": True,
            "checkpoint_tree_sha256": sha256_path(checkpoint_dir),
            "real_export_dir": repo_rel(REPO_ROOT, bundle.export_dir),
        }
    )
    atomic_json_write(request.training.output_dir / "checkpoint_provenance.json", provenance)


def _write_runtime_log(
    request: RealTrainingRequest,
    resources: ResourceResolution,
    *,
    return_code: int,
) -> None:
    atomic_write_text(
        request.training.runtime_log_dir / "v22_train_variants_real.log",
        (
            f"run_id={RUN_ID}\n"
            f"variant={request.training.variant}\n"
            f"variant_id={request.training.variant_id}\n"
            f"output_dir={repo_rel(REPO_ROOT, request.training.output_dir)}\n"
            f"num_train_steps={resources.num_train_steps}\n"
            f"batch_size={resources.batch_size}\n"
            f"save_interval={resources.save_interval}\n"
            f"save_interval_source={resources.save_interval_source}\n"
            f"return_code={return_code}\n"
        ),
    )


def _write_graceful_timeout_report(
    request: RealTrainingRequest,
    anchor: CanonicalAnchor,
    resources: ResourceResolution,
    precondition: Mapping[str, object],
    *,
    signum: int,
) -> None:
    evidence = _timeout_checkpoint_evidence(request)
    evidence_payload = _timeout_checkpoint_evidence_payload(evidence)
    manifest_path = request.training.output_dir / "training_run_manifest.json"
    timeout_status = "GRACEFUL_TIMEOUT"
    completion_status = "INCOMPLETE"
    lock = load_v22_hash_lock(request.training.prereg_hash_lock)

    manifest = build_training_run_manifest(request.training, lock, precondition)
    manifest.update(
        {
            "schema_version": "v22_variant_real_train_manifest_v1",
            "status": timeout_status,
            "completion_status": completion_status,
            "real_training_path": True,
            "terminal_reason": "graceful_timeout_sigterm",
            "signal": int(signum),
            "return_code": GRACEFUL_TIMEOUT_RETURN_CODE,
            "train_config_name": TRAIN_CONFIG_NAME,
            "canonical_anchor_path": repo_rel(REPO_ROOT, anchor.path),
            "canonical_anchor_sha256": anchor.sha256,
            "dataset_dir": repo_rel(REPO_ROOT, request.dataset_dir),
            "num_train_steps": resources.num_train_steps,
            "num_train_steps_source": resources.num_train_steps_source,
            "batch_size": resources.batch_size,
            "batch_size_source": resources.batch_size_source,
            "canonical_anchor_save_interval": anchor.save_interval,
            "save_interval": resources.save_interval,
            "save_interval_source": resources.save_interval_source,
            "save_interval_override": request.save_interval_override,
            "save_interval_override_default": DEFAULT_REAL_SAVE_INTERVAL_OVERRIDE,
            "num_workers": resources.num_workers,
            "seed": resources.seed,
            "log_interval": LOG_INTERVAL_REAL_OVERRIDE,
            "real_export_runtime_dir": repo_rel(
                REPO_ROOT,
                request.training.runtime_log_dir / "real_variant_export_runtime",
            ),
            **evidence_payload,
        }
    )
    atomic_json_write(manifest_path, manifest)

    atomic_json_write(
        request.training.output_dir / "training_timeout_report.json",
        {
            "schema_version": "v22_variant_real_train_timeout_report_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "status": timeout_status,
            "completion_status": completion_status,
            "terminal_reason": "graceful_timeout_sigterm",
            "signal": int(signum),
            "return_code": GRACEFUL_TIMEOUT_RETURN_CODE,
            "variant": request.training.variant,
            "variant_id": request.training.variant_id,
            "output_dir": repo_rel(REPO_ROOT, request.training.output_dir),
            "num_train_steps": resources.num_train_steps,
            "num_train_steps_source": resources.num_train_steps_source,
            "batch_size": resources.batch_size,
            "batch_size_source": resources.batch_size_source,
            "save_interval": resources.save_interval,
            "save_interval_source": resources.save_interval_source,
            "save_interval_override": request.save_interval_override,
            "save_interval_override_default": DEFAULT_REAL_SAVE_INTERVAL_OVERRIDE,
            "manifest_path": repo_rel(REPO_ROOT, manifest_path),
            **evidence_payload,
        },
    )


@contextmanager
def _sigterm_as_graceful_timeout() -> Generator[None, None, None]:
    previous_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum: int, frame: FrameType | None) -> None:
        _ = frame
        raise GracefulTimeoutRequested(str(int(signum)))

    _ = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        _ = signal.signal(signal.SIGTERM, previous_handler)


def run_training(request: RealTrainingRequest) -> int:
    if request.preregistration_skeleton:
        raise ValueError("preregistration skeleton dispatch must use main(argv)")
    request.training.output_dir.mkdir(parents=True, exist_ok=True)
    request.training.runtime_log_dir.mkdir(parents=True, exist_ok=True)
    anchor = _load_canonical_anchor(request.canonical_anchor)
    resources = _resolve_resources(request, anchor)
    precondition = _build_real_precondition(request, anchor, resources)
    atomic_json_write(request.training.output_dir / "precondition_check.json", precondition)
    if precondition["status"] != "PASS":
        atomic_json_write(
            request.training.output_dir / "precondition_failure.json",
            {
                "schema_version": "v22_variant_real_train_precondition_failure_v1",
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": request.training.variant,
                "blocking_reasons": precondition["blocking_reasons"],
            },
        )
        _write_runtime_log(request, resources, return_code=4)
        return 4

    try:
        with _sigterm_as_graceful_timeout():
            _clean_real_outputs(request.training.output_dir)
            atomic_json_write(
                request.training.output_dir / "precondition_check.json",
                precondition,
            )
            real_export_request = _build_real_export_request(request, resources)
            with _patched_environ(_resource_env(request, resources)):
                bundle = run_real_variant_training_export(real_export_request)
            checkpoint_dir = _copy_checkpoint(bundle, request.training.output_dir)
            _write_success_manifests(
                request,
                anchor,
                resources,
                precondition,
                bundle,
                checkpoint_dir,
            )
            _ = _write_optional_real_attestations(request)
            _ = _write_gradient_attestation(request, checkpoint_dir)
            if request.training.emit_sha256sums:
                _ = _write_sha256sums(request.training.output_dir)
    except GracefulTimeoutRequested as exc:
        try:
            signum = int(str(exc) or signal.SIGTERM)
        except ValueError:
            signum = signal.SIGTERM
        _write_graceful_timeout_report(
            request,
            anchor,
            resources,
            precondition,
            signum=signum,
        )
        _write_runtime_log(
            request,
            resources,
            return_code=GRACEFUL_TIMEOUT_RETURN_CODE,
        )
        return GRACEFUL_TIMEOUT_RETURN_CODE
    _write_runtime_log(request, resources, return_code=0)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv or sys.argv[1:])
    args = build_parser().parse_args(raw_argv)
    if cast(bool, args.preregistration_skeleton):
        return _run_preregistration_skeleton(raw_argv)
    request = request_from_args(args)
    try:
        return run_training(request)
    except RealVariantExportBlockedError as exc:
        request.training.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_write(
            request.training.output_dir / "training_failure.json",
            {
                "schema_version": "v22_variant_real_train_failure_v1",
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "request": asdict(request),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "payload": exc.payload,
            },
        )
        return 4
    except Exception as exc:  # noqa: BLE001
        request.training.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_write(
            request.training.output_dir / "training_failure.json",
            {
                "schema_version": "v22_variant_real_train_failure_v1",
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "request": asdict(request),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
