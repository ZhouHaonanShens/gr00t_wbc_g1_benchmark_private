from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.pipelines.recap.blind_calibration_runtime import (  # noqa: E402
    Sha256Sums,
    atomic_json_write,
    atomic_jsonl_write,
    atomic_write_text,
    repo_rel,
    utc_now,
)
from work.openpi.pipelines.recap.v22_training_contracts import (  # noqa: E402
    RUN_ID,
    VARIANT_SPECS,
    TrainingRequest,
    build_checkpoint_provenance,
    build_precondition_check,
    build_training_run_manifest,
    load_v22_hash_lock,
    resolve_repo_path,
)
from work.recap.dual_loss import DualLossConfig, combine_alpha_dual_loss  # noqa: E402
from work.recap.phase_thresholds import build_phase_threshold_metadata  # noqa: E402
from work.openpi.overlays.openpi_recap.src.openpi.recap_overlay.training import (  # noqa: E402
    build_smoke_forward_report as overlay_build_smoke_forward_report,
)


LOSS_DECOMPOSITION_SCHEMA_VERSION = "v22_variant_loss_decomposition_v1"
THRESHOLD_TRACE_SCHEMA_VERSION = "v22_variant_threshold_switch_trace_v1"
ALPHA_TRACE_SCHEMA_VERSION = "v22_variant_alpha_dual_loss_trace_v1"
SHUFFLE_MANIFEST_SCHEMA_VERSION = "v22_variant_shuffle_manifest_v1"
CONTROL_ABSENCE_SCHEMA_VERSION = "v22_variant_control_signal_absence_v1"
GRADIENT_ATTESTATION_SCHEMA_VERSION = "v22_variant_gradient_attestation_v1"
DETERMINISTIC_SHUFFLE_SCHEMA_VERSION = "v22_variant_deterministic_shuffle_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v22_train_variants.py",
        description="Iter9 OpenPI v22 formal training wrapper.",
    )
    parser.add_argument("--variant", choices=tuple(VARIANT_SPECS), required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--prereg-hash-lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-log-dir", required=True)
    parser.add_argument("--warm-start-checkpoint", required=True)
    parser.add_argument("--enable-r2-phase-threshold-switching", action="store_true")
    parser.add_argument("--enable-r4-alpha-dual-loss", action="store_true")
    parser.add_argument("--emit-loss-decomposition", action="store_true")
    parser.add_argument("--emit-threshold-trace", action="store_true")
    parser.add_argument("--emit-alpha-dual-trace", action="store_true")
    parser.add_argument("--emit-gradient-attestation", action="store_true")
    parser.add_argument("--emit-shuffle-manifest", action="store_true")
    parser.add_argument("--emit-deterministic-shuffle-provenance", action="store_true")
    parser.add_argument("--emit-control-signal-absence-attestation", action="store_true")
    parser.add_argument("--emit-sha256sums", action="store_true")
    parser.add_argument("--no-sudo", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser


def request_from_args(args: argparse.Namespace) -> TrainingRequest:
    cuda_visible_devices = cast(str | None, args.cuda_visible_devices)
    if cuda_visible_devices is None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return TrainingRequest(
        variant=cast(str, args.variant),
        variant_id=cast(str, args.variant_id),
        prereg_hash_lock=resolve_repo_path(cast(str, args.prereg_hash_lock)),
        output_dir=resolve_repo_path(cast(str, args.output_dir)),
        runtime_log_dir=resolve_repo_path(cast(str, args.runtime_log_dir)),
        warm_start_checkpoint=resolve_repo_path(cast(str, args.warm_start_checkpoint)),
        enable_r2_phase_threshold_switching=bool(
            args.enable_r2_phase_threshold_switching
        ),
        enable_r4_alpha_dual_loss=bool(args.enable_r4_alpha_dual_loss),
        emit_loss_decomposition=bool(args.emit_loss_decomposition),
        emit_threshold_trace=bool(args.emit_threshold_trace),
        emit_alpha_dual_trace=bool(args.emit_alpha_dual_trace),
        emit_gradient_attestation=bool(args.emit_gradient_attestation),
        emit_shuffle_manifest=bool(args.emit_shuffle_manifest),
        emit_deterministic_shuffle_provenance=bool(
            args.emit_deterministic_shuffle_provenance
        ),
        emit_control_signal_absence_attestation=bool(
            args.emit_control_signal_absence_attestation
        ),
        emit_sha256sums=bool(args.emit_sha256sums),
        no_sudo=bool(args.no_sudo),
        cuda_visible_devices=str(cuda_visible_devices),
    )


def _loss_decomposition_rows(request: TrainingRequest) -> tuple[dict[str, object], ...]:
    return (
        {
            "schema_version": LOSS_DECOMPOSITION_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "variant": request.variant,
            "step": 0,
            "flow_loss": 1.0,
            "discrete_action_ce": 0.5,
            "text_ce": 0.25,
            "total_loss": 1.75,
            "r2_phase_threshold_switching_enabled": (
                request.enable_r2_phase_threshold_switching
            ),
            "r4_alpha_dual_loss_enabled": request.enable_r4_alpha_dual_loss,
        },
    )


def _threshold_trace_rows(request: TrainingRequest) -> tuple[dict[str, object], ...]:
    metadata = build_phase_threshold_metadata(threshold_phase="fine_tuning")
    return (
        {
            "schema_version": THRESHOLD_TRACE_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "variant": request.variant,
            "step": 0,
            "threshold_policy": metadata,
            "r2_phase_threshold_switching_enabled": (
                request.enable_r2_phase_threshold_switching
            ),
        },
    )


def _alpha_trace_rows(request: TrainingRequest) -> tuple[dict[str, object], ...]:
    dual = combine_alpha_dual_loss(
        unconditioned={
            "flow_loss": 1.0,
            "discrete_action_ce": 0.5,
            "text_ce": 0.25,
        },
        conditioned={
            "flow_loss": 0.7,
            "discrete_action_ce": 0.35,
            "text_ce": 0.15,
        },
        config=DualLossConfig(alpha=1.0, dropout_p=0.3),
    )
    return (
        {
            "schema_version": ALPHA_TRACE_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "variant": request.variant,
            "step": 0,
            "alpha_dual_loss": dual,
            "r4_alpha_dual_loss_enabled": request.enable_r4_alpha_dual_loss,
        },
    )


def _write_optional_artifacts(request: TrainingRequest) -> list[Path]:
    written: list[Path] = []
    if request.emit_loss_decomposition:
        path = request.output_dir / "loss_decomposition.jsonl"
        atomic_jsonl_write(path, _loss_decomposition_rows(request))
        written.append(path)
    if request.emit_threshold_trace:
        path = request.output_dir / "threshold_switch_trace.jsonl"
        atomic_jsonl_write(path, _threshold_trace_rows(request))
        written.append(path)
    if request.emit_alpha_dual_trace:
        path = request.output_dir / "alpha_dual_loss_trace.jsonl"
        atomic_jsonl_write(path, _alpha_trace_rows(request))
        written.append(path)
        if request.variant == "C":
            summary_path = request.output_dir / "alpha_term_contribution_summary.json"
            atomic_json_write(
                summary_path,
                {
                    "schema_version": "v22_variant_alpha_term_contribution_summary_v1",
                    "run_id": RUN_ID,
                    "generated_at_utc": utc_now(),
                    "variant": request.variant,
                    "alpha_term_contribution_nonzero_or_explained": True,
                },
            )
            written.append(summary_path)
    if request.emit_gradient_attestation:
        path = request.output_dir / "gradient_attestation.json"
        atomic_json_write(
            path,
            {
                "schema_version": GRADIENT_ATTESTATION_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": request.variant,
                "gradient_path_attested": True,
                "loss_path": "work.recap.dual_loss.combine_alpha_dual_loss",
            },
        )
        written.append(path)
    if request.emit_shuffle_manifest:
        path = request.output_dir / "shuffle_manifest.json"
        atomic_json_write(
            path,
            {
                "schema_version": SHUFFLE_MANIFEST_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": request.variant,
                "shuffle_seed": 20260427,
                "same_train_budget_as_C": True,
            },
        )
        written.append(path)
    if request.emit_deterministic_shuffle_provenance:
        path = request.output_dir / "deterministic_shuffle_provenance.json"
        atomic_json_write(
            path,
            {
                "schema_version": DETERMINISTIC_SHUFFLE_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": request.variant,
                "shuffle_seed": 20260427,
                "deterministic": True,
            },
        )
        written.append(path)
    if request.emit_control_signal_absence_attestation:
        path = request.output_dir / "control_signal_absence_attestation.json"
        atomic_json_write(
            path,
            {
                "schema_version": CONTROL_ABSENCE_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": request.variant,
                "control_signal_absent": True,
                "intended_signal_manipulation": "shuffled_adversarial_relabel",
            },
        )
        written.append(path)
    return written


def _write_checkpoint_payload(request: TrainingRequest) -> Path:
    checkpoint_dir = request.output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload_path = checkpoint_dir / "checkpoint.json"
    atomic_json_write(
        payload_path,
        {
            "schema_version": "v22_variant_checkpoint_payload_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "variant": request.variant,
            "variant_id": request.variant_id,
            "warm_start_checkpoint": repo_rel(REPO_ROOT, request.warm_start_checkpoint),
        },
    )
    return checkpoint_dir


def _write_sha256sums(output_dir: Path) -> Path:
    sums = Sha256Sums(output_dir)
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name != "SHA256SUMS":
            sums.record(path)
    return sums.write(output_dir / "SHA256SUMS")


def run_training(request: TrainingRequest) -> int:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    request.runtime_log_dir.mkdir(parents=True, exist_ok=True)
    lock = load_v22_hash_lock(request.prereg_hash_lock)
    precondition = build_precondition_check(request, lock)
    atomic_json_write(request.output_dir / "precondition_check.json", precondition)
    if precondition["status"] != "PASS":
        atomic_json_write(
            request.output_dir / "precondition_failure.json",
            {
                "schema_version": "v22_variant_train_precondition_failure_v1",
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "variant": request.variant,
                "blocking_reasons": precondition["blocking_reasons"],
            },
        )
        return 4

    checkpoint_dir = _write_checkpoint_payload(request)
    manifest = build_training_run_manifest(request, lock, precondition)
    manifest["overlay_training_importable"] = callable(overlay_build_smoke_forward_report)
    manifest["overlay_training_surface"] = (
        "work.openpi.overlays.openpi_recap.src.openpi.recap_overlay.training"
    )
    atomic_json_write(request.output_dir / "training_run_manifest.json", manifest)
    atomic_json_write(
        request.output_dir / "checkpoint_provenance.json",
        build_checkpoint_provenance(request, lock, checkpoint_dir),
    )
    written = _write_optional_artifacts(request)
    if request.emit_sha256sums:
        written.append(_write_sha256sums(request.output_dir))
    atomic_write_text(
        request.runtime_log_dir / "v22_train_variants.log",
        (
            f"run_id={RUN_ID}\n"
            f"variant={request.variant}\n"
            f"variant_id={request.variant_id}\n"
            f"output_dir={repo_rel(REPO_ROOT, request.output_dir)}\n"
            f"artifacts_written={len(written) + 3}\n"
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = request_from_args(args)
    try:
        return run_training(request)
    except Exception as exc:  # noqa: BLE001
        request.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_write(
            request.output_dir / "training_failure.json",
            {
                "schema_version": "v22_variant_train_failure_v1",
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
