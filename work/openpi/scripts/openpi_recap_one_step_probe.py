#!/usr/bin/env python3
"""Thin p1 one-step probe wrapper for the OpenPI LIBERO full-update lane."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterator, cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.dataloader import write_json  # noqa: E402
from work.openpi.prompting.routes import RECAP_RELABEL_CONSUMER_MODE  # noqa: E402
from work.openpi.recap.real_variant_export import (  # noqa: E402
    OPENPI_VENV_PYTHON,
    RealVariantExportBlockedError,
    RealVariantExportRequest,
    run_real_variant_training_export,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "agent" / "artifacts" / "openpi_libero_recap_v2_full_update"
)
P0_SCOPE_AUDIT_DIRNAME = "p0_scope_audit"
P1_ONE_STEP_DIRNAME = "p1_one_step"
P0_SCOPE_AUDIT_MANIFEST_NAME = "scope_audit_manifest.json"
ONE_STEP_PROBE_NAME = "one_step_probe.json"
PROBE_METRICS_NAME = "probe_metrics.json"
REAL_RUNTIME_DIRNAME = "real_variant_runtime"
LANE_NAME = "openpi_libero_recap_v2_full_update"
PHASE_NAME = "p1_one_step"
BLOCKED_DATASET_NOT_MATERIALIZED = "BLOCKED(dataset_not_materialized)"
DATASET_NOT_MATERIALIZED_CODE = "dataset_not_materialized"
REAL_VARIANT_NAME = "openpi_libero_recap_v2_full_update_one_step_probe"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openpi_recap_one_step_probe.py",
        description=(
            "Emit the p1 one-step probe verdict and execute the real OpenPI "
            "one-step path only when the p0 dataset-materialization gate is green."
        ),
    )
    _ = parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    _ = parser.add_argument(
        "--p0-scope-audit-manifest",
        default=str(
            DEFAULT_OUTPUT_ROOT
            / P0_SCOPE_AUDIT_DIRNAME
            / P0_SCOPE_AUDIT_MANIFEST_NAME
        ),
    )
    _ = parser.add_argument("--gpu", type=int, default=2)
    return parser


def _p1_dir(output_root: Path) -> Path:
    return output_root / P1_ONE_STEP_DIRNAME


def _one_step_probe_path(output_root: Path) -> Path:
    return _p1_dir(output_root) / ONE_STEP_PROBE_NAME


def _probe_metrics_path(output_root: Path) -> Path:
    return _p1_dir(output_root) / PROBE_METRICS_NAME


def _real_runtime_dir(output_root: Path) -> Path:
    return _p1_dir(output_root) / REAL_RUNTIME_DIRNAME


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return {str(key): value for key, value in cast(dict[object, object], payload).items()}


def _coerce_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(raw, (int, float)):
        return bool(raw)
    return False


def _resolve_gpu_pci_bus_id(gpu: int) -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) == 2 and parts[0] == str(gpu):
            return parts[1]
    return ""


def _dataset_join_root_status(manifest: dict[str, object]) -> dict[str, object]:
    raw_status = manifest.get("dataset_join_root_status")
    if not isinstance(raw_status, dict):
        raise ValueError("p0 scope audit manifest missing dataset_join_root_status object")
    status = cast(dict[object, object], raw_status)
    return {str(key): value for key, value in status.items()}


def _dataset_is_materialized(manifest: dict[str, object]) -> bool:
    status = _dataset_join_root_status(manifest)
    return _coerce_bool(status.get("materialized", False))


def _planned_runtime_env(gpu: int) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "JAX_PLATFORMS": "cuda",
        "JAX_PLATFORM_NAME": "cuda",
    }


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_wrapper_command(raw_args: list[str]) -> list[str]:
    return [str(sys.executable), str(Path(__file__).resolve()), *raw_args]


def _command_shell(command: list[str]) -> str:
    return shlex.join(command)


def _command_context(
    *,
    raw_args: list[str],
    output_root: Path,
    p0_manifest_path: Path,
    gpu: int,
) -> dict[str, object]:
    command = _build_wrapper_command(raw_args)
    return {
        "command": command,
        "command_shell": _command_shell(command),
        "working_directory": str(REPO_ROOT),
        "python_executable": sys.executable,
        "script_path": str(Path(__file__).resolve()),
        "p0_scope_audit_manifest_path": str(p0_manifest_path),
        "planned_runtime_env": _planned_runtime_env(gpu),
        "gpu": gpu,
        "output_context": {
            "output_root": str(output_root),
            "phase_dir": str(_p1_dir(output_root)),
            "one_step_probe_json_path": str(_one_step_probe_path(output_root)),
            "probe_metrics_path": str(_probe_metrics_path(output_root)),
            "real_runtime_dir": str(_real_runtime_dir(output_root)),
        },
        "expected_openpi_python": str(OPENPI_VENV_PYTHON),
    }


def _blocked_command_context(
    *,
    raw_args: list[str],
    output_root: Path,
    p0_manifest_path: Path,
) -> dict[str, object]:
    command_context = _command_context(
        raw_args=raw_args,
        output_root=output_root,
        p0_manifest_path=p0_manifest_path,
        gpu=2,
    )
    command_context["planned_runtime_env"] = {
        "CUDA_VISIBLE_DEVICES": None,
        "JAX_PLATFORMS": None,
        "JAX_PLATFORM_NAME": None,
    }
    command_context["gpu"] = None
    return command_context


def _base_payload(
    *,
    raw_args: list[str],
    manifest: dict[str, object],
    output_root: Path,
    p0_manifest_path: Path,
    gpu: int,
    resolved_egl_device_pci_bus_id: str,
) -> dict[str, object]:
    return {
        "schema_version": "openpi_libero_p1_one_step_probe_v1",
        "artifact_kind": "p1_one_step_probe",
        "lane": LANE_NAME,
        "phase": PHASE_NAME,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gpu": gpu,
        "cuda_visible_devices": str(gpu),
        "resolved_egl_device_pci_bus_id": resolved_egl_device_pci_bus_id,
        "p0_scope_audit_manifest_path": str(p0_manifest_path),
        "dataset_join_root": str(manifest.get("dataset_join_root", "")),
        "dataset_join_root_status": _dataset_join_root_status(manifest),
        "command_context": _command_context(
            raw_args=raw_args,
            output_root=output_root,
            p0_manifest_path=p0_manifest_path,
            gpu=gpu,
        ),
        "python": {
            "observed": sys.executable,
            "expected": str(OPENPI_VENV_PYTHON),
        },
    }


def _build_blocked_payload(
    *,
    raw_args: list[str],
    manifest: dict[str, object],
    output_root: Path,
    p0_manifest_path: Path,
    gpu: int,
    resolved_egl_device_pci_bus_id: str,
) -> dict[str, object]:
    payload = _base_payload(
        raw_args=raw_args,
        manifest=manifest,
        output_root=output_root,
        p0_manifest_path=p0_manifest_path,
        gpu=gpu,
        resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
    )
    payload.update(
        {
            "gpu": None,
            "cuda_visible_devices": None,
            "resolved_egl_device_pci_bus_id": None,
            "command_context": _blocked_command_context(
                raw_args=raw_args,
                output_root=output_root,
                p0_manifest_path=p0_manifest_path,
            ),
            "status": BLOCKED_DATASET_NOT_MATERIALIZED,
            "verdict": BLOCKED_DATASET_NOT_MATERIALIZED,
            "skip_before_execute": True,
            "runtime_started": False,
            "probe_pass": False,
            "blocking_reasons": [DATASET_NOT_MATERIALIZED_CODE],
            "loss_values": [],
            "any_grad_nonzero": False,
            "any_param_delta_nonzero": False,
            "failure": {
                "kind": "blocker",
                "blocking_reason": DATASET_NOT_MATERIALIZED_CODE,
                "reason": "dataset join root is not materialized; skipping one-step probe before JAX/CUDA startup",
            },
        }
    )
    return payload


def _build_real_variant_request(output_root: Path, dataset_join_root: Path) -> RealVariantExportRequest:
    return RealVariantExportRequest(
        variant="one_step_probe",
        variant_name=REAL_VARIANT_NAME,
        dataset_dir=dataset_join_root,
        runtime_dir=_real_runtime_dir(output_root),
        consumer_mode=RECAP_RELABEL_CONSUMER_MODE,
        fixed_indicator_mode=None,
        probe_metrics_path=_probe_metrics_path(output_root),
    )


def _load_probe_metrics(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    return payload


def _build_runtime_failure_payload(
    *,
    raw_args: list[str],
    manifest: dict[str, object],
    output_root: Path,
    p0_manifest_path: Path,
    gpu: int,
    resolved_egl_device_pci_bus_id: str,
    blocker_payload: dict[str, object],
) -> dict[str, object]:
    payload = _base_payload(
        raw_args=raw_args,
        manifest=manifest,
        output_root=output_root,
        p0_manifest_path=p0_manifest_path,
        gpu=gpu,
        resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
    )
    blocker_code = str(blocker_payload.get("blocker_code", "real_variant_training_failed"))
    payload.update(
        {
            "status": "FAIL",
            "verdict": "FAIL",
            "skip_before_execute": False,
            "runtime_started": blocker_code == "real_variant_training_failed",
            "probe_pass": False,
            "blocking_reasons": [blocker_code],
            "loss_values": [],
            "any_grad_nonzero": False,
            "any_param_delta_nonzero": False,
            "failure": blocker_payload,
        }
    )
    return payload


def _build_probe_result_payload(
    *,
    raw_args: list[str],
    manifest: dict[str, object],
    output_root: Path,
    p0_manifest_path: Path,
    gpu: int,
    resolved_egl_device_pci_bus_id: str,
    probe_metrics: dict[str, object],
    runtime_log_path: Path,
    export_dir: Path,
) -> dict[str, object]:
    payload = _base_payload(
        raw_args=raw_args,
        manifest=manifest,
        output_root=output_root,
        p0_manifest_path=p0_manifest_path,
        gpu=gpu,
        resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
    )
    loss_values = probe_metrics.get("loss_values", [])
    if not isinstance(loss_values, list):
        raise ValueError("probe metrics payload must expose loss_values as a list")
    probe_pass = _coerce_bool(probe_metrics.get("probe_pass", False))
    blocking_reasons = [] if probe_pass else ["missing_probe_metrics"]
    payload.update(
        {
            "status": "PASS" if probe_pass else "FAIL",
            "verdict": "PASS" if probe_pass else "FAIL",
            "skip_before_execute": False,
            "runtime_started": True,
            "probe_pass": probe_pass,
            "blocking_reasons": blocking_reasons,
            "loss_values": loss_values,
            "any_grad_nonzero": _coerce_bool(
                probe_metrics.get("any_grad_nonzero", False)
            ),
            "any_param_delta_nonzero": _coerce_bool(
                probe_metrics.get("any_param_delta_nonzero", False)
            ),
            "runtime_output": {
                "runtime_log_path": str(runtime_log_path),
                "export_dir": str(export_dir),
                "probe_metrics_path": str(_probe_metrics_path(output_root)),
            },
            "probe_metrics": probe_metrics,
        }
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    args = _build_parser().parse_args(raw_args)
    gpu = int(args.gpu)
    if gpu != 2:
        raise ValueError("this p1 one-step probe is pinned to gpu=2")

    output_root = Path(str(args.output_root)).resolve()
    p0_manifest_path = Path(str(args.p0_scope_audit_manifest)).resolve()
    manifest = _load_json_object(p0_manifest_path)
    resolved_egl_device_pci_bus_id = _resolve_gpu_pci_bus_id(gpu)

    if not _dataset_is_materialized(manifest):
        write_json(
            _one_step_probe_path(output_root),
            _build_blocked_payload(
                raw_args=raw_args,
                manifest=manifest,
                output_root=output_root,
                p0_manifest_path=p0_manifest_path,
                gpu=gpu,
                resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
            ),
        )
        return 0

    dataset_join_root = Path(str(manifest.get("dataset_join_root", ""))).resolve()
    request = _build_real_variant_request(output_root, dataset_join_root)
    with _temporary_env(_planned_runtime_env(gpu)):
        try:
            export_bundle = run_real_variant_training_export(request)
        except RealVariantExportBlockedError as exc:
            write_json(
                _one_step_probe_path(output_root),
                _build_runtime_failure_payload(
                    raw_args=raw_args,
                    manifest=manifest,
                    output_root=output_root,
                    p0_manifest_path=p0_manifest_path,
                    gpu=gpu,
                    resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
                    blocker_payload=exc.payload,
                ),
            )
            return 1

    probe_metrics = _load_probe_metrics(_probe_metrics_path(output_root))
    payload = _build_probe_result_payload(
        raw_args=raw_args,
        manifest=manifest,
        output_root=output_root,
        p0_manifest_path=p0_manifest_path,
        gpu=gpu,
        resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
        probe_metrics=probe_metrics,
        runtime_log_path=export_bundle.runtime_log_path,
        export_dir=export_bundle.export_dir,
    )
    write_json(_one_step_probe_path(output_root), payload)
    return 0 if _coerce_bool(payload["probe_pass"]) else 1


__all__ = [
    "BLOCKED_DATASET_NOT_MATERIALIZED",
    "DATASET_NOT_MATERIALIZED_CODE",
    "DEFAULT_OUTPUT_ROOT",
    "ONE_STEP_PROBE_NAME",
    "P0_SCOPE_AUDIT_MANIFEST_NAME",
    "P0_SCOPE_AUDIT_DIRNAME",
    "P1_ONE_STEP_DIRNAME",
    "REAL_VARIANT_NAME",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
