#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "stage3_single_gpu_smoke_verdict_v1"
TASK_NAME = "stage3_single_gpu_formal_geometry_smoke"
UNLOCK_SIGNAL = "single_gpu_formal_baseline_allowed"
EXPECTED_CUDA_VISIBLE_DEVICES = "1"
EXPECTED_NUM_GPUS = 1
EXPECTED_USE_DDP = False
EXPECTED_TORCHRUN_INVOKED = False
EXPECTED_GLOBAL_BATCH_SIZE = 4
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 4
EXPECTED_PER_DEVICE_BATCH_SIZE = 1
EXPECTED_EFFECTIVE_UPDATE_BATCH = 4
EXPECTED_GLOBAL_STEP = 8
EXPECTED_CHECKPOINT_DIRNAME = "checkpoint-8"
SUMMARY_FILENAME = "delegate_finetune_summary.json"
TRAINER_STATE_FILENAME = "trainer_state.json"
TORCH_CUDA_MEMORY_FILENAME = "torch_cuda_memory_rank0.json"
NVIDIA_SMI_ACTIVE_FILENAME = "nvidia_smi_active_sampling_rank0.json"
VERSION_SURFACE_FILENAME = "version_surface.json"

RUNTIME_FAILURE_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "oom",
        re.compile(
            r"cuda out of memory|outofmemoryerror|cublas_status_alloc_failed|\boom\b",
            re.IGNORECASE,
        ),
        "OOM observed in runtime log",
    ),
    (
        "illegal_memory_access",
        re.compile(r"illegal memory access", re.IGNORECASE),
        "illegal memory access observed in runtime log",
    ),
    (
        "child_failed",
        re.compile(r"childfailederror", re.IGNORECASE),
        "ChildFailedError observed in runtime log",
    ),
    (
        "nccl",
        re.compile(r"\bnccl\b", re.IGNORECASE),
        "NCCL observed in runtime log",
    ),
    (
        "distributed_init",
        re.compile(
            r"torch\.distributed\.init_process_group|init_process_group\(|distributed init",
            re.IGNORECASE,
        ),
        "torch.distributed init observed in runtime log",
    ),
    (
        "data_parallel",
        re.compile(r"\bdataparallel\b|\bdistributeddataparallel\b", re.IGNORECASE),
        "DataParallel/distributed wrapper observed in runtime log",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="30e_stage3_single_gpu_smoke_verdict.py",
        description=(
            "Read a stage3 single-GPU smoke output directory plus runtime logs and emit a fail-closed verdict JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--smoke-output-dir", type=Path, required=True)
    _ = parser.add_argument("--runtime-log-dir", type=Path, required=True)
    _ = parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _check_exact(
    *,
    actual: Any,
    expected: Any,
    missing_reason: str,
    mismatch_reason: str,
    blocking_reasons: list[str],
) -> bool:
    if actual is None:
        _append_reason(blocking_reasons, missing_reason)
        return False
    if actual != expected:
        _append_reason(blocking_reasons, mismatch_reason)
        return False
    return True


def _discover_runtime_logs(runtime_log_dir: Path) -> list[Path]:
    if not runtime_log_dir.is_dir():
        return []
    return sorted(path for path in runtime_log_dir.iterdir() if path.is_file() and path.suffix == ".log")


def _select_authoritative_runtime_log(
    *,
    runtime_log_dir: Path,
    summary_payload: Mapping[str, Any] | None,
) -> tuple[Path | None, str]:
    summary_runtime_log = None
    if summary_payload is not None:
        candidate = _nested_get(summary_payload, "runtime_log_path")
        if isinstance(candidate, str) and candidate.strip():
            path = Path(candidate)
            if path.is_file() and path.parent == runtime_log_dir.resolve():
                summary_runtime_log = path
    if summary_runtime_log is not None:
        return summary_runtime_log, "summary.runtime_log_path"

    logs = _discover_runtime_logs(runtime_log_dir)
    if len(logs) == 1:
        return logs[0], "single_log_in_runtime_log_dir"
    if len(logs) > 1:
        return max(logs, key=lambda path: path.name), "latest_log_name_in_runtime_log_dir"
    return None, "no_runtime_log_found"


def _scan_runtime_log(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "runtime_log_path": None,
        "torchrun_invoked": None,
        "failure_token_hits": [],
    }
    if path is None or not path.is_file():
        return result

    text = path.read_text(encoding="utf-8", errors="replace")
    torchrun_match = re.findall(
        r"training_launcher_uses_torchrun=(True|False)",
        text,
        flags=re.IGNORECASE,
    )
    failure_hits: list[dict[str, str]] = []
    for token_name, pattern, _ in RUNTIME_FAILURE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        excerpt_start = max(0, match.start() - 80)
        excerpt_end = min(len(text), match.end() + 80)
        failure_hits.append(
            {
                "token": token_name,
                "match": match.group(0),
                "excerpt": text[excerpt_start:excerpt_end].replace("\n", "\\n"),
            }
        )

    result.update(
        {
            "available": True,
            "runtime_log_path": str(path),
            "torchrun_invoked": (
                None if not torchrun_match else torchrun_match[-1].strip().lower() == "true"
            ),
            "failure_token_hits": failure_hits,
        }
    )
    return result


def build_verdict(
    *,
    smoke_output_dir: Path,
    runtime_log_dir: Path,
) -> dict[str, Any]:
    blocking_reasons: list[str] = []

    smoke_output_dir = smoke_output_dir.resolve()
    runtime_log_dir = runtime_log_dir.resolve()

    summary_path = smoke_output_dir / SUMMARY_FILENAME
    trainer_state_path = smoke_output_dir / TRAINER_STATE_FILENAME
    checkpoint_path = smoke_output_dir / EXPECTED_CHECKPOINT_DIRNAME
    metadata_dir = smoke_output_dir / "repo_local_metadata"
    version_surface_path = metadata_dir / VERSION_SURFACE_FILENAME
    torch_cuda_memory_path = metadata_dir / TORCH_CUDA_MEMORY_FILENAME
    nvidia_smi_active_path = metadata_dir / NVIDIA_SMI_ACTIVE_FILENAME

    summary_payload = _read_json_object(summary_path)
    trainer_state_payload = _read_json_object(trainer_state_path)
    version_surface_payload = _read_json_object(version_surface_path)
    torch_cuda_memory_payload = _read_json_object(torch_cuda_memory_path)
    nvidia_smi_active_payload = _read_json_object(nvidia_smi_active_path)

    if summary_payload is None:
        _append_reason(blocking_reasons, f"missing {SUMMARY_FILENAME}")
        effective_config: Mapping[str, Any] = {}
    else:
        raw_effective_config = summary_payload.get("effective_config")
        if isinstance(raw_effective_config, Mapping):
            effective_config = raw_effective_config
        else:
            effective_config = {}
            _append_reason(blocking_reasons, "delegate_finetune_summary.effective_config missing")

    if trainer_state_payload is None:
        _append_reason(blocking_reasons, f"missing {TRAINER_STATE_FILENAME}")

    authoritative_runtime_log, runtime_log_selection = _select_authoritative_runtime_log(
        runtime_log_dir=runtime_log_dir,
        summary_payload=summary_payload,
    )
    runtime_log_scan = _scan_runtime_log(authoritative_runtime_log)
    if authoritative_runtime_log is None:
        _append_reason(blocking_reasons, "authoritative runtime log missing")

    cuda_visible_devices = None
    for payload in (
        version_surface_payload,
        torch_cuda_memory_payload,
        nvidia_smi_active_payload,
    ):
        raw = _nested_get(payload or {}, "cuda_visible_devices")
        if isinstance(raw, str) and raw.strip():
            cuda_visible_devices = raw.strip()
            break

    num_gpus = _parse_int(_nested_get(effective_config, "num_gpus"))
    use_ddp = _parse_bool(_nested_get(effective_config, "use_ddp"))
    global_batch_size = _parse_int(_nested_get(effective_config, "global_batch_size"))
    gradient_accumulation_steps = _parse_int(
        _nested_get(effective_config, "gradient_accumulation_steps")
    )
    per_device_batch_size = _parse_int(_nested_get(effective_config, "per_device_batch_size"))
    effective_update_batch = _parse_int(
        _nested_get(effective_config, "effective_update_batch")
    )
    trainer_state_global_step = _parse_int(
        _nested_get(trainer_state_payload or {}, "global_step")
    )
    torchrun_invoked = runtime_log_scan["torchrun_invoked"]

    launch_policy = {
        "cuda_visible_devices": cuda_visible_devices,
        "num_gpus": num_gpus,
        "use_ddp": use_ddp,
        "torchrun_invoked": torchrun_invoked,
        "authoritative_runtime_log_path": runtime_log_scan["runtime_log_path"],
        "runtime_log_selection": runtime_log_selection,
        "summary_path": str(summary_path),
        "version_surface_path": (
            None if version_surface_payload is None else str(version_surface_path)
        ),
    }

    geometry_checks = {
        "global_batch_size": {
            "expected": EXPECTED_GLOBAL_BATCH_SIZE,
            "actual": global_batch_size,
            "pass": _check_exact(
                actual=global_batch_size,
                expected=EXPECTED_GLOBAL_BATCH_SIZE,
                missing_reason="effective_config.global_batch_size missing",
                mismatch_reason="global_batch_size != 4",
                blocking_reasons=blocking_reasons,
            ),
        },
        "gradient_accumulation_steps": {
            "expected": EXPECTED_GRADIENT_ACCUMULATION_STEPS,
            "actual": gradient_accumulation_steps,
            "pass": _check_exact(
                actual=gradient_accumulation_steps,
                expected=EXPECTED_GRADIENT_ACCUMULATION_STEPS,
                missing_reason="effective_config.gradient_accumulation_steps missing",
                mismatch_reason="gradient_accumulation_steps != 4",
                blocking_reasons=blocking_reasons,
            ),
        },
        "per_device_batch_size": {
            "expected": EXPECTED_PER_DEVICE_BATCH_SIZE,
            "actual": per_device_batch_size,
            "pass": _check_exact(
                actual=per_device_batch_size,
                expected=EXPECTED_PER_DEVICE_BATCH_SIZE,
                missing_reason="effective_config.per_device_batch_size missing",
                mismatch_reason="per_device_batch_size != 1",
                blocking_reasons=blocking_reasons,
            ),
        },
        "effective_update_batch": {
            "expected": EXPECTED_EFFECTIVE_UPDATE_BATCH,
            "actual": effective_update_batch,
            "pass": _check_exact(
                actual=effective_update_batch,
                expected=EXPECTED_EFFECTIVE_UPDATE_BATCH,
                missing_reason="effective_config.effective_update_batch missing",
                mismatch_reason="effective_update_batch != 4",
                blocking_reasons=blocking_reasons,
            ),
        },
    }

    launch_checks = {
        "cuda_visible_devices_check": {
            "expected": EXPECTED_CUDA_VISIBLE_DEVICES,
            "actual": cuda_visible_devices,
            "pass": _check_exact(
                actual=cuda_visible_devices,
                expected=EXPECTED_CUDA_VISIBLE_DEVICES,
                missing_reason="CUDA_VISIBLE_DEVICES missing",
                mismatch_reason='CUDA_VISIBLE_DEVICES != "1"',
                blocking_reasons=blocking_reasons,
            ),
        },
        "num_gpus_check": {
            "expected": EXPECTED_NUM_GPUS,
            "actual": num_gpus,
            "pass": _check_exact(
                actual=num_gpus,
                expected=EXPECTED_NUM_GPUS,
                missing_reason="effective_config.num_gpus missing",
                mismatch_reason="num_gpus != 1",
                blocking_reasons=blocking_reasons,
            ),
        },
        "use_ddp_check": {
            "expected": EXPECTED_USE_DDP,
            "actual": use_ddp,
            "pass": _check_exact(
                actual=use_ddp,
                expected=EXPECTED_USE_DDP,
                missing_reason="effective_config.use_ddp missing",
                mismatch_reason="use_ddp != false",
                blocking_reasons=blocking_reasons,
            ),
        },
        "torchrun_invoked_check": {
            "expected": EXPECTED_TORCHRUN_INVOKED,
            "actual": torchrun_invoked,
            "pass": _check_exact(
                actual=torchrun_invoked,
                expected=EXPECTED_TORCHRUN_INVOKED,
                missing_reason="torchrun_invoked missing",
                mismatch_reason="torchrun_invoked != false",
                blocking_reasons=blocking_reasons,
            ),
        },
    }

    checkpoint_dir_exists = checkpoint_path.is_dir()
    if not checkpoint_dir_exists:
        _append_reason(blocking_reasons, "checkpoint-8 missing")
    trainer_state_step_pass = _check_exact(
        actual=trainer_state_global_step,
        expected=EXPECTED_GLOBAL_STEP,
        missing_reason="trainer_state.global_step missing",
        mismatch_reason="trainer_state.global_step != 8",
        blocking_reasons=blocking_reasons,
    )
    checkpoint_checks = {
        "checkpoint_dir": str(checkpoint_path),
        "checkpoint_8_exists": checkpoint_dir_exists,
        "trainer_state_path": str(trainer_state_path),
        "trainer_state_global_step": trainer_state_global_step,
        "trainer_state_global_step_pass": trainer_state_step_pass,
        "selected_checkpoint_path": _nested_get(summary_payload or {}, "selected_checkpoint_path"),
        "selected_checkpoint_exists": _nested_get(summary_payload or {}, "selected_checkpoint_exists"),
    }

    failure_reason_by_token = {token: reason for token, _, reason in RUNTIME_FAILURE_PATTERNS}
    for hit in runtime_log_scan["failure_token_hits"]:
        reason = failure_reason_by_token.get(hit["token"])
        if reason is not None:
            _append_reason(blocking_reasons, reason)

    torch_cuda_memory_checks = {
        "path": None if torch_cuda_memory_payload is None else str(torch_cuda_memory_path),
        "available": False,
        "cuda_visible_devices": None,
        "device_count": None,
        "runtime_pid": None,
        "max_memory_allocated_bytes": None,
        "pass": None,
    }
    if torch_cuda_memory_payload is not None:
        torch_device_count = _parse_int(torch_cuda_memory_payload.get("device_count"))
        torch_cuda_visible_devices = torch_cuda_memory_payload.get("cuda_visible_devices")
        torch_cuda_memory_checks.update(
            {
                "available": bool(torch_cuda_memory_payload.get("available") is True),
                "cuda_visible_devices": torch_cuda_visible_devices,
                "device_count": torch_device_count,
                "runtime_pid": _parse_int(torch_cuda_memory_payload.get("runtime_pid")),
                "max_memory_allocated_bytes": _parse_int(
                    torch_cuda_memory_payload.get("max_memory_allocated_bytes")
                ),
                "pass": (
                    torch_cuda_memory_payload.get("available") is True
                    and torch_device_count == 1
                    and torch_cuda_visible_devices == EXPECTED_CUDA_VISIBLE_DEVICES
                ),
            }
        )

    nvidia_smi_checks = {
        "path": None if nvidia_smi_active_payload is None else str(nvidia_smi_active_path),
        "available": False,
        "cuda_visible_devices": None,
        "matching_runtime_pid_count": None,
        "active_compute_app_count": None,
        "pass": None,
    }
    if nvidia_smi_active_payload is not None:
        matching_entries = nvidia_smi_active_payload.get("matching_runtime_pid_entries")
        active_compute_apps = nvidia_smi_active_payload.get("active_compute_apps")
        nvidia_smi_checks.update(
            {
                "available": bool(nvidia_smi_active_payload.get("available") is True),
                "cuda_visible_devices": nvidia_smi_active_payload.get("cuda_visible_devices"),
                "matching_runtime_pid_count": (
                    len(matching_entries) if isinstance(matching_entries, Sequence) else None
                ),
                "active_compute_app_count": (
                    len(active_compute_apps) if isinstance(active_compute_apps, Sequence) else None
                ),
                "pass": (
                    nvidia_smi_active_payload.get("available") is True
                    and nvidia_smi_active_payload.get("cuda_visible_devices")
                    == EXPECTED_CUDA_VISIBLE_DEVICES
                ),
            }
        )

    runtime_checks = {
        "runtime_log_dir": str(runtime_log_dir),
        "runtime_log_dir_exists": runtime_log_dir.is_dir(),
        "available_runtime_logs": [str(path) for path in _discover_runtime_logs(runtime_log_dir)],
        "selected_runtime_log": runtime_log_scan["runtime_log_path"],
        "runtime_log_selection": runtime_log_selection,
        "torchrun_invoked": torchrun_invoked,
        "failure_token_hits": runtime_log_scan["failure_token_hits"],
        "torch_cuda_memory": torch_cuda_memory_checks,
        "nvidia_smi_active_sampling": nvidia_smi_checks,
    }

    launch_policy.update(launch_checks)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_NAME,
        "run_slug": smoke_output_dir.name,
        "pass": len(blocking_reasons) == 0,
        "unlock_signal": UNLOCK_SIGNAL,
        "launch_policy": launch_policy,
        "geometry_checks": geometry_checks,
        "checkpoint_checks": checkpoint_checks,
        "runtime_checks": runtime_checks,
        "blocking_reasons": blocking_reasons,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_verdict(
        smoke_output_dir=args.smoke_output_dir,
        runtime_log_dir=args.runtime_log_dir,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
