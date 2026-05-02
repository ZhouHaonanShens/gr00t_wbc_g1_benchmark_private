from __future__ import annotations

from pathlib import Path

from work.openpi.contracts import OpenPIRuntimePaths

from . import bridge


DEFAULT_HOST = bridge.DEFAULT_HOST
DEFAULT_PORT = bridge.DEFAULT_PORT
NUM_STEPS_WAIT = bridge.NUM_STEPS_WAIT
LIBERO_NATIVE_SMOKE_ENTRY = bridge.LIBERO_NATIVE_SMOKE_ENTRY
FailFastError = bridge.FailFastError


def build_runtime_paths(
    *,
    topic: str = bridge.TOPIC,
    evidence_path: Path | None = None,
    artifact_root: Path | None = None,
    runtime_root: Path | None = None,
) -> OpenPIRuntimePaths:
    return bridge._required_paths(
        topic=topic,
        evidence_path=evidence_path,
        artifact_root=artifact_root,
        runtime_root=runtime_root,
    )


def run_stock_smoke_harness(
    args: object,
    *,
    paths: OpenPIRuntimePaths | None = None,
) -> int:
    return bridge._run_harness(args, paths=paths)


def pick_free_port(host: str, start_port: int) -> int:
    return bridge._pick_free_port(host, start_port)


def prepare_libero_config_dir(openpi_root: Path, runtime_dir: Path) -> Path:
    return bridge._prepare_libero_config_dir(openpi_root, runtime_dir)


def max_steps_for_task_suite(task_suite_name: str) -> int:
    return bridge._get_max_steps(task_suite_name)


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "FailFastError",
    "LIBERO_NATIVE_SMOKE_ENTRY",
    "NUM_STEPS_WAIT",
    "build_runtime_paths",
    "max_steps_for_task_suite",
    "pick_free_port",
    "prepare_libero_config_dir",
    "run_stock_smoke_harness",
]

