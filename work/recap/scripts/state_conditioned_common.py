from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from work.recap import state_conditioned_bucket_a_import


DEFAULT_BUCKET_DIR = Path("agent/artifacts/state_conditioned_materialization/bucket_a")
DEFAULT_DEV_DIR = Path("agent/artifacts/state_conditioned_materialization/devbench")
DEFAULT_COLLECTION_DIR = Path(
    "agent/artifacts/state_conditioned_materialization/collection"
)
DEFAULT_HARVEST_DIR = Path("agent/artifacts/state_conditioned_materialization/harvest")


def exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def validate_existing_dir(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"{arg_name} directory does not exist: {resolved}")
    return resolved


def validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    return state_conditioned_bucket_a_import._write_jsonl(path, records)


def read_json(path: Path) -> dict[str, Any]:
    return state_conditioned_bucket_a_import._read_json(path)


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    return state_conditioned_bucket_a_import._read_jsonl_dicts(path)


__all__ = [
    "DEFAULT_BUCKET_DIR",
    "DEFAULT_COLLECTION_DIR",
    "DEFAULT_DEV_DIR",
    "DEFAULT_HARVEST_DIR",
    "exception_message",
    "read_json",
    "read_jsonl_dicts",
    "validate_existing_dir",
    "validate_existing_file",
    "validate_output_dir",
    "write_json",
    "write_jsonl",
]
