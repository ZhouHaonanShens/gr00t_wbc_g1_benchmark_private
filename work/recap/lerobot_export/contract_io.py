from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(
            f"Invalid JSON object in {path}: expected object, got {type(obj).__name__}"
        )
    return obj


def write_json_object(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=4, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Invalid record type in {path} at line {lineno}: expected object, got {type(obj).__name__}"
                )
            out.append(obj)
    return out


def write_jsonl_objects(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True))
            f.write("\n")
    tmp.replace(path)


def as_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected int-like, got bool ({context})")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Expected integer-valued number, got {value!r} ({context})"
            )
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(f"Expected int-like str, got {value!r} ({context})") from e
    raise ValueError(f"Expected int-like, got {type(value).__name__} ({context})")
