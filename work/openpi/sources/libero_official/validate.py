#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TextIO, cast


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d"
)
DEFAULT_OUT = (
    REPO_ROOT / "agent/artifacts/openpi_recap_v1/official_8d_source_check.json"
)
CONTRACT_REF = "agent/exchange/openpi_libero_official_8d_source_prereq_v1.md"
BLOCKED_EXIT_CODE = 42
MISSING_BLOCKER_CODE = "missing_official_native_8d_source"
INCOMPLETE_BLOCKER_CODE = "incomplete_official_native_8d_source"
REQUIRED_RELATIVE_FILES: tuple[str, ...] = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the canonical LIBERO official/native 8D dataset source is present and complete."
        )
    )
    _ = parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset root to validate.",
    )
    _ = parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help="Optional JSON output path.",
    )
    return parser


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    return resolved.resolve()


def _required_files_report(dataset_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "relative_path": relative_path,
            "exists": bool((dataset_dir / relative_path).is_file()),
        }
        for relative_path in REQUIRED_RELATIVE_FILES
    ]


def _episode_parquet_files(dataset_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet")))


def build_source_prereq_report(dataset_dir: str | Path) -> dict[str, object]:
    resolved_dir = _resolve_path(str(dataset_dir))
    required_files = _required_files_report(resolved_dir)

    report: dict[str, object] = {
        "status": "blocked",
        "dataset_dir": str(resolved_dir),
        "required_files": required_files,
        "sample_parquet_count": 0,
        "contract_ref": CONTRACT_REF,
        "blocker_code": None,
    }

    if not resolved_dir.is_dir():
        report["blocker_code"] = MISSING_BLOCKER_CODE
        return report

    parquet_count = len(_episode_parquet_files(resolved_dir))
    report["sample_parquet_count"] = parquet_count

    if (
        any(not cast(bool, item["exists"]) for item in required_files)
        or parquet_count < 1
    ):
        report["blocker_code"] = INCOMPLETE_BLOCKER_CODE
        return report

    report["status"] = "ready"
    report["blocker_code"] = None
    return report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        _ = handle.write("\n")
    _ = tmp.replace(path)


def _emit_report(report: dict[str, object], *, stream: TextIO) -> None:
    json.dump(report, stream, ensure_ascii=True, sort_keys=True)
    _ = stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = _resolve_path(cast(str, args.dataset_dir))
    report = build_source_prereq_report(dataset_dir)

    out_raw = cast(str, args.out)
    if out_raw:
        _write_json(_resolve_path(out_raw), report)

    if report["status"] != "ready":
        _emit_report(report, stream=sys.stderr)
        return BLOCKED_EXIT_CODE

    _emit_report(report, stream=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
