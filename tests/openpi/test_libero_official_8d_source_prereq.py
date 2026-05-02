from __future__ import annotations

from collections.abc import Callable
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "agent/exchange/openpi_libero_official_8d_source_prereq_v1.md"
VALIDATOR = REPO_ROOT / "work/openpi/scripts/validate_libero_official_8d_source.py"
OFFICIAL_DIR = (
    REPO_ROOT
    / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_validator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_libero_official_8d_source", VALIDATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load validator module from {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR_MODULE = _load_validator_module()
BLOCKED_EXIT_CODE = cast(int, getattr(VALIDATOR_MODULE, "BLOCKED_EXIT_CODE"))
CONTRACT_REF = cast(str, getattr(VALIDATOR_MODULE, "CONTRACT_REF"))
INCOMPLETE_BLOCKER_CODE = cast(
    str, getattr(VALIDATOR_MODULE, "INCOMPLETE_BLOCKER_CODE")
)
MISSING_BLOCKER_CODE = cast(str, getattr(VALIDATOR_MODULE, "MISSING_BLOCKER_CODE"))
build_source_prereq_report = cast(
    Callable[[str | Path], dict[str, object]],
    getattr(VALIDATOR_MODULE, "build_source_prereq_report"),
)
main = cast(Callable[[list[str] | None], int], getattr(VALIDATOR_MODULE, "main"))


def _read_json_object(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def test_source_prereq_contract_freezes_canonical_path_and_stop_rule() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO official/native 8D source prerequisite 合同 v1",
        "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d/",
        "tasks 5–10 的唯一 canonical demo source prerequisite",
        '如果 validator 结果不是 `status="ready"`，则 downstream tasks 必须立即停止',
        "missing_official_native_8d_source",
        "incomplete_official_native_8d_source",
    ]
    for item in required:
        assert item in text, f"missing source prerequisite contract item: {item}"


def test_validator_reports_ready_for_canonical_official_source(tmp_path: Path) -> None:
    report = build_source_prereq_report(OFFICIAL_DIR)
    assert report["status"] == "ready"
    assert report["dataset_dir"] == str(OFFICIAL_DIR.resolve())
    assert report["contract_ref"] == CONTRACT_REF
    assert report["blocker_code"] is None
    assert cast(int, report["sample_parquet_count"]) >= 1

    result_path = tmp_path / "official_8d_source_check.json"
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--dataset-dir",
            str(OFFICIAL_DIR),
            "--out",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    written = _read_json_object(result_path)
    assert written["status"] == "ready"
    assert cast(int, written["sample_parquet_count"]) >= 1
    assert written["contract_ref"] == CONTRACT_REF


def test_validator_fails_fast_with_missing_source_blocker(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"
    result_path = tmp_path / "missing_source_check.json"
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--dataset-dir",
            str(missing_dir),
            "--out",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == BLOCKED_EXIT_CODE
    assert MISSING_BLOCKER_CODE in result.stderr

    written = _read_json_object(result_path)
    assert written["status"] == "blocked"
    assert written["blocker_code"] == MISSING_BLOCKER_CODE
    assert written["sample_parquet_count"] == 0


def test_validator_fails_fast_with_incomplete_source_blocker(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "official_native_8d_partial"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    _ = (meta_dir / "info.json").write_text("{}\n", encoding="utf-8")

    report_path = tmp_path / "partial_source_check.json"
    rc = main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--out",
            str(report_path),
        ]
    )

    assert rc == BLOCKED_EXIT_CODE
    written = _read_json_object(report_path)
    assert written["status"] == "blocked"
    assert written["blocker_code"] == INCOMPLETE_BLOCKER_CODE
    assert written["sample_parquet_count"] == 0
