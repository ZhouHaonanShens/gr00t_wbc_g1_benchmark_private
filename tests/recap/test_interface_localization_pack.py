from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


interface_localization_contract = importlib.import_module(
    "work.recap.scripts.interface_localization_contract"
)
interface_localization_pack = importlib.import_module(
    "work.recap.scripts.interface_localization_pack"
)


STATIC_GENERATED_AT = "2026-03-28T21:15:08+00:00"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        interface_localization_pack.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_pack_and_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    evidence_json = tmp_path / "task-9-interface-localization-pack.json"

    exit_code = interface_localization_pack.main(
        [
            "--output-dir",
            str(output_dir),
            "--evidence-json",
            str(evidence_json),
        ]
    )

    assert exit_code == 0
    pack = _read_json(output_dir / interface_localization_pack.PACK_JSON_NAME)
    evidence = _read_json(evidence_json)

    assert pack["schema_version"] == interface_localization_pack.PACK_SCHEMA_VERSION
    assert pack["artifact_kind"] == interface_localization_pack.PACK_ARTIFACT_KIND
    assert pack["task_code"] == "T9"
    assert pack[
        "baseline_tuple_digest"
    ] == interface_localization_pack._baseline_tuple_digest(
        interface_localization_contract.build_interface_localization_contract()
    )
    assert [
        entry["boundary_name"] for entry in pack["final_boundary_statuses"]
    ] == list(interface_localization_contract.BOUNDARY_ORDER)
    assert any(
        entry["boundary_name"] == "server_policy_adapter"
        and entry["final_status"] == "blocked_missing_upstream"
        for entry in pack["final_boundary_statuses"]
    )
    assert any(
        entry["boundary_name"] == "dex3_finger_hand_path"
        and entry["final_status"] == "blocked_missing_upstream"
        for entry in pack["final_boundary_statuses"]
    )
    assert pack["generated_outputs"]["response_summary_layer"]["updated"] is False
    assert any(
        item["artifact_id"] == "task8_right_hand_split_audit"
        for item in pack["key_supporting_artifacts"]
    )
    assert any(
        item["finding_code"] == "missing_gr00t_module_blocks_custom_runtime_visibility"
        for item in pack["blocker_findings"]
    )
    assert (
        evidence["schema_version"]
        == interface_localization_pack.TASK9_EVIDENCE_SCHEMA_VERSION
    )
    assert (
        evidence["artifact_kind"]
        == interface_localization_pack.TASK9_EVIDENCE_ARTIFACT_KIND
    )
    assert evidence["generated_outputs"]["interface_localization_pack"]["path"] == str(
        output_dir / interface_localization_pack.PACK_JSON_NAME
    )


def test_build_pack_rejects_missing_required_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required input file"):
        interface_localization_pack.build_interface_localization_pack(
            REPO_ROOT,
            input_dir=tmp_path / "missing_inputs",
            output_dir=tmp_path / "out",
            output_json=tmp_path / "out" / interface_localization_pack.PACK_JSON_NAME,
            runtime_log_dir=tmp_path / "runtime",
            evidence_json=tmp_path / "task-9-interface-localization-pack.json",
            generated_at=STATIC_GENERATED_AT,
        )


def test_builder_summarizes_current_repo_state() -> None:
    output_dir = REPO_ROOT / "agent" / "artifacts" / "interface_localization_sprint"
    evidence_json = (
        REPO_ROOT / ".sisyphus" / "evidence" / "task-9-interface-localization-pack.json"
    )
    runtime_log_dir = (
        REPO_ROOT / "agent" / "runtime_logs" / "interface_localization_sprint"
    )

    pack = interface_localization_pack.build_interface_localization_pack(
        REPO_ROOT,
        input_dir=output_dir,
        output_dir=output_dir,
        output_json=output_dir / interface_localization_pack.PACK_JSON_NAME,
        runtime_log_dir=runtime_log_dir,
        evidence_json=evidence_json,
        generated_at=STATIC_GENERATED_AT,
    )

    assert (
        pack["baseline_tuple_digest"]
        == "fef4b961cb30c0b02a1ced735fd0c3d69bcab4d438af95065758679f98b2a9aa"
    )
    assert pack["blocked_surface_summary"]["blocked_surface_count"] == 7
    assert pack["blocked_surface_summary"]["blocked_surface_count_by_source"] == {
        "agent/artifacts/interface_localization_sprint/conditional_blockers.json": 2,
        "agent/artifacts/interface_localization_sprint/recap_numeric_custom_path.json": 4,
        "agent/artifacts/interface_localization_sprint/right_arm_vs_right_hand_split_audit.json": 1,
    }
    assert pack["recommended_next_step"]["priority_blockers"] == [
        "python_module.gr00t",
        "dex3_finger_hand_path.upstream_source_route",
    ]
    by_name = {
        entry["boundary_name"]: entry for entry in pack["final_boundary_statuses"]
    }
    assert by_name["collector_policy_callsite"]["final_status"] == "survived"
    assert (
        by_name["server_policy_adapter"]["final_status"] == "blocked_missing_upstream"
    )
    assert (
        by_name["dex3_finger_hand_path"]["final_status"] == "blocked_missing_upstream"
    )
