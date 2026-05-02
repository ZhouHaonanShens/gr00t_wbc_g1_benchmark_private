from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import interface_localization_trace


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        interface_localization_trace.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_interface_trace_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    runtime_log_dir = tmp_path / "runtime_logs"
    evidence_json = tmp_path / "task-5-interface-trace.json"

    exit_code = interface_localization_trace.main(
        [
            "--output-dir",
            str(output_dir),
            "--runtime-log-dir",
            str(runtime_log_dir),
            "--evidence-json",
            str(evidence_json),
        ]
    )

    assert exit_code == 0
    trace_csv_path = output_dir / interface_localization_trace.INTERFACE_TRACE_CSV_NAME
    summary_json_path = (
        output_dir / interface_localization_trace.RESPONSE_SUMMARY_JSON_NAME
    )
    runtime_log_json_path = (
        runtime_log_dir / interface_localization_trace.TRACE_RUNTIME_LOG_JSON_NAME
    )

    assert trace_csv_path.is_file()
    assert summary_json_path.is_file()
    assert runtime_log_json_path.is_file()
    assert evidence_json.is_file()

    summary_payload = _read_json(summary_json_path)
    evidence_payload = _read_json(evidence_json)
    fieldnames, rows = _read_csv_rows(trace_csv_path)

    expected_payload = interface_localization_trace.build_interface_trace_payload(
        REPO_ROOT,
        output_dir=output_dir,
        runtime_log_dir=runtime_log_dir,
        summary_json=summary_json_path,
        replay_fixture_path=runtime_log_dir / "interface_trace_fixture.npz",
    )

    assert fieldnames == list(interface_localization_trace.TRACE_CSV_FIELDNAMES)
    assert summary_payload == expected_payload["response_summary"]
    assert summary_payload["schema_version"] == (
        interface_localization_trace.RESPONSE_SUMMARY_SCHEMA_VERSION
    )
    assert summary_payload["artifact_kind"] == (
        interface_localization_trace.RESPONSE_SUMMARY_ARTIFACT_KIND
    )
    assert summary_payload["backpointer"]["writer_script"] == (
        "work/recap/scripts/interface_localization_trace.py"
    )
    assert summary_payload["backpointer"]["interface_trace_csv"] == str(trace_csv_path)
    assert summary_payload["backpointer"]["runtime_log_dir"] == str(runtime_log_dir)
    assert evidence_payload["backpointer"]["response_summary_json"] == str(
        summary_json_path
    )
    assert len(rows) == summary_payload["summary"]["row_count"]


def test_trace_harness_keeps_field_set_minimal_and_watchlist_scoped(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    runtime_log_dir = tmp_path / "runtime_logs"
    payload = interface_localization_trace.build_interface_trace_payload(
        REPO_ROOT,
        output_dir=output_dir,
        runtime_log_dir=runtime_log_dir,
        summary_json=output_dir
        / interface_localization_trace.RESPONSE_SUMMARY_JSON_NAME,
        replay_fixture_path=runtime_log_dir / "interface_trace_fixture.npz",
    )

    rows = payload["trace_rows"]
    assert rows
    expected_field_names = {
        "condition_text",
        "recap_value",
        "obs_body_q",
        "obs_body_dq",
        "motion_ref",
        "upper_body_target",
        "raw_action_norm",
        "q_target",
        "q_measured",
        "q_error",
    }
    assert {str(row["field_name"]) for row in rows} == expected_field_names
    assert {str(row["watch_bucket"]) for row in rows} == {
        "body_wrist_upper_limb_chain",
        "dex3_finger_hand_path",
    }
    for row in rows:
        assert set(row.keys()) == set(interface_localization_trace.TRACE_CSV_FIELDNAMES)
        assert str(row["boundary_name"])
        assert str(row["status"])
        assert str(row["provenance_class"])
        assert str(row["condition_label"])
        assert str(row["watch_bucket"])
        value_repr = str(row["value_repr"])
        if str(row["field_name"]) in {
            "obs_body_q",
            "obs_body_dq",
            "motion_ref",
            "upper_body_target",
        }:
            preview = json.loads(value_repr)
            assert preview["count"] == 7
            assert len(preview["preview"]) <= 3


def test_trace_harness_emits_explicit_blocked_rows_for_unavailable_surfaces(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    runtime_log_dir = tmp_path / "runtime_logs"
    payload = interface_localization_trace.build_interface_trace_payload(
        REPO_ROOT,
        output_dir=output_dir,
        runtime_log_dir=runtime_log_dir,
        summary_json=output_dir
        / interface_localization_trace.RESPONSE_SUMMARY_JSON_NAME,
        replay_fixture_path=runtime_log_dir / "interface_trace_fixture.npz",
    )

    rows = payload["trace_rows"]
    blocked_rows = [
        row for row in rows if str(row["status"]) == "blocked_missing_upstream"
    ]

    assert blocked_rows
    assert payload["response_summary"]["summary"]["blocked_row_count"] == len(
        blocked_rows
    )
    assert payload["response_summary"]["summary"]["blocked_field_names"] == [
        "q_target",
        "q_measured",
        "q_error",
    ]
    assert {str(row["field_name"]) for row in blocked_rows} == set(
        interface_localization_trace.BLOCKED_FIELD_NAMES
    )
    for row in blocked_rows:
        assert str(row["blocked_reason"])
        assert str(row["provenance_class"]) == "server_live"
    assert payload["response_summary"]["summary"]["blocked_surface_count_by_field"] == {
        "q_target": 4,
        "q_measured": 4,
        "q_error": 4,
    }
