from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import interface_localization_numeric_gap


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_boundary(
    payload: dict[str, Any], boundary_name: str
) -> list[dict[str, Any]]:
    return [
        dict(row) for row in payload["rows"] if row["boundary_name"] == boundary_name
    ]


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        interface_localization_numeric_gap.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_numeric_custom_path_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"

    exit_code = interface_localization_numeric_gap.main(
        [
            "--output-dir",
            str(output_dir),
            "--path-mode",
            "custom_adv",
        ]
    )

    assert exit_code == 0
    output_json = (
        output_dir / interface_localization_numeric_gap.NUMERIC_CUSTOM_PATH_JSON_NAME
    )
    assert output_json.is_file()

    payload = _read_json(output_json)
    assert payload["schema_version"] == (
        interface_localization_numeric_gap.NUMERIC_CUSTOM_PATH_SCHEMA_VERSION
    )
    assert payload["artifact_kind"] == (
        interface_localization_numeric_gap.NUMERIC_CUSTOM_PATH_ARTIFACT_KIND
    )
    assert payload["path_mode"] == "custom_adv"
    assert payload["backpointer"]["writer_script"] == (
        "work/recap/scripts/interface_localization_numeric_gap.py"
    )
    assert payload["backpointer"]["pytest_command"] == (
        "python3 -m pytest tests/recap/test_numeric_sentinel_custom_path.py -q"
    )
    for boundary_name in (
        "collector_policy_callsite",
        "policy_input_collation",
        "model_condition_injection",
        "policy_output_action",
    ):
        assert boundary_name in payload["summary"]["rows_by_boundary"]


def test_builder_supports_success_mode_with_deterministic_sentinel_summary(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    output_json = (
        output_dir / interface_localization_numeric_gap.NUMERIC_CUSTOM_PATH_JSON_NAME
    )
    payload = interface_localization_numeric_gap.build_numeric_custom_path_payload(
        REPO_ROOT,
        output_dir=output_dir,
        output_json=output_json,
        availability_overrides={"python_module.gr00t": True},
    )

    assert payload["dependency_context"]["python_module.gr00t"]["status"] == "survived"
    assert (
        payload["dependency_context"]["custom_advantage_aware_server_cli"]["status"]
        == "survived"
    )
    assert payload["summary"]["rows_by_status"] == {"survived": 8}
    assert payload["summary"]["blocked_row_count"] == 0
    assert payload["baseline_tuple_digest"]
    assert len(payload["summary"]["sentinel_records"]) == 2

    for record in payload["summary"]["sentinel_records"]:
        assert record["sentinel_id"].startswith("custom_adv_numeric_sentinel__")
        assert record["watch_bucket"] == "body_wrist_upper_limb_chain"
        assert record["seed"] == 0
        assert record["condition_label"] in {"SEARCH_NOMINAL", "SEARCH_RECOVERY"}
        assert record["baseline_tuple_digest"] == payload["baseline_tuple_digest"]
        assert record["window_description"] == (
            interface_localization_numeric_gap.SENTINEL_WINDOW_DESCRIPTION
        )
        assert record["amplitude_description"] == (
            interface_localization_numeric_gap.SENTINEL_AMPLITUDE_DESCRIPTION
        )
        assert record["boundary_status"] == {
            "collector_policy_callsite": "survived",
            "policy_input_collation": "survived",
            "model_condition_injection": "survived",
            "policy_output_action": "survived",
        }

    model_rows = _rows_by_boundary(payload, "model_condition_injection")
    action_rows = _rows_by_boundary(payload, "policy_output_action")
    assert {row["status"] for row in model_rows} == {"survived"}
    assert {row["status"] for row in action_rows} == {"survived"}
    assert all(row["blocked_reason"] == "" for row in model_rows + action_rows)


def test_builder_keeps_artifact_and_marks_blocked_missing_upstream_when_gr00t_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    output_json = (
        output_dir / interface_localization_numeric_gap.NUMERIC_CUSTOM_PATH_JSON_NAME
    )
    payload = interface_localization_numeric_gap.build_numeric_custom_path_payload(
        REPO_ROOT,
        output_dir=output_dir,
        output_json=output_json,
        availability_overrides={"python_module.gr00t": False},
    )

    written_path = interface_localization_numeric_gap.write_artifact(
        output_json=output_json,
        payload=payload,
    )

    assert written_path.is_file()
    materialized = _read_json(written_path)
    assert materialized == payload
    assert materialized["dependency_context"]["python_module.gr00t"]["status"] == (
        "blocked_missing_upstream"
    )
    assert materialized["dependency_context"]["python_module.gr00t"][
        "missing_modules"
    ] == ["gr00t"]
    assert materialized["summary"]["blocked_row_count"] == 4
    assert materialized["summary"]["rows_by_status"] == {
        "blocked_missing_upstream": 4,
        "survived": 4,
    }

    collector_rows = _rows_by_boundary(materialized, "collector_policy_callsite")
    collation_rows = _rows_by_boundary(materialized, "policy_input_collation")
    model_rows = _rows_by_boundary(materialized, "model_condition_injection")
    action_rows = _rows_by_boundary(materialized, "policy_output_action")

    assert {row["status"] for row in collector_rows} == {"survived"}
    assert {row["status"] for row in collation_rows} == {"survived"}
    assert {row["status"] for row in model_rows} == {"blocked_missing_upstream"}
    assert {row["status"] for row in action_rows} == {"blocked_missing_upstream"}
    assert all(row["blocked_reason"] for row in model_rows + action_rows)
