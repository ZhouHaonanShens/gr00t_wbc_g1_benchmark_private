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


def _stock_path_overrides(*, available: bool) -> dict[str, bool]:
    return {
        "path.submodules/Isaac-GR00T": available,
        "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py": available,
    }


def test_main_writes_numeric_stock_path_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"

    exit_code = interface_localization_numeric_gap.main(
        [
            "--output-dir",
            str(output_dir),
            "--path-mode",
            interface_localization_numeric_gap.STOCK_PATH_MODE,
        ]
    )

    assert exit_code == 0
    output_json = (
        output_dir / interface_localization_numeric_gap.NUMERIC_STOCK_PATH_JSON_NAME
    )
    assert output_json.is_file()

    payload = _read_json(output_json)
    assert payload["schema_version"] == (
        interface_localization_numeric_gap.NUMERIC_STOCK_PATH_SCHEMA_VERSION
    )
    assert payload["artifact_kind"] == (
        interface_localization_numeric_gap.NUMERIC_STOCK_PATH_ARTIFACT_KIND
    )
    assert payload["path_mode"] == interface_localization_numeric_gap.STOCK_PATH_MODE
    assert payload["backpointer"]["writer_script"] == (
        "work/recap/scripts/interface_localization_numeric_gap.py"
    )
    assert payload["backpointer"]["pytest_command"] == (
        "python3 -m pytest tests/recap/test_numeric_sentinel_stock_path.py -q"
    )
    assert payload["stock_path_evidence"]["reference_lines"]["stock_entrypoint"] == (
        interface_localization_numeric_gap.STOCK_ENTRYPOINT_REFERENCE_LINES
    )


def test_builder_supports_stock_available_mode_with_purity_gate_evidence(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    output_json = (
        output_dir / interface_localization_numeric_gap.NUMERIC_STOCK_PATH_JSON_NAME
    )
    payload = interface_localization_numeric_gap.build_numeric_stock_path_payload(
        REPO_ROOT,
        output_dir=output_dir,
        output_json=output_json,
        availability_overrides=_stock_path_overrides(available=True),
    )

    assert payload["dependency_context"]["path.submodules/Isaac-GR00T"]["status"] == (
        "survived"
    )
    assert (
        payload["dependency_context"][
            "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"
        ]["status"]
        == "survived"
    )
    assert (
        payload["dependency_context"]["stock_mainline_server_entrypoint"]["status"]
        == "survived"
    )
    assert payload["summary"]["rows_by_status"] == {"survived": 8}
    assert payload["summary"]["blocked_row_count"] == 0
    assert len(payload["summary"]["sentinel_records"]) == 2

    for record in payload["summary"]["sentinel_records"]:
        assert record["sentinel_id"].startswith("stock_mainline_numeric_sentinel__")
        assert record["watch_bucket"] == interface_localization_numeric_gap.WATCH_BUCKET
        assert record["baseline_tuple_digest"] == payload["baseline_tuple_digest"]
        assert record["boundary_status"] == {
            "collector_policy_callsite": "survived",
            "policy_input_collation": "survived",
            "model_condition_injection": "survived",
            "policy_output_action": "survived",
        }

    collector_rows = _rows_by_boundary(payload, "collector_policy_callsite")
    collation_rows = _rows_by_boundary(payload, "policy_input_collation")
    model_rows = _rows_by_boundary(payload, "model_condition_injection")
    action_rows = _rows_by_boundary(payload, "policy_output_action")

    assert {row["field_name"] for row in collector_rows} == {
        "stock_path.numeric_sentinel_probe"
    }
    assert {row["field_name"] for row in collation_rows} == {"server_provenance_probe"}
    assert {row["field_name"] for row in model_rows} == {"provenance.purity_gate"}
    assert {row["field_name"] for row in action_rows} == {
        "branch_aware_mainline_decision"
    }

    model_payload = json.loads(model_rows[0]["value_repr"])
    assert model_payload["purity_mode"] == "mainline_no_overlay"
    assert model_payload["required_fields"]["overlay_from"] is None
    assert model_payload["required_fields"]["task_text_field"] == "prompt_raw"
    assert model_payload["custom_counterexample_reference_lines"] == (
        interface_localization_numeric_gap.CUSTOM_PROVENANCE_COUNTEREXAMPLE_REFERENCE_LINES
    )
    assert model_payload["rejected_custom_markers"]["adv_embedding_from"] is None

    action_payload = json.loads(action_rows[0]["value_repr"])
    assert action_payload["selected_branch"] == "unitree_g1"
    assert action_payload["selected_next_step"] == (
        "audit_recap_injection_action_target_and_relative_action_interpretation"
    )
    assert action_payload["recommended_next_step_by_branch"]["unitree_g1"] == (
        "audit_recap_injection_action_target_and_relative_action_interpretation"
    )


def test_builder_marks_all_stock_boundaries_blocked_when_upstream_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    output_json = (
        output_dir / interface_localization_numeric_gap.NUMERIC_STOCK_PATH_JSON_NAME
    )
    payload = interface_localization_numeric_gap.build_numeric_stock_path_payload(
        REPO_ROOT,
        output_dir=output_dir,
        output_json=output_json,
        availability_overrides=_stock_path_overrides(available=False),
    )

    written_path = interface_localization_numeric_gap.write_artifact(
        output_json=output_json,
        payload=payload,
    )

    assert written_path.is_file()
    materialized = _read_json(written_path)
    assert materialized == payload
    assert materialized["summary"]["rows_by_status"] == {"blocked_missing_upstream": 8}
    assert materialized["summary"]["blocked_row_count"] == 8
    assert len(materialized["blocked_surfaces"]) == 8

    missing_entrypoint_path = str(
        (
            REPO_ROOT
            / interface_localization_numeric_gap.STOCK_ENTRYPOINT_RELATIVE_PATH
        ).resolve()
    )
    for boundary_name in interface_localization_numeric_gap.BOUNDARY_ORDER:
        rows = _rows_by_boundary(materialized, boundary_name)
        assert {row["status"] for row in rows} == {"blocked_missing_upstream"}
        assert all(row["blocked_reason"] for row in rows)
        assert all(missing_entrypoint_path in row["blocked_reason"] for row in rows)


def test_builder_rejects_custom_provenance_for_stock_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    output_json = (
        output_dir / interface_localization_numeric_gap.NUMERIC_STOCK_PATH_JSON_NAME
    )
    custom_like_provenance = {
        "advantage_contract_version": "full_recap_continuous_adv_v2",
        "advantage_injection_rule": interface_localization_numeric_gap.ADVANTAGE_INJECTION_RULE,
        "adv_embedding_from": "/tmp/custom_adv_embedding.pt",
        "legacy_negate_enabled": False,
        "overlay_from": None,
        "require_advantage_embedding": True,
        "task_text_field": "prompt_raw",
    }

    with pytest.raises(ValueError, match="custom_adv_embedding_marker_forbidden"):
        interface_localization_numeric_gap.build_numeric_stock_path_payload(
            REPO_ROOT,
            output_dir=output_dir,
            output_json=output_json,
            availability_overrides=_stock_path_overrides(available=True),
            runtime_provenance_override=custom_like_provenance,
        )
