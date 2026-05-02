from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import interface_localization_text_rewrite_map


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        interface_localization_text_rewrite_map.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_rewrite_map_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"

    exit_code = interface_localization_text_rewrite_map.main(
        ["--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    payload = _read_json(
        output_dir / interface_localization_text_rewrite_map.TEXT_REWRITE_MAP_JSON_NAME
    )

    assert (
        payload
        == interface_localization_text_rewrite_map.build_text_source_and_rewrite_map(
            REPO_ROOT,
            output_dir=output_dir,
        )
    )
    assert (
        payload["schema_version"]
        == interface_localization_text_rewrite_map.TEXT_REWRITE_MAP_SCHEMA_VERSION
    )
    assert (
        payload["artifact_kind"]
        == interface_localization_text_rewrite_map.TEXT_REWRITE_MAP_ARTIFACT_KIND
    )
    assert payload["provenance_class"] == "static"
    assert payload["backpointer"]["writer_script"] == (
        "work/recap/scripts/interface_localization_text_rewrite_map.py"
    )


def test_rewrite_map_explicitly_distinguishes_consumption_surfaces() -> None:
    payload = interface_localization_text_rewrite_map.build_text_source_and_rewrite_map(
        REPO_ROOT
    )

    assert payload["summary"]["numeric_mainline_consumes"] == "prompt_raw"
    assert payload["summary"]["numeric_mainline_consumes_compat_alias_for"] == (
        "authoritative_text_carrier_source_field"
    )
    assert payload["summary"]["exporter_default_consumes"] == "carrier_text_v1"
    assert payload["summary"]["exporter_default_authority_status"] == (
        "mainline_authority"
    )
    assert payload["summary"]["runtime_override_surface"] == (
        "annotation.human.task_description"
    )
    assert payload["stage_order"] == [
        "collector.prompt_raw",
        "collector.prompt_conditioned",
        "labeler.prompt_conditioned",
        "export.task_text_field",
        "export.dual_task_text",
        "runtime_override.annotation.human.task_description",
        "text_indicator_policy.prompt_raw",
    ]

    collector_raw = payload["stages"]["collector.prompt_raw"]["semantic_selection"]
    assert collector_raw["source_field"] == "normalize_prompt(obs)"
    assert collector_raw["selected_field"] == "prompt_raw"

    collector_conditioned = payload["stages"]["collector.prompt_conditioned"][
        "semantic_selection"
    ]
    assert collector_conditioned["source_field"] == "prompt_raw"
    assert collector_conditioned["effective_consumed_text"] == "prompt_conditioned"
    assert collector_conditioned["live_model_consumption_proven"] is True

    labeler_conditioned = payload["stages"]["labeler.prompt_conditioned"][
        "semantic_selection"
    ]
    assert labeler_conditioned["source_field"] == "prompt_raw"
    assert "advantage positive" in labeler_conditioned["rewrite_rule"]
    assert labeler_conditioned["live_model_consumption_proven"] is False

    export_single = payload["stages"]["export.task_text_field"]["semantic_selection"]
    assert export_single["selected_field"] == "carrier_text_v1"
    assert export_single["surface_authority_key"] == "carrier_text_v1"
    assert export_single["authority_status"] == "mainline_authority"
    assert export_single["effective_consumed_text_surface"] == (
        "exported dataset single-task-text surface"
    )

    export_dual = payload["stages"]["export.dual_task_text"]["semantic_selection"]
    assert export_dual["selected_field"] == "dual_task_text"
    assert export_dual["effective_consumed_text"] == "prompt_raw + prompt_conditioned"

    runtime_override = payload["stages"][
        "runtime_override.annotation.human.task_description"
    ]["semantic_selection"]
    assert runtime_override["selected_field"] == "annotation.human.task_description"
    assert runtime_override["effective_consumed_text_surface"] == (
        "annotation.human.task_description"
    )

    text_indicator = payload["stages"]["text_indicator_policy.prompt_raw"][
        "semantic_selection"
    ]
    assert text_indicator["source_field"] == "observation.language[self.language_key]"
    assert text_indicator["selected_field"] == "prompt_raw"
    assert text_indicator["effective_consumed_text"] == "canonical_text_from_prompt_raw"


def test_rewrite_map_reuses_contract_vocabulary_and_records_mismatch() -> None:
    payload = interface_localization_text_rewrite_map.build_text_source_and_rewrite_map(
        REPO_ROOT
    )

    assert payload["input_baseline_summary"]["status_allowlist"] == [
        "survived",
        "died",
        "mutated",
        "rerouted",
        "bypassed",
        "blocked_missing_upstream",
    ]
    assert payload["input_baseline_summary"]["provenance_class_allowlist"] == [
        "static",
        "synthetic",
        "replay_live",
        "server_live",
    ]

    for stage_name in payload["stage_order"]:
        stage = payload["stages"][stage_name]
        assert stage["status"] == "survived"
        assert stage["provenance_class"] == "static"
        assert stage["references"]
        semantic = stage["semantic_selection"]
        assert semantic["source_field"]
        assert semantic["rewrite_rule"]
        assert semantic["selected_field"]
        assert semantic["effective_consumed_text_surface"]

    mismatch = payload["relationships"][0]
    assert mismatch["relationship_name"] == "numeric_mainline_vs_exporter_default"
    assert mismatch["display_relationship_name"] == (
        "authoritative_source_field_vs_exporter_default"
    )
    assert mismatch["left_value"] == "prompt_raw"
    assert mismatch["right_value"] == "carrier_text_v1"
    assert mismatch["relationship"] == "shared_mainline_authority_via_carrier_text_v1"
