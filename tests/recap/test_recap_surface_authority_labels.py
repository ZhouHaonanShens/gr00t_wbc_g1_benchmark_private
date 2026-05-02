from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import advantage
from work.recap.scripts import interface_localization_pack
from work.recap.scripts import interface_localization_text_rewrite_map
from work.recap.scripts import state_conditioned_contract_gate


def test_rewrite_map_exposes_authority_metadata_and_diagnostic_labels() -> None:
    payload = interface_localization_text_rewrite_map.build_text_source_and_rewrite_map(
        REPO_ROOT
    )
    labels = cast(dict[str, Any], payload["surface_authority_labels"])
    summary = cast(dict[str, Any], payload["summary"])

    carrier = cast(dict[str, Any], labels["carrier_text_v1"])
    metadata = cast(dict[str, Any], labels["policy_condition_metadata"])
    legacy = cast(dict[str, Any], labels["prompt_conditioned"])
    diagnostic = cast(dict[str, Any], labels["numeric_advantage"])
    export_single = cast(
        dict[str, Any],
        payload["stages"]["export.task_text_field"]["semantic_selection"],
    )

    assert carrier["authority_status"] == "mainline_authority"
    assert carrier["mainline_authority"] is True
    assert carrier["carrier_field"] == "carrier_text_v1"
    assert carrier["source_prompt_field"] == "prompt_raw"

    assert metadata["authority_status"] == "metadata_only"
    assert metadata["mainline_authority"] is False
    assert metadata["diagnostic_only"] is False
    assert metadata["metadata_fields"] == [
        "policy_condition.phase",
        "policy_condition.mode",
        "policy_condition_text",
    ]

    assert legacy["authority_status"] == "legacy_non_authority"
    assert legacy["mainline_authority"] is False
    assert legacy["diagnostic_only"] is False

    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["mainline_authority"] is False
    assert diagnostic["authority_scope"] == (
        advantage.NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE
    )

    assert summary["numeric_mainline_consumes"] == "prompt_raw"
    assert summary["numeric_mainline_consumes_compat_alias_for"] == (
        "authoritative_text_carrier_source_field"
    )
    assert summary["exporter_default_consumes"] == "carrier_text_v1"
    assert summary["exporter_default_authority_status"] == "mainline_authority"
    assert export_single["selected_field"] == "carrier_text_v1"
    assert export_single["surface_authority_key"] == "carrier_text_v1"
    assert export_single["authority_status"] == "mainline_authority"


def test_interface_localization_pack_relabels_mainline_and_diagnostic_surfaces() -> (
    None
):
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
        generated_at="2026-04-11T00:00:00+00:00",
    )
    labels = cast(dict[str, Any], pack["surface_authority_labels"])
    by_name = {
        cast(str, entry["boundary_name"]): cast(dict[str, Any], entry)
        for entry in cast(list[dict[str, Any]], pack["final_boundary_statuses"])
    }

    assert cast(dict[str, Any], labels["carrier_text_v1"])["authority_status"] == (
        "mainline_authority"
    )
    assert (
        cast(dict[str, Any], labels["policy_condition_metadata"])["authority_status"]
        == "metadata_only"
    )
    assert cast(dict[str, Any], labels["numeric_advantage"])["diagnostic_only"] is True

    assert by_name["prompt_raw_source"]["surface_authority_key"] == "carrier_text_v1"
    assert by_name["prompt_conditioned_write"]["surface_authority_key"] == (
        "prompt_conditioned"
    )
    assert by_name["export_task_text_selection"]["surface_authority_key"] == (
        "carrier_text_v1"
    )
    assert by_name["collector_policy_callsite"]["authority_status"] == "diagnostic_only"
    assert by_name["policy_output_action"]["authority_status"] == "diagnostic_only"


def test_state_conditioned_contract_gate_surfaces_authority_metadata_split() -> None:
    freeze = state_conditioned_contract_gate.build_state_conditioned_freeze()
    validated = state_conditioned_contract_gate.validate_contract_candidate(
        state_conditioned_contract_gate.build_reference_contract_example()
    )

    mainline = cast(dict[str, Any], freeze["mainline_training_text"])
    policy_text_surface = cast(dict[str, Any], freeze["policy_text_surface"])
    validated_mainline = cast(dict[str, Any], validated["mainline_training_text"])
    validated_policy_text = cast(dict[str, Any], validated["policy_text"])

    assert mainline["authority_status"] == "mainline_authority"
    assert mainline["mainline_authority"] is True
    assert mainline["diagnostic_only"] is False
    assert mainline["legacy_text_non_authority_fields"] == [
        "prompt_conditioned",
        "dual_task_text",
    ]
    assert mainline["diagnostic_only_fields"] == ["advantage_input"]

    assert policy_text_surface["field"] == "policy_condition_text"
    assert policy_text_surface["authority_status"] == "metadata_only"
    assert policy_text_surface["mainline_authority"] is False
    assert policy_text_surface["diagnostic_only"] is False

    assert validated_mainline["authority_status"] == "mainline_authority"
    assert validated_mainline["diagnostic_only_fields"] == ["advantage_input"]
    assert validated_policy_text["authority_status"] == "metadata_only"
    assert validated_policy_text["metadata_fields"] == [
        "policy_condition.phase",
        "policy_condition.mode",
        "policy_condition_text",
    ]
