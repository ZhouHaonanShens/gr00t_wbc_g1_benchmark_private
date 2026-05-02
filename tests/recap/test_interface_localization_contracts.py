from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import interface_localization_contract


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        interface_localization_contract.main(["--help"])
    assert exc_info.value.code == 0


def test_markdown_freeze_block_matches_canonical_contract() -> None:
    _markdown, raw_spec, validated = (
        interface_localization_contract.load_markdown_contract_spec(
            REPO_ROOT / interface_localization_contract.DEFAULT_CONTRACT_DOC_PATH
        )
    )

    assert (
        raw_spec
        == interface_localization_contract.build_interface_localization_contract()
    )
    assert (
        validated["schema_version"]
        == interface_localization_contract.CONTRACT_SCHEMA_VERSION
    )
    assert raw_spec["boundary_ontology"]["boundary_order"] == list(
        interface_localization_contract.BOUNDARY_ORDER
    )
    assert raw_spec["status_ontology"]["status_order"] == [
        "survived",
        "died",
        "mutated",
        "rerouted",
        "bypassed",
        "blocked_missing_upstream",
    ]
    assert raw_spec["provenance_classes"]["class_order"] == [
        "static",
        "synthetic",
        "replay_live",
        "server_live",
    ]
    assert (
        raw_spec["baseline_tuple"]["right_arm_boundary_name"]
        == "body_wrist_upper_limb_chain"
    )
    assert (
        raw_spec["baseline_tuple"]["right_hand_boundary_name"]
        == "dex3_finger_hand_path"
    )


def test_main_materializes_contract_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"

    exit_code = interface_localization_contract.main(["--output-dir", str(output_dir)])

    assert exit_code == 0
    baseline = _read_json(
        output_dir / interface_localization_contract.BASELINE_JSON_NAME
    )
    contract = _read_json(
        output_dir / interface_localization_contract.CONTRACT_JSON_NAME
    )

    assert baseline == interface_localization_contract.build_baseline_tuple_artifact()
    assert (
        contract
        == interface_localization_contract.build_interface_localization_contract()
    )
    assert contract["artifact_contract"]["future_artifact_minimum_fields"] == [
        "provenance_class",
        "generation_command",
        "input_baseline_summary",
        "backpointer",
    ]


def test_baseline_tuple_freezes_plan_required_surface() -> None:
    baseline = interface_localization_contract.build_baseline_tuple_artifact()
    tuple_payload = baseline["baseline_tuple"]

    assert baseline["baseline_tuple_field_order"] == [
        "embodiment",
        "benchmark_env_name",
        "benchmark_task",
        "simulator",
        "controller_stack",
        "scene_motion_slice_identifier",
        "checkpoint_identifier",
        "seed_set_identifier",
        "condition_pair_identifier",
        "server_mode_identifier",
        "serving_path_axis_identifier",
        "task_text_surface_identifier",
        "replay_init_source_identifier",
        "right_arm_boundary_name",
        "right_hand_boundary_name",
    ]
    assert tuple_payload["scene_motion_slice_identifier"].startswith(
        "TASK1_SELECTION_DEFERRED__"
    )
    assert tuple_payload["checkpoint_identifier"].startswith(
        "TASK1_SELECTION_DEFERRED__"
    )
    assert tuple_payload["seed_set_identifier"].startswith("TASK1_SELECTION_DEFERRED__")
    assert tuple_payload["condition_pair_identifier"].startswith(
        "TASK1_SELECTION_DEFERRED__"
    )
    assert tuple_payload["server_mode_identifier"] == "replay_sim_only"
    assert (
        tuple_payload["serving_path_axis_identifier"]
        == "stock_serving_path_vs_custom_advantage_aware_path"
    )
    assert (
        tuple_payload["task_text_surface_identifier"]
        == "prompt_raw_vs_prompt_conditioned_vs_runtime_override"
    )
    assert tuple_payload["replay_init_source_identifier"].startswith(
        "TASK1_SELECTION_DEFERRED__"
    )


def test_validate_contract_rejects_new_baseline_field_drift() -> None:
    candidate = copy.deepcopy(
        interface_localization_contract.build_interface_localization_contract()
    )
    candidate["baseline_tuple"]["checkpoint_identifier"] = "drifted_checkpoint"

    with pytest.raises(
        ValueError,
        match="baseline_tuple\\.checkpoint_identifier mismatch",
    ):
        interface_localization_contract.assert_contract_matches_canonical(candidate)


def test_default_output_dir_contract_matches_expected_branch_location() -> None:
    parser = interface_localization_contract.build_parser()
    args = parser.parse_args([])
    output_dir = interface_localization_contract.resolve_output_dir(REPO_ROOT, args)

    assert (
        output_dir
        == (
            REPO_ROOT
            / interface_localization_contract.DEFAULT_ARTIFACT_DIR
            / interface_localization_contract.DEFAULT_OUTPUT_SUBDIR
        ).resolve()
    )
