from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_action_chain_telemetry


def test_unitree_g1_telemetry_restates_frozen_execution_surface_contract() -> None:
    report = gr00t_action_chain_telemetry.build_telemetry_report("UNITREE_G1")

    assert report["policy_horizon_expected"] == 30
    assert report["n_action_steps_expected"] == 20
    assert report["execution_surface_contract"] == {
        "policy_horizon_expected": 30,
        "n_action_steps_expected": 20,
        "relative_action_keys": ["left_arm", "right_arm"],
        "absolute_action_keys": [
            "left_hand",
            "right_hand",
            "waist",
            "base_height_command",
            "navigate_command",
        ],
        "action_representation_by_key": {
            "left_arm": "RELATIVE",
            "right_arm": "RELATIVE",
            "left_hand": "ABSOLUTE",
            "right_hand": "ABSOLUTE",
            "waist": "ABSOLUTE",
            "base_height_command": "ABSOLUTE",
            "navigate_command": "ABSOLUTE",
        },
        "relative_to_absolute_rule": {
            "enabled_for_relative_action_keys": True,
            "reference_state_timestep": "last",
            "reference_state_keys": {
                "left_arm": "left_arm",
                "right_arm": "right_arm",
            },
        },
        "must_not_conflate_horizon_and_execution": True,
        "repo_local_formalization": {
            "field_names_are_repo_local": True,
            "upstream_policy_horizon_authority": "action.delta_indices",
            "upstream_execution_steps_authority": "rollout --n_action_steps",
            "note": (
                "Telemetry re-states repo-local contract field names for drift detection; "
                "they summarize upstream semantics but are not upstream official JSON field names."
            ),
        },
    }


def test_unitree_g1_telemetry_keeps_relative_and_absolute_groups_separate() -> None:
    report = gr00t_action_chain_telemetry.build_telemetry_report("UNITREE_G1")

    left_arm = report["per_group_stats"]["left_arm"]
    right_arm = report["per_group_stats"]["right_arm"]
    left_hand = report["per_group_stats"]["left_hand"]
    right_hand = report["per_group_stats"]["right_hand"]
    waist = report["per_group_stats"]["waist"]
    navigate = report["per_group_stats"]["navigate_command"]
    base_height = report["per_group_stats"]["base_height_command"]

    assert left_arm["action_representation"] == "RELATIVE"
    assert right_arm["action_representation"] == "RELATIVE"
    assert left_arm["reference_state_key"] == "left_arm"
    assert right_arm["reference_state_key"] == "right_arm"

    assert left_hand["action_representation"] == "ABSOLUTE"
    assert right_hand["action_representation"] == "ABSOLUTE"
    assert waist["action_representation"] == "ABSOLUTE"
    assert navigate["action_representation"] == "ABSOLUTE"
    assert base_height["action_representation"] == "ABSOLUTE"
    assert left_hand["reference_state_key"] is None
    assert right_hand["reference_state_key"] is None
    assert waist["reference_state_key"] is None
    assert navigate["reference_state_key"] is None
    assert base_height["reference_state_key"] is None


def test_grouped_action_chain_sidecar_keeps_canonical_stage_names_and_split_groups() -> (
    None
):
    sidecar = gr00t_action_chain_telemetry.build_grouped_action_chain_sidecar(
        "UNITREE_G1",
        stage_group_values={
            "raw_action": {
                "right_arm": [0.1] * 7,
                "right_hand": [0.2] * 7,
            },
            "decoded_action": {
                "action.right_arm": [0.3] * 7,
                "action.right_hand": [0.4] * 7,
            },
            "absolute_action": {
                "right_arm": [0.5] * 7,
                "right_hand": [0.6] * 7,
            },
            "controller_input": {
                "right_arm": [0.7] * 7,
                "right_hand": [0.8] * 7,
            },
        },
        stage_unavailable_reasons={
            "raw_action": "raw stage only provided for targeted groups in this test"
        },
    )

    assert sidecar["canonical_stage_names"] == [
        "raw_action",
        "decoded_action",
        "absolute_action",
        "controller_input",
    ]
    assert sidecar["action_group_order"] == [
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    ]

    right_arm = sidecar["per_group_stage_surfaces"]["right_arm"]
    right_hand = sidecar["per_group_stage_surfaces"]["right_hand"]
    left_arm = sidecar["per_group_stage_surfaces"]["left_arm"]

    assert right_arm["action_representation"] == "RELATIVE"
    assert right_arm["reference_state_key"] == "right_arm"
    assert right_hand["action_representation"] == "ABSOLUTE"
    assert right_hand["reference_state_key"] is None
    assert right_arm["stages"]["decoded_action"]["available"] is True
    assert right_hand["stages"]["decoded_action"]["available"] is True
    assert left_arm["stages"]["raw_action"]["available"] is False
    assert "seven-group split" in left_arm["stages"]["raw_action"]["unavailable_reason"]


def test_mode_pair_summaries_cover_required_triplet_pairs() -> None:
    omit_sidecar = gr00t_action_chain_telemetry.build_grouped_action_chain_sidecar(
        "UNITREE_G1",
        stage_group_values={
            "raw_action": {"right_arm": [0.0] * 7, "right_hand": [0.0] * 7},
            "decoded_action": {"right_arm": [0.1] * 7, "right_hand": [0.2] * 7},
            "absolute_action": {"right_arm": [0.3] * 7, "right_hand": [0.4] * 7},
            "controller_input": {
                "right_arm": [0.5] * 7,
                "right_hand": [0.6] * 7,
            },
        },
    )
    positive_sidecar = gr00t_action_chain_telemetry.build_grouped_action_chain_sidecar(
        "UNITREE_G1",
        stage_group_values={
            "raw_action": {"right_arm": [0.1] * 7, "right_hand": [0.0] * 7},
            "decoded_action": {"right_arm": [0.2] * 7, "right_hand": [0.2] * 7},
            "absolute_action": {"right_arm": [0.4] * 7, "right_hand": [0.4] * 7},
            "controller_input": {
                "right_arm": [0.6] * 7,
                "right_hand": [0.6] * 7,
            },
        },
    )
    negative_sidecar = gr00t_action_chain_telemetry.build_grouped_action_chain_sidecar(
        "UNITREE_G1",
        stage_group_values={
            "raw_action": {"right_arm": [-0.1] * 7, "right_hand": [0.0] * 7},
            "decoded_action": {
                "right_arm": [-0.2] * 7,
                "right_hand": [0.2] * 7,
            },
            "absolute_action": {
                "right_arm": [-0.4] * 7,
                "right_hand": [0.4] * 7,
            },
            "controller_input": {
                "right_arm": [-0.6] * 7,
                "right_hand": [0.6] * 7,
            },
        },
    )

    pair_summaries = (
        gr00t_action_chain_telemetry.build_action_chain_mode_pair_summaries(
            "UNITREE_G1",
            mode_sidecars={
                "omit": omit_sidecar,
                "positive": positive_sidecar,
                "negative": negative_sidecar,
            },
        )
    )

    assert set(pair_summaries) == {
        "positive_vs_negative",
        "positive_vs_omit",
        "negative_vs_omit",
    }
    positive_vs_negative = pair_summaries["positive_vs_negative"]
    assert positive_vs_negative["canonical_stage_names"] == [
        "raw_action",
        "decoded_action",
        "absolute_action",
        "controller_input",
    ]
    assert positive_vs_negative["per_group"]["right_arm"]["action_representation"] == (
        "RELATIVE"
    )
    assert (
        positive_vs_negative["per_group"]["right_hand"]["action_representation"]
        == "ABSOLUTE"
    )
    assert (
        positive_vs_negative["per_group"]["right_arm"]["stages"]["decoded_action"][
            "difference_present"
        ]
        is True
    )
    assert positive_vs_negative["difference_groups_by_stage"]["decoded_action"] == [
        "right_arm"
    ]

    execution_surface = (
        gr00t_action_chain_telemetry.build_pair_execution_surface_summary(
            positive_vs_negative,
            terminal_stage="controller_input",
        )
    )
    assert execution_surface["terminal_stage_used"] == "controller_input"
    assert (
        execution_surface["per_group"]["right_arm"]["difference_disappeared_at"] is None
    )
    assert (
        execution_surface["per_group"]["right_arm"]["deepest_distinct_checkpoint"]
        == "controller_input"
    )
    assert (
        execution_surface["per_group"]["right_hand"]["difference_disappeared_at"]
        == "model"
    )
