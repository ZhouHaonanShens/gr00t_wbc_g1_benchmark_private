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


from work.recap.scripts import gr00t_ladder_policy_gate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison_surface(branch: str) -> dict[str, Any]:
    spec = gr00t_ladder_policy_gate.BRANCH_SPECS[branch]
    controller_family = (
        "GR00T-WholeBodyControl"
        if branch == gr00t_ladder_policy_gate.BRANCH_UNITREE_G1
        else "branch_specific_controller_stack"
    )
    return {
        "branch": {
            "branch_key": spec.branch_key,
            "branch_scope": spec.branch_scope,
            "public_anchor_comparable": spec.public_anchor_comparable,
        },
        "embodiment": {
            "embodiment_tag": branch,
            "modality_config_path": f"work/configs/{spec.branch_key}/modality_config.py",
            "modality_config_digest": f"{spec.branch_key}-modality-config-v1",
        },
        "controller": {
            "controller_family": controller_family,
            "action_horizon": 30,
            "relative_action_policy": "unitree_g1_arm_relative_else_absolute",
            "action_keys": ["left_arm", "right_arm", "waist"],
            "state_keys": ["left_arm", "right_arm", "waist"],
        },
        "prompt_interface": {
            "prompt_template_id": "gr00t_g1_pnp_apple_to_plate_public_eval_v1",
            "condition_injection": "task_text_only",
            "condition_schema": "gr00t_policy_condition_v1",
        },
        "training": {
            "parameter_update": {
                "visual_unfreeze": False,
                "lora_enabled": False,
                "lora_rank": 0,
                "selective_unfreeze_modules": [],
                "tune_visual": False,
                "tune_projector": False,
                "tune_diffusion_model": False,
                "tune_llm": False,
            },
            "optimizer": {
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "gradient_clip_norm": 1.0,
            },
            "schedule": {
                "max_steps": 100,
                "save_steps": 100,
                "save_total_limit": 1,
                "warmup_ratio": 0.05,
                "global_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_gpus": 1,
                "dataloader_num_workers": 0,
            },
        },
        "dataset": {
            "dataset_mix": ["bucket_a:0.70", "bucket_b:0.30"],
            "admission": {
                "branch_inclusion": [branch],
                "dataset_source_ids": ["canonical_bucket_a"],
                "dataset_fingerprints": ["dataset-fingerprint-v1"],
                "admission_policy_version": "admission_v1",
            },
            "normalization": {
                "explicit_stats_policy": "branch_specific_stats_v1",
                "stats_fingerprint": "stats-fingerprint-v1",
                "stats_owner": spec.branch_key,
                "explicit_diff_reason": "none",
                "hidden_stats_fingerprint": "hidden-stats-fingerprint-v1",
                "implicit_cross_branch_stats_reuse": False,
            },
            "sampling": {
                "seed_policy": "fixed_seed_manifest_v1",
                "episode_sampling_policy": "equal_weight_frozen_v1",
            },
        },
    }


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_ladder_policy_gate.main(["--help"])
    assert exc_info.value.code == 0


def test_unitree_cli_materializes_branch_scoped_p_and_d_gates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "ladder_policy"

    exit_code = gr00t_ladder_policy_gate.main(
        ["--branch", "UNITREE_G1", "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    p_gate = _read_json(
        output_dir
        / gr00t_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH["UNITREE_G1"]["P"]
    )
    d_gate = _read_json(
        output_dir
        / gr00t_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH["UNITREE_G1"]["D"]
    )

    assert exit_code == 0
    assert captured.err == ""
    assert payload["branch"] == "UNITREE_G1"
    assert Path(payload["p_ladder_policy_gate_path"]) == (
        output_dir
        / gr00t_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH["UNITREE_G1"]["P"]
    )
    assert Path(payload["d_ladder_policy_gate_path"]) == (
        output_dir
        / gr00t_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH["UNITREE_G1"]["D"]
    )
    assert p_gate["artifact_kind"] == gr00t_ladder_policy_gate.REPORT_ARTIFACT_KIND
    assert d_gate["artifact_kind"] == gr00t_ladder_policy_gate.REPORT_ARTIFACT_KIND
    assert p_gate["ladder_axis"] == "P"
    assert d_gate["ladder_axis"] == "D"
    assert p_gate["branch_scope"] == "official_public_anchor_line"
    assert d_gate["branch_scope"] == "official_public_anchor_line"
    assert p_gate["public_anchor_comparable"] is True
    assert d_gate["public_anchor_comparable"] is True
    assert (
        "training.parameter_update.visual_unfreeze"
        in p_gate["allowed_difference_paths"]
    )
    assert "controller.controller_family" in p_gate["forbidden_difference_paths"]
    assert "dataset.dataset_mix" in d_gate["allowed_difference_paths"]
    assert "dataset.sampling.seed_policy" in d_gate["forbidden_difference_paths"]
    assert sorted(p_gate["promotion_requirements"].keys()) == sorted(
        gr00t_ladder_policy_gate.PROMOTION_REQUIREMENT_ORDER
    )


def test_new_embodiment_cli_materializes_internal_only_branch_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "ladder_policy"

    exit_code = gr00t_ladder_policy_gate.main(
        ["--branch", "NEW_EMBODIMENT", "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    p_gate = _read_json(
        output_dir
        / gr00t_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH["NEW_EMBODIMENT"]["P"]
    )
    d_gate = _read_json(
        output_dir
        / gr00t_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH["NEW_EMBODIMENT"]["D"]
    )

    assert exit_code == 0
    assert captured.err == ""
    assert payload["branch_scope"] == "branch_internal_only"
    assert payload["public_anchor_comparable"] is False
    assert p_gate["branch_scope"] == "branch_internal_only"
    assert d_gate["branch_scope"] == "branch_internal_only"
    assert p_gate["public_anchor_comparable"] is False
    assert d_gate["public_anchor_comparable"] is False
    assert p_gate["official_comparable_line"] is False
    assert d_gate["internal_only_comparable_line"] is True


def test_p_ladder_allows_parameter_only_differences() -> None:
    gate = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="UNITREE_G1",
        axis="P",
    )
    reference = _comparison_surface("UNITREE_G1")
    candidate = copy.deepcopy(reference)
    candidate["training"]["parameter_update"]["visual_unfreeze"] = True
    candidate["training"]["optimizer"]["learning_rate"] = 5e-5

    report = gr00t_ladder_policy_gate.build_ladder_diff_report(
        gate,
        reference,
        candidate,
    )

    assert report["comparability_status"] == "PASS"
    assert report["observed_difference_paths"] == [
        "training.optimizer.learning_rate",
        "training.parameter_update.visual_unfreeze",
    ]
    assert report["offending_field_paths"] == []
    assert report["unexpected_difference_paths"] == []
    assert report["triggered_regression_blockers"] == []


def test_p_ladder_blocks_controller_drift() -> None:
    gate = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="UNITREE_G1",
        axis="P",
    )
    reference = _comparison_surface("UNITREE_G1")
    candidate = copy.deepcopy(reference)
    candidate["controller"]["controller_family"] = "different_controller_stack"

    report = gr00t_ladder_policy_gate.build_ladder_diff_report(
        gate,
        reference,
        candidate,
    )

    assert report["comparability_status"] == "BLOCK"
    assert report["offending_field_paths"] == ["controller.controller_family"]
    assert report["unexpected_difference_paths"] == []
    assert report["blocking_reasons"] == ["forbidden_difference_paths_present"]
    assert report["triggered_regression_blockers"] == [
        "controller_embodiment_prompt_interface_drift"
    ]


def test_d_ladder_allows_dataset_mix_and_explicit_normalization_differences() -> None:
    gate = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="NEW_EMBODIMENT",
        axis="D",
    )
    reference = _comparison_surface("NEW_EMBODIMENT")
    candidate = copy.deepcopy(reference)
    candidate["dataset"]["dataset_mix"] = ["bucket_a:0.50", "bucket_c:0.50"]
    candidate["dataset"]["normalization"]["explicit_diff_reason"] = (
        "recomputed_for_new_mix"
    )
    candidate["dataset"]["normalization"]["stats_fingerprint"] = "stats-fingerprint-v2"

    report = gr00t_ladder_policy_gate.build_ladder_diff_report(
        gate,
        reference,
        candidate,
    )

    assert report["comparability_status"] == "PASS"
    assert report["observed_difference_paths"] == [
        "dataset.dataset_mix",
        "dataset.normalization.explicit_diff_reason",
        "dataset.normalization.stats_fingerprint",
    ]
    assert report["offending_field_paths"] == []
    assert report["unexpected_difference_paths"] == []


def test_d_ladder_blocks_training_scope_drift() -> None:
    gate = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="NEW_EMBODIMENT",
        axis="D",
    )
    reference = _comparison_surface("NEW_EMBODIMENT")
    candidate = copy.deepcopy(reference)
    candidate["training"]["schedule"]["max_steps"] = 200

    report = gr00t_ladder_policy_gate.build_ladder_diff_report(
        gate,
        reference,
        candidate,
    )

    assert report["comparability_status"] == "BLOCK"
    assert report["offending_field_paths"] == ["training.schedule.max_steps"]
    assert report["triggered_regression_blockers"] == [
        "parameter_or_training_scope_drift"
    ]


def test_d_ladder_blocks_hidden_sampling_drift() -> None:
    gate = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="NEW_EMBODIMENT",
        axis="D",
    )
    reference = _comparison_surface("NEW_EMBODIMENT")
    candidate = copy.deepcopy(reference)
    candidate["dataset"]["sampling"]["seed_policy"] = "different_seed_policy"

    report = gr00t_ladder_policy_gate.build_ladder_diff_report(
        gate,
        reference,
        candidate,
    )

    assert report["comparability_status"] == "BLOCK"
    assert report["offending_field_paths"] == ["dataset.sampling.seed_policy"]
    assert report["triggered_regression_blockers"] == [
        "hidden_normalization_or_sampling_drift"
    ]


def test_promotion_report_requires_fixed_replicates_and_rejects_lucky_seed_only() -> (
    None
):
    gate = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="UNITREE_G1",
        axis="P",
    )
    good_report = gr00t_ladder_policy_gate.build_promotion_report(
        gate,
        {
            "fixed_replicate_policy": True,
            "fixed_seed_policy": True,
            "no_systemic_break": True,
            "provenance_pass": True,
            "diagnostics_not_regressing": True,
            "single_lucky_seed_only_improvement": False,
        },
    )
    blocked_report = gr00t_ladder_policy_gate.build_promotion_report(
        gate,
        {
            "fixed_replicate_policy": False,
            "fixed_seed_policy": True,
            "no_systemic_break": True,
            "provenance_pass": False,
            "diagnostics_not_regressing": False,
            "single_lucky_seed_only_improvement": True,
        },
    )

    assert good_report["promotion_allowed"] is True
    assert good_report["promotion_status"] == "PASS"
    assert blocked_report["promotion_allowed"] is False
    assert blocked_report["promotion_status"] == "BLOCK"
    assert blocked_report["failure_reasons"] == [
        "fixed_replicate_policy_required",
        "provenance_pass_required",
        "diagnostics_not_regressing_required",
        "no_single_lucky_seed_promotion_required",
    ]
