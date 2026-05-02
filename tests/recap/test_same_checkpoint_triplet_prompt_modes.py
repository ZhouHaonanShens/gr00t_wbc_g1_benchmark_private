from __future__ import annotations
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import experiment_matrix
from work.recap import label_policy
from work.recap import scope_experiment
from work.recap import text_indicator
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval
from work.recap.run_manifest import build_run_manifest_from_sources
from work.recap.scripts import gr00t_baseline_freeze_matrix


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _make_checkpoint(root: Path, name: str) -> Path:
    checkpoint_dir = root / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"test-checkpoint")
    return checkpoint_dir


def _state_conditioned_metadata(checkpoint_dir: Path) -> dict[str, object]:
    training_route = {
        "carrier_route": "carrier_text_v1",
        "carrier_schema_version": "recap_text_indicator_v1",
        "prompt_source_field": "prompt_raw",
        "indicator_source": "indicator_mode",
        "runtime_route": "carrier_text_v1",
        "runtime_policy_class": "TextIndicatorGr00tPolicy",
    }
    return {
        "training_route": training_route,
        "comparable_run_spec": {
            "dataset_fingerprint": "dataset-fingerprint-123",
            "checkpoint_rule": {
                "selected_checkpoint_path": str(checkpoint_dir),
            },
            "stable_base": {
                "embodiment_tag": "UNITREE_G1",
            },
        },
    }


def _eval_summary(checkpoint_dir: Path) -> dict[str, object]:
    return {
        "execution_surface_contract": {
            "policy_horizon_expected": 30,
            "n_action_steps": 20,
            "relative_action_keys": ["left_arm", "right_arm"],
            "absolute_action_keys": ["left_hand", "right_hand", "waist"],
            "action_representation_by_key": {
                "left_arm": "RELATIVE",
                "right_arm": "RELATIVE",
                "left_hand": "ABSOLUTE",
                "right_hand": "ABSOLUTE",
                "waist": "ABSOLUTE",
            },
            "must_not_conflate_horizon_and_execution": True,
        },
        "evaluation_binding": {
            "eval_uses_finetuned": True,
            "server_load_mode": "model_path",
            "server_load_path": str(checkpoint_dir),
            "base_model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
        },
        "server_provenance": {
            "policy_model_path": str(checkpoint_dir),
            "base_model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            "overlay_include_regex": "advantage_embedding.*",
        },
    }


def _finetune_summary(checkpoint_dir: Path) -> dict[str, object]:
    return {
        "selected_checkpoint_path": str(checkpoint_dir),
        "effective_config": {
            "trainable_module_regex": "transformer\\.layers\\..*",
        },
    }


def _controller_audit() -> dict[str, object]:
    return {
        "controller_provenance": {
            "embodiment_tag": "UNITREE_G1",
            "official_env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
            "wbc_policy_class": "G1DecoupledWholeBodyPolicy",
        }
    }


def _build_valid_manifest(checkpoint_dir: Path) -> dict[str, object]:
    return build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )


def _build_label_policy_extension(prompt_raw: str) -> dict[str, object]:
    label_rows = [
        {
            "sample_id": "sample_001",
            "source_sample_key": "episode_001::t0",
            "training_view": "C1",
            "carrier_text_v1": text_indicator.build_canonical_text_indicator(
                prompt_raw,
                text_indicator.TEXT_INDICATOR_POSITIVE,
            ),
            "policy_condition.phase": "TRANSPORT",
            "policy_condition.mode": "RECOVERY",
            "indicator_I": 1,
            "epsilon_l": 0.20,
            "repeat_index": 0,
            "recovery_oversample_factor": 1,
        },
        {
            "sample_id": "sample_002",
            "source_sample_key": "episode_001::t1",
            "training_view": "C1",
            "carrier_text_v1": text_indicator.build_canonical_text_indicator(
                prompt_raw,
                text_indicator.TEXT_INDICATOR_NEGATIVE,
            ),
            "policy_condition.phase": "APPROACH",
            "policy_condition.mode": "NOMINAL",
            "indicator_I": 0,
            "epsilon_l": 0.20,
            "repeat_index": 0,
            "recovery_oversample_factor": 1,
        },
    ]
    stats = {
        "schema_version": "g1_state_conditioned_equal_data_training_set_v1",
        "artifact_kind": "state_conditioned_sft_stats",
        "counts": {"unified_base_row_count": len(label_rows)},
    }
    return label_policy.build_label_policy(label_rows=label_rows, stats=stats)


def _build_manifest_for_experiment_matrix(
    checkpoint_dir: Path,
    *,
    prompt_raw: str,
) -> dict[str, object]:
    return build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
        extensions={
            scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY: {"preset_id": "S1"},
            label_policy.LABEL_POLICY_EXTENSION_KEY: _build_label_policy_extension(
                prompt_raw
            ),
        },
    )


def _baseline_freeze_payload() -> dict[str, object]:
    return {
        "schema_version": gr00t_baseline_freeze_matrix.REPORT_SCHEMA_VERSION,
        "artifact_kind": gr00t_baseline_freeze_matrix.REPORT_ARTIFACT_KIND,
        "baselines": {
            gr00t_baseline_freeze_matrix.B0_BASELINE_ID: {
                "display_label": gr00t_baseline_freeze_matrix.DISPLAY_LABEL_B0,
                "branch_key": "unitree_g1",
                "mainline_authority": True,
                "legacy_backpointer_only": False,
                "summary": {},
            },
            gr00t_baseline_freeze_matrix.B1_BASELINE_ID: {
                "display_label": gr00t_baseline_freeze_matrix.DISPLAY_LABEL_B1,
                "branch_key": "unitree_g1",
                "mainline_authority": False,
                "legacy_backpointer_only": True,
                "summary": {},
            },
        },
    }


def _fake_mode_surface_executor(
    *,
    indicator_mode: str,
    prompt_text: str,
    policy_options: Mapping[str, object],
) -> dict[str, object]:
    def _group_surface(scale: float) -> dict[str, list[float]]:
        return {
            "left_arm": [scale + 0.01 * idx for idx in range(7)],
            "right_arm": [scale + 0.02 * idx for idx in range(7)],
            "left_hand": [scale + 0.03 * idx for idx in range(7)],
            "right_hand": [scale + 0.04 * idx for idx in range(7)],
            "waist": [scale + 0.05 * idx for idx in range(3)],
            "base_height_command": [scale + 0.06],
            "navigate_command": [scale + 0.07 * idx for idx in range(3)],
        }

    raw_by_mode = {
        "omit": [[0.0, 0.1], [0.2, 0.3]],
        "positive": [[0.4, 0.5], [0.6, 0.7]],
        "negative": [[-0.4, -0.5], [-0.6, -0.7]],
    }
    decoded_by_mode = {
        "omit": _group_surface(0.10),
        "positive": _group_surface(0.50),
        "negative": _group_surface(-0.50),
    }
    absolute_by_mode = {
        "omit": _group_surface(0.20),
        "positive": _group_surface(0.60),
        "negative": _group_surface(-0.60),
    }
    controller_by_mode = {
        "omit": _group_surface(0.30),
        "positive": _group_surface(0.70),
        "negative": _group_surface(-0.70),
    }
    raw_by_group = {
        "omit": _group_surface(0.01),
        "positive": _group_surface(0.11),
        "negative": _group_surface(-0.11),
    }
    return {
        "raw_action_chunk": raw_by_mode[indicator_mode],
        "raw_action_by_group": raw_by_group[indicator_mode],
        "decoded_action": decoded_by_mode[indicator_mode],
        "absolute_action": absolute_by_mode[indicator_mode],
        "controller_input": controller_by_mode[indicator_mode],
        "surface_source": "pytest.fake_executor",
        "runtime_metadata": {
            "prompt_text": prompt_text,
            "policy_options": dict(policy_options),
        },
    }


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_same_checkpoint_triplet_eval.main(["--help"])
    assert exc_info.value.code == 0


def test_bundle_writes_all_three_modes_with_separated_machine_readable_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "triplet"
    summary_path = (
        output_dir / gr00t_same_checkpoint_triplet_eval.DEFAULT_SUMMARY_JSON_NAME
    )
    bundle = gr00t_same_checkpoint_triplet_eval.build_same_checkpoint_triplet_bundle(
        checkpoint_loaded="/tmp/checkpoints/same-checkpoint",
        prompt_raw="pick up the apple and place it on the plate",
        output_dir=output_dir,
        summary_json_path=summary_path,
        observation_seed=123,
        observation_surface={
            "annotation.human.task_description": [
                "pick up the apple and place it on the plate"
            ],
            "q": [[0.0, 1.0]],
        },
        mode_surface_executor=_fake_mode_surface_executor,
    )

    written_paths = (
        gr00t_same_checkpoint_triplet_eval.write_same_checkpoint_triplet_bundle(bundle)
    )
    summary_payload = _read_json(written_paths["summary"])

    assert summary_payload["schema_version"] == (
        gr00t_same_checkpoint_triplet_eval.REPORT_SCHEMA_VERSION
    )
    assert summary_payload["artifact_kind"] == (
        gr00t_same_checkpoint_triplet_eval.REPORT_ARTIFACT_KIND
    )
    assert summary_payload["triplet_variable"] == "indicator_mode"
    assert summary_payload["numeric_advantage_main_control"] is False
    assert summary_payload["indicator_modes"] == ["omit", "positive", "negative"]
    assert summary_payload["same_checkpoint_locked"] is True
    assert summary_payload["same_observation_locked"] is True
    assert summary_payload["summary"]["modes_emitted"] == 3

    per_mode_artifact_paths = summary_payload["summary"]["per_mode_artifact_paths"]
    assert set(per_mode_artifact_paths) == {"omit", "positive", "negative"}
    assert len(set(per_mode_artifact_paths.values())) == 3

    per_mode_prompt_text = summary_payload["summary"]["per_mode_prompt_text"]
    assert per_mode_prompt_text["omit"] == "pick up the apple and place it on the plate"
    assert per_mode_prompt_text["positive"].endswith("Advantage: positive")
    assert per_mode_prompt_text["negative"].endswith("Advantage: negative")

    for indicator_mode in ["omit", "positive", "negative"]:
        mode_payload = _read_json(Path(per_mode_artifact_paths[indicator_mode]))
        assert mode_payload["schema_version"] == (
            gr00t_same_checkpoint_triplet_eval.MODE_SCHEMA_VERSION
        )
        assert mode_payload["artifact_kind"] == (
            gr00t_same_checkpoint_triplet_eval.MODE_ARTIFACT_KIND
        )
        assert mode_payload["indicator_mode"] == indicator_mode
        assert mode_payload["prompt_source_field"] == "prompt_raw"
        assert mode_payload["carrier_route"] == "carrier_text_v1"
        assert mode_payload["policy_class_name"] == "TextIndicatorGr00tPolicy"
        assert mode_payload["raw_action_chunk_available"] is True
        assert mode_payload["decoded_action_available"] is True
        assert mode_payload["absolute_action_available"] is True
        assert mode_payload["controller_input_available"] is True
        assert mode_payload["token_ids"] is None
        assert mode_payload["token_ids_available"] is False
        assert mode_payload["token_ids_unavailable_reason"]
        assert mode_payload["artifact_path"] == per_mode_artifact_paths[indicator_mode]
        assert mode_payload["backpointer"]["summary_json"] == str(
            summary_path.resolve()
        )
        assert mode_payload["action_delta_sidecar"]["canonical_stage_names"] == [
            "raw_action",
            "decoded_action",
            "absolute_action",
            "controller_input",
        ]
        assert (
            mode_payload["action_delta_sidecar"]["per_group_stage_surfaces"][
                "right_arm"
            ]["action_representation"]
            == "RELATIVE"
        )
        assert (
            mode_payload["action_delta_sidecar"]["per_group_stage_surfaces"][
                "right_hand"
            ]["action_representation"]
            == "ABSOLUTE"
        )
        assert (
            mode_payload["action_delta_sidecar"]["per_group_stage_surfaces"][
                "right_arm"
            ]["stages"]["controller_input"]["available"]
            is True
        )

    action_delta_audit = summary_payload["action_delta_audit"]
    assert action_delta_audit["audit_status"] == "READY"
    assert action_delta_audit["mode_pair_summary_keys"] == [
        "positive_vs_negative",
        "positive_vs_omit",
        "negative_vs_omit",
    ]
    assert set(action_delta_audit["mode_pair_summaries"]) == {
        "positive_vs_negative",
        "positive_vs_omit",
        "negative_vs_omit",
    }
    assert (
        action_delta_audit["per_mode_sidecar_backpointers"]["positive"]["json_field"]
        == "action_delta_sidecar"
    )
    assert (
        action_delta_audit["mode_pair_summaries"]["positive_vs_negative"]["per_group"][
            "right_arm"
        ]["stages"]["decoded_action"]["difference_present"]
        is True
    )
    assert (
        action_delta_audit["mode_pair_summaries"]["positive_vs_negative"]["per_group"][
            "right_hand"
        ]["action_representation"]
        == "ABSOLUTE"
    )


def test_experiment_matrix_backpointers_reuse_triplet_summary_and_action_delta_surface(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-experiment-matrix")
    prompt_raw = "pick up the apple and place it on the plate"
    manifest = _build_manifest_for_experiment_matrix(
        checkpoint_dir,
        prompt_raw=prompt_raw,
    )
    manifest_path = _write_json(tmp_path / "run_manifest.json", manifest)
    output_dir = tmp_path / "triplet"
    summary_path = (
        output_dir / gr00t_same_checkpoint_triplet_eval.DEFAULT_SUMMARY_JSON_NAME
    )
    bundle = gr00t_same_checkpoint_triplet_eval.build_same_checkpoint_triplet_bundle(
        checkpoint_loaded=str(checkpoint_dir),
        prompt_raw=prompt_raw,
        output_dir=output_dir,
        summary_json_path=summary_path,
        observation_seed=123,
        observation_surface={"q": [[0.0, 1.0]]},
        mode_surface_executor=_fake_mode_surface_executor,
    )
    written_paths = (
        gr00t_same_checkpoint_triplet_eval.write_same_checkpoint_triplet_bundle(bundle)
    )

    matrix = experiment_matrix.materialize_experiment_matrix(
        baseline_freeze_payload=_baseline_freeze_payload(),
        experiment_row_specs=[
            experiment_matrix.build_experiment_row_spec(
                display_label="E1",
                run_manifest_path=manifest_path,
                triplet_summary_path=written_paths["summary"],
            )
        ],
    )
    rows = cast(Mapping[str, Mapping[str, object]], matrix["rows"])
    assert isinstance(rows, Mapping)
    e1 = rows[experiment_matrix.E1_ROW_ID]
    backpointers = cast(Mapping[str, object], e1["backpointers"])

    assert backpointers["run_manifest_path"] == str(manifest_path.resolve())
    assert backpointers["scope_experiment"] == {
        "json_field": "extensions.scope_experiment",
        "preset_id": "S1",
        "schema_version": scope_experiment.SCOPE_EXPERIMENT_SCHEMA_VERSION,
    }
    label_policy_backpointer = cast(Mapping[str, object], backpointers["label_policy"])
    triplet_summary_backpointer = cast(
        Mapping[str, object], backpointers["triplet_summary"]
    )

    assert label_policy_backpointer["json_field"] == "extensions.label_policy"
    assert triplet_summary_backpointer["path"] == str(
        written_paths["summary"].resolve()
    )
    assert triplet_summary_backpointer["same_checkpoint_locked"] is True
    assert backpointers["action_delta_audit"] == {
        "path": str(written_paths["summary"].resolve()),
        "json_field": "action_delta_audit",
        "audit_status": "READY",
    }


@pytest.mark.parametrize(
    "mode_sequence,mode_surface_by_mode,match_text",
    [
        (
            ["omit", "positive", "positive"],
            None,
            "mode collapse detected",
        ),
        (
            None,
            {
                "omit": {"decoded_action": {"action.right_arm": [0.1]}},
                "positive": {"decoded_action": {"action.right_arm": [0.2]}},
            },
            "requires all three canonical modes",
        ),
    ],
)
def test_mode_collapse_or_missing_mode_is_rejected(
    tmp_path: Path,
    mode_sequence: list[str] | None,
    mode_surface_by_mode: dict[str, dict[str, object]] | None,
    match_text: str,
) -> None:
    with pytest.raises(ValueError, match=match_text):
        _ = gr00t_same_checkpoint_triplet_eval.build_same_checkpoint_triplet_bundle(
            checkpoint_loaded="/tmp/checkpoints/same-checkpoint",
            prompt_raw="pick up the apple and place it on the plate",
            output_dir=tmp_path / "triplet",
            summary_json_path=tmp_path / "triplet" / "summary.json",
            observation_seed=7,
            mode_sequence=mode_sequence,
            mode_surface_by_mode=mode_surface_by_mode,
        )


def test_provenance_gate_blocks_base_fallback_and_skips_mode_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    manifest = _build_valid_manifest(checkpoint_dir)
    manifest["evaluation_binding"] = {
        **cast(dict[str, object], manifest["evaluation_binding"]),
        "server_load_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
        "base_model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
    }
    manifest_path = _write_json(tmp_path / "run_manifest.json", manifest)
    output_dir = tmp_path / "triplet_out"

    rc = gr00t_same_checkpoint_triplet_eval.main(
        [
            "--run-manifest-json",
            str(manifest_path),
            "--prompt-raw",
            "pick up the apple and place it on the plate",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 1
    summary = _read_json(
        output_dir / gr00t_same_checkpoint_triplet_eval.DEFAULT_SUMMARY_JSON_NAME
    )
    failure_note_path = Path(str(summary["failure_note_path"]))

    assert summary["formal_eligibility"] == "BLOCK"
    assert summary["mode_artifacts"] == []
    assert summary["summary"]["modes_emitted"] == 0
    assert summary["triplet_gate"]["checkpoint_provenance"]["is_base_fallback"] is True
    assert any(
        issue["code"] == "checkpoint_provenance_blocked"
        for issue in summary["triplet_gate"]["issues"]
    )
    assert failure_note_path.is_file()
    assert not (output_dir / "same_checkpoint_triplet_omit.json").exists()
    assert not (output_dir / "same_checkpoint_triplet_positive.json").exists()
    assert not (output_dir / "same_checkpoint_triplet_negative.json").exists()


def test_provenance_gate_blocks_manifest_core_mismatch_with_machine_reason(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    manifest = _build_valid_manifest(checkpoint_dir)
    manifest["core"] = {
        **cast(dict[str, object], manifest["core"]),
        "carrier_schema_version": "legacy_text_indicator_v0",
    }
    manifest_path = _write_json(tmp_path / "run_manifest.json", manifest)
    output_dir = tmp_path / "triplet_out"

    rc = gr00t_same_checkpoint_triplet_eval.main(
        [
            "--run-manifest-json",
            str(manifest_path),
            "--prompt-raw",
            "pick up the apple and place it on the plate",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 1
    summary = _read_json(
        output_dir / gr00t_same_checkpoint_triplet_eval.DEFAULT_SUMMARY_JSON_NAME
    )

    assert summary["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "manifest_core_mismatch"
        and issue["field_path"] == "core.carrier_schema_version"
        for issue in summary["triplet_gate"]["issues"]
    )
    assert summary["summary"]["blocked_before_mode_artifacts"] is True
    assert summary["mode_artifacts"] == []


def test_action_delta_audit_marks_unavailable_stages_without_fabrication(
    tmp_path: Path,
) -> None:
    bundle = gr00t_same_checkpoint_triplet_eval.build_same_checkpoint_triplet_bundle(
        checkpoint_loaded="/tmp/checkpoints/same-checkpoint",
        prompt_raw="pick up the apple and place it on the plate",
        output_dir=tmp_path / "triplet",
        summary_json_path=tmp_path / "triplet" / "summary.json",
        observation_seed=9,
        mode_surface_by_mode={
            "omit": {
                "decoded_action": {"action.right_arm": [0.1] * 7},
            },
            "positive": {
                "decoded_action": {"action.right_arm": [0.2] * 7},
            },
            "negative": {
                "decoded_action": {"action.right_arm": [-0.2] * 7},
            },
        },
    )

    mode_payload = cast(Mapping[str, object], bundle["mode_payloads"]["omit"])
    sidecar = cast(Mapping[str, object], mode_payload["action_delta_sidecar"])
    per_group = cast(Mapping[str, object], sidecar["per_group_stage_surfaces"])
    right_arm = cast(Mapping[str, object], per_group["right_arm"])
    right_arm_stages = cast(Mapping[str, object], right_arm["stages"])
    absolute_stage = cast(Mapping[str, object], right_arm_stages["absolute_action"])
    controller_stage = cast(Mapping[str, object], right_arm_stages["controller_input"])

    assert mode_payload["absolute_action"] is None
    assert mode_payload["absolute_action_available"] is False
    assert "not exposed separately" in str(
        mode_payload["absolute_action_unavailable_reason"]
    )
    assert mode_payload["controller_input"] is None
    assert mode_payload["controller_input_available"] is False
    assert absolute_stage["available"] is False
    assert controller_stage["available"] is False
    assert "not exposed separately" in str(absolute_stage["unavailable_reason"])
    assert "controller_input was not exposed separately" in str(
        controller_stage["unavailable_reason"]
    )


def test_policy_server_returned_surface_binds_absolute_and_controller_input() -> None:
    action = {
        "action.right_arm": [[[0.1] * 7 for _ in range(30)]],
        "action.right_hand": [[[0.2] * 7 for _ in range(30)]],
    }

    surface = gr00t_same_checkpoint_triplet_eval._policy_server_returned_action_surface(
        action=action,
        action_info={"source": "unit-test"},
        policy_options={"indicator_mode": "positive", "seed": 9},
        host="127.0.0.1",
        server_port=5562,
    )

    assert surface["surface_source"] == "policy_server.get_action"
    assert surface["decoded_action"] == action
    assert surface["absolute_action"] == action
    assert surface["controller_input"] == action
    assert surface["post_transform_action"] == action
    assert surface["raw_action_chunk_unavailable_reason"]
    assert surface["token_ids_unavailable_reason"]

    bundle = gr00t_same_checkpoint_triplet_eval.build_same_checkpoint_triplet_bundle(
        checkpoint_loaded="/tmp/checkpoints/same-checkpoint",
        prompt_raw="pick up the apple and place it on the plate",
        output_dir=Path("/tmp/triplet"),
        summary_json_path=Path("/tmp/triplet/summary.json"),
        observation_seed=9,
        mode_surface_by_mode={
            "omit": surface,
            "positive": surface,
            "negative": surface,
        },
    )
    mode_payload = cast(Mapping[str, object], bundle["mode_payloads"]["positive"])
    assert mode_payload["absolute_action_available"] is True
    assert mode_payload["controller_input_available"] is True
    assert mode_payload["post_transform_action_available"] is True

    sidecar = cast(Mapping[str, object], mode_payload["action_delta_sidecar"])
    coverage = cast(Mapping[str, Mapping[str, object]], sidecar["stage_group_coverage"])
    assert coverage["absolute_action"]["available_group_count"] == 2
    assert coverage["controller_input"]["available_group_count"] == 2
