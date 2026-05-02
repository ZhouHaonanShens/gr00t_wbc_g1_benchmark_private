from __future__ import annotations

from collections.abc import Mapping
import json
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
from work.recap.run_manifest import build_run_manifest_from_sources
from work.recap.run_manifest import validate_run_manifest
from work.recap.scripts import gr00t_baseline_freeze_matrix
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval


def _make_checkpoint(root: Path, name: str) -> Path:
    checkpoint_dir = root / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"test-checkpoint")
    return checkpoint_dir


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _state_conditioned_metadata(checkpoint_dir: Path) -> dict[str, object]:
    return {
        "comparable_run_spec": {
            "dataset_fingerprint": "dataset-fingerprint-123",
            "stable_base": {"embodiment_tag": "UNITREE_G1"},
            "checkpoint_rule": {
                "selected_checkpoint_path": str(checkpoint_dir),
            },
        }
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
            "overlay_include_regex": scope_experiment.ACTION_HEAD_ONLY_EVAL_OVERLAY_REGEX,
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


def _build_manifest_with_scope(
    checkpoint_dir: Path,
    preset_id: str,
) -> dict[str, object]:
    return build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
        extensions={
            scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY: scope_experiment.build_scope_experiment_extension(
                preset_id,
                paired_triplet_artifacts={
                    "summary_json": f"agent/artifacts/triplet/{preset_id.lower()}_summary.json",
                },
            )
        },
    )


def _build_label_policy_extension(
    prompt_raw: str,
    *,
    enable_positive_duplication: bool,
    enable_task_phase_epsilon: bool,
) -> dict[str, object]:
    positive_epsilon = 0.15 if enable_task_phase_epsilon else 0.20
    negative_epsilon = 0.35 if enable_task_phase_epsilon else 0.20
    label_rows: list[dict[str, object]] = [
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
            "epsilon_l": positive_epsilon,
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
            "epsilon_l": negative_epsilon,
            "repeat_index": 0,
            "recovery_oversample_factor": 1,
        },
    ]
    if enable_positive_duplication:
        label_rows.append(
            {
                "sample_id": "sample_003",
                "source_sample_key": "episode_001::t0::dup1",
                "training_view": "C1",
                "carrier_text_v1": text_indicator.build_canonical_text_indicator(
                    prompt_raw,
                    text_indicator.TEXT_INDICATOR_POSITIVE,
                ),
                "policy_condition.phase": "TRANSPORT",
                "policy_condition.mode": "RECOVERY",
                "indicator_I": 1,
                "epsilon_l": positive_epsilon,
                "repeat_index": 1,
                "recovery_oversample_factor": 3,
            }
        )
    stats = {
        "schema_version": "g1_state_conditioned_equal_data_training_set_v1",
        "artifact_kind": "state_conditioned_sft_stats",
        "counts": {"unified_base_row_count": len(label_rows)},
    }
    return label_policy.build_label_policy(label_rows=label_rows, stats=stats)


def _build_manifest_with_scope_and_label(
    checkpoint_dir: Path,
    *,
    preset_id: str,
    label_policy_extension: Mapping[str, object],
) -> dict[str, object]:
    return build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
        extensions={
            scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY: scope_experiment.build_scope_experiment_extension(
                preset_id,
                paired_triplet_artifacts={
                    "summary_json": f"agent/artifacts/triplet/{preset_id.lower()}_summary.json",
                },
            ),
            label_policy.LABEL_POLICY_EXTENSION_KEY: dict(label_policy_extension),
        },
    )


def _fake_mode_surface_executor(
    *,
    indicator_mode: str,
    prompt_text: str,
    policy_options: Mapping[str, object],
) -> dict[str, object]:
    del prompt_text
    del policy_options

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

    scale_by_mode = {
        "omit": 0.1,
        "positive": 0.5,
        "negative": -0.5,
    }
    scale = scale_by_mode[indicator_mode]
    return {
        "raw_action_chunk": [[scale, scale + 0.1], [scale + 0.2, scale + 0.3]],
        "raw_action_by_group": _group_surface(scale + 0.01),
        "decoded_action": _group_surface(scale + 0.10),
        "absolute_action": _group_surface(scale + 0.20),
        "controller_input": _group_surface(scale + 0.30),
        "surface_source": "pytest.fake_executor",
    }


def _write_triplet_summary(
    *,
    tmp_path: Path,
    topic: str,
    checkpoint_dir: Path,
    prompt_raw: str,
) -> Path:
    output_dir = tmp_path / "triplet" / topic
    summary_path = (
        output_dir / gr00t_same_checkpoint_triplet_eval.DEFAULT_SUMMARY_JSON_NAME
    )
    bundle = gr00t_same_checkpoint_triplet_eval.build_same_checkpoint_triplet_bundle(
        checkpoint_loaded=str(checkpoint_dir),
        prompt_raw=prompt_raw,
        output_dir=output_dir,
        summary_json_path=summary_path,
        observation_seed=7,
        observation_surface={"q": [[0.0, 1.0]]},
        mode_surface_executor=_fake_mode_surface_executor,
    )
    written = gr00t_same_checkpoint_triplet_eval.write_same_checkpoint_triplet_bundle(
        bundle
    )
    return written["summary"]


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
                "summary": {"baseline_role": "public_anchor"},
            },
            gr00t_baseline_freeze_matrix.B1_BASELINE_ID: {
                "display_label": gr00t_baseline_freeze_matrix.DISPLAY_LABEL_B1,
                "branch_key": "unitree_g1",
                "mainline_authority": False,
                "legacy_backpointer_only": True,
                "summary": {"baseline_role": "legacy_negative_control"},
            },
        },
    }


def _materialize_standard_experiment_matrix(tmp_path: Path) -> dict[str, object]:
    prompt_raw = "pick up the apple and place it on the plate"
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-e-matrix")

    e1_manifest = _build_manifest_with_scope_and_label(
        checkpoint_dir,
        preset_id="S1",
        label_policy_extension=_build_label_policy_extension(
            prompt_raw,
            enable_positive_duplication=False,
            enable_task_phase_epsilon=False,
        ),
    )
    e2_manifest = _build_manifest_with_scope_and_label(
        checkpoint_dir,
        preset_id="S2",
        label_policy_extension=_build_label_policy_extension(
            prompt_raw,
            enable_positive_duplication=False,
            enable_task_phase_epsilon=False,
        ),
    )
    e3_manifest = _build_manifest_with_scope_and_label(
        checkpoint_dir,
        preset_id="S2",
        label_policy_extension=_build_label_policy_extension(
            prompt_raw,
            enable_positive_duplication=True,
            enable_task_phase_epsilon=False,
        ),
    )
    e4_manifest = _build_manifest_with_scope_and_label(
        checkpoint_dir,
        preset_id="S2",
        label_policy_extension=_build_label_policy_extension(
            prompt_raw,
            enable_positive_duplication=True,
            enable_task_phase_epsilon=True,
        ),
    )

    e1_manifest_path = _write_json(tmp_path / "e1" / "run_manifest.json", e1_manifest)
    e2_manifest_path = _write_json(tmp_path / "e2" / "run_manifest.json", e2_manifest)
    e3_manifest_path = _write_json(tmp_path / "e3" / "run_manifest.json", e3_manifest)
    e4_manifest_path = _write_json(tmp_path / "e4" / "run_manifest.json", e4_manifest)

    e1_triplet_path = _write_triplet_summary(
        tmp_path=tmp_path,
        topic="e1",
        checkpoint_dir=checkpoint_dir,
        prompt_raw=prompt_raw,
    )
    e2_triplet_path = _write_triplet_summary(
        tmp_path=tmp_path,
        topic="e2",
        checkpoint_dir=checkpoint_dir,
        prompt_raw=prompt_raw,
    )
    e3_triplet_path = _write_triplet_summary(
        tmp_path=tmp_path,
        topic="e3",
        checkpoint_dir=checkpoint_dir,
        prompt_raw=prompt_raw,
    )
    e4_triplet_path = _write_triplet_summary(
        tmp_path=tmp_path,
        topic="e4",
        checkpoint_dir=checkpoint_dir,
        prompt_raw=prompt_raw,
    )

    matrix = experiment_matrix.materialize_experiment_matrix(
        baseline_freeze_payload=_baseline_freeze_payload(),
        experiment_row_specs=[
            experiment_matrix.build_experiment_row_spec(
                display_label="E1",
                run_manifest_path=e1_manifest_path,
                triplet_summary_path=e1_triplet_path,
            ),
            experiment_matrix.build_experiment_row_spec(
                display_label="E2",
                run_manifest_path=e2_manifest_path,
                triplet_summary_path=e2_triplet_path,
            ),
            experiment_matrix.build_experiment_row_spec(
                display_label="E3",
                run_manifest_path=e3_manifest_path,
                triplet_summary_path=e3_triplet_path,
            ),
            experiment_matrix.build_experiment_row_spec(
                display_label="E4",
                run_manifest_path=e4_manifest_path,
                triplet_summary_path=e4_triplet_path,
            ),
        ],
    )
    matrix["_test_manifests"] = {
        "E1": e1_manifest,
        "E2": e2_manifest,
        "E3": e3_manifest,
        "E4": e4_manifest,
    }
    matrix["_test_paths"] = {
        "E1": {"run_manifest": str(e1_manifest_path), "triplet": str(e1_triplet_path)},
        "E2": {"run_manifest": str(e2_manifest_path), "triplet": str(e2_triplet_path)},
        "E3": {"run_manifest": str(e3_manifest_path), "triplet": str(e3_triplet_path)},
        "E4": {"run_manifest": str(e4_manifest_path), "triplet": str(e4_triplet_path)},
    }
    return matrix


def test_scope_preset_registry_exposes_exact_machine_ids_and_derived_regexes() -> None:
    assert scope_experiment.SCOPE_PRESET_IDS == ("S1", "S2", "S3")

    s1 = scope_experiment.build_scope_experiment_extension("S1")
    s2 = scope_experiment.build_scope_experiment_extension("S2")
    s3 = scope_experiment.build_scope_experiment_extension("S3")

    assert s1["semantic_components"]["action_head"] is True
    assert s1["current_eval_lane"]["coverage"] == "full"
    assert s2["semantic_components"]["text_ingress"] is True
    assert s2["current_eval_lane"]["coverage"] == "partial_action_head_only"
    assert s3["semantic_components"]["top_backbone_fusion_blocks"]["enabled"] is True
    assert s3["semantic_components"]["top_backbone_fusion_blocks"]["layer_indices"] == [
        12,
        13,
        14,
        15,
    ]

    for payload in (s1, s2, s3):
        derived_core_fields = payload["derived_core_fields"]
        assert isinstance(derived_core_fields, Mapping)
        assert derived_core_fields["trainable_module_regex"]
        assert derived_core_fields["eval_overlay_regex"]
        assert (
            derived_core_fields["eval_overlay_regex"]
            == scope_experiment.ACTION_HEAD_ONLY_EVAL_OVERLAY_REGEX
        )


def test_scope_experiment_helper_rejects_implicit_scope_regex_only_when_preset_required() -> (
    None
):
    with pytest.raises(ValueError, match="regex-only scope"):
        scope_experiment.resolve_scope_experiment_from_manifest(
            {
                "core": {
                    "trainable_module_regex": r"^action_head\\..*",
                    "eval_overlay_regex": scope_experiment.ACTION_HEAD_ONLY_EVAL_OVERLAY_REGEX,
                },
                "extensions": {},
            },
            require_preset_metadata=True,
        )


@pytest.mark.parametrize("preset_id", ["S1", "S2", "S3"])
def test_scope_preset_manifest_happy_path_keeps_core_compatibility_fields(
    tmp_path: Path,
    preset_id: str,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, f"checkpoint-{preset_id.lower()}")
    manifest = _build_manifest_with_scope(checkpoint_dir, preset_id)

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)
    normalized_scope = scope_experiment.resolve_scope_experiment_from_manifest(
        validation["normalized_manifest"],
        require_preset_metadata=True,
    )
    core = manifest["core"]

    assert validation["formal_eligibility"] == "ALLOW"
    assert normalized_scope is not None
    assert isinstance(core, Mapping)
    assert normalized_scope["preset_id"] == preset_id
    assert (
        core["trainable_module_regex"]
        == normalized_scope["derived_core_fields"]["trainable_module_regex"]
    )
    assert (
        core["eval_overlay_regex"]
        == normalized_scope["derived_core_fields"]["eval_overlay_regex"]
    )


def test_scope_preset_triplet_binding_still_works_with_scope_metadata_present(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-s1")
    manifest = _build_manifest_with_scope(checkpoint_dir, "S1")

    gate = gr00t_same_checkpoint_triplet_eval.build_triplet_binding_gate(
        run_manifest_payload=manifest,
        run_manifest_path=tmp_path / "run_manifest.json",
        output_dir=tmp_path / "triplet_out",
        repo_root=REPO_ROOT,
        declared_checkpoint_loaded=str(checkpoint_dir),
        observation_seed=7,
        observation_signature_sha256="obs-sha",
    )

    assert gate["formal_eligibility"] == "ALLOW"
    assert gate["run_manifest"]["core"]["checkpoint_loaded"] == str(checkpoint_dir)
    scope_payload = gate["run_manifest"]["core"]
    assert (
        scope_payload["trainable_module_regex"]
        == scope_experiment.build_scope_experiment_extension("S1")[
            "derived_core_fields"
        ]["trainable_module_regex"]
    )


def test_experiment_matrix_keeps_display_labels_separate_from_machine_row_ids(
    tmp_path: Path,
) -> None:
    matrix = _materialize_standard_experiment_matrix(tmp_path)
    rows = cast(Mapping[str, Mapping[str, Any]], matrix["rows"])
    machine_id_policy = cast(Mapping[str, object], matrix["machine_id_policy"])
    display_rows = cast(list[object], matrix["display_rows"])
    assert isinstance(rows, Mapping)

    assert machine_id_policy["display_labels_are_not_machine_ids"] is True
    assert display_rows == [
        {
            "display_label": "B0",
            "row_id": gr00t_baseline_freeze_matrix.B0_BASELINE_ID,
        },
        {
            "display_label": "B1",
            "row_id": gr00t_baseline_freeze_matrix.B1_BASELINE_ID,
        },
        {"display_label": "E1", "row_id": experiment_matrix.E1_ROW_ID},
        {"display_label": "E2", "row_id": experiment_matrix.E2_ROW_ID},
        {"display_label": "E3", "row_id": experiment_matrix.E3_ROW_ID},
        {"display_label": "E4", "row_id": experiment_matrix.E4_ROW_ID},
    ]
    assert "B0" not in rows
    assert "B1" not in rows
    assert "E1" not in rows
    assert "E2" not in rows
    assert "E3" not in rows
    assert "E4" not in rows


def test_experiment_matrix_counts_extension_only_scope_and_label_changes_as_real_axes(
    tmp_path: Path,
) -> None:
    matrix = _materialize_standard_experiment_matrix(tmp_path)
    rows = cast(Mapping[str, Mapping[str, Any]], matrix["rows"])
    assert isinstance(rows, Mapping)
    manifests = cast(Mapping[str, Mapping[str, Any]], matrix["_test_manifests"])
    assert isinstance(manifests, Mapping)

    e2_manifest = manifests["E2"]
    e3_manifest = manifests["E3"]
    e4_manifest = manifests["E4"]
    assert isinstance(e2_manifest, Mapping)
    assert isinstance(e3_manifest, Mapping)
    assert isinstance(e4_manifest, Mapping)

    assert e2_manifest["core_digest"] == e3_manifest["core_digest"]
    assert e3_manifest["core_digest"] == e4_manifest["core_digest"]
    assert rows[experiment_matrix.E2_ROW_ID]["changed_axes"] == ["scope_preset"]
    assert rows[experiment_matrix.E3_ROW_ID]["changed_axes"] == [
        "positive_duplication_policy"
    ]
    assert rows[experiment_matrix.E4_ROW_ID]["changed_axes"] == [
        "task_phase_aware_epsilon"
    ]


def test_experiment_matrix_forces_multi_variable_rows_to_migration_only(
    tmp_path: Path,
) -> None:
    matrix = _materialize_standard_experiment_matrix(tmp_path)
    rows = cast(Mapping[str, Mapping[str, Any]], matrix["rows"])
    assert isinstance(rows, Mapping)

    e1 = rows[experiment_matrix.E1_ROW_ID]
    assert e1["compare_to_row_id"] == gr00t_baseline_freeze_matrix.B0_BASELINE_ID
    assert e1["changed_axes"] == ["text_indicator_carrier", "scope_preset"]
    assert e1["migration_only"] is True
    assert e1["attribution_allowed"] is False
    assert e1["comparability_level"] == "migration_only"
    assert "multiple_primary_axes_changed" in e1["attribution_blockers"]


def test_experiment_matrix_allows_single_axis_attribution_and_downgrades_partial_scope(
    tmp_path: Path,
) -> None:
    matrix = _materialize_standard_experiment_matrix(tmp_path)
    rows = cast(Mapping[str, Mapping[str, Any]], matrix["rows"])
    assert isinstance(rows, Mapping)

    e2 = rows[experiment_matrix.E2_ROW_ID]
    e3 = rows[experiment_matrix.E3_ROW_ID]
    e4 = rows[experiment_matrix.E4_ROW_ID]

    assert e2["migration_only"] is False
    assert e2["attribution_allowed"] is True
    assert e2["comparability_level"] == "partial_action_head_only"
    assert e3["migration_only"] is False
    assert e3["attribution_allowed"] is True
    assert e3["comparability_level"] == "partial_action_head_only"
    assert e4["migration_only"] is False
    assert e4["attribution_allowed"] is True
    assert e4["comparability_level"] == "partial_action_head_only"


def test_experiment_matrix_scope_s3_rows_stay_partial_not_full(
    tmp_path: Path,
) -> None:
    prompt_raw = "pick up the apple and place it on the plate"
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-s3")
    e2_manifest = _build_manifest_with_scope_and_label(
        checkpoint_dir,
        preset_id="S2",
        label_policy_extension=_build_label_policy_extension(
            prompt_raw,
            enable_positive_duplication=False,
            enable_task_phase_epsilon=False,
        ),
    )
    s3_manifest = _build_manifest_with_scope_and_label(
        checkpoint_dir,
        preset_id="S3",
        label_policy_extension=_build_label_policy_extension(
            prompt_raw,
            enable_positive_duplication=False,
            enable_task_phase_epsilon=False,
        ),
    )
    e2_manifest_path = _write_json(tmp_path / "s2" / "run_manifest.json", e2_manifest)
    s3_manifest_path = _write_json(tmp_path / "s3" / "run_manifest.json", s3_manifest)
    e2_triplet_path = _write_triplet_summary(
        tmp_path=tmp_path,
        topic="s2",
        checkpoint_dir=checkpoint_dir,
        prompt_raw=prompt_raw,
    )
    s3_triplet_path = _write_triplet_summary(
        tmp_path=tmp_path,
        topic="s3",
        checkpoint_dir=checkpoint_dir,
        prompt_raw=prompt_raw,
    )

    matrix = experiment_matrix.materialize_experiment_matrix(
        baseline_freeze_payload=_baseline_freeze_payload(),
        experiment_row_specs=[
            experiment_matrix.build_experiment_row_spec(
                display_label="E2",
                row_id="g1_matrix_scope_s2_reference",
                compare_to_row_id=gr00t_baseline_freeze_matrix.B0_BASELINE_ID,
                run_manifest_path=e2_manifest_path,
                triplet_summary_path=e2_triplet_path,
            ),
            experiment_matrix.build_experiment_row_spec(
                display_label="scope_s3",
                row_id="g1_matrix_scope_s3_partial",
                compare_to_row_id="g1_matrix_scope_s2_reference",
                run_manifest_path=s3_manifest_path,
                triplet_summary_path=s3_triplet_path,
            ),
        ],
    )
    rows = cast(Mapping[str, Mapping[str, Any]], matrix["rows"])
    assert isinstance(rows, Mapping)

    s3_row = rows["g1_matrix_scope_s3_partial"]
    assert s3_row["changed_axes"] == ["scope_preset"]
    assert s3_row["migration_only"] is False
    assert s3_row["attribution_allowed"] is True
    assert s3_row["comparability_level"] == "partial_action_head_only"


def test_experiment_matrix_rejects_missing_triplet_action_delta_backpointer(
    tmp_path: Path,
) -> None:
    prompt_raw = "pick up the apple and place it on the plate"
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-missing-triplet")
    manifest = _build_manifest_with_scope_and_label(
        checkpoint_dir,
        preset_id="S1",
        label_policy_extension=_build_label_policy_extension(
            prompt_raw,
            enable_positive_duplication=False,
            enable_task_phase_epsilon=False,
        ),
    )
    manifest_path = _write_json(tmp_path / "broken" / "run_manifest.json", manifest)
    broken_triplet_path = _write_json(
        tmp_path / "broken" / "same_checkpoint_triplet_eval.json",
        {
            "schema_version": gr00t_same_checkpoint_triplet_eval.REPORT_SCHEMA_VERSION,
            "artifact_kind": gr00t_same_checkpoint_triplet_eval.REPORT_ARTIFACT_KIND,
        },
    )

    with pytest.raises(ValueError, match="action_delta_audit"):
        experiment_matrix.materialize_experiment_matrix(
            baseline_freeze_payload=_baseline_freeze_payload(),
            experiment_row_specs=[
                experiment_matrix.build_experiment_row_spec(
                    display_label="E1",
                    run_manifest_path=manifest_path,
                    triplet_summary_path=broken_triplet_path,
                )
            ],
        )
