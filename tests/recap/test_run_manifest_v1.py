from __future__ import annotations

from collections.abc import Mapping
import inspect
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap import prompt_builder
from work.recap import critic_promotion
from work.recap import label_policy
from work.recap import policy as recap_policy
from work.recap import scope_experiment
from work.recap import text_indicator
from work.recap.lerobot_export import dataset_export
from work.recap.run_manifest import build_run_manifest_from_sources
from work.recap.run_manifest import controller_config_hash
from work.recap.run_manifest import INDICATOR_SOURCE_FIELD
from work.recap.run_manifest import PROMPT_SOURCE_FIELD
from work.recap.run_manifest import TEXT_CARRIER_ROUTE
from work.recap.run_manifest import TEXT_CARRIER_SCHEMA_VERSION
from work.recap.run_manifest import validate_run_manifest
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval
from work.recap.scripts import gr00t_run_manifest_gate
from work.recap.scripts import state_conditioned_train


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
        "carrier_route": state_conditioned_train.MAINLINE_CARRIER_ROUTE,
        "carrier_schema_version": state_conditioned_train.MAINLINE_CARRIER_SCHEMA_VERSION,
        "prompt_source_field": state_conditioned_train.MAINLINE_PROMPT_SOURCE_FIELD,
        "indicator_source": state_conditioned_train.MAINLINE_INDICATOR_SOURCE_FIELD,
        "runtime_route": state_conditioned_train.MAINLINE_RUNTIME_ROUTE,
        "runtime_policy_class": state_conditioned_train.MAINLINE_RUNTIME_POLICY_CLASS,
        "runtime_indicator_mode_required": True,
        "runtime_supported_indicator_modes": list(
            state_conditioned_train.MAINLINE_RUNTIME_INDICATOR_MODES
        ),
        "mainline_authority": True,
        "diagnostic_only": False,
    }
    return {
        "training_route": training_route,
        "comparable_run_spec": {
            "dataset_fingerprint": "dataset-fingerprint-123",
            "carrier_schema_version": state_conditioned_train.MAINLINE_CARRIER_SCHEMA_VERSION,
            "carrier_route": state_conditioned_train.MAINLINE_CARRIER_ROUTE,
            "prompt_source_field": state_conditioned_train.MAINLINE_PROMPT_SOURCE_FIELD,
            "indicator_source": state_conditioned_train.MAINLINE_INDICATOR_SOURCE_FIELD,
            "training_route": training_route,
            "stable_base": {
                "embodiment_tag": "UNITREE_G1",
            },
            "checkpoint_rule": {
                "selected_checkpoint_path": str(checkpoint_dir),
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


def _label_policy_extension_payload() -> dict[str, object]:
    prompt = "pick up the apple and place it on the plate"
    label_rows = [
        {
            "sample_id": "sample_001",
            "source_sample_key": "episode_001::t0",
            "training_view": "C1",
            "carrier_text_v1": text_indicator.build_canonical_text_indicator(
                prompt,
                text_indicator.TEXT_INDICATOR_POSITIVE,
            ),
            "policy_condition.phase": "TRANSPORT",
            "policy_condition.mode": "RECOVERY",
            "indicator_I": 1,
            "epsilon_l": 0.15,
            "repeat_index": 0,
            "recovery_oversample_factor": 3,
        },
        {
            "sample_id": "sample_002",
            "source_sample_key": "episode_001::t1",
            "training_view": "C1",
            "carrier_text_v1": text_indicator.build_canonical_text_indicator(
                prompt,
                text_indicator.TEXT_INDICATOR_NEGATIVE,
            ),
            "policy_condition.phase": "APPROACH",
            "policy_condition.mode": "NOMINAL",
            "indicator_I": 0,
            "epsilon_l": 0.30,
            "repeat_index": 0,
            "recovery_oversample_factor": 3,
        },
    ]
    stats = {
        "schema_version": "g1_state_conditioned_equal_data_training_set_v1",
        "artifact_kind": "state_conditioned_sft_stats",
        "counts": {"unified_base_row_count": len(label_rows)},
    }
    return label_policy.build_label_policy(label_rows=label_rows, stats=stats)


def test_build_and_validate_run_manifest_happy_path(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    controller_audit = _controller_audit()
    controller_provenance = controller_audit["controller_provenance"]
    assert isinstance(controller_provenance, Mapping)
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=controller_audit,
        branch="UNITREE_G1",
        commit="abc123def456",
        extensions={"custom": {"kept_outside_core": True}},
    )

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)

    assert validation["formal_eligibility"] == "ALLOW"
    assert validation["issues"] == []
    assert manifest["core"]["carrier_schema_version"] == "recap_text_indicator_v1"
    assert manifest["core"]["prompt_source_field"] == "prompt_raw"
    assert manifest["core"]["checkpoint_loaded"] == str(checkpoint_dir)
    assert manifest["core"]["trainable_module_regex"] == "transformer\\.layers\\..*"
    assert manifest["core"]["eval_overlay_regex"] == "advantage_embedding.*"
    assert manifest["core"]["controller_config_hash"] == controller_config_hash(
        branch="UNITREE_G1",
        policy_horizon=30,
        n_action_steps=20,
        relative_contract=manifest["core"]["relative_absolute_action_contract"],
        controller_provenance=controller_provenance,
    )


def test_gate_blocks_missing_new_required_core_field(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )
    del manifest["core"]["checkpoint_loaded"]

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)

    assert validation["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "missing_required_core_field"
        and issue["field_path"] == "core.checkpoint_loaded"
        for issue in validation["issues"]
    )


def test_gate_blocks_wrong_type_for_n_action_steps(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )
    manifest["core"]["n_action_steps"] = "20"

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)

    assert validation["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "wrong_type" and issue["field_path"] == "core.n_action_steps"
        for issue in validation["issues"]
    )


def test_exporter_binding_uses_same_text_carrier_authority_names() -> None:
    task_text_default = (
        inspect.signature(dataset_export.export_recap_to_lerobot_v2)
        .parameters["task_text_field"]
        .default
    )

    assert task_text_default == dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD
    assert task_text_default == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    assert dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION == TEXT_CARRIER_SCHEMA_VERSION
    assert (
        TEXT_CARRIER_SCHEMA_VERSION
        == text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION
    )
    assert dataset_export.EXPORTER_CARRIER_ROUTE == TEXT_CARRIER_ROUTE
    assert TEXT_CARRIER_ROUTE == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    assert dataset_export.EXPORTER_PROMPT_SOURCE_FIELD == PROMPT_SOURCE_FIELD
    assert (
        PROMPT_SOURCE_FIELD == text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD
    )
    assert dataset_export.EXPORTER_PROMPT_ROUTE == prompt_builder.PHASE1_PROMPT_ROUTE
    assert dataset_export.EXPORTER_CONDITIONING_MODE == prompt_builder.CONDITIONING_MODE
    assert dataset_export.EXPORTER_INDICATOR_MODE_FIELD == INDICATOR_SOURCE_FIELD


def test_training_runtime_binding_constants_match_mainline_contract() -> None:
    metadata = _state_conditioned_metadata(Path("/tmp/checkpoint-placeholder"))
    comparable_run_spec = metadata["comparable_run_spec"]
    assert isinstance(comparable_run_spec, Mapping)
    training_route = metadata["training_route"]
    assert isinstance(training_route, Mapping)

    assert state_conditioned_train.MAINLINE_CARRIER_SCHEMA_VERSION == (
        TEXT_CARRIER_SCHEMA_VERSION
    )
    assert state_conditioned_train.MAINLINE_CARRIER_ROUTE == TEXT_CARRIER_ROUTE
    assert state_conditioned_train.MAINLINE_PROMPT_SOURCE_FIELD == PROMPT_SOURCE_FIELD
    assert state_conditioned_train.MAINLINE_INDICATOR_SOURCE_FIELD == (
        INDICATOR_SOURCE_FIELD
    )
    assert state_conditioned_train.MAINLINE_RUNTIME_ROUTE == (
        recap_policy.MAINLINE_RUNTIME_ROUTE
    )
    assert state_conditioned_train.MAINLINE_RUNTIME_POLICY_CLASS == (
        recap_policy.MAINLINE_RUNTIME_POLICY_CLASS_NAME
    )
    assert set(state_conditioned_train.MAINLINE_RUNTIME_INDICATOR_MODES) == set(
        recap_policy.MAINLINE_RUNTIME_INDICATOR_MODES
    )
    assert comparable_run_spec["carrier_schema_version"] == TEXT_CARRIER_SCHEMA_VERSION
    assert comparable_run_spec["carrier_route"] == TEXT_CARRIER_ROUTE
    assert comparable_run_spec["prompt_source_field"] == PROMPT_SOURCE_FIELD
    assert comparable_run_spec["indicator_source"] == INDICATOR_SOURCE_FIELD
    assert comparable_run_spec["training_route"] == training_route
    assert training_route["runtime_route"] == recap_policy.MAINLINE_RUNTIME_ROUTE
    assert training_route["runtime_policy_class"] == (
        recap_policy.MAINLINE_RUNTIME_POLICY_CLASS_NAME
    )
    assert training_route["runtime_indicator_mode_required"] is True
    runtime_supported_indicator_modes = training_route[
        "runtime_supported_indicator_modes"
    ]
    assert isinstance(runtime_supported_indicator_modes, list)
    assert set(runtime_supported_indicator_modes) == {
        text_indicator.TEXT_INDICATOR_OMIT,
        text_indicator.TEXT_INDICATOR_POSITIVE,
        text_indicator.TEXT_INDICATOR_NEGATIVE,
    }


def test_gate_script_adapter_happy_path_writes_manifest(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    state_metadata_path = _write_json(
        tmp_path / "state_conditioned_metadata.json",
        _state_conditioned_metadata(checkpoint_dir),
    )
    finetune_summary_path = _write_json(
        tmp_path / "finetune_summary.json",
        _finetune_summary(checkpoint_dir),
    )
    eval_summary_path = _write_json(
        tmp_path / "eval_summary.json",
        _eval_summary(checkpoint_dir),
    )
    controller_audit_path = _write_json(
        tmp_path / "controller_audit.json",
        _controller_audit(),
    )

    output_dir = tmp_path / "gate_happy_out"
    rc = gr00t_run_manifest_gate.main(
        [
            "--state-conditioned-metadata",
            str(state_metadata_path),
            "--finetune-summary",
            str(finetune_summary_path),
            "--eval-summary",
            str(eval_summary_path),
            "--controller-audit-json",
            str(controller_audit_path),
            "--branch",
            "UNITREE_G1",
            "--commit",
            "abc123def456",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    report = json.loads(
        (output_dir / gr00t_run_manifest_gate.RUN_MANIFEST_REPORT_JSON_NAME).read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (output_dir / gr00t_run_manifest_gate.RUN_MANIFEST_JSON_NAME).read_text(
            encoding="utf-8"
        )
    )

    assert report["formal_eligibility"] == "ALLOW"
    assert manifest["core"]["checkpoint_loaded"] == str(checkpoint_dir)
    assert manifest["core"]["controller_config_hash"]


def test_gate_script_blocks_checkpoint_binding_mismatch_and_writes_failure_note(
    tmp_path: Path,
) -> None:
    selected_checkpoint = _make_checkpoint(tmp_path, "checkpoint-selected")
    wrong_checkpoint = _make_checkpoint(tmp_path, "checkpoint-wrong")
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(selected_checkpoint),
        finetune_summary=_finetune_summary(selected_checkpoint),
        eval_summary=_eval_summary(selected_checkpoint),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )
    manifest["evaluation_binding"]["server_load_path"] = str(wrong_checkpoint)

    manifest_path = _write_json(tmp_path / "input_manifest.json", manifest)
    output_dir = tmp_path / "gate_out"

    rc = gr00t_run_manifest_gate.main(
        [
            "--manifest-json",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 1
    report = json.loads(
        (output_dir / gr00t_run_manifest_gate.RUN_MANIFEST_REPORT_JSON_NAME).read_text(
            encoding="utf-8"
        )
    )
    failure_note = (
        output_dir / gr00t_run_manifest_gate.FAILURE_NOTE_MARKDOWN_NAME
    ).read_text(encoding="utf-8")

    assert report["formal_eligibility"] == "BLOCK"
    assert any(issue["code"] == "checkpoint_mismatch" for issue in report["issues"])
    assert "checkpoint_selected" in failure_note


def test_triplet_binding_blocks_checkpoint_argument_mismatch(tmp_path: Path) -> None:
    selected_checkpoint = _make_checkpoint(tmp_path, "checkpoint-selected")
    wrong_checkpoint = _make_checkpoint(tmp_path, "checkpoint-wrong")
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(selected_checkpoint),
        finetune_summary=_finetune_summary(selected_checkpoint),
        eval_summary=_eval_summary(selected_checkpoint),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )

    gate = gr00t_same_checkpoint_triplet_eval.build_triplet_binding_gate(
        run_manifest_payload=manifest,
        run_manifest_path=tmp_path / "run_manifest.json",
        output_dir=tmp_path / "triplet_out",
        repo_root=REPO_ROOT,
        declared_checkpoint_loaded=str(wrong_checkpoint),
        observation_seed=7,
        observation_signature_sha256="obs-sha",
    )

    assert gate["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "checkpoint_argument_mismatch"
        and issue["field_path"] == "checkpoint_loaded"
        for issue in gate["issues"]
    )


def test_triplet_binding_allows_equivalent_checkpoint_symlink(
    tmp_path: Path,
) -> None:
    selected_checkpoint = _make_checkpoint(tmp_path, "checkpoint-selected")
    symlink_checkpoint = tmp_path / "checkpoint-link"
    symlink_checkpoint.symlink_to(selected_checkpoint, target_is_directory=True)
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(selected_checkpoint),
        finetune_summary=_finetune_summary(selected_checkpoint),
        eval_summary=_eval_summary(selected_checkpoint),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )

    gate = gr00t_same_checkpoint_triplet_eval.build_triplet_binding_gate(
        run_manifest_payload=manifest,
        run_manifest_path=tmp_path / "run_manifest.json",
        output_dir=tmp_path / "triplet_out",
        repo_root=REPO_ROOT,
        declared_checkpoint_loaded=str(symlink_checkpoint),
        observation_seed=7,
        observation_signature_sha256="obs-sha",
    )

    assert gate["formal_eligibility"] == "ALLOW"
    assert not any(
        issue["code"] == "checkpoint_argument_mismatch"
        for issue in gate["issues"]
    )


def test_triplet_binding_blocks_manifest_core_schema_mismatch(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )
    manifest["core"] = {
        **dict(manifest["core"]),
        "carrier_schema_version": "legacy_text_indicator_v0",
    }

    gate = gr00t_same_checkpoint_triplet_eval.build_triplet_binding_gate(
        run_manifest_payload=manifest,
        run_manifest_path=tmp_path / "run_manifest.json",
        output_dir=tmp_path / "triplet_out",
        repo_root=REPO_ROOT,
        declared_checkpoint_loaded=str(checkpoint_dir),
        observation_seed=7,
        observation_signature_sha256="obs-sha",
    )

    assert gate["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "manifest_core_mismatch"
        and issue["field_path"] == "core.carrier_schema_version"
        for issue in gate["issues"]
    )


def test_run_manifest_scope_presets_add_additive_scope_extension(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-s2")
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
        extensions={
            scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY: {"preset_id": "S2"}
        },
    )

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)
    normalized_scope = validation["normalized_manifest"]["extensions"][
        scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY
    ]

    assert validation["formal_eligibility"] == "ALLOW"
    assert normalized_scope["preset_id"] == "S2"
    assert (
        normalized_scope["current_eval_lane"]["coverage"] == "partial_action_head_only"
    )
    assert (
        manifest["core"]["trainable_module_regex"]
        == normalized_scope["derived_core_fields"]["trainable_module_regex"]
    )
    assert (
        manifest["core"]["eval_overlay_regex"]
        == normalized_scope["derived_core_fields"]["eval_overlay_regex"]
    )


def test_run_manifest_scope_presets_block_core_regex_drift(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-s3")
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
        extensions={
            scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY: {"preset_id": "S3"}
        },
    )
    manifest["core"]["trainable_module_regex"] = (
        r"^action_head\\.advantage_embedding\\..*"
    )

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)

    assert validation["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "scope_preset_core_mismatch"
        and issue["field_path"] == "core.trainable_module_regex"
        for issue in validation["issues"]
    )


def test_run_manifest_label_policy_extension_is_additive_and_kept_outside_core(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    base_manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
        extensions={
            label_policy.LABEL_POLICY_EXTENSION_KEY: _label_policy_extension_payload()
        },
    )

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)

    assert validation["formal_eligibility"] == "ALLOW"
    assert manifest["core"] == base_manifest["core"]
    assert manifest["core_digest"] == base_manifest["core_digest"]
    assert label_policy.LABEL_POLICY_EXTENSION_KEY not in manifest["core"]
    normalized_label_policy = validation["normalized_manifest"]["extensions"][
        label_policy.LABEL_POLICY_EXTENSION_KEY
    ]
    assert normalized_label_policy["schema_version"] == (
        label_policy.LABEL_POLICY_SCHEMA_VERSION
    )
    assert normalized_label_policy["artifact_kind"] == (
        label_policy.LABEL_POLICY_ARTIFACT_KIND
    )


def test_run_manifest_label_policy_extension_blocks_invalid_schema(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )
    manifest["extensions"] = {
        label_policy.LABEL_POLICY_EXTENSION_KEY: {
            "schema_version": "broken_label_policy_v0",
            "artifact_kind": label_policy.LABEL_POLICY_ARTIFACT_KIND,
        }
    }

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)

    assert validation["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "invalid_label_policy"
        and issue["field_path"]
        == f"extensions.{label_policy.LABEL_POLICY_EXTENSION_KEY}"
        for issue in validation["issues"]
    )


def test_run_manifest_keeps_critic_promotion_extension_outside_core(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-critic-promotion")
    base_manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
    )
    critic_promotion_extension = critic_promotion.build_critic_promotion_verdict(
        offline_audit_payload={"pass": True},
        downstream_gate_payload={"gate_passed": True, "gate_status": "DIAGNOSTIC_PASS"},
        gates_a_f_bundle={
            gate_name: True for gate_name in critic_promotion.GATES_A_F_ORDER
        },
    )
    manifest = build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(checkpoint_dir),
        finetune_summary=_finetune_summary(checkpoint_dir),
        eval_summary=_eval_summary(checkpoint_dir),
        controller_audit=_controller_audit(),
        branch="UNITREE_G1",
        commit="abc123def456",
        extensions={
            critic_promotion.CRITIC_PROMOTION_EXTENSION_KEY: critic_promotion_extension
        },
    )

    validation = validate_run_manifest(manifest, repo_root=REPO_ROOT)

    assert validation["formal_eligibility"] == "ALLOW"
    assert manifest["core"] == base_manifest["core"]
    assert manifest["core_digest"] == base_manifest["core_digest"]
    normalized_extension = validation["normalized_manifest"]["extensions"][
        critic_promotion.CRITIC_PROMOTION_EXTENSION_KEY
    ]
    assert normalized_extension["promotion_status"] == "PASS"
    assert (
        normalized_extension["critic_role"]
        == critic_promotion.CRITIC_ROLE_PRIMARY_RELABEL_SOURCE
    )
