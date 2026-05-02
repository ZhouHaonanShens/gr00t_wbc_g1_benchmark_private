from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import drop_events
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import build_apple_recap_final_report
from work.recap.scripts import build_readonly_refs
from work.recap.scripts import build_uplift_schemas
from work.recap.scripts import critic_build_episode_traces
from work.recap.scripts import critic_build_sample_pack
from work.recap.scripts import critic_scorecard_all_splits
from work.recap.scripts import gr00t_action_absorption_audit
from work.recap.scripts import gr00t_carrier_panel_gate
from work.recap.scripts import inspect_mainline_carrier
from work.recap.scripts import relabel_counterfactual_rewards


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sign_payload(payload: dict[str, Any]) -> str:
    signature_basis = {
        key: value for key, value in payload.items() if key != "report_signature_sha256"
    }
    return apple_recap_execution_contract._sha256_payload(signature_basis)


def _freshness_bundle() -> dict[str, str]:
    return {
        "execution_sha": "abc123frozenexecutionsha",
        "manifest_hash": "manifest_hash_v1",
        "checkpoint_id": "checkpoint-1000",
        "seed_bundle_id": "20000:20009",
        "timestamp": "2026-04-12T00:00:00+00:00",
    }


def _row(
    row_id: str,
    display_label: str,
    *,
    mainline_authority: bool,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "display_label": display_label,
        "row_kind": "baseline" if display_label.startswith("B") else "experiment",
        "mainline_authority": mainline_authority,
        "compare_to_row_id": None,
        "changed_axes": [],
        "summary": {"display_label": display_label},
    }


def _experiment_matrix_payload() -> dict[str, Any]:
    display_rows = [
        {"display_label": "B0", "row_id": "g1_b0_public_anchor"},
        {"display_label": "E1", "row_id": "g1_e1_text_indicator_s1"},
        {"display_label": "E2", "row_id": "g1_e2_text_indicator_s2"},
        {
            "display_label": "E3",
            "row_id": "g1_e3_text_indicator_s2_positive_duplication",
        },
        {"display_label": "E4", "row_id": "g1_e4_text_indicator_s2_task_phase_epsilon"},
    ]
    rows = {
        "g1_b0_public_anchor": _row(
            "g1_b0_public_anchor", "B0", mainline_authority=True
        ),
        "g1_e1_text_indicator_s1": _row(
            "g1_e1_text_indicator_s1", "E1", mainline_authority=True
        ),
        "g1_e2_text_indicator_s2": _row(
            "g1_e2_text_indicator_s2", "E2", mainline_authority=True
        ),
        "g1_e3_text_indicator_s2_positive_duplication": _row(
            "g1_e3_text_indicator_s2_positive_duplication",
            "E3",
            mainline_authority=True,
        ),
        "g1_e4_text_indicator_s2_task_phase_epsilon": _row(
            "g1_e4_text_indicator_s2_task_phase_epsilon",
            "E4",
            mainline_authority=True,
        ),
    }
    payload = {
        "schema_version": "gr00t_experiment_matrix_v1",
        "artifact_kind": "gr00t_experiment_matrix",
        "generated_at": "2026-04-12T00:00:00+00:00",
        "display_rows": display_rows,
        "row_id_order": [item["row_id"] for item in display_rows],
        "rows": rows,
    }
    payload["report_signature_sha256"] = _sign_payload(payload)
    return payload


def _authority_fixture_specs(repo_root: Path) -> list[dict[str, str]]:
    _write_json(
        repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/public_anchor/public_anchor_formal.json",
        {
            "schema_version": "public_anchor_formal_v1",
            "artifact_kind": "public_anchor_formal",
            "report_signature_sha256": "public-anchor-signature",
        },
    )
    _write_json(
        repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/same_checkpoint_triplet/same_checkpoint_triplet_eval.json",
        {
            "schema_version": "same_checkpoint_triplet_eval_v1",
            "artifact_kind": "same_checkpoint_triplet_eval",
            "report_signature_sha256": "triplet-signature",
        },
    )
    _write_text(
        repo_root / "agent/artifacts/vlm_critic_scorecard/score_rows_v1.csv",
        "sample_id,predicted_value,return_G\nsample_001,-10,-12\n",
    )
    _write_text(
        repo_root
        / "agent/artifacts/recap_datasets/recap_mainline_fresh_20260311_121500_k0/episodes.jsonl",
        '{"episode_id":"episode_001"}\n',
    )
    _write_json(
        repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json",
        _experiment_matrix_payload(),
    )
    _write_json(
        repo_root
        / "agent/artifacts/recap_temporal_critic_upgrade/reward_audit/reward_gate.json",
        {
            "schema_version": "reward_gate_v1",
            "artifact_kind": "reward_gate",
            "report_signature_sha256": "reward-gate-signature",
        },
    )
    _write_text(
        repo_root / "agent/exchange/AppleToPlate_RECAP_status_critic_reward_audit.md",
        "# audit\n",
    )
    return [
        {
            "artifact_id": "public_anchor_formal",
            "authority_role": "official_public_anchor",
            "relative_path": "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/public_anchor/public_anchor_formal.json",
        },
        {
            "artifact_id": "same_checkpoint_triplet_eval",
            "authority_role": "diagnostic_triplet_eval",
            "relative_path": "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/same_checkpoint_triplet/same_checkpoint_triplet_eval.json",
        },
        {
            "artifact_id": "critic_score_rows_v1",
            "authority_role": "critic_held_out_score_rows",
            "relative_path": "agent/artifacts/vlm_critic_scorecard/score_rows_v1.csv",
        },
        {
            "artifact_id": "recap_authority_episodes",
            "authority_role": "mainline_reward_authority_dataset",
            "relative_path": "agent/artifacts/recap_datasets/recap_mainline_fresh_20260311_121500_k0/episodes.jsonl",
        },
        {
            "artifact_id": "gr00t_experiment_matrix",
            "authority_role": "experiment_matrix_backpointer",
            "relative_path": "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json",
        },
        {
            "artifact_id": "reward_gate",
            "authority_role": "reward_publish_gate",
            "relative_path": "agent/artifacts/recap_temporal_critic_upgrade/reward_audit/reward_gate.json",
        },
        {
            "artifact_id": "critic_reward_audit_markdown",
            "authority_role": "single_file_audit_summary",
            "relative_path": "agent/exchange/AppleToPlate_RECAP_status_critic_reward_audit.md",
        },
    ]


def _with_freshness(
    payload: dict[str, Any], freshness: dict[str, str]
) -> dict[str, Any]:
    payload = dict(payload)
    payload["freshness"] = dict(freshness)
    return payload


def _source_ref(
    repo_root: Path, *, artifact_id: str, relative_path: str
) -> dict[str, Any]:
    return apple_recap_execution_contract.build_read_only_authority_ref(
        repo_root=repo_root,
        artifact_id=artifact_id,
        authority_role="upstream",
        relative_path=relative_path,
    )


def _build_fixture_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repo_root = tmp_path / "repo"
    execution_root = repo_root / "agent" / "artifacts" / "apple_recap_exec"
    phase_dir = execution_root / "phase_a_tooling_draft"
    reward_dir = execution_root / "reward"
    freshness = _freshness_bundle()

    authority_specs = _authority_fixture_specs(repo_root)
    execution_contract = (
        apple_recap_execution_contract.build_execution_freeze_contract_draft(
            repo_root=repo_root,
            generated_at=freshness["timestamp"],
            execution_sha=freshness["execution_sha"],
            read_only_authority_ref_specs=authority_specs,
        )
    )
    execution_contract = _with_freshness(execution_contract, freshness)
    _write_json(
        execution_root / apple_recap_execution_contract.EXECUTION_CONTRACT_JSON_NAME,
        execution_contract,
    )

    baseline_refs = build_readonly_refs.build_baseline_refs_manifest(
        repo_root=repo_root,
        generated_at=freshness["timestamp"],
        execution_sha=freshness["execution_sha"],
        read_only_authority_ref_specs=authority_specs,
    )
    baseline_refs = _with_freshness(baseline_refs, freshness)
    baseline_refs["report_signature_sha256"] = _sign_payload(baseline_refs)
    baseline_refs_path = _write_json(
        phase_dir / build_readonly_refs.DEFAULT_OUTPUT.name,
        baseline_refs,
    )

    frozen_matrix = build_uplift_schemas.build_experiment_matrix_frozen(
        experiment_matrix_payload=_experiment_matrix_payload(),
        experiment_matrix_json=repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json",
        repo_root=repo_root,
        generated_at=freshness["timestamp"],
        execution_sha=freshness["execution_sha"],
    )
    frozen_matrix = _with_freshness(frozen_matrix, freshness)
    frozen_matrix["report_signature_sha256"] = _sign_payload(frozen_matrix)
    _write_json(phase_dir / build_uplift_schemas.FROZEN_MATRIX_JSON_NAME, frozen_matrix)

    uplift_schema = build_uplift_schemas.build_uplift_verdict_schema(
        frozen_matrix_payload=frozen_matrix,
        generated_at=freshness["timestamp"],
        execution_sha=freshness["execution_sha"],
    )
    uplift_schema = _with_freshness(uplift_schema, freshness)
    uplift_schema["report_signature_sha256"] = _sign_payload(uplift_schema)
    _write_json(
        phase_dir / build_uplift_schemas.UPLIFT_VERDICT_SCHEMA_JSON_NAME, uplift_schema
    )

    with (phase_dir / build_uplift_schemas.LEDGER_CSV_NAME).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(list(build_uplift_schemas.LEDGER_COLUMNS))

    carrier_report = {
        "schema_version": inspect_mainline_carrier.SCHEMA_VERSION,
        "training_text_field": "carrier_text_v1",
        "authority_violation_count": 0,
        "policy_condition_metadata_only": True,
        "full_scan_row_count": 2,
        "sample_row_count": 2,
        "carrier_parity_match_count": 2,
        "carrier_parity_mismatch_count": 0,
        "indicator_mode_counts": {"positive": 1, "omit": 1},
        "field_presence_summary": {},
        "authority_violation_examples": [],
        "sample_rows_artifact": inspect_mainline_carrier.SAMPLE_ROWS_CSV_NAME,
        "inspection_markdown_artifact": inspect_mainline_carrier.INSPECTION_MD_NAME,
        "parity_report_artifact": inspect_mainline_carrier.PARITY_REPORT_JSON_NAME,
        "freshness": dict(freshness),
        "source_artifacts": [
            _source_ref(
                repo_root,
                artifact_id="execution_contract",
                relative_path=_repo_relative(
                    repo_root,
                    execution_root
                    / apple_recap_execution_contract.EXECUTION_CONTRACT_JSON_NAME,
                ),
            )
        ],
    }
    _write_json(
        phase_dir / inspect_mainline_carrier.PARITY_REPORT_JSON_NAME, carrier_report
    )
    _write_text(
        phase_dir / inspect_mainline_carrier.INSPECTION_MD_NAME,
        "# carrier inspection\n",
    )

    carrier_panel_gate = {
        "schema_version": gr00t_carrier_panel_gate.REPORT_SCHEMA_VERSION,
        "artifact_kind": gr00t_carrier_panel_gate.REPORT_ARTIFACT_KIND,
        "status": "PASS",
        "gate_strength": "robust",
        "freshness": dict(freshness),
        "source_artifacts": [
            _source_ref(
                repo_root,
                artifact_id="baseline_refs_manifest",
                relative_path=_repo_relative(repo_root, baseline_refs_path),
            )
        ],
    }
    carrier_panel_gate["report_signature_sha256"] = _sign_payload(carrier_panel_gate)
    _write_json(phase_dir / "carrier_panel_gate.json", carrier_panel_gate)

    action_absorption = {
        "schema_version": gr00t_action_absorption_audit.REPORT_SCHEMA_VERSION,
        "artifact_kind": gr00t_action_absorption_audit.REPORT_ARTIFACT_KIND,
        "input_status": gr00t_action_absorption_audit.TARGET_STATUS,
        "target_status": gr00t_action_absorption_audit.TARGET_STATUS,
        "audit_status": "ready",
        "eligible_for_root_cause_audit": True,
        "freshness": dict(freshness),
        "source_artifacts": [
            _source_ref(
                repo_root,
                artifact_id="carrier_panel_gate",
                relative_path="agent/artifacts/apple_recap_exec/phase_a_tooling_draft/carrier_panel_gate.json",
            )
        ],
    }
    action_absorption["report_signature_sha256"] = _sign_payload(action_absorption)
    _write_json(
        phase_dir
        / gr00t_action_absorption_audit.ACTION_ABSORPTION_ROOT_CAUSE_JSON_NAME,
        action_absorption,
    )

    critic_scorecard = {
        "schema_version": critic_scorecard_all_splits.SCHEMA_VERSION,
        "artifact_kind": critic_scorecard_all_splits.ARTIFACT_KIND,
        "generated_at": freshness["timestamp"],
        "freshness": dict(freshness),
        "splits": {"test": {"row_count": 12}},
    }
    _write_json(phase_dir / "critic_scorecard_all_splits.json", critic_scorecard)

    critic_sample_pack = {
        "schema_version": critic_build_sample_pack.SCHEMA_VERSION,
        "artifact_kind": critic_build_sample_pack.ARTIFACT_KIND,
        "generated_at": freshness["timestamp"],
        "freshness": dict(freshness),
        "samples": [{"sample_id": "sample_001"}],
    }
    _write_json(phase_dir / "critic_sample_pack.json", critic_sample_pack)

    critic_episode_traces = {
        "schema_version": critic_build_episode_traces.SCHEMA_VERSION,
        "artifact_kind": critic_build_episode_traces.ARTIFACT_KIND,
        "generated_at": freshness["timestamp"],
        "freshness": dict(freshness),
        "episodes": [{"recap_episode_id": "episode_001"}],
    }
    _write_json(phase_dir / "critic_episode_traces.json", critic_episode_traces)

    reward_recommendation = {
        "schema_version": drop_events.REWARD_RECOMMENDATION_SCHEMA_VERSION,
        "artifact_kind": drop_events.REWARD_RECOMMENDATION_ARTIFACT_KIND,
        "formal_eligibility": "ALLOW",
        "reward_recommendation": drop_events.RECOMMENDATION_ELIGIBLE_FOR_MAINLINE,
        "mainline_reward_rerun_allowed": True,
        "failure_reasons": [],
        "freshness": dict(freshness),
    }
    _write_json(
        reward_dir / relabel_counterfactual_rewards.REWARD_RECOMMENDATION_JSON_NAME,
        reward_recommendation,
    )

    counterfactual_summary = {
        "schema_version": drop_events.COUNTERFACTUAL_REWARD_SUMMARY_SCHEMA_VERSION,
        "artifact_kind": drop_events.COUNTERFACTUAL_REWARD_SUMMARY_ARTIFACT_KIND,
        "mainline_candidate_variant": drop_events.COUNTERFACTUAL_VARIANT_V1,
        "freshness": dict(freshness),
        "variants": {
            drop_events.COUNTERFACTUAL_VARIANT_V1: {"affected_episode_count": 1}
        },
    }
    _write_json(
        reward_dir / relabel_counterfactual_rewards.COUNTERFACTUAL_SUMMARY_JSON_NAME,
        counterfactual_summary,
    )
    _write_text(
        reward_dir
        / relabel_counterfactual_rewards.REWARD_COUNTERFACTUAL_REPORT_MD_NAME,
        "# reward counterfactual report\n",
    )

    return repo_root, execution_root, freshness


def _repo_relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def test_happy_path_builds_markdown_and_json_pack(tmp_path: Path) -> None:
    repo_root, execution_root, freshness = _build_fixture_repo(tmp_path)
    out_md = repo_root / "agent/exchange/AppleToPlate_RECAP_final_report.md"
    out_json = execution_root / "final_report/final_verdict_pack.json"

    payload = build_apple_recap_final_report.materialize_apple_recap_final_report(
        execution_root=execution_root,
        out_md=out_md,
        out_json=out_json,
        repo_root=repo_root,
        generated_at=freshness["timestamp"],
    )
    written = _read_json(out_json)
    markdown = out_md.read_text(encoding="utf-8")

    assert payload == written
    assert payload["formal_eligibility"] == "ALLOW"
    assert payload["tooling_phase_only"] is True
    assert payload["freeze_context"]["execution_sha"] == freshness["execution_sha"]
    assert (
        payload["report_artifacts"]["markdown"]
        == "agent/exchange/AppleToPlate_RECAP_final_report.md"
    )
    assert payload["questions"]["Q1"]["reviewer_verdict"] is None
    assert payload["questions"]["Q2"]["status"] == "tooling_placeholder"
    assert payload["questions"]["Q3"]["referenced_artifact_ids"] == [
        "reward_recommendation",
        "counterfactual_reward_summary",
        "reward_counterfactual_report_markdown",
    ]
    assert "T8 tooling-phase" in markdown
    assert "Reviewer answer slot" in markdown
    assert "carrier_panel_gate" in markdown


def test_builder_rejects_stale_artifact_when_freshness_does_not_match(
    tmp_path: Path,
) -> None:
    repo_root, execution_root, _freshness = _build_fixture_repo(tmp_path)
    scorecard_path = (
        execution_root / "phase_a_tooling_draft/critic_scorecard_all_splits.json"
    )
    scorecard = _read_json(scorecard_path)
    scorecard["freshness"]["execution_sha"] = "wrong-execution-sha"
    _write_json(scorecard_path, scorecard)

    with pytest.raises(ValueError, match="stale artifact critic_scorecard_all_splits"):
        build_apple_recap_final_report.build_final_report_pack(
            execution_root=execution_root,
            repo_root=repo_root,
        )


def test_builder_rejects_missing_required_artifact(tmp_path: Path) -> None:
    repo_root, execution_root, _freshness = _build_fixture_repo(tmp_path)
    missing_path = (
        execution_root
        / "reward"
        / relabel_counterfactual_rewards.REWARD_RECOMMENDATION_JSON_NAME
    )
    missing_path.unlink()

    with pytest.raises(
        ValueError, match="missing required artifact reward_recommendation"
    ):
        build_apple_recap_final_report.build_final_report_pack(
            execution_root=execution_root,
            repo_root=repo_root,
        )


def test_builder_rejects_off_freeze_authority_ref_mismatch(tmp_path: Path) -> None:
    repo_root, execution_root, _freshness = _build_fixture_repo(tmp_path)
    baseline_refs_path = (
        execution_root
        / "phase_a_tooling_draft"
        / build_readonly_refs.DEFAULT_OUTPUT.name
    )
    payload = _read_json(baseline_refs_path)
    payload["read_only_authority_refs"][0]["content_sha256"] = "0" * 64
    payload["report_signature_sha256"] = _sign_payload(payload)
    _write_json(baseline_refs_path, payload)

    with pytest.raises(ValueError, match="authority ref digest mismatch"):
        build_apple_recap_final_report.build_final_report_pack(
            execution_root=execution_root,
            repo_root=repo_root,
        )


def test_builder_rejects_noncanonical_execution_root_lane(tmp_path: Path) -> None:
    repo_root, execution_root, _freshness = _build_fixture_repo(tmp_path)
    current_root = repo_root / "agent/artifacts/current/apple_recap_exec"
    shutil.copytree(execution_root, current_root)

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        build_apple_recap_final_report.build_final_report_pack(
            execution_root=current_root,
            repo_root=repo_root,
        )


def test_builder_rejects_invalid_authority_ref_before_stale_classification(
    tmp_path: Path,
) -> None:
    repo_root, execution_root, _freshness = _build_fixture_repo(tmp_path)
    baseline_refs_path = (
        execution_root
        / "phase_a_tooling_draft"
        / build_readonly_refs.DEFAULT_OUTPUT.name
    )
    payload = _read_json(baseline_refs_path)
    del payload["read_only_authority_refs"][0]["content_sha256"]
    payload["report_signature_sha256"] = _sign_payload(payload)
    _write_json(baseline_refs_path, payload)

    with pytest.raises(ValueError, match="invalid_input"):
        build_apple_recap_final_report.build_final_report_pack(
            execution_root=execution_root,
            repo_root=repo_root,
        )
