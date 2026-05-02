from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import successor_execution


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _authority_ref_fixtures(repo_root: Path) -> None:
    _write_json(
        repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/public_anchor/public_anchor_formal.json",
        {
            "schema_version": "public_anchor_formal_v1",
            "artifact_kind": "public_anchor_formal",
            "report_signature_sha256": "public-anchor-signature",
            "success_count": 5,
            "success_rate": 0.5,
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
        row["row_id"]: {
            "row_id": row["row_id"],
            "display_label": row["display_label"],
            "row_kind": "baseline" if row["display_label"] == "B0" else "experiment",
            "mainline_authority": True,
            "compare_to_row_id": None,
            "changed_axes": [],
            "summary": {"display_label": row["display_label"]},
        }
        for row in display_rows
    }
    return {
        "schema_version": "gr00t_experiment_matrix_v1",
        "artifact_kind": "gr00t_experiment_matrix",
        "generated_at": "2026-04-12T00:00:00+00:00",
        "display_rows": display_rows,
        "row_id_order": [row["row_id"] for row in display_rows],
        "rows": rows,
        "report_signature_sha256": "experiment-matrix-signature",
    }


def _blocked_root_authority(repo_root: Path) -> None:
    blocked_root = repo_root / successor_execution.BLOCKED_EXECUTION_ROOT
    _write_json(
        blocked_root
        / apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME,
        {
            "schema_version": "apple_recap_execution_freeze_contract_v1",
            "artifact_kind": "apple_recap_execution_freeze_contract",
            "execution_sha": successor_execution.BLOCKED_EXECUTION_SHA,
            "freshness": {"execution_sha": successor_execution.BLOCKED_EXECUTION_SHA},
        },
    )
    _write_json(
        blocked_root / "uplift_verdict.json",
        {
            "schema_version": "apple_recap_blocked_closeout_v1",
            "artifact_kind": "apple_recap_blocked_closeout",
            "execution_sha": successor_execution.BLOCKED_EXECUTION_SHA,
            "gating_eligible": False,
        },
    )
    _write_json(
        blocked_root / "baseline_refs_manifest.json",
        {"artifact_kind": "apple_recap_baseline_refs_manifest"},
    )
    _write_json(
        blocked_root / "experiment_matrix_frozen.json",
        {"artifact_kind": "apple_recap_experiment_matrix_frozen"},
    )
    _write_json(
        blocked_root / "B0_repro_band.json",
        {"artifact_kind": "apple_recap_b0_repro_band"},
    )
    _write_json(
        blocked_root / "carrier_parity_report.json",
        {
            "schema_version": "carrier_parity_report_v1",
            "artifact_kind": "carrier_parity_report",
            "authority_violation_count": 61246,
        },
    )
    _write_text(
        blocked_root / "carrier_sample_rows.csv",
        "row_number,sample_id\n1,sample_001\n",
    )
    _write_text(
        blocked_root / "carrier_inspection.md",
        "# blocked carrier inspection\n",
    )


def _baseline_suite(repo_root: Path) -> None:
    suite_root = repo_root / successor_execution.DEFAULT_BASELINE_SUITE_DIR
    _write_json(
        suite_root / "official_10ep_20000_20009/public_anchor_formal.json",
        {
            "schema_version": "gr00t_public_anchor_formal_v1",
            "artifact_kind": "gr00t_public_anchor_formal",
            "success_count": 5,
            "success_rate": 0.5,
        },
    )
    _write_json(
        suite_root / "repro_rerun_b_21000_21009_live/b0_bundle_eval.json",
        {
            "schema_version": "apple_recap_b0_bundle_eval_v1",
            "artifact_kind": "apple_recap_b0_bundle_eval",
            "success_count": 9,
            "success_rate": 0.9,
        },
    )
    _write_json(
        suite_root / "repro_rerun_c_22000_22009_live/b0_bundle_eval.json",
        {
            "schema_version": "apple_recap_b0_bundle_eval_v1",
            "artifact_kind": "apple_recap_b0_bundle_eval",
            "success_count": 6,
            "success_rate": 0.6,
        },
    )
    _write_json(
        suite_root / "extended_50ep_30000_30049_live/b0_bundle_eval.json",
        {
            "schema_version": "apple_recap_b0_bundle_eval_v1",
            "artifact_kind": "apple_recap_b0_bundle_eval",
            "success_count": 26,
            "success_rate": 0.52,
        },
    )


def _dummy_source_dataset(repo_root: Path) -> None:
    dataset_dir = repo_root / successor_execution.DEFAULT_SOURCE_DATASET_DIR
    _write_jsonl(
        dataset_dir / "episodes.jsonl",
        [
            {
                "episode_id": "episode_001",
                "prompt_raw": "pick up the apple and place it on the plate",
                "prompt_conditioned": "advantage positive pick up the apple and place it on the plate",
                "npz_path": "arrays/episode_001.npz",
            }
        ],
    )
    _write_jsonl(
        dataset_dir / "transitions.jsonl",
        [
            {
                "episode_id": "episode_001",
                "t": 0,
                "n_action_steps_executed": 1,
                "inner_rewards": [0.0],
                "inner_dones": [False],
            }
        ],
    )
    source_rows = []
    for row in _successor_label_rows():
        source_row = dict(row)
        del source_row["carrier_text_v1"]
        source_rows.append(source_row)
    _write_jsonl(dataset_dir / "m2_labels" / "labels.jsonl", source_rows)
    _write_json(
        dataset_dir / "m2_labels" / "stats.json",
        {
            "n_transitions": len(source_rows),
            "n_episodes": 1,
            "epsilon_strategy": "quantile",
            "epsilon_value": 0.1,
            "pos_ratio": 0.5,
            "advantage_mean": 0.0,
            "advantage_min": -0.75,
            "advantage_max": 0.75,
        },
    )
    (dataset_dir / "arrays").mkdir(parents=True, exist_ok=True)
    _write_json(
        repo_root / successor_execution.DEFAULT_SOURCE_CRITIC_DIR / "config.json",
        {
            "artifact_version": "multimodal_distributional_v1",
            "critic_type": "multimodal_distributional_v1",
            "smoke_backend": "qwen3_vl_late_fusion_v1",
        },
    )


def _write_worktree_file(repo_root: Path) -> None:
    _write_text(
        repo_root / "work/recap/scripts/dummy_successor_logic.py",
        "SUCCESSOR = True\n",
    )


def _git_stub_factory(
    responses: dict[tuple[str, ...], str],
):
    def _stub(
        repo_root: Path,
        *args: str,
        allow_failure: bool = False,
        default: str = "",
    ) -> str:
        key = tuple(args)
        if key in responses:
            return responses[key]
        if allow_failure:
            return str(default)
        raise AssertionError(f"unexpected git invocation: {key!r}")

    return _stub


def _successor_label_rows() -> list[dict[str, Any]]:
    prompt_raw = "pick up the apple and place it on the plate"
    positive_carrier = text_indicator.build_canonical_text_indicator(
        prompt_raw,
        text_indicator.TEXT_INDICATOR_POSITIVE,
    )
    negative_carrier = text_indicator.build_canonical_text_indicator(
        prompt_raw,
        text_indicator.TEXT_INDICATOR_NEGATIVE,
    )
    return [
        {
            "schema_version": "recap-v0",
            "code_version": "successor-test",
            "iter_tag": "recap_mainline_fresh_20260311_121500_k0",
            "episode_id": "episode_001",
            "t": 0,
            "return_G": 1.0,
            "value_V": 0.25,
            "advantage_A": 0.75,
            "epsilon_l": 0.1,
            "indicator_I": 1,
            "is_correction": False,
            "prompt_raw": prompt_raw,
            "prompt_conditioned": "advantage positive pick up the apple and place it on the plate",
            "carrier_text_v1": positive_carrier,
        },
        {
            "schema_version": "recap-v0",
            "code_version": "successor-test",
            "iter_tag": "recap_mainline_fresh_20260311_121500_k0",
            "episode_id": "episode_001",
            "t": 1,
            "return_G": -1.0,
            "value_V": -0.25,
            "advantage_A": -0.75,
            "epsilon_l": 0.1,
            "indicator_I": 0,
            "is_correction": False,
            "prompt_raw": prompt_raw,
            "prompt_conditioned": "advantage negative pick up the apple and place it on the plate",
            "carrier_text_v1": negative_carrier,
        },
    ]


def _prepare_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    _authority_ref_fixtures(repo_root)
    _blocked_root_authority(repo_root)
    _baseline_suite(repo_root)
    _dummy_source_dataset(repo_root)
    _write_worktree_file(repo_root)
    return repo_root


def test_materialize_successor_authority_keeps_blocked_root_hash_stable_and_binds_new_sha(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = _prepare_repo(tmp_path)
    successor_root = repo_root / successor_execution.SUCCESSOR_EXECUTION_ROOT
    git_sha = "feedfacefeedfacefeedfacefeedfacefeedface"
    responses = {
        ("rev-parse", "HEAD"): git_sha + "\n",
        ("branch", "--show-current"): "successor/r10d\n",
        ("rev-parse", "--abbrev-ref", "@{upstream}"): "origin/successor/r10d\n",
        (
            "status",
            "--short",
            "--branch",
        ): "## successor/r10d...origin/successor/r10d\n",
        (
            "status",
            "--porcelain=v1",
            "--branch",
            "-uall",
        ): "## successor/r10d...origin/successor/r10d\n",
    }
    monkeypatch.setattr(
        apple_recap_execution_contract,
        "_git_text",
        _git_stub_factory(responses),
    )
    result = successor_execution.materialize_successor_authority(
        repo_root=repo_root,
        successor_root=successor_root,
        blocked_root=repo_root / successor_execution.BLOCKED_EXECUTION_ROOT,
        source_dataset_dir=repo_root / successor_execution.DEFAULT_SOURCE_DATASET_DIR,
        critic_dir=repo_root / successor_execution.DEFAULT_SOURCE_CRITIC_DIR,
        baseline_suite_dir=repo_root / successor_execution.DEFAULT_BASELINE_SUITE_DIR,
        experiment_matrix_json=repo_root
        / successor_execution.DEFAULT_EXPERIMENT_MATRIX_JSON,
        freeze_timestamp="2026-04-12T06:30:00+00:00",
    )

    assert result["execution_sha"] == git_sha
    assert result["execution_sha"] != successor_execution.BLOCKED_EXECUTION_SHA
    assert result["blocked_root_hash_before"] == result["blocked_root_hash_after"]

    expected_execution_root = "agent/artifacts/apple_recap_exec_successor"
    for relative_path in successor_execution.SUCCESSOR_AUTHORITY_RELATIVE_PATHS:
        payload_or_text_path = successor_root / relative_path
        assert payload_or_text_path.exists()
        if payload_or_text_path.suffix == ".json":
            payload = _read_json(payload_or_text_path)
            validation = successor_execution.validate_successor_authority_artifact(
                payload,
                artifact_relative_path=relative_path,
                expected_execution_root=expected_execution_root,
                expected_execution_sha=git_sha,
            )
            assert validation["formal_eligibility"] == "ALLOW"

    freeze_contract = _read_json(successor_root / "execution_freeze_contract.json")
    assert freeze_contract["execution_root"] == expected_execution_root
    assert freeze_contract["execution_sha"] == git_sha
    assert freeze_contract["successor_context"]["mode"] == "authoritative_export_fix"

    carrier_report = _read_json(successor_root / "carrier_parity_report.json")
    assert carrier_report["authority_violation_count"] == 0
    assert carrier_report["authority_level"] == "authoritative_successor"
    assert carrier_report["gating_eligible"] is True
    assert carrier_report["successor_context"]["mode"] == "authoritative_export_fix"
    assert carrier_report["source_artifacts"][1]["relative_path"].endswith(
        "authoritative_export_fix_labels/materialization_manifest.json"
    )

    carrier_rows = _read_csv(successor_root / "carrier_sample_rows.csv")
    assert carrier_rows[0]["execution_root"] == expected_execution_root
    assert carrier_rows[0]["execution_sha"] == git_sha
    assert carrier_rows[0]["authority_level"] == "authoritative_successor"

    carrier_markdown = (successor_root / "carrier_inspection.md").read_text(
        encoding="utf-8"
    )
    assert (
        "execution_root: `agent/artifacts/apple_recap_exec_successor`"
        in carrier_markdown
    )
    assert "successor_mode: `authoritative_export_fix`" in carrier_markdown


def test_successor_root_write_attempt_into_blocked_root_fails_closed(
    tmp_path: Path,
) -> None:
    repo_root = _prepare_repo(tmp_path)
    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        successor_execution.materialize_successor_authority(
            repo_root=repo_root,
            successor_root=repo_root / successor_execution.BLOCKED_EXECUTION_ROOT,
            blocked_root=repo_root / successor_execution.BLOCKED_EXECUTION_ROOT,
        )


def test_successor_authority_rejects_noncanonical_current_lane_before_materialization(
    tmp_path: Path,
) -> None:
    repo_root = _prepare_repo(tmp_path)

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        successor_execution.materialize_successor_authority(
            repo_root=repo_root,
            successor_root=repo_root
            / "agent/artifacts/apple_recap_exec_successor/current",
            blocked_root=repo_root / successor_execution.BLOCKED_EXECUTION_ROOT,
        )


def test_successor_authority_rejects_noncanonical_reference_only_baseline_suite(
    tmp_path: Path,
) -> None:
    repo_root = _prepare_repo(tmp_path)

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        successor_execution.materialize_successor_authority(
            repo_root=repo_root,
            successor_root=repo_root / successor_execution.SUCCESSOR_EXECUTION_ROOT,
            blocked_root=repo_root / successor_execution.BLOCKED_EXECUTION_ROOT,
            source_dataset_dir=repo_root
            / successor_execution.DEFAULT_SOURCE_DATASET_DIR,
            critic_dir=repo_root / successor_execution.DEFAULT_SOURCE_CRITIC_DIR,
            baseline_suite_dir=repo_root
            / "agent/artifacts/apple_recap_exec/reference_only/baseline_suite",
            experiment_matrix_json=repo_root
            / successor_execution.DEFAULT_EXPERIMENT_MATRIX_JSON,
        )


def test_validate_successor_authority_rejects_ineligible_or_non_authoritative_payload() -> (
    None
):
    payload = {
        "execution_root": "agent/artifacts/apple_recap_exec_successor",
        "execution_sha": "feedfacefeedfacefeedfacefeedfacefeedface",
        "freshness": {
            "execution_sha": "feedfacefeedfacefeedfacefeedfacefeedface",
        },
        "authority_level": "research",
        "gating_eligible": False,
        "successor_context": {"mode": "authoritative_export_fix"},
    }
    validation = successor_execution.validate_successor_authority_artifact(
        payload,
        artifact_relative_path="carrier_parity_report.json",
        expected_execution_root="agent/artifacts/apple_recap_exec_successor",
        expected_execution_sha="feedfacefeedfacefeedfacefeedfacefeedface",
    )

    assert validation["formal_eligibility"] == "BLOCK"
    issue_codes = {issue["code"] for issue in validation["issues"]}
    assert "successor_gating_ineligible" in issue_codes
    assert "authority_level_mismatch" in issue_codes


def test_validate_successor_authority_rejects_phase_a_and_research_backfill_masquerade() -> (
    None
):
    payload = {
        "execution_root": "agent/artifacts/apple_recap_exec_successor",
        "execution_sha": "feedfacefeedfacefeedfacefeedfacefeedface",
        "freshness": {
            "execution_sha": "feedfacefeedfacefeedfacefeedfacefeedface",
        },
        "authority_level": "authoritative_successor",
        "gating_eligible": True,
        "successor_context": {"mode": "authoritative_export_fix"},
        "source_artifacts": [
            {
                "relative_path": "agent/artifacts/apple_recap_exec_successor/phase_a_tooling_draft/carrier_parity_report.json"
            },
            {
                "relative_path": "agent/artifacts/recap_datasets/fullsize_relabel_v1_carrier_backfill_v1/research_probe/probe_manifest.json"
            },
        ],
    }
    validation = successor_execution.validate_successor_authority_artifact(
        payload,
        artifact_relative_path="phase_a_tooling_draft/carrier_parity_report.json",
        expected_execution_root="agent/artifacts/apple_recap_exec_successor",
        expected_execution_sha="feedfacefeedfacefeedfacefeedfacefeedface",
    )

    assert validation["formal_eligibility"] == "BLOCK"
    issue_codes = [issue["code"] for issue in validation["issues"]]
    assert "phase_a_authority_forbidden" in issue_codes
    assert "research_backfill_authority_forbidden" in issue_codes
