from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import apple_recap_execution_contract


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
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
        {
            "schema_version": "gr00t_experiment_matrix_v1",
            "artifact_kind": "gr00t_experiment_matrix",
            "report_signature_sha256": "experiment-matrix-signature",
        },
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


def _build_payload(repo_root: Path) -> dict[str, Any]:
    return apple_recap_execution_contract.build_execution_freeze_contract_draft(
        repo_root=repo_root,
        generated_at="2026-04-12T00:00:00+00:00",
        read_only_authority_ref_specs=_authority_fixture_specs(repo_root),
    )


def test_materialize_execution_contract_draft_writes_complete_payload(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    output_dir = repo_root / apple_recap_execution_contract.DEFAULT_OUTPUT_DIR

    payload = (
        apple_recap_execution_contract.materialize_execution_freeze_contract_draft(
            output_dir=output_dir,
            repo_root=repo_root,
            generated_at="2026-04-12T00:00:00+00:00",
            read_only_authority_ref_specs=_authority_fixture_specs(repo_root),
        )
    )
    written = _read_json(
        output_dir / apple_recap_execution_contract.EXECUTION_CONTRACT_JSON_NAME
    )
    validation = (
        apple_recap_execution_contract.validate_execution_freeze_contract_draft(
            payload,
            repo_root=repo_root,
        )
    )

    assert written == payload
    assert validation["formal_eligibility"] == "ALLOW"
    assert validation["issues"] == []
    assert (
        payload["execution_sha"] == apple_recap_execution_contract.UNSET_EXECUTION_SHA
    )
    assert (
        payload["core"]["commit"] == apple_recap_execution_contract.UNSET_EXECUTION_SHA
    )
    assert payload["core_digest"] == apple_recap_execution_contract.core_digest(
        payload["core"]
    )
    assert payload["critic_baseline_authority"] == "task7_real_critic_v2"
    assert payload["critic_candidate_track"] == "task7_real_critic_v3"
    assert (
        payload["threshold_policy"] == apple_recap_execution_contract.THRESHOLD_POLICY
    )
    assert len(payload["historical_reference_commits"]) == 2
    assert len(payload["read_only_authority_refs"]) == 7
    assert (
        payload["read_only_authority_refs"][0]["artifact_kind"]
        == "public_anchor_formal"
    )
    assert (
        payload["read_only_authority_refs"][0]["schema_version"]
        == "public_anchor_formal_v1"
    )
    assert (
        payload["read_only_authority_refs"][0]["report_signature_sha256"]
        == "public-anchor-signature"
    )
    assert payload[
        "report_signature_sha256"
    ] == apple_recap_execution_contract._signature_for_contract(payload)


def test_build_read_only_authority_ref_rejects_off_repo_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    off_repo_path = tmp_path / "outside.json"
    _write_json(
        off_repo_path,
        {
            "schema_version": "off_repo_fixture_v1",
            "artifact_kind": "off_repo_fixture",
            "report_signature_sha256": "fixture-signature",
        },
    )

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="off_repo_fixture",
            authority_role="upstream",
            relative_path=off_repo_path,
        )


def test_materialize_execution_contract_draft_rejects_noncanonical_current_lane(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    noncanonical_output_dir = repo_root / "agent/artifacts/apple_recap_exec/current"

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        apple_recap_execution_contract.materialize_execution_freeze_contract_draft(
            output_dir=noncanonical_output_dir,
            repo_root=repo_root,
            generated_at="2026-04-12T00:00:00+00:00",
            read_only_authority_ref_specs=_authority_fixture_specs(repo_root),
        )


def test_materialize_execution_contract_draft_rejects_noncanonical_alias_authority_ref(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    output_dir = repo_root / apple_recap_execution_contract.DEFAULT_OUTPUT_DIR
    specs = _authority_fixture_specs(repo_root)
    alias_relative_path = (
        "agent/artifacts/alias/public_anchor/public_anchor_formal.json"
    )
    _write_json(
        repo_root / alias_relative_path,
        {
            "schema_version": "public_anchor_formal_v1",
            "artifact_kind": "public_anchor_formal",
            "report_signature_sha256": "alias-public-anchor-signature",
        },
    )
    specs[0] = {
        **specs[0],
        "relative_path": alias_relative_path,
    }

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        apple_recap_execution_contract.materialize_execution_freeze_contract_draft(
            output_dir=output_dir,
            repo_root=repo_root,
            generated_at="2026-04-12T00:00:00+00:00",
            read_only_authority_ref_specs=specs,
        )


def test_validator_blocks_missing_required_field(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    payload = _build_payload(repo_root)
    del payload["critic_candidate_track"]
    payload["report_signature_sha256"] = (
        apple_recap_execution_contract._signature_for_contract(payload)
    )

    validation = (
        apple_recap_execution_contract.validate_execution_freeze_contract_draft(
            payload,
            repo_root=repo_root,
        )
    )

    assert validation["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "wrong_type"
        and issue["field_path"] == "critic_candidate_track"
        for issue in validation["issues"]
    )


def test_validator_fail_closes_when_execution_sha_reuses_historical_commit(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    payload = apple_recap_execution_contract.build_execution_freeze_contract_draft(
        repo_root=repo_root,
        generated_at="2026-04-12T00:00:00+00:00",
        execution_sha=apple_recap_execution_contract.DEFAULT_HISTORICAL_REFERENCE_COMMITS[
            0
        ],
        read_only_authority_ref_specs=_authority_fixture_specs(repo_root),
    )

    validation = (
        apple_recap_execution_contract.validate_execution_freeze_contract_draft(
            payload,
            repo_root=repo_root,
        )
    )

    assert validation["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "historical_execution_authority_conflict"
        and issue["field_path"] == "execution_sha"
        for issue in validation["issues"]
    )


def test_build_execution_contract_draft_preserves_unset_execution_sha(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"

    payload = _build_payload(repo_root)
    validation = (
        apple_recap_execution_contract.validate_execution_freeze_contract_draft(
            copy.deepcopy(payload),
            repo_root=repo_root,
        )
    )

    assert payload["execution_sha"] == "UNSET_UNTIL_T1B"
    assert payload["core"]["commit"] == "UNSET_UNTIL_T1B"
    assert validation["formal_eligibility"] == "ALLOW"
    assert validation["normalized_contract"]["execution_sha"] == "UNSET_UNTIL_T1B"


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


def _materialize_draft_for_finalize(
    repo_root: Path, output_dir: Path
) -> dict[str, Any]:
    payload = apple_recap_execution_contract.build_execution_freeze_contract_draft(
        repo_root=repo_root,
        generated_at="2026-04-12T00:00:00+00:00",
        read_only_authority_ref_specs=_authority_fixture_specs(repo_root),
    )
    _write_json(
        output_dir / apple_recap_execution_contract.EXECUTION_CONTRACT_JSON_NAME,
        payload,
    )
    return payload


def test_materialize_final_execution_freeze_writes_snapshot_and_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_dir = repo_root / "agent/artifacts/apple_recap_exec"
    work_file = repo_root / "work/recap/scripts/demo_tool.py"
    work_file.parent.mkdir(parents=True, exist_ok=True)
    work_file.write_text("print('freeze')\n", encoding="utf-8")
    _materialize_draft_for_finalize(repo_root, output_dir)

    responses = {
        ("rev-parse", "HEAD"): "feedfacefeedfacefeedfacefeedfacefeedface\n",
        ("branch", "--show-current"): "0x\n",
        ("rev-parse", "--abbrev-ref", "@{upstream}"): "private/0x\n",
        ("status", "--short", "--branch"): "## 0x...private/0x [ahead 3]\n",
        (
            "status",
            "--porcelain=v1",
            "--branch",
            "-uall",
        ): "## 0x...private/0x [ahead 3]\n",
    }
    monkeypatch.setattr(
        apple_recap_execution_contract,
        "_git_text",
        _git_stub_factory(responses),
    )

    payload = apple_recap_execution_contract.materialize_final_execution_freeze(
        output_dir=output_dir,
        repo_root=repo_root,
        freeze_timestamp="2026-04-12T01:23:45+00:00",
    )
    snapshot = _read_json(
        output_dir / apple_recap_execution_contract.REPO_SNAPSHOT_JSON_NAME
    )
    repo_commit = (
        output_dir / apple_recap_execution_contract.REPO_COMMIT_TXT_NAME
    ).read_text(encoding="utf-8")
    repo_status = (
        output_dir / apple_recap_execution_contract.REPO_STATUS_TXT_NAME
    ).read_text(encoding="utf-8")
    validation = (
        apple_recap_execution_contract.validate_execution_freeze_contract_final(
            payload,
            repo_root=repo_root,
        )
    )
    integrity = apple_recap_execution_contract.check_execution_freeze_integrity(
        final_contract_json=(
            output_dir
            / apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME
        ),
        repo_root=repo_root,
    )

    assert validation["formal_eligibility"] == "ALLOW"
    assert payload["execution_sha"] == "feedfacefeedfacefeedfacefeedfacefeedface"
    assert payload["policy"]["freeze_phase"] == "phase_b_final_execution_freeze"
    assert payload["policy"]["execution_sha_placeholder_allowed"] is False
    assert payload["freshness"]["manifest_hash"] == snapshot["toolchain_manifest_hash"]
    assert (
        payload["freeze_integrity"]["toolchain_manifest_hash"]
        == snapshot["toolchain_manifest_hash"]
    )
    assert payload["freeze_integrity"]["allowed_post_freeze_write_scopes"] == list(
        apple_recap_execution_contract.ALLOWED_POST_FREEZE_WRITE_SCOPES
    )
    assert repo_commit.strip() == payload["execution_sha"]
    assert repo_status == "## 0x...private/0x [ahead 3]\n"
    assert snapshot["working_tree_clean_except_allowed"] is True
    assert snapshot["frozen_worktree_overrides"] == []
    assert integrity["formal_eligibility"] == "ALLOW"
    assert integrity["issues"] == []


def test_materialize_final_execution_freeze_blocks_dirty_worktree_logic_and_tests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_dir = repo_root / "agent/artifacts/apple_recap_exec"
    work_file = repo_root / "work/recap/scripts/demo_tool.py"
    extra_dirty_file = repo_root / "tests/recap/test_demo_tool.py"
    work_file.parent.mkdir(parents=True, exist_ok=True)
    extra_dirty_file.parent.mkdir(parents=True, exist_ok=True)
    work_file.write_text("print('freeze')\n", encoding="utf-8")
    extra_dirty_file.write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    _materialize_draft_for_finalize(repo_root, output_dir)

    responses = {
        ("rev-parse", "HEAD"): "feedfacefeedfacefeedfacefeedfacefeedface\n",
        ("branch", "--show-current"): "0x\n",
        ("rev-parse", "--abbrev-ref", "@{upstream}"): "private/0x\n",
        (
            "status",
            "--short",
            "--branch",
        ): "## 0x...private/0x [ahead 3]\n M work/recap/scripts/demo_tool.py\n?? tests/recap/test_demo_tool.py\n",
        (
            "status",
            "--porcelain=v1",
            "--branch",
            "-uall",
        ): "## 0x...private/0x [ahead 3]\n M work/recap/scripts/demo_tool.py\n?? tests/recap/test_demo_tool.py\n",
    }
    monkeypatch.setattr(
        apple_recap_execution_contract,
        "_git_text",
        _git_stub_factory(responses),
    )

    with pytest.raises(
        ValueError,
        match=r"final freeze blocks dirty runnable authority paths under work/\*\* or tests/\*\*",
    ):
        apple_recap_execution_contract.materialize_final_execution_freeze(
            output_dir=output_dir,
            repo_root=repo_root,
            freeze_timestamp="2026-04-12T01:23:45+00:00",
        )


def test_freeze_integrity_blocks_post_freeze_worktree_logic_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_dir = repo_root / "agent/artifacts/apple_recap_exec"
    work_file = repo_root / "work/recap/scripts/demo_tool.py"
    work_file.parent.mkdir(parents=True, exist_ok=True)
    work_file.write_text("print('freeze')\n", encoding="utf-8")
    _materialize_draft_for_finalize(repo_root, output_dir)

    responses = {
        ("rev-parse", "HEAD"): "feedfacefeedfacefeedfacefeedfacefeedface\n",
        ("branch", "--show-current"): "0x\n",
        ("rev-parse", "--abbrev-ref", "@{upstream}"): "private/0x\n",
        ("status", "--short", "--branch"): "## 0x...private/0x [ahead 3]\n",
        (
            "status",
            "--porcelain=v1",
            "--branch",
            "-uall",
        ): "## 0x...private/0x [ahead 3]\n",
    }
    monkeypatch.setattr(
        apple_recap_execution_contract,
        "_git_text",
        _git_stub_factory(responses),
    )
    apple_recap_execution_contract.materialize_final_execution_freeze(
        output_dir=output_dir,
        repo_root=repo_root,
        freeze_timestamp="2026-04-12T01:23:45+00:00",
    )

    work_file.write_text("print('drifted after freeze')\n", encoding="utf-8")
    integrity = apple_recap_execution_contract.check_execution_freeze_integrity(
        final_contract_json=(
            output_dir
            / apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME
        ),
        repo_root=repo_root,
    )

    assert integrity["formal_eligibility"] == "BLOCK"
    assert any(
        issue["code"] == "toolchain_manifest_hash_drift"
        for issue in integrity["issues"]
    )


def test_freeze_integrity_blocks_legacy_snapshot_with_frozen_runnable_dirty_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_dir = repo_root / "agent/artifacts/apple_recap_exec"
    work_file = repo_root / "work/recap/scripts/demo_tool.py"
    work_file.parent.mkdir(parents=True, exist_ok=True)
    work_file.write_text("print('freeze')\n", encoding="utf-8")
    _materialize_draft_for_finalize(repo_root, output_dir)

    responses = {
        ("rev-parse", "HEAD"): "feedfacefeedfacefeedfacefeedfacefeedface\n",
        ("branch", "--show-current"): "0x\n",
        ("rev-parse", "--abbrev-ref", "@{upstream}"): "private/0x\n",
        ("status", "--short", "--branch"): "## 0x...private/0x [ahead 3]\n",
        (
            "status",
            "--porcelain=v1",
            "--branch",
            "-uall",
        ): "## 0x...private/0x [ahead 3]\n",
    }
    monkeypatch.setattr(
        apple_recap_execution_contract,
        "_git_text",
        _git_stub_factory(responses),
    )
    apple_recap_execution_contract.materialize_final_execution_freeze(
        output_dir=output_dir,
        repo_root=repo_root,
        freeze_timestamp="2026-04-12T01:23:45+00:00",
    )

    snapshot_path = output_dir / apple_recap_execution_contract.REPO_SNAPSHOT_JSON_NAME
    snapshot = _read_json(snapshot_path)
    snapshot["frozen_worktree_overrides"] = [
        {
            "relative_path": "work/recap/scripts/demo_tool.py",
            "status_code": " M",
            "raw_line": " M work/recap/scripts/demo_tool.py",
            "path_kind": "file",
            "content_sha256": apple_recap_execution_contract._sha256_file(work_file),
        }
    ]
    _write_json(snapshot_path, snapshot)

    integrity = apple_recap_execution_contract.check_execution_freeze_integrity(
        final_contract_json=(
            output_dir
            / apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME
        ),
        repo_root=repo_root,
    )

    assert integrity["formal_eligibility"] == "BLOCK"
    assert integrity["frozen_runnable_dirty_paths"] == [
        "work/recap/scripts/demo_tool.py"
    ]
    assert any(
        issue["code"] == "invalid_frozen_runnable_dirty_path"
        for issue in integrity["issues"]
    )


def test_freeze_integrity_allows_post_freeze_artifact_scope_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_dir = repo_root / "agent/artifacts/apple_recap_exec"
    work_file = repo_root / "work/recap/scripts/demo_tool.py"
    work_file.parent.mkdir(parents=True, exist_ok=True)
    work_file.write_text("print('freeze')\n", encoding="utf-8")
    _materialize_draft_for_finalize(repo_root, output_dir)

    clean_responses = {
        ("rev-parse", "HEAD"): "feedfacefeedfacefeedfacefeedfacefeedface\n",
        ("branch", "--show-current"): "0x\n",
        ("rev-parse", "--abbrev-ref", "@{upstream}"): "private/0x\n",
        ("status", "--short", "--branch"): "## 0x...private/0x [ahead 3]\n",
        (
            "status",
            "--porcelain=v1",
            "--branch",
            "-uall",
        ): "## 0x...private/0x [ahead 3]\n",
    }
    monkeypatch.setattr(
        apple_recap_execution_contract,
        "_git_text",
        _git_stub_factory(clean_responses),
    )
    apple_recap_execution_contract.materialize_final_execution_freeze(
        output_dir=output_dir,
        repo_root=repo_root,
        freeze_timestamp="2026-04-12T01:23:45+00:00",
    )

    allowed_only_responses = {
        **clean_responses,
        ("status", "--short", "--branch"): (
            "## 0x...private/0x [ahead 3]\n"
            "?? agent/exchange/AppleToPlate_RECAP_final_report.md\n"
        ),
        (
            "status",
            "--porcelain=v1",
            "--branch",
            "-uall",
        ): (
            "## 0x...private/0x [ahead 3]\n"
            "?? agent/exchange/AppleToPlate_RECAP_final_report.md\n"
        ),
    }
    monkeypatch.setattr(
        apple_recap_execution_contract,
        "_git_text",
        _git_stub_factory(allowed_only_responses),
    )
    allowed_report = repo_root / "agent/exchange/AppleToPlate_RECAP_final_report.md"
    allowed_report.parent.mkdir(parents=True, exist_ok=True)
    allowed_report.write_text("# tooling placeholder\n", encoding="utf-8")
    integrity = apple_recap_execution_contract.check_execution_freeze_integrity(
        final_contract_json=(
            output_dir
            / apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME
        ),
        repo_root=repo_root,
    )

    assert integrity["formal_eligibility"] == "ALLOW"
    assert integrity["issues"] == []
