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


from work.recap import experiment_matrix
from work.recap.scripts import build_readonly_refs
from work.recap.scripts import build_uplift_schemas


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


def _row(
    row_id: str, display_label: str, *, mainline_authority: bool
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
        {"display_label": "B1", "row_id": "g1_b1_oldworld_c1"},
        {"display_label": "E1", "row_id": "g1_e1_text_indicator_s1"},
        {"display_label": "E2", "row_id": "g1_e2_text_indicator_s2"},
        {
            "display_label": "E3",
            "row_id": "g1_e3_text_indicator_s2_positive_duplication",
        },
        {"display_label": "E4", "row_id": "g1_e4_text_indicator_s2_task_phase_epsilon"},
        {"display_label": "DX", "row_id": "g1_dx_debug_row"},
    ]
    rows = {
        "g1_b0_public_anchor": _row(
            "g1_b0_public_anchor",
            "B0",
            mainline_authority=True,
        ),
        "g1_b1_oldworld_c1": _row(
            "g1_b1_oldworld_c1",
            "B1",
            mainline_authority=False,
        ),
        "g1_e1_text_indicator_s1": _row(
            "g1_e1_text_indicator_s1",
            "E1",
            mainline_authority=True,
        ),
        "g1_e2_text_indicator_s2": _row(
            "g1_e2_text_indicator_s2",
            "E2",
            mainline_authority=True,
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
        "g1_dx_debug_row": _row(
            "g1_dx_debug_row",
            "DX",
            mainline_authority=False,
        ),
    }
    payload = {
        "schema_version": experiment_matrix.EXPERIMENT_MATRIX_SCHEMA_VERSION,
        "artifact_kind": experiment_matrix.EXPERIMENT_MATRIX_ARTIFACT_KIND,
        "generated_at": "2026-04-12T00:00:00+00:00",
        "display_rows": display_rows,
        "row_id_order": [row["row_id"] for row in display_rows],
        "rows": rows,
    }
    payload["report_signature_sha256"] = "experiment-matrix-signature"
    return payload


def test_build_readonly_refs_happy_path_with_json_metadata(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    output = tmp_path / "out" / build_readonly_refs.DEFAULT_OUTPUT.name

    payload = build_readonly_refs.materialize_baseline_refs_manifest(
        output=output,
        repo_root=repo_root,
        generated_at="2026-04-12T00:00:00+00:00",
        read_only_authority_ref_specs=_authority_fixture_specs(repo_root),
    )
    written = _read_json(output)

    assert written == payload
    assert payload["schema_version"] == build_readonly_refs.SCHEMA_VERSION
    assert payload["artifact_kind"] == build_readonly_refs.ARTIFACT_KIND
    assert payload["execution_sha"] == "UNSET_UNTIL_T1B"
    assert payload["core"]["commit"] == "UNSET_UNTIL_T1B"
    assert payload["core_digest"]
    assert (
        payload["freeze_policy"]["missing_required_authority_ref_behavior"]
        == "fail_closed"
    )
    assert len(payload["read_only_authority_refs"]) == 7

    first_ref = payload["read_only_authority_refs"][0]
    assert first_ref["artifact_id"] == "public_anchor_formal"
    assert first_ref["authority_role"] == "official_public_anchor"
    assert first_ref["must_exist"] is True
    assert first_ref["read_only"] is True
    assert first_ref["artifact_kind"] == "public_anchor_formal"
    assert first_ref["schema_version"] == "public_anchor_formal_v1"
    assert first_ref["report_signature_sha256"] == "public-anchor-signature"
    assert payload["report_signature_sha256"]


def test_build_readonly_refs_fails_closed_when_authority_ref_missing(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    specs = _authority_fixture_specs(repo_root)
    missing_path = repo_root / specs[0]["relative_path"]
    missing_path.unlink()

    with pytest.raises(ValueError, match="read-only authority ref does not exist"):
        build_readonly_refs.build_baseline_refs_manifest(
            repo_root=repo_root,
            generated_at="2026-04-12T00:00:00+00:00",
            read_only_authority_ref_specs=specs,
        )


def test_build_uplift_schemas_excludes_diagnostic_rows_from_mainline_matrix(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _authority_fixture_specs(repo_root)
    output_dir = tmp_path / "out"
    experiment_matrix_json = (
        repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json"
    )

    result = build_uplift_schemas.materialize_uplift_schemas(
        output_dir=output_dir,
        experiment_matrix_json=experiment_matrix_json,
        repo_root=repo_root,
        generated_at="2026-04-12T00:00:00+00:00",
    )
    frozen = _read_json(Path(result["experiment_matrix_frozen"]))

    assert [row["display_label"] for row in frozen["display_rows"]] == [
        "B0",
        "E1",
        "E2",
        "E3",
        "E4",
    ]
    assert frozen["row_id_order"] == [
        "g1_b0_public_anchor",
        "g1_e1_text_indicator_s1",
        "g1_e2_text_indicator_s2",
        "g1_e3_text_indicator_s2_positive_duplication",
        "g1_e4_text_indicator_s2_task_phase_epsilon",
    ]
    assert [row["display_label"] for row in frozen["diagnostic_display_rows"]] == [
        "B1",
        "DX",
    ]
    assert set(frozen["rows"]) == set(frozen["row_id_order"])
    assert set(frozen["diagnostic_rows"]) == {
        "g1_b1_oldworld_c1",
        "g1_dx_debug_row",
    }
    assert frozen["rows"]["g1_b0_public_anchor"]["diagnostic_only"] is False
    assert (
        frozen["diagnostic_rows"]["g1_b1_oldworld_c1"]["main_verdict_eligible"] is False
    )
    assert (
        frozen["diagnostic_rows"]["g1_dx_debug_row"]["external_reference_only"] is True
    )
    assert frozen["freeze_policy"]["diagnostic_rows_not_part_of_main_verdict"] is True


def test_build_uplift_schemas_emits_ledger_columns_and_verdict_schema(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _authority_fixture_specs(repo_root)
    output_dir = tmp_path / "out"
    experiment_matrix_json = (
        repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json"
    )

    result = build_uplift_schemas.materialize_uplift_schemas(
        output_dir=output_dir,
        experiment_matrix_json=experiment_matrix_json,
        repo_root=repo_root,
        generated_at="2026-04-12T00:00:00+00:00",
    )

    with Path(result["run_ledger_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.reader(handle))
    schema_payload = _read_json(Path(result["uplift_verdict_schema"]))

    assert rows == [list(build_uplift_schemas.LEDGER_COLUMNS)]
    assert (
        schema_payload["schema_version"]
        == build_uplift_schemas.UPLIFT_VERDICT_SCHEMA_VERSION
    )
    assert (
        schema_payload["artifact_kind"]
        == build_uplift_schemas.UPLIFT_VERDICT_ARTIFACT_KIND
    )
    assert schema_payload["ledger_columns"] == list(build_uplift_schemas.LEDGER_COLUMNS)
    assert schema_payload["required_mainline_rows"] == ["B0", "E1", "E2"]
    assert schema_payload["mainline_rows_required_for_verdict"] == [
        "B0",
        "E1",
        "E2",
        "E3",
        "E4",
    ]
    assert schema_payload["diagnostic_rows_allowed_outside_mainline"] is True
    assert schema_payload["verdict_record"]["required"] == list(
        build_uplift_schemas.LEDGER_COLUMNS
    )
    assert schema_payload["verdict_record"]["properties"]["row_id"]["enum"] == [
        "B0",
        "E1",
        "E2",
    ]
    assert schema_payload["verdict_record"]["properties"]["row_signal"]["enum"] == [
        "screen_positive",
        "screen_flat",
        "screen_negative",
        "screen_inconclusive",
    ]
    assert schema_payload["verdict_record"]["properties"]["verdict"]["enum"] == [
        "accepted_uplift",
        "no_material_uplift",
        "rejected_regression",
        "inconclusive_rerun_needed",
    ]
