from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.safe_sft.t8_2_final_verifier import (  # noqa: E402
    ALLOWED_T8_2_FINAL_DECISIONS,
    main,
    sha256_file,
    verify_t8_2_artifact_root,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_complete_artifact_root(tmp_path: Path, *, selected_base_success: int = 10, final_decision: str = "SAFE_SFT_NONCOLLAPSE_PRELIM") -> Path:
    root = tmp_path / "t8_2_run"
    root.mkdir()
    source_dir = tmp_path / "source_t8_1"
    source_dir.mkdir()
    source_final = source_dir / "final_decision.json"
    source_summary = source_dir / "t8_1_summary.md"
    source_interpretation = source_dir / "t8_1_verified_interpretation.md"
    source_t8_dir = source_dir / "t8_source_run"
    source_t8_dir.mkdir()
    _write_json(source_final, {"final_decision": "BASE_SEEDS_TOO_HARD"})
    source_summary.write_text("# T8.1 summary\n", encoding="utf-8")
    source_interpretation.write_text("# T8.1 interpretation\n", encoding="utf-8")

    evidence_rows = []
    for name, path in (
        ("t8_1_final", source_final),
        ("t8_1_summary", source_summary),
        ("t8_1_interpretation", source_interpretation),
    ):
        evidence_rows.append(
            {"name": name, "path": str(path), "exists": True, "sha256": sha256_file(path)}
        )
    evidence_rows.append({"name": "t8_source_run", "path": str(source_t8_dir), "exists": True})
    _write_json(
        root / "evidence_lock.json",
        {
            "status": "PASS",
            "t8_1_final_decision": "BASE_SEEDS_TOO_HARD",
            "forbidden_branch_status": "FORBIDDEN_ROUTES_REJECTED",
            "required": evidence_rows,
        },
    )
    _write_json(
        root / "command_manifest.json",
        {
            "commands": [
                "timeout 60 env CUDA_VISIBLE_DEVICES=1 python -m work.recap.safe_sft.t8_2_seed_scout --eval"
            ],
            "submodule_status_short": [],
            "training_allowed": False,
            "optimizer_step_allowed": False,
            "checkpoint_update_allowed": False,
            "guarded_recap_allowed": False,
            "fatg_allowed": False,
            "per_edge_lora_allowed": False,
            "full_scope_update_allowed": False,
            "lora_merge_allowed": False,
        },
    )
    _write_json(
        root / "seed_scan_manifest.json",
        {
            "seed_start": 2026050800,
            "max_candidate_seeds": 200,
            "max_wall_time": "24h",
            "max_episode_steps": 720,
            "base_policy_path": "/models/base",
            "canonical_surface_path": "/models/canonical",
            "exact_command_template": "timeout ... env CUDA_VISIBLE_DEVICES=1 ...",
        },
    )

    candidate_fields = [
        "seed",
        "base_success",
        "base_reached",
        "base_lifted",
        "base_failure_mode",
        "reached_t",
        "lifted_t",
        "success_t",
        "apple_to_plate_min_after_lift",
        "stratum",
        "selected",
        "exclusion_reason",
        "steps_jsonl",
    ]
    candidate_rows = []
    for offset in range(selected_base_success):
        candidate_rows.append(
            {
                "seed": str(1000 + offset),
                "base_success": "true",
                "base_reached": "true",
                "base_lifted": "true",
                "base_failure_mode": "success",
                "reached_t": "3",
                "lifted_t": "5",
                "success_t": "8",
                "apple_to_plate_min_after_lift": "0.04",
                "stratum": "BASE_SUCCESS",
                "selected": "true",
                "exclusion_reason": "",
                "steps_jsonl": "steps.jsonl",
            }
        )
    candidate_rows.append(
        {
            "seed": "9999",
            "base_success": "false",
            "base_reached": "false",
            "base_lifted": "false",
            "base_failure_mode": "never_reached_apple",
            "reached_t": "",
            "lifted_t": "",
            "success_t": "",
            "apple_to_plate_min_after_lift": "",
            "stratum": "BASE_NEVER_REACHED",
            "selected": "false",
            "exclusion_reason": "underfilled_non_material_appendix",
            "steps_jsonl": "",
        }
    )
    _write_csv(root / "candidate_seed_scout.csv", candidate_fields, candidate_rows)
    _write_json(
        root / "seed_bank.json",
        {
            "counts_by_stratum": {"BASE_SUCCESS": selected_base_success},
            "selected_seeds": [1000 + offset for offset in range(selected_base_success)],
        },
    )
    _write_json(root / "seed_bank_deficits.json", {"BASE_SUCCESS": max(0, 10 - selected_base_success)})

    paired_rows: list[dict[str, Any]] = []
    for offset in range(selected_base_success):
        seed = 1000 + offset
        for row_id in ("B0", "B1", "B2", "S2"):
            paired_rows.append(
                {
                    "seed": seed,
                    "stratum": "BASE_SUCCESS",
                    "row_id": row_id,
                    "policy_or_splice": row_id,
                    "success": True,
                    "reached": True,
                    "lifted": True,
                    "failure_mode": "success",
                    "reached_t": 3,
                    "lifted_t": 5,
                    "success_t": 8,
                    "apple_to_plate_min_after_lift": 0.04,
                    "reached_plate_proxy": True,
                    "forbidden_scope_pass": True,
                    "steps_jsonl": "steps.jsonl",
                }
            )
    _write_jsonl(root / "paired_eval_per_seed.jsonl", paired_rows)
    summary_fields = [
        "ID",
        "seeds",
        "success",
        "reached",
        "lifted",
        "lift_given_reached",
        "success_given_lifted",
        "apple_to_plate_min_after_lift",
        "reached_plate_proxy",
        "failure_modes",
    ]
    _write_csv(
        root / "paired_eval_summary.csv",
        summary_fields,
        [
            {
                "ID": row_id,
                "seeds": str(selected_base_success),
                "success": str(selected_base_success),
                "reached": str(selected_base_success),
                "lifted": str(selected_base_success),
                "lift_given_reached": "1.0",
                "success_given_lifted": "1.0",
                "apple_to_plate_min_after_lift": "0.04",
                "reached_plate_proxy": str(selected_base_success),
                "failure_modes": "{}",
            }
            for row_id in ("B0", "B1", "B2", "S2")
        ],
    )
    _write_json(root / "control_regression_report.json", {"status": "PASS"})
    _write_csv(
        root / "candidate_eval_summary.csv",
        ["ID", "n_seeds", "success", "reached", "lifted", "delta_vs_B0", "delta_vs_B2"],
        [{"ID": "S2", "n_seeds": selected_base_success, "success": selected_base_success, "reached": selected_base_success, "lifted": selected_base_success, "delta_vs_B0": 0, "delta_vs_B2": 0}],
    )
    _write_json(root / "stratum_effects.json", {"status": "PASS", "BASE_SUCCESS": {"S2_delta_vs_B0": 0}})

    post_lift_fields = [
        "seed",
        "stratum",
        "row_id",
        "episode_id",
        "lifted_t",
        "apple_height_peak",
        "carried_duration_after_lift",
        "min_apple_to_plate_dist_after_lift",
        "delta_apple_to_plate_dist_after_lift",
        "moved_toward_plate_after_lift",
        "reached_plate_proxy",
        "release_or_open_proxy_after_lift",
        "hand_close_open_profile_summary",
        "arm_energy_after_lift",
        "base_nav_energy_after_lift",
        "nav_projection_to_plate",
        "chunk_boundary_jump_q99",
        "final_failure_mode",
    ]
    _write_csv(root / "post_lift_place_audit.csv", post_lift_fields, [])
    _write_json(
        root / "post_lift_place_audit.json",
        {
            "status": "PASS",
            "coverage": {"complete": True},
            "answers": {
                "apple_moves_toward_plate_after_lift": "no lifted failures",
                "any_policy_reaches_plate_proxy": True,
                "hand_release_or_hold_timing": "no lifted failures",
                "transport_driver": "not applicable",
                "dominant_transition_failure": "none",
            },
        },
    )
    _write_jsonl(root / "lifted_episode_index.jsonl", [])

    _write_json(
        root / "final_decision.json",
        {
            "final_decision": final_decision,
            "allowed_final_decisions": list(ALLOWED_T8_2_FINAL_DECISIONS),
            "guarded_recap_allowed": False,
            "fatg_allowed": False,
            "training_allowed": False,
        },
    )
    (root / "t8_2_summary.md").write_text(
        "\n".join(
            [
                "# T8.2 summary",
                "- JSON: evidence_lock.json, seed_bank.json, final_decision.json",
                "- CSV: candidate_seed_scout.csv, paired_eval_summary.csv, post_lift_place_audit.csv",
                f"- final_decision: {final_decision}",
                "- Guarded RECAP/FATG/per-edge/full-scope remain forbidden.",
                "- recommended_next_branch: broader paired no-RECAP eval.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_allowed_final_enum_list_matches_t8_2_user_list() -> None:
    assert ALLOWED_T8_2_FINAL_DECISIONS == (
        "BASE_PROTOCOL_TOO_WEAK",
        "SAFE_SFT_NONCOLLAPSE_PRELIM",
        "REACH_NAV_BLOCKER",
        "POST_LIFT_PLACE_BLOCKER",
        "POST_LIFT_SPLICE_IDENTIFIED_FIX",
        "HAND_REGRESSION",
        "READY_FOR_SAFE_SFT_30",
        "GUARDED_RECAP_STILL_FORBIDDEN",
    )


def test_verifier_passes_complete_artifact_root_and_writes_report(tmp_path: Path) -> None:
    artifact_root = _make_complete_artifact_root(tmp_path)

    payload = verify_t8_2_artifact_root(artifact_root)

    assert payload["status"] == "PASS"
    assert payload["final_decision"] == "SAFE_SFT_NONCOLLAPSE_PRELIM"
    assert "evidence_lock.json" in payload["cited_json_artifacts"]
    assert "candidate_seed_scout.csv" in payload["cited_csv_artifacts"]
    written = json.loads((artifact_root / "post_run_verification.json").read_text(encoding="utf-8"))
    assert written["status"] == "PASS"


def test_verifier_rejects_legacy_or_multi_final_enum(tmp_path: Path) -> None:
    artifact_root = _make_complete_artifact_root(tmp_path, final_decision="SAFE_SFT_NONCOLLAPSE_PRELIM")
    _write_json(
        artifact_root / "final_decision.json",
        {
            "final_decision": "BASE_SEEDS_TOO_HARD",
            "allowed_final_decisions": list(ALLOWED_T8_2_FINAL_DECISIONS),
            "decisions": ["BASE_SEEDS_TOO_HARD", "GUARDED_RECAP_STILL_FORBIDDEN"],
        },
    )

    payload = verify_t8_2_artifact_root(artifact_root)

    final_check = next(check for check in payload["checks"] if check["name"] == "final_enum")
    assert payload["status"] == "FAIL"
    assert "legacy T8/T8.1 enum" in " ".join(final_check["details"]["errors"])
    assert "multi-decision" in " ".join(final_check["details"]["errors"])


def test_forbidden_route_in_manifest_fails_before_final_claim(tmp_path: Path) -> None:
    artifact_root = _make_complete_artifact_root(tmp_path)
    _write_json(
        artifact_root / "command_manifest.json",
        {
            "commands": ["timeout 60 env CUDA_VISIBLE_DEVICES=1 python train.py --guarded-recap"],
            "submodule_status_short": [],
            "training_allowed": False,
        },
    )

    payload = verify_t8_2_artifact_root(artifact_root)

    guard_check = next(check for check in payload["checks"] if check["name"] == "forbidden_scope_guard")
    assert payload["status"] == "FAIL"
    assert "guarded_recap" in " ".join(guard_check["details"]["errors"])


def test_underfilled_base_success_requires_base_protocol_final(tmp_path: Path) -> None:
    artifact_root = _make_complete_artifact_root(tmp_path, selected_base_success=9, final_decision="SAFE_SFT_NONCOLLAPSE_PRELIM")

    payload = verify_t8_2_artifact_root(artifact_root)

    seed_check = next(check for check in payload["checks"] if check["name"] == "seed_bank_schema")
    assert payload["status"] == "FAIL"
    assert "BASE_SUCCESS < 10" in " ".join(seed_check["details"]["errors"])


def test_base_protocol_too_weak_skip_bundle_passes_without_pair_or_postlift_artifacts(tmp_path: Path) -> None:
    artifact_root = _make_complete_artifact_root(
        tmp_path,
        selected_base_success=3,
        final_decision="BASE_PROTOCOL_TOO_WEAK",
    )
    for name in (
        "control_regression_report.json",
        "stratum_effects.json",
        "post_lift_place_audit.json",
        "paired_eval_summary.csv",
        "candidate_eval_summary.csv",
        "post_lift_place_audit.csv",
        "paired_eval_per_seed.jsonl",
        "lifted_episode_index.jsonl",
    ):
        (artifact_root / name).unlink()
    _write_json(
        artifact_root / "paired_eval_skip_report.json",
        {
            "status": "NOT_RUN_BASE_PROTOCOL_TOO_WEAK",
            "reason": "selected BASE_SUCCESS < 10 after declared seed scan window",
        },
    )
    (artifact_root / "t8_2_summary.md").write_text(
        "\n".join(
            [
                "# T8.2 summary",
                "- JSON: evidence_lock.json, seed_bank.json, final_decision.json",
                "- CSV: candidate_seed_scout.csv",
                "- final_decision: BASE_PROTOCOL_TOO_WEAK",
                "- Guarded RECAP/FATG/per-edge/full-scope remain forbidden.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = verify_t8_2_artifact_root(artifact_root)

    assert payload["status"] == "PASS"
    paired_check = next(check for check in payload["checks"] if check["name"] == "paired_eval_schema")
    post_lift_check = next(check for check in payload["checks"] if check["name"] == "post_lift_schema")
    assert paired_check["status"] == "PASS"
    assert post_lift_check["status"] == "PASS"


def test_cli_returns_nonzero_for_invalid_artifact_root(tmp_path: Path) -> None:
    artifact_root = _make_complete_artifact_root(tmp_path)
    _write_json(artifact_root / "final_decision.json", {"final_decision": ["BASE_PROTOCOL_TOO_WEAK"]})

    assert main(["--artifact-root", str(artifact_root)]) == 2
