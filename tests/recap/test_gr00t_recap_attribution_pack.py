from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_recap_attribution_pack


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task1_evidence(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_1_eval_contract_evidence",
            "public_anchor_snapshot": {
                "env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
                "policy_horizon_expected": 30,
                "n_action_steps": 20,
                "UNITREE_G1_public_anchor_comparable": True,
                "NEW_EMBODIMENT_public_anchor_comparable": False,
                "branch_comparability_tag": "unitree_vs_new_embodiment_split_v1",
            },
        },
    )


def _task_evidence(path: Path, *, task: int, slug: str) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": f"task_{task}_{slug}_evidence",
            "task_code": f"T{task}",
            "status": "PASS",
            "report_signature_sha256": f"task-{task}-signature",
        },
    )


def _checkpoint_report(path: Path, *, allow: bool = True) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "gr00t_checkpoint_provenance_gate_v1",
            "artifact_kind": "gr00t_checkpoint_provenance_gate",
            "formal_eligibility": "ALLOW" if allow else "BLOCK",
            "loadability_status": (
                "LOADABLE_CHECKPOINT_CONFIRMED"
                if allow
                else "WRONG_CHECKPOINT_OR_BASE_FALLBACK"
            ),
            "status": "PASS" if allow else "BLOCK",
            "report_signature_sha256": "checkpoint-report-signature",
        },
    )


def _public_anchor(path: Path, *, success_count: int = 5) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "gr00t_public_anchor_formal_v1",
            "artifact_kind": "gr00t_public_anchor_formal",
            "success_count": success_count,
            "success_rate": 0.5 if success_count else 0.0,
            "systemic_break_flags": [],
            "report_signature_sha256": "public-anchor-signature",
        },
    )


def _controller_audit(
    path: Path, *, branch: str, pass_equivalence: bool = True
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": f"{branch.lower()}_controller_audit_v1",
        "artifact_kind": f"{branch.lower()}_controller_audit",
        "mismatch_fields": [],
        "report_signature_sha256": f"controller-{branch.lower()}-signature",
    }
    if branch == "UNITREE_G1":
        payload["equivalent_to_official_unitree_g1"] = pass_equivalence
    else:
        payload["formal_branch_eligibility"] = "ALLOW" if pass_equivalence else "BLOCK"
    return _write_json(path, payload)


def _action_telemetry(
    path: Path,
    *,
    public_anchor_comparable: bool,
    controller_absorbed_groups: list[str] | None = None,
    model_insensitive_groups: list[str] | None = None,
    zero_motion_groups: list[str] | None = None,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "gr00t_action_chain_telemetry_v1",
            "artifact_kind": "gr00t_action_chain_telemetry",
            "public_anchor_comparable": public_anchor_comparable,
            "controller_absorbed_groups": controller_absorbed_groups or [],
            "model_insensitive_groups": model_insensitive_groups or [],
            "zero_motion_flags": {"all_zero_in_both_groups": zero_motion_groups or []},
            "report_signature_sha256": (
                f"telemetry-{'public' if public_anchor_comparable else 'internal'}"
            ),
        },
    )


def _teacher_reachability(
    path: Path,
    *,
    branch: str,
    healthy_replay: bool = True,
) -> Path:
    prefix = branch.lower()
    return _write_json(
        path,
        {
            "schema_version": "gr00t_teacher_reachability_gate_v1",
            "artifact_kind": "gr00t_teacher_reachability_gate",
            "branch": branch,
            "allow_formal_ladders": True,
            "teacher_reachable_scene_ids": [f"{prefix}::S_drop", f"{prefix}::S_lost"],
            "reachable_scene_ids": [f"{prefix}::S_drop", f"{prefix}::S_lost"],
            "replay_upper_bound": {
                "success_count": 2 if healthy_replay else 0,
                "success_rate": 0.5 if healthy_replay else 0.0,
            },
            "report_signature_sha256": f"reachability-{prefix}-signature",
        },
    )


def _branch_entry(
    *,
    branch_key: str,
    branch: str,
    branch_scope: str,
    public_anchor_comparable: bool,
    official_comparable_line: bool,
    internal_only_comparable_line: bool,
    controller_path: Path,
    action_path: Path,
    reachability_path: Path,
    public_anchor_path: Path | None,
    public_anchor_success_rate: float | None,
    condition_flip_response_ratio: float,
) -> dict[str, Any]:
    return {
        "branch_key": branch_key,
        "embodiment_tag": branch,
        "branch_scope": branch_scope,
        "public_anchor_comparable": public_anchor_comparable,
        "official_comparable_line": official_comparable_line,
        "internal_only_comparable_line": internal_only_comparable_line,
        "controller_equivalence": {"artifact_path": str(controller_path)},
        "action_telemetry": {"artifact_path": str(action_path)},
        "teacher_reachability": {"artifact_path": str(reachability_path)},
        "prerequisite_status": {
            "status": "PASS",
            "failed_checks": [],
        },
        "recommended_next_step": "proceed_to_task_12_ladder_policy_gate",
        "diagnostic_summary": {
            "public_anchor_success_rate": public_anchor_success_rate,
            "condition_flip_min_response_ratio": condition_flip_response_ratio,
            "teacher_reachable_scene_count": 2,
            "teacher_student_branch_match_rate": 0.75,
            "action_telemetry_controller_absorbed_groups": ["left_arm"],
            "action_telemetry_model_insensitive_groups": ["right_hand"],
            "action_telemetry_zero_motion_groups": ["right_hand"],
        },
        **(
            {"public_anchor_status": {"artifact_path": str(public_anchor_path)}}
            if public_anchor_path is not None
            else {}
        ),
    }


def _dual_branch_scorecard(
    path: Path, *, unitree_entry: dict[str, Any], new_entry: dict[str, Any]
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "gr00t_dual_branch_scorecard_v1",
            "artifact_kind": "gr00t_dual_branch_scorecard",
            "branch_order": ["unitree_g1", "new_embodiment"],
            "branches": [unitree_entry, new_entry],
            "report_signature_sha256": "dual-branch-signature",
        },
    )


def _simple_gate(path: Path, *, artifact_kind: str, schema_version: str) -> Path:
    return _write_json(
        path,
        {
            "schema_version": schema_version,
            "artifact_kind": artifact_kind,
            "report_signature_sha256": f"{artifact_kind}-signature",
        },
    )


def _p_scorecard_payload(
    *,
    branch: str,
    branch_key: str,
    branch_scope: str,
    public_anchor_comparable: bool,
    rung: str,
    effective: bool,
) -> dict[str, Any]:
    return {
        "schema_version": f"gr00t_p_ladder_{branch_key}_v1",
        "artifact_kind": f"gr00t_p_ladder_{branch_key}_scorecard",
        "axis": "P",
        "branch": branch,
        "branch_key": branch_key,
        "branch_scope": branch_scope,
        "rung": rung,
        "public_anchor_comparable": public_anchor_comparable,
        "frozen_formal_protocol": {"env_name": "formal_env"},
        "source_artifacts": {"dual_branch_scorecard": {"path": "dual.json"}},
        "comparability": {"artifact_kind": "gr00t_ladder_policy_gate"},
        "status": "BLOCK" if rung == "P3" else "PASS",
        "promotion_status": "BLOCK" if rung == "P3" else "PASS",
        "blocking_reasons": ["positive_slope_required_for_p3"] if rung == "P3" else [],
        "positive_slope_report": {
            "positive_slope_detected": effective,
            "qualifying_metric_names": ["success_count"] if effective else [],
        },
        "report_signature_sha256": f"{branch_key}-{rung}-scorecard-signature",
    }


def _d_scorecard_payload(
    *,
    branch: str,
    branch_key: str,
    branch_scope: str,
    public_anchor_comparable: bool,
    rung: str,
    baseline_success_rate: float,
    baseline_condition_ratio: float,
    effective: bool,
    branch_local_only_rung: bool,
) -> dict[str, Any]:
    blocked = branch_key == "unitree_g1" and rung == "D4"
    success_rate = baseline_success_rate + 0.2 if effective else baseline_success_rate
    condition_ratio = (
        baseline_condition_ratio + 0.05 if effective else baseline_condition_ratio
    )
    return {
        "schema_version": f"gr00t_d_ladder_{branch_key}_v1",
        "artifact_kind": f"gr00t_d_ladder_{branch_key}_scorecard",
        "axis": "D",
        "branch": branch,
        "branch_key": branch_key,
        "branch_scope": branch_scope,
        "rung": rung,
        "public_anchor_comparable": public_anchor_comparable,
        "frozen_formal_protocol": {"env_name": "formal_env"},
        "source_artifacts": {"dual_branch_scorecard": {"path": "dual.json"}},
        "comparability": {"artifact_kind": "gr00t_ladder_policy_gate"},
        "status": "BLOCK" if blocked else "PASS",
        "promotion_status": "BLOCK" if blocked else "PASS",
        "blocking_reasons": ["unitree_g1_d4_branch_only_redirect"] if blocked else [],
        "execution_disposition": (
            "BLOCK_REDIRECT_TO_NEW_EMBODIMENT"
            if blocked
            else "EXECUTE_ON_BRANCH"
            if branch_local_only_rung
            else "EXECUTE_ON_TRUNK"
        ),
        "branch_local_only_rung": branch_local_only_rung,
        "success_rate": success_rate,
        "condition_flip_response_ratio": condition_ratio,
        "teacher_gap": 1.0,
        "baseline_metrics": {
            "success_rate": baseline_success_rate,
            "condition_flip_response_ratio": baseline_condition_ratio,
            "teacher_gap": 1.0,
        },
        "report_signature_sha256": f"{branch_key}-{rung}-scorecard-signature",
    }


def _manifest_payload(*, branch_key: str, axis: str, rung: str) -> dict[str, Any]:
    return {
        "schema_version": f"manifest_{branch_key}_{axis}_{rung}_v1",
        "artifact_kind": f"manifest_{branch_key}_{axis}_{rung}",
        "branch_key": branch_key,
        "axis": axis,
        "rung": rung,
        "status": "PASS",
        "report_signature_sha256": f"manifest-{branch_key}-{axis}-{rung}",
    }


def _build_fixture(
    tmp_path: Path,
    *,
    unitree_parameter_effective: bool,
    new_data_effective: bool,
    recap_branch_key: str | None = None,
    add_invalid_exploratory_new_p1: bool = False,
) -> dict[str, Any]:
    evidence_dir = tmp_path / "evidence"
    artifacts_dir = tmp_path / "artifacts"
    output_dir = artifacts_dir / "final_wave"

    task1_evidence = _task1_evidence(evidence_dir / "task-1-eval-contract.json")
    task12_evidence = _task_evidence(
        evidence_dir / "task-12-ladder-policy-gate.json",
        task=12,
        slug="ladder_policy_gate",
    )
    task14_evidence = _task_evidence(
        evidence_dir / "task-14-p-ladder-new-embodiment.json",
        task=14,
        slug="p_ladder_new_embodiment",
    )
    task15_evidence = _task_evidence(
        evidence_dir / "task-15-d-ladder-policy-gate.json",
        task=15,
        slug="d_ladder_policy_gate",
    )
    task16_evidence = _task_evidence(
        evidence_dir / "task-16-d-ladder-unitree-g1.json",
        task=16,
        slug="d_ladder_unitree_g1",
    )
    task17_evidence = _task_evidence(
        evidence_dir / "task-17-d-ladder-new-embodiment.json",
        task=17,
        slug="d_ladder_new_embodiment",
    )
    task18_evidence = evidence_dir / "task-18-attribution-pack.json"
    checkpoint_report = _checkpoint_report(
        artifacts_dir / "checkpoint_provenance_report.json"
    )

    unitree_public_anchor = _public_anchor(
        artifacts_dir / "unitree_g1" / "public_anchor" / "public_anchor_formal.json"
    )
    unitree_controller = _controller_audit(
        artifacts_dir / "unitree_g1" / "controller_audit_unitree_g1.json",
        branch="UNITREE_G1",
    )
    new_controller = _controller_audit(
        artifacts_dir / "new_embodiment" / "controller_audit_new_embodiment.json",
        branch="NEW_EMBODIMENT",
    )
    unitree_action = _action_telemetry(
        artifacts_dir / "unitree_g1" / "action_chain_telemetry_unitree_g1.json",
        public_anchor_comparable=True,
        controller_absorbed_groups=["left_arm"],
        model_insensitive_groups=["right_hand"],
        zero_motion_groups=["right_hand"],
    )
    new_action = _action_telemetry(
        artifacts_dir / "new_embodiment" / "action_chain_telemetry_new_embodiment.json",
        public_anchor_comparable=False,
        controller_absorbed_groups=["left_arm"],
        model_insensitive_groups=["right_hand"],
        zero_motion_groups=["right_hand"],
    )
    unitree_reachability = _teacher_reachability(
        artifacts_dir / "teacher_reachability_gate_unitree_g1.json",
        branch="UNITREE_G1",
    )
    new_reachability = _teacher_reachability(
        artifacts_dir / "teacher_reachability_gate_new_embodiment.json",
        branch="NEW_EMBODIMENT",
    )

    unitree_entry = _branch_entry(
        branch_key="unitree_g1",
        branch="UNITREE_G1",
        branch_scope="official_public_anchor_line",
        public_anchor_comparable=True,
        official_comparable_line=True,
        internal_only_comparable_line=False,
        controller_path=unitree_controller,
        action_path=unitree_action,
        reachability_path=unitree_reachability,
        public_anchor_path=unitree_public_anchor,
        public_anchor_success_rate=0.5,
        condition_flip_response_ratio=0.18,
    )
    new_entry = _branch_entry(
        branch_key="new_embodiment",
        branch="NEW_EMBODIMENT",
        branch_scope="branch_internal_only",
        public_anchor_comparable=False,
        official_comparable_line=False,
        internal_only_comparable_line=True,
        controller_path=new_controller,
        action_path=new_action,
        reachability_path=new_reachability,
        public_anchor_path=None,
        public_anchor_success_rate=None,
        condition_flip_response_ratio=0.15,
    )
    dual_branch = _dual_branch_scorecard(
        artifacts_dir / "dual_branch_scorecard.json",
        unitree_entry=unitree_entry,
        new_entry=new_entry,
    )

    unitree_p_root = artifacts_dir / "unitree_g1" / "p"
    new_p_root = artifacts_dir / "new_embodiment" / "p"
    new_p_smoke_root = artifacts_dir / "new_embodiment" / "p_smoke_check"
    unitree_d_root = artifacts_dir / "unitree_g1" / "d"
    new_d_root = artifacts_dir / "new_embodiment" / "d"

    for rung in ("P0", "P1", "P2", "P3"):
        unitree_effective = unitree_parameter_effective and rung in {"P1", "P2"}
        _write_json(
            unitree_p_root / rung / "scorecard.json",
            _p_scorecard_payload(
                branch="UNITREE_G1",
                branch_key="unitree_g1",
                branch_scope="official_public_anchor_line",
                public_anchor_comparable=True,
                rung=rung,
                effective=unitree_effective,
            ),
        )
        _write_json(
            unitree_p_root / rung / "manifest.json",
            _manifest_payload(branch_key="unitree_g1", axis="P", rung=rung),
        )

        new_effective = False if recap_branch_key == "new_embodiment" else False
        _write_json(
            new_p_smoke_root / rung / "scorecard.json",
            _p_scorecard_payload(
                branch="NEW_EMBODIMENT",
                branch_key="new_embodiment",
                branch_scope="branch_internal_only",
                public_anchor_comparable=False,
                rung=rung,
                effective=new_effective,
            ),
        )
        _write_json(
            new_p_smoke_root / rung / "manifest.json",
            _manifest_payload(branch_key="new_embodiment", axis="P", rung=rung),
        )

    if add_invalid_exploratory_new_p1:
        _write_json(
            new_p_root / "P1" / "scorecard.json",
            {
                "schema_version": "exploratory_scorecard_v1",
                "artifact_kind": "exploratory_only_scorecard",
                "axis": "P",
                "branch": "NEW_EMBODIMENT",
                "branch_key": "new_embodiment",
                "branch_scope": "branch_internal_only",
                "rung": "P1",
                "public_anchor_comparable": False,
            },
        )
        _write_json(
            new_p_root / "P1" / "manifest.json",
            _manifest_payload(branch_key="new_embodiment", axis="P", rung="P1"),
        )

    for rung in ("D0", "D1", "D2", "D3", "D4"):
        unitree_effective = False
        if recap_branch_key != "unitree_g1":
            unitree_effective = False
        _write_json(
            unitree_d_root / rung / "scorecard.json",
            _d_scorecard_payload(
                branch="UNITREE_G1",
                branch_key="unitree_g1",
                branch_scope="official_public_anchor_line",
                public_anchor_comparable=True,
                rung=rung,
                baseline_success_rate=0.5,
                baseline_condition_ratio=0.18,
                effective=False,
                branch_local_only_rung=False,
            ),
        )
        _write_json(
            unitree_d_root / rung / "manifest.json",
            _manifest_payload(branch_key="unitree_g1", axis="D", rung=rung),
        )

        new_effective = new_data_effective and rung in {"D1", "D2", "D3"}
        if recap_branch_key == "new_embodiment":
            new_effective = False
        _write_json(
            new_d_root / rung / "scorecard.json",
            _d_scorecard_payload(
                branch="NEW_EMBODIMENT",
                branch_key="new_embodiment",
                branch_scope="branch_internal_only",
                public_anchor_comparable=False,
                rung=rung,
                baseline_success_rate=0.0,
                baseline_condition_ratio=0.15,
                effective=new_effective,
                branch_local_only_rung=(rung == "D4"),
            ),
        )
        _write_json(
            new_d_root / rung / "manifest.json",
            _manifest_payload(branch_key="new_embodiment", axis="D", rung=rung),
        )

    p_gate_unitree = _simple_gate(
        artifacts_dir / "p_ladder_policy_gate_unitree.json",
        artifact_kind="gr00t_ladder_policy_gate",
        schema_version="gr00t_ladder_policy_gate_v1",
    )
    p_gate_new = _simple_gate(
        artifacts_dir / "p_ladder_policy_gate_new.json",
        artifact_kind="gr00t_ladder_policy_gate",
        schema_version="gr00t_ladder_policy_gate_v1",
    )
    d_gate_unitree = _simple_gate(
        artifacts_dir / "d_ladder_policy_gate_unitree.json",
        artifact_kind="gr00t_ladder_policy_gate",
        schema_version="gr00t_ladder_policy_gate_v1",
    )
    d_gate_new = _simple_gate(
        artifacts_dir / "d_ladder_policy_gate_new.json",
        artifact_kind="gr00t_ladder_policy_gate",
        schema_version="gr00t_ladder_policy_gate_v1",
    )
    d_admission_unitree = _simple_gate(
        artifacts_dir / "d_ladder_admission_unitree.json",
        artifact_kind="gr00t_d_ladder_admission_report",
        schema_version="gr00t_d_ladder_admission_report_v1",
    )
    d_admission_new = _simple_gate(
        artifacts_dir / "d_ladder_admission_new.json",
        artifact_kind="gr00t_d_ladder_admission_report",
        schema_version="gr00t_d_ladder_admission_report_v1",
    )
    dataset_registry = _simple_gate(
        artifacts_dir / "dataset_source_registry.json",
        artifact_kind="gr00t_dataset_source_registry",
        schema_version="gr00t_dataset_source_registry_v1",
    )

    return {
        "output_dir": output_dir,
        "task1_evidence": task1_evidence,
        "task18_evidence": task18_evidence,
        "task12_evidence": task12_evidence,
        "task14_evidence": task14_evidence,
        "task15_evidence": task15_evidence,
        "task16_evidence": task16_evidence,
        "task17_evidence": task17_evidence,
        "checkpoint_report": checkpoint_report,
        "dual_branch": dual_branch,
        "p_gate_unitree": p_gate_unitree,
        "p_gate_new": p_gate_new,
        "d_gate_unitree": d_gate_unitree,
        "d_gate_new": d_gate_new,
        "d_admission_unitree": d_admission_unitree,
        "d_admission_new": d_admission_new,
        "dataset_registry": dataset_registry,
        "unitree_p_root": unitree_p_root,
        "new_p_root": new_p_root,
        "new_p_smoke_root": new_p_smoke_root,
        "unitree_d_root": unitree_d_root,
        "new_d_root": new_d_root,
    }


def _run_cli(paths: dict[str, Any]) -> int:
    return gr00t_recap_attribution_pack.main(
        [
            "--output-dir",
            str(paths["output_dir"]),
            "--task1-evidence",
            str(paths["task1_evidence"]),
            "--task12-evidence",
            str(paths["task12_evidence"]),
            "--task14-evidence",
            str(paths["task14_evidence"]),
            "--task15-evidence",
            str(paths["task15_evidence"]),
            "--task16-evidence",
            str(paths["task16_evidence"]),
            "--task17-evidence",
            str(paths["task17_evidence"]),
            "--task18-evidence-path",
            str(paths["task18_evidence"]),
            "--checkpoint-provenance-report",
            str(paths["checkpoint_report"]),
            "--dual-branch-scorecard-json",
            str(paths["dual_branch"]),
            "--p-ladder-policy-gate-unitree",
            str(paths["p_gate_unitree"]),
            "--p-ladder-policy-gate-new-embodiment",
            str(paths["p_gate_new"]),
            "--d-ladder-policy-gate-unitree",
            str(paths["d_gate_unitree"]),
            "--d-ladder-policy-gate-new-embodiment",
            str(paths["d_gate_new"]),
            "--d-ladder-admission-gate-unitree",
            str(paths["d_admission_unitree"]),
            "--d-ladder-admission-gate-new-embodiment",
            str(paths["d_admission_new"]),
            "--dataset-source-registry-json",
            str(paths["dataset_registry"]),
            "--unitree-p-root",
            str(paths["unitree_p_root"]),
            "--new-embodiment-p-root",
            str(paths["new_p_root"]),
            "--new-embodiment-p-root",
            str(paths["new_p_smoke_root"]),
            "--unitree-d-root",
            str(paths["unitree_d_root"]),
            "--new-embodiment-d-root",
            str(paths["new_d_root"]),
        ]
    )


def test_materialize_attribution_pack_maps_parameter_and_data_signals(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(
        tmp_path,
        unitree_parameter_effective=True,
        new_data_effective=True,
    )

    assert _run_cli(paths) == 0

    final_payload = _read_json(paths["output_dir"] / "final_attribution_matrix.json")
    comparison_payload = _read_json(paths["output_dir"] / "branch_comparison_pack.json")
    freeze_payload = _read_json(paths["output_dir"] / "wave_freeze_manifest.json")
    evidence_payload = _read_json(paths["task18_evidence"])

    assert (
        final_payload["artifact_kind"]
        == gr00t_recap_attribution_pack.FINAL_ATTRIBUTION_ARTIFACT_KIND
    )
    assert (
        comparison_payload["artifact_kind"]
        == gr00t_recap_attribution_pack.BRANCH_COMPARISON_ARTIFACT_KIND
    )
    assert (
        freeze_payload["artifact_kind"]
        == gr00t_recap_attribution_pack.WAVE_FREEZE_ARTIFACT_KIND
    )

    branches = {row["branch_key"]: row for row in final_payload["branches"]}
    assert branches["unitree_g1"]["parameter_scope_hypothesis"]["strength"] == "strong"
    assert (
        branches["new_embodiment"]["data_distribution_hypothesis"]["strength"]
        == "strong"
    )

    assert [
        row["branch_key"] for row in final_payload["officially_comparable_conclusions"]
    ] == ["unitree_g1"]
    assert [
        row["branch_key"] for row in final_payload["internally_comparable_conclusions"]
    ] == ["new_embodiment"]

    axes = {row["metric"]: row for row in comparison_payload["comparison_axes"]}
    assert (
        axes["public_anchor_success_rate"]["comparability"]
        == "official_public_anchor_only"
    )
    assert axes["public_anchor_success_rate"]["new_embodiment"] is None

    inventory_ids = {row["artifact_id"] for row in freeze_payload["inventory"]}
    assert "task18_final_attribution_matrix" in inventory_ids
    assert "task18_branch_comparison_pack" in inventory_ids

    assert evidence_payload["artifact_kind"] == "task_18_attribution_pack_evidence"
    assert evidence_payload["comparability_contract"] == {
        "official_comparable_line": "unitree_g1",
        "internal_only_comparable_line": "new_embodiment",
        "public_anchor_projection_forbidden_to": ["new_embodiment"],
        "cross_branch_single_ranking_forbidden": True,
    }
    assert set(evidence_payload["generated_outputs"]) == {
        "final_attribution_matrix",
        "branch_comparison_pack",
        "wave_freeze_manifest",
    }
    assert (
        Path(evidence_payload["generated_outputs"]["final_attribution_matrix"]["path"])
        == paths["output_dir"] / "final_attribution_matrix.json"
    )
    assert evidence_payload["final_conclusions"]["recommended_next_step_by_branch"] == {
        "unitree_g1": final_payload["branches"][0]["recommended_next_step"],
        "new_embodiment": final_payload["branches"][1]["recommended_next_step"],
    }
    supporting_ids = {
        row["artifact_id"] for row in evidence_payload["key_supporting_artifacts"]
    }
    assert {
        "task12_ladder_policy_gate_evidence",
        "task14_p_ladder_new_embodiment_evidence",
        "task15_d_ladder_policy_gate_evidence",
        "task16_d_ladder_unitree_g1_evidence",
        "task17_d_ladder_new_embodiment_evidence",
    }.issubset(supporting_ids)


def test_recap_interface_strengthens_when_teacher_reachable_but_p_and_d_unmoved(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(
        tmp_path,
        unitree_parameter_effective=False,
        new_data_effective=False,
        recap_branch_key="unitree_g1",
    )

    assert _run_cli(paths) == 0

    final_payload = _read_json(paths["output_dir"] / "final_attribution_matrix.json")
    branches = {row["branch_key"]: row for row in final_payload["branches"]}
    unitree = branches["unitree_g1"]

    assert unitree["parameter_scope_hypothesis"]["strength"] == "not_supported"
    assert unitree["data_distribution_hypothesis"]["strength"] == "not_supported"
    assert unitree["recap_interface_hypothesis"]["strength"] == "strong"
    assert (
        unitree["recommended_next_step"]
        == "audit_recap_injection_action_target_and_relative_action_interpretation"
    )


def test_exploratory_candidates_are_excluded_from_final_attribution(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(
        tmp_path,
        unitree_parameter_effective=False,
        new_data_effective=False,
        add_invalid_exploratory_new_p1=True,
    )

    assert _run_cli(paths) == 0

    final_payload = _read_json(paths["output_dir"] / "final_attribution_matrix.json")
    excluded_paths = {
        row["path"]: row["reason"] for row in final_payload["excluded_artifacts"]
    }
    exploratory_scorecard = paths["new_p_root"] / "P1" / "scorecard.json"
    matching_reason = next(
        (
            reason
            for path, reason in excluded_paths.items()
            if Path(path) == exploratory_scorecard
        ),
        None,
    )
    assert matching_reason == "missing_frozen_formal_protocol"

    branches = {row["branch_key"]: row for row in final_payload["branches"]}
    discovery = branches["new_embodiment"]["selected_formal_artifacts"]["discovery"][
        "p"
    ]["discoveries"]["P1"]
    selected_scorecard = Path(discovery["scorecard_path"])
    assert selected_scorecard == paths["new_p_smoke_root"] / "P1" / "scorecard.json"
