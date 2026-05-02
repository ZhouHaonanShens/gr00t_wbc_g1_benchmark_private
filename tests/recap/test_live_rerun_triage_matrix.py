from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.datasets import flux_grouped_dataset
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import audit_g1_execution_surface
from work.recap.scripts import build_flux_graft_final_report
from work.recap.scripts import gr00t_screening_authoritative
from work.recap.scripts import gr00t_screening_probe_bypass_diagnostic
from work.recap.scripts import gr00t_training_promotion_gate


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _inventory_payload() -> dict[str, object]:
    contract = gr00t_screening_authoritative.build_authoritative_binding_join_contract()
    return {
        "schema_version": flux_grouped_dataset.SCHEMA_VERSION,
        "artifact_kind": flux_grouped_dataset.ARTIFACT_KIND,
        "dataset_dir": "/tmp/fixture_dataset",
        "dataset_name": "fixture_dataset",
        "verdict": flux_grouped_dataset.VERDICT_COMPLETE,
        "dataset_source": {"dataset_dir": "/tmp/fixture_dataset"},
        "dataset_fingerprint": "fixture_dataset_fingerprint_sha256",
        "stats_fingerprint": "fixture_stats_fingerprint_sha256",
        "prompt_source": {
            "prompt_source_field": contract["prompt_source"],
            "prompt_route": "recap_conditioned_prompt_token_v1",
            "conditioning_mode": "prompt_text_only",
            "provenance_complete": True,
        },
        "task_description_source": {"task_text_field": "carrier_text_v1"},
        "camera_inventory": {"view_count": 2},
        "action_state_normalization_source": {
            "norm_stats_policy": contract["action_state_norm_source"],
            "norm_stats_source": contract["norm_stats_source"],
        },
        "schema_compatibility": {
            "status": "compatible",
            "state_dim": 8,
            "action_dim": 7,
        },
        "grouped_stats": {"episode_row_count": 1},
        "binding_join_contract": {
            "dataset_fingerprint": "fixture_dataset_fingerprint_sha256",
            **contract,
        },
        "dataset_adapter": {"schema_version": "flux_parquet_dataset_adapter_v1"},
        "blocking_reasons": [],
    }


def _screening_payload(
    repo_root: Path,
    inventory_json: Path,
    *,
    row_label: str,
    formal_eligibility: str = "ALLOW",
    reason_code: str = "ok",
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    servable = formal_eligibility == "ALLOW"
    if formal_eligibility != "ALLOW":
        issues = [
            {
                "code": "binding_join_mismatch",
                "field_path": "binding_join_contract.prompt_source",
                "message": "fixture screening blocked",
                "expected": "prompt_raw",
                "observed": "prompt_conditioned",
            }
        ]
        servable = False
    inference_model_ref = {
        "artifact_id": f"inference_model_{row_label.lower()}",
        "relative_path": f"agent/artifacts/flux/{row_label.lower()}_inference_model_ref.json",
        "surface_role": "inference_model_ref",
    }
    payload = {
        "schema_version": gr00t_screening_authoritative.SCREENING_SCHEMA_VERSION,
        "artifact_kind": gr00t_screening_authoritative.SCREENING_ARTIFACT_KIND,
        "screening_mode": "authoritative",
        "config_module": f"configs.apple_recap.flux.fixture_{row_label.lower()}",
        "dataset_inventory_json": str(inventory_json),
        "output_dir": str(repo_root / "screening" / row_label.lower()),
        "artifact_path": str(repo_root / "screening" / f"{row_label.lower()}.json"),
        "formal_eligibility": formal_eligibility,
        "reason_code": reason_code,
        "issues": issues,
        "inventory_verdict": flux_grouped_dataset.VERDICT_COMPLETE,
        "inventory_blocking_reasons": [],
        "dataset_fingerprint": "fixture_dataset_fingerprint_sha256",
        "stats_fingerprint": "fixture_stats_fingerprint_sha256",
        "prompt_source": "prompt_raw",
        "schema_compatibility": {
            "status": "compatible",
            "state_dim": 8,
            "action_dim": 7,
        },
        "inventory_binding_join_contract": {"prompt_source": "prompt_raw"},
        "authoritative_binding_join_contract": {"prompt_source": "prompt_raw"},
        "binding_join_evaluation": {
            "matched": formal_eligibility == "ALLOW",
            "mismatched_fields": []
            if formal_eligibility == "ALLOW"
            else ["prompt_source"],
            "join_key_fields": list(gr00t_screening_authoritative.JOIN_KEY_FIELDS),
        },
        "action_space_compatibility": {
            "status": "compatible" if formal_eligibility == "ALLOW" else "blocked"
        },
        "servable": servable,
        "inference_model_ref": inference_model_ref,
        "materialized_model_ref": inference_model_ref if servable else None,
        "inference_model_materialized": servable,
        "train_model_materialized": False,
        "source_artifacts": [
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="dataset_inventory_bundle",
                authority_role="authoritative_rerun_dataset_inventory",
                relative_path=inventory_json,
            )
        ],
    }
    payload["report_signature_sha256"] = (
        gr00t_screening_authoritative._signature_for_payload(payload)
    )
    return payload


def _exact_probe_payload(*, probe_passed: bool) -> dict[str, object]:
    if probe_passed:
        return {
            "status": "PASS",
            "formal_eligibility": "ALLOW",
            "reason_code": "ok",
            "probe_passed": True,
        }
    return {
        "status": "FAIL",
        "reason_code": "probe_failed",
        "probe_passed": False,
    }


def _summary_payload(*, success_count: int, episodes: int = 10) -> dict[str, object]:
    return {
        "episodes": episodes,
        "requested_episodes": episodes,
        "success_count": success_count,
        "success_rate": float(success_count) / float(episodes) if episodes else 0.0,
    }


def _late_stage_probe_inputs() -> dict[str, object]:
    return {
        "episode_record": {"failure_reason": "done_without_success"},
        "step_records": [
            {
                "t": 0,
                "policy_condition": {"phase": "APPROACH"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.50,
            },
            {
                "t": 1,
                "policy_condition": {"phase": "GRASP"},
                "privileged": {"apple_in_hand": True},
                "apple_to_plate_l2": 0.30,
            },
            {
                "t": 2,
                "policy_condition": {"phase": "TRANSPORT"},
                "privileged": {"apple_in_hand": True},
                "apple_to_plate_l2": 0.09,
            },
            {
                "t": 3,
                "policy_condition": {"phase": "PLACE"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.04,
            },
        ],
    }


def _weak_probe_inputs() -> dict[str, object]:
    return {
        "episode_record": {"failure_reason": "terminated_without_success"},
        "step_records": [
            {
                "t": 0,
                "policy_condition": {"phase": "APPROACH"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.60,
            },
            {
                "t": 1,
                "policy_condition": {"phase": "APPROACH"},
                "privileged": {"apple_in_hand": False},
                "apple_to_plate_l2": 0.58,
            },
        ],
    }


def _execution_audit_payload() -> dict[str, object]:
    return audit_g1_execution_surface.build_execution_surface_audit(
        runtime_trace={
            "status": "READY",
            "controller_output_available": False,
            "controller_output_unavailable_reason": "controller_output unavailable in current live seam",
            "upstream_distinction": {
                "prompt_or_token_distinct": True,
                "raw_or_decoded_distinct": True,
            },
            "stage_max_mean_abs_delta_over_contract_range": {
                "decoded_action": 0.08,
                "absolute_action": 0.04,
                "controller_input": 0.0,
                "controller_output": None,
            },
        }
    )


def _diagnostic_payload() -> dict[str, Any]:
    return (
        gr00t_screening_probe_bypass_diagnostic.build_probe_bypass_diagnostic_payload(
            row_evidence_by_label={
                "B0": _weak_probe_inputs(),
                "E1": _late_stage_probe_inputs(),
            },
            include_e1=True,
        )
    )


def _promotion_payload(
    repo_root: Path,
    *,
    inventory_json: Path,
    screening_json: Path,
    allow: bool = True,
    reason_code: str = "ok",
) -> dict[str, Any]:
    return {
        "schema_version": gr00t_training_promotion_gate.SCHEMA_VERSION,
        "artifact_kind": gr00t_training_promotion_gate.ARTIFACT_KIND,
        "gate_name": gr00t_training_promotion_gate.GATE_NAME,
        "output_dir": str(repo_root / "promotion"),
        "artifact_path": str(
            repo_root
            / "promotion"
            / gr00t_training_promotion_gate.TRAINING_PROMOTION_GATE_JSON_NAME
        ),
        "promotion_allowed": allow,
        "allow_plan_next_training_stage": allow,
        "promotion_status": "PASS" if allow else "BLOCK",
        "reason_code": reason_code,
        "failure_reasons": [] if allow else [reason_code],
        "checks": {},
        "issues": []
        if allow
        else [
            {"code": reason_code, "field_path": "checks", "message": "fixture blocked"}
        ],
        "source_artifacts": [
            {
                "artifact_id": "dataset_inventory",
                "relative_path": apple_recap_execution_contract._repo_relative_path(
                    repo_root, inventory_json
                ),
            },
            {
                "artifact_id": "authoritative_screening",
                "relative_path": apple_recap_execution_contract._repo_relative_path(
                    repo_root, screening_json
                ),
            },
        ],
        "diagnostic_only": True,
        "mainline_authority": False,
        "main_verdict_eligible": False,
        "external_reference_only": True,
        "release_gate": False,
        "gate_semantics": "plan_next_training_stage_only",
    }


def _write_probe_inputs(
    root: Path, topic: str, payload: dict[str, object]
) -> dict[str, Path]:
    episode_json = _write_json(root / topic / "episode.json", payload["episode_record"])
    steps_jsonl = _write_jsonl(
        root / topic / "steps.jsonl",
        list(cast(list[dict[str, object]], payload["step_records"])),
    )
    return {
        "episode_json": episode_json,
        "steps_jsonl": steps_jsonl,
    }


def _materialize_repo(tmp_path: Path) -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    inventory_json = _write_json(
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json",
        _inventory_payload(),
    )
    return {"repo_root": repo_root, "inventory_json": inventory_json}


def _write_screening(
    repo_root: Path, inventory_json: Path, row_label: str, **kwargs: Any
) -> Path:
    payload = _screening_payload(
        repo_root, inventory_json, row_label=row_label, **kwargs
    )
    return _write_json(
        repo_root
        / "agent/artifacts/apple_recap_flux_graft/authoritative_screening"
        / f"{row_label.lower()}_screening.json",
        payload,
    )


def _write_promotion(
    repo_root: Path,
    *,
    inventory_json: Path,
    screening_json: Path,
    allow: bool = True,
    reason_code: str = "ok",
) -> Path:
    return _write_json(
        repo_root
        / "agent/artifacts/apple_recap_flux_graft/training_promotion_gate/training_promotion_gate.json",
        _promotion_payload(
            repo_root,
            inventory_json=inventory_json,
            screening_json=screening_json,
            allow=allow,
            reason_code=reason_code,
        ),
    )


def _materialize_triage(
    tmp_path: Path,
    *,
    probe_passed: bool,
    b0_success_count: int,
    e1_probe_inputs: dict[str, object] | None = None,
    e1_success_count: int | None = None,
    e2_success_count: int | None = None,
    promotion_allow: bool = True,
    promotion_reason_code: str = "ok",
    include_diagnostic: bool = False,
    stale_promotion_screening_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    repo = _materialize_repo(tmp_path)
    repo_root = repo["repo_root"]
    inventory_json = repo["inventory_json"]
    exact_probe_json = _write_json(
        repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/exact_probe.json",
        _exact_probe_payload(probe_passed=probe_passed),
    )
    b0_screening_json = _write_screening(repo_root, inventory_json, "B0")
    b0_summary_json = _write_json(
        repo_root / "agent/artifacts/apple_recap_flux_graft/b0_summary.json",
        _summary_payload(success_count=b0_success_count),
    )
    row_artifacts_by_label: dict[str, dict[str, object]] = {
        "B0": {
            "screening_json": b0_screening_json,
            "summary_json": b0_summary_json,
        }
    }
    if e1_success_count is not None or e1_probe_inputs is not None:
        e1_screening_json = _write_screening(repo_root, inventory_json, "E1")
        row_artifacts_by_label["E1"] = {"screening_json": e1_screening_json}
        if e1_success_count is not None:
            row_artifacts_by_label["E1"]["summary_json"] = _write_json(
                repo_root / "agent/artifacts/apple_recap_flux_graft/e1_summary.json",
                _summary_payload(success_count=e1_success_count),
            )
        if e1_probe_inputs is not None:
            probe_paths = _write_probe_inputs(
                repo_root / "agent/artifacts/apple_recap_flux_graft/probe_inputs",
                "e1",
                e1_probe_inputs,
            )
            row_artifacts_by_label["E1"].update(probe_paths)
            row_artifacts_by_label["E1"]["execution_audit_json"] = _write_json(
                repo_root
                / "agent/artifacts/apple_recap_flux_graft/e1_execution_audit.json",
                _execution_audit_payload(),
            )
    if e2_success_count is not None:
        e2_screening_json = _write_screening(repo_root, inventory_json, "E2")
        row_artifacts_by_label["E2"] = {
            "screening_json": e2_screening_json,
            "summary_json": _write_json(
                repo_root / "agent/artifacts/apple_recap_flux_graft/e2_summary.json",
                _summary_payload(success_count=e2_success_count),
            ),
        }

    diagnostic_json = None
    if include_diagnostic:
        diagnostic_json = _write_json(
            repo_root
            / "agent/artifacts/apple_recap_flux_graft/diagnostic_probe_bypass/diagnostic_probe_vs_screening_gap.json",
            _diagnostic_payload(),
        )
    screening_for_promotion = b0_screening_json
    if stale_promotion_screening_path is not None:
        screening_for_promotion = repo_root / stale_promotion_screening_path
    promotion_json = _write_promotion(
        repo_root,
        inventory_json=inventory_json,
        screening_json=screening_for_promotion,
        allow=promotion_allow,
        reason_code=promotion_reason_code,
    )
    triage_output_dir = (
        repo_root / "agent/artifacts/apple_recap_flux_graft/live_model_triage"
    )
    payload = gr00t_screening_authoritative.materialize_live_model_triage(
        exact_probe_json=exact_probe_json,
        diagnostic_json=diagnostic_json,
        promotion_gate_json=promotion_json,
        row_artifacts_by_label=row_artifacts_by_label,
        output_dir=triage_output_dir,
        repo_root=repo_root,
    )
    return payload, {
        "repo_root": repo_root,
        "inventory_json": inventory_json,
        "b0_screening_json": b0_screening_json,
        "promotion_json": promotion_json,
        "triage_json": triage_output_dir
        / gr00t_screening_authoritative.LIVE_MODEL_TRIAGE_JSON_NAME,
    }


def test_probe_failed_but_b0_nonzero_does_not_leak_diagnostic_positive(
    tmp_path: Path,
) -> None:
    payload, _ = _materialize_triage(
        tmp_path,
        probe_passed=False,
        b0_success_count=2,
        include_diagnostic=True,
    )

    assert payload["triage_status"] == "READY"
    assert payload["triage_result"] == "probe_failed_but_b0_nonzero"
    assert payload["reason_code"] == "probe_failed_but_b0_nonzero"
    assert (
        payload["diagnostic_reference"]["comparison_verdict"]
        == "probe_likely_too_strict"
    )


def test_probe_failed_and_b0_zero_is_conclusive_when_baseline_is_dead(
    tmp_path: Path,
) -> None:
    payload, _ = _materialize_triage(
        tmp_path,
        probe_passed=False,
        b0_success_count=0,
    )

    assert payload["triage_status"] == "READY"
    assert payload["triage_result"] == "probe_failed_and_b0_zero"


def test_probe_passed_b0_nonzero_without_post_b0_rows(tmp_path: Path) -> None:
    payload, _ = _materialize_triage(
        tmp_path,
        probe_passed=True,
        b0_success_count=1,
    )

    assert payload["triage_status"] == "READY"
    assert payload["triage_result"] == "probe_passed_b0_nonzero"


def test_probe_passed_e1_signal_present_can_come_from_exact_probe_like_row_evidence(
    tmp_path: Path,
) -> None:
    payload, _ = _materialize_triage(
        tmp_path,
        probe_passed=True,
        b0_success_count=0,
        e1_success_count=0,
        e1_probe_inputs=_late_stage_probe_inputs(),
    )

    assert payload["triage_status"] == "READY"
    assert payload["triage_result"] == "probe_passed_e1_signal_present"
    assert payload["authoritative_rows"]["E1"]["signal_flags"]["probe_positive"] is True
    assert payload["authoritative_rows"]["E1"]["execution_audit"]["verdict"] in {
        "postprocess",
        "controller_distortion",
        "unknown",
        "blocked",
        "policy",
    }


def test_probe_passed_e1e2_flat_uses_available_post_b0_rows(tmp_path: Path) -> None:
    payload, _ = _materialize_triage(
        tmp_path,
        probe_passed=True,
        b0_success_count=1,
        e1_success_count=0,
        e2_success_count=0,
    )

    assert payload["triage_status"] == "READY"
    assert payload["triage_result"] == "probe_passed_e1e2_flat"


def test_triage_blocks_when_promotion_gate_blocks_training_planning(
    tmp_path: Path,
) -> None:
    payload, _ = _materialize_triage(
        tmp_path,
        probe_passed=True,
        b0_success_count=1,
        promotion_allow=False,
        promotion_reason_code="smoke_blocked",
    )

    assert payload["triage_status"] == "BLOCKED"
    assert payload["triage_result"] is None
    assert payload["reason_code"] == "smoke_blocked"


def test_triage_marks_promotion_backpointer_mismatch_as_stale(tmp_path: Path) -> None:
    payload, _ = _materialize_triage(
        tmp_path,
        probe_passed=True,
        b0_success_count=1,
        stale_promotion_screening_path="agent/artifacts/apple_recap_flux_graft/other_screening.json",
    )

    assert payload["triage_status"] == "STALE"
    assert payload["triage_result"] is None
    assert payload["reason_code"].startswith("stale_")


def test_final_report_builder_summarizes_live_model_triage_input(
    tmp_path: Path,
) -> None:
    triage_payload, paths = _materialize_triage(
        tmp_path,
        probe_passed=True,
        b0_success_count=1,
        e1_success_count=0,
        e2_success_count=0,
    )
    repo_root = paths["repo_root"]
    out_md = repo_root / "agent/exchange/Flux_Graft_RECAP_final_report.md"
    out_json = (
        repo_root
        / "agent/artifacts/apple_recap_flux_graft/final_report/final_verdict_pack.json"
    )

    payload = build_flux_graft_final_report.materialize_flux_graft_final_report(
        screening_json=paths["b0_screening_json"],
        dataset_inventory_json=paths["inventory_json"],
        out_md=out_md,
        out_json=out_json,
        repo_root=repo_root,
    )

    assert payload["global_verdict"] == "AUTHORITATIVE_POSITIVE"
    assert payload["triage_json"].endswith("triage_result.json")
    assert payload["summary"]["live_model_triage"]["input_status"] == "auto_discovered"
    assert payload["summary"]["live_model_triage"]["status"] == (
        "AUTHORITATIVE_POSITIVE"
    )
    assert (
        payload["summary"]["live_model_triage"]["triage_status"]
        == triage_payload["triage_status"]
    )
    assert (
        payload["summary"]["live_model_triage"]["triage_result"]
        == triage_payload["triage_result"]
    )
    assert payload["triage_plane"]["input_status"] == "auto_discovered"
    assert payload["triage_plane"]["status"] == "AUTHORITATIVE_POSITIVE"
    assert payload["non_authoritative_context"]["promotion_context"]["status"] == (
        "available"
    )
    assert (
        payload["non_authoritative_context"]["promotion_context"]["promotion_allowed"]
        is True
    )
    assert payload["non_authoritative_context"]["diagnostic_context"]["status"] == (
        "not_provided"
    )
    assert payload["non_authoritative_context"]["rtc_context"]["status"] == (
        "not_provided"
    )
    assert "live_model_triage" in payload["artifacts"]
    markdown = out_md.read_text(encoding="utf-8")
    assert "## Triage plane" in markdown
    assert "## Non-authoritative context" in markdown


def test_final_report_preserves_blocked_triage_reason_without_parity_language(
    tmp_path: Path,
) -> None:
    _triage_payload, paths = _materialize_triage(
        tmp_path,
        probe_passed=True,
        b0_success_count=1,
        promotion_allow=False,
        promotion_reason_code="smoke_blocked",
    )
    repo_root = paths["repo_root"]

    payload = build_flux_graft_final_report.build_final_report_pack(
        screening_json=paths["b0_screening_json"],
        dataset_inventory_json=paths["inventory_json"],
        repo_root=repo_root,
    )

    markdown = build_flux_graft_final_report.build_report_markdown(payload)
    assert payload["global_verdict"] == "BLOCKED"
    assert payload["reason_code"] == "smoke_blocked"
    assert payload["triage_plane"]["status"] == "BLOCKED"
    assert payload["triage_plane"]["reason_code"] == "smoke_blocked"
    assert "does not imply parity failure" in markdown


def test_final_report_exposes_rtc_as_optional_non_authoritative_context(
    tmp_path: Path,
) -> None:
    _triage_payload, paths = _materialize_triage(
        tmp_path,
        probe_passed=True,
        b0_success_count=0,
        e1_success_count=0,
        e1_probe_inputs=_late_stage_probe_inputs(),
    )
    repo_root = paths["repo_root"]

    payload = build_flux_graft_final_report.build_final_report_pack(
        screening_json=paths["b0_screening_json"],
        dataset_inventory_json=paths["inventory_json"],
        repo_root=repo_root,
    )

    rtc_context = payload["non_authoritative_context"]["rtc_context"]
    assert rtc_context["status"] == "available"
    assert rtc_context["row_id"] == "E1"
    assert rtc_context["verdict"] in {
        "postprocess",
        "controller_distortion",
        "unknown",
        "blocked",
        "policy",
    }
