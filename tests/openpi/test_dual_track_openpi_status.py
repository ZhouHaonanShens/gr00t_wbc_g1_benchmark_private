from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.dual_track_status import (  # noqa: E402
    DATASET_NOT_MATERIALIZED,
    LABEL_SEMANTICS_BLOCK,
    RUNTIME_LEVEL_NONE,
    RUNTIME_LEVEL_P0_LOADER,
    RUNTIME_LEVEL_P1_ONE_STEP,
    build_openpi_exploratory_signal,
    build_openpi_formal_status,
    build_openpi_summary_section,
    validate_exploratory_signal,
    validate_formal_status,
)


def _p0_manifest(*, materialized: bool) -> dict[str, object]:
    return {
        "dataset_join_root_status": {
            "materialized": materialized,
            "status": "materialized=true" if materialized else "BLOCKED(dataset_not_materialized)",
            "blocking_reason": "" if materialized else DATASET_NOT_MATERIALIZED,
        }
    }


def _join_report_with_openpi_identity_blockers() -> dict[str, object]:
    return {
        "hard_blockers": [
            {"code": "missing_cross_dataset_task_identity"},
            {"code": "missing_verified_episode_frame_crosswalk"},
            {"code": "missing_cross_dataset_frame_identity"},
            {"code": "task_text_universe_not_proven_compatible"},
            {"code": "weak_key_only_join_rejected"},
        ],
        "materialization_allowed": False,
        "materialization_emitted": False,
    }


def test_openpi_formal_status_blocks_unmaterialized_identity_and_semantics_gaps() -> None:
    status = build_openpi_formal_status(
        p0_scope_audit_manifest=_p0_manifest(materialized=False),
        join_report=_join_report_with_openpi_identity_blockers(),
        label_semantics_pass=False,
        authority_inputs=["p0_scope_audit/scope_audit_manifest.json"],
        validator_outputs=["join_report.json"],
    )

    validate_formal_status(status)
    assert status["status"] == "BLOCK"
    assert status["formal_claim_allowed"] is False
    assert status["next_gate_allowed"] is False
    assert status["entered_next_gate"] is False
    assert status["runtime_level"] == "blocked_materialization_reverify_failed"
    assert status["blocking_reasons"] == [
        DATASET_NOT_MATERIALIZED,
        "missing_cross_dataset_task_identity",
        "missing_verified_episode_frame_crosswalk",
        "missing_cross_dataset_frame_identity",
        "task_text_universe_not_proven_compatible",
        "weak_key_only_join_rejected",
        LABEL_SEMANTICS_BLOCK,
    ]


def test_openpi_formal_status_uses_atomic_blockers_not_compound_status_values() -> None:
    status = build_openpi_formal_status(
        p0_scope_audit_manifest=_p0_manifest(materialized=True),
        join_report={"hard_blockers": []},
        label_semantics_pass=False,
        authority_inputs=[],
    )

    validate_formal_status(status)
    assert status["status"] == "BLOCK"
    assert "(" not in str(status["status"])
    assert status["runtime_level"] == "blocked_formal_prereq"
    assert status["blocking_reasons"] == [LABEL_SEMANTICS_BLOCK]


def test_exploratory_signal_cannot_unlock_openpi_formal_status_or_summary() -> None:
    formal_status = build_openpi_formal_status(
        p0_scope_audit_manifest=_p0_manifest(materialized=False),
        join_report=_join_report_with_openpi_identity_blockers(),
        label_semantics_pass=False,
        authority_inputs=["scope_audit_manifest.json"],
    )
    exploratory = build_openpi_exploratory_signal(
        status="SIGNAL",
        method="tiny_overfit",
        inputs=["exploratory_only_tiny_dataset"],
        outputs=["one_step_probe.json"],
        observed_signal={"loss_decreased": True, "sample_file_count": 2},
    )

    validate_exploratory_signal(exploratory)
    section = build_openpi_summary_section(
        formal_status=formal_status,
        exploratory_signal=exploratory,
    )

    assert section["formal"] == {
        "status": "BLOCK",
        "formal_claim_allowed": False,
        "runtime_level": "blocked_materialization_reverify_failed",
        "artifact": "",
    }
    assert section["exploratory"]["status"] == "SIGNAL"
    assert exploratory["formal_claim_allowed"] is False
    assert exploratory["must_not_unlock_formal_gate"] is True


def test_openpi_materialization_only_does_not_claim_runtime_pass() -> None:
    status = build_openpi_formal_status(
        p0_scope_audit_manifest=_p0_manifest(materialized=True),
        join_report={"hard_blockers": []},
        label_semantics_pass=True,
        authority_inputs=["scope_audit_manifest.json", "join_report.json"],
        validator_outputs=["label_semantics_validator.json"],
    )

    validate_formal_status(status)
    assert status["status"] == "BLOCK"
    assert status["formal_claim_allowed"] is False
    assert status["next_gate_allowed"] is False
    assert status["runtime_level"] == "materialization_ready"
    assert status["blocking_reasons"] == [
        "p0_loader_runtime_evidence_pending",
        "p1_one_step_runtime_evidence_pending",
    ]


def test_openpi_p0_runtime_level_does_not_claim_p1_or_benchmark() -> None:
    status = build_openpi_formal_status(
        p0_scope_audit_manifest=_p0_manifest(materialized=True),
        join_report={"hard_blockers": []},
        label_semantics_pass=True,
        runtime_level=RUNTIME_LEVEL_P0_LOADER,
        runtime_evidence=["p0_runtime_loader_smoke.json"],
        authority_inputs=["scope_audit_manifest.json", "join_report.json"],
    )

    validate_formal_status(status)
    assert status["status"] == "BLOCK"
    assert status["formal_claim_allowed"] is False
    assert status["runtime_level"] == RUNTIME_LEVEL_P0_LOADER
    assert status["blocking_reasons"] == ["p1_one_step_runtime_evidence_pending"]
    assert "benchmark" not in str(status).lower()


def test_openpi_formal_pass_requires_p1_runtime_evidence() -> None:
    status = build_openpi_formal_status(
        p0_scope_audit_manifest=_p0_manifest(materialized=True),
        join_report={"hard_blockers": []},
        label_semantics_pass=True,
        runtime_level=RUNTIME_LEVEL_P1_ONE_STEP,
        runtime_evidence=["one_step_probe.json", "one_step_runtime.log"],
        authority_inputs=["scope_audit_manifest.json", "join_report.json"],
        validator_outputs=["label_semantics_validator.json"],
    )

    validate_formal_status(status)
    assert status["status"] == "PASS"
    assert status["formal_claim_allowed"] is True
    assert status["next_gate_allowed"] is True
    assert status["runtime_level"] == RUNTIME_LEVEL_P1_ONE_STEP
    assert status["runtime_claims"] == [
        "materialization_ready",
        RUNTIME_LEVEL_P0_LOADER,
        RUNTIME_LEVEL_P1_ONE_STEP,
    ]
    assert status["blocking_reasons"] == []


def test_openpi_runtime_level_blocks_higher_level_overclaims() -> None:
    status = build_openpi_formal_status(
        p0_scope_audit_manifest=_p0_manifest(materialized=True),
        join_report={"hard_blockers": []},
        label_semantics_pass=True,
        authority_inputs=["scope_audit_manifest.json"],
        runtime_level="p1_one_step_pass",
    )
    status["runtime_claims"] = ["p1_one_step_pass", "p2_overfit_or_tiny_update_pass"]

    try:
        validate_formal_status(status)
    except AssertionError:
        pass
    else:  # pragma: no cover - defensive branch
        raise AssertionError("p1_one_step_pass must not claim p2 runtime pass")
