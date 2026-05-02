from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import audit_carrier_lineage


def test_unique_first_failing_stage_when_earlier_stage_passes() -> None:
    canonical = ["rk1", "rk2"]
    stage_summaries = [
        audit_carrier_lineage.summarize_stage_coverage(
            audit_carrier_lineage.build_stage_coverage(
                stage_name=audit_carrier_lineage.STAGE_LABEL_MATERIALIZATION,
                canonical_row_keys=canonical,
                pass_row_keys=canonical,
            ),
            canonical_row_keys=canonical,
        ),
        audit_carrier_lineage.summarize_stage_coverage(
            audit_carrier_lineage.build_stage_coverage(
                stage_name=audit_carrier_lineage.STAGE_STATE_CONDITIONED_BUILD,
                canonical_row_keys=canonical,
                fail_row_keys=canonical,
            ),
            canonical_row_keys=canonical,
        ),
    ]

    outcome = audit_carrier_lineage.resolve_audit_outcome(
        stage_summaries,
        canonical_row_keys=canonical,
    )

    assert outcome["audit_status"] == audit_carrier_lineage.AUDIT_STATUS_COMPLETE
    assert (
        outcome["first_failing_stage"]
        == audit_carrier_lineage.STAGE_STATE_CONDITIONED_BUILD
    )


def test_fail_closed_when_earlier_stage_evidence_is_missing() -> None:
    canonical = ["rk1", "rk2"]
    stage_summaries = [
        audit_carrier_lineage.summarize_stage_coverage(
            audit_carrier_lineage.build_stage_coverage(
                stage_name=audit_carrier_lineage.STAGE_LABEL_MATERIALIZATION,
                canonical_row_keys=canonical,
                pass_row_keys=["rk1"],
                unresolved_row_keys=["rk2"],
            ),
            canonical_row_keys=canonical,
        ),
        audit_carrier_lineage.summarize_stage_coverage(
            audit_carrier_lineage.build_stage_coverage(
                stage_name=audit_carrier_lineage.STAGE_STATE_CONDITIONED_BUILD,
                canonical_row_keys=canonical,
                fail_row_keys=canonical,
            ),
            canonical_row_keys=canonical,
        ),
    ]

    outcome = audit_carrier_lineage.resolve_audit_outcome(
        stage_summaries,
        canonical_row_keys=canonical,
    )

    assert outcome["audit_status"] == audit_carrier_lineage.AUDIT_STATUS_FAIL_CLOSED
    assert outcome["first_failing_stage"] is None


def test_fail_closed_when_two_stages_fail_on_different_subsets() -> None:
    canonical = ["rk1", "rk2"]
    stage_summaries = [
        audit_carrier_lineage.summarize_stage_coverage(
            audit_carrier_lineage.build_stage_coverage(
                stage_name=audit_carrier_lineage.STAGE_LABEL_MATERIALIZATION,
                canonical_row_keys=canonical,
                fail_row_keys=["rk1"],
                pass_row_keys=["rk2"],
            ),
            canonical_row_keys=canonical,
        ),
        audit_carrier_lineage.summarize_stage_coverage(
            audit_carrier_lineage.build_stage_coverage(
                stage_name=audit_carrier_lineage.STAGE_STATE_CONDITIONED_BUILD,
                canonical_row_keys=canonical,
                pass_row_keys=["rk1"],
                fail_row_keys=["rk2"],
            ),
            canonical_row_keys=canonical,
        ),
    ]

    outcome = audit_carrier_lineage.resolve_audit_outcome(
        stage_summaries,
        canonical_row_keys=canonical,
    )

    assert outcome["audit_status"] == audit_carrier_lineage.AUDIT_STATUS_FAIL_CLOSED
    assert outcome["first_failing_stage"] is None


def test_unique_first_failing_stage_survives_provenance_blockers() -> None:
    canonical = ["rk1", "rk2"]
    stage_summaries = [
        audit_carrier_lineage.summarize_stage_coverage(
            audit_carrier_lineage.build_stage_coverage(
                stage_name=audit_carrier_lineage.STAGE_LABEL_MATERIALIZATION,
                canonical_row_keys=canonical,
                fail_row_keys=canonical,
            ),
            canonical_row_keys=canonical,
        )
    ]

    outcome = audit_carrier_lineage.resolve_audit_outcome(
        stage_summaries,
        canonical_row_keys=canonical,
        provenance_blockers=[
            {
                "code": "output_dataset_backpointer_mismatch",
                "message": "source_dataset_ref.output_dataset_dir drifted",
            }
        ],
    )

    assert outcome["audit_status"] == audit_carrier_lineage.AUDIT_STATUS_COMPLETE
    assert (
        outcome["first_failing_stage"]
        == audit_carrier_lineage.STAGE_LABEL_MATERIALIZATION
    )
    assert "lineage_binding_blocked" in outcome["reason_codes"]


def test_stable_row_key_remains_stable_when_sample_id_is_blank() -> None:
    manifest_context = {
        "execution_sha": "abc123",
        "manifest_hash": "hash-001",
    }
    key_a = audit_carrier_lineage.stable_row_lineage_key(
        source_artifact_id="frozen_mainline_labels",
        resolved_path=REPO_ROOT / "agent/artifacts/demo/labels.jsonl",
        content_sha256="digest-001",
        row_locator="line:12",
        episode_id="episode_001",
        t=7,
        manifest_context=manifest_context,
    )
    key_b = audit_carrier_lineage.stable_row_lineage_key(
        source_artifact_id="frozen_mainline_labels",
        resolved_path=REPO_ROOT / "agent/artifacts/demo/labels.jsonl",
        content_sha256="digest-001",
        row_locator="line:12",
        episode_id="episode_001",
        t=7,
        manifest_context=manifest_context,
    )
    key_c = audit_carrier_lineage.stable_row_lineage_key(
        source_artifact_id="frozen_mainline_labels",
        resolved_path=REPO_ROOT / "agent/artifacts/demo/labels.jsonl",
        content_sha256="digest-001",
        row_locator="line:13",
        episode_id="episode_001",
        t=7,
        manifest_context=manifest_context,
    )

    assert key_a == key_b
    assert key_a != key_c
