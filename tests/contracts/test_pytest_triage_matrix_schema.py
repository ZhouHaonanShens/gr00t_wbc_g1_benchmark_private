from __future__ import annotations

from pathlib import Path

from work.recap.iter6_worker2 import TYPE_LABELS, build_triage_payloads


def test_iter6_pytest_triage_matrix_classifies_all_54_failures_and_3_errors() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payloads = build_triage_payloads(repo_root)
    matrix = payloads["triage_matrix"]
    summary = payloads["summary"]

    assert matrix["schema_version"] == "iter6_pytest_triage_matrix_v1"
    assert matrix["failed_tests_expected"] == 54
    assert matrix["error_tests_expected"] == 3
    assert len(matrix["entries"]) == 57
    assert matrix["unclassified_failures"] == 0
    assert matrix["errors_unclassified"] == 0
    assert set(summary["counts_by_type"]) == set(TYPE_LABELS)
    assert summary["total_items_classified"] == 57

    for entry in matrix["entries"]:
        assert entry["type"] in TYPE_LABELS
        assert entry["rationale"]
        assert isinstance(entry["quarantine_allowed"], bool)
        assert entry["code_change_estimate_lines"] >= 0
        if entry["type"] == "A":
            assert entry["required_fix"]
            assert entry["quarantine_allowed"] is False
        else:
            assert entry["quarantine_allowed"] is True


def test_iter6_v22_critical_manifest_has_actionable_type_a_entries() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = build_triage_payloads(repo_root)["v22_critical_test_manifest"]

    assert manifest["schema_version"] == "iter6_v22_critical_test_manifest_v1"
    assert manifest["type_a_count"] > 0
    assert manifest["all_type_a_have_required_fix"] is True
    assert manifest["posttest_success_predicate"]
    for entry in manifest["type_a_tests"]:
        assert entry["required_fix"]
        assert entry["code_change_estimate_lines"] <= 200
        assert len(entry["estimated_modules_touched"]) <= 3
