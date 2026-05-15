from __future__ import annotations

import pytest

from work.recap.r5_fidelity_audit import verdicts
from work.recap.r5_fidelity_audit.contract import R5AuditError, require_question


def _r(question: str, repo: str, active: str):
    return verdicts.compose_question_result(
        question=require_question(question),
        repo_presence=repo,
        active_path_consumption=active,
        evidence_files=("a.json",),
        evidence_artifacts=("b.json",),
        conclusion="short text",
        confidence="MEDIUM",
    )


def test_compose_question_result_rejects_invalid_evidence() -> None:
    with pytest.raises(R5AuditError):
        verdicts.compose_question_result(
            question=require_question("Q1"),
            repo_presence="YES",
            active_path_consumption="NO",
            evidence_files=(),
            evidence_artifacts=("x",),
            conclusion="x",
            confidence="LOW",
        )


def test_compose_question_result_stores_analyzer_metadata_in_details() -> None:
    result = verdicts.compose_question_result(
        question="Q1",
        repo_presence="IMPLEMENTED",
        active_path_consumption="PARTIAL",
        evidence_files=("a",),
        evidence_artifacts=("b",),
        conclusion="short text",
        confidence="HIGH",
        analyzer_name="custom_analyzer",
    )
    assert result.question.qid == "Q1"
    assert result.details["analyzer_name"] == "custom_analyzer"


def test_overall_full_fidelity_priority_requires_nine_active_results() -> None:
    results = tuple(_r(f"Q{i}", "IMPLEMENTED", "IMPLEMENTED") for i in range(1, 10))
    assert verdicts.overall_fidelity_label(results) == "FULL_FIDELITY"
    assert verdicts.overall_fidelity_label(results[:8]) == "PARTIAL_FIDELITY"


def test_overall_detached_runtime_path_priority() -> None:
    results = (
        _r("Q1", "IMPLEMENTED", "IMPLEMENTED"),
        _r("Q2", "IMPLEMENTED", "ABSENT"),
        _r("Q3", "ABSENT", "ABSENT"),
    )
    assert verdicts.overall_fidelity_label(results) == "DETACHED_RUNTIME_PATH"


def test_overall_unclear_priority() -> None:
    results = (
        _r("Q1", "IMPLEMENTED", "IMPLEMENTED"),
        _r("Q2", "UNCLEAR", "ABSENT"),
        _r("Q3", "IMPLEMENTED", "IMPLEMENTED"),
    )
    assert verdicts.overall_fidelity_label(results) == "UNCLEAR"


def test_overall_partial_when_absent_repo_with_absent_active() -> None:
    results = (
        _r("Q1", "ABSENT", "ABSENT"),
        _r("Q2", "IMPLEMENTED", "IMPLEMENTED"),
    )
    assert verdicts.overall_fidelity_label(results) == "PARTIAL_FIDELITY"


def test_validate_question_results_roundtrips_mapping() -> None:
    payload = {
        "Q1": {
            "analyzer_name": "x",
            "repo_presence": "IMPLEMENTED",
            "active_path_consumption": "PARTIAL",
            "evidence_files": ("a",),
            "evidence_artifacts": ("b",),
            "conclusion": "short text",
            "confidence": "HIGH",
        }
    }
    normalized = verdicts.validate_question_results(payload)
    assert len(normalized) == 1
    assert normalized[0].question.qid == "Q1"
    assert normalized[0].details["analyzer_name"] == "x"
