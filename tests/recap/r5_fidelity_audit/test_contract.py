from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

import work.recap.r5_fidelity_audit as r5
from work.recap.r5_fidelity_audit.contract import (
    ALLOWED_CONFIDENCE,
    ALLOWED_VERDICTS,
    Confidence,
    FIDELITY_QUESTIONS,
    FULL_REPORT_FILENAME,
    FidelityQuestionResult,
    FidelityQuestionSpec,
    R5AuditError,
    Verdict,
    require_question,
)


def test_closed_question_contract_and_public_api() -> None:
    assert tuple(question.qid for question in FIDELITY_QUESTIONS) == tuple(f"Q{i}" for i in range(1, 10))
    assert len({question.qid for question in FIDELITY_QUESTIONS}) == 9
    assert all(question.analyzer_name == f"_analyze_{question.qid.lower()}" for question in FIDELITY_QUESTIONS)
    assert all(question.evidence_files for question in FIDELITY_QUESTIONS)
    assert all(question.evidence_artifacts for question in FIDELITY_QUESTIONS)
    assert r5.__all__ == (
        "FIDELITY_QUESTIONS",
        "FidelityQuestionResult",
        "audit_question",
        "FULL_REPORT_FILENAME",
    )
    assert FULL_REPORT_FILENAME == "gr00t_recap_fidelity_fact_report_v1.md"


def test_literal_aliases_are_closed() -> None:
    assert get_args(Verdict) == ALLOWED_VERDICTS
    assert get_args(Confidence) == ALLOWED_CONFIDENCE
    assert ALLOWED_VERDICTS == ("IMPLEMENTED", "PARTIAL", "ABSENT", "UNCLEAR")
    assert ALLOWED_CONFIDENCE == ("HIGH", "MEDIUM", "LOW")


def test_frozen_dataclasses_and_question_lookup() -> None:
    question = require_question("q1")
    assert isinstance(question, FidelityQuestionSpec)
    with pytest.raises(FrozenInstanceError):
        question.qid = "QX"  # type: ignore[misc]
    result = FidelityQuestionResult(
        question=question,
        repo_presence="PARTIAL",
        active_path_consumption="ABSENT",
        confidence="MEDIUM",
        evidence_files=("work/recap/advantage.py",),
        evidence_artifacts=("agent/exchange/example.md",),
        conclusion="Static audit result.",
    )
    with pytest.raises(FrozenInstanceError):
        result.conclusion = "mutated"  # type: ignore[misc]
    with pytest.raises(R5AuditError):
        require_question("Q99")
    assert issubclass(R5AuditError, RuntimeError)
