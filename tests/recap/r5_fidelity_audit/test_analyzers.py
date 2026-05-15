from __future__ import annotations

import pytest

from work.recap.r5_fidelity_audit import analyzers
from work.recap.r5_fidelity_audit.contract import FIDELITY_QUESTIONS, R5AuditError


def _facts(*tokens: str, active: dict[str, bool] | None = None) -> dict[str, object]:
    repo_text = "\n".join(tokens)
    return {
        "repo_texts": {"work/recap/example.py": repo_text},
        "symbol_hits": {token: ({"path": "work/recap/example.py"},) for token in tokens},
        "evidence_artifacts": ("agent/artifacts/example.json",),
        "active_path_hints": active or {},
    }


def test_audit_question_rejects_unknown_question() -> None:
    with pytest.raises(R5AuditError):
        analyzers.audit_question("Q99")


def test_all_questions_return_conservative_contract_results() -> None:
    facts = _facts(
        "recap",
        "advantage_embedding",
        "indicator",
        "threshold",
        "prompt",
        "dual",
        "conditional_loss",
        "dropout",
        "runtime",
        "action_head",
        "axis",
        "A.2",
        "A.3",
        "A.4",
        "A.5",
        active={"Q1": True, "Q4": True, "Q9": True},
    )
    for question in FIDELITY_QUESTIONS:
        result = analyzers.audit_question(question.qid, facts)
        assert result.question.qid == question.qid
        assert result.repo_presence in {"IMPLEMENTED", "PARTIAL", "ABSENT", "UNCLEAR"}
        assert result.active_path_consumption in {"IMPLEMENTED", "PARTIAL", "ABSENT", "UNCLEAR"}
        assert result.evidence_files
        assert result.evidence_artifacts
        assert len(result.conclusion) <= 200


def test_q2_is_absent_without_active_path_signal_even_if_repo_symbols_present() -> None:
    result = analyzers.audit_question("Q2", _facts("critic", "value_function"))
    assert result.repo_presence == "IMPLEMENTED"
    assert result.active_path_consumption == "ABSENT"


def test_q6_requires_active_signal() -> None:
    result = analyzers.audit_question("Q6", _facts("dual", "conditional", "loss"))
    assert result.repo_presence == "IMPLEMENTED"
    assert result.active_path_consumption == "ABSENT"


def test_q8_does_not_infer_runtime_active_from_symbols_alone() -> None:
    result = analyzers.audit_question("Q8", _facts("runtime", "rollout", "action_head", "indicator"))
    assert result.repo_presence == "PARTIAL"
    assert result.active_path_consumption == "ABSENT"


def test_q9_marks_absent_repo_if_no_cell_or_artifact_signal() -> None:
    result = analyzers.audit_question("Q9", {})
    assert result.repo_presence == "ABSENT"
    assert result.active_path_consumption == "ABSENT"


def test_q9_partial_when_only_some_evidence_grade_cells_seen() -> None:
    result = analyzers.audit_question(
        "Q9",
        {
            "evidence_cell_ids": ("A.2", "A.3"),
            "evidence_files": ("work/recap/r2_authentic_eval/exclusion.py",),
            "evidence_artifacts": ("agent/artifacts/q9.json",),
        },
    )
    assert result.repo_presence == "PARTIAL"
    assert result.active_path_consumption == "ABSENT"
