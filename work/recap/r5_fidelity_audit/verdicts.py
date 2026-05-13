from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from work.recap.r5_fidelity_audit.contract import (
    ABSENT, ALLOWED_CONFIDENCE, ALLOWED_VERDICTS, HIGH, IMPLEMENTED, LOW, MEDIUM, PARTIAL,
    UNCLEAR, FidelityQuestionResult, R5AuditError, require_question,
)

FULL_FIDELITY = "FULL_FIDELITY"; DETACHED_RUNTIME_PATH = "DETACHED_RUNTIME_PATH"; PARTIAL_FIDELITY = "PARTIAL_FIDELITY"

def _as_tuple(values: Sequence[str] | Iterable[str]) -> tuple[str, ...]:
    return tuple(str(v) for v in values if str(v))

def _analyzer_name(question: Any) -> str:
    return str(getattr(question, "analyzer_name", f"_analyze_{getattr(question, 'qid', question).lower()}"))

def compose_question_result(question: Any, *, repo_presence: str, active_path_consumption: str, confidence: str, conclusion: str, evidence_files: Sequence[str] | Iterable[str], evidence_artifacts: Sequence[str] | Iterable[str], analyzer_name: str | None = None, details: Mapping[str, Any] | None = None, **_: Any) -> FidelityQuestionResult:
    q = require_question(question)
    if repo_presence not in ALLOWED_VERDICTS: raise R5AuditError(f"invalid repo_presence verdict: {repo_presence}")
    if active_path_consumption not in ALLOWED_VERDICTS: raise R5AuditError(f"invalid active_path_consumption verdict: {active_path_consumption}")
    if confidence not in ALLOWED_CONFIDENCE: raise R5AuditError(f"invalid confidence: {confidence}")
    text = conclusion.strip()
    if not text or "\n" in text or len(text) > 200: raise R5AuditError("conclusion must be one non-empty paragraph of at most 200 characters")
    files, artifacts = _as_tuple(evidence_files), _as_tuple(evidence_artifacts)
    if not files: raise R5AuditError("at least one evidence_files entry is required")
    if not artifacts: raise R5AuditError("at least one evidence_artifacts entry is required")
    meta = dict(details or {}, qid=q.qid, title=q.title, analyzer_name=analyzer_name or _analyzer_name(q))
    return FidelityQuestionResult(q, repo_presence, active_path_consumption, confidence, files, artifacts, text, meta)

def overall_fidelity_label(results: Iterable[Any]) -> str:
    items = tuple(results); pairs = tuple((str(r.repo_presence), str(r.active_path_consumption)) for r in items)
    if any(UNCLEAR in pair for pair in pairs): return UNCLEAR
    if len(items) == 9 and all(active == IMPLEMENTED for _, active in pairs): return FULL_FIDELITY
    if any(active == ABSENT and repo in {IMPLEMENTED, PARTIAL} for repo, active in pairs): return DETACHED_RUNTIME_PATH
    if any(active == ABSENT and repo == ABSENT for repo, active in pairs): return PARTIAL_FIDELITY
    return PARTIAL_FIDELITY

def validate_question_results(results: Mapping[str, Any]) -> tuple[Any, ...]:
    out = []
    for qid, payload in results.items():
        if hasattr(payload, "repo_presence") and hasattr(payload, "active_path_consumption"):
            out.append(payload); continue
        if not isinstance(payload, Mapping): raise R5AuditError(f"result[{qid}] must be a mapping")
        out.append(compose_question_result(qid, repo_presence=str(payload.get("repo_presence", UNCLEAR)), active_path_consumption=str(payload.get("active_path_consumption", UNCLEAR)), confidence=str(payload.get("confidence", LOW)), conclusion=str(payload.get("conclusion", "missing conclusion")), evidence_files=payload.get("evidence_files", ()), evidence_artifacts=payload.get("evidence_artifacts", ()), analyzer_name=str(payload.get("analyzer_name", f"_analyze_{qid.lower()}")), details=payload.get("details") if isinstance(payload.get("details"), Mapping) else None))
    return tuple(out)
