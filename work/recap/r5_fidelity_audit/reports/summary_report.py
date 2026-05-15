from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

def _get(obj: Any, key: str, default: Any = "") -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)

def _qid(result: Any) -> str:
    q = _get(result, "qid") or _get(result, "question")
    return str(_get(q, "qid", q)).upper()

def _title(result: Any) -> str:
    q = _get(result, "question")
    if isinstance(q, str) and q.upper().startswith("Q") and q[1:].isdigit(): return ""
    return str(_get(q, "title") or _get(q, "intent") or q or "")

def matrix_data(results: Sequence[Any], overall_label: str) -> dict[str, Any]:
    return {"schema_version": "r5_fidelity_matrix_v1", "overall_label": overall_label, "runtime_invocations": [], "questions": [{"qid": _qid(r), "title": _title(r), "repo_presence": str(_get(r, "repo_presence", "UNCLEAR")), "active_path_consumption": str(_get(r, "active_path_consumption", "UNCLEAR")), "confidence": str(_get(r, "confidence", "LOW")), "conclusion": str(_get(r, "conclusion", ""))} for r in results]}

def render_summary(results: Sequence[Any], overall_label: str) -> str:
    rows = ["| question_id | title | repo_presence | active_path_consumption | conclusion (1-line) |", "|---|---|---|---|---|"]
    rows += [f"| {_qid(r)} | {_title(r)} | `{_get(r, 'repo_presence', 'UNCLEAR')}` | `{_get(r, 'active_path_consumption', 'UNCLEAR')}` | `{_get(r, 'conclusion', '')}` |" for r in results]
    return f"""# GR00T RECAP Fidelity Fact Report v1

## Fidelity matrix

{chr(10).join(rows)}

## Exit label

`{overall_label}`
"""

summary_data = matrix_data
render_summary_report = render_summary
