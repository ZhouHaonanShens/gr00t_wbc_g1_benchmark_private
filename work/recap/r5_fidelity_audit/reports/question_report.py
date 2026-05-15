from __future__ import annotations

from collections.abc import Mapping
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

def _bullets(values: Any) -> str:
    vals = [values] if isinstance(values, (str, bytes)) else list(values or ())
    return "\n".join(f"- `{v.decode() if isinstance(v, bytes) else v}`" for v in vals) or "- `NOT_RECORDED`"

def render_question_report(result: Any) -> str:
    return f"""# R5 Fidelity Question Report — {_qid(result)}

- title: {_title(result)}
- repo_presence: `{_get(result, 'repo_presence', 'UNCLEAR')}`
- active_path_consumption: `{_get(result, 'active_path_consumption', 'UNCLEAR')}`
- confidence: `{_get(result, 'confidence', 'LOW')}`
- conclusion: {_get(result, 'conclusion', '')}

## Evidence files

{_bullets(_get(result, 'evidence_files', ())) }

## Evidence artifacts

{_bullets(_get(result, 'evidence_artifacts', ())) }
"""

render = render_question_report
