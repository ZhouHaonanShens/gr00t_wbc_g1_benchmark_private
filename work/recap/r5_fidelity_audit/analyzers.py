from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from work.recap.r5_fidelity_audit.contract import FIDELITY_QUESTIONS, R5AuditError, require_question
from work.recap.r5_fidelity_audit.verdicts import ABSENT, HIGH, IMPLEMENTED, LOW, MEDIUM, PARTIAL, compose_question_result

_ANALYZERS: dict[str, Callable[[Any, Mapping[str, Any]], Any]] = {}

def audit_question(qid_or_question: str | Any, collected: Mapping[str, Any] | None = None) -> Any:
    q = require_question(qid_or_question)
    facts = collected if collected is not None else _collect_default_inputs()
    return _ANALYZERS[q.qid](q, facts)

def _collect_default_inputs() -> Mapping[str, Any]:
    from work.recap.r5_fidelity_audit import collectors
    data = collectors.collect_static_sources()
    return data if isinstance(data, Mapping) else {}

def _texts(c: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in ("repo_texts", "repo_files", "file_texts"):
        val = c.get(key)
        if isinstance(val, Mapping):
            for path, text in val.items():
                if isinstance(text, str):
                    out[str(path)] = text
    for path in c.get("source_paths", ()) or ():
        out.setdefault(str(path), "")
    return out

def _flat(c: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for path, text in _texts(c).items():
        parts.extend((path, text))
    for key in ("phase_a_literals", "dataset_meta", "ckpt_cfg_by_cell", "active_path_hints"):
        value = c.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts).lower()

def _has(c: Mapping[str, Any], *tokens: str) -> bool:
    text = _flat(c)
    if any(token.lower() in text for token in tokens):
        return True

    hits = c.get("symbol_hits")
    wanted = {token.lower() for token in tokens}
    if isinstance(hits, Mapping):
        return any(str(symbol).lower() in wanted and bool(value) for symbol, value in hits.items())
    if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)):
        return any(any(token in str(hit).lower() for token in wanted) for hit in hits)
    return False

def _hit_path(hit: Any) -> str:
    return str(hit.get("path") or hit.get("file") or hit) if isinstance(hit, Mapping) else str(hit)

def _paths(q: Any, c: Mapping[str, Any], *tokens: str) -> tuple[str, ...]:
    out = [p for p, t in _texts(c).items() if not tokens or any(x.lower() in (p + t).lower() for x in tokens)]
    hits = c.get("symbol_hits")
    if isinstance(hits, Mapping):
        for sym, vals in hits.items():
            if tokens and not any(x.lower() in str(sym).lower() for x in tokens): continue
            if isinstance(vals, Sequence) and not isinstance(vals, (str, bytes)): out += [_hit_path(v) for v in vals]
            elif vals: out.append(str(sym))
    return tuple(dict.fromkeys([p for p in out if p] or list(q.repo_files_to_inspect[:1])))

def _arts(q: Any, c: Mapping[str, Any], *tokens: str) -> tuple[str, ...]:
    out: list[str] = []
    for key in ("evidence_artifacts", "artifact_paths"):
        vals = c.get(key)
        vals = vals.values() if isinstance(vals, Mapping) else vals
        if isinstance(vals, Sequence) and not isinstance(vals, (str, bytes)): out += [str(v) for v in vals if v]
    presence = c.get("artifact_presence")
    if isinstance(presence, Mapping): out += [str(p) for p, ok in presence.items() if ok and (not tokens or any(t.lower() in str(p).lower() for t in tokens))]
    return tuple(dict.fromkeys(out or list(q.artifact_paths_to_inspect[:1])))

def _hint(c: Mapping[str, Any], qid: str, *tokens: str) -> bool | None:
    hints = c.get("active_path_hints")
    if isinstance(hints, Mapping) and qid in hints:
        return bool(hints[qid])
    if f"{qid}_active" in c:
        return bool(c[f"{qid}_active"])
    cfg = str(c.get("ckpt_cfg_by_cell", "")).lower()
    return bool(tokens and cfg and any(t.lower() in cfg for t in tokens)) if cfg else None

def _enabled(c: Mapping[str, Any], *tokens: str) -> bool:
    text = str(c.get("ckpt_cfg_by_cell", "")).lower()
    return bool(text and any(t.lower() in text for t in tokens) and not any(x in text for x in ("false", "disabled", "none")))

def _lit(c: Mapping[str, Any], *tokens: str) -> bool:
    text = (str(c.get("phase_a_literals", "")) + str(c.get("dataset_meta", ""))).lower()
    return any(t.lower() in text for t in tokens)

def _res(q: Any, repo: str, active: str, conf: str, conclusion: str, c: Mapping[str, Any], *tokens: str) -> Any:
    return compose_question_result(
        q,
        repo_presence=repo,
        active_path_consumption=active,
        confidence=conf,
        conclusion=conclusion,
        evidence_files=_paths(q, c, *tokens),
        evidence_artifacts=_arts(q, c, *tokens),
    )

def _active(c: Mapping[str, Any], qid: str, *tokens: str) -> str:
    hint = _hint(c, qid, *tokens)
    if hint is True or _enabled(c, *tokens):
        return IMPLEMENTED
    return ABSENT

def _analyze_q1_scope(q: Any, c: Mapping[str, Any]) -> Any:
    hits = sum(_has(c, t) for t in ("recap", "advantage", "indicator", "critic", "condition"))
    return _res(q, IMPLEMENTED if hits >= 3 else PARTIAL if hits else ABSENT, PARTIAL if _active(c, "Q1", "recap", "indicator") == IMPLEMENTED else ABSENT, MEDIUM, "RECAP component coverage is separated from active-path consumption evidence.", c, "recap", "indicator", "critic")

def _analyze_q2_value_function(q: Any, c: Mapping[str, Any]) -> Any:
    repo = IMPLEMENTED if _has(c, "critic", "value_function", "value model", "vlm_critic") else ABSENT
    active = _active(c, "Q2", "critic", "value_function", "value_model") if repo != ABSENT else ABSENT
    return _res(q, repo, active, HIGH if active == IMPLEMENTED else MEDIUM, "Value/critic symbols count active only with checkpoint or config consumption evidence.", c, "critic", "value")

def _analyze_q3_advantage_source(q: Any, c: Mapping[str, Any]) -> Any:
    learned, static = _lit(c, "learned_advantage", "learned advantage", "advantage_embedding=learned"), _has(c, "advantage_embedding", "static relabel", "advantage")
    return _res(q, IMPLEMENTED if learned else PARTIAL if static else ABSENT, IMPLEMENTED if learned and _enabled(c, "advantage") else ABSENT, HIGH if learned else MEDIUM, "Advantage embedding is learned only with explicit learned/config evidence.", c, "advantage")

def _analyze_q4_indicator_threshold(q: Any, c: Mapping[str, Any]) -> Any:
    threshold = _lit(c, "threshold", "delta>0", "binary_improvement") or _has(c, "threshold", "binary_improvement", "improvement_indicator")
    return _res(q, IMPLEMENTED if threshold else ABSENT, IMPLEMENTED if threshold and (_enabled(c, "indicator") or _hint(c, "Q4", "threshold") is True) else ABSENT, MEDIUM, "Binary indicator threshold evidence must be explicit in literals or training config.", c, "threshold", "indicator")

def _analyze_q5_indicator_placement(q: Any, c: Mapping[str, Any]) -> Any:
    placement = _has(c, "indicator", "input_sequence", "prompt", "token") or _lit(c, "indicator_position", "input_sequence")
    return _res(q, IMPLEMENTED if placement else ABSENT, IMPLEMENTED if placement and _enabled(c, "indicator", "input") else ABSENT, MEDIUM, "Training-side placement requires placement symbols and active dataset/config wiring.", c, "indicator", "prompt", "input")

def _analyze_q6_loss_objective(q: Any, c: Mapping[str, Any]) -> Any:
    dual, single = _has(c, "dual", "conditional", "unconditional", "conditional_loss"), _has(c, "mse", "single_loss")
    active = _active(c, "Q6", "dual", "conditional_loss") if dual else ABSENT
    return _res(q, IMPLEMENTED if dual else PARTIAL if single else ABSENT, active, HIGH if active == IMPLEMENTED else MEDIUM, "Dual-loss fidelity is active only when conditional loss appears in training config/artifacts.", c, "loss", "conditional", "mse")

def _analyze_q7_indicator_dropout(q: Any, c: Mapping[str, Any]) -> Any:
    drop = _has(c, "indicator_dropout", "drop_indicator", "omit_indicator", "dropout")
    return _res(q, IMPLEMENTED if drop else ABSENT, IMPLEMENTED if drop and _enabled(c, "dropout", "indicator") else ABSENT, MEDIUM, "Indicator dropout is absent unless explicit dropout code is wired into training config.", c, "dropout", "indicator")

def _analyze_q8_runtime_indicator_consumption(q: Any, c: Mapping[str, Any]) -> Any:
    runtime = _has(c, "runtime", "rollout", "action_head", "serving", "indicator")
    active = _active(c, "Q8", "runtime_indicator", "action_head", "serving_indicator") if runtime else ABSENT
    return _res(q, IMPLEMENTED if active == IMPLEMENTED else PARTIAL if runtime else ABSENT, active, HIGH if active == IMPLEMENTED else MEDIUM, "Runtime indicator consumption requires serving/rollout evidence, not symbol presence alone.", c, "runtime", "action", "indicator")

def _q9_cells() -> tuple[str, ...]:
    from work.recap.r2_authentic_eval import exclusion
    return tuple(str(cell) for cell in exclusion.EVIDENCE_GRADE_CELL_IDS)

def _analyze_q9_cell_diff(q: Any, c: Mapping[str, Any]) -> Any:
    expected, cfg = set(_q9_cells()), c.get("ckpt_cfg_by_cell")
    seen = set(str(x) for x in c.get("evidence_cell_ids", ()) or ()) or (set(str(x) for x in cfg) if isinstance(cfg, Mapping) else set())
    cfg_map = cfg if isinstance(cfg, Mapping) else {}
    sourced = {str(cell) for cell, payload in cfg_map.items() if isinstance(payload, Mapping) and payload.get("source_paths")}
    observed = seen or sourced
    repo = IMPLEMENTED if expected and expected <= observed else PARTIAL if observed & expected or _has(c, "axis", "A.2", "A.3", "A.4", "A.5") else ABSENT
    active = IMPLEMENTED if _hint(c, "Q9", "axis") is True or bool(expected and expected <= sourced) else ABSENT
    return _res(q, repo, active, LOW if repo == ABSENT else MEDIUM, "A.2-A.5 axis fidelity uses the R2 evidence-grade cell SSOT and never infers A.1 coverage.", c, "A.2", "A.3", "axis", "exclusion")

_analyze_q1 = _analyze_q1_scope
_analyze_q2 = _analyze_q2_value_function
_analyze_q3 = _analyze_q3_advantage_source
_analyze_q4 = _analyze_q4_indicator_threshold
_analyze_q5 = _analyze_q5_indicator_placement
_analyze_q6 = _analyze_q6_loss_objective
_analyze_q7 = _analyze_q7_indicator_dropout
_analyze_q8 = _analyze_q8_runtime_indicator_consumption
_analyze_q9 = _analyze_q9_cell_diff

def _register() -> dict[str, Callable[[Any, Mapping[str, Any]], Any]]:
    funcs = {q.qid: globals()[q.analyzer_name] for q in FIDELITY_QUESTIONS}
    funcs.update({f"_analyze_{i}": funcs[f"Q{i}"] for i in range(1, 10)})
    return funcs

_ANALYZERS = _register()
