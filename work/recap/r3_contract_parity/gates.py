from __future__ import annotations

from work.recap.r3_contract_parity.contract import FAIL_HOT, PASS, WARN, R3AuditError, ParityAxisResult, PatternId


def cell_overall_verdict(results: tuple[ParityAxisResult, ...]) -> str:
    if not results:
        raise R3AuditError("empty parity axis result set")
    verdicts = [result.verdict for result in results]
    if FAIL_HOT in verdicts:
        return FAIL_HOT
    if WARN in verdicts:
        return WARN
    return PASS


def collect_pattern_hits(results: tuple[ParityAxisResult, ...]) -> tuple[PatternId, ...]:
    hits: list[PatternId] = []
    for result in results:
        if result.verdict == FAIL_HOT and result.pattern_id != "none" and result.pattern_id not in hits:
            hits.append(result.pattern_id)
    return tuple(hits)
