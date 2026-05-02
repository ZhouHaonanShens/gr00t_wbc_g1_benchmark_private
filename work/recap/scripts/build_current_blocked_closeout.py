#!/usr/bin/env python3

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import apple_recap_execution_contract


DEFAULT_EXECUTION_ROOT = Path("agent/artifacts/apple_recap_exec")
DEFAULT_CARRIER_REPORT = DEFAULT_EXECUTION_ROOT / "carrier_parity_report.json"
DEFAULT_T10_EVIDENCE = Path(".sisyphus/evidence/task-11-carrier-parity.json")
DEFAULT_FREEZE_CONTRACT = DEFAULT_EXECUTION_ROOT / "execution_freeze_contract.json"
DEFAULT_LEDGER_CSV = DEFAULT_EXECUTION_ROOT / "B0_E1_E2_run_ledger.csv"
DEFAULT_DRAFT_FINAL_PACK = (
    DEFAULT_EXECUTION_ROOT / "final_report" / "final_verdict_pack.json"
)
DEFAULT_UPLIFT_VERDICT_JSON = DEFAULT_EXECUTION_ROOT / "uplift_verdict.json"
DEFAULT_UPLIFT_VERDICT_MD = DEFAULT_EXECUTION_ROOT / "uplift_verdict.md"
DEFAULT_BLOCK_REASON_JSON = (
    DEFAULT_EXECUTION_ROOT / "block_reasons" / "carrier_export_authority_violation.json"
)

CLOSEOUT_SCHEMA_VERSION = "apple_recap_blocked_closeout_v1"
CLOSEOUT_ARTIFACT_KIND = "apple_recap_blocked_closeout"
BLOCK_REASON_SCHEMA_VERSION = "apple_recap_block_reason_v1"
BLOCK_REASON_ARTIFACT_KIND = "apple_recap_block_reason"
EXPECTED_BLOCK_REASON = "carrier_export_authority_violation"
EXPECTED_STATUS = "BLOCK"
EXPECTED_TERMINAL_STATE = "blocked"
EXPECTED_AUTHORITY_LEVEL = "blocked_closeout"
EXPECTED_THEORY_VERDICT = "not yet proven"
EXPECTED_BLOCK_STAGE = "T10_formal_carrier_parity"
EXPECTED_MISSING_FIELD = "carrier_text_v1"
EXPECTED_SOURCE_FIELDS_PRESENT: tuple[str, ...] = ("prompt_raw", "indicator_I")
EXPECTED_TASK_ID = "T10"


def _resolve_repo_path(repo_root: Path, raw: Path | str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_relative_path(repo_root: Path, path: Path | str) -> str:
    resolved = _resolve_repo_path(repo_root, path)
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, Any], payload)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _signature_for_payload(payload: dict[str, Any]) -> str:
    signature_basis = {
        str(key): value
        for key, value in payload.items()
        if key != "report_signature_sha256"
    }
    return apple_recap_execution_contract._sha256_payload(signature_basis)


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _parse_timestamp(value: object, *, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name=field_name)
    try:
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    return normalized


def _require_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return cast(dict[str, Any], value)


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _extract_freshness(payload: dict[str, Any], *, field_name: str) -> dict[str, str]:
    freshness = payload.get(field_name, payload if field_name == "" else None)
    if field_name and not isinstance(freshness, dict):
        raise TypeError(f"{field_name} must be an object")
    source = cast(dict[str, Any], freshness if field_name else payload)
    return {
        "execution_sha": _non_empty_string(
            source.get("execution_sha"),
            field_name=f"{field_name or 'payload'}.execution_sha",
        ),
        "manifest_hash": _non_empty_string(
            source.get("manifest_hash"),
            field_name=f"{field_name or 'payload'}.manifest_hash",
        ),
        "checkpoint_id": _non_empty_string(
            source.get("checkpoint_id"),
            field_name=f"{field_name or 'payload'}.checkpoint_id",
        ),
        "seed_bundle_id": _non_empty_string(
            source.get("seed_bundle_id"),
            field_name=f"{field_name or 'payload'}.seed_bundle_id",
        ),
        "timestamp": _parse_timestamp(
            source.get("timestamp"), field_name=f"{field_name or 'payload'}.timestamp"
        ),
    }


def _assert_same_freshness(*bundles: tuple[str, dict[str, str]]) -> dict[str, str]:
    if not bundles:
        raise ValueError("at least one freshness bundle is required")
    anchor_name, anchor = bundles[0]
    for name, bundle in bundles[1:]:
        if bundle != anchor:
            raise ValueError(
                f"freshness mismatch between {anchor_name} and {name}: {anchor!r} != {bundle!r}"
            )
    return dict(anchor)


def _load_ledger_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("run ledger must contain at least one row")
    return rows


def _extract_execution_verdict(
    rows: list[dict[str, str]], *, execution_sha: str
) -> str:
    verdicts = {
        _non_empty_string(row.get("verdict"), field_name="ledger.verdict")
        for row in rows
        if _non_empty_string(
            row.get("execution_sha"), field_name="ledger.execution_sha"
        )
        == execution_sha
    }
    if len(verdicts) != 1:
        raise ValueError(
            f"ledger must yield exactly one execution verdict, got {sorted(verdicts)!r}"
        )
    return next(iter(verdicts))


def _extract_stage_status(rows: list[dict[str, str]], *, row_id: str) -> str:
    matches = [row for row in rows if row.get("row_id") == row_id]
    if len(matches) != 1:
        raise ValueError(f"ledger must contain exactly one {row_id} row")
    return _non_empty_string(
        matches[0].get("stage_status"), field_name=f"ledger.{row_id}.stage_status"
    )


def _derive_missing_field(carrier_report: dict[str, Any]) -> str:
    training_text_field = _non_empty_string(
        carrier_report.get("training_text_field"),
        field_name="carrier_report.training_text_field",
    )
    presence_summary = _require_mapping(
        carrier_report.get("field_presence_summary"),
        field_name="carrier_report.field_presence_summary",
    )
    summary = presence_summary.get(training_text_field)
    if not isinstance(summary, dict):
        raise ValueError(
            f"carrier_report.field_presence_summary missing training field {training_text_field!r}"
        )
    full_scan_row_count = _require_int(
        carrier_report.get("full_scan_row_count"),
        field_name="carrier_report.full_scan_row_count",
    )
    if (
        _require_int(
            summary.get("missing_count"),
            field_name="field_presence_summary.missing_count",
        )
        != full_scan_row_count
    ):
        raise ValueError(
            "training_text_field is not missing for the full scan row count"
        )
    if (
        _require_int(
            summary.get("present_count"),
            field_name="field_presence_summary.present_count",
        )
        != 0
    ):
        raise ValueError("training_text_field unexpectedly present in frozen labels")
    return training_text_field


def _derive_source_fields_present(carrier_report: dict[str, Any]) -> list[str]:
    presence_summary = _require_mapping(
        carrier_report.get("field_presence_summary"),
        field_name="carrier_report.field_presence_summary",
    )
    full_scan_row_count = _require_int(
        carrier_report.get("full_scan_row_count"),
        field_name="carrier_report.full_scan_row_count",
    )
    present_fields: list[str] = []
    for field_name in EXPECTED_SOURCE_FIELDS_PRESENT:
        summary = presence_summary.get(field_name)
        if not isinstance(summary, dict):
            raise ValueError(
                f"carrier_report.field_presence_summary missing {field_name!r}"
            )
        if (
            _require_int(
                summary.get("present_count"),
                field_name=f"field_presence_summary.{field_name}.present_count",
            )
            == full_scan_row_count
        ):
            present_fields.append(field_name)
    if present_fields != list(EXPECTED_SOURCE_FIELDS_PRESENT):
        raise ValueError(
            f"unexpected source fields present: {present_fields!r}; expected {list(EXPECTED_SOURCE_FIELDS_PRESENT)!r}"
        )
    return present_fields


def _derive_block_stage(t10_evidence: dict[str, Any]) -> str:
    task_id = _non_empty_string(
        t10_evidence.get("task_id"), field_name="t10_evidence.task_id"
    )
    if task_id != EXPECTED_TASK_ID:
        raise ValueError(f"expected task_id={EXPECTED_TASK_ID}, got {task_id}")
    task_slug = _non_empty_string(
        t10_evidence.get("task_slug"), field_name="t10_evidence.task_slug"
    )
    return f"{task_id}_formal_{task_slug.replace('-', '_')}"


def _derive_block_reason(
    *,
    carrier_report: dict[str, Any],
    t10_evidence: dict[str, Any],
    missing_field: str,
    source_fields_present: list[str],
) -> str:
    authority_violation_count = _require_int(
        carrier_report.get("authority_violation_count"),
        field_name="carrier_report.authority_violation_count",
    )
    blocker_summary = _non_empty_string(
        t10_evidence.get("blocker_reason"), field_name="t10_evidence.blocker_reason"
    )
    if authority_violation_count <= 0:
        raise ValueError("carrier parity report is not blocked")
    if missing_field != EXPECTED_MISSING_FIELD:
        raise ValueError(f"unexpected missing field: {missing_field}")
    if source_fields_present != list(EXPECTED_SOURCE_FIELDS_PRESENT):
        raise ValueError(
            "source field presence does not match carrier export authority violation"
        )
    if "do not materialize carrier_text_v1" not in blocker_summary:
        raise ValueError(
            "T10 evidence does not describe the carrier export authority violation"
        )
    return EXPECTED_BLOCK_REASON


def _source_ref(
    repo_root: Path, *, artifact_id: str, authority_role: str, relative_path: Path | str
) -> dict[str, Any]:
    return apple_recap_execution_contract.build_read_only_authority_ref(
        repo_root=repo_root,
        artifact_id=artifact_id,
        authority_role=authority_role,
        relative_path=_repo_relative_path(repo_root, relative_path),
    )


def _build_non_authoritative_input(
    repo_root: Path,
    *,
    artifact_id: str,
    relative_path: Path | str,
    reason: str,
) -> dict[str, Any]:
    payload = _source_ref(
        repo_root,
        artifact_id=artifact_id,
        authority_role="non_authoritative_draft_input",
        relative_path=relative_path,
    )
    payload["reason"] = reason
    payload["authority_disposition"] = "must_not_override_blocked_closeout"
    return payload


def build_current_blocked_closeout(
    *,
    repo_root: Path = REPO_ROOT,
    execution_root: Path | str = DEFAULT_EXECUTION_ROOT,
    carrier_report_json: Path | str = DEFAULT_CARRIER_REPORT,
    t10_evidence_json: Path | str = DEFAULT_T10_EVIDENCE,
    freeze_contract_json: Path | str = DEFAULT_FREEZE_CONTRACT,
    ledger_csv: Path | str = DEFAULT_LEDGER_CSV,
    draft_final_pack_json: Path | str = DEFAULT_DRAFT_FINAL_PACK,
) -> dict[str, Any]:
    resolved_execution_root = _resolve_repo_path(repo_root, execution_root)
    carrier_report_path = _resolve_repo_path(repo_root, carrier_report_json)
    t10_evidence_path = _resolve_repo_path(repo_root, t10_evidence_json)
    freeze_contract_path = _resolve_repo_path(repo_root, freeze_contract_json)
    ledger_path = _resolve_repo_path(repo_root, ledger_csv)
    draft_final_pack_path = _resolve_repo_path(repo_root, draft_final_pack_json)

    carrier_report = _read_json(carrier_report_path)
    t10_evidence = _read_json(t10_evidence_path)
    freeze_contract = _read_json(freeze_contract_path)
    ledger_rows = _load_ledger_rows(ledger_path)

    freshness = _assert_same_freshness(
        ("carrier_report", _extract_freshness(carrier_report, field_name="freshness")),
        ("t10_evidence", _extract_freshness(t10_evidence, field_name="")),
        (
            "freeze_contract",
            _extract_freshness(freeze_contract, field_name="freshness"),
        ),
    )
    execution_sha = freshness["execution_sha"]
    missing_field = _derive_missing_field(carrier_report)
    source_fields_present = _derive_source_fields_present(carrier_report)
    block_stage = _derive_block_stage(t10_evidence)
    block_reason = _derive_block_reason(
        carrier_report=carrier_report,
        t10_evidence=t10_evidence,
        missing_field=missing_field,
        source_fields_present=source_fields_present,
    )
    execution_verdict = _extract_execution_verdict(
        ledger_rows, execution_sha=execution_sha
    )
    authority_violation_count = _require_int(
        carrier_report.get("authority_violation_count"),
        field_name="carrier_report.authority_violation_count",
    )
    full_scan_row_count = _require_int(
        carrier_report.get("full_scan_row_count"),
        field_name="carrier_report.full_scan_row_count",
    )
    if (
        _require_int(
            t10_evidence.get("authority_violation_count"),
            field_name="t10_evidence.authority_violation_count",
        )
        != authority_violation_count
    ):
        raise ValueError(
            "authority_violation_count mismatch between carrier report and T10 evidence"
        )
    if (
        _require_int(
            t10_evidence.get("full_scan_row_count"),
            field_name="t10_evidence.full_scan_row_count",
        )
        != full_scan_row_count
    ):
        raise ValueError(
            "full_scan_row_count mismatch between carrier report and T10 evidence"
        )
    blocker = _require_mapping(
        carrier_report.get("blocker"), field_name="carrier_report.blocker"
    )
    carrier_status = _non_empty_string(
        blocker.get("status"),
        field_name="carrier_report.blocker.status",
    )
    if carrier_status != EXPECTED_STATUS:
        raise ValueError(f"unexpected blocker status: {carrier_status}")
    if (
        _non_empty_string(t10_evidence.get("status"), field_name="t10_evidence.status")
        != "blocked"
    ):
        raise ValueError("T10 evidence status must be blocked")
    if execution_sha != _non_empty_string(
        freeze_contract.get("execution_sha"), field_name="freeze_contract.execution_sha"
    ):
        raise ValueError("execution_sha mismatch between freeze contract and freshness")

    blocker_summary = _non_empty_string(
        blocker.get("summary"),
        field_name="carrier_report.blocker.summary",
    )
    first_violation = t10_evidence.get("first_violation")
    if not isinstance(first_violation, dict):
        raise TypeError("t10_evidence.first_violation must be an object")

    source_artifacts = [
        _source_ref(
            repo_root,
            artifact_id="carrier_parity_report",
            authority_role="formal_block_report",
            relative_path=carrier_report_path,
        ),
        _source_ref(
            repo_root,
            artifact_id="task_11_carrier_parity_evidence",
            authority_role="formal_block_evidence",
            relative_path=t10_evidence_path,
        ),
        _source_ref(
            repo_root,
            artifact_id="execution_freeze_contract",
            authority_role="execution_freeze_authority",
            relative_path=freeze_contract_path,
        ),
        _source_ref(
            repo_root,
            artifact_id="b0_e1_e2_run_ledger",
            authority_role="execution_context_ledger",
            relative_path=ledger_path,
        ),
    ]
    non_authoritative_inputs = [
        _build_non_authoritative_input(
            repo_root,
            artifact_id="draft_final_verdict_pack",
            relative_path=draft_final_pack_path,
            reason=(
                "This draft reviewer-grade final pack predates the formal blocked closeout "
                "and must not exercise mainline authority for the frozen execution."
            ),
        )
    ]

    closeout_payload: dict[str, Any] = {
        "schema_version": CLOSEOUT_SCHEMA_VERSION,
        "artifact_kind": CLOSEOUT_ARTIFACT_KIND,
        "generated_at": freshness["timestamp"],
        "execution_root": _repo_relative_path(repo_root, resolved_execution_root),
        "execution_sha": execution_sha,
        "freshness": dict(freshness),
        "status": EXPECTED_STATUS,
        "terminal_state": EXPECTED_TERMINAL_STATE,
        "block_stage": block_stage,
        "block_reason": block_reason,
        "theory_verdict": EXPECTED_THEORY_VERDICT,
        "execution_verdict": execution_verdict,
        "authority_level": EXPECTED_AUTHORITY_LEVEL,
        "gating_eligible": False,
        "requires_successor_execution": True,
        "current_execution_reopen_forbidden": True,
        "authority_violation_count": authority_violation_count,
        "full_scan_row_count": full_scan_row_count,
        "missing_field": missing_field,
        "source_fields_present": list(source_fields_present),
        "blocker_summary": blocker_summary,
        "ledger_context": {
            "row_ids": [row["row_id"] for row in ledger_rows],
            "b0_stage_status": _extract_stage_status(ledger_rows, row_id="B0"),
            "e1_stage_status": _extract_stage_status(ledger_rows, row_id="E1"),
            "e2_stage_status": _extract_stage_status(ledger_rows, row_id="E2"),
        },
        "source_artifacts": source_artifacts,
        "non_authoritative_inputs": non_authoritative_inputs,
        "superseded_inputs": [
            {
                "relative_path": _repo_relative_path(repo_root, draft_final_pack_path),
                "authority_disposition": "non_authoritative_draft_input",
                "reason": (
                    "Formal blocked closeout supersedes this draft-style final pack for the current frozen execution."
                ),
            }
        ],
    }
    closeout_payload["report_signature_sha256"] = _signature_for_payload(
        closeout_payload
    )

    block_reason_payload: dict[str, Any] = {
        "schema_version": BLOCK_REASON_SCHEMA_VERSION,
        "artifact_kind": BLOCK_REASON_ARTIFACT_KIND,
        "generated_at": freshness["timestamp"],
        "execution_sha": execution_sha,
        "freshness": dict(freshness),
        "status": EXPECTED_STATUS,
        "terminal_state": EXPECTED_TERMINAL_STATE,
        "block_stage": block_stage,
        "block_reason": block_reason,
        "authority_level": EXPECTED_AUTHORITY_LEVEL,
        "gating_eligible": False,
        "requires_successor_execution": True,
        "current_execution_reopen_forbidden": True,
        "authority_violation_count": authority_violation_count,
        "full_scan_row_count": full_scan_row_count,
        "missing_field": missing_field,
        "source_fields_present": list(source_fields_present),
        "issue": apple_recap_execution_contract._issue(
            "missing_required_carrier_export",
            f"field_presence_summary.{missing_field}",
            (
                f"Frozen mainline labels retain {', '.join(source_fields_present)} but do not materialize {missing_field}; "
                f"{authority_violation_count}/{full_scan_row_count} rows violate carrier export authority."
            ),
        ),
        "first_violation": dict(first_violation),
        "source_artifacts": list(source_artifacts),
        "non_authoritative_inputs": list(non_authoritative_inputs),
    }
    block_reason_payload["report_signature_sha256"] = _signature_for_payload(
        block_reason_payload
    )

    return {
        "uplift_verdict": closeout_payload,
        "block_reason": block_reason_payload,
        "uplift_verdict_markdown": render_blocked_closeout_markdown(
            closeout_payload=closeout_payload,
            block_reason_payload=block_reason_payload,
        ),
    }


def render_blocked_closeout_markdown(
    *,
    closeout_payload: dict[str, Any],
    block_reason_payload: dict[str, Any],
) -> str:
    freshness = cast(dict[str, str], closeout_payload["freshness"])
    source_artifacts = cast(list[dict[str, Any]], closeout_payload["source_artifacts"])
    non_authoritative_inputs = cast(
        list[dict[str, Any]], closeout_payload["non_authoritative_inputs"]
    )
    issue = cast(dict[str, str], block_reason_payload["issue"])
    lines = [
        "# AppleToPlate frozen execution blocked closeout",
        "",
        "## 结论",
        "",
        f"- 当前 frozen execution 已正式终局关闭：`status={closeout_payload['status']}`，`terminal_state={closeout_payload['terminal_state']}`。",
        f"- closeout authority level：`{closeout_payload['authority_level']}`；当前 execution 不再具备继续承担 mainline authority 的资格。",
        f"- consumer-safe 语义：`gating_eligible={str(closeout_payload['gating_eligible']).lower()}`、`requires_successor_execution={str(closeout_payload['requires_successor_execution']).lower()}`、`current_execution_reopen_forbidden={str(closeout_payload['current_execution_reopen_forbidden']).lower()}`。",
        "",
        "## Canonical freshness / context",
        "",
        f"- execution_sha: `{freshness['execution_sha']}`",
        f"- manifest_hash: `{freshness['manifest_hash']}`",
        f"- checkpoint_id: `{freshness['checkpoint_id']}`",
        f"- seed_bundle_id: `{freshness['seed_bundle_id']}`",
        f"- timestamp: `{freshness['timestamp']}`",
        "",
        "## BLOCK 原因",
        "",
        f"- block_stage: `{closeout_payload['block_stage']}`",
        f"- block_reason: `{closeout_payload['block_reason']}`",
        f"- missing_field: `{closeout_payload['missing_field']}`",
        f"- source_fields_present: `{', '.join(cast(list[str], closeout_payload['source_fields_present']))}`",
        f"- authority_violation_count / full_scan_row_count: `{closeout_payload['authority_violation_count']} / {closeout_payload['full_scan_row_count']}`",
        f"- blocker summary: {closeout_payload['blocker_summary']}",
        f"- issue: `{issue['code']}` at `{issue['field_path']}` — {issue['message']}",
        "",
        "## Execution verdict",
        "",
        f"- theory_verdict: `{closeout_payload['theory_verdict']}`",
        f"- execution_verdict: `{closeout_payload['execution_verdict']}`",
        f"- B0/E1/E2 stage status: `B0={closeout_payload['ledger_context']['b0_stage_status']}`, `E1={closeout_payload['ledger_context']['e1_stage_status']}`, `E2={closeout_payload['ledger_context']['e2_stage_status']}`",
        "",
        "## Source artifacts",
        "",
    ]
    for artifact in source_artifacts:
        lines.append(
            f"- `{artifact['artifact_id']}` ({artifact['authority_role']}): `{artifact['relative_path']}`"
        )
    lines.extend(
        [
            "",
            "## Non-authoritative / superseded inputs",
            "",
        ]
    )
    for artifact in non_authoritative_inputs:
        lines.append(
            f"- `{artifact['artifact_id']}`: `{artifact['relative_path']}` — {artifact['reason']}"
        )
    lines.extend(
        [
            "",
            "本 Markdown 是 blocked closeout surface，不是 reviewer-grade final report，也不得被解释为“当前 execution 仍可继续 pending”。",
        ]
    )
    return "\n".join(lines) + "\n"


def materialize_current_blocked_closeout(
    *,
    repo_root: Path = REPO_ROOT,
    execution_root: Path | str = DEFAULT_EXECUTION_ROOT,
    uplift_verdict_json: Path | str = DEFAULT_UPLIFT_VERDICT_JSON,
    uplift_verdict_md: Path | str = DEFAULT_UPLIFT_VERDICT_MD,
    block_reason_json: Path | str = DEFAULT_BLOCK_REASON_JSON,
) -> dict[str, Any]:
    bundle = build_current_blocked_closeout(
        repo_root=repo_root,
        execution_root=execution_root,
    )
    uplift_json_path = _resolve_repo_path(repo_root, uplift_verdict_json)
    uplift_md_path = _resolve_repo_path(repo_root, uplift_verdict_md)
    block_reason_path = _resolve_repo_path(repo_root, block_reason_json)
    _write_json(uplift_json_path, cast(dict[str, Any], bundle["uplift_verdict"]))
    _write_text(uplift_md_path, cast(str, bundle["uplift_verdict_markdown"]))
    _write_json(block_reason_path, cast(dict[str, Any], bundle["block_reason"]))
    return {
        "uplift_verdict_json": str(uplift_json_path),
        "uplift_verdict_md": str(uplift_md_path),
        "block_reason_json": str(block_reason_path),
        **bundle,
    }


def main() -> int:
    materialize_current_blocked_closeout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
