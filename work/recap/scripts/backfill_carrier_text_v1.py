#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import inspect_mainline_carrier
from work.recap.scripts.state_conditioned_common import read_json
from work.recap.scripts.state_conditioned_common import validate_existing_file
from work.recap.scripts.state_conditioned_common import validate_output_dir
from work.recap.scripts.state_conditioned_common import write_json


DEFAULT_SOURCE_LABELS = Path(
    "agent/artifacts/recap_datasets/fullsize_relabel_v1/m2_labels/labels.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "agent/artifacts/recap_datasets/fullsize_relabel_v1_carrier_backfill_v1"
)
DEFAULT_DERIVED_LABELS = DEFAULT_OUTPUT_ROOT / "m2_labels" / "labels.jsonl"
DEFAULT_ROW_DIFF_SUMMARY = DEFAULT_OUTPUT_ROOT / "row_diff_summary.json"
DEFAULT_BACKFILL_MANIFEST = DEFAULT_OUTPUT_ROOT / "backfill_manifest.json"
DEFAULT_PROVENANCE_MARKDOWN = DEFAULT_OUTPUT_ROOT / "carrier_backfill_provenance.md"
DEFAULT_RESEARCH_PROBE_DIR = DEFAULT_OUTPUT_ROOT / "research_probe"
DEFAULT_RESEARCH_PROBE_MANIFEST = DEFAULT_RESEARCH_PROBE_DIR / "probe_manifest.json"

DEFAULT_CARRIER_LINEAGE_AUDIT = Path(
    "agent/artifacts/apple_recap_exec/carrier_lineage_audit.json"
)
DEFAULT_UPLIFT_VERDICT = Path("agent/artifacts/apple_recap_exec/uplift_verdict.json")
DEFAULT_FREEZE_CONTRACT = Path(
    "agent/artifacts/apple_recap_exec/execution_freeze_contract.json"
)

DERIVED_LABELS_SCHEMA_VERSION = "carrier_text_v1_backfill_v1"
ROW_DIFF_SUMMARY_SCHEMA_VERSION = "carrier_text_v1_backfill_row_diff_summary_v1"
BACKFILL_MANIFEST_SCHEMA_VERSION = "carrier_text_v1_backfill_manifest_v1"
PROBE_MANIFEST_SCHEMA_VERSION = "carrier_text_v1_backfill_research_probe_v1"
BACKFILL_ARTIFACT_KIND = "carrier_text_v1_backfill_research_variant"
ROW_DIFF_SUMMARY_ARTIFACT_KIND = "carrier_text_v1_backfill_row_diff_summary"
PROBE_MANIFEST_ARTIFACT_KIND = "carrier_text_v1_backfill_research_probe"
RESEARCH_AUTHORITY_LEVEL = "research"
RESEARCH_VARIANT_ID = "carrier_backfill_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a research-only derived labels variant that deterministically "
            "backfills carrier_text_v1 from prompt_raw + indicator_I without mutating "
            "the frozen source dataset."
        )
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        default=DEFAULT_SOURCE_LABELS,
        help="Frozen source labels JSONL to read without mutation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Derived research-only output root.",
    )
    parser.add_argument(
        "--carrier-lineage-audit",
        type=Path,
        default=DEFAULT_CARRIER_LINEAGE_AUDIT,
        help="Carrier lineage audit JSON used to anchor frozen execution freshness.",
    )
    parser.add_argument(
        "--uplift-verdict",
        type=Path,
        default=DEFAULT_UPLIFT_VERDICT,
        help="Blocked closeout verdict JSON used to anchor frozen execution freshness.",
    )
    parser.add_argument(
        "--freeze-contract",
        type=Path,
        default=DEFAULT_FREEZE_CONTRACT,
        help="Execution freeze contract JSON used to cross-check frozen execution freshness.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=inspect_mainline_carrier.DEFAULT_SAMPLE_LIMIT,
        help="Sample limit forwarded to inspect_mainline_carrier for the research probe.",
    )
    return parser


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _signature_for_payload(payload: Mapping[str, Any]) -> str:
    signature_basis = {
        str(key): value
        for key, value in dict(payload).items()
        if key != "report_signature_sha256"
    }
    canonical_bytes = json.dumps(
        signature_basis,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _write_lines(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
    tmp.replace(path)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_freshness(
    payload: Mapping[str, Any], *, field_name: str
) -> dict[str, str]:
    source = payload if not field_name else payload.get(field_name)
    if not isinstance(source, Mapping):
        raise TypeError(f"{field_name or 'payload'} must be an object")
    fields = (
        "checkpoint_id",
        "execution_sha",
        "manifest_hash",
        "seed_bundle_id",
        "timestamp",
    )
    out: dict[str, str] = {}
    for field in fields:
        raw = source.get(field)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                f"{field_name or 'payload'}.{field} must be a non-empty string"
            )
        out[field] = raw.strip()
    return out


def _assert_same_freshness(*bundles: tuple[str, dict[str, str]]) -> dict[str, str]:
    if not bundles:
        raise ValueError("at least one freshness bundle is required")
    anchor_name, anchor_bundle = bundles[0]
    for name, bundle in bundles[1:]:
        if bundle != anchor_bundle:
            raise ValueError(
                f"freshness mismatch between {anchor_name} and {name}: {anchor_bundle!r} != {bundle!r}"
            )
    return dict(anchor_bundle)


def _has_non_empty_text(raw: Any) -> bool:
    if raw is None:
        return False
    return bool(str(raw).strip())


def _build_row_locator(row: Mapping[str, Any], *, line_number: int) -> dict[str, Any]:
    return {
        "line_number": int(line_number),
        "episode_id": row.get("episode_id"),
        "t": row.get("t"),
        "sample_id": row.get("sample_id"),
    }


def _canonical_carrier_from_row(row: Mapping[str, Any]) -> str:
    prompt_text = text_indicator.require_prompt_raw(
        row.get(text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD),
        field_name=text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
    )
    indicator_mode = text_indicator.indicator_mode_from_indicator_value(
        row.get("indicator_I"),
        field_name="indicator_I",
    )
    return text_indicator.build_canonical_text_indicator(prompt_text, indicator_mode)


def build_backfilled_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_dict = dict(row)
    carrier_raw = row_dict.get(text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD)
    if _has_non_empty_text(carrier_raw):
        metadata: dict[str, Any] = {
            "changed": False,
            "precondition_failed": False,
            "existing_carrier_preserved": True,
            "existing_carrier_is_canonical": False,
            "precondition_failure_reason": None,
        }
        try:
            expected_carrier = _canonical_carrier_from_row(row_dict)
        except (TypeError, ValueError):
            return row_dict, metadata
        metadata["existing_carrier_is_canonical"] = str(carrier_raw) == expected_carrier
        return row_dict, metadata

    try:
        canonical_carrier = _canonical_carrier_from_row(row_dict)
    except (TypeError, ValueError) as exc:
        return row_dict, {
            "changed": False,
            "precondition_failed": True,
            "existing_carrier_preserved": False,
            "existing_carrier_is_canonical": False,
            "precondition_failure_reason": str(exc),
        }

    updated = dict(row_dict)
    updated[text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD] = canonical_carrier
    return updated, {
        "changed": True,
        "precondition_failed": False,
        "existing_carrier_preserved": False,
        "existing_carrier_is_canonical": False,
        "precondition_failure_reason": None,
    }


def _json_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _build_row_diff_summary(
    *,
    repo_root: Path,
    source_labels_path: Path,
    derived_labels_path: Path,
    total_row_count: int,
    changed_rows: list[dict[str, Any]],
    precondition_failures: list[dict[str, Any]],
    existing_canonical_preserved_count: int,
    existing_noncanonical_or_unverifiable_preserved_count: int,
    generated_at: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": ROW_DIFF_SUMMARY_SCHEMA_VERSION,
        "artifact_kind": ROW_DIFF_SUMMARY_ARTIFACT_KIND,
        "generated_at": generated_at,
        "source_labels_path": _repo_relative_path(repo_root, source_labels_path),
        "derived_labels_path": _repo_relative_path(repo_root, derived_labels_path),
        "full_scan_row_count": int(total_row_count),
        "changed_row_count": int(len(changed_rows)),
        "untouched_row_count": int(total_row_count - len(changed_rows)),
        "precondition_failed_row_count": int(len(precondition_failures)),
        "existing_canonical_carrier_preserved_count": int(
            existing_canonical_preserved_count
        ),
        "existing_noncanonical_or_unverifiable_carrier_preserved_count": int(
            existing_noncanonical_or_unverifiable_preserved_count
        ),
        "rows_still_missing_carrier_text_v1_count": int(len(precondition_failures)),
        "changed_row_examples": changed_rows[:10],
        "precondition_failure_examples": precondition_failures[:10],
    }
    summary["report_signature_sha256"] = _signature_for_payload(summary)
    return summary


def _build_provenance_markdown(
    *,
    manifest: Mapping[str, Any],
    row_diff_summary: Mapping[str, Any],
    probe_manifest: Mapping[str, Any],
) -> str:
    freshness = cast(Mapping[str, Any], manifest["frozen_execution_context"])
    freshness_bundle = cast(Mapping[str, Any], freshness["freshness"])
    research_probe = cast(Mapping[str, Any], manifest["research_probe"])
    lines = [
        "# carrier_text_v1 research-only deterministic backfill provenance",
        "",
        "## 结论",
        "",
        "- 本目录是 **research-only derived variant**，不是 canonical authority export。",
        "- 本目录 **不得** 解锁 mainline authority，也 **不得** 作为 gating 通过依据。",
        "- 原始冻结数据集保持只读；本变体只是把缺失的 `carrier_text_v1` 以确定性规则物化到派生 labels 中。",
        "",
        "## 固定回填规则",
        "",
        "```text",
        "if carrier_text_v1 is missing and prompt_raw exists and indicator_I exists:",
        "    carrier_text_v1 = build_canonical_text_indicator(prompt_raw, indicator_I)",
        "```",
        "",
        "## 研究边界",
        "",
        f"- authority_level: `{manifest['authority_level']}`",
        f"- gating_eligible: `{str(manifest['gating_eligible']).lower()}`",
        f"- not_canonical_authority: `{str(manifest['not_canonical_authority']).lower()}`",
        f"- derived_from: `{manifest['derived_from']}`",
        f"- derived_labels_path: `{manifest['derived_labels_path']}`",
        f"- research_probe.output_dir: `{research_probe['output_dir']}`",
        "- research probe 仅用于观察派生 labels 在 parity 检查下的表现；即使 probe 无 authority violation，也不能覆盖当前 blocked closeout。",
        "",
        "## 冻结执行 freshness 上下文",
        "",
        f"- execution_sha: `{freshness_bundle['execution_sha']}`",
        f"- manifest_hash: `{freshness_bundle['manifest_hash']}`",
        f"- checkpoint_id: `{freshness_bundle['checkpoint_id']}`",
        f"- seed_bundle_id: `{freshness_bundle['seed_bundle_id']}`",
        f"- timestamp: `{freshness_bundle['timestamp']}`",
        f"- current_execution_reopen_forbidden: `{str(freshness['current_execution_reopen_forbidden']).lower()}`",
        f"- blocked_closeout_status: `{freshness['blocked_closeout_status']}`",
        f"- blocked_closeout_stage: `{freshness['blocked_closeout_stage']}`",
        f"- carrier_lineage_first_failing_stage: `{freshness['carrier_lineage_first_failing_stage']}`",
        "",
        "## 行级摘要",
        "",
        f"- full_scan_row_count: {row_diff_summary['full_scan_row_count']}",
        f"- changed_row_count: {row_diff_summary['changed_row_count']}",
        f"- untouched_row_count: {row_diff_summary['untouched_row_count']}",
        f"- precondition_failed_row_count: {row_diff_summary['precondition_failed_row_count']}",
        f"- rows_still_missing_carrier_text_v1_count: {row_diff_summary['rows_still_missing_carrier_text_v1_count']}",
        "",
        "## 关键说明",
        "",
        "- 本变体只修补 research probe 需要的 `carrier_text_v1` 表面，不触碰 `work/recap/labeler.py`、exporter/source live logic 或 successor authority 路径。",
        "- `research_probe/` 子目录下的所有产物都必须与主线 authority root 分离理解；它们不是 `apple_recap_exec`，也不是 `apple_recap_exec_successor`。",
        f"- research probe authority_violation_count: {probe_manifest['authority_violation_count']}",
    ]
    return "\n".join(lines)


def _build_probe_manifest(
    *,
    repo_root: Path,
    generated_at: str,
    derived_labels_path: Path,
    research_probe_dir: Path,
    probe_report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = cast(Mapping[str, Any], probe_report.get("artifacts", {}))
    manifest: dict[str, Any] = {
        "schema_version": PROBE_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": PROBE_MANIFEST_ARTIFACT_KIND,
        "generated_at": generated_at,
        "authority_level": RESEARCH_AUTHORITY_LEVEL,
        "gating_eligible": False,
        "not_canonical_authority": True,
        "mainline_unlock_forbidden": True,
        "probe_script": "work/recap/scripts/inspect_mainline_carrier.py",
        "allow_authority_violations": True,
        "derived_labels_path": _repo_relative_path(repo_root, derived_labels_path),
        "output_dir": _repo_relative_path(repo_root, research_probe_dir),
        "full_scan_row_count": int(probe_report["full_scan_row_count"]),
        "authority_violation_count": int(probe_report["authority_violation_count"]),
        "output_artifacts": {
            str(key): _repo_relative_path(repo_root, Path(str(value)))
            for key, value in artifacts.items()
        },
    }
    manifest["report_signature_sha256"] = _signature_for_payload(manifest)
    return manifest


def _build_backfill_manifest(
    *,
    repo_root: Path,
    generated_at: str,
    source_labels_path: Path,
    derived_labels_path: Path,
    output_root: Path,
    research_probe_dir: Path,
    row_diff_summary_path: Path,
    provenance_markdown_path: Path,
    probe_manifest_path: Path,
    source_labels_sha256_before: str,
    source_labels_sha256_after: str,
    derived_labels_sha256: str,
    row_diff_summary: Mapping[str, Any],
    probe_manifest: Mapping[str, Any],
    frozen_execution_context: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": BACKFILL_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": BACKFILL_ARTIFACT_KIND,
        "generated_at": generated_at,
        "variant_id": RESEARCH_VARIANT_ID,
        "authority_level": RESEARCH_AUTHORITY_LEVEL,
        "gating_eligible": False,
        "not_canonical_authority": True,
        "mainline_unlock_forbidden": True,
        "research_only": True,
        "derived_from": _repo_relative_path(repo_root, source_labels_path),
        "variant_root": _repo_relative_path(repo_root, output_root),
        "derived_labels_path": _repo_relative_path(repo_root, derived_labels_path),
        "row_diff_summary_path": _repo_relative_path(repo_root, row_diff_summary_path),
        "provenance_markdown_path": _repo_relative_path(
            repo_root, provenance_markdown_path
        ),
        "source_dataset_mutated": False,
        "source_labels_sha256_before": source_labels_sha256_before,
        "source_labels_sha256_after": source_labels_sha256_after,
        "derived_labels_sha256": derived_labels_sha256,
        "deterministic_backfill_rule": {
            "target_field": text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD,
            "source_fields": [
                text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
                "indicator_I",
            ],
            "rule": (
                "if carrier_text_v1 is missing and prompt_raw exists and indicator_I exists, "
                "materialize carrier_text_v1 with build_canonical_text_indicator(prompt_raw, indicator_I)"
            ),
            "preserve_existing_carrier_text_v1": True,
            "mutates_frozen_source": False,
            "derivation_kind": "deterministic_research_only_backfill",
        },
        "row_diff_counts": {
            "full_scan_row_count": int(row_diff_summary["full_scan_row_count"]),
            "changed_row_count": int(row_diff_summary["changed_row_count"]),
            "untouched_row_count": int(row_diff_summary["untouched_row_count"]),
            "precondition_failed_row_count": int(
                row_diff_summary["precondition_failed_row_count"]
            ),
        },
        "research_probe": {
            "authority_level": RESEARCH_AUTHORITY_LEVEL,
            "gating_eligible": False,
            "not_canonical_authority": True,
            "output_dir": _repo_relative_path(repo_root, research_probe_dir),
            "probe_manifest_path": _repo_relative_path(repo_root, probe_manifest_path),
            "full_scan_row_count": int(probe_manifest["full_scan_row_count"]),
            "authority_violation_count": int(
                probe_manifest["authority_violation_count"]
            ),
            "allow_authority_violations": True,
            "probe_script": probe_manifest["probe_script"],
            "output_artifacts": probe_manifest["output_artifacts"],
        },
        "frozen_execution_context": dict(frozen_execution_context),
        "source_artifacts": [dict(item) for item in source_artifacts],
    }
    manifest["report_signature_sha256"] = _signature_for_payload(manifest)
    return manifest


def materialize_carrier_backfill_variant(
    *,
    repo_root: Path = REPO_ROOT,
    source_labels: Path | str = DEFAULT_SOURCE_LABELS,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    carrier_lineage_audit_json: Path | str = DEFAULT_CARRIER_LINEAGE_AUDIT,
    uplift_verdict_json: Path | str = DEFAULT_UPLIFT_VERDICT,
    freeze_contract_json: Path | str = DEFAULT_FREEZE_CONTRACT,
    sample_limit: int = inspect_mainline_carrier.DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    resolved_source_labels = validate_existing_file(
        _resolve_repo_path(repo_root, source_labels),
        arg_name="source_labels",
    )
    resolved_output_root = validate_output_dir(
        _resolve_repo_path(repo_root, output_root)
    )
    derived_labels_path = resolved_output_root / "m2_labels" / "labels.jsonl"
    row_diff_summary_path = resolved_output_root / DEFAULT_ROW_DIFF_SUMMARY.name
    manifest_path = resolved_output_root / DEFAULT_BACKFILL_MANIFEST.name
    provenance_markdown_path = resolved_output_root / DEFAULT_PROVENANCE_MARKDOWN.name
    research_probe_dir = resolved_output_root / DEFAULT_RESEARCH_PROBE_DIR.name
    research_probe_manifest_path = (
        research_probe_dir / DEFAULT_RESEARCH_PROBE_MANIFEST.name
    )

    carrier_lineage_audit_path = validate_existing_file(
        _resolve_repo_path(repo_root, carrier_lineage_audit_json),
        arg_name="carrier_lineage_audit_json",
    )
    uplift_verdict_path = validate_existing_file(
        _resolve_repo_path(repo_root, uplift_verdict_json),
        arg_name="uplift_verdict_json",
    )
    freeze_contract_path = validate_existing_file(
        _resolve_repo_path(repo_root, freeze_contract_json),
        arg_name="freeze_contract_json",
    )

    generated_at = _now_iso()
    source_labels_sha256_before = _sha256_file(resolved_source_labels)

    carrier_lineage_audit = read_json(carrier_lineage_audit_path)
    uplift_verdict = read_json(uplift_verdict_path)
    freeze_contract = read_json(freeze_contract_path)
    freshness = _assert_same_freshness(
        (
            "carrier_lineage_audit",
            _extract_freshness(carrier_lineage_audit, field_name="freshness"),
        ),
        ("uplift_verdict", _extract_freshness(uplift_verdict, field_name="freshness")),
        (
            "execution_freeze_contract",
            _extract_freshness(freeze_contract, field_name="freshness"),
        ),
    )

    output_lines: list[str] = []
    changed_rows: list[dict[str, Any]] = []
    precondition_failures: list[dict[str, Any]] = []
    total_row_count = 0
    existing_canonical_preserved_count = 0
    existing_noncanonical_or_unverifiable_preserved_count = 0

    with resolved_source_labels.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.rstrip("\n")
            if not stripped.strip():
                raise ValueError(
                    f"source_labels contains an empty line at line_number={line_number}"
                )
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"source_labels line {line_number} must decode to an object, got {type(payload).__name__}"
                )
            row = cast(dict[str, Any], payload)
            total_row_count += 1
            updated_row, metadata = build_backfilled_row(row)

            if bool(metadata["changed"]):
                changed_rows.append(
                    {
                        **_build_row_locator(row, line_number=line_number),
                        "backfilled_carrier_text_v1": updated_row[
                            text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
                        ],
                    }
                )
                output_lines.append(_json_line(updated_row))
                continue

            output_lines.append(stripped)
            if bool(metadata["precondition_failed"]):
                precondition_failures.append(
                    {
                        **_build_row_locator(row, line_number=line_number),
                        "reason": metadata["precondition_failure_reason"],
                    }
                )
                continue

            if bool(metadata["existing_carrier_is_canonical"]):
                existing_canonical_preserved_count += 1
            else:
                existing_noncanonical_or_unverifiable_preserved_count += 1

    _write_lines(derived_labels_path, output_lines)
    source_labels_sha256_after = _sha256_file(resolved_source_labels)
    if source_labels_sha256_before != source_labels_sha256_after:
        raise RuntimeError(
            "source_labels changed during research-only backfill; frozen source must remain untouched"
        )
    derived_labels_sha256 = _sha256_file(derived_labels_path)

    row_diff_summary = _build_row_diff_summary(
        repo_root=repo_root,
        source_labels_path=resolved_source_labels,
        derived_labels_path=derived_labels_path,
        total_row_count=total_row_count,
        changed_rows=changed_rows,
        precondition_failures=precondition_failures,
        existing_canonical_preserved_count=existing_canonical_preserved_count,
        existing_noncanonical_or_unverifiable_preserved_count=(
            existing_noncanonical_or_unverifiable_preserved_count
        ),
        generated_at=generated_at,
    )
    write_json(row_diff_summary_path, row_diff_summary)

    probe_report = inspect_mainline_carrier.run_inspection(
        labels_path=derived_labels_path,
        output_dir=research_probe_dir,
        sample_limit=sample_limit,
        fail_on_authority_violation=False,
    )
    probe_manifest = _build_probe_manifest(
        repo_root=repo_root,
        generated_at=generated_at,
        derived_labels_path=derived_labels_path,
        research_probe_dir=research_probe_dir,
        probe_report=probe_report,
    )
    write_json(research_probe_manifest_path, probe_manifest)

    frozen_execution_context = {
        "execution_root": "agent/artifacts/apple_recap_exec",
        "freshness": dict(freshness),
        "carrier_lineage_first_failing_stage": carrier_lineage_audit.get(
            "first_failing_stage"
        ),
        "blocked_closeout_status": uplift_verdict.get("status"),
        "blocked_closeout_stage": uplift_verdict.get("block_stage"),
        "blocked_closeout_reason": uplift_verdict.get("block_reason"),
        "current_execution_reopen_forbidden": uplift_verdict.get(
            "current_execution_reopen_forbidden"
        ),
    }
    source_artifacts = [
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="frozen_mainline_labels",
            authority_role="frozen_mainline_dataset",
            relative_path=resolved_source_labels,
        ),
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="carrier_lineage_audit",
            authority_role="lineage_audit_context",
            relative_path=carrier_lineage_audit_path,
        ),
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="uplift_verdict",
            authority_role="blocked_closeout_context",
            relative_path=uplift_verdict_path,
        ),
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="execution_freeze_contract",
            authority_role="execution_freeze_authority",
            relative_path=freeze_contract_path,
        ),
    ]
    backfill_manifest = _build_backfill_manifest(
        repo_root=repo_root,
        generated_at=generated_at,
        source_labels_path=resolved_source_labels,
        derived_labels_path=derived_labels_path,
        output_root=resolved_output_root,
        research_probe_dir=research_probe_dir,
        row_diff_summary_path=row_diff_summary_path,
        provenance_markdown_path=provenance_markdown_path,
        probe_manifest_path=research_probe_manifest_path,
        source_labels_sha256_before=source_labels_sha256_before,
        source_labels_sha256_after=source_labels_sha256_after,
        derived_labels_sha256=derived_labels_sha256,
        row_diff_summary=row_diff_summary,
        probe_manifest=probe_manifest,
        frozen_execution_context=frozen_execution_context,
        source_artifacts=source_artifacts,
    )
    write_json(manifest_path, backfill_manifest)

    provenance_markdown = _build_provenance_markdown(
        manifest=backfill_manifest,
        row_diff_summary=row_diff_summary,
        probe_manifest=probe_manifest,
    )
    _write_text(provenance_markdown_path, provenance_markdown)

    return {
        "derived_labels_path": str(derived_labels_path),
        "row_diff_summary": row_diff_summary,
        "row_diff_summary_path": str(row_diff_summary_path),
        "backfill_manifest": backfill_manifest,
        "backfill_manifest_path": str(manifest_path),
        "carrier_backfill_provenance_markdown": provenance_markdown,
        "carrier_backfill_provenance_path": str(provenance_markdown_path),
        "research_probe_manifest": probe_manifest,
        "research_probe_manifest_path": str(research_probe_manifest_path),
        "research_probe_report": probe_report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = materialize_carrier_backfill_variant(
        source_labels=args.source_labels,
        output_root=args.output_root,
        carrier_lineage_audit_json=args.carrier_lineage_audit,
        uplift_verdict_json=args.uplift_verdict,
        freeze_contract_json=args.freeze_contract,
        sample_limit=args.sample_limit,
    )
    json.dump(result["backfill_manifest"], sys.stdout, ensure_ascii=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
