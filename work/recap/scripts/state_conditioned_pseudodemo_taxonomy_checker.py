#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import re
import sys
from typing import NoReturn, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_TAXONOMY = Path(
    "agent/artifacts/state_conditioned_materialization/audit/pseudodemo_label_taxonomy.md"
)
DEFAULT_AUDIT_PACK = Path(
    "agent/artifacts/state_conditioned_materialization/audit/pseudodemo_label_audit_pack.md"
)

PASS_SENTINEL = "PSEUDODEMO_TAXONOMY_CHECK_PASS"
FAIL_SENTINEL = "PSEUDODEMO_TAXONOMY_CHECK_FAIL"
ALLOWED_DECISION_CLASSES = {"valid", "ambiguous", "invalid"}
CLAIM_REQUIRED_FIELDS = (
    "claim_id",
    "decision_class",
    "taxonomy_subtype",
    "sample_refs",
    "evidence_refs",
    "status",
    "reviewer",
    "review_notes",
)
PRIMARY_EVIDENCE_PREFIXES = ("snapshot:", "manifest:", "label:")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate evidence-backed taxonomy claims for the state-conditioned "
            "pseudodemo audit pack."
        )
    )
    _ = parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
        help="Markdown taxonomy file that contains claim blocks.",
    )
    _ = parser.add_argument(
        "--audit-pack",
        type=Path,
        default=DEFAULT_AUDIT_PACK,
        help="Markdown audit pack file that defines the audit-sample registry.",
    )
    return parser


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    print(FAIL_SENTINEL, file=sys.stderr)
    raise SystemExit(1)


def _read_text(path: Path, *, label: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        _fail(f"missing required {label}: {resolved}")
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"failed to read {label} {resolved}: {exc}")


def _parse_registry(audit_pack_text: str) -> set[str]:
    registry = {
        match.group(1)
        for match in re.finditer(
            r"^\|\s+(audit-sample-\d{3})\s+\|", audit_pack_text, re.MULTILINE
        )
    }
    if not registry:
        _fail("audit pack is missing the audit-sample registry table")
    return registry


def _parse_claim_blocks(taxonomy_text: str) -> list[dict[str, str]]:
    blocks = cast(
        list[str], re.findall(r"```text\s*(.*?)```", taxonomy_text, flags=re.DOTALL)
    )
    claims: list[dict[str, str]] = []
    for block in blocks:
        raw_lines = [
            line.strip() for line in block.strip().splitlines() if line.strip()
        ]
        if not any(line.startswith("claim_id:") for line in raw_lines):
            continue
        claim: dict[str, str] = {}
        for line in raw_lines:
            if ":" not in line:
                _fail(
                    f"claim block contains malformed line without ':' separator: {line!r}"
                )
            key, value = line.split(":", 1)
            claim[key.strip()] = value.strip()
        claims.append(claim)
    if not claims:
        _fail("taxonomy markdown does not contain any claim blocks")
    return claims


def _parse_bracket_list(raw_value: str, *, field_name: str, claim_id: str) -> list[str]:
    value = raw_value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        _fail(f"{claim_id}: {field_name} must use [item1, item2] list syntax")
    inner = value[1:-1].strip()
    if not inner:
        return []
    items = [item.strip().strip("'\"") for item in inner.split(",")]
    normalized = [item for item in items if item]
    if len(normalized) != len(items):
        _fail(f"{claim_id}: {field_name} contains an empty list item")
    return normalized


def _has_any_prefix(values: Iterable[str], prefixes: tuple[str, ...]) -> bool:
    return any(any(value.startswith(prefix) for prefix in prefixes) for value in values)


def _validate_claim(
    claim: dict[str, str],
    *,
    index: int,
    registry: set[str],
) -> list[str]:
    missing_fields = [field for field in CLAIM_REQUIRED_FIELDS if field not in claim]
    if missing_fields:
        return [
            f"claim #{index} is missing required fields: {', '.join(missing_fields)}"
        ]

    errors: list[str] = []
    claim_id = claim["claim_id"].strip()
    if not claim_id:
        return [f"claim #{index} has an empty claim_id"]

    decision_class = claim["decision_class"].strip()
    if decision_class not in ALLOWED_DECISION_CLASSES:
        errors.append(
            f"{claim_id}: decision_class must be one of {sorted(ALLOWED_DECISION_CLASSES)}, got {decision_class!r}"
        )

    taxonomy_subtype = claim["taxonomy_subtype"].strip()
    if not taxonomy_subtype.startswith(f"{decision_class}."):
        errors.append(
            f"{claim_id}: taxonomy_subtype {taxonomy_subtype!r} does not match decision_class {decision_class!r}"
        )

    sample_refs = _parse_bracket_list(
        claim["sample_refs"], field_name="sample_refs", claim_id=claim_id
    )
    if not sample_refs:
        errors.append(
            f"{claim_id}: sample_refs must be non-empty in a passing taxonomy file"
        )
    unknown_refs = [
        sample_ref for sample_ref in sample_refs if sample_ref not in registry
    ]
    if unknown_refs:
        errors.append(
            f"{claim_id}: sample_refs not found in audit registry: {', '.join(unknown_refs)}"
        )

    evidence_refs = _parse_bracket_list(
        claim["evidence_refs"], field_name="evidence_refs", claim_id=claim_id
    )
    if not evidence_refs:
        errors.append(f"{claim_id}: evidence_refs must be non-empty")
    if not _has_any_prefix(evidence_refs, ("audit_pack:",)):
        errors.append(
            f"{claim_id}: evidence_refs must include at least one audit_pack:* item"
        )
    if not _has_any_prefix(evidence_refs, PRIMARY_EVIDENCE_PREFIXES):
        errors.append(
            f"{claim_id}: evidence_refs must include at least one snapshot:*, manifest:* or label:* item"
        )
    return errors


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    audit_pack_path = cast(Path, args.audit_pack)
    taxonomy_path = cast(Path, args.taxonomy)

    audit_pack_text = _read_text(audit_pack_path, label="audit pack")
    taxonomy_text = _read_text(taxonomy_path, label="taxonomy markdown")

    registry = _parse_registry(audit_pack_text)
    claims = _parse_claim_blocks(taxonomy_text)
    errors: list[str] = []
    for index, claim in enumerate(claims, start=1):
        errors.extend(_validate_claim(claim, index=index, registry=registry))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(FAIL_SENTINEL, file=sys.stderr)
        return 1

    print(
        f"validated {len(claims)} claim(s) against {len(registry)} audit registry entries"
    )
    print(PASS_SENTINEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
