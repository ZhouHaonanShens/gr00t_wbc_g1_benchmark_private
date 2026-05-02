#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


DEFAULT_LABELS = Path(
    "agent/artifacts/state_conditioned_materialization/training/state_conditioned_sft_labels.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/state_conditioned_materialization/carrier_inspection"
)

SAMPLE_ROWS_CSV_NAME = "carrier_sample_rows.csv"
INSPECTION_MD_NAME = "carrier_inspection.md"
PARITY_REPORT_JSON_NAME = "carrier_parity_report.json"
SCHEMA_VERSION = "carrier_parity_report_v1"
DEFAULT_SAMPLE_LIMIT = 12

FIELD_NAMES_TO_SUMMARIZE: tuple[str, ...] = (
    "prompt_raw",
    "indicator_I",
    "carrier_text_v1",
    "policy_condition_text",
    "canonical_policy_condition_text",
)

SAMPLE_CSV_FIELDNAMES: tuple[str, ...] = (
    "row_number",
    "sample_id",
    "source_episode_id",
    "prompt_raw",
    "indicator_I",
    "indicator_mode",
    "carrier_text_v1",
    "expected_carrier_text_v1",
    "carrier_matches_canonical",
    "policy_condition_text",
    "canonical_policy_condition_text",
    "authority_violation_reason",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator
from work.recap.scripts.state_conditioned_common import read_jsonl_dicts as _read_jsonl
from work.recap.scripts.state_conditioned_common import (
    validate_existing_file as _validate_existing_file,
)
from work.recap.scripts.state_conditioned_common import (
    validate_output_dir as _validate_output_dir,
)
from work.recap.scripts.state_conditioned_common import write_json as _write_json
from work.recap.state_conditioned import build_training_set


TRAINING_TEXT_FIELD = build_training_set.MAINLINE_TRAINING_TEXT_FIELD
if TRAINING_TEXT_FIELD != text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD:
    raise RuntimeError(
        "Mainline carrier inspection expects build_training_set.MAINLINE_TRAINING_TEXT_FIELD "
        f"to stay on {text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD!r}, "
        f"got {TRAINING_TEXT_FIELD!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect export-side mainline carrier parity for carrier_text_v1 and emit "
            "CSV/Markdown/JSON inspection artifacts."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Training-label artifact to inspect (.jsonl or .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for carrier inspection outputs.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help="How many representative rows to emit to carrier_sample_rows.csv.",
    )
    parser.add_argument(
        "--allow-authority-violations",
        action="store_true",
        help="Write reports without raising when canonical carrier authority violations are found.",
    )
    return parser


def _validate_sample_limit(sample_limit: int) -> int:
    if not isinstance(sample_limit, int) or sample_limit <= 0:
        raise ValueError(f"sample_limit must be a positive int, got {sample_limit!r}")
    return int(sample_limit)


def _has_non_empty_text(value: object) -> bool:
    if value is None:
        return False
    return bool(str(value).strip())


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {str(key): value for key, value in row.items() if key is not None}
            )
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    resolved = _validate_existing_file(path, arg_name="labels")
    suffix = resolved.suffix.lower()
    if suffix == ".jsonl":
        return list(_read_jsonl(resolved))
    if suffix == ".csv":
        return _read_csv_rows(resolved)
    raise ValueError(f"labels must be a .jsonl or .csv file, got {resolved}")


def _example_values(rows: Sequence[Mapping[str, Any]], *, field_name: str) -> list[str]:
    examples: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(field_name)
        if not _has_non_empty_text(value):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        examples.append(text)
        if len(examples) >= 3:
            break
    return examples


def summarize_field_presence(
    rows: Sequence[Mapping[str, Any]], *, field_name: str
) -> dict[str, Any]:
    present_count = sum(
        1 for row in rows if field_name in row and row.get(field_name) is not None
    )
    non_empty_count = sum(1 for row in rows if _has_non_empty_text(row.get(field_name)))
    missing_count = max(len(rows) - present_count, 0)
    empty_count = max(present_count - non_empty_count, 0)
    return {
        "present_count": present_count,
        "missing_count": missing_count,
        "empty_count": empty_count,
        "non_empty_count": non_empty_count,
        "example_values": _example_values(rows, field_name=field_name),
    }


def inspect_row(row: Mapping[str, Any], *, row_number: int) -> dict[str, Any]:
    prompt_raw = row.get("prompt_raw")
    indicator_raw = row.get("indicator_I")
    carrier_raw = row.get(TRAINING_TEXT_FIELD)
    policy_condition_text = row.get("policy_condition_text")
    canonical_policy_condition_text = row.get("canonical_policy_condition_text")

    authority_reasons: list[str] = []
    prompt_text: str | None = None
    indicator_mode: str | None = None
    carrier_text: str | None = None
    expected_carrier_text = ""

    try:
        prompt_text = text_indicator.require_prompt_raw(
            prompt_raw,
            field_name="prompt_raw",
        )
    except (TypeError, ValueError) as exc:
        authority_reasons.append(str(exc))

    try:
        indicator_mode = text_indicator.indicator_mode_from_indicator_value(
            indicator_raw,
            field_name="indicator_I",
        )
    except (TypeError, ValueError) as exc:
        authority_reasons.append(str(exc))

    try:
        carrier_text = text_indicator.require_prompt_raw(
            carrier_raw,
            field_name=TRAINING_TEXT_FIELD,
        )
    except (TypeError, ValueError) as exc:
        authority_reasons.append(str(exc))

    if prompt_text is not None and indicator_mode is not None:
        expected_carrier_text = text_indicator.build_canonical_text_indicator(
            prompt_text,
            indicator_mode,
        )

    carrier_matches_canonical = (
        bool(expected_carrier_text)
        and carrier_text is not None
        and carrier_text == expected_carrier_text
    )
    if (
        expected_carrier_text
        and carrier_text is not None
        and not carrier_matches_canonical
    ):
        authority_reasons.append(
            "carrier_text_v1 must match build_canonical_text_indicator(prompt_raw, indicator_I)"
        )

    return {
        "row_number": row_number,
        "sample_id": _stringify(row.get("sample_id")),
        "source_episode_id": _stringify(row.get("source_episode_id")),
        "prompt_raw": _stringify(prompt_raw),
        "indicator_I": _stringify(indicator_raw),
        "indicator_mode": indicator_mode or "",
        "carrier_text_v1": _stringify(carrier_raw),
        "expected_carrier_text_v1": expected_carrier_text,
        "carrier_matches_canonical": carrier_matches_canonical,
        "policy_condition_text": _stringify(policy_condition_text),
        "canonical_policy_condition_text": _stringify(canonical_policy_condition_text),
        "authority_violation_reason": " | ".join(authority_reasons),
    }


def inspect_rows(
    rows: Sequence[Mapping[str, Any]], *, sample_limit: int = DEFAULT_SAMPLE_LIMIT
) -> dict[str, Any]:
    normalized_sample_limit = _validate_sample_limit(sample_limit)
    sample_rows: list[dict[str, Any]] = []
    authority_violations: list[dict[str, Any]] = []
    indicator_mode_counts: Counter[str] = Counter()
    carrier_parity_match_count = 0

    for row_index, row in enumerate(rows, start=1):
        inspected = inspect_row(row, row_number=row_index)
        indicator_mode = str(inspected["indicator_mode"])
        if indicator_mode:
            indicator_mode_counts[indicator_mode] += 1
        if bool(inspected["carrier_matches_canonical"]):
            carrier_parity_match_count += 1
        if len(sample_rows) < normalized_sample_limit:
            sample_rows.append(inspected)
        if str(inspected["authority_violation_reason"]):
            authority_violations.append(
                {
                    "row_number": int(inspected["row_number"]),
                    "sample_id": str(inspected["sample_id"]),
                    "reason": str(inspected["authority_violation_reason"]),
                }
            )

    full_scan_row_count = len(rows)
    field_presence_summary = {
        field_name: summarize_field_presence(rows, field_name=field_name)
        for field_name in FIELD_NAMES_TO_SUMMARIZE
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "training_text_field": TRAINING_TEXT_FIELD,
        "authority_violation_count": len(authority_violations),
        "policy_condition_metadata_only": True,
        "full_scan_row_count": full_scan_row_count,
        "sample_row_count": len(sample_rows),
        "carrier_parity_match_count": carrier_parity_match_count,
        "carrier_parity_mismatch_count": full_scan_row_count
        - carrier_parity_match_count,
        "indicator_mode_counts": dict(sorted(indicator_mode_counts.items())),
        "field_presence_summary": field_presence_summary,
        "authority_violation_examples": authority_violations[:25],
        "sample_rows_artifact": SAMPLE_ROWS_CSV_NAME,
        "inspection_markdown_artifact": INSPECTION_MD_NAME,
        "parity_report_artifact": PARITY_REPORT_JSON_NAME,
    }
    return {
        "report": report,
        "sample_rows": sample_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in SAMPLE_CSV_FIELDNAMES}
            )
    tmp.replace(path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    tmp.replace(path)
    return path


def build_inspection_markdown(
    report: Mapping[str, Any], *, sample_rows: Sequence[Mapping[str, Any]]
) -> str:
    lines: list[str] = [
        "# carrier_text_v1 parity inspection",
        "",
        f"- training_text_field: `{report['training_text_field']}`",
        f"- full_scan_row_count: {report['full_scan_row_count']}",
        f"- authority_violation_count: {report['authority_violation_count']}",
        f"- policy_condition_metadata_only: {str(report['policy_condition_metadata_only']).lower()}",
        "",
        "## 字段汇总",
        "",
        "| field | present | non-empty | missing | empty | examples |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    field_presence_summary = report.get("field_presence_summary")
    if not isinstance(field_presence_summary, Mapping):
        raise TypeError("report.field_presence_summary must be an object")
    for field_name in FIELD_NAMES_TO_SUMMARIZE:
        summary = field_presence_summary.get(field_name)
        if not isinstance(summary, Mapping):
            raise TypeError(
                f"report.field_presence_summary[{field_name!r}] must be an object"
            )
        examples = summary.get("example_values")
        example_text = (
            "<br>".join(str(item) for item in examples)
            if isinstance(examples, list)
            else ""
        )
        lines.append(
            "| "
            + field_name
            + f" | {summary.get('present_count', 0)}"
            + f" | {summary.get('non_empty_count', 0)}"
            + f" | {summary.get('missing_count', 0)}"
            + f" | {summary.get('empty_count', 0)}"
            + f" | {example_text} |"
        )

    lines.extend(["", "## indicator_I canonical modes", ""])
    indicator_mode_counts = report.get("indicator_mode_counts")
    if isinstance(indicator_mode_counts, Mapping) and indicator_mode_counts:
        lines.extend(
            [
                "| indicator_mode | count |",
                "| --- | ---: |",
            ]
        )
        for indicator_mode, count in indicator_mode_counts.items():
            lines.append(f"| {indicator_mode} | {count} |")
    else:
        lines.append("无可解析的 indicator_I 行。")

    lines.extend(["", "## 样例行", ""])
    if sample_rows:
        lines.extend(
            [
                "| row | sample_id | indicator_mode | carrier_matches_canonical | policy_condition_text |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in sample_rows:
            lines.append(
                "| "
                + str(row.get("row_number", ""))
                + " | "
                + str(row.get("sample_id", ""))
                + " | "
                + str(row.get("indicator_mode", ""))
                + " | "
                + str(row.get("carrier_matches_canonical", ""))
                + " | "
                + str(row.get("policy_condition_text", ""))
                + " |"
            )
    else:
        lines.append("未找到任何可写入样例的行。")

    lines.extend(["", "## Authority violations", ""])
    authority_violation_examples = report.get("authority_violation_examples")
    if isinstance(authority_violation_examples, list) and authority_violation_examples:
        for violation in authority_violation_examples:
            if not isinstance(violation, Mapping):
                continue
            lines.append(
                "- row "
                + str(violation.get("row_number", "?"))
                + ": "
                + str(violation.get("reason", "unknown violation"))
            )
    else:
        lines.append("- none")

    return "\n".join(lines)


def run_inspection(
    *,
    labels_path: Path,
    output_dir: Path,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    fail_on_authority_violation: bool = True,
) -> dict[str, Any]:
    rows = read_rows(labels_path)
    resolved_output_dir = _validate_output_dir(output_dir)
    inspection = inspect_rows(rows, sample_limit=sample_limit)
    report = dict(inspection["report"])
    sample_rows = list(inspection["sample_rows"])

    sample_csv_path = resolved_output_dir / SAMPLE_ROWS_CSV_NAME
    inspection_md_path = resolved_output_dir / INSPECTION_MD_NAME
    parity_report_path = resolved_output_dir / PARITY_REPORT_JSON_NAME

    _write_csv(sample_csv_path, sample_rows)
    _write_text(
        inspection_md_path,
        build_inspection_markdown(report, sample_rows=sample_rows),
    )
    report["artifacts"] = {
        "carrier_sample_rows_csv": str(sample_csv_path),
        "carrier_inspection_md": str(inspection_md_path),
        "carrier_parity_report_json": str(parity_report_path),
    }
    _write_json(parity_report_path, report)

    if fail_on_authority_violation and int(report["authority_violation_count"]) > 0:
        authority_violation_examples = report.get("authority_violation_examples")
        first_reason = "unknown authority violation"
        if (
            isinstance(authority_violation_examples, list)
            and authority_violation_examples
        ):
            first = authority_violation_examples[0]
            if isinstance(first, Mapping):
                first_reason = str(first.get("reason", first_reason))
        raise ValueError(
            "carrier inspection found "
            + str(report["authority_violation_count"])
            + " authority violation(s); first: "
            + first_reason
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_inspection(
        labels_path=args.labels,
        output_dir=args.output_dir,
        sample_limit=args.sample_limit,
        fail_on_authority_violation=not args.allow_authority_violations,
    )
    json.dump(report, sys.stdout, ensure_ascii=True, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
