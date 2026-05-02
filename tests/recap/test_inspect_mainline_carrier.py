from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator
from work.recap.scripts import inspect_mainline_carrier


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _base_row(
    *,
    sample_id: str = "sample_001",
    prompt_raw: str = "pick up the apple and place it on the plate",
    indicator_I: int = 1,
    carrier_text_v1: str | None = None,
    policy_condition_text: str = "Phase=TRANSPORT; Mode=RECOVERY",
    canonical_policy_condition_text: str = "Phase=TRANSPORT; Mode=RECOVERY",
) -> dict[str, object]:
    indicator_mode = text_indicator.indicator_mode_from_indicator_value(
        indicator_I,
        field_name="indicator_I",
    )
    carrier_text = carrier_text_v1 or text_indicator.build_canonical_text_indicator(
        prompt_raw,
        indicator_mode,
    )
    return {
        "sample_id": sample_id,
        "source_episode_id": f"episode_{sample_id}",
        "prompt_raw": prompt_raw,
        "indicator_I": indicator_I,
        "carrier_text_v1": carrier_text,
        "policy_condition_text": policy_condition_text,
        "canonical_policy_condition_text": canonical_policy_condition_text,
    }


def test_run_inspection_emits_reports_and_canonical_parity(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    output_dir = tmp_path / "inspection"
    rows = [
        _base_row(sample_id="sample_001", indicator_I=1),
        _base_row(sample_id="sample_002", indicator_I=0),
    ]
    _write_jsonl(labels_path, rows)

    report = inspect_mainline_carrier.run_inspection(
        labels_path=labels_path,
        output_dir=output_dir,
    )

    assert report["training_text_field"] == "carrier_text_v1"
    assert report["authority_violation_count"] == 0
    assert report["full_scan_row_count"] == 2
    assert report["carrier_parity_match_count"] == 2

    report_json = _read_json(
        output_dir / inspect_mainline_carrier.PARITY_REPORT_JSON_NAME
    )
    assert report_json["training_text_field"] == "carrier_text_v1"
    assert report_json["authority_violation_count"] == 0

    sample_rows = _read_csv(output_dir / inspect_mainline_carrier.SAMPLE_ROWS_CSV_NAME)
    assert len(sample_rows) == 2
    assert sample_rows[0]["carrier_matches_canonical"] == "True"

    inspection_md = (
        output_dir / inspect_mainline_carrier.INSPECTION_MD_NAME
    ).read_text(encoding="utf-8")
    assert "carrier_text_v1 parity inspection" in inspection_md
    assert "prompt_raw" in inspection_md


def test_run_inspection_fails_closed_on_mixed_carrier_authority_violation(
    tmp_path: Path,
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    output_dir = tmp_path / "inspection"
    rows = [
        _base_row(
            carrier_text_v1="Phase=TRANSPORT; Mode=RECOVERY",
            policy_condition_text="Phase=TRANSPORT; Mode=RECOVERY",
        )
    ]
    _write_jsonl(labels_path, rows)

    with pytest.raises(ValueError, match=r"authority violation\(s\)"):
        inspect_mainline_carrier.run_inspection(
            labels_path=labels_path,
            output_dir=output_dir,
        )

    report_json = _read_json(
        output_dir / inspect_mainline_carrier.PARITY_REPORT_JSON_NAME
    )
    assert report_json["authority_violation_count"] == 1
    violations = report_json["authority_violation_examples"]
    assert isinstance(violations, list)
    assert "build_canonical_text_indicator" in violations[0]["reason"]


def test_policy_condition_text_stays_metadata_only_in_report(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    output_dir = tmp_path / "inspection"
    row = _base_row(
        policy_condition_text="Phase=SEARCH; Mode=NOMINAL",
        canonical_policy_condition_text="Phase=SEARCH; Mode=NOMINAL",
    )
    _write_jsonl(labels_path, [row])

    report = inspect_mainline_carrier.run_inspection(
        labels_path=labels_path,
        output_dir=output_dir,
    )

    assert report["policy_condition_metadata_only"] is True
    sample_rows = _read_csv(output_dir / inspect_mainline_carrier.SAMPLE_ROWS_CSV_NAME)
    assert sample_rows[0]["policy_condition_text"] == "Phase=SEARCH; Mode=NOMINAL"
    assert report["training_text_field"] == "carrier_text_v1"


def test_report_freezes_training_text_field_to_carrier_text_v1(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    output_dir = tmp_path / "inspection"
    _write_jsonl(labels_path, [_base_row()])

    report = inspect_mainline_carrier.run_inspection(
        labels_path=labels_path,
        output_dir=output_dir,
    )
    report_json = _read_json(
        output_dir / inspect_mainline_carrier.PARITY_REPORT_JSON_NAME
    )

    assert (
        report["training_text_field"]
        == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
    assert (
        report_json["training_text_field"]
        == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
