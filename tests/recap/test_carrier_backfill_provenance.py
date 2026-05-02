from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator


backfill_carrier_text_v1 = importlib.import_module(
    "work.recap.scripts.backfill_carrier_text_v1"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"expected JSON object row at {path}, got {type(payload).__name__}"
                )
            rows.append(dict(payload))
    return rows


def _freshness_bundle() -> dict[str, str]:
    return {
        "checkpoint_id": "phase_a_tooling_frozen",
        "execution_sha": "29d7396b51d5f3db1204f59df2e376ebd7e64ef9",
        "manifest_hash": "3de2e772d69955993ae7acd2528e5046b4fb764228aa8d60e1e78e773553e401",
        "seed_bundle_id": "20000:20009",
        "timestamp": "2026-04-12T04:27:21+00:00",
    }


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _write_minimal_context(repo_root: Path) -> tuple[Path, Path, Path]:
    lineage_path = (
        repo_root / "agent/artifacts/apple_recap_exec/carrier_lineage_audit.json"
    )
    uplift_path = repo_root / "agent/artifacts/apple_recap_exec/uplift_verdict.json"
    freeze_path = (
        repo_root / "agent/artifacts/apple_recap_exec/execution_freeze_contract.json"
    )
    freshness = _freshness_bundle()
    _write_json(
        lineage_path,
        {
            "schema_version": "carrier_lineage_audit_v1",
            "artifact_kind": "carrier_lineage_audit",
            "freshness": freshness,
            "first_failing_stage": "label_materialization",
        },
    )
    _write_json(
        uplift_path,
        {
            "schema_version": "apple_recap_blocked_closeout_v1",
            "artifact_kind": "apple_recap_blocked_closeout",
            "freshness": freshness,
            "status": "BLOCK",
            "block_stage": "T10_formal_carrier_parity",
            "block_reason": "carrier_export_authority_violation",
            "current_execution_reopen_forbidden": True,
            "gating_eligible": False,
        },
    )
    _write_json(
        freeze_path,
        {
            "schema_version": "apple_recap_execution_freeze_contract_v1",
            "artifact_kind": "apple_recap_execution_freeze_contract",
            "freshness": freshness,
            "execution_sha": freshness["execution_sha"],
        },
    )
    return lineage_path, uplift_path, freeze_path


def test_materialize_carrier_backfill_variant_keeps_source_read_only_and_marks_research_probe(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_labels = (
        repo_root
        / "agent/artifacts/recap_datasets/fullsize_relabel_v1/m2_labels/labels.jsonl"
    )
    row_needs_backfill = {
        "episode_id": "episode_001",
        "t": 0,
        "prompt_raw": "pick up the apple and place it on the plate",
        "indicator_I": 0,
    }
    canonical_existing = text_indicator.build_canonical_text_indicator(
        "walk left with the apple", "positive"
    )
    row_existing_canonical = {
        "episode_id": "episode_002",
        "t": 1,
        "prompt_raw": "walk left with the apple",
        "indicator_I": 1,
        "carrier_text_v1": canonical_existing,
    }
    row_missing_precondition = {
        "episode_id": "episode_003",
        "t": 2,
        "prompt_raw": "place the apple on the plate",
    }
    original_lines = [
        json.dumps(row_needs_backfill, ensure_ascii=True, separators=(",", ":")),
        json.dumps(row_existing_canonical, ensure_ascii=True, separators=(",", ":")),
        json.dumps(row_missing_precondition, ensure_ascii=True, separators=(",", ":")),
    ]
    _write_jsonl_lines(source_labels, original_lines)
    original_source_text = source_labels.read_text(encoding="utf-8")

    lineage_path, uplift_path, freeze_path = _write_minimal_context(repo_root)
    output_root = (
        repo_root
        / "agent/artifacts/recap_datasets/fullsize_relabel_v1_carrier_backfill_v1"
    )

    result = backfill_carrier_text_v1.materialize_carrier_backfill_variant(
        repo_root=repo_root,
        source_labels=source_labels,
        output_root=output_root,
        carrier_lineage_audit_json=lineage_path,
        uplift_verdict_json=uplift_path,
        freeze_contract_json=freeze_path,
        sample_limit=5,
    )

    assert source_labels.read_text(encoding="utf-8") == original_source_text

    derived_labels_path = output_root / "m2_labels/labels.jsonl"
    derived_lines = derived_labels_path.read_text(encoding="utf-8").splitlines()
    derived_rows = _read_jsonl(derived_labels_path)
    assert derived_rows[0][
        "carrier_text_v1"
    ] == text_indicator.build_canonical_text_indicator(
        row_needs_backfill["prompt_raw"], "negative"
    )
    assert derived_rows[1]["carrier_text_v1"] == canonical_existing
    assert derived_lines[1] == original_lines[1]
    assert "carrier_text_v1" not in derived_rows[2]
    assert derived_lines[2] == original_lines[2]

    manifest = _read_json(Path(str(result["backfill_manifest_path"])))
    assert manifest["authority_level"] == "research"
    assert manifest["gating_eligible"] is False
    assert manifest["not_canonical_authority"] is True
    assert manifest["derived_from"] == _repo_relative(
        source_labels, repo_root=repo_root
    )
    assert (
        manifest["source_labels_sha256_before"]
        == manifest["source_labels_sha256_after"]
    )

    row_diff_summary = _read_json(Path(str(result["row_diff_summary_path"])))
    assert row_diff_summary["changed_row_count"] == 1
    assert row_diff_summary["untouched_row_count"] == 2
    assert row_diff_summary["precondition_failed_row_count"] == 1

    probe_manifest = _read_json(Path(str(result["research_probe_manifest_path"])))
    assert probe_manifest["authority_level"] == "research"
    assert probe_manifest["gating_eligible"] is False
    assert probe_manifest["not_canonical_authority"] is True
    assert probe_manifest["output_dir"].endswith("research_probe")
    for artifact_path in probe_manifest["output_artifacts"].values():
        assert "/research_probe/" in f"/{artifact_path}"
        assert "apple_recap_exec_successor" not in str(artifact_path)

    provenance_markdown = (output_root / "carrier_backfill_provenance.md").read_text(
        encoding="utf-8"
    )
    assert "research-only derived variant" in provenance_markdown
    assert "不得" in provenance_markdown
    assert "不能" in provenance_markdown or "不得" in provenance_markdown


def test_repository_backfill_artifacts_stay_research_only_and_non_gating() -> None:
    output_root = (
        REPO_ROOT
        / "agent/artifacts/recap_datasets/fullsize_relabel_v1_carrier_backfill_v1"
    )
    manifest = _read_json(output_root / "backfill_manifest.json")
    row_diff_summary = _read_json(output_root / "row_diff_summary.json")
    probe_manifest = _read_json(output_root / "research_probe/probe_manifest.json")
    provenance_markdown = (output_root / "carrier_backfill_provenance.md").read_text(
        encoding="utf-8"
    )

    assert manifest["authority_level"] == "research"
    assert manifest["gating_eligible"] is False
    assert manifest["not_canonical_authority"] is True
    assert manifest["derived_from"] == (
        "agent/artifacts/recap_datasets/fullsize_relabel_v1/m2_labels/labels.jsonl"
    )
    assert manifest["research_probe"]["authority_level"] == "research"
    assert manifest["research_probe"]["gating_eligible"] is False
    assert manifest["research_probe"]["probe_manifest_path"] == (
        "agent/artifacts/recap_datasets/fullsize_relabel_v1_carrier_backfill_v1/"
        "research_probe/probe_manifest.json"
    )
    assert manifest["research_probe"]["output_dir"] == (
        "agent/artifacts/recap_datasets/fullsize_relabel_v1_carrier_backfill_v1/"
        "research_probe"
    )

    assert row_diff_summary["full_scan_row_count"] == 61246
    assert row_diff_summary["changed_row_count"] == 61246
    assert row_diff_summary["untouched_row_count"] == 0
    assert row_diff_summary["precondition_failed_row_count"] == 0
    assert row_diff_summary["rows_still_missing_carrier_text_v1_count"] == 0

    assert probe_manifest["authority_level"] == "research"
    assert probe_manifest["gating_eligible"] is False
    assert probe_manifest["not_canonical_authority"] is True
    assert probe_manifest["authority_violation_count"] == 0
    for artifact_path in probe_manifest["output_artifacts"].values():
        assert str(artifact_path).startswith(
            "agent/artifacts/recap_datasets/fullsize_relabel_v1_carrier_backfill_v1/research_probe/"
        )

    assert "research-only derived variant" in provenance_markdown
    assert "不得" in provenance_markdown
    assert "不能覆盖当前 blocked closeout" in provenance_markdown
