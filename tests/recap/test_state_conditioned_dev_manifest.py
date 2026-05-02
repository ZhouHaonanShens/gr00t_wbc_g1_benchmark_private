from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.scripts import state_conditioned_bucket_a_import
from work.recap.scripts import state_conditioned_bucket_a_sidecar
from work.recap.scripts import state_conditioned_dev_manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_bucket_fixture(
    tmp_path: Path,
    *,
    ready: bool,
    train_seeds: list[int] | None = None,
) -> Path:
    bucket_dir = tmp_path / "bucket_a"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    seeds = (
        list(train_seeds)
        if train_seeds is not None
        else [700 + episode_index for episode_index in range(24)]
    )
    manifest_episodes: list[dict[str, Any]] = []
    for episode_index, seed in enumerate(seeds):
        dataset_dir = tmp_path / f"dataset_{episode_index:03d}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_episodes.append(
            {
                "episode_id": f"fresh_accept_{episode_index:03d}",
                "accepted": True,
                "debug_only": False,
                "fresh_nominal_recollection": True,
                "reused_existing_live_dataset": False,
                "seed": int(seed),
                "source_dataset_dir": str(dataset_dir),
            }
        )

    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "ready": bool(ready),
            "required_distinct_accepted_episode_count": 24,
            "accepted_episode_count": 24,
            "distinct_accepted_episode_count": 24,
        },
    )
    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "required_distinct_episode_count": 24,
            "episodes": manifest_episodes,
        },
    )
    _write_jsonl(
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_SIDECAR_JSON_NAME,
        [{"episode_id": "fresh_accept_000", "t": 0}],
    )
    _write_json(
        bucket_dir
        / state_conditioned_bucket_a_sidecar.BUCKET_A_JOIN_COVERAGE_JSON_NAME,
        {
            "coverage_ratio": 1.0,
            "accepted_episode_count": 24,
        },
    )
    _write_json(
        bucket_dir
        / state_conditioned_bucket_a_sidecar.BUCKET_A_EXPORTER_MANIFEST_JSON_NAME,
        {
            "accepted_episode_count": 24,
            "field_groups": {},
        },
    )
    return bucket_dir


def _fake_baseline_runner(
    *,
    output_dir: Path,
    manifest_path: Path,
    entries: list[dict[str, Any]],
    stratum_counts: dict[str, int],
) -> dict[str, Any]:
    del output_dir
    assert (
        manifest_path.name == state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME
    )
    assert len(entries) == 32
    return {
        "baseline_invocation": {
            "runner": "fake_runner",
            "manifest_path": str(manifest_path),
            "invocation_mode": "test_only",
        },
        "aggregate_metrics": {
            "requested_entries": 32,
            "evaluated_episodes": 32,
            "success_count": 20,
            "success_rate": 20.0 / 32.0,
        },
        "per_stratum": {
            stratum_id: {
                "requested_count": int(count),
                "evaluated_episodes": int(count),
                "success_count": 5,
                "success_rate": 5.0 / float(count),
            }
            for stratum_id, count in stratum_counts.items()
        },
    }


def test_happy_path_writes_fixed_32_entry_dev_manifest_and_scorecard(
    tmp_path: Path,
) -> None:
    bucket_dir = _build_bucket_fixture(tmp_path, ready=True)
    output_dir = tmp_path / "devbench"

    result = state_conditioned_dev_manifest.materialize_state_conditioned_dev_manifest(
        bucket_dir=bucket_dir,
        output_dir=output_dir,
        baseline_runner=_fake_baseline_runner,
    )

    fixed_strata_definition = _read_json(
        output_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME
    )
    baseline_manifest = _read_json(
        output_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME
    )
    scorecard = _read_json(
        output_dir / state_conditioned_dev_manifest.BASELINE_DEV_SCORECARD_JSON_NAME
    )

    assert Path(result["fixed_strata_definition_path"]).is_file()
    assert Path(result["baseline_manifest_path"]).is_file()
    assert Path(result["baseline_dev_scorecard_path"]).is_file()

    assert fixed_strata_definition["paired_seed_count"] == 8
    assert fixed_strata_definition["expected_total_entries"] == 32
    assert [row["stratum_id"] for row in fixed_strata_definition["strata"]] == [
        "nominal",
        "drop_during_transport",
        "failed_grasp_visible",
        "failed_grasp_occluded",
    ]

    entries = baseline_manifest["entries"]
    assert len(entries) == 32
    assert baseline_manifest["counts"]["per_stratum"] == {
        "drop_during_transport": 8,
        "failed_grasp_occluded": 8,
        "failed_grasp_visible": 8,
        "nominal": 8,
    }
    assert baseline_manifest["train_lineage"]["overlap_seed_count"] == 0
    assert len({entry["paired_key"] for entry in entries}) == 32
    assert {
        (entry["paired_identity"]["seed"], entry["paired_identity"]["stratum_id"])
        for entry in entries
    } == {(entry["seed"], entry["stratum_id"]) for entry in entries}
    assert scorecard["manifest_path"] == str(
        output_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME
    )
    assert scorecard["baseline_invocation"]["runner"] == "fake_runner"
    assert scorecard["aggregate_metrics"]["evaluated_episodes"] == 32
    assert scorecard["aggregate_metrics"]["success_count"] == 20
    assert (
        scorecard["counts"]["per_stratum"] == baseline_manifest["counts"]["per_stratum"]
    )


def test_duplicate_paired_key_fails(tmp_path: Path) -> None:
    bucket_dir = _build_bucket_fixture(tmp_path, ready=True)

    with pytest.raises(ValueError, match="duplicate paired key"):
        state_conditioned_dev_manifest.materialize_state_conditioned_dev_manifest(
            bucket_dir=bucket_dir,
            output_dir=tmp_path / "devbench_duplicate",
            baseline_runner=_fake_baseline_runner,
            paired_seeds=[31001, 31001, 31003, 31004, 31005, 31006, 31007, 31008],
        )


def test_empty_stratum_fails(tmp_path: Path) -> None:
    bucket_dir = _build_bucket_fixture(tmp_path, ready=True)

    with pytest.raises(ValueError, match="empty stratum is forbidden"):
        state_conditioned_dev_manifest.materialize_state_conditioned_dev_manifest(
            bucket_dir=bucket_dir,
            output_dir=tmp_path / "devbench_empty",
            baseline_runner=_fake_baseline_runner,
            paired_seeds=[],
        )


def test_train_lineage_overlap_fails(tmp_path: Path) -> None:
    bucket_dir = _build_bucket_fixture(
        tmp_path,
        ready=True,
        train_seeds=[31001 + episode_index for episode_index in range(24)],
    )

    with pytest.raises(ValueError, match="overlaps canonical train lineage"):
        state_conditioned_dev_manifest.materialize_state_conditioned_dev_manifest(
            bucket_dir=bucket_dir,
            output_dir=tmp_path / "devbench_overlap",
            baseline_runner=_fake_baseline_runner,
        )


def test_gate_a_not_ready_fails_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket_dir = _build_bucket_fixture(tmp_path, ready=False)

    exit_code = state_conditioned_dev_manifest.main(
        ["--bucket-dir", str(bucket_dir), "--output-dir", str(tmp_path / "devbench")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "bucket_A_gate_a_ready.json.ready == true" in captured.err
    assert "Traceback" not in captured.err
