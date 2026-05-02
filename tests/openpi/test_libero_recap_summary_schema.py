from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.protocol import build_frozen_comparison_manifest
from work.openpi.recap.summary import (
    SUMMARY_FIELDS,
    build_paired_summary,
    validate_summary_fields,
)


def _summary(
    variant: str, checkpoint_source: str, checkpoint_dir: str
) -> dict[str, object]:
    manifest = build_frozen_comparison_manifest(
        suite="libero_spatial",
        task_ids="0,1",
        seed_manifest="7,17",
        num_trials_per_task=2,
    )
    return {
        "variant": variant,
        "checkpoint_source": checkpoint_source,
        "checkpoint_dir": checkpoint_dir,
        "suite": manifest.suite,
        "task_ids": [int(value) for value in manifest.task_ids],
        "seed_manifest": [int(value) for value in manifest.seed_manifest],
        "num_trials_per_task": int(manifest.num_trials_per_task),
        "episode_count": int(manifest.episode_count),
        "success_rate": 0.5,
        "failure_count": 4,
        "deviation_notes": [],
    }


def test_summary_fields_are_frozen_for_task7() -> None:
    assert SUMMARY_FIELDS == (
        "variant",
        "checkpoint_source",
        "checkpoint_dir",
        "suite",
        "task_ids",
        "seed_manifest",
        "num_trials_per_task",
        "episode_count",
        "success_rate",
        "failure_count",
        "deviation_notes",
    )


def test_validate_summary_fields_requires_deviation_notes_even_if_empty() -> None:
    summary = _summary("recap_only", "repo_local", "/tmp/recap/best")
    validated = validate_summary_fields(summary)

    assert validated["variant"] == "recap_only"
    assert validated["task_ids"] == [0, 1]
    assert validated["seed_manifest"] == [7, 17]
    assert validated["episode_count"] == 8
    assert validated["deviation_notes"] == []


def test_build_paired_summary_keeps_stock_and_recap_only_rows_on_same_manifest() -> (
    None
):
    stock = _summary(
        "stock",
        "upstream_openpi_default_or_explicit_cli",
        str(REPO_ROOT / "agent/artifacts/openpi_libero_native/summary.json"),
    )
    recap = _summary(
        "recap_only",
        "repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
        str(
            REPO_ROOT
            / "agent/artifacts/checkpoints/openpi_libero_variants/recap_only_v1/best"
        ),
    )
    recap["success_rate"] = 0.25
    recap["failure_count"] = 6
    recap["deviation_notes"] = ["offline proxy"]

    paired = build_paired_summary(stock_summary=stock, recap_summary=recap)

    assert paired["schema_version"] == "openpi_libero_recap_paired_summary_v1"
    assert paired["summary_fields"] == list(SUMMARY_FIELDS)
    rows = paired["paired_summary"]
    assert isinstance(rows, list)
    assert [row["variant"] for row in rows] == ["stock", "recap_only"]
    assert rows[0]["task_ids"] == rows[1]["task_ids"] == [0, 1]
    assert rows[0]["seed_manifest"] == rows[1]["seed_manifest"] == [7, 17]
    assert rows[1]["deviation_notes"] == ["offline proxy"]
