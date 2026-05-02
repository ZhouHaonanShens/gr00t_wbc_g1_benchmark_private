from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.control_gate import (  # noqa: E402
    CONTROL_GATE_SCHEMA_VERSION,
    RELABEL_CONTROL_OUTPUT_DIR,
    SOURCE_EQUIVALENCE_REPORT_NAME,
    build_source_equivalence_report,
    materialize_source_equivalence_report,
)
from work.openpi.recap.protocol import RECAP_ONLY_VARIANT  # noqa: E402


def _mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"expected mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _materialization_report(
    *, final_status: str, output_dataset_dir: Path
) -> dict[str, object]:
    return {
        "final_status": final_status,
        "output_dataset_dir": str(output_dataset_dir),
        "source_dataset_dir": str(
            REPO_ROOT
            / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d"
        ),
        "route_id": "official_native_8d_recap_relabels_v1",
        "selected_episode_count": 1693,
        "selected_frame_count": 273465,
    }


def _train_manifest(source_dir: Path, checkpoint_dir: Path) -> dict[str, object]:
    return {
        "variant": RECAP_ONLY_VARIANT,
        "checkpoint_dir": str(checkpoint_dir),
        "train_source": {
            "dataset_dir": str(source_dir),
            "dataset_name": source_dir.name,
        },
    }


def _checkpoint_provenance(source_dir: Path, checkpoint_dir: Path) -> dict[str, object]:
    return {
        "variant": RECAP_ONLY_VARIANT,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_source": "repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
        "variant_derivation": {
            "source_dataset_dir": str(source_dir),
            "source_dataset_name": source_dir.name,
        },
    }


def _paired_summary(checkpoint_dir: Path) -> dict[str, object]:
    return {
        "schema_version": "openpi_libero_recap_paired_summary_v1",
        "summary_fields": [],
        "paired_summary": [
            {
                "variant": "stock",
                "checkpoint_dir": str(
                    REPO_ROOT / "agent/artifacts/openpi_libero_native/summary.json"
                ),
                "success_rate": 1.0,
                "failure_count": 0,
            },
            {
                "variant": RECAP_ONLY_VARIANT,
                "checkpoint_dir": str(checkpoint_dir),
                "success_rate": 0.25,
                "failure_count": 6,
            },
        ],
    }


def test_build_source_equivalence_report_blocks_when_relabel_source_not_materialized() -> (
    None
):
    checkpoint_dir = (
        REPO_ROOT
        / "agent/artifacts/checkpoints/openpi_libero_variants/recap_only_v1/best"
    )
    existing_source_dir = (
        REPO_ROOT / "agent/artifacts/lerobot_datasets/recap_reward_approved_v1"
    )
    relabel_output_dir = RELABEL_CONTROL_OUTPUT_DIR

    report = build_source_equivalence_report(
        materialization_report=_materialization_report(
            final_status="blocked",
            output_dataset_dir=relabel_output_dir,
        ),
        train_manifest=_train_manifest(existing_source_dir, checkpoint_dir),
        checkpoint_provenance=_checkpoint_provenance(
            existing_source_dir, checkpoint_dir
        ),
        paired_summary=_paired_summary(checkpoint_dir),
        rerun_output_dir=relabel_output_dir,
    )

    assert report["schema_version"] == CONTROL_GATE_SCHEMA_VERSION
    assert report["status"] == "blocked"
    assert report["reuse_existing_control"] is False
    assert report["rerun_control"] is True
    assert report["rerun_possible_now"] is False
    assert _mapping(report["source_equivalence"])["strongly_proven"] is False
    blocker = _mapping(report["blocker"])
    assert blocker["code"] == "missing_materialized_relabel8d_source"
    assert blocker["authority_final_status"] == "blocked"


def test_build_source_equivalence_report_requires_rerun_when_relabel_source_exists_but_sources_differ(
    tmp_path: Path,
) -> None:
    checkpoint_dir = (
        REPO_ROOT
        / "agent/artifacts/checkpoints/openpi_libero_variants/recap_only_v1/best"
    )
    existing_source_dir = (
        REPO_ROOT / "agent/artifacts/lerobot_datasets/recap_reward_approved_v1"
    )
    relabel_output_dir = tmp_path / "recap_only_relabel8d_v1"

    report = build_source_equivalence_report(
        materialization_report=_materialization_report(
            final_status="materialized",
            output_dataset_dir=relabel_output_dir,
        ),
        train_manifest=_train_manifest(existing_source_dir, checkpoint_dir),
        checkpoint_provenance=_checkpoint_provenance(
            existing_source_dir, checkpoint_dir
        ),
        paired_summary=_paired_summary(checkpoint_dir),
        rerun_output_dir=relabel_output_dir,
    )

    assert report["status"] == "rerun_required"
    assert report["reuse_existing_control"] is False
    assert report["rerun_control"] is True
    assert report["rerun_possible_now"] is True
    assert "blocker" not in report
    assert _mapping(report["rerun_target"])["rerun_artifact_exists"] is False


def test_build_source_equivalence_report_reuses_existing_control_only_on_exact_relabel_source_match(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "recap_only_relabel8d_v1" / "best"
    relabel_output_dir = tmp_path / "recap_only_relabel8d_v1"

    report = build_source_equivalence_report(
        materialization_report=_materialization_report(
            final_status="materialized",
            output_dataset_dir=relabel_output_dir,
        ),
        train_manifest=_train_manifest(relabel_output_dir, checkpoint_dir),
        checkpoint_provenance=_checkpoint_provenance(
            relabel_output_dir, checkpoint_dir
        ),
        paired_summary=_paired_summary(checkpoint_dir),
        rerun_output_dir=relabel_output_dir,
    )

    assert report["status"] == "reuse_existing_control"
    assert report["reuse_existing_control"] is True
    assert report["rerun_control"] is False
    assert report["rerun_possible_now"] is False
    assert _mapping(report["source_equivalence"])["strongly_proven"] is True
    assert _mapping(report["rerun_target"])["rerun_artifact_exists"] is False


def test_materialize_source_equivalence_report_locks_current_repo_to_rerun_required_gate(
    tmp_path: Path,
) -> None:
    report_path = materialize_source_equivalence_report(
        materialization_report_path=REPO_ROOT
        / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1/materialization_report.json",
        train_manifest_path=REPO_ROOT
        / "agent/artifacts/checkpoints/openpi_libero_variants/recap_only_v1/train_manifest.json",
        checkpoint_provenance_path=REPO_ROOT
        / "agent/artifacts/checkpoints/openpi_libero_variants/recap_only_v1/checkpoint_provenance.json",
        paired_summary_path=REPO_ROOT
        / "agent/artifacts/openpi_libero_recap_eval/recap_only_best/paired_summary.json",
        output_dir=tmp_path / "recap_only_relabel8d_v1",
    )

    assert report_path.name == SOURCE_EQUIVALENCE_REPORT_NAME
    payload = _mapping(json.loads(report_path.read_text(encoding="utf-8")))
    assert payload["status"] == "rerun_required"
    assert payload["reuse_existing_control"] is False
    assert payload["rerun_control"] is True
    assert payload["rerun_possible_now"] is True
    assert "blocker" not in payload
    rerun_target = _mapping(payload["rerun_target"])
    assert rerun_target["rerun_artifact_exists"] is False
    assert rerun_target["dataset_dir"] == str(
        REPO_ROOT
        / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1"
    )
