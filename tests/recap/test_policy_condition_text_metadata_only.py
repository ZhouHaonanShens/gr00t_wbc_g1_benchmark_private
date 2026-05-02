from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.data import contract_mapping
from work.recap.lerobot_export import dataset_export


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_policy_condition_text_cannot_satisfy_mainline_carrier_authority(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_json(
        dataset_dir / "meta/info.json",
        {
            "features": {
                "action": {"dtype": "float32", "shape": [32], "names": None},
                "observation.state": {
                    "dtype": "float32",
                    "shape": [45],
                    "names": None,
                },
                "annotation.human.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None,
                },
            },
            "task_text_field": "policy_condition_text",
            "carrier_route": "policy_condition_text",
            "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
            "recap_export.dual_task_text": False,
        },
    )
    _write_json(
        dataset_dir / "meta/modality.json",
        {"video": {"ego_view": {"original_key": "observation.images.ego_view"}}},
    )

    with pytest.raises(
        ValueError,
        match=(
            "Phase 1 dataset handoff requires task_text_field=carrier_text_v1; "
            "policy_condition_text remains metadata-only"
        ),
    ):
        _ = contract_mapping.build_phase1_dataset_mapping_spec(dataset_dir)
