from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_package_workflow_exposes_expected_export_entrypoints() -> None:
    package_module = importlib.import_module("work.recap.lerobot_export.workflow")

    assert callable(package_module.main)
    assert hasattr(package_module, "LeRobotVideoExportWorkflow")
    assert hasattr(package_module, "RecapExportLeRobotWithVideoScriptApp")


def test_package_dataset_export_exposes_expected_contract_surface() -> None:
    package_module = importlib.import_module("work.recap.lerobot_export.dataset_export")

    assert callable(package_module.export_recap_to_lerobot_v2)
    assert callable(package_module.resolve_lerobot_v2_dataset_dir)
    assert isinstance(package_module.STATE_KEY_ORDER_LOCK, list)


def test_package_video_export_exposes_expected_video_helpers() -> None:
    package_module = importlib.import_module("work.recap.lerobot_export.video_export")

    assert callable(package_module.export_recap_to_lerobot_v2_with_video)
    assert callable(package_module.attach_videos_to_existing_lerobot_dataset)
    assert callable(package_module.resolve_episode_video_path)
