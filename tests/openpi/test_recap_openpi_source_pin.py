from __future__ import annotations

from pathlib import Path
import sys
from typing import cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.repack_recap_model import (
    load_overlay_materialization_manifest,
    validate_materialized_overlay_tree,
)
from work.openpi.overlays.openpi_recap.materialize import (
    DEFAULT_OVERLAY_ROOT,
    DEFAULT_PINNED_COMMIT,
    KEY_SOURCE_FILE_HASHES,
    OverlayMaterializationError,
    materialize_overlay,
    verify_pinned_source_tree,
)


SOURCE_TREE = REPO_ROOT / "submodules" / "openpi"
OVERLAY_ROOT = DEFAULT_OVERLAY_ROOT
LEGACY_OVERLAY_ROOT_FRAGMENT = "/".join(("upstream_overlay", "openpi_recap"))


def test_materializer_records_source_pin_and_overlay_file_list(tmp_path: Path) -> None:
    output_dir = tmp_path / "openpi_fdc03f5_recap"

    report = materialize_overlay(
        source_tree=SOURCE_TREE,
        pinned_commit=DEFAULT_PINNED_COMMIT,
        overlay_root=OVERLAY_ROOT,
        output_dir=output_dir,
    )

    assert cast(str, report["source_tree_commit"]) == DEFAULT_PINNED_COMMIT
    assert cast(str, report["pinned_commit"]) == DEFAULT_PINNED_COMMIT
    assert cast(str, report["overlay_root"]) == str(DEFAULT_OVERLAY_ROOT.resolve())
    assert LEGACY_OVERLAY_ROOT_FRAGMENT not in cast(str, report["overlay_root"])
    assert str(output_dir.resolve()) == cast(str, report["output_tree"])
    overlay_file_list = cast(list[str], report["overlay_file_list"])
    assert "src/openpi/recap_overlay/training.py" in overlay_file_list

    manifest = load_overlay_materialization_manifest(output_dir)
    assert cast(str, manifest["source_tree_commit"]) == DEFAULT_PINNED_COMMIT
    assert cast(str, manifest["overlay_root"]) == str(DEFAULT_OVERLAY_ROOT.resolve())
    assert len(cast(list[object], manifest["key_source_files"])) == len(
        KEY_SOURCE_FILE_HASHES
    )
    assert sorted(cast(list[str], manifest["overlay_file_list"])) == sorted(
        [
            "src/openpi/policies/policy_config.py",
            "src/openpi/recap_overlay/__init__.py",
            "src/openpi/recap_overlay/config.py",
            "src/openpi/recap_overlay/modeling.py",
            "src/openpi/recap_overlay/training.py",
            "src/openpi/training/config.py",
        ]
    )

    validation = validate_materialized_overlay_tree(output_dir)
    required_backups = cast(list[str], validation["required_backup_files"])
    assert (
        "src/openpi/training/_upstream_openpi_recap_training_config.py"
        in required_backups
    )
    assert (
        output_dir / "src/openpi/training/_upstream_openpi_recap_training_config.py"
    ).is_file()


def test_verify_pinned_source_tree_fails_on_commit_mismatch() -> None:
    with pytest.raises(
        OverlayMaterializationError, match="source tree commit mismatch"
    ):
        verify_pinned_source_tree(
            source_tree=SOURCE_TREE,
            pinned_commit="deadbeef",
        )


def test_verify_pinned_source_tree_fails_on_missing_required_key_file() -> None:
    with pytest.raises(
        OverlayMaterializationError,
        match="missing key source file required by task-11 source pin",
    ):
        verify_pinned_source_tree(
            source_tree=SOURCE_TREE,
            pinned_commit=DEFAULT_PINNED_COMMIT,
            key_source_hashes={"src/openpi/__missing__.py": "abc"},
        )
