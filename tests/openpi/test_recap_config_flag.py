from __future__ import annotations

from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.repack_recap_model import (
    build_config_summary,
    inspect_training_config_overlay_source,
    load_overlay_module,
)
from work.openpi.overlays.openpi_recap.materialize import (
    DEFAULT_OVERLAY_ROOT,
    DEFAULT_PINNED_COMMIT,
    materialize_overlay,
)


SOURCE_TREE = REPO_ROOT / "submodules" / "openpi"
OVERLAY_ROOT = DEFAULT_OVERLAY_ROOT


def _materialized_tree(tmp_path: Path) -> Path:
    output_dir = tmp_path / "openpi_fdc03f5_recap"
    if not output_dir.exists():
        _ = materialize_overlay(
            source_tree=SOURCE_TREE,
            pinned_commit=DEFAULT_PINNED_COMMIT,
            overlay_root=OVERLAY_ROOT,
            output_dir=output_dir,
        )
    return output_dir


def test_pi05_libero_recap_flag_is_registered_in_overlay_source(tmp_path: Path) -> None:
    tree = _materialized_tree(tmp_path)
    source_summary = inspect_training_config_overlay_source(tree)
    config_summary = build_config_summary(tree, "pi05_libero_recap")
    overlay_config_module = load_overlay_module(tree, "openpi.recap_overlay.config")
    policy_metadata = cast(
        dict[str, object], overlay_config_module.build_recap_policy_metadata()
    )

    assert cast(bool, source_summary["has_cli"]) is True
    assert cast(bool, source_summary["has_get_config"]) is True

    metadata_keys = set(policy_metadata.keys())
    assert {
        "recap_enabled",
        "recap_base_config",
        "recap_cfg_supported",
        "recap_conditioning_position",
        "recap_loss_keys",
    }.issubset(metadata_keys)
    assert cast(str, config_summary["name"]) == "pi05_libero_recap"
    assert cast(str, config_summary["base_config_name"]) == "pi05_libero"
    assert cast(bool, config_summary["supports_cfg"]) is True
    assert cast(bool, config_summary["condition_on_advantage_text"]) is True
    assert cast(str, source_summary["path"]).endswith("src/openpi/training/config.py")
