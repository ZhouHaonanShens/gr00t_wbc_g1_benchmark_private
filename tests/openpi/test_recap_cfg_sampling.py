from __future__ import annotations

from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.repack_recap_model import run_smoke_forward
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


def test_cfg_sampling_respects_guidance_scale_boundaries(tmp_path: Path) -> None:
    tree = _materialized_tree(tmp_path)
    scale_zero = run_smoke_forward(
        openpi_tree=tree,
        config_name="pi05_libero_recap",
        checkpoint_source="gs://openpi-assets/checkpoints/pi05_libero",
        cfg_scale=0.0,
    )
    scale_one = run_smoke_forward(
        openpi_tree=tree,
        config_name="pi05_libero_recap",
        checkpoint_source="gs://openpi-assets/checkpoints/pi05_libero",
        cfg_scale=1.0,
    )

    scale_zero_cfg = cast(dict[str, object], scale_zero["cfg"])
    scale_zero_unconditioned = cast(dict[str, object], scale_zero["unconditioned"])
    scale_one_cfg = cast(dict[str, object], scale_one["cfg"])
    scale_one_conditioned = cast(dict[str, object], scale_one["conditioned"])

    assert float(cast(float, scale_zero_cfg["total_loss"])) == float(
        cast(float, scale_zero_unconditioned["total_loss"])
    )
    assert float(cast(float, scale_one_cfg["total_loss"])) == float(
        cast(float, scale_one_conditioned["total_loss"])
    )


def test_cfg_sampling_can_go_beyond_conditioned_path_for_positive_guidance(
    tmp_path: Path,
) -> None:
    payload = run_smoke_forward(
        openpi_tree=_materialized_tree(tmp_path),
        config_name="pi05_libero_recap",
        checkpoint_source="gs://openpi-assets/checkpoints/pi05_libero",
        cfg_scale=1.5,
    )

    cfg = cast(dict[str, object], payload["cfg"])
    conditioned = cast(dict[str, object], payload["conditioned"])

    assert float(cast(float, cfg["cfg_scale"])) == 1.5
    assert float(cast(float, cfg["total_loss"])) < float(
        cast(float, conditioned["total_loss"])
    )
