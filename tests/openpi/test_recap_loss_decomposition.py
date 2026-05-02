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


def _assert_total_loss_is_decomposed(path_payload: dict[str, object]) -> None:
    flow_loss = float(cast(float, path_payload["flow_loss"]))
    discrete_action_ce = float(cast(float, path_payload["discrete_action_ce"]))
    text_ce = float(cast(float, path_payload["text_ce"]))
    total_loss = float(cast(float, path_payload["total_loss"]))
    assert total_loss == round(flow_loss + discrete_action_ce + text_ce, 6)


def test_loss_decomposition_matches_total_loss_for_all_smoke_paths(
    tmp_path: Path,
) -> None:
    payload = run_smoke_forward(
        openpi_tree=_materialized_tree(tmp_path),
        config_name="pi05_libero_recap",
        checkpoint_source="gs://openpi-assets/checkpoints/pi05_libero",
    )

    conditioned = cast(dict[str, object], payload["conditioned"])
    unconditioned = cast(dict[str, object], payload["unconditioned"])
    cfg = cast(dict[str, object], payload["cfg"])

    _assert_total_loss_is_decomposed(conditioned)
    _assert_total_loss_is_decomposed(unconditioned)
    _assert_total_loss_is_decomposed(cfg)

    assert float(cast(float, conditioned["total_loss"])) < float(
        cast(float, unconditioned["total_loss"])
    )
