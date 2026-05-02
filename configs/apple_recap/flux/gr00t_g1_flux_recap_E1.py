from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from configs.apple_recap.flux.gr00t_g1_flux_recap_base import PINNED_FLUXVLA_COMMIT
from configs.apple_recap.flux.gr00t_g1_flux_recap_base import build_variant_config
from work.recap.models import flux_recap_vla


CONFIG = build_variant_config(
    variant_id="gr00t_g1_flux_recap_E1",
    train_relative_path=(
        "agent/artifacts/apple_recap_flux_graft/models/gr00t_g1_flux_recap_E1/train_model_ref.json"
    ),
    inference_relative_path=(
        "agent/artifacts/apple_recap_flux_graft/models/gr00t_g1_flux_recap_E1/inference_model_ref.json"
    ),
)


def build_config_surface(*, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    return flux_recap_vla.build_flux_recap_model_surface(
        repo_root=repo_root,
        variant_id=str(CONFIG["variant_id"]),
        pinned_fluxvla_commit=str(PINNED_FLUXVLA_COMMIT),
        train_model_spec=cast(
            Mapping[str, object],
            deepcopy(CONFIG["train_model"]),
        ),
        inference_model_spec=cast(
            Mapping[str, object],
            deepcopy(CONFIG["inference_model"]),
        ),
    )


__all__ = ["CONFIG", "PINNED_FLUXVLA_COMMIT", "build_config_surface"]
