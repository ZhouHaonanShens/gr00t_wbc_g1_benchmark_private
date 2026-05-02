from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.models import flux_recap_vla


PINNED_FLUXVLA_COMMIT = "UNSET_FLUXVLA_COMMIT"


def build_variant_config(
    *,
    variant_id: str,
    train_relative_path: str,
    inference_relative_path: str,
) -> dict[str, object]:
    return {
        "variant_id": str(variant_id),
        "pinned_fluxvla_commit": PINNED_FLUXVLA_COMMIT,
        "authoritative_consumer_surface": flux_recap_vla.AUTHORITATIVE_CONSUMER_SURFACE,
        "train_model": {
            "artifact_id": f"{variant_id}_train_model",
            "authority_role": "flux_recap_train_model",
            "relative_path": str(train_relative_path),
            "registered_class": flux_recap_vla.TRAIN_MODEL_CLASS,
            "surface_role": flux_recap_vla.TRAIN_MODEL_ROLE,
        },
        "inference_model": {
            "artifact_id": f"{variant_id}_inference_model",
            "authority_role": "flux_recap_inference_model",
            "relative_path": str(inference_relative_path),
            "registered_class": flux_recap_vla.INFERENCE_MODEL_CLASS,
            "surface_role": flux_recap_vla.INFERENCE_MODEL_ROLE,
        },
    }


CONFIG = build_variant_config(
    variant_id="gr00t_g1_flux_recap_base",
    train_relative_path=(
        "agent/artifacts/apple_recap_flux_graft/models/gr00t_g1_flux_recap_base/train_model_ref.json"
    ),
    inference_relative_path=(
        "agent/artifacts/apple_recap_flux_graft/models/gr00t_g1_flux_recap_base/inference_model_ref.json"
    ),
)


def build_config_surface(*, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    return flux_recap_vla.build_flux_recap_model_surface(
        repo_root=repo_root,
        variant_id=str(CONFIG["variant_id"]),
        pinned_fluxvla_commit=str(CONFIG["pinned_fluxvla_commit"]),
        train_model_spec=cast(
            Mapping[str, object],
            deepcopy(CONFIG["train_model"]),
        ),
        inference_model_spec=cast(
            Mapping[str, object],
            deepcopy(CONFIG["inference_model"]),
        ),
    )


__all__ = [
    "CONFIG",
    "PINNED_FLUXVLA_COMMIT",
    "build_config_surface",
    "build_variant_config",
]
