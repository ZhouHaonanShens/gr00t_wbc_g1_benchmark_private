from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from work.recap.lerobot_export import dataset_export as lerobot_v2_export


PHASE1_IMAGE_FEATURE_KEY = "observation.images.ego_view"
PHASE1_STATE_FEATURE_KEY = "observation.state"
PHASE1_ACTION_FEATURE_KEY = "action"
PHASE1_PROMPT_FEATURE_PRIMARY_KEY = "annotation.human.task_description"
PHASE1_PROMPT_FEATURE_FALLBACK_KEY = "annotation.human.action.task_description"

OPENPI_PRIMARY_IMAGE_KEY = "observation/image"
OPENPI_WRIST_IMAGE_KEY = "observation/wrist_image"
OPENPI_STATE_KEY = "observation/state"
OPENPI_PROMPT_KEY = "prompt"

IMAGE_BRIDGE_MODE = "duplicate_ego_view_for_wrist_image"
PHASE1_DATASET_HANDOFF_KIND = "lerobot_v2_with_video_single_prompt"


@dataclass(frozen=True)
class Phase1DatasetMappingSpec:
    dataset_dir: Path
    dataset_handoff_kind: str
    source_image_feature_key: str
    source_state_feature_key: str
    source_action_feature_key: str
    source_prompt_feature_key: str
    openpi_primary_image_key: str
    openpi_wrist_image_key: str
    openpi_state_key: str
    openpi_prompt_key: str
    image_bridge_mode: str
    state_dim: int
    action_dim: int


def _read_json(path: Path) -> dict[str, object]:
    data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}, got {type(data).__name__}")
    raw_dict = cast(dict[object, object], data)
    return {str(key): value for key, value in raw_dict.items()}


def _require_feature_shape(
    features: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> int:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        raise ValueError(f"missing feature mapping {key!r} ({context})")
    feature_mapping = cast(Mapping[str, object], feature)
    shape = feature_mapping.get("shape")
    shape_list = cast(list[object], shape) if isinstance(shape, list) else None
    if shape_list is None or len(shape_list) != 1 or not isinstance(shape_list[0], int):
        raise ValueError(f"invalid shape for {key!r} ({context}): {shape!r}")
    if int(shape_list[0]) <= 0:
        raise ValueError(f"non-positive shape for {key!r} ({context}): {shape!r}")
    return int(shape_list[0])


def _require_mainline_carrier_authority(info: Mapping[str, object]) -> None:
    task_text_field = str(info.get("task_text_field", "")).strip()
    if task_text_field != lerobot_v2_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD:
        raise ValueError(
            "Phase 1 dataset handoff requires task_text_field=carrier_text_v1; "
            + "policy_condition_text remains metadata-only and prompt_conditioned remains legacy-only"
        )

    carrier_route = str(info.get("carrier_route", "")).strip()
    if carrier_route != lerobot_v2_export.EXPORTER_CARRIER_ROUTE:
        raise ValueError(
            "Phase 1 dataset handoff requires carrier_route=carrier_text_v1"
        )

    carrier_schema_version = str(info.get("carrier_schema_version", "")).strip()
    if carrier_schema_version != lerobot_v2_export.EXPORTER_CARRIER_SCHEMA_VERSION:
        raise ValueError(
            "Phase 1 dataset handoff requires canonical carrier_schema_version for carrier_text_v1"
        )

    prompt_source_field = str(info.get("prompt_source_field", "")).strip()
    if prompt_source_field != lerobot_v2_export.EXPORTER_PROMPT_SOURCE_FIELD:
        raise ValueError(
            "Phase 1 dataset handoff requires prompt_source_field=prompt_raw for the carrier_text_v1 authority contract"
        )

    prompt_route = str(info.get("prompt_route", "")).strip()
    if prompt_route != lerobot_v2_export.EXPORTER_PROMPT_ROUTE:
        raise ValueError(
            "Phase 1 dataset handoff requires the canonical prompt_route for the carrier_text_v1 authority contract"
        )

    conditioning_mode = str(info.get("conditioning_mode", "")).strip()
    if conditioning_mode != lerobot_v2_export.EXPORTER_CONDITIONING_MODE:
        raise ValueError(
            "Phase 1 dataset handoff requires the canonical conditioning_mode for the carrier_text_v1 authority contract"
        )


def resolve_phase1_dataset_dir(
    iter_tag: str, *, repo_root: str | Path | None = None
) -> Path:
    return lerobot_v2_export.resolve_lerobot_v2_dataset_dir(
        iter_tag=iter_tag,
        repo_root=repo_root,
    )


def build_phase1_dataset_mapping_spec(
    dataset_dir: str | Path,
) -> Phase1DatasetMappingSpec:
    dataset_dir_path = Path(dataset_dir).resolve()
    info_path = dataset_dir_path / "meta" / "info.json"
    modality_path = dataset_dir_path / "meta" / "modality.json"

    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    if not modality_path.is_file():
        raise FileNotFoundError(modality_path)

    info = _read_json(info_path)
    modality = _read_json(modality_path)

    dual_task_text = info.get("recap_export.dual_task_text")
    task_text_mode = info.get("task_text_mode")
    if bool(dual_task_text) or task_text_mode == "mix50":
        raise ValueError(
            "Phase 1 dataset handoff rejects ambiguous prompt sources: "
            + "dual_task_text/task_text_mode=mix50 must be disabled"
        )
    _require_mainline_carrier_authority(info)

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("info.json missing features mapping")
    features_mapping = cast(Mapping[str, object], features)

    video_modality = modality.get("video")
    if not isinstance(video_modality, Mapping):
        raise ValueError("modality.json missing video modality mapping")
    video_mapping = cast(Mapping[str, object], video_modality)
    has_direct_video_key = PHASE1_IMAGE_FEATURE_KEY in video_mapping
    has_original_key_match = any(
        isinstance(entry, Mapping)
        and cast(Mapping[str, object], entry).get("original_key")
        == PHASE1_IMAGE_FEATURE_KEY
        for entry in video_mapping.values()
    )
    if not has_direct_video_key and not has_original_key_match:
        raise ValueError(
            f"video modality missing required key {PHASE1_IMAGE_FEATURE_KEY!r}"
        )

    prompt_key = (
        PHASE1_PROMPT_FEATURE_PRIMARY_KEY
        if PHASE1_PROMPT_FEATURE_PRIMARY_KEY in features_mapping
        else PHASE1_PROMPT_FEATURE_FALLBACK_KEY
    )
    if prompt_key not in features_mapping:
        raise ValueError(
            "dataset is missing both prompt feature keys: "
            + f"{PHASE1_PROMPT_FEATURE_PRIMARY_KEY!r}, "
            + f"{PHASE1_PROMPT_FEATURE_FALLBACK_KEY!r}"
        )

    state_dim = _require_feature_shape(
        features_mapping, PHASE1_STATE_FEATURE_KEY, context="info.json"
    )
    action_dim = _require_feature_shape(
        features_mapping, PHASE1_ACTION_FEATURE_KEY, context="info.json"
    )

    return Phase1DatasetMappingSpec(
        dataset_dir=dataset_dir_path,
        dataset_handoff_kind=PHASE1_DATASET_HANDOFF_KIND,
        source_image_feature_key=PHASE1_IMAGE_FEATURE_KEY,
        source_state_feature_key=PHASE1_STATE_FEATURE_KEY,
        source_action_feature_key=PHASE1_ACTION_FEATURE_KEY,
        source_prompt_feature_key=str(prompt_key),
        openpi_primary_image_key=OPENPI_PRIMARY_IMAGE_KEY,
        openpi_wrist_image_key=OPENPI_WRIST_IMAGE_KEY,
        openpi_state_key=OPENPI_STATE_KEY,
        openpi_prompt_key=OPENPI_PROMPT_KEY,
        image_bridge_mode=IMAGE_BRIDGE_MODE,
        state_dim=state_dim,
        action_dim=action_dim,
    )
