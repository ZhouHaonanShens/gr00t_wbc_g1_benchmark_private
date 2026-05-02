from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


SCOPE_EXPERIMENT_SCHEMA_VERSION = "gr00t_scope_experiment_v1"
SCOPE_EXPERIMENT_EXTENSION_KEY = "scope_experiment"
SCOPE_EXPERIMENT_VARIABLE_NAME = "scope_preset"

ACTION_HEAD_ONLY_EVAL_OVERLAY_REGEX = r"^action_head\..*"
S3_TOP_BACKBONE_LAYER_INDICES: tuple[int, ...] = (12, 13, 14, 15)

_TRAINABLE_REGEX_FIELD = "trainable_module_regex"
_EVAL_REGEX_FIELD = "eval_overlay_regex"
_DERIVED_CORE_FIELD_NAMES: tuple[str, str] = (
    _TRAINABLE_REGEX_FIELD,
    _EVAL_REGEX_FIELD,
)


def _compile_regex_union(pattern_bodies: tuple[str, ...]) -> str:
    if not pattern_bodies:
        raise ValueError("scope preset regex body list must be non-empty")
    if len(pattern_bodies) == 1:
        return "^" + pattern_bodies[0] + "$"
    return "^(?:" + "|".join(pattern_bodies) + ")$"


def _copy_mapping(value: Mapping[str, object]) -> dict[str, Any]:
    return deepcopy(dict(value))


_SCOPE_PRESET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "S1": {
        "preset_id": "S1",
        "description": "action-head only",
        "trainable_component_ids": ["action_head"],
        "semantic_components": {
            "action_head": True,
            "text_ingress": False,
            "conditioning_projector": False,
            "nearby_fusion_ingress": False,
            "top_backbone_fusion_blocks": {
                "enabled": False,
                "layer_indices": [],
            },
        },
        "trainable_parameter_families": [
            "action_head",
        ],
        "trainable_regex_bodies": (r"action_head\..*",),
        "current_eval_lane": {
            "eval_script": "work/recap/scripts/3D_recap_eval.py",
            "overlay_route": "action_head_only_overlay",
            "coverage": "full",
            "overlay_represents_full_trainable_scope": True,
            "covered_trainable_component_ids": ["action_head"],
            "uncovered_trainable_component_ids": [],
        },
    },
    "S2": {
        "preset_id": "S2",
        "description": (
            "action-head plus text ingress, conditioning projector, and nearby fusion ingress"
        ),
        "trainable_component_ids": [
            "action_head",
            "text_ingress",
            "conditioning_projector",
            "nearby_fusion_ingress",
        ],
        "semantic_components": {
            "action_head": True,
            "text_ingress": True,
            "conditioning_projector": True,
            "nearby_fusion_ingress": True,
            "top_backbone_fusion_blocks": {
                "enabled": False,
                "layer_indices": [],
            },
        },
        "trainable_parameter_families": [
            "action_head",
            "backbone.model.language_model.model.embed_tokens",
            "backbone.model.mlp1",
        ],
        "trainable_regex_bodies": (
            r"action_head\..*",
            r"backbone\.model\.language_model\.model\.embed_tokens\..*",
            r"backbone\.model\.mlp1\..*",
        ),
        "current_eval_lane": {
            "eval_script": "work/recap/scripts/3D_recap_eval.py",
            "overlay_route": "action_head_only_overlay",
            "coverage": "partial_action_head_only",
            "overlay_represents_full_trainable_scope": False,
            "covered_trainable_component_ids": ["action_head"],
            "uncovered_trainable_component_ids": [
                "text_ingress",
                "conditioning_projector",
                "nearby_fusion_ingress",
            ],
        },
    },
    "S3": {
        "preset_id": "S3",
        "description": ("S2 plus a small top-backbone and fusion block extension"),
        "trainable_component_ids": [
            "action_head",
            "text_ingress",
            "conditioning_projector",
            "nearby_fusion_ingress",
            "top_backbone_fusion_blocks",
        ],
        "semantic_components": {
            "action_head": True,
            "text_ingress": True,
            "conditioning_projector": True,
            "nearby_fusion_ingress": True,
            "top_backbone_fusion_blocks": {
                "enabled": True,
                "layer_indices": list(S3_TOP_BACKBONE_LAYER_INDICES),
            },
        },
        "trainable_parameter_families": [
            "action_head",
            "backbone.model.language_model.model.embed_tokens",
            "backbone.model.mlp1",
            "backbone.model.language_model.model.layers.12-15",
        ],
        "trainable_regex_bodies": (
            r"action_head\..*",
            r"backbone\.model\.language_model\.model\.embed_tokens\..*",
            r"backbone\.model\.mlp1\..*",
            r"backbone\.model\.language_model\.model\.layers\.(?:12|13|14|15)\..*",
        ),
        "current_eval_lane": {
            "eval_script": "work/recap/scripts/3D_recap_eval.py",
            "overlay_route": "action_head_only_overlay",
            "coverage": "partial_action_head_only",
            "overlay_represents_full_trainable_scope": False,
            "covered_trainable_component_ids": ["action_head"],
            "uncovered_trainable_component_ids": [
                "text_ingress",
                "conditioning_projector",
                "nearby_fusion_ingress",
                "top_backbone_fusion_blocks",
            ],
        },
    },
}


SCOPE_PRESET_IDS: tuple[str, ...] = tuple(_SCOPE_PRESET_DEFINITIONS.keys())

V2_TRAIN_SCOPE_IDS: tuple[str, ...] = (
    "current_partial",
    "full_action",
    "full_policy",
    "strict_full",
)

_V2_TRAIN_SCOPE_LEGACY_SHIM: dict[str, dict[str, Any]] = {
    "current_partial": {
        "legacy_scope_semantics_preserved": True,
        "legacy_scope_presets_unchanged": list(SCOPE_PRESET_IDS),
        "legacy_scope_bridge_note": (
            "v2 current_partial is additive taxonomy only; it does not rename or rewrite legacy S1/S2/S3 semantics"
        ),
    },
    "full_action": {
        "legacy_scope_semantics_preserved": True,
        "legacy_scope_presets_unchanged": list(SCOPE_PRESET_IDS),
        "legacy_scope_bridge_note": (
            "v2 full_action is reported alongside the unchanged legacy S1/S2/S3 presets"
        ),
    },
    "full_policy": {
        "legacy_scope_semantics_preserved": True,
        "legacy_scope_presets_unchanged": list(SCOPE_PRESET_IDS),
        "legacy_scope_bridge_note": (
            "v2 full_policy is reported alongside the unchanged legacy S1/S2/S3 presets"
        ),
    },
    "strict_full": {
        "legacy_scope_semantics_preserved": True,
        "legacy_scope_presets_unchanged": list(SCOPE_PRESET_IDS),
        "legacy_scope_bridge_note": (
            "v2 strict_full is candidate-only and cannot be relabeled as any legacy S1/S2/S3 preset"
        ),
    },
}


def build_v2_train_scope_shim_metadata(requested_scope: object) -> dict[str, Any]:
    if not isinstance(requested_scope, str) or not requested_scope.strip():
        raise ValueError(
            "requested_scope must be a non-empty scope string; expected one of "
            + f"{list(V2_TRAIN_SCOPE_IDS)}"
        )
    normalized_scope = requested_scope.strip()
    metadata = _V2_TRAIN_SCOPE_LEGACY_SHIM.get(normalized_scope)
    if metadata is None:
        raise ValueError(
            f"requested_scope must be one of {list(V2_TRAIN_SCOPE_IDS)}, got {normalized_scope!r}"
        )
    payload = _copy_mapping(metadata)
    payload["requested_scope"] = normalized_scope
    return payload


def _resolve_scope_preset_definition(
    preset_id: object,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(preset_id, str) or not preset_id.strip():
        raise ValueError(
            f"{field_name} must be a non-empty preset id string; expected one of {list(SCOPE_PRESET_IDS)}"
        )
    normalized_preset_id = preset_id.strip()
    preset = _SCOPE_PRESET_DEFINITIONS.get(normalized_preset_id)
    if preset is None:
        raise ValueError(
            f"{field_name} must be one of {list(SCOPE_PRESET_IDS)}, got {normalized_preset_id!r}"
        )
    return preset


def derived_scope_core_fields(preset_id: object) -> dict[str, str]:
    preset = _resolve_scope_preset_definition(preset_id, field_name="preset_id")
    trainable_regex = _compile_regex_union(tuple(preset["trainable_regex_bodies"]))
    return {
        _TRAINABLE_REGEX_FIELD: trainable_regex,
        _EVAL_REGEX_FIELD: ACTION_HEAD_ONLY_EVAL_OVERLAY_REGEX,
    }


def build_scope_experiment_extension(
    preset_id: object,
    *,
    paired_triplet_artifacts: Mapping[str, object] | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    preset = _resolve_scope_preset_definition(preset_id, field_name="preset_id")
    payload: dict[str, Any] = {
        "schema_version": SCOPE_EXPERIMENT_SCHEMA_VERSION,
        "preset_id": str(preset["preset_id"]),
        "machine_readable_variable": SCOPE_EXPERIMENT_VARIABLE_NAME,
        "description": str(preset["description"]),
        "trainable_component_ids": list(preset["trainable_component_ids"]),
        "semantic_components": _copy_mapping(preset["semantic_components"]),
        "trainable_parameter_families": list(preset["trainable_parameter_families"]),
        "derived_core_fields": derived_scope_core_fields(preset_id),
        "current_eval_lane": _copy_mapping(preset["current_eval_lane"]),
    }
    if isinstance(paired_triplet_artifacts, Mapping):
        payload["paired_triplet_artifacts"] = _copy_mapping(paired_triplet_artifacts)
    if isinstance(extra_metadata, Mapping):
        for key, value in extra_metadata.items():
            if key in payload:
                continue
            payload[str(key)] = deepcopy(value)
    return payload


def normalize_scope_experiment_payload(
    payload: object,
    *,
    field_name: str = SCOPE_EXPERIMENT_EXTENSION_KEY,
) -> dict[str, Any]:
    if isinstance(payload, str):
        return build_scope_experiment_extension(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{field_name} must be a preset id string or object, got {type(payload).__name__}"
        )

    raw_schema_version = payload.get("schema_version")
    if (
        raw_schema_version is not None
        and raw_schema_version != SCOPE_EXPERIMENT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{field_name}.schema_version must equal {SCOPE_EXPERIMENT_SCHEMA_VERSION!r}"
        )

    raw_paired_triplet_artifacts = payload.get("paired_triplet_artifacts")
    paired_triplet_artifacts = (
        _copy_mapping(raw_paired_triplet_artifacts)
        if isinstance(raw_paired_triplet_artifacts, Mapping)
        else None
    )
    normalized = build_scope_experiment_extension(
        payload.get("preset_id"),
        paired_triplet_artifacts=paired_triplet_artifacts,
    )
    declared_derived_core_fields = payload.get("derived_core_fields")
    if isinstance(declared_derived_core_fields, Mapping):
        for field in _DERIVED_CORE_FIELD_NAMES:
            declared_value = declared_derived_core_fields.get(field)
            if declared_value is None:
                continue
            expected_value = normalized["derived_core_fields"][field]
            if str(declared_value) != expected_value:
                raise ValueError(
                    f"{field_name}.derived_core_fields.{field} must match preset {normalized['preset_id']}"
                )

    canonical_keys = set(normalized)
    for key, value in payload.items():
        normalized_key = str(key)
        if normalized_key in canonical_keys:
            continue
        normalized[normalized_key] = deepcopy(value)
    return normalized


def resolve_scope_experiment_from_manifest(
    manifest_payload: Mapping[str, Any],
    *,
    require_preset_metadata: bool = False,
) -> dict[str, Any] | None:
    extensions = manifest_payload.get("extensions")
    raw_scope = None
    if isinstance(extensions, Mapping):
        raw_scope = extensions.get(SCOPE_EXPERIMENT_EXTENSION_KEY)
    if raw_scope is None:
        if not require_preset_metadata:
            return None
        core = manifest_payload.get("core")
        if isinstance(core, Mapping):
            has_compatibility_regex = any(
                isinstance(core.get(field_name), str)
                and str(core.get(field_name)).strip()
                for field_name in _DERIVED_CORE_FIELD_NAMES
            )
            if has_compatibility_regex:
                raise ValueError(
                    "scope preset metadata is required; implicit regex-only scope is unsupported"
                )
        raise ValueError(
            "scope preset metadata is required in extensions.scope_experiment"
        )

    normalized = normalize_scope_experiment_payload(
        raw_scope,
        field_name=f"extensions.{SCOPE_EXPERIMENT_EXTENSION_KEY}",
    )
    core = manifest_payload.get("core")
    if isinstance(core, Mapping):
        derived_core_fields = normalized["derived_core_fields"]
        for field_name in _DERIVED_CORE_FIELD_NAMES:
            raw_value = core.get(field_name)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            normalized_value = raw_value.strip()
            if normalized_value != derived_core_fields[field_name]:
                raise ValueError(
                    "extensions.scope_experiment preset "
                    + f"{normalized['preset_id']} requires core.{field_name}="
                    + f"{derived_core_fields[field_name]!r}, got {normalized_value!r}"
                )
    return normalized


__all__ = [
    "ACTION_HEAD_ONLY_EVAL_OVERLAY_REGEX",
    "S3_TOP_BACKBONE_LAYER_INDICES",
    "SCOPE_EXPERIMENT_EXTENSION_KEY",
    "SCOPE_EXPERIMENT_SCHEMA_VERSION",
    "SCOPE_EXPERIMENT_VARIABLE_NAME",
    "SCOPE_PRESET_IDS",
    "V2_TRAIN_SCOPE_IDS",
    "build_scope_experiment_extension",
    "build_v2_train_scope_shim_metadata",
    "derived_scope_core_fields",
    "normalize_scope_experiment_payload",
    "resolve_scope_experiment_from_manifest",
]
