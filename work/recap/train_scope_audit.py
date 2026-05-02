from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
from typing import Any, cast


TRAIN_SCOPE_TAXONOMY_SCHEMA_VERSION = "recap_train_scope_taxonomy_v1"
TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY = "train_scope_taxonomy"
STATIC_SCOPE_AUDIT_SCHEMA_VERSION = "recap_full_update_scope_audit_static_v1"
STATIC_SCOPE_AUDIT_ARTIFACT_KIND = "recap_full_update_scope_audit_static"
STATIC_SCOPE_AUDIT_FILENAME = "full_update_scope_audit.json"
RUNTIME_RESOLUTION_STATUS_NOT_ATTEMPTED = "NOT_ATTEMPTED"
FULL_UPDATE_SCOPE_NAMES: tuple[str, ...] = (
    "strict_full",
    "full_policy",
    "full_action",
)
RECAP_TRAIN_SCOPE_CHOICES: tuple[str, ...] = (
    "current_partial",
    "full_action",
    "full_policy",
    "strict_full",
)
PAPER_METHOD_GAP: tuple[str, ...] = (
    "no_binarized_advantage_indicator",
    "no_advantage_condition_dropout",
    "no_cfg_policy_extraction",
    "scalar_embedding_not_language_indicator",
)

_METHOD_FAITHFULNESS: dict[str, Any] = {
    "recap_method_contract": "continuous_numeric_advantage_v2",
    "paper_equivalent": False,
    "paper_method_gap": list(PAPER_METHOD_GAP),
}
_ALLOWED_RECAP_METHOD_CONTRACTS = frozenset(
    {
        "continuous_numeric_advantage_v2",
        "binary_text_indicator_v1",
    }
)

_SCOPE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "current_partial": {
        "scope_faithfulness": "legacy_partial_control",
        "required_trainable_families": [
            "action_head.advantage_embedding.*",
            "action_head.vlln.*",
            "existing_nonzero_lr_condition_or_projector_hot_groups",
        ],
        "explicit_exclusions": [
            "legacy_cold_diffusion_trunk_groups (diffusion_trunk_lr_scale=0.0)",
        ],
    },
    "full_action": {
        "scope_faithfulness": "maximal_feasible_action_scope_candidate",
        "required_trainable_families": [
            "action_head.*",
        ],
        "explicit_exclusions": [
            "projector.*",
            "vla_action_interface.*",
            "backbone.*",
        ],
    },
    "full_policy": {
        "scope_faithfulness": "maximal_feasible_policy_scope_candidate",
        "required_trainable_families": [
            "action_head.*",
            "projector.*",
            "vla_action_interface.*",
        ],
        "explicit_exclusions": [
            "vision_language_backbone.*",
            "top_llm_layers.*",
        ],
    },
    "strict_full": {
        "scope_faithfulness": "strict_full_scope_candidate",
        "required_trainable_families": [
            "model.named_parameters()",
        ],
        "explicit_exclusions": [],
    },
}


def _copy_mapping(value: Mapping[str, object]) -> dict[str, Any]:
    return deepcopy(dict(value))


def parse_scope_flag(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "--recap-train-scope must be a non-empty string; expected one of "
            + f"{list(RECAP_TRAIN_SCOPE_CHOICES)}"
        )
    normalized = raw.strip()
    if normalized not in RECAP_TRAIN_SCOPE_CHOICES:
        raise ValueError(
            "--recap-train-scope must be one of "
            + f"{list(RECAP_TRAIN_SCOPE_CHOICES)}, got {normalized!r}"
        )
    return normalized


def add_scope_flag_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str,
    help_text: str,
) -> None:
    normalized_default = parse_scope_flag(default)
    parser.add_argument(
        "--recap-train-scope",
        type=parse_scope_flag,
        choices=RECAP_TRAIN_SCOPE_CHOICES,
        default=normalized_default,
        help=help_text,
    )


def build_scope_summary(
    requested_scope: object,
    *,
    legacy_scope_bridge: Mapping[str, object] | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    normalized_scope = parse_scope_flag(requested_scope)
    definition = _copy_mapping(_SCOPE_DEFINITIONS[normalized_scope])
    payload: dict[str, Any] = {
        "schema_version": TRAIN_SCOPE_TAXONOMY_SCHEMA_VERSION,
        "train_scope_requested": normalized_scope,
        "scope_faithfulness": str(definition["scope_faithfulness"]),
        "method_faithfulness": deepcopy(_METHOD_FAITHFULNESS),
        "required_trainable_families": list(definition["required_trainable_families"]),
        "explicit_exclusions": list(definition["explicit_exclusions"]),
    }
    if isinstance(legacy_scope_bridge, Mapping):
        payload["legacy_scope_shim"] = _copy_mapping(legacy_scope_bridge)
    if isinstance(extra_metadata, Mapping):
        for key, value in extra_metadata.items():
            if key in payload:
                continue
            payload[str(key)] = deepcopy(value)
    return payload


def normalize_train_scope_payload(
    payload: object,
    *,
    field_name: str = TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY,
) -> dict[str, Any]:
    if isinstance(payload, str):
        return build_scope_summary(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{field_name} must be a scope string or object, got {type(payload).__name__}"
        )

    legacy_scope_bridge = payload.get("legacy_scope_shim")
    normalized = build_scope_summary(
        payload.get("train_scope_requested"),
        legacy_scope_bridge=(legacy_scope_bridge if isinstance(legacy_scope_bridge, Mapping) else None),
    )

    raw_schema_version = payload.get("schema_version")
    if (
        raw_schema_version is not None
        and raw_schema_version != TRAIN_SCOPE_TAXONOMY_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{field_name}.schema_version must equal {TRAIN_SCOPE_TAXONOMY_SCHEMA_VERSION!r}"
        )

    for field_name_suffix in (
        "scope_faithfulness",
        "required_trainable_families",
        "explicit_exclusions",
    ):
        declared_value = payload.get(field_name_suffix)
        if declared_value is None:
            continue
        if declared_value != normalized[field_name_suffix]:
            raise ValueError(
                f"{field_name}.{field_name_suffix} must match requested scope "
                + f"{normalized['train_scope_requested']!r}"
            )

    declared_method = payload.get("method_faithfulness")
    if declared_method is not None:
        declared_contract = (
            declared_method.get("recap_method_contract")
            if isinstance(declared_method, Mapping)
            else None
        )
        if (
            declared_method != normalized["method_faithfulness"]
            and declared_contract not in _ALLOWED_RECAP_METHOD_CONTRACTS
        ):
            raise ValueError(
                f"{field_name}.method_faithfulness must match the "
                + "continuous_numeric_advantage_v2 contract"
            )
        if isinstance(declared_method, Mapping):
            normalized["method_faithfulness"] = _copy_mapping(declared_method)

    canonical_keys = set(normalized)
    for key, value in payload.items():
        normalized_key = str(key)
        if normalized_key in canonical_keys:
            continue
        normalized[normalized_key] = deepcopy(value)
    return normalized


def _safe_tensor_shape(param: Any) -> list[int]:
    try:
        return [int(dim) for dim in tuple(getattr(param, "shape", ()))]
    except Exception:
        return []


def _safe_tensor_numel(param: Any) -> int:
    try:
        return int(param.numel())
    except Exception:
        return 0


def _safe_tensor_dtype(param: Any) -> str:
    return str(getattr(param, "dtype", "unknown"))


def _safe_tensor_device(param: Any) -> str:
    return str(getattr(param, "device", "unknown"))


def _safe_is_meta(param: Any) -> bool:
    return bool(getattr(param, "is_meta", False))


def _coerce_float(raw: object, *, default: float = 0.0) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return float(default)
    return float(default)


def _coerce_int(raw: object, *, default: int = 0) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return int(default)
    return int(default)


def _iter_named_parameters_allow_duplicates(model: Any) -> list[tuple[str, Any]]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model must define named_parameters()")

    try:
        signature = inspect.signature(named_parameters)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and "remove_duplicate" in signature.parameters:
        iterator = named_parameters(remove_duplicate=False)
    else:
        iterator = named_parameters()
    if not isinstance(iterator, Iterable):
        raise TypeError("model.named_parameters() must return an iterable")

    return [
        (str(name), param)
        for name, param in cast(Iterable[tuple[object, Any]], iterator)
    ]


def _logical_bucket_name_for_param(name: str) -> str:
    if name.startswith("action_head.advantage_embedding.") or name.startswith(
        "action_head.vlln."
    ):
        return "condition_hot"
    if name.startswith("action_head.model."):
        return "diffusion_trunk_cold"
    return "default"


def _family_matches(
    *,
    family: str,
    parameter_row: Mapping[str, Any],
) -> bool:
    name = str(parameter_row["name"])
    optimizer_lrs = [float(value) for value in parameter_row["optimizer_lr_values"]]
    if family == "model.named_parameters()":
        return True
    if family == "existing_nonzero_lr_condition_or_projector_hot_groups":
        if not any(lr > 0.0 for lr in optimizer_lrs):
            return False
        return (
            name.startswith("action_head.advantage_embedding.")
            or name.startswith("action_head.vlln.")
            or name.startswith("projector.")
        )
    if family == "legacy_cold_diffusion_trunk_groups (diffusion_trunk_lr_scale=0.0)":
        return name.startswith("action_head.model.") and any(lr <= 0.0 for lr in optimizer_lrs)
    if family.endswith(".*"):
        prefix = family[:-2]
        return name == prefix or name.startswith(prefix + ".")
    return name == family


def _build_optimizer_index(
    optimizer: Any,
    *,
    named_parameter_rows: Sequence[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    raw_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(raw_groups, list):
        raise TypeError("optimizer.param_groups must be a list")

    names_by_param_id: dict[int, list[str]] = defaultdict(list)
    requires_grad_by_param_id: dict[int, bool] = {}
    numel_by_param_id: dict[int, int] = {}
    for row in named_parameter_rows:
        param_id = int(row["param_id"])
        name = str(row["name"])
        if name not in names_by_param_id[param_id]:
            names_by_param_id[param_id].append(name)
        requires_grad_by_param_id[param_id] = bool(row["requires_grad"])
        numel_by_param_id[param_id] = int(row["numel"])

    optimizer_entries_by_param_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    duplicate_optimizer_params: list[dict[str, Any]] = []
    optimizer_group_rows: list[dict[str, Any]] = []

    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, Mapping):
            raise TypeError(f"optimizer.param_groups[{group_index}] must be a mapping")
        group_params = raw_group.get("params", [])
        if not isinstance(group_params, list):
            raise TypeError(
                f"optimizer.param_groups[{group_index}]['params'] must be a list"
            )

        lr = _coerce_float(raw_group.get("lr", 0.0))
        weight_decay = _coerce_float(raw_group.get("weight_decay", 0.0))
        counts_in_group = Counter(id(param) for param in group_params)
        for param_id, count in counts_in_group.items():
            if count > 1:
                duplicate_optimizer_params.append(
                    {
                        "param_id": param_id,
                        "param_identity": f"param_{param_id}",
                        "model_names": list(names_by_param_id.get(param_id, [])),
                        "group_indices": [group_index],
                        "occurrence_count": int(count),
                        "duplicate_kind": "within_group",
                    }
                )

        group_entry_rows: list[dict[str, Any]] = []
        seen_param_ids: set[int] = set()
        for item_index, param in enumerate(group_params):
            param_id = id(param)
            entry = {
                "group_index": group_index,
                "item_index": item_index,
                "param_id": param_id,
                "param_identity": f"param_{param_id}",
                "lr": lr,
                "weight_decay": weight_decay,
            }
            optimizer_entries_by_param_id[param_id].append(entry)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            group_entry_rows.append(
                {
                    "param_id": param_id,
                    "param_identity": f"param_{param_id}",
                    "names": list(names_by_param_id.get(param_id, [])),
                    "requires_grad": bool(requires_grad_by_param_id.get(param_id, False)),
                    "numel": int(numel_by_param_id.get(param_id, _safe_tensor_numel(param))),
                }
            )

        trainable_names = sorted(
            {
                name
                for entry in group_entry_rows
                if bool(entry["requires_grad"])
                for name in entry["names"]
            }
        )
        frozen_names = sorted(
            {
                name
                for entry in group_entry_rows
                if not bool(entry["requires_grad"])
                for name in entry["names"]
            }
        )
        zero_lr_trainable_names = trainable_names if lr <= 0.0 else []
        optimizer_group_rows.append(
            {
                "group_index": group_index,
                "lr": lr,
                "weight_decay": weight_decay,
                "param_count": len(group_entry_rows),
                "total_numel": int(sum(int(entry["numel"]) for entry in group_entry_rows)),
                "bucket_name_preview": sorted(
                    {
                        _logical_bucket_name_for_param(name)
                        for entry in group_entry_rows
                        for name in entry["names"]
                    }
                ),
                "param_rows": group_entry_rows,
                "trainable_names": trainable_names,
                "frozen_names": frozen_names,
                "zero_lr_trainable_names": zero_lr_trainable_names,
            }
        )

    seen_cross_group: set[tuple[int, tuple[int, ...]]] = set()
    for param_id, entries in optimizer_entries_by_param_id.items():
        distinct_groups = sorted({int(entry["group_index"]) for entry in entries})
        if len(distinct_groups) <= 1:
            continue
        key = (param_id, tuple(distinct_groups))
        if key in seen_cross_group:
            continue
        seen_cross_group.add(key)
        duplicate_optimizer_params.append(
            {
                "param_id": param_id,
                "param_identity": f"param_{param_id}",
                "model_names": list(names_by_param_id.get(param_id, [])),
                "group_indices": distinct_groups,
                "occurrence_count": len(entries),
                "duplicate_kind": "cross_group",
            }
        )

    duplicate_optimizer_params.sort(
        key=lambda row: (
            str(row["duplicate_kind"]),
            list(row["group_indices"]),
            list(row["model_names"]),
        )
    )
    optimizer_group_rows.sort(key=lambda row: int(row["group_index"]))
    return optimizer_group_rows, optimizer_entries_by_param_id, duplicate_optimizer_params


def _build_parameter_rows(
    named_parameters: Sequence[tuple[str, Any]],
    optimizer_entries_by_param_id: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    parameter_rows: list[dict[str, Any]] = []
    for occurrence_index, (name, param) in enumerate(named_parameters):
        param_id = id(param)
        optimizer_entries = list(optimizer_entries_by_param_id.get(param_id, []))
        parameter_rows.append(
            {
                "name": str(name),
                "occurrence_index": occurrence_index,
                "param_id": param_id,
                "param_identity": f"param_{param_id}",
                "requires_grad": bool(getattr(param, "requires_grad", False)),
                "numel": _safe_tensor_numel(param),
                "shape": _safe_tensor_shape(param),
                "dtype": _safe_tensor_dtype(param),
                "device": _safe_tensor_device(param),
                "is_meta": _safe_is_meta(param),
                "logical_bucket_hint": _logical_bucket_name_for_param(str(name)),
                "in_optimizer": bool(optimizer_entries),
                "optimizer_group_indices": sorted(
                    {int(entry["group_index"]) for entry in optimizer_entries}
                ),
                "optimizer_lr_values": sorted(
                    {float(entry["lr"]) for entry in optimizer_entries}
                ),
                "optimizer_weight_decay_values": sorted(
                    {float(entry["weight_decay"]) for entry in optimizer_entries}
                ),
            }
        )
    parameter_rows.sort(key=lambda row: (str(row["name"]), int(row["occurrence_index"])))
    return parameter_rows


def _build_candidate_scope_coverage(
    *,
    scope_summary: Mapping[str, Any],
    parameter_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    required_trainable_families = [
        str(family) for family in scope_summary.get("required_trainable_families", [])
    ]
    explicit_exclusions = [
        str(family) for family in scope_summary.get("explicit_exclusions", [])
    ]

    required_family_rows: list[dict[str, Any]] = []
    required_trainable_names_missing: list[str] = []
    for family in required_trainable_families:
        matched_rows = [
            row for row in parameter_rows if _family_matches(family=family, parameter_row=row)
        ]
        matched_names = [str(row["name"]) for row in matched_rows]
        matched_trainable_names = [
            str(row["name"]) for row in matched_rows if bool(row["requires_grad"])
        ]
        missing_names = sorted(set(matched_names) - set(matched_trainable_names))
        required_trainable_names_missing.extend(missing_names)
        required_family_rows.append(
            {
                "family": family,
                "matched_parameter_names": matched_names,
                "matched_trainable_parameter_names": matched_trainable_names,
                "missing_trainable_parameter_names": missing_names,
                "satisfied": bool(matched_rows) and not missing_names,
            }
        )

    exclusion_rows: list[dict[str, Any]] = []
    forbidden_trainable_names: list[str] = []
    for family in explicit_exclusions:
        matched_rows = [
            row for row in parameter_rows if _family_matches(family=family, parameter_row=row)
        ]
        matched_names = [str(row["name"]) for row in matched_rows]
        matched_trainable_names = [
            str(row["name"]) for row in matched_rows if bool(row["requires_grad"])
        ]
        forbidden_trainable_names.extend(matched_trainable_names)
        exclusion_rows.append(
            {
                "family": family,
                "matched_parameter_names": matched_names,
                "matched_trainable_parameter_names": matched_trainable_names,
                "satisfied": not matched_trainable_names,
            }
        )

    return {
        "required_trainable_families": required_trainable_families,
        "explicit_exclusions": explicit_exclusions,
        "diffusion_trunk_requires_grad_count": sum(
            1
            for row in parameter_rows
            if str(row.get("name", "")).startswith("action_head.model.")
            and bool(row.get("requires_grad", False))
        ),
        "required_family_rows": required_family_rows,
        "explicit_exclusion_rows": exclusion_rows,
        "required_trainable_names_missing": sorted(set(required_trainable_names_missing)),
        "forbidden_trainable_names": sorted(set(forbidden_trainable_names)),
    }


def _build_optimizer_integrity(
    *,
    parameter_rows: Sequence[dict[str, Any]],
    optimizer_group_rows: Sequence[dict[str, Any]],
    duplicate_optimizer_params: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    trainable_params_missing_from_optimizer = sorted(
        {
            str(row["name"])
            for row in parameter_rows
            if bool(row["requires_grad"]) and not bool(row["in_optimizer"])
        }
    )
    frozen_params_present_in_optimizer = sorted(
        {
            str(row["name"])
            for row in parameter_rows
            if (not bool(row["requires_grad"])) and bool(row["in_optimizer"])
        }
    )
    zero_lr_trainable_param_groups = [
        {
            "group_index": int(group_row["group_index"]),
            "lr": float(group_row["lr"]),
            "weight_decay": float(group_row["weight_decay"]),
            "trainable_names": list(group_row["trainable_names"]),
            "zero_lr_trainable_names": list(group_row["zero_lr_trainable_names"]),
        }
        for group_row in optimizer_group_rows
        if bool(group_row["zero_lr_trainable_names"])
    ]

    model_param_ids = {int(row["param_id"]) for row in parameter_rows}
    optimizer_param_ids: set[int] = set()
    for group_row in optimizer_group_rows:
        for param_row in group_row["param_rows"]:
            optimizer_param_ids.add(int(param_row["param_id"]))

    advantage_embedding_missing_from_optimizer = any(
        name.startswith("action_head.advantage_embedding.")
        for name in trainable_params_missing_from_optimizer
    )

    return {
        "optimizer_group_count": len(optimizer_group_rows),
        "trainable_params_missing_from_optimizer": trainable_params_missing_from_optimizer,
        "advantage_embedding_not_in_optimizer": advantage_embedding_missing_from_optimizer,
        "frozen_params_present_in_optimizer": frozen_params_present_in_optimizer,
        "duplicate_optimizer_params": [dict(row) for row in duplicate_optimizer_params],
        "optimizer_params_without_model_match": sorted(
            f"param_{param_id}" for param_id in optimizer_param_ids - model_param_ids
        ),
        "zero_lr_trainable_param_groups": zero_lr_trainable_param_groups,
    }


def estimate_memory_feasibility(
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    scope_requested: object | None = None,
    audit: Mapping[str, Any] | None = None,
    available_memory_bytes: int | None = None,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    optimizer_state_bytes: int = 8,
    master_weight_bytes: int = 4,
) -> dict[str, Any]:
    if audit is None:
        if model is None or optimizer is None:
            raise ValueError(
                "estimate_memory_feasibility requires either audit or model+optimizer"
            )
        audit = compute_static_scope_audit(
            model=model,
            optimizer=optimizer,
            scope_requested=("current_partial" if scope_requested is None else scope_requested),
        )

    parameter_coverage = audit.get("parameter_coverage")
    if not isinstance(parameter_coverage, Mapping):
        parameter_coverage = {}
    parameter_rows = parameter_coverage.get("parameter_rows", [])
    if not isinstance(parameter_rows, list):
        parameter_rows = []
    trainable_numel = sum(
        _coerce_int(row.get("numel", 0))
        for row in parameter_rows
        if isinstance(row, Mapping) and bool(row.get("requires_grad", False))
    )
    estimated_parameter_bytes = int(trainable_numel) * int(parameter_bytes)
    estimated_gradient_bytes = int(trainable_numel) * int(gradient_bytes)
    estimated_optimizer_state_bytes = int(trainable_numel) * int(optimizer_state_bytes)
    estimated_master_weight_bytes = int(trainable_numel) * int(master_weight_bytes)
    estimated_total_bytes = (
        estimated_parameter_bytes
        + estimated_gradient_bytes
        + estimated_optimizer_state_bytes
        + estimated_master_weight_bytes
    )
    fits_available_memory = (
        None
        if available_memory_bytes is None
        else estimated_total_bytes <= int(available_memory_bytes)
    )
    return {
        "audit_phase": "static",
        "estimator_kind": "heuristic_trainable_param_memory",
        "trainable_numel": int(trainable_numel),
        "estimated_parameter_bytes": estimated_parameter_bytes,
        "estimated_gradient_bytes": estimated_gradient_bytes,
        "estimated_optimizer_state_bytes": estimated_optimizer_state_bytes,
        "estimated_master_weight_bytes": estimated_master_weight_bytes,
        "estimated_total_bytes": estimated_total_bytes,
        "available_memory_bytes": (
            None if available_memory_bytes is None else int(available_memory_bytes)
        ),
        "fits_available_memory": fits_available_memory,
        "assumptions": {
            "parameter_bytes": int(parameter_bytes),
            "gradient_bytes": int(gradient_bytes),
            "optimizer_state_bytes": int(optimizer_state_bytes),
            "master_weight_bytes": int(master_weight_bytes),
        },
    }


def compute_static_scope_audit(
    model: Any,
    optimizer: Any,
    scope_requested: object,
) -> dict[str, Any]:
    scope_summary = normalize_train_scope_payload(scope_requested)
    named_parameters = _iter_named_parameters_allow_duplicates(model)
    optimizer_group_rows, optimizer_entries_by_param_id, duplicate_optimizer_params = (
        _build_optimizer_index(optimizer, named_parameter_rows=[
            {
                "name": name,
                "param_id": id(param),
                "requires_grad": bool(getattr(param, "requires_grad", False)),
                "numel": _safe_tensor_numel(param),
            }
            for name, param in named_parameters
        ])
    )
    parameter_rows = _build_parameter_rows(named_parameters, optimizer_entries_by_param_id)
    candidate_scope_coverage = _build_candidate_scope_coverage(
        scope_summary=scope_summary,
        parameter_rows=parameter_rows,
    )
    optimizer_integrity = _build_optimizer_integrity(
        parameter_rows=parameter_rows,
        optimizer_group_rows=optimizer_group_rows,
        duplicate_optimizer_params=duplicate_optimizer_params,
    )

    block_reasons: list[str] = []
    if candidate_scope_coverage["required_trainable_names_missing"]:
        block_reasons.append("SCOPE_REQUIRED_TRAINABLE_MISSING")
    if candidate_scope_coverage["forbidden_trainable_names"]:
        block_reasons.append("SCOPE_FORBIDDEN_TRAINABLE")
    if optimizer_integrity["trainable_params_missing_from_optimizer"]:
        block_reasons.append("TRAINABLE_MISSING_FROM_OPTIMIZER")
    if optimizer_integrity["advantage_embedding_not_in_optimizer"]:
        block_reasons.append("ADVANTAGE_EMBEDDING_NOT_IN_OPTIMIZER")
    if optimizer_integrity["duplicate_optimizer_params"]:
        block_reasons.append("DUPLICATE_OPTIMIZER_PARAM")
    if optimizer_integrity["optimizer_params_without_model_match"]:
        block_reasons.append("OPTIMIZER_PARAM_NOT_IN_MODEL")
    if optimizer_integrity["frozen_params_present_in_optimizer"]:
        block_reasons.append("FROZEN_PARAM_IN_OPTIMIZER")
    if optimizer_integrity["zero_lr_trainable_param_groups"]:
        block_reasons.append("ZERO_LR_TRAINABLE_PARAM_GROUP")

    audit: dict[str, Any] = {
        "schema_version": STATIC_SCOPE_AUDIT_SCHEMA_VERSION,
        "artifact_kind": STATIC_SCOPE_AUDIT_ARTIFACT_KIND,
        "audit_phase": "static",
        "train_scope_requested": str(scope_summary["train_scope_requested"]),
        "scope_faithfulness": str(scope_summary["scope_faithfulness"]),
        "method_faithfulness": deepcopy(scope_summary["method_faithfulness"]),
        "strict_full_runtime_attempted": False,
        "runtime_resolution_status": RUNTIME_RESOLUTION_STATUS_NOT_ATTEMPTED,
        "static_verdict": "BLOCK" if block_reasons else "PASS",
        "static_block_reasons": block_reasons,
        "candidate_scope_coverage": candidate_scope_coverage,
        "optimizer_integrity": optimizer_integrity,
        "optimizer_buckets": optimizer_group_rows,
        "parameter_coverage": {
            "total_parameter_rows": len(parameter_rows),
            "trainable_parameter_rows": sum(
                1 for row in parameter_rows if bool(row["requires_grad"])
            ),
            "frozen_parameter_rows": sum(
                1 for row in parameter_rows if not bool(row["requires_grad"])
            ),
            "parameter_rows": parameter_rows,
        },
        "scope_taxonomy": {
            "schema_version": str(scope_summary["schema_version"]),
            "required_trainable_families": list(
                scope_summary["required_trainable_families"]
            ),
            "explicit_exclusions": list(scope_summary["explicit_exclusions"]),
        },
    }
    audit["memory_feasibility"] = estimate_memory_feasibility(audit=audit)
    return audit


def emit_scope_audit_json(dest: str | Path, audit: Mapping[str, Any]) -> Path:
    path = Path(dest).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(dict(audit), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    return path


__all__ = [
    "FULL_UPDATE_SCOPE_NAMES",
    "PAPER_METHOD_GAP",
    "RECAP_TRAIN_SCOPE_CHOICES",
    "RUNTIME_RESOLUTION_STATUS_NOT_ATTEMPTED",
    "STATIC_SCOPE_AUDIT_ARTIFACT_KIND",
    "STATIC_SCOPE_AUDIT_FILENAME",
    "STATIC_SCOPE_AUDIT_SCHEMA_VERSION",
    "TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY",
    "TRAIN_SCOPE_TAXONOMY_SCHEMA_VERSION",
    "add_scope_flag_argument",
    "build_scope_summary",
    "compute_static_scope_audit",
    "emit_scope_audit_json",
    "estimate_memory_feasibility",
    "normalize_train_scope_payload",
    "parse_scope_flag",
]
