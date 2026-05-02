from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from work.openpi.data.contract_mapping import build_phase1_dataset_mapping_spec
from work.openpi.norm.policy import (
    build_phase1_norm_policy,
    build_phase1_norm_provenance,
)
from work.openpi.recap.dataset import dataset_bundle_to_dict, resolve_recap_dataset


SCHEMA_VERSION = "flux_parquet_dataset_adapter_v1"
ARTIFACT_KIND = "flux_parquet_dataset_adapter"
DATASET_VERDICT_COMPLETE = "inventory-complete"
DATASET_VERDICT_MISSING = "inventory-missing"

REQUIRED_META_FILE_NAMES: tuple[str, ...] = (
    "info.json",
    "modality.json",
    "stats.json",
    "tasks.jsonl",
    "episodes.jsonl",
)


@dataclass(frozen=True)
class FluxParquetDatasetAdapter:
    dataset_dir: Path
    dataset_name: str
    dataset_source: dict[str, object]
    dataset_fingerprint: str | None
    stats_fingerprint: str | None
    prompt_source: dict[str, object]
    task_description_source: dict[str, object]
    action_state_normalization_source: dict[str, object]
    schema_compatibility: dict[str, object]
    dataset_bundle: dict[str, object] | None
    blocking_reasons: tuple[dict[str, str], ...]


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object at {path}, got {type(payload).__name__}"
        )
    return dict(cast(dict[str, Any], payload))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"expected JSON object line at {path}, got {type(payload).__name__}"
            )
        rows.append(dict(cast(dict[str, Any], payload)))
    return tuple(rows)


def _issue(code: str, field_path: str, message: str) -> dict[str, str]:
    return {
        "code": str(code),
        "field_path": str(field_path),
        "message": str(message),
    }


def _append_issue(
    issues: list[dict[str, str]],
    *,
    code: str,
    field_path: str,
    message: str,
) -> None:
    candidate = _issue(code, field_path, message)
    if candidate not in issues:
        issues.append(candidate)


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    return None


def _as_object_dict(payload: dict[str, object]) -> dict[str, object]:
    return dict(payload)


def _required_meta_paths(dataset_dir: Path) -> dict[str, Path]:
    meta_dir = dataset_dir / "meta"
    return {
        name: meta_dir / name
        for name in (
            *REQUIRED_META_FILE_NAMES,
            "dataset_fingerprint.json",
            "episode_universe_hash.txt",
            "episodes_stats.jsonl",
        )
    }


def _dataset_fingerprint(
    *,
    dataset_dir: Path,
    info_payload: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> str | None:
    meta_paths = _required_meta_paths(dataset_dir)
    fingerprint_path = meta_paths["dataset_fingerprint.json"]
    if fingerprint_path.is_file():
        try:
            fingerprint_payload = _read_json(fingerprint_path)
        except Exception as exc:
            _append_issue(
                issues,
                code="invalid_dataset_fingerprint",
                field_path="meta/dataset_fingerprint.json",
                message=str(exc),
            )
            return None
        fingerprint = _optional_text(fingerprint_payload.get("fingerprint_sha256"))
        if fingerprint is None:
            _append_issue(
                issues,
                code="missing_dataset_fingerprint",
                field_path="meta/dataset_fingerprint.json.fingerprint_sha256",
                message="dataset_fingerprint.json is missing fingerprint_sha256",
            )
        return fingerprint

    required_paths = (
        meta_paths["info.json"],
        meta_paths["tasks.jsonl"],
        meta_paths["episodes.jsonl"],
        meta_paths["stats.json"],
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        _append_issue(
            issues,
            code="missing_dataset_fingerprint_inputs",
            field_path="meta",
            message=(
                "dataset fingerprint fallback requires info.json, tasks.jsonl, "
                f"episodes.jsonl, and stats.json; missing {missing}"
            ),
        )
        return None
    route_id = _optional_text((info_payload or {}).get("route_id"))
    if route_id is None:
        _append_issue(
            issues,
            code="missing_dataset_route_id",
            field_path="meta/info.json.route_id",
            message="dataset info.json is missing route_id needed for provenance binding",
        )
    digest = hashlib.sha256()
    for path in required_paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_source_payload(
    *,
    dataset_dir: Path,
    info_payload: dict[str, Any] | None,
) -> dict[str, object]:
    meta_paths = _required_meta_paths(dataset_dir)
    parquet_files = tuple(sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet")))
    tasks_rows = (
        _read_jsonl(meta_paths["tasks.jsonl"])
        if meta_paths["tasks.jsonl"].is_file()
        else ()
    )
    episodes_rows = (
        _read_jsonl(meta_paths["episodes.jsonl"])
        if meta_paths["episodes.jsonl"].is_file()
        else ()
    )
    episodes_stats_rows = (
        _read_jsonl(meta_paths["episodes_stats.jsonl"])
        if meta_paths["episodes_stats.jsonl"].is_file()
        else ()
    )
    return {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "meta_dir": str(dataset_dir / "meta"),
        "data_dir": str(dataset_dir / "data"),
        "route_id": _optional_text((info_payload or {}).get("route_id")),
        "schema_version": _optional_text((info_payload or {}).get("schema_version")),
        "info_path": str(meta_paths["info.json"]),
        "modality_path": str(meta_paths["modality.json"]),
        "stats_path": str(meta_paths["stats.json"]),
        "tasks_path": str(meta_paths["tasks.jsonl"]),
        "episodes_path": str(meta_paths["episodes.jsonl"]),
        "episodes_stats_path": (
            str(meta_paths["episodes_stats.jsonl"])
            if meta_paths["episodes_stats.jsonl"].is_file()
            else None
        ),
        "parquet_files": [str(path.resolve()) for path in parquet_files],
        "parquet_file_count": len(parquet_files),
        "total_tasks": _optional_int((info_payload or {}).get("total_tasks"))
        or len(tasks_rows),
        "total_episodes": _optional_int((info_payload or {}).get("total_episodes"))
        or len(episodes_rows),
        "total_frames": _optional_int((info_payload or {}).get("total_frames")),
        "episodes_stats_row_count": len(episodes_stats_rows),
        "required_meta_files": [
            {
                "name": name,
                "path": str(meta_paths[name]),
                "present": meta_paths[name].is_file(),
            }
            for name in REQUIRED_META_FILE_NAMES
        ],
    }


def _schema_compatibility_payload(
    *,
    dataset_dir: Path,
    issues: list[dict[str, str]],
) -> dict[str, object]:
    try:
        mapping = build_phase1_dataset_mapping_spec(dataset_dir)
    except Exception as exc:
        _append_issue(
            issues,
            code="schema_incompatible",
            field_path="schema_compatibility",
            message=str(exc),
        )
        return {
            "status": "incompatible",
            "provenance_complete": False,
            "reason": str(exc),
        }
    return {
        "status": "compatible",
        "provenance_complete": True,
        "dataset_handoff_kind": mapping.dataset_handoff_kind,
        "source_image_feature_key": mapping.source_image_feature_key,
        "source_state_feature_key": mapping.source_state_feature_key,
        "source_action_feature_key": mapping.source_action_feature_key,
        "source_prompt_feature_key": mapping.source_prompt_feature_key,
        "openpi_primary_image_key": mapping.openpi_primary_image_key,
        "openpi_wrist_image_key": mapping.openpi_wrist_image_key,
        "openpi_state_key": mapping.openpi_state_key,
        "openpi_prompt_key": mapping.openpi_prompt_key,
        "image_bridge_mode": mapping.image_bridge_mode,
        "state_dim": int(mapping.state_dim),
        "action_dim": int(mapping.action_dim),
    }


def _dataset_bundle_payload(
    *,
    dataset_dir: Path,
    issues: list[dict[str, str]],
) -> dict[str, object] | None:
    try:
        bundle = resolve_recap_dataset(dataset_dir, preview_limit=1)
    except Exception as exc:
        _append_issue(
            issues,
            code="invalid_dataset_bundle",
            field_path="dataset_bundle",
            message=str(exc),
        )
        return None
    return dataset_bundle_to_dict(bundle)


def _prompt_source_payload(
    *,
    info_payload: dict[str, Any] | None,
    dataset_bundle: dict[str, object] | None,
    schema_compatibility: dict[str, object],
    issues: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, object]]:
    prompt_source_field = None
    prompt_route = None
    conditioning_mode = None
    if dataset_bundle is not None:
        prompt_source_field = _optional_text(dataset_bundle.get("source_prompt_field"))
        prompt_route = _optional_text(dataset_bundle.get("prompt_route"))
        conditioning_mode = _optional_text(dataset_bundle.get("conditioning_mode"))
    task_text_field = _optional_text((info_payload or {}).get("task_text_field"))
    carrier_route = _optional_text((info_payload or {}).get("carrier_route"))
    carrier_schema_version = _optional_text(
        (info_payload or {}).get("carrier_schema_version")
    )
    source_prompt_feature_key = _optional_text(
        schema_compatibility.get("source_prompt_feature_key")
    )
    provenance_complete = all(
        value is not None
        for value in (
            prompt_source_field,
            prompt_route,
            conditioning_mode,
            task_text_field,
            carrier_route,
            carrier_schema_version,
            source_prompt_feature_key,
        )
    )
    if not provenance_complete:
        _append_issue(
            issues,
            code="missing_prompt_provenance",
            field_path="task_description_source",
            message="dataset inventory requires prompt/task-description provenance bound to carrier_text_v1",
        )
    prompt_source = _as_object_dict(
        {
            "prompt_source_field": prompt_source_field,
            "prompt_route": prompt_route,
            "conditioning_mode": conditioning_mode,
            "provenance_complete": provenance_complete,
        }
    )
    task_description_source = _as_object_dict(
        {
            "task_text_field": task_text_field,
            "carrier_route": carrier_route,
            "carrier_schema_version": carrier_schema_version,
            "source_prompt_feature_key": source_prompt_feature_key,
            "prompt_source_field": prompt_source_field,
            "prompt_route": prompt_route,
            "conditioning_mode": conditioning_mode,
            "provenance_complete": provenance_complete,
        }
    )
    return prompt_source, task_description_source


def _action_state_normalization_payload(
    *,
    dataset_dir: Path,
    stats_fingerprint: str | None,
    issues: list[dict[str, str]],
) -> dict[str, object]:
    try:
        norm_policy = build_phase1_norm_policy(dataset_dir)
        norm_provenance = build_phase1_norm_provenance(norm_policy)
    except Exception as exc:
        _append_issue(
            issues,
            code="missing_action_state_normalization_source",
            field_path="action_state_normalization_source",
            message=str(exc),
        )
        return {
            "provenance_complete": False,
            "norm_stats_policy": None,
            "norm_stats_source": None,
            "norm_stats_path": None,
            "asset_id": None,
            "reference_checkpoint_asset_id": None,
            "stats_fingerprint": stats_fingerprint,
        }
    return {
        **norm_provenance,
        "stats_fingerprint": stats_fingerprint,
        "provenance_complete": stats_fingerprint is not None,
    }


def build_flux_parquet_dataset_adapter(
    dataset_dir: str | Path,
) -> FluxParquetDatasetAdapter:
    resolved_dataset_dir = Path(dataset_dir).expanduser().resolve()
    issues: list[dict[str, str]] = []
    if not resolved_dataset_dir.is_dir():
        _append_issue(
            issues,
            code="missing_dataset_dir",
            field_path="dataset_source.dataset_dir",
            message=f"dataset directory does not exist: {resolved_dataset_dir}",
        )
    meta_paths = _required_meta_paths(resolved_dataset_dir)
    info_payload: dict[str, Any] | None = None
    if meta_paths["info.json"].is_file():
        try:
            info_payload = _read_json(meta_paths["info.json"])
        except Exception as exc:
            _append_issue(
                issues,
                code="invalid_info_json",
                field_path="meta/info.json",
                message=str(exc),
            )
    else:
        _append_issue(
            issues,
            code="missing_info_json",
            field_path="meta/info.json",
            message=f"missing required dataset meta file: {meta_paths['info.json']}",
        )
    for required_meta_name in REQUIRED_META_FILE_NAMES:
        path = meta_paths[required_meta_name]
        if not path.is_file():
            _append_issue(
                issues,
                code="missing_required_meta_file",
                field_path=f"meta/{required_meta_name}",
                message=f"missing required dataset meta file: {path}",
            )
    dataset_source = _dataset_source_payload(
        dataset_dir=resolved_dataset_dir,
        info_payload=info_payload,
    )
    stats_path = meta_paths["stats.json"]
    stats_fingerprint = _sha256_file(stats_path) if stats_path.is_file() else None
    if stats_fingerprint is None:
        _append_issue(
            issues,
            code="missing_stats_fingerprint",
            field_path="stats_fingerprint",
            message="stats fingerprint requires meta/stats.json",
        )
    dataset_fingerprint = _dataset_fingerprint(
        dataset_dir=resolved_dataset_dir,
        info_payload=info_payload,
        issues=issues,
    )
    schema_compatibility = _schema_compatibility_payload(
        dataset_dir=resolved_dataset_dir,
        issues=issues,
    )
    dataset_bundle = _dataset_bundle_payload(
        dataset_dir=resolved_dataset_dir,
        issues=issues,
    )
    prompt_source, task_description_source = _prompt_source_payload(
        info_payload=info_payload,
        dataset_bundle=dataset_bundle,
        schema_compatibility=schema_compatibility,
        issues=issues,
    )
    action_state_normalization_source = _action_state_normalization_payload(
        dataset_dir=resolved_dataset_dir,
        stats_fingerprint=stats_fingerprint,
        issues=issues,
    )
    return FluxParquetDatasetAdapter(
        dataset_dir=resolved_dataset_dir,
        dataset_name=resolved_dataset_dir.name,
        dataset_source=dataset_source,
        dataset_fingerprint=dataset_fingerprint,
        stats_fingerprint=stats_fingerprint,
        prompt_source=prompt_source,
        task_description_source=task_description_source,
        action_state_normalization_source=action_state_normalization_source,
        schema_compatibility=schema_compatibility,
        dataset_bundle=dataset_bundle,
        blocking_reasons=tuple(issues),
    )


__all__ = [
    "ARTIFACT_KIND",
    "DATASET_VERDICT_COMPLETE",
    "DATASET_VERDICT_MISSING",
    "FluxParquetDatasetAdapter",
    "SCHEMA_VERSION",
    "build_flux_parquet_dataset_adapter",
]
