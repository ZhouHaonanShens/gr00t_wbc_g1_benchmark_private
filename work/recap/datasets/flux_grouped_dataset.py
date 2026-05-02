from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_ABSOLUTE_ACTION_KEYS
from work.recap.scripts.gr00t_eval_contract_gate import (
    DEFAULT_ACTION_REPRESENTATION_BY_KEY,
)
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_CAMERA_CONFIG
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_N_ACTION_STEPS
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_POLICY_HORIZON_EXPECTED
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_RELATIVE_ACTION_KEYS
from work.recap.scripts.gr00t_eval_contract_gate import build_eval_contract_freeze

from .flux_parquet_dataset import ARTIFACT_KIND as ADAPTER_ARTIFACT_KIND
from .flux_parquet_dataset import FluxParquetDatasetAdapter
from .flux_parquet_dataset import SCHEMA_VERSION as ADAPTER_SCHEMA_VERSION
from .flux_parquet_dataset import build_flux_parquet_dataset_adapter


SCHEMA_VERSION = "flux_dataset_inventory_bundle_v1"
ARTIFACT_KIND = "flux_dataset_inventory_bundle"
VERDICT_COMPLETE = "inventory-complete"
VERDICT_MISSING = "inventory-missing"


@dataclass(frozen=True)
class FluxDatasetInventoryBundle:
    dataset_dir: Path
    verdict: str
    dataset_source: dict[str, object]
    dataset_fingerprint: str | None
    stats_fingerprint: str | None
    prompt_source: dict[str, object]
    task_description_source: dict[str, object]
    camera_inventory: dict[str, object]
    action_state_normalization_source: dict[str, object]
    schema_compatibility: dict[str, object]
    grouped_stats: dict[str, object]
    binding_join_contract: dict[str, object]
    dataset_adapter: dict[str, object]
    blocking_reasons: tuple[dict[str, str], ...]


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_signature(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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


def _video_entries(dataset_dir: Path) -> dict[str, dict[str, object]]:
    modality_path = dataset_dir / "meta" / "modality.json"
    if not modality_path.is_file():
        raise FileNotFoundError(modality_path)
    payload = _read_json(modality_path)
    raw_video = payload.get("video")
    if not isinstance(raw_video, dict):
        raise ValueError("modality.json missing video modality mapping")
    return {
        str(key): dict(cast(dict[str, object], value))
        for key, value in raw_video.items()
        if isinstance(value, dict)
    }


def _camera_inventory(
    *,
    adapter: FluxParquetDatasetAdapter,
    issues: list[dict[str, str]],
) -> dict[str, object]:
    try:
        video_entries = _video_entries(adapter.dataset_dir)
    except Exception as exc:
        _append_issue(
            issues,
            code="missing_camera_inventory",
            field_path="camera_inventory",
            message=str(exc),
        )
        return {
            "provenance_complete": False,
            "video_feature_keys": [],
            "original_observation_keys": [],
            "view_count": 0,
            "expected_eval_camera_config": dict(DEFAULT_CAMERA_CONFIG),
            "image_bridge_mode": adapter.schema_compatibility.get("image_bridge_mode"),
        }
    original_keys = sorted(
        {
            str(value.get("original_key", key)).strip()
            for key, value in video_entries.items()
            if str(value.get("original_key", key)).strip()
        }
    )
    view_count = len(video_entries)
    provenance_complete = bool(video_entries)
    if not provenance_complete:
        _append_issue(
            issues,
            code="missing_camera_inventory",
            field_path="camera_inventory.video_feature_keys",
            message="dataset modality.json did not declare any video entries",
        )
    return {
        "provenance_complete": provenance_complete,
        "video_feature_keys": sorted(video_entries.keys()),
        "original_observation_keys": original_keys,
        "view_count": view_count,
        "expected_eval_camera_config": dict(DEFAULT_CAMERA_CONFIG),
        "image_bridge_mode": adapter.schema_compatibility.get("image_bridge_mode"),
    }


def _grouped_stats(
    *,
    adapter: FluxParquetDatasetAdapter,
) -> dict[str, object]:
    meta_dir = adapter.dataset_dir / "meta"
    info_payload = (
        _read_json(meta_dir / "info.json") if (meta_dir / "info.json").is_file() else {}
    )
    tasks_rows = (
        _read_jsonl(meta_dir / "tasks.jsonl")
        if (meta_dir / "tasks.jsonl").is_file()
        else ()
    )
    episodes_rows = (
        _read_jsonl(meta_dir / "episodes.jsonl")
        if (meta_dir / "episodes.jsonl").is_file()
        else ()
    )
    episode_stats_rows = (
        _read_jsonl(meta_dir / "episodes_stats.jsonl")
        if (meta_dir / "episodes_stats.jsonl").is_file()
        else ()
    )
    dataset_bundle = adapter.dataset_bundle or {}
    return {
        "task_row_count": len(tasks_rows),
        "episode_row_count": len(episodes_rows),
        "episode_stats_row_count": len(episode_stats_rows),
        "parquet_file_count": _optional_int(
            adapter.dataset_source.get("parquet_file_count")
        )
        or 0,
        "total_tasks": _optional_int(adapter.dataset_source.get("total_tasks"))
        or len(tasks_rows),
        "total_episodes": _optional_int(adapter.dataset_source.get("total_episodes"))
        or len(episodes_rows),
        "total_frames": int(
            _optional_int(adapter.dataset_source.get("total_frames"))
            or _optional_int(dataset_bundle.get("total_rows"))
            or 0
        ),
        "fps": info_payload.get("fps"),
    }


def _expected_action_space_contract(
    *,
    adapter: FluxParquetDatasetAdapter,
) -> dict[str, object]:
    eval_contract = build_eval_contract_freeze()
    server_contract = cast(dict[str, object], eval_contract["server_contract"])
    return {
        "expected_embodiment_tag": str(server_contract["embodiment_tag"]),
        "policy_horizon_expected": int(DEFAULT_POLICY_HORIZON_EXPECTED),
        "n_action_steps": int(DEFAULT_N_ACTION_STEPS),
        "state_dim": adapter.schema_compatibility.get("state_dim"),
        "action_dim": adapter.schema_compatibility.get("action_dim"),
        "relative_action_keys": list(DEFAULT_RELATIVE_ACTION_KEYS),
        "absolute_action_keys": list(DEFAULT_ABSOLUTE_ACTION_KEYS),
        "action_representation_by_key": dict(DEFAULT_ACTION_REPRESENTATION_BY_KEY),
    }


def _binding_join_contract(
    *,
    adapter: FluxParquetDatasetAdapter,
    issues: list[dict[str, str]],
) -> dict[str, object]:
    expected_action_space_contract = _expected_action_space_contract(adapter=adapter)
    prompt_source = adapter.prompt_source.get("prompt_source_field")
    norm_stats_source = adapter.action_state_normalization_source.get(
        "norm_stats_source"
    )
    action_state_norm_source = adapter.action_state_normalization_source.get(
        "norm_stats_policy"
    )
    expected_embodiment_tag = expected_action_space_contract["expected_embodiment_tag"]
    contract = {
        "dataset_fingerprint": adapter.dataset_fingerprint,
        "prompt_source": prompt_source,
        "norm_stats_source": norm_stats_source,
        "action_state_norm_source": action_state_norm_source,
        "expected_embodiment_tag": expected_embodiment_tag,
        "expected_action_space_signature": _stable_signature(
            expected_action_space_contract
        ),
        "expected_action_space_contract": expected_action_space_contract,
    }
    required_paths = {
        "dataset_fingerprint": contract["dataset_fingerprint"],
        "prompt_source": contract["prompt_source"],
        "norm_stats_source": contract["norm_stats_source"],
        "action_state_norm_source": contract["action_state_norm_source"],
        "expected_embodiment_tag": contract["expected_embodiment_tag"],
        "expected_action_space_signature": contract["expected_action_space_signature"],
    }
    for field_name, value in required_paths.items():
        if value in (None, ""):
            _append_issue(
                issues,
                code="missing_binding_join_contract_field",
                field_path=f"binding_join_contract.{field_name}",
                message=f"binding_join_contract.{field_name} is required",
            )
    return contract


def build_flux_dataset_inventory_bundle(
    dataset_dir: str | Path,
) -> FluxDatasetInventoryBundle:
    adapter = build_flux_parquet_dataset_adapter(dataset_dir)
    issues = list(adapter.blocking_reasons)
    camera_inventory = _camera_inventory(adapter=adapter, issues=issues)
    grouped_stats = _grouped_stats(adapter=adapter)
    binding_join_contract = _binding_join_contract(adapter=adapter, issues=issues)
    verdict = VERDICT_COMPLETE if not issues else VERDICT_MISSING
    dataset_adapter: dict[str, object] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "artifact_kind": ADAPTER_ARTIFACT_KIND,
        "dataset_bundle_present": adapter.dataset_bundle is not None,
        "blocking_reason_count": len(issues),
    }
    return FluxDatasetInventoryBundle(
        dataset_dir=adapter.dataset_dir,
        verdict=verdict,
        dataset_source=dict(adapter.dataset_source),
        dataset_fingerprint=adapter.dataset_fingerprint,
        stats_fingerprint=adapter.stats_fingerprint,
        prompt_source=dict(adapter.prompt_source),
        task_description_source=dict(adapter.task_description_source),
        camera_inventory=camera_inventory,
        action_state_normalization_source=dict(
            adapter.action_state_normalization_source
        ),
        schema_compatibility=dict(adapter.schema_compatibility),
        grouped_stats=grouped_stats,
        binding_join_contract=binding_join_contract,
        dataset_adapter=dataset_adapter,
        blocking_reasons=tuple(issues),
    )


def inventory_bundle_to_dict(
    bundle: FluxDatasetInventoryBundle,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "dataset_dir": str(bundle.dataset_dir),
        "dataset_name": bundle.dataset_dir.name,
        "verdict": bundle.verdict,
        "dataset_source": dict(bundle.dataset_source),
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "stats_fingerprint": bundle.stats_fingerprint,
        "prompt_source": dict(bundle.prompt_source),
        "task_description_source": dict(bundle.task_description_source),
        "camera_inventory": dict(bundle.camera_inventory),
        "action_state_normalization_source": dict(
            bundle.action_state_normalization_source
        ),
        "schema_compatibility": dict(bundle.schema_compatibility),
        "grouped_stats": dict(bundle.grouped_stats),
        "binding_join_contract": dict(bundle.binding_join_contract),
        "dataset_adapter": dict(bundle.dataset_adapter),
        "blocking_reasons": [dict(reason) for reason in bundle.blocking_reasons],
    }


__all__ = [
    "ARTIFACT_KIND",
    "FluxDatasetInventoryBundle",
    "SCHEMA_VERSION",
    "VERDICT_COMPLETE",
    "VERDICT_MISSING",
    "build_flux_dataset_inventory_bundle",
    "inventory_bundle_to_dict",
]
