from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import errno
import json
import os
from pathlib import Path
import shutil
from typing import Any, cast

from .dataset import RecapDatasetBundle, dataset_bundle_to_dict
from .protocol import (
    CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
    FrozenComparisonManifest,
    RECAP_ONLY_VARIANT,
    REPO_ROOT,
    TRAIN_MANIFEST_SCHEMA_VERSION,
    build_stock_origin_payload,
    comparison_manifest_to_dict,
)


@dataclass(frozen=True)
class RecapCheckpointBundle:
    output_dir: Path
    checkpoint_dir: Path
    train_manifest_path: Path
    checkpoint_provenance_path: Path
    checkpoint_payload_path: Path


@dataclass(frozen=True)
class TrainCheckpointMetadata:
    variant_name: str
    dataset_route_id: str
    dataset_fingerprint: str
    episode_universe_hash: str
    base_checkpoint_id: str
    train_budget_id: str
    consumer_mode: str
    gate_eval_manifest_hash: str
    reuse_existing_checkpoint: bool
    reuse_verdict: str


LIBERO_ASSET_SUBDIR = Path("physical-intelligence") / "libero"
SERVEABLE_ARTIFACT_HARDLINK_MODE = "durable_file_hardlink_tree"
SERVEABLE_ARTIFACT_COPY_MODE = "durable_directory_copy"
SERVEABLE_ARTIFACT_MIXED_MODE = "durable_mixed_tree_materialization"
SERVEABLE_SOURCE_LAYOUT_SCHEMA_VERSION = "openpi_servable_checkpoint_source_layout_v1"


@dataclass(frozen=True)
class TreeMaterializationResult:
    mode: str
    same_filesystem: bool


def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _replace_tree_with_copy(src: Path, dst: Path) -> None:
    _remove_existing_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copytree(src.resolve(), dst)


def _share_filesystem(src: Path, dst_parent: Path) -> bool:
    return src.resolve().stat().st_dev == dst_parent.resolve().stat().st_dev


def _raise_unsupported_hardlink_entry(entry: Path) -> None:
    raise OSError(
        errno.EOPNOTSUPP,
        f"hardlink tree requires regular files and directories: {entry}",
    )


def _replace_tree_with_hardlinks(src: Path, dst: Path) -> None:
    resolved_src = src.resolve()
    _remove_existing_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)
    for root, dir_names, file_names in os.walk(resolved_src, topdown=True):
        root_path = Path(root)
        relative_root = root_path.relative_to(resolved_src)
        dst_root = dst if relative_root == Path(".") else dst / relative_root
        for dir_name in dir_names:
            source_dir = root_path / dir_name
            if source_dir.is_symlink() or not source_dir.is_dir():
                _raise_unsupported_hardlink_entry(source_dir)
            (dst_root / dir_name).mkdir(parents=True, exist_ok=True)
        for file_name in file_names:
            source_file = root_path / file_name
            if source_file.is_symlink() or not source_file.is_file():
                _raise_unsupported_hardlink_entry(source_file)
            os.link(source_file, dst_root / file_name)


def _replace_tree_with_durable_links_or_copy(
    src: Path,
    dst: Path,
) -> TreeMaterializationResult:
    dst.parent.mkdir(parents=True, exist_ok=True)
    same_filesystem = _share_filesystem(src, dst.parent)
    if same_filesystem:
        try:
            _replace_tree_with_hardlinks(src, dst)
        except OSError:
            _remove_existing_path(dst)
        else:
            return TreeMaterializationResult(
                mode=SERVEABLE_ARTIFACT_HARDLINK_MODE,
                same_filesystem=True,
            )
    _replace_tree_with_copy(src, dst)
    return TreeMaterializationResult(
        mode=SERVEABLE_ARTIFACT_COPY_MODE,
        same_filesystem=same_filesystem,
    )


def _require_servable_checkpoint_layout(checkpoint_dir: Path) -> Path:
    required_paths = (
        checkpoint_dir / "params" / "_METADATA",
        checkpoint_dir / "assets" / LIBERO_ASSET_SUBDIR / "norm_stats.json",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "servable checkpoint source is incomplete: " + ", ".join(missing)
        )
    return checkpoint_dir.resolve()


def _materialize_servable_checkpoint_assets(
    checkpoint_dir: Path, *, source_checkpoint_dir: Path
) -> tuple[TreeMaterializationResult, TreeMaterializationResult]:
    resolved_source = _require_servable_checkpoint_layout(source_checkpoint_dir)
    params_result = _replace_tree_with_durable_links_or_copy(
        resolved_source / "params",
        checkpoint_dir / "params",
    )
    assets_result = _replace_tree_with_durable_links_or_copy(
        resolved_source / "assets" / LIBERO_ASSET_SUBDIR,
        checkpoint_dir / "assets" / LIBERO_ASSET_SUBDIR,
    )
    return params_result, assets_result


def _describe_tree_layout(
    path: Path,
    *,
    materialization_result: TreeMaterializationResult | None = None,
) -> dict[str, object]:
    layout: dict[str, object] = {
        "path": str(path),
        "present": bool(path.exists() or path.is_symlink()),
        "is_symlink": bool(path.is_symlink()),
        "resolved_path": str(path.resolve(strict=False)),
    }
    if materialization_result is not None:
        layout["materialization_mode"] = materialization_result.mode
        layout["same_filesystem_as_source"] = materialization_result.same_filesystem
    return layout


def _infer_tree_layout_mode(*, layouts: tuple[dict[str, object], ...]) -> str:
    if any(bool(layout.get("is_symlink", False)) for layout in layouts):
        return "directory_symlink"
    return "directory_copy"


def _infer_bundle_layout_mode(
    *,
    materialization_results: tuple[TreeMaterializationResult, ...],
) -> str:
    modes = {result.mode for result in materialization_results}
    if len(modes) == 1:
        return next(iter(modes))
    return SERVEABLE_ARTIFACT_MIXED_MODE


def _build_servable_source_layout(
    *,
    checkpoint_dir: Path,
    source_checkpoint_dir: Path,
    params_result: TreeMaterializationResult,
    assets_result: TreeMaterializationResult,
) -> dict[str, object]:
    source_params_layout = _describe_tree_layout(source_checkpoint_dir / "params")
    source_assets_layout = _describe_tree_layout(
        source_checkpoint_dir / "assets" / LIBERO_ASSET_SUBDIR
    )
    bundle_params_layout = _describe_tree_layout(
        checkpoint_dir / "params",
        materialization_result=params_result,
    )
    bundle_assets_layout = _describe_tree_layout(
        checkpoint_dir / "assets" / LIBERO_ASSET_SUBDIR,
        materialization_result=assets_result,
    )
    return {
        "schema_version": SERVEABLE_SOURCE_LAYOUT_SCHEMA_VERSION,
        "source_checkpoint_dir": str(source_checkpoint_dir),
        "source_layout_mode": _infer_tree_layout_mode(
            layouts=(source_params_layout, source_assets_layout)
        ),
        "source_params_layout": source_params_layout,
        "source_libero_assets_layout": source_assets_layout,
        "bundle_checkpoint_dir": str(checkpoint_dir),
        "bundle_layout_mode": _infer_bundle_layout_mode(
            materialization_results=(params_result, assets_result)
        ),
        "bundle_params_layout": bundle_params_layout,
        "bundle_libero_assets_layout": bundle_assets_layout,
    }


def _build_servable_artifact_payload(
    *,
    checkpoint_dir: Path,
    source_checkpoint_dir: Path,
    params_result: TreeMaterializationResult,
    assets_result: TreeMaterializationResult,
) -> dict[str, object]:
    artifact_mirror_mode = _infer_bundle_layout_mode(
        materialization_results=(params_result, assets_result)
    )
    return {
        "artifact_mirror_mode": artifact_mirror_mode,
        "servable_source_layout": _build_servable_source_layout(
            checkpoint_dir=checkpoint_dir,
            source_checkpoint_dir=source_checkpoint_dir,
            params_result=params_result,
            assets_result=assets_result,
        ),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}, got {type(data).__name__}")
    return {str(key): value for key, value in cast(dict[object, object], data).items()}


def checkpoint_metadata_to_dict(
    metadata: TrainCheckpointMetadata,
) -> dict[str, object]:
    return {
        "variant_name": metadata.variant_name,
        "dataset_route_id": metadata.dataset_route_id,
        "dataset_fingerprint": metadata.dataset_fingerprint,
        "episode_universe_hash": metadata.episode_universe_hash,
        "base_checkpoint_id": metadata.base_checkpoint_id,
        "train_budget_id": metadata.train_budget_id,
        "consumer_mode": metadata.consumer_mode,
        "gate_eval_manifest_hash": metadata.gate_eval_manifest_hash,
        "reuse_existing_checkpoint": bool(metadata.reuse_existing_checkpoint),
        "reuse_verdict": metadata.reuse_verdict,
    }


def _preview_mapping(dataset_bundle: RecapDatasetBundle) -> dict[str, object]:
    if not dataset_bundle.record_preview:
        return {}
    preview = dataset_bundle.record_preview[0]
    return {str(key): value for key, value in preview.items()}


def _preview_bool(dataset_bundle: RecapDatasetBundle, field_name: str) -> bool:
    raw = _preview_mapping(dataset_bundle).get(field_name, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() == "true"


def _preview_text(dataset_bundle: RecapDatasetBundle, field_name: str) -> str:
    raw = _preview_mapping(dataset_bundle).get(field_name, "")
    return str(raw).strip()


def _critic_contract(dataset_bundle: RecapDatasetBundle) -> dict[str, object]:
    raw = dataset_bundle.recap_contract
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}


def _value_source(dataset_bundle: RecapDatasetBundle) -> str:
    contract = _critic_contract(dataset_bundle)
    return str(contract.get("value_source", "baseline")).strip() or "baseline"


def _critic_checkpoint_ref(dataset_bundle: RecapDatasetBundle) -> str:
    contract = _critic_contract(dataset_bundle)
    for key in ("critic_checkpoint_ref", "critic_dir"):
        value = str(contract.get(key, "")).strip()
        if value:
            return value
    return "not_applicable"


def _critic_metrics_ref(dataset_bundle: RecapDatasetBundle) -> str | None:
    value = str(_critic_contract(dataset_bundle).get("critic_metrics_path", "")).strip()
    return value or None


def _critic_provenance_ref(dataset_bundle: RecapDatasetBundle) -> str | None:
    value = str(
        _critic_contract(dataset_bundle).get("critic_provenance_path", "")
    ).strip()
    return value or None


def _public_value_scale(dataset_bundle: RecapDatasetBundle) -> str:
    contract = _critic_contract(dataset_bundle)
    return str(contract.get("value_scale", "raw_return")).strip() or "raw_return"


def _value_adapter_name(dataset_bundle: RecapDatasetBundle) -> str | None:
    value = str(_critic_contract(dataset_bundle).get("value_adapter", "")).strip()
    return value or None


def build_train_manifest(
    *,
    dataset_bundle: RecapDatasetBundle,
    manifest: FrozenComparisonManifest,
    output_dir: Path,
    variant: str = RECAP_ONLY_VARIANT,
    train_metadata: TrainCheckpointMetadata | None = None,
) -> dict[str, object]:
    dataset_payload = dataset_bundle_to_dict(dataset_bundle)
    payload: dict[str, object] = {
        "schema_version": TRAIN_MANIFEST_SCHEMA_VERSION,
        "variant": variant,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_dir": str((output_dir / "best").resolve()),
        "comparison_manifest": comparison_manifest_to_dict(manifest),
        "train_source": dataset_payload,
        "training_route": {
            "carrier": "prompt_text_only",
            "prompt_route": dataset_bundle.prompt_route,
            "conditioning_mode": dataset_bundle.conditioning_mode,
            "source_prompt_field": dataset_bundle.source_prompt_field,
            "consumer_mode": dataset_bundle.consumer_mode,
            "fixed_indicator_mode": dataset_bundle.fixed_indicator_mode,
            "indicator_source": _preview_text(dataset_bundle, "indicator_source"),
            "prompt_text_surface": _preview_text(dataset_bundle, "prompt_text_surface"),
            "per_sample_indicator_consumption": _preview_bool(
                dataset_bundle, "per_sample_indicator_consumption"
            ),
            "prompt_conditioned_dependency": _preview_bool(
                dataset_bundle, "prompt_conditioned_dependency"
            ),
            "advantage_input_dependency": _preview_bool(
                dataset_bundle, "advantage_input_dependency"
            ),
            "value_source": _value_source(dataset_bundle),
            "value_scale": _public_value_scale(dataset_bundle),
            "critic_checkpoint_ref": _critic_checkpoint_ref(dataset_bundle),
            "critic_metrics_ref": _critic_metrics_ref(dataset_bundle),
            "critic_provenance_ref": _critic_provenance_ref(dataset_bundle),
            "value_adapter": _value_adapter_name(dataset_bundle),
            "offline_labels_only": True,
            "value_head": False,
            "online_loop": False,
        },
        "critic_checkpoint_ref": _critic_checkpoint_ref(dataset_bundle),
    }
    if train_metadata is not None:
        payload.update(checkpoint_metadata_to_dict(train_metadata))
    return payload


def build_checkpoint_provenance(
    *,
    dataset_bundle: RecapDatasetBundle,
    manifest: FrozenComparisonManifest,
    checkpoint_dir: Path,
    train_manifest_path: Path,
    variant: str = RECAP_ONLY_VARIANT,
    checkpoint_source: str = "repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
    train_metadata: TrainCheckpointMetadata | None = None,
    serveable_artifact_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "variant": variant,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_source": checkpoint_source,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stock_baseline_origin": build_stock_origin_payload(),
        "variant_derivation": {
            "source_dataset_dir": str(dataset_bundle.dataset_dir),
            "source_dataset_name": dataset_bundle.dataset_name,
            "train_manifest": str(train_manifest_path.resolve()),
            "suite": manifest.suite,
            "task_ids": [int(value) for value in manifest.task_ids],
            "seed_manifest": [int(value) for value in manifest.seed_manifest],
            "num_trials_per_task": int(manifest.num_trials_per_task),
            "prompt_route": dataset_bundle.prompt_route,
            "conditioning_mode": dataset_bundle.conditioning_mode,
            "source_prompt_field": dataset_bundle.source_prompt_field,
            "consumer_mode": dataset_bundle.consumer_mode,
            "fixed_indicator_mode": dataset_bundle.fixed_indicator_mode,
            "indicator_source": _preview_text(dataset_bundle, "indicator_source"),
            "prompt_text_surface": _preview_text(dataset_bundle, "prompt_text_surface"),
            "per_sample_indicator_consumption": _preview_bool(
                dataset_bundle, "per_sample_indicator_consumption"
            ),
            "prompt_conditioned_dependency": _preview_bool(
                dataset_bundle, "prompt_conditioned_dependency"
            ),
            "advantage_input_dependency": _preview_bool(
                dataset_bundle, "advantage_input_dependency"
            ),
            "value_source": _value_source(dataset_bundle),
            "value_scale": _public_value_scale(dataset_bundle),
            "critic_checkpoint_ref": _critic_checkpoint_ref(dataset_bundle),
            "critic_metrics_ref": _critic_metrics_ref(dataset_bundle),
            "critic_provenance_ref": _critic_provenance_ref(dataset_bundle),
            "value_adapter": _value_adapter_name(dataset_bundle),
            "offline_labels": [
                "recap_m2.return_G",
                "recap_m2.value_V",
                "recap_m2.advantage_A",
                "recap_m2.advantage_input",
                "recap_m2.indicator_I",
            ],
            "no_value_head": True,
            "no_online_loop": True,
            "no_state_tokens": True,
        },
        "critic_checkpoint_ref": _critic_checkpoint_ref(dataset_bundle),
    }
    if train_metadata is not None:
        payload.update(checkpoint_metadata_to_dict(train_metadata))
    if serveable_artifact_payload is not None:
        payload.update(serveable_artifact_payload)
    return payload


def materialize_recap_checkpoint(
    *,
    output_dir: str | Path,
    dataset_bundle: RecapDatasetBundle,
    manifest: FrozenComparisonManifest,
    variant: str = RECAP_ONLY_VARIANT,
    checkpoint_source: str = "repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
    train_metadata: TrainCheckpointMetadata | None = None,
    serveable_checkpoint_source_dir: str | Path | None = None,
) -> RecapCheckpointBundle:
    output_dir_path = Path(output_dir).resolve()
    checkpoint_dir = output_dir_path / "best"
    train_manifest_path = output_dir_path / "train_manifest.json"
    checkpoint_provenance_path = output_dir_path / "checkpoint_provenance.json"
    checkpoint_payload_path = checkpoint_dir / "checkpoint.json"

    serveable_artifact_payload: dict[str, object] | None = None
    if serveable_checkpoint_source_dir is not None:
        resolved_source_checkpoint_dir = Path(serveable_checkpoint_source_dir).resolve()
        params_result, assets_result = _materialize_servable_checkpoint_assets(
            checkpoint_dir,
            source_checkpoint_dir=resolved_source_checkpoint_dir,
        )
        serveable_artifact_payload = _build_servable_artifact_payload(
            checkpoint_dir=checkpoint_dir,
            source_checkpoint_dir=resolved_source_checkpoint_dir,
            params_result=params_result,
            assets_result=assets_result,
        )

    train_manifest = build_train_manifest(
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        output_dir=output_dir_path,
        variant=variant,
        train_metadata=train_metadata,
    )
    checkpoint_provenance = build_checkpoint_provenance(
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        checkpoint_dir=checkpoint_dir,
        train_manifest_path=train_manifest_path,
        variant=variant,
        checkpoint_source=checkpoint_source,
        train_metadata=train_metadata,
        serveable_artifact_payload=serveable_artifact_payload,
    )
    checkpoint_payload: dict[str, object] = {
        "schema_version": "openpi_libero_recap_checkpoint_payload_v1",
        "variant": variant,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": str(checkpoint_dir),
        "train_manifest": str(train_manifest_path),
        "checkpoint_provenance": str(checkpoint_provenance_path),
        "offline_success_proxy": float(dataset_bundle.indicator_positive_fraction),
        "offline_failure_proxy": max(
            0.0, 1.0 - float(dataset_bundle.indicator_positive_fraction)
        ),
        "record_count": int(dataset_bundle.total_rows),
    }
    if serveable_artifact_payload is not None:
        checkpoint_payload.update(serveable_artifact_payload)
    write_json(train_manifest_path, train_manifest)
    write_json(checkpoint_provenance_path, checkpoint_provenance)
    write_json(checkpoint_dir / "train_manifest.json", train_manifest)
    write_json(checkpoint_dir / "checkpoint_provenance.json", checkpoint_provenance)
    write_json(checkpoint_payload_path, checkpoint_payload)
    return RecapCheckpointBundle(
        output_dir=output_dir_path,
        checkpoint_dir=checkpoint_dir,
        train_manifest_path=train_manifest_path,
        checkpoint_provenance_path=checkpoint_provenance_path,
        checkpoint_payload_path=checkpoint_payload_path,
    )


__all__ = [
    "RecapCheckpointBundle",
    "TrainCheckpointMetadata",
    "build_checkpoint_provenance",
    "build_train_manifest",
    "checkpoint_metadata_to_dict",
    "materialize_recap_checkpoint",
    "read_json",
    "write_json",
]
