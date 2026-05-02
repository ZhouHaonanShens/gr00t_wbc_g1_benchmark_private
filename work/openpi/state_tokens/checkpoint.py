from __future__ import annotations

import datetime as dt
from pathlib import Path

from work.openpi.recap.checkpoint import (
    RecapCheckpointBundle,
    TrainCheckpointMetadata,
    checkpoint_metadata_to_dict,
    write_json,
)
from work.openpi.recap.protocol import (
    FrozenComparisonManifest,
    REPO_ROOT,
    build_stock_origin_payload,
    comparison_manifest_to_dict,
)

from .dataset import StateTokenDatasetBundle, dataset_bundle_to_dict
from .protocol import (
    CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
    OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
    REQUIRED_NATIVE_STATE_DIM,
    RECAP_STATE_TOKENS_VARIANT,
    SOURCE_STATE,
    SOURCE_STATE_PADDING,
    STATE_TOKEN_ROUTE,
    STATE_TOKEN_SEMANTICS,
    TRAIN_MANIFEST_SCHEMA_VERSION,
    TRANSFORM_ORDER,
)


def _require_native_state_dim(dataset_bundle: StateTokenDatasetBundle) -> None:
    observed = int(dataset_bundle.observed_dataset_state_dim)
    if observed != REQUIRED_NATIVE_STATE_DIM:
        raise ValueError(
            "native_discrete_state_input_v1 requires observed_dataset_state_dim=8, "
            + f"got {observed!r}"
        )


def _preview_mapping(dataset_bundle: StateTokenDatasetBundle) -> dict[str, object]:
    if not dataset_bundle.recap_bundle.record_preview:
        return {}
    preview = dataset_bundle.recap_bundle.record_preview[0]
    return {str(key): value for key, value in preview.items()}


def _preview_bool(dataset_bundle: StateTokenDatasetBundle, field_name: str) -> bool:
    raw = _preview_mapping(dataset_bundle).get(field_name, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() == "true"


def _preview_text(dataset_bundle: StateTokenDatasetBundle, field_name: str) -> str:
    raw = _preview_mapping(dataset_bundle).get(field_name, "")
    return str(raw).strip()


def build_train_manifest(
    *,
    dataset_bundle: StateTokenDatasetBundle,
    manifest: FrozenComparisonManifest,
    output_dir: Path,
    train_metadata: TrainCheckpointMetadata | None = None,
) -> dict[str, object]:
    _require_native_state_dim(dataset_bundle)
    source_bundle = dataset_bundle.source_bundle
    recap_bundle = dataset_bundle.recap_bundle
    payload: dict[str, object] = {
        "schema_version": TRAIN_MANIFEST_SCHEMA_VERSION,
        "variant": RECAP_STATE_TOKENS_VARIANT,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_dir": str((output_dir / "best").resolve()),
        "comparison_manifest": comparison_manifest_to_dict(manifest),
        "train_source": dataset_bundle_to_dict(dataset_bundle),
        "training_route": {
            "base_variant": "recap_only",
            "only_added_experimental_variable": "native discrete_state_input=True",
            "source_dataset_dir": str(source_bundle.dataset_dir),
            "source_dataset_name": source_bundle.dataset_name,
            "source_dataset_route_id": source_bundle.route_id,
            "source_dataset_schema_version": source_bundle.schema_version,
            "official_native_source_dataset_dir": str(source_bundle.source_dataset_dir),
            "official_native_source_dataset_name": source_bundle.source_dataset_name,
            "prompt_route": recap_bundle.prompt_route,
            "conditioning_mode": recap_bundle.conditioning_mode,
            "source_prompt_field": recap_bundle.source_prompt_field,
            "consumer_mode": recap_bundle.consumer_mode,
            "fixed_indicator_mode": recap_bundle.fixed_indicator_mode,
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
            "offline_labels_only": True,
            "value_head": False,
            "online_loop": False,
            "discrete_state_input": True,
            "state_token_route": dataset_bundle.state_token_route,
            "state_token_semantics": dataset_bundle.state_token_semantics,
            "source_state": dataset_bundle.source_state,
            "source_state_padding": dataset_bundle.source_state_padding,
            "transform_order": dataset_bundle.transform_order,
            "contract_state_dim": REQUIRED_NATIVE_STATE_DIM,
            "observed_dataset_state_dim": int(
                dataset_bundle.observed_dataset_state_dim
            ),
            "source_action_dim": int(source_bundle.action_dim),
            "source_total_tasks": int(source_bundle.total_tasks),
            "source_total_episodes": int(source_bundle.total_episodes),
            "aligned_record_count": int(dataset_bundle.aligned_record_count),
            "recap_label_dataset_dir": str(recap_bundle.dataset_dir),
            "recap_label_dataset_name": recap_bundle.dataset_name,
            "recap_label_route_id": OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
            "no_second_tokenizer": True,
            "no_custom_vocabulary": True,
            "no_symbolic_phase_token": True,
            "no_task_phase_id": True,
            "no_rl_token": True,
            "no_next_state_head": True,
        },
    }
    if train_metadata is not None:
        payload.update(checkpoint_metadata_to_dict(train_metadata))
    return payload


def build_checkpoint_provenance(
    *,
    dataset_bundle: StateTokenDatasetBundle,
    manifest: FrozenComparisonManifest,
    checkpoint_dir: Path,
    train_manifest_path: Path,
    train_metadata: TrainCheckpointMetadata | None = None,
) -> dict[str, object]:
    _require_native_state_dim(dataset_bundle)
    source_bundle = dataset_bundle.source_bundle
    recap_bundle = dataset_bundle.recap_bundle
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "variant": RECAP_STATE_TOKENS_VARIANT,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_source": "repo_local_openpi_recap_state_tokens_native_discrete_state_input_v1",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "state_token_route": dataset_bundle.state_token_route,
        "stock_baseline_origin": build_stock_origin_payload(),
        "variant_derivation": {
            "base_variant": "recap_only",
            "only_added_experimental_variable": "native discrete_state_input=True",
            "source_dataset_dir": str(source_bundle.dataset_dir),
            "source_dataset_name": source_bundle.dataset_name,
            "source_dataset_route_id": source_bundle.route_id,
            "source_dataset_schema_version": source_bundle.schema_version,
            "official_native_source_dataset_dir": str(source_bundle.source_dataset_dir),
            "official_native_source_dataset_name": source_bundle.source_dataset_name,
            "source_total_episodes": int(source_bundle.total_episodes),
            "source_total_frames": int(source_bundle.total_frames),
            "source_total_tasks": int(source_bundle.total_tasks),
            "source_action_dim": int(source_bundle.action_dim),
            "train_manifest": str(train_manifest_path.resolve()),
            "recap_label_dataset_dir": str(recap_bundle.dataset_dir),
            "recap_label_dataset_name": recap_bundle.dataset_name,
            "recap_label_route_id": OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
            "suite": manifest.suite,
            "task_ids": [int(value) for value in manifest.task_ids],
            "seed_manifest": [int(value) for value in manifest.seed_manifest],
            "num_trials_per_task": int(manifest.num_trials_per_task),
            "prompt_route": recap_bundle.prompt_route,
            "conditioning_mode": recap_bundle.conditioning_mode,
            "source_prompt_field": recap_bundle.source_prompt_field,
            "consumer_mode": recap_bundle.consumer_mode,
            "fixed_indicator_mode": recap_bundle.fixed_indicator_mode,
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
            "offline_labels": [
                "recap_m2.return_G",
                "recap_m2.value_V",
                "recap_m2.advantage_A",
                "recap_m2.advantage_input",
                "recap_m2.indicator_I",
            ],
            "discrete_state_input": True,
            "state_token_route": dataset_bundle.state_token_route,
            "state_token_semantics": dataset_bundle.state_token_semantics,
            "source_state": dataset_bundle.source_state,
            "source_state_padding": dataset_bundle.source_state_padding,
            "transform_order": dataset_bundle.transform_order,
            "contract_state_dim": REQUIRED_NATIVE_STATE_DIM,
            "observed_dataset_state_dim": int(
                dataset_bundle.observed_dataset_state_dim
            ),
            "no_value_head": True,
            "no_online_loop": True,
            "no_symbolic_phase_token": True,
            "no_task_phase_id": True,
            "no_rl_token": True,
            "no_next_state_head": True,
            "no_custom_token_vocabulary": True,
            "no_second_tokenizer": True,
        },
    }
    if train_metadata is not None:
        payload.update(checkpoint_metadata_to_dict(train_metadata))
    return payload


def materialize_state_token_checkpoint(
    *,
    output_dir: str | Path,
    dataset_bundle: StateTokenDatasetBundle,
    manifest: FrozenComparisonManifest,
    train_metadata: TrainCheckpointMetadata | None = None,
) -> RecapCheckpointBundle:
    _require_native_state_dim(dataset_bundle)
    output_dir_path = Path(output_dir).resolve()
    checkpoint_dir = output_dir_path / "best"
    train_manifest_path = output_dir_path / "train_manifest.json"
    checkpoint_provenance_path = output_dir_path / "checkpoint_provenance.json"
    checkpoint_payload_path = checkpoint_dir / "checkpoint.json"

    train_manifest = build_train_manifest(
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        output_dir=output_dir_path,
        train_metadata=train_metadata,
    )
    checkpoint_provenance = build_checkpoint_provenance(
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        checkpoint_dir=checkpoint_dir,
        train_manifest_path=train_manifest_path,
        train_metadata=train_metadata,
    )
    recap_bundle = dataset_bundle.recap_bundle
    source_bundle = dataset_bundle.source_bundle
    checkpoint_payload = {
        "schema_version": "openpi_libero_state_tokens_checkpoint_payload_v1",
        "variant": RECAP_STATE_TOKENS_VARIANT,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": str(checkpoint_dir),
        "train_manifest": str(train_manifest_path),
        "checkpoint_provenance": str(checkpoint_provenance_path),
        "offline_success_proxy": float(recap_bundle.indicator_positive_fraction),
        "offline_failure_proxy": max(
            0.0, 1.0 - float(recap_bundle.indicator_positive_fraction)
        ),
        "record_count": int(dataset_bundle.aligned_record_count),
        "source_dataset_dir": str(source_bundle.dataset_dir),
        "source_dataset_name": source_bundle.dataset_name,
        "source_dataset_route_id": source_bundle.route_id,
        "source_dataset_schema_version": source_bundle.schema_version,
        "official_native_source_dataset_dir": str(source_bundle.source_dataset_dir),
        "official_native_source_dataset_name": source_bundle.source_dataset_name,
        "recap_label_dataset_dir": str(recap_bundle.dataset_dir),
        "recap_label_dataset_name": recap_bundle.dataset_name,
        "recap_label_route_id": OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
        "state_token_route": STATE_TOKEN_ROUTE,
        "source_state": SOURCE_STATE,
        "source_state_padding": SOURCE_STATE_PADDING,
        "state_token_semantics": STATE_TOKEN_SEMANTICS,
        "transform_order": TRANSFORM_ORDER,
        "discrete_state_input": True,
        "observed_dataset_state_dim": int(dataset_bundle.observed_dataset_state_dim),
    }

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
    "build_checkpoint_provenance",
    "build_train_manifest",
    "materialize_state_token_checkpoint",
]
