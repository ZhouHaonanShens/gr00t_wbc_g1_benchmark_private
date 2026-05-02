from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import cast

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.dataset import RecapDatasetBundle
from work.openpi.recap.checkpoint import TrainCheckpointMetadata
from work.openpi.recap.protocol import (
    FrozenComparisonManifest,
    build_frozen_comparison_manifest,
)
from work.openpi.state_tokens.checkpoint import (
    build_checkpoint_provenance,
    build_train_manifest,
    materialize_state_token_checkpoint,
)
from work.openpi.state_tokens.dataset import (
    OfficialNativeLiberoDatasetBundle,
    StateTokenDatasetBundle,
    resolve_official_native_8d_dataset,
    resolve_state_token_dataset,
)
from work.openpi.state_tokens.protocol import (
    BLOCKER_CODE_CONTROL_PARITY_NOT_SATISFIED,
    BLOCKER_CODE_INVALID_NATIVE_PROVENANCE,
    BLOCKER_CODE_INVALID_TRAINING_SOURCE,
    BLOCKER_CODE_MISSING_CONTROL_PARITY_ARTIFACT,
    NOT_APPLICABLE_STATE_TOKEN_ROUTE,
    OFFICIAL_NATIVE_DATASET_DIR,
    OFFICIAL_NATIVE_DATASET_NAME,
    OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
    OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
    REQUIRED_NATIVE_STATE_DIM,
    RECAP_STATE_TOKENS_VARIANT,
    SOURCE_STATE,
    SOURCE_STATE_PADDING,
    StateTokenContractError,
    STATE_TOKEN_ROUTE,
    STATE_TOKEN_SEMANTICS,
    TRANSFORM_ORDER,
)
from work.openpi.state_tokens.summary import (
    SUMMARY_FIELDS,
    build_eval_wrapper,
    build_recap_only_comparison_summary,
    build_state_token_summary,
    build_stock_comparison_summary,
    build_three_way_paired_summary,
    require_control_parity_ready,
    resolve_control_parity_reference,
)
from tests.openpi.carrier_text_v1_fixture import (  # noqa: E402
    carrier_text_v1_handoff_metadata,
)


def _manifest() -> FrozenComparisonManifest:
    return build_frozen_comparison_manifest(
        suite="libero_spatial",
        task_ids="0,1",
        seed_manifest="7,17",
        num_trials_per_task=2,
    )


def _mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"expected mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _training_source_bundle(
    *, state_dim: int = REQUIRED_NATIVE_STATE_DIM
) -> OfficialNativeLiberoDatasetBundle:
    return OfficialNativeLiberoDatasetBundle(
        dataset_dir=REPO_ROOT
        / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1",
        dataset_name=OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
        total_episodes=1693,
        total_frames=273465,
        total_tasks=40,
        fps=10,
        state_dim=state_dim,
        action_dim=7,
        task_texts=("put the bowl on the plate",),
        source_dataset_dir=OFFICIAL_NATIVE_DATASET_DIR,
        source_dataset_name=OFFICIAL_NATIVE_DATASET_NAME,
        schema_version="openpi_libero_official_8d_recap_relabels_v1",
        route_id=OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
    )


def _dataset_bundle() -> StateTokenDatasetBundle:
    source_bundle = _training_source_bundle()
    recap_bundle = RecapDatasetBundle(
        dataset_dir=REPO_ROOT
        / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1",
        dataset_name=OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
        parquet_files=(REPO_ROOT / "fake_episode.parquet",),
        total_rows=10,
        prompt_route="recap_conditioned_prompt_token_v1",
        conditioning_mode="prompt_text_only",
        source_prompt_field="prompt_raw",
        indicator_positive_fraction=0.25,
        indicator_positive_count=2,
        indicator_negative_count=8,
        advantage_input_mean=0.1,
        advantage_input_abs_mean=0.2,
        action_dim=7,
        state_dim=8,
        record_preview=(),
        recap_contract={"schema_version": "openpi_libero_recap_record_v1"},
    )
    return StateTokenDatasetBundle(
        source_bundle=source_bundle,
        recap_bundle=recap_bundle,
        aligned_record_count=10,
        state_token_route=STATE_TOKEN_ROUTE,
        source_state=SOURCE_STATE,
        source_state_padding=SOURCE_STATE_PADDING,
        transform_order=TRANSFORM_ORDER,
        state_token_semantics=STATE_TOKEN_SEMANTICS,
        discrete_state_input=True,
        observed_dataset_state_dim=REQUIRED_NATIVE_STATE_DIM,
    )


def _train_metadata() -> TrainCheckpointMetadata:
    return TrainCheckpointMetadata(
        variant_name="recap_state_tokens_relabel8d_v2",
        dataset_route_id=OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
        dataset_fingerprint="fixture_dataset_fingerprint_sha256",
        episode_universe_hash="fixture_episode_universe_hash_sha256",
        base_checkpoint_id="pi05_libero_anchor",
        train_budget_id="libero_cmp_budget_v2",
        consumer_mode="informative_adv",
        gate_eval_manifest_hash="fixture_gate_eval_manifest_hash_sha256",
        reuse_existing_checkpoint=False,
        reuse_verdict="materialize_new_checkpoint",
    )


def _invalid_dataset_bundle() -> StateTokenDatasetBundle:
    source_bundle = _training_source_bundle(state_dim=43)
    recap_bundle = RecapDatasetBundle(
        dataset_dir=REPO_ROOT
        / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1",
        dataset_name=OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
        parquet_files=(REPO_ROOT / "fake_episode.parquet",),
        total_rows=10,
        prompt_route="recap_conditioned_prompt_token_v1",
        conditioning_mode="prompt_text_only",
        source_prompt_field="prompt_raw",
        indicator_positive_fraction=0.25,
        indicator_positive_count=2,
        indicator_negative_count=8,
        advantage_input_mean=0.1,
        advantage_input_abs_mean=0.2,
        action_dim=7,
        state_dim=43,
        record_preview=(),
        recap_contract={"schema_version": "openpi_libero_recap_record_v1"},
    )
    return StateTokenDatasetBundle(
        source_bundle=source_bundle,
        recap_bundle=recap_bundle,
        aligned_record_count=10,
        state_token_route=STATE_TOKEN_ROUTE,
        source_state=SOURCE_STATE,
        source_state_padding=SOURCE_STATE_PADDING,
        transform_order=TRANSFORM_ORDER,
        state_token_semantics=STATE_TOKEN_SEMANTICS,
        discrete_state_input=True,
        observed_dataset_state_dim=43,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_minimal_state_token_dataset(
    root: Path,
    *,
    dataset_name: str = OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
    route_id: str = OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
    source_dataset_name: str = OFFICIAL_NATIVE_DATASET_NAME,
    state_dim: int = REQUIRED_NATIVE_STATE_DIM,
    action_dim: int = 7,
) -> Path:
    dataset_dir = root / dataset_name
    _write_json(
        dataset_dir / "meta" / "info.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
            "route_id": route_id,
            **carrier_text_v1_handoff_metadata(),
            "source_dataset_dir": str(OFFICIAL_NATIVE_DATASET_DIR),
            "source_dataset_name": source_dataset_name,
            "fps": 10,
            "total_episodes": 1,
            "total_frames": 2,
            "total_tasks": 1,
            "features": {
                "observation.images.ego_view": {
                    "dtype": "image",
                    "shape": [256, 256, 3],
                },
                "observation.state": {"dtype": "float32", "shape": [state_dim]},
                "action": {"dtype": "float32", "shape": [action_dim]},
                "annotation.human.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                },
                "annotation.human.action.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                },
            },
            "recap_advantage_input_contract": {
                "contract_version": "full_recap_continuous_adv_v2"
            },
        },
    )
    _write_json(
        dataset_dir / "meta" / "modality.json",
        {
            "video": {
                "observation.images.ego_view": {
                    "original_key": "observation.images.ego_view"
                }
            },
            "state": {"observation.state": {}},
            "action": {"action": {}},
            "annotation": {"annotation.human.task_description": {}},
        },
    )
    _write_jsonl(
        dataset_dir / "meta" / "tasks.jsonl",
        [{"task": "put the bowl on the plate", "task_index": 0}],
    )
    frame = pd.DataFrame(
        {
            "action": [[0.1] * action_dim, [0.2] * action_dim],
            "episode_index": [0, 0],
            "observation.state": [[0.0] * state_dim, [1.0] * state_dim],
            "recap_m2.advantage_A": [0.5, -0.5],
            "recap_m2.advantage_input": [0.25, -0.25],
            "recap_m2.indicator_I": [1, 0],
            "recap_m2.prompt_conditioned": [
                "advantage positive put the bowl on the plate",
                "advantage negative put the bowl on the plate",
            ],
            "recap_m2.prompt_raw": [
                "put the bowl on the plate",
                "put the bowl on the plate",
            ],
            "recap_m2.return_G": [0.0, -1.0],
            "recap_m2.value_V": [-0.5, -0.5],
        }
    )
    parquet_path = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    return dataset_dir


def _write_control_gate_report(
    path: Path,
    *,
    checkpoint_dir: Path,
    status: str = "blocked",
) -> None:
    payload: dict[str, object] = {
        "schema_version": "openpi_libero_recap_control_gate_v1",
        "route_id": "task9b_recap_only_relabel8d_control_gate_v1",
        "variant": "recap_only",
        "status": status,
        "decision_reason": "control parity test fixture",
        "rerun_control": status != "reuse_existing_control",
        "rerun_possible_now": status == "rerun_required",
        "existing_control": {
            "checkpoint_dir": str(checkpoint_dir.resolve()),
        },
        "rerun_target": {
            "checkpoint_dir": str(
                (
                    checkpoint_dir.parent.parent / "recap_only_relabel8d_v1" / "best"
                ).resolve()
            ),
        },
    }
    if status == "blocked":
        payload["blocker"] = {
            "code": "missing_materialized_relabel8d_source",
            "reason": "control parity remains blocked in test fixture",
        }
    _write_json(
        path,
        payload,
    )


def test_train_manifest_and_provenance_freeze_native_discrete_state_only() -> None:
    manifest = _manifest()
    dataset_bundle = _dataset_bundle()
    train_metadata = _train_metadata()
    output_dir = (
        REPO_ROOT
        / "agent/artifacts/checkpoints/openpi_libero_variants/recap_state_tokens_v1"
    )
    train_manifest = build_train_manifest(
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        output_dir=output_dir,
        train_metadata=train_metadata,
    )
    provenance = build_checkpoint_provenance(
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        checkpoint_dir=output_dir / "best",
        train_manifest_path=output_dir / "train_manifest.json",
        train_metadata=train_metadata,
    )

    assert train_manifest["variant"] == RECAP_STATE_TOKENS_VARIANT
    assert train_manifest["variant_name"] == "recap_state_tokens_relabel8d_v2"
    assert train_manifest["dataset_route_id"] == OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID
    assert train_manifest["dataset_fingerprint"] == "fixture_dataset_fingerprint_sha256"
    assert (
        train_manifest["episode_universe_hash"]
        == "fixture_episode_universe_hash_sha256"
    )
    assert train_manifest["base_checkpoint_id"] == "pi05_libero_anchor"
    assert train_manifest["train_budget_id"] == "libero_cmp_budget_v2"
    assert train_manifest["consumer_mode"] == "informative_adv"
    assert (
        train_manifest["gate_eval_manifest_hash"]
        == "fixture_gate_eval_manifest_hash_sha256"
    )
    assert train_manifest["reuse_existing_checkpoint"] is False
    assert train_manifest["reuse_verdict"] == "materialize_new_checkpoint"
    training_route = _mapping(train_manifest["training_route"])
    assert (
        training_route["only_added_experimental_variable"]
        == "native discrete_state_input=True"
    )
    assert training_route["conditioning_mode"] == "prompt_text_only"
    assert training_route["consumer_mode"] == "informative_adv"
    assert training_route["fixed_indicator_mode"] is None
    assert training_route["discrete_state_input"] is True
    assert training_route["state_token_route"] == STATE_TOKEN_ROUTE
    assert training_route["state_token_semantics"] == STATE_TOKEN_SEMANTICS
    assert training_route["source_state"] == SOURCE_STATE
    assert training_route["source_state_padding"] == SOURCE_STATE_PADDING
    assert training_route["transform_order"] == TRANSFORM_ORDER
    assert (
        training_route["source_dataset_name"]
        == OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME
    )
    assert (
        training_route["source_dataset_route_id"]
        == OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID
    )
    assert (
        training_route["official_native_source_dataset_name"]
        == OFFICIAL_NATIVE_DATASET_NAME
    )
    assert (
        training_route["recap_label_dataset_name"]
        == OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME
    )
    assert (
        training_route["recap_label_route_id"] == OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID
    )
    assert training_route["no_second_tokenizer"] is True
    assert training_route["no_rl_token"] is True

    assert (
        provenance["checkpoint_source"]
        == "repo_local_openpi_recap_state_tokens_native_discrete_state_input_v1"
    )
    assert provenance["variant_name"] == "recap_state_tokens_relabel8d_v2"
    assert provenance["dataset_route_id"] == OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID
    assert provenance["dataset_fingerprint"] == "fixture_dataset_fingerprint_sha256"
    assert provenance["episode_universe_hash"] == "fixture_episode_universe_hash_sha256"
    assert provenance["base_checkpoint_id"] == "pi05_libero_anchor"
    assert provenance["train_budget_id"] == "libero_cmp_budget_v2"
    assert provenance["consumer_mode"] == "informative_adv"
    assert (
        provenance["gate_eval_manifest_hash"]
        == "fixture_gate_eval_manifest_hash_sha256"
    )
    assert provenance["reuse_existing_checkpoint"] is False
    assert provenance["reuse_verdict"] == "materialize_new_checkpoint"
    assert provenance["state_token_route"] == STATE_TOKEN_ROUTE
    variant_derivation = _mapping(provenance["variant_derivation"])
    assert (
        variant_derivation["only_added_experimental_variable"]
        == "native discrete_state_input=True"
    )
    assert variant_derivation["consumer_mode"] == "informative_adv"
    assert variant_derivation["fixed_indicator_mode"] is None
    assert variant_derivation["discrete_state_input"] is True
    assert variant_derivation["state_token_route"] == STATE_TOKEN_ROUTE
    assert variant_derivation["contract_state_dim"] == 8
    assert variant_derivation["observed_dataset_state_dim"] == 8
    assert (
        variant_derivation["source_dataset_name"]
        == OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME
    )
    assert (
        variant_derivation["source_dataset_route_id"]
        == OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID
    )
    assert (
        variant_derivation["official_native_source_dataset_name"]
        == OFFICIAL_NATIVE_DATASET_NAME
    )
    assert (
        variant_derivation["recap_label_dataset_name"]
        == OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME
    )
    assert variant_derivation["no_symbolic_phase_token"] is True
    assert variant_derivation["no_task_phase_id"] is True
    assert variant_derivation["no_custom_token_vocabulary"] is True


def test_materialize_state_token_checkpoint_writes_best_bundle_and_root_sidecars(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    dataset_bundle = _dataset_bundle()
    train_metadata = _train_metadata()

    checkpoint = materialize_state_token_checkpoint(
        output_dir=tmp_path / "recap_state_tokens_v1",
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        train_metadata=train_metadata,
    )

    assert checkpoint.checkpoint_dir == (tmp_path / "recap_state_tokens_v1" / "best")
    assert checkpoint.train_manifest_path.is_file()
    assert checkpoint.checkpoint_provenance_path.is_file()
    assert (checkpoint.checkpoint_dir / "checkpoint.json").is_file()
    root_provenance = _mapping(
        cast(
            object,
            json.loads(
                checkpoint.checkpoint_provenance_path.read_text(encoding="utf-8")
            ),
        )
    )
    best_payload = _mapping(
        cast(
            object,
            json.loads(
                (checkpoint.checkpoint_dir / "checkpoint.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
    )

    assert root_provenance["state_token_route"] == STATE_TOKEN_ROUTE
    assert (
        root_provenance["dataset_fingerprint"] == "fixture_dataset_fingerprint_sha256"
    )
    assert (
        root_provenance["episode_universe_hash"]
        == "fixture_episode_universe_hash_sha256"
    )
    assert root_provenance["base_checkpoint_id"] == "pi05_libero_anchor"
    assert root_provenance["train_budget_id"] == "libero_cmp_budget_v2"
    assert root_provenance["consumer_mode"] == "informative_adv"
    assert root_provenance["reuse_verdict"] == "materialize_new_checkpoint"
    assert (
        _mapping(root_provenance["variant_derivation"])["discrete_state_input"] is True
    )
    assert (
        best_payload["source_dataset_name"]
        == OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME
    )
    assert (
        best_payload["source_dataset_route_id"]
        == OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID
    )
    assert (
        best_payload["official_native_source_dataset_name"]
        == OFFICIAL_NATIVE_DATASET_NAME
    )
    assert best_payload["state_token_route"] == STATE_TOKEN_ROUTE
    assert best_payload["discrete_state_input"] is True


def test_resolve_official_native_8d_dataset_anchors_task9_source() -> None:
    bundle = resolve_official_native_8d_dataset()

    assert bundle.dataset_name == OFFICIAL_NATIVE_DATASET_NAME
    assert bundle.state_dim == REQUIRED_NATIVE_STATE_DIM
    assert bundle.action_dim == 7
    assert bundle.total_tasks == 40
    assert bundle.total_episodes == 1693
    assert "put the bowl on the plate" in bundle.task_texts


def test_resolve_state_token_dataset_accepts_relabeled_official_native_8d_source(
    tmp_path: Path,
) -> None:
    dataset_dir = _write_minimal_state_token_dataset(tmp_path)

    bundle = resolve_state_token_dataset(dataset_dir, preview_limit=2)

    assert bundle.dataset_name == OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME
    assert bundle.total_rows == 2
    assert bundle.state_token_route == STATE_TOKEN_ROUTE
    assert bundle.discrete_state_input is True
    assert bundle.source_bundle.source_dataset_name == OFFICIAL_NATIVE_DATASET_NAME
    assert bundle.source_bundle.route_id == OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID
    assert (
        bundle.recap_bundle.dataset_name == OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME
    )
    assert bundle.recap_bundle.prompt_route == "recap_conditioned_prompt_token_v1"


def test_resolve_state_token_dataset_rejects_wrong_route_id(tmp_path: Path) -> None:
    dataset_dir = _write_minimal_state_token_dataset(
        tmp_path,
        route_id="wrong_route_id",
    )

    try:
        _ = resolve_state_token_dataset(dataset_dir)
    except StateTokenContractError as exc:
        assert exc.payload["status"] == "blocked"
        assert exc.payload["blocker_code"] == BLOCKER_CODE_INVALID_TRAINING_SOURCE
        assert (
            exc.payload["required_training_route_id"]
            == OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID
        )
        assert exc.payload["observed_route_id"] == "wrong_route_id"
    else:
        raise AssertionError("expected invalid route_id to be rejected")


def test_checkpoint_builders_reject_native_route_with_non_8d_observed_state() -> None:
    manifest = _manifest()
    dataset_bundle = _invalid_dataset_bundle()
    output_dir = (
        REPO_ROOT / "agent/artifacts/checkpoints/openpi_libero_variants/invalid"
    )

    try:
        _ = build_train_manifest(
            dataset_bundle=dataset_bundle,
            manifest=manifest,
            output_dir=output_dir,
        )
    except ValueError as exc:
        assert "observed_dataset_state_dim=8" in str(exc)
    else:
        raise AssertionError(
            "expected build_train_manifest to reject non-8D native route"
        )


def test_summary_builder_rejects_native_route_with_non_8d_provenance(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    state_token_dir = tmp_path / "recap_state_tokens_v1" / "best"
    _write_json(
        state_token_dir / "checkpoint.json",
        {
            "offline_success_proxy": 0.375,
            "state_token_route": STATE_TOKEN_ROUTE,
        },
    )
    _write_json(
        state_token_dir / "checkpoint_provenance.json",
        {
            "checkpoint_source": "repo_local_openpi_recap_state_tokens_native_discrete_state_input_v1",
            "state_token_route": STATE_TOKEN_ROUTE,
            "variant_derivation": {
                "state_token_route": STATE_TOKEN_ROUTE,
                "observed_dataset_state_dim": 43,
                "source_dataset_name": OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
                "source_dataset_route_id": OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
                "official_native_source_dataset_name": OFFICIAL_NATIVE_DATASET_NAME,
            },
        },
    )

    try:
        _ = build_state_token_summary(state_token_dir, manifest=manifest)
    except StateTokenContractError as exc:
        assert exc.payload["status"] == "blocked"
        assert exc.payload["blocker_code"] == BLOCKER_CODE_INVALID_NATIVE_PROVENANCE
        assert exc.payload["observed_dataset_state_dim"] == 43
    else:
        raise AssertionError(
            "expected build_state_token_summary to reject non-8D provenance"
        )


def test_recap_only_comparison_summary_requires_task9b_control_gate(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    recap_only_dir = tmp_path / "recap_only_v1" / "best"
    _write_json(
        recap_only_dir / "checkpoint.json",
        {
            "offline_success_proxy": 0.25,
        },
    )
    _write_json(
        recap_only_dir / "checkpoint_provenance.json",
        {
            "checkpoint_source": "repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
        },
    )

    try:
        _ = build_recap_only_comparison_summary(
            recap_only_dir,
            manifest=manifest,
            control_gate_report_path=tmp_path
            / "missing_source_equivalence_report.json",
        )
    except StateTokenContractError as exc:
        assert exc.payload["status"] == "blocked"
        assert (
            exc.payload["blocker_code"] == BLOCKER_CODE_MISSING_CONTROL_PARITY_ARTIFACT
        )
    else:
        raise AssertionError("expected missing 9B gate artifact to block comparison")


def test_blocked_control_gate_blocks_state_token_ablation(tmp_path: Path) -> None:
    recap_only_dir = tmp_path / "recap_only_v1" / "best"
    recap_only_dir.mkdir(parents=True, exist_ok=True)
    control_gate_report = tmp_path / "source_equivalence_report.json"
    _write_control_gate_report(
        control_gate_report,
        checkpoint_dir=recap_only_dir,
        status="blocked",
    )

    try:
        _ = require_control_parity_ready(
            tmp_path / "recap_state_tokens_v1" / "best",
            control_gate_report_path=control_gate_report,
            stage="train_preflight",
        )
    except StateTokenContractError as exc:
        assert exc.payload["status"] == "blocked"
        assert exc.payload["blocker_code"] == BLOCKER_CODE_CONTROL_PARITY_NOT_SATISFIED
        assert exc.payload["control_gate_status"] == "blocked"
        task9b_blocker = _mapping(exc.payload["task9b_blocker"])
        assert task9b_blocker["code"] == "missing_materialized_relabel8d_source"
    else:
        raise AssertionError("expected blocked Task 9B gate to block Task 9D")


def test_three_way_summary_keeps_stock_recap_only_and_state_tokens_rows(
    tmp_path: Path,
) -> None:
    manifest = _manifest()

    stock_summary_path = tmp_path / "stock_summary.json"
    _write_json(
        stock_summary_path,
        {
            "provenance": {
                "checkpoint_source": "upstream_openpi_default_or_explicit_cli",
                "task_ids": [0],
                "seed_manifest": [7],
                "num_trials_per_task": 1,
            },
            "client": {"success_rate": 1.0},
        },
    )

    existing_control_dir = tmp_path / "recap_only_v1" / "best"
    _write_json(
        existing_control_dir / "checkpoint.json",
        {
            "offline_success_proxy": 0.25,
        },
    )
    _write_json(
        existing_control_dir / "checkpoint_provenance.json",
        {
            "checkpoint_source": "repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
        },
    )
    recap_only_dir = tmp_path / "recap_only_relabel8d_v1" / "best"
    _write_json(
        recap_only_dir / "checkpoint.json",
        {
            "offline_success_proxy": 0.29995794708646445,
        },
    )
    _write_json(
        recap_only_dir / "checkpoint_provenance.json",
        {
            "checkpoint_source": "repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
        },
    )
    control_gate_report = tmp_path / "source_equivalence_report.json"
    _write_control_gate_report(
        control_gate_report,
        checkpoint_dir=existing_control_dir,
        status="rerun_required",
    )

    state_token_dir = tmp_path / "recap_state_tokens_v1" / "best"
    _write_json(
        state_token_dir / "checkpoint.json",
        {
            "offline_success_proxy": 0.375,
            "state_token_route": STATE_TOKEN_ROUTE,
        },
    )
    _write_json(
        state_token_dir / "checkpoint_provenance.json",
        {
            "checkpoint_source": "repo_local_openpi_recap_state_tokens_native_discrete_state_input_v1",
            "state_token_route": STATE_TOKEN_ROUTE,
            "variant_derivation": {
                "state_token_route": STATE_TOKEN_ROUTE,
                "observed_dataset_state_dim": REQUIRED_NATIVE_STATE_DIM,
                "source_dataset_name": OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
                "source_dataset_route_id": OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
                "official_native_source_dataset_name": OFFICIAL_NATIVE_DATASET_NAME,
            },
        },
    )

    stock_summary = build_stock_comparison_summary(
        stock_summary_path, manifest=manifest
    )
    control_parity = resolve_control_parity_reference(
        recap_only_dir,
        control_gate_report_path=control_gate_report,
    )
    recap_only_summary = build_recap_only_comparison_summary(
        recap_only_dir,
        manifest=manifest,
        control_gate_report_path=control_gate_report,
    )
    state_token_summary = build_state_token_summary(
        state_token_dir,
        manifest=manifest,
    )
    paired_summary = build_three_way_paired_summary(
        stock_summary=stock_summary,
        recap_only_summary=recap_only_summary,
        state_token_summary=state_token_summary,
        control_parity=control_parity,
    )
    payload = build_eval_wrapper(
        variant=RECAP_STATE_TOKENS_VARIANT,
        summary=state_token_summary,
        paired_summary=paired_summary,
    )

    assert SUMMARY_FIELDS[-1] == "state_token_route"
    assert stock_summary["state_token_route"] == NOT_APPLICABLE_STATE_TOKEN_ROUTE
    assert recap_only_summary["state_token_route"] == NOT_APPLICABLE_STATE_TOKEN_ROUTE
    assert state_token_summary["state_token_route"] == STATE_TOKEN_ROUTE
    rows = cast(list[Mapping[str, object]], paired_summary["paired_summary"])
    assert [row["variant"] for row in rows] == [
        "stock",
        "recap_only",
        RECAP_STATE_TOKENS_VARIANT,
    ]
    assert [row["state_token_route"] for row in rows] == [
        NOT_APPLICABLE_STATE_TOKEN_ROUTE,
        NOT_APPLICABLE_STATE_TOKEN_ROUTE,
        STATE_TOKEN_ROUTE,
    ]
    assert _mapping(payload["summary"])["state_token_route"] == STATE_TOKEN_ROUTE
    assert _mapping(payload["paired_summary"])["summary_fields"] == list(SUMMARY_FIELDS)
    control_parity_payload = _mapping(
        _mapping(payload["paired_summary"])["control_parity"]
    )
    assert control_parity_payload["status"] == "rerun_required"
    assert control_parity_payload["authorized_checkpoint_dir"] == str(
        recap_only_dir.resolve()
    )
    recap_notes = cast(list[str], recap_only_summary["deviation_notes"])
    assert any("relabel8d rerun checkpoint" in note for note in recap_notes)
