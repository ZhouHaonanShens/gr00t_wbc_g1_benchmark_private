from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[4]


from work.openpi.eval.protocols.manifest import (  # noqa: E402
    EXPECTED_ALLOWED_VARIANTS,
    build_rollout_manifest_v21,
    write_rollout_manifest_v21,
)
from work.openpi.eval import OpenPIEvalApp, RolloutEvaluationScenario  # noqa: E402
from work.openpi.eval.scenarios import CheckpointSelection, TraceMetricConfig  # noqa: E402
from work.openpi.eval.workflows.rollout_support import (  # noqa: E402
    CHECKPOINT_INSTANCE_BINDING_KEY_FILES,
    CHECKPOINT_INSTANCE_BINDING_SCHEMA_VERSION,
    EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
)
import work.openpi.eval.workflows.rollout_support as libero_rollout_eval_v21  # noqa: E402
from work.openpi.recap.control_gate import (  # noqa: E402
    build_repaired_gate_rows,
    build_repaired_headline_results,
    build_task11_blocker_verdict,
)
from work.openpi.recap.runtime_prompt import (  # noqa: E402
    build_runtime_prompt_bundle,
    resolve_runtime_indicator_config,
)
from work.openpi.model import (  # noqa: E402
    build_effective_runtime_spec as build_effective_runtime_spec_model,
    effective_runtime_spec_hash,
)
from work.openpi.recap.dataset_aggregation import (  # noqa: E402
    CollectionBundle,
    ITERATION_MANIFEST_NAME,
    MergedDatasetBundle,
    build_iteration_manifest,
    compute_selection_source_hash,
    read_json,
    write_json,
)
from work.openpi.recap.protocol import (  # noqa: E402
    REPAIRED_METRIC_PROFILE_ID,
    build_train_runtime_dir,
)
from work.openpi.recap.train_config import (  # noqa: E402
    STAGE_OMIT_CONTROL,
    STAGE_RECAP_INFORMATIVE,
    STAGE_SHUFFLED_INDICATOR,
)
import work.openpi.pipelines.recap.critic_training as train_recap_critic  # noqa: E402
import work.openpi.pipelines.recap.policy_training as libero_recap_train  # noqa: E402
from work.openpi.pipelines.recap.collect import (  # noqa: E402
    CollectConfig,
    run_collection,
)
from work.openpi.pipelines.recap.merge import (  # noqa: E402
    MergeConfig,
    run_merge,
)
from work.openpi.runtime.config import (  # noqa: E402
    DEFAULT_RUNTIME_BRIDGE_CONFIG,
    RuntimeBridgeConfig,
)


CRITIC_RETRAIN_ROUTE_ID = "openpi_libero_recap_task10_critic_retrain_v1"
POLICY_RETRAIN_ROUTE_ID = "openpi_libero_recap_task10_policy_retrain_v1"
ITERATION_EVAL_ROUTE_ID = "openpi_libero_recap_task10_eval_v1"
ITERATION_EVAL_SUMMARY_SCHEMA_VERSION = "openpi_libero_recap_iteration_eval_summary_v1"
ITERATION_HEADLINE_SCHEMA_VERSION = (
    "openpi_libero_recap_iteration_headline_comparisons_v1"
)
ITERATION_GATE_RESULTS_SCHEMA_VERSION = "openpi_libero_recap_iteration_gate_results_v1"
DEFAULT_REPAIRED_MATRIX_SUMMARY_PATH = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "openpi_recap_v1"
    / "repaired_matrix_summary.json"
)
DEFAULT_TRACKED_SUMMARY_PATH = (
    REPO_ROOT / "agent" / "exchange" / "openpi_recap_iteration_smoke_summary_v1.md"
)
STAGE_RUNTIME_CACHE_DIRNAMES = (
    "subprocess_cache",
    "upstream_train_checkpoints",
)


def _build_train_runtime_dir(output_dir: Path, *, variant: str) -> Path:
    # Preserve the historical monkeypatch seam on the facade module while
    # keeping the real workflow implementation in this file.
    import work.openpi.pipelines.recap.iteration as iteration_facade

    return iteration_facade.build_train_runtime_dir(output_dir, variant=variant)


@dataclass(frozen=True)
class CriticRetrainBundle:
    output_dir: Path
    checkpoint_dir: Path
    train_summary_path: Path
    train_summary: dict[str, object]


@dataclass(frozen=True)
class PolicyStageBundle:
    stage: str
    repaired_variant_ids: tuple[str, ...]
    output_dir: Path
    checkpoint_dir: Path
    train_manifest_path: Path
    checkpoint_provenance_path: Path
    runtime_summary_path: Path
    runtime_summary: dict[str, object]


@dataclass(frozen=True)
class EvalVariantSpec:
    repaired_variant_id: str
    carrier_variant_id: str
    checkpoint_dir: Path
    indicator_mode: str
    output_dir: Path
    canonical_source_dir: Path | None = None
    effective_runtime_spec_hash: str | None = None
    same_effective_runtime_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalVariantBundle:
    repaired_variant_id: str
    carrier_variant_id: str
    checkpoint_dir: Path
    indicator_mode: str
    output_dir: Path
    summary_path: Path
    summary: dict[str, object]
    effective_runtime_spec_hash: str | None
    same_effective_runtime_aliases: tuple[str, ...]


@dataclass(frozen=True)
class IterationConfig:
    iter_id: str
    seed_policy_checkpoint: Path
    critic_checkpoint: Path | None
    indicator_mode: str
    task_suite_name: str
    task_ids: str
    episodes: int
    output_dir: Path
    demo_dir: Path
    correction_dir: Path | None
    critic_config: Path | None
    repaired_matrix_summary_path: Path
    tracked_summary_path: Path
    prepared_dataset_dir: Path | None = None
    informative_prepared_dataset_dir: Path | None = None
    runtime: RuntimeBridgeConfig = DEFAULT_RUNTIME_BRIDGE_CONFIG

    @property
    def collect_dir(self) -> Path:
        return self.output_dir / "collect"

    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / "dataset"

    @property
    def critic_retrain_dir(self) -> Path:
        return self.output_dir / "critic_retrain"

    @property
    def policy_retrain_dir(self) -> Path:
        return self.output_dir / "policy_retrain"

    @property
    def eval_dir(self) -> Path:
        return self.output_dir / "eval"

    @property
    def eval_manifest_path(self) -> Path:
        return self.eval_dir / "rollout_manifest_v21.json"

    @property
    def eval_summary_path(self) -> Path:
        return self.eval_dir / "eval_summary.json"

    @property
    def repaired_comparisons_path(self) -> Path:
        return self.eval_dir / "repaired_headline_comparisons.json"

    @property
    def gate_results_path(self) -> Path:
        return self.eval_dir / "repaired_gate_results.json"

    @property
    def blocker_verdict_path(self) -> Path:
        return self.eval_dir / "blocker_verdict.json"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / ITERATION_MANIFEST_NAME

    @property
    def collect_merge_only(self) -> bool:
        return self.critic_config is None


class IterationWorkflow:
    """Orchestrate one canonical RECAP iteration from collect to final verdict.

    In full mode the workflow runs collect, merge, critic retrain, repaired
    policy retrains, and iteration eval before writing the iteration manifest.
    When ``collect_merge_only`` is active it deliberately stops after the data
    surface so compatibility paths can reuse the same contract.
    """

    def __init__(self, config: IterationConfig) -> None:
        self.config: IterationConfig = config

    def run(self) -> dict[str, object]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        collection_bundle = self._run_collection()
        merged_dataset = self._run_merge()
        critic_retrain_payload: dict[str, object] | None = None
        policy_retrain_payload: dict[str, object] | None = None
        iteration_eval_payload: dict[str, object] | None = None

        if not self.config.collect_merge_only:
            retrain_dataset_dir = merged_dataset.output_dir
            critic_bundle = self._run_critic_retrain(retrain_dataset_dir)
            policy_bundles = self._run_policy_retrains(
                retrain_dataset_dir,
                critic_bundle.checkpoint_dir,
            )
            iteration_eval_payload = self._run_iteration_eval(
                collection_bundle=collection_bundle,
                policy_bundles=policy_bundles,
            )
            critic_retrain_payload = self._critic_manifest_payload(
                critic_bundle,
                source_dataset_ref=retrain_dataset_dir,
            )
            policy_retrain_payload = self._policy_manifest_payload(
                policy_bundles,
                critic_bundle.checkpoint_dir,
                source_dataset_ref=retrain_dataset_dir,
            )

        manifest = build_iteration_manifest(
            iter_id=self._iter_label(),
            collection_bundle=collection_bundle,
            merged_dataset=merged_dataset,
            critic_retrain=critic_retrain_payload,
            policy_retrain=policy_retrain_payload,
            iteration_eval=iteration_eval_payload,
        )
        _ = write_json(self.config.manifest_path, manifest)
        if iteration_eval_payload is not None:
            self._write_tracked_summary(manifest)
        return manifest

    def _run_collection(self) -> CollectionBundle:
        return run_collection(
            CollectConfig(
                policy_checkpoint=self.config.seed_policy_checkpoint,
                critic_checkpoint=self.config.critic_checkpoint,
                indicator_mode=self.config.indicator_mode,
                task_suite_name=self.config.task_suite_name,
                task_ids=tuple(
                    int(value)
                    for value in self.config.task_ids.split(",")
                    if value.strip()
                ),
                episodes=self.config.episodes,
                output_dir=self.config.collect_dir,
                demo_dir=self.config.demo_dir,
                runtime=self.config.runtime,
            )
        )

    def _run_merge(self) -> MergedDatasetBundle:
        return run_merge(
            MergeConfig(
                demo_dir=self.config.demo_dir,
                autonomous_dir=self.config.collect_dir,
                output_dir=self.config.dataset_dir,
                correction_dir=self.config.correction_dir,
            )
        )

    def _run_critic_retrain(self, dataset_dir: Path) -> CriticRetrainBundle:
        critic_config = self.config.critic_config
        if critic_config is None:
            raise RuntimeError("critic retrain requested without --critic-config")
        exit_code = train_recap_critic.main(
            [
                "--config",
                str(critic_config),
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(self.config.critic_retrain_dir),
            ]
        )
        if exit_code != 0:
            raise RuntimeError(f"critic retrain failed with exit_code={exit_code}")
        train_summary_path = self.config.critic_retrain_dir / "train_summary.json"
        return CriticRetrainBundle(
            output_dir=self.config.critic_retrain_dir,
            checkpoint_dir=self.config.critic_retrain_dir / "best",
            train_summary_path=train_summary_path,
            train_summary=read_json(train_summary_path),
        )

    def _run_policy_retrains(
        self,
        dataset_dir: Path,
        critic_checkpoint_dir: Path,
    ) -> tuple[PolicyStageBundle, ...]:
        stage_specs = (
            (STAGE_OMIT_CONTROL, ("B0_omit_control_v2",)),
            (STAGE_SHUFFLED_INDICATOR, ("X_shuffled_indicator_v2",)),
            (
                STAGE_RECAP_INFORMATIVE,
                (
                    "C0_recap_informative_positiveinfer_v2",
                    "C1_recap_informative_cfg_v2",
                ),
            ),
        )
        bundles: list[PolicyStageBundle] = []
        for stage, repaired_variant_ids in stage_specs:
            output_dir = self.config.policy_retrain_dir / stage
            argv = [
                "--stage",
                stage,
                "--dataset-dir",
                str(dataset_dir),
                "--critic-checkpoint",
                str(critic_checkpoint_dir),
                "--output-dir",
                str(output_dir),
            ]
            prepared_dataset_dir = self._prepared_dataset_dir_for_stage(
                stage=stage,
                dataset_dir=dataset_dir,
            )
            if prepared_dataset_dir is not None:
                argv.extend(
                    [
                        "--prepared-dataset-dir",
                        str(prepared_dataset_dir),
                    ]
                )
            exit_code = libero_recap_train.main(argv)
            if exit_code != 0:
                raise RuntimeError(
                    f"policy retrain failed for stage={stage} exit_code={exit_code}"
                )
            runtime_summary_path = (
                _build_train_runtime_dir(output_dir, variant=stage) / "summary.json"
            )
            runtime_summary = read_json(runtime_summary_path)
            self._cleanup_completed_stage_runtime(stage=stage, output_dir=output_dir)
            bundles.append(
                PolicyStageBundle(
                    stage=stage,
                    repaired_variant_ids=repaired_variant_ids,
                    output_dir=output_dir,
                    checkpoint_dir=output_dir / "best",
                    train_manifest_path=output_dir / "best" / "train_manifest.json",
                    checkpoint_provenance_path=output_dir
                    / "best"
                    / "checkpoint_provenance.json",
                    runtime_summary_path=runtime_summary_path,
                    runtime_summary=runtime_summary,
                )
            )
        return tuple(bundles)

    def _prepared_dataset_dir_for_stage(
        self,
        *,
        stage: str,
        dataset_dir: Path,
    ) -> Path | None:
        if stage == STAGE_RECAP_INFORMATIVE:
            candidate = self.config.informative_prepared_dataset_dir
            if candidate is not None and candidate.is_dir():
                return candidate
        candidate = self.config.prepared_dataset_dir
        if candidate is not None and candidate.is_dir():
            return candidate
        return None

    def _cleanup_completed_stage_runtime(self, *, stage: str, output_dir: Path) -> None:
        runtime_dir = _build_train_runtime_dir(output_dir, variant=stage)
        self._assert_stage_runtime_cleanup_safe(
            stage=stage,
            output_dir=output_dir,
            runtime_dir=runtime_dir,
        )
        for dirname in STAGE_RUNTIME_CACHE_DIRNAMES:
            cache_dir = runtime_dir / dirname
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

    def _assert_stage_runtime_cleanup_safe(
        self,
        *,
        stage: str,
        output_dir: Path,
        runtime_dir: Path,
    ) -> None:
        missing_paths = [
            str(path)
            for path in (
                output_dir / "best",
                runtime_dir / "summary.json",
                runtime_dir / "train.log",
                runtime_dir / "real_variant_training.log",
                runtime_dir / "real_variant_export",
            )
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "refusing to clean runtime caches before stage artifacts are complete "
                + f"for stage={stage}: {', '.join(missing_paths)}"
            )

    def _run_iteration_eval(
        self,
        *,
        collection_bundle: CollectionBundle,
        policy_bundles: Sequence[PolicyStageBundle],
    ) -> dict[str, object]:
        repaired_matrix_summary = read_json(self.config.repaired_matrix_summary_path)
        eval_manifest_path = self._write_eval_manifest(collection_bundle)
        eval_bundles = self._run_eval_variants(eval_manifest_path, policy_bundles)
        observed_metrics = {
            bundle.repaired_variant_id: self._extract_metric_points(bundle.summary)
            for bundle in eval_bundles
        }
        headline_results = build_repaired_headline_results(
            repaired_matrix_summary,
            metrics_by_variant=observed_metrics,
        )
        gate_rows = build_repaired_gate_rows(
            repaired_matrix_summary,
            metrics_by_variant=observed_metrics,
        )
        blocker_verdict = build_task11_blocker_verdict(gate_rows)

        comparisons_payload = {
            "schema_version": ITERATION_HEADLINE_SCHEMA_VERSION,
            "iter_id": self._iter_label(),
            "repaired_matrix_summary_ref": str(
                self.config.repaired_matrix_summary_path
            ),
            "metric_profile_id": REPAIRED_METRIC_PROFILE_ID,
            "comparison_order": [
                str(row.get("comparison_id", "")) for row in headline_results
            ],
            "results": headline_results,
        }
        gate_results_payload = {
            "schema_version": ITERATION_GATE_RESULTS_SCHEMA_VERSION,
            "iter_id": self._iter_label(),
            "repaired_matrix_summary_ref": str(
                self.config.repaired_matrix_summary_path
            ),
            "gate_order": [str(row.get("gate", "")) for row in gate_rows],
            "gates": gate_rows,
        }
        _ = write_json(self.config.repaired_comparisons_path, comparisons_payload)
        _ = write_json(self.config.gate_results_path, gate_results_payload)
        _ = write_json(self.config.blocker_verdict_path, blocker_verdict)

        eval_summary_payload = {
            "schema_version": ITERATION_EVAL_SUMMARY_SCHEMA_VERSION,
            "route_id": ITERATION_EVAL_ROUTE_ID,
            "iter_id": self._iter_label(),
            "repaired_matrix_summary_ref": str(
                self.config.repaired_matrix_summary_path
            ),
            "rollout_manifest_ref": str(eval_manifest_path),
            "metric_profile_id": REPAIRED_METRIC_PROFILE_ID,
            "variant_results": [
                self._eval_variant_payload(bundle) for bundle in eval_bundles
            ],
            "observed_metrics": observed_metrics,
            "headline_comparisons_ref": str(self.config.repaired_comparisons_path),
            "gate_results_ref": str(self.config.gate_results_path),
            "blocker_verdict_ref": str(self.config.blocker_verdict_path),
        }
        _ = write_json(self.config.eval_summary_path, eval_summary_payload)
        return {
            "route_id": ITERATION_EVAL_ROUTE_ID,
            "eval_manifest_ref": str(eval_manifest_path),
            "eval_summary_ref": str(self.config.eval_summary_path),
            "repaired_comparisons_ref": str(self.config.repaired_comparisons_path),
            "gate_results_ref": str(self.config.gate_results_path),
            "blocker_verdict_ref": str(self.config.blocker_verdict_path),
            "observed_variant_ids": [
                bundle.repaired_variant_id for bundle in eval_bundles
            ],
        }

    def _write_eval_manifest(self, collection_bundle: CollectionBundle) -> Path:
        collection_manifest = cast(Mapping[str, object], collection_bundle.manifest)
        source_rollout_manifest_ref = str(
            collection_manifest.get("rollout_manifest_ref", "")
        ).strip()
        if not source_rollout_manifest_ref:
            raise ValueError("collection manifest missing rollout_manifest_ref")
        source_rollout_manifest = read_json(source_rollout_manifest_ref)
        selection_payload = {
            "iter_id": self._iter_label(),
            "collection_manifest_ref": str(collection_bundle.manifest_path),
            "source_rollout_manifest_ref": source_rollout_manifest_ref,
            "repaired_variant_ids": [
                "B1_fixed_positive_sft_v2",
                "B0_omit_control_v2",
                "X_shuffled_indicator_v2",
                "C0_recap_informative_positiveinfer_v2",
                "C1_recap_informative_cfg_v2",
            ],
        }
        eval_manifest = build_rollout_manifest_v21(
            authority_id="fresh_rollout_v21_lite",
            manifest_name=f"task10_eval_{self._iter_label()}",
            task_ids=cast(Sequence[int], source_rollout_manifest["task_ids"]),
            seed_manifest=cast(
                Sequence[int] | None, source_rollout_manifest.get("seed_manifest")
            ),
            per_task_seed_manifest=cast(
                Mapping[int | str, Sequence[int]] | None,
                source_rollout_manifest.get("per_task_seed_manifest"),
            ),
            num_trials_per_task=int(
                cast(Any, source_rollout_manifest["num_trials_per_task"])
            ),
            variant_scope=EXPECTED_ALLOWED_VARIANTS,
            selection_policy="task10_iteration_eval_from_collection_v1",
            selection_source=str(collection_bundle.manifest_path),
            selection_source_hash=compute_selection_source_hash(selection_payload),
        )
        self.config.eval_dir.mkdir(parents=True, exist_ok=True)
        return write_rollout_manifest_v21(self.config.eval_manifest_path, eval_manifest)

    def _run_eval_variants(
        self,
        eval_manifest_path: Path,
        policy_bundles: Sequence[PolicyStageBundle],
    ) -> tuple[EvalVariantBundle, ...]:
        stage_map = {bundle.stage: bundle for bundle in policy_bundles}
        raw_specs = (
            EvalVariantSpec(
                repaired_variant_id="B1_fixed_positive_sft_v2",
                carrier_variant_id="fixedadv_relabel8d_control_v1",
                checkpoint_dir=self.config.seed_policy_checkpoint,
                indicator_mode="cfg",
                output_dir=self.config.eval_dir / "B1_fixed_positive_sft_v2",
            ),
            EvalVariantSpec(
                repaired_variant_id="B0_omit_control_v2",
                carrier_variant_id="fixedadv_relabel8d_control_v1",
                checkpoint_dir=stage_map[STAGE_OMIT_CONTROL].checkpoint_dir,
                indicator_mode="cfg",
                output_dir=self.config.eval_dir / "B0_omit_control_v2",
            ),
            EvalVariantSpec(
                repaired_variant_id="X_shuffled_indicator_v2",
                carrier_variant_id="recap_shuffledadv_diag_v1",
                checkpoint_dir=stage_map[STAGE_SHUFFLED_INDICATOR].checkpoint_dir,
                indicator_mode="cfg",
                output_dir=self.config.eval_dir / "X_shuffled_indicator_v2",
            ),
            EvalVariantSpec(
                repaired_variant_id="C0_recap_informative_positiveinfer_v2",
                carrier_variant_id="recap_only_relabel8d_v2",
                checkpoint_dir=stage_map[STAGE_RECAP_INFORMATIVE].checkpoint_dir,
                indicator_mode="positive",
                output_dir=self.config.eval_dir
                / "C0_recap_informative_positiveinfer_v2",
            ),
            EvalVariantSpec(
                repaired_variant_id="C1_recap_informative_cfg_v2",
                carrier_variant_id="recap_only_relabel8d_v2",
                checkpoint_dir=stage_map[STAGE_RECAP_INFORMATIVE].checkpoint_dir,
                indicator_mode="cfg",
                output_dir=self.config.eval_dir / "C1_recap_informative_cfg_v2",
            ),
        )
        specs = self._canonicalize_eval_specs(raw_specs)
        bundles: list[EvalVariantBundle] = []
        for spec in specs:
            eval_runtime_args = self._resolved_runtime_eval_args(
                carrier_variant_id=spec.carrier_variant_id,
                checkpoint_dir=spec.checkpoint_dir,
                indicator_mode=spec.indicator_mode,
            )
            scenario = RolloutEvaluationScenario(
                checkpoint=CheckpointSelection(
                    variant=spec.carrier_variant_id,
                    checkpoint_ref=str(spec.checkpoint_dir),
                ),
                manifest_path=eval_manifest_path,
                metrics=TraceMetricConfig(metric_profile="budget_ladder_v1"),
                output_dir=spec.output_dir,
                indicator_mode=spec.indicator_mode,
                canonical_source_dir=spec.canonical_source_dir,
                resolved_runtime_indicator_mode=(
                    eval_runtime_args[
                        eval_runtime_args.index("--resolved-runtime-indicator-mode") + 1
                    ]
                    if "--resolved-runtime-indicator-mode" in eval_runtime_args
                    else ""
                ),
                resolved_runtime_indicator_source=(
                    eval_runtime_args[
                        eval_runtime_args.index("--resolved-runtime-indicator-source")
                        + 1
                    ]
                    if "--resolved-runtime-indicator-source" in eval_runtime_args
                    else ""
                ),
                resolved_runtime_consumer_mode=(
                    eval_runtime_args[
                        eval_runtime_args.index("--resolved-runtime-consumer-mode") + 1
                    ]
                    if "--resolved-runtime-consumer-mode" in eval_runtime_args
                    else ""
                ),
                resolved_runtime_fixed_indicator_mode=(
                    eval_runtime_args[
                        eval_runtime_args.index(
                            "--resolved-runtime-fixed-indicator-mode"
                        )
                        + 1
                    ]
                    if "--resolved-runtime-fixed-indicator-mode" in eval_runtime_args
                    else ""
                ),
                resolved_runtime_critic_checkpoint_ref=(
                    eval_runtime_args[
                        eval_runtime_args.index(
                            "--resolved-runtime-critic-checkpoint-ref"
                        )
                        + 1
                    ]
                    if "--resolved-runtime-critic-checkpoint-ref" in eval_runtime_args
                    else ""
                ),
                runtime=self.config.runtime,
            )
            exit_code = int(libero_rollout_eval_v21.main(scenario.to_cli_args()))
            if exit_code != 0:
                raise RuntimeError(
                    "rollout eval failed for repaired_variant="
                    + f"{spec.repaired_variant_id} exit_code={exit_code}"
                )
            summary_path = spec.output_dir / "summary.json"
            bundles.append(
                EvalVariantBundle(
                    repaired_variant_id=spec.repaired_variant_id,
                    carrier_variant_id=spec.carrier_variant_id,
                    checkpoint_dir=spec.checkpoint_dir,
                    indicator_mode=spec.indicator_mode,
                    output_dir=spec.output_dir,
                    summary_path=summary_path,
                    summary=read_json(summary_path),
                    effective_runtime_spec_hash=spec.effective_runtime_spec_hash,
                    same_effective_runtime_aliases=spec.same_effective_runtime_aliases,
                )
            )
        return tuple(bundles)

    def _canonicalize_eval_specs(
        self,
        raw_specs: Sequence[EvalVariantSpec],
    ) -> tuple[EvalVariantSpec, ...]:
        alias_groups: dict[str, list[str]] = {}
        spec_hashes: dict[str, str] = {}
        for spec in raw_specs:
            effective_runtime_spec = self._effective_runtime_spec(
                carrier_variant_id=spec.carrier_variant_id,
                checkpoint_dir=spec.checkpoint_dir,
                indicator_mode=spec.indicator_mode,
            )
            effective_hash = effective_runtime_spec_hash(effective_runtime_spec)
            spec_hashes[spec.repaired_variant_id] = effective_hash
            alias_groups.setdefault(effective_hash, []).append(spec.repaired_variant_id)

        canonical_specs: list[EvalVariantSpec] = []
        for spec in raw_specs:
            effective_hash = spec_hashes[spec.repaired_variant_id]
            alias_group = tuple(alias_groups[effective_hash])
            canonical_specs.append(
                EvalVariantSpec(
                    repaired_variant_id=spec.repaired_variant_id,
                    carrier_variant_id=spec.carrier_variant_id,
                    checkpoint_dir=spec.checkpoint_dir,
                    indicator_mode=spec.indicator_mode,
                    output_dir=spec.output_dir,
                    canonical_source_dir=(
                        self.config.eval_dir
                        / "_canonical_rollout_sources"
                        / effective_hash
                        if len(alias_group) > 1
                        else None
                    ),
                    effective_runtime_spec_hash=effective_hash,
                    same_effective_runtime_aliases=alias_group,
                )
            )
        return tuple(canonical_specs)

    def _resolved_runtime_eval_args(
        self,
        *,
        carrier_variant_id: str,
        checkpoint_dir: Path,
        indicator_mode: str,
    ) -> list[str]:
        train_manifest = self._read_optional_json(
            checkpoint_dir / "train_manifest.json"
        )
        checkpoint_provenance = self._read_optional_json(
            checkpoint_dir / "checkpoint_provenance.json"
        )
        config = resolve_runtime_indicator_config(
            requested_indicator_mode=indicator_mode,
            variant=carrier_variant_id,
            train_manifest=train_manifest,
            checkpoint_provenance=checkpoint_provenance,
        )
        args = [
            "--resolved-runtime-indicator-mode",
            str(config.indicator_mode),
            "--resolved-runtime-indicator-source",
            str(config.indicator_source),
            "--resolved-runtime-consumer-mode",
            str(config.consumer_mode),
            "--resolved-runtime-critic-checkpoint-ref",
            str(config.critic_checkpoint_ref),
        ]
        if config.fixed_indicator_mode:
            args.extend(
                [
                    "--resolved-runtime-fixed-indicator-mode",
                    str(config.fixed_indicator_mode),
                ]
            )
        return args

    def _effective_runtime_spec(
        self,
        *,
        carrier_variant_id: str,
        checkpoint_dir: Path,
        indicator_mode: str,
    ) -> dict[str, str]:
        train_manifest = self._read_optional_json(
            checkpoint_dir / "train_manifest.json"
        )
        checkpoint_provenance = self._read_optional_json(
            checkpoint_dir / "checkpoint_provenance.json"
        )
        config = resolve_runtime_indicator_config(
            requested_indicator_mode=indicator_mode,
            variant=carrier_variant_id,
            train_manifest=train_manifest,
            checkpoint_provenance=checkpoint_provenance,
        )
        prompt_bundle = build_runtime_prompt_bundle(
            "runtime prompt surface preview",
            config=config,
        )
        return build_effective_runtime_spec_model(
            variant=carrier_variant_id,
            checkpoint_ref=str(checkpoint_dir),
            runtime_indicator_config=config,
            prompt_surface_bundle=prompt_bundle,
            key_files=CHECKPOINT_INSTANCE_BINDING_KEY_FILES,
            binding_schema_version=CHECKPOINT_INSTANCE_BINDING_SCHEMA_VERSION,
            runtime_spec_schema_version=EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
        )

    def _read_optional_json(self, path: Path) -> dict[str, object] | None:
        if not path.is_file():
            return None
        return read_json(path)

    def _dataset_surface_payload(self, dataset_dir: Path) -> dict[str, object]:
        info = read_json(dataset_dir / "meta" / "info.json")
        return {
            "dataset_dir": str(dataset_dir),
            "route_id": str(
                info.get("route_id") or info.get("merged_dataset_route_id") or ""
            ),
            "schema_version": str(
                info.get("schema_version")
                or info.get("merged_dataset_schema_version")
                or ""
            ),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "episodes_added": info.get("episodes_added"),
            "corrections_added": info.get("corrections_added"),
            "dataset_mix": info.get("dataset_mix"),
            "episodes_jsonl": str(dataset_dir / "meta" / "episodes.jsonl"),
            "episode_lineage_jsonl": info.get("episode_lineage_jsonl"),
        }

    def _critic_manifest_payload(
        self,
        bundle: CriticRetrainBundle,
        *,
        source_dataset_ref: Path,
    ) -> dict[str, object]:
        return {
            "route_id": CRITIC_RETRAIN_ROUTE_ID,
            "config_ref": str(self.config.critic_config),
            "source_dataset_ref": str(source_dataset_ref),
            "source_dataset_surface": self._dataset_surface_payload(source_dataset_ref),
            "output_dir": str(bundle.output_dir),
            "checkpoint_dir": str(bundle.checkpoint_dir),
            "train_summary_ref": str(bundle.train_summary_path),
        }

    def _policy_manifest_payload(
        self,
        bundles: Sequence[PolicyStageBundle],
        critic_checkpoint_dir: Path,
        *,
        source_dataset_ref: Path,
    ) -> dict[str, object]:
        return {
            "route_id": POLICY_RETRAIN_ROUTE_ID,
            "source_dataset_ref": str(source_dataset_ref),
            "source_dataset_surface": self._dataset_surface_payload(source_dataset_ref),
            "critic_checkpoint_ref": str(critic_checkpoint_dir),
            "stage_outputs": [self._policy_stage_payload(bundle) for bundle in bundles],
        }

    def _policy_stage_payload(self, bundle: PolicyStageBundle) -> dict[str, object]:
        return {
            "stage": bundle.stage,
            "repaired_variant_ids": list(bundle.repaired_variant_ids),
            "output_dir": str(bundle.output_dir),
            "checkpoint_dir": str(bundle.checkpoint_dir),
            "train_manifest_ref": str(bundle.train_manifest_path),
            "checkpoint_provenance_ref": str(bundle.checkpoint_provenance_path),
            "runtime_summary_ref": str(bundle.runtime_summary_path),
        }

    def _eval_variant_payload(self, bundle: EvalVariantBundle) -> dict[str, object]:
        required_outputs = self._nested_mapping(bundle.summary, "required_outputs")
        return {
            "repaired_variant_id": bundle.repaired_variant_id,
            "carrier_variant_id": bundle.carrier_variant_id,
            "checkpoint_dir": str(bundle.checkpoint_dir),
            "indicator_mode": bundle.indicator_mode,
            "output_dir": str(bundle.output_dir),
            "summary_ref": str(bundle.summary_path),
            "metric_ladder_summary_ref": str(
                required_outputs.get("metric_ladder_summary")
                or bundle.output_dir / "metric_ladder_summary.json"
            ),
            "per_episode_trace_ref": str(
                required_outputs.get("per_episode_trace")
                or bundle.output_dir / "per_episode_trace.jsonl"
            ),
            "eval_manifest_ref": str(
                required_outputs.get("eval_manifest")
                or bundle.output_dir / "eval_manifest.json"
            ),
            "runtime_prompting": self._nested_mapping(
                bundle.summary, "runtime_prompting"
            ),
            "requested_runtime_prompting": self._nested_mapping(
                bundle.summary, "requested_runtime_prompting"
            ),
            "effective_runtime_spec": self._nested_mapping(
                bundle.summary, "effective_runtime_spec"
            ),
            "rollout_source_binding": self._nested_mapping(
                bundle.summary, "rollout_source_binding"
            ),
            "effective_runtime_spec_hash": bundle.effective_runtime_spec_hash,
            "same_effective_runtime_aliases": list(
                bundle.same_effective_runtime_aliases
            ),
            "observed_metrics": self._extract_metric_points(bundle.summary),
        }

    def _extract_metric_points(
        self,
        summary: Mapping[str, object],
    ) -> dict[str, float]:
        metric_ladder_summary = self._nested_mapping(summary, "metric_ladder_summary")
        metrics = self._nested_mapping(metric_ladder_summary, "metrics")
        resolved: dict[str, float] = {}
        for metric_id, payload in metrics.items():
            if not isinstance(payload, Mapping):
                continue
            point_estimate = payload.get("point_estimate")
            if isinstance(point_estimate, bool) or point_estimate is None:
                continue
            resolved[str(metric_id)] = float(cast(Any, point_estimate))
        return resolved

    def _nested_mapping(
        self,
        payload: Mapping[str, object],
        key: str,
    ) -> dict[str, object]:
        raw = payload.get(key)
        if not isinstance(raw, Mapping):
            return {}
        return {str(child_key): value for child_key, value in raw.items()}

    def _write_tracked_summary(self, manifest: Mapping[str, object]) -> None:
        tracked_summary = self._render_tracked_summary(manifest)
        self.config.tracked_summary_path.parent.mkdir(parents=True, exist_ok=True)
        _ = self.config.tracked_summary_path.write_text(
            tracked_summary, encoding="utf-8"
        )

    def _render_tracked_summary(self, manifest: Mapping[str, object]) -> str:
        iteration_eval = self._nested_mapping(manifest, "iteration_eval")
        critic_retrain = self._nested_mapping(manifest, "critic_retrain")
        policy_retrain = self._nested_mapping(manifest, "policy_retrain")
        comparisons_ref = str(
            iteration_eval.get("repaired_comparisons_ref", "")
        ).strip()
        blocker_verdict_ref = str(iteration_eval.get("blocker_verdict_ref", "")).strip()
        if not comparisons_ref or not blocker_verdict_ref:
            raise ValueError("iteration_eval refs are incomplete for tracked summary")
        comparisons = read_json(comparisons_ref)
        blocker_verdict = read_json(blocker_verdict_ref)
        comparison_rows = cast(
            Sequence[Mapping[str, object]], comparisons.get("results", [])
        )
        policy_stage_outputs = cast(
            Sequence[Mapping[str, object]], policy_retrain.get("stage_outputs", [])
        )
        lines = [
            "# OpenPI RECAP iteration smoke 摘要 v1",
            "",
            "## 1. authority 路径",
            "",
            f"- prerequisite_authority=`{self._relative_path(REPO_ROOT / 'agent/exchange/openpi_libero_official_8d_source_prereq_v1.md')}`",
            f"- iter0_iteration_manifest=`{self._relative_path(self.config.manifest_path)}`",
            f"- repaired_matrix_summary=`{self._relative_path(self.config.repaired_matrix_summary_path)}`",
            "",
            "## 2. critic / policy / eval 产物",
            "",
            f"- critic_retrain=`{self._relative_path(critic_retrain.get('checkpoint_dir'))}`",
            f"- critic_train_summary=`{self._relative_path(critic_retrain.get('train_summary_ref'))}`",
            f"- eval_summary=`{self._relative_path(iteration_eval.get('eval_summary_ref'))}`",
            f"- repaired_comparisons=`{self._relative_path(iteration_eval.get('repaired_comparisons_ref'))}`",
            f"- blocker_verdict=`{self._relative_path(iteration_eval.get('blocker_verdict_ref'))}`",
        ]
        for stage_output in policy_stage_outputs:
            lines.append(
                f"- policy_{stage_output.get('stage')}_checkpoint=`{self._relative_path(stage_output.get('checkpoint_dir'))}`"
            )
        lines.extend(["", "## 3. headline comparison 结论", ""])
        for row in comparison_rows:
            comparison_id = str(row.get("comparison_id", "")).strip()
            status = str(row.get("status", "")).strip()
            lines.append(
                f"- `{comparison_id}`：{status}；结论：{self._headline_conclusion(row)}"
            )
        lines.extend(
            [
                "",
                "## 4. task 11 blocker 结论",
                "",
                f"- task11_ready={str(blocker_verdict.get('ready_for_task11')).lower()}",
                f"- blocking_gates={blocker_verdict.get('blocking_gates', [])}",
                f"- pending_gates={blocker_verdict.get('pending_gates', [])}",
                f"- concise_conclusion={self._blocker_conclusion(blocker_verdict)}",
                "",
                "本页只投影 authority 路径、gate verdict 与简明中文结论；重 JSON / trace / checkpoint 仍留在 `agent/artifacts/openpi_recap_loop/**`。",
                "",
            ]
        )
        return "\n".join(lines)

    def _headline_conclusion(self, row: Mapping[str, object]) -> str:
        comparison_id = str(row.get("comparison_id", "")).strip()
        status = str(row.get("status", "")).strip()
        if comparison_id == "C0_vs_B1":
            if status == "pass":
                return "本轮 C0 在 budget-aware headline 上不落后于 B1。"
            if status == "pending_evidence":
                return "本轮 C0 与 B1 的 repaired headline 证据仍待补齐。"
            return "本轮 C0 在 budget-aware headline 上仍落后于 B1。"
        if comparison_id == "C0_vs_X":
            if status == "pass":
                return "真实 indicator 仍优于 shuffled diagnostic。"
            if status == "pending_evidence":
                return "真实 indicator 与 shuffled diagnostic 的差异仍待补证。"
            return (
                "真实 indicator 还没有在 repaired headline 上压过 shuffled diagnostic。"
            )
        if comparison_id == "B1_vs_B0":
            if status == "pass":
                return "B1 与 B0 在 budget-aware headline 上保持可区分。"
            if status == "pending_evidence":
                return "B1 与 B0 的 budget-aware 对比证据仍待补齐。"
            return "B1 与 B0 目前还没有在 budget-aware headline 上拉开差异。"
        if comparison_id == "C1_vs_C0":
            if status == "pass":
                return "cfg runtime 与 explicit positive runtime 仍保持可比。"
            if status == "pending_evidence":
                return "cfg runtime 与 explicit positive runtime 的可比性仍待补证。"
            return "cfg runtime 还没有达到与 explicit positive runtime 可比的 repaired headline。"
        return str(row.get("decision_text", "")).strip() or "结论未生成。"

    def _blocker_conclusion(self, blocker_verdict: Mapping[str, object]) -> str:
        if blocker_verdict.get("ready_for_task11") is True:
            return "G0-G6 已全部通过，task 11 可以直接消费本轮 verdict。"
        return (
            "task 11 仍被 repaired gates 阻塞；"
            + f"blocking={blocker_verdict.get('blocking_gates', [])}，"
            + f"pending={blocker_verdict.get('pending_gates', [])}。"
        )

    def _relative_path(self, raw_path: object) -> str:
        text = str(raw_path or "").strip()
        if not text:
            return "not_available"
        path = Path(text).expanduser().resolve()
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    def _iter_label(self) -> str:
        text = self.config.iter_id.strip()
        if text.startswith("iter"):
            return text
        return f"iter{text}"


def run_iteration(config: IterationConfig) -> dict[str, object]:
    return IterationWorkflow(config).run()


__all__ = [
    "CRITIC_RETRAIN_ROUTE_ID",
    "CriticRetrainBundle",
    "DEFAULT_REPAIRED_MATRIX_SUMMARY_PATH",
    "DEFAULT_TRACKED_SUMMARY_PATH",
    "EvalVariantBundle",
    "EvalVariantSpec",
    "ITERATION_EVAL_ROUTE_ID",
    "ITERATION_EVAL_SUMMARY_SCHEMA_VERSION",
    "ITERATION_GATE_RESULTS_SCHEMA_VERSION",
    "ITERATION_HEADLINE_SCHEMA_VERSION",
    "IterationConfig",
    "IterationWorkflow",
    "POLICY_RETRAIN_ROUTE_ID",
    "PolicyStageBundle",
    "libero_recap_train",
    "libero_rollout_eval_v21",
    "run_iteration",
    "train_recap_critic",
]
