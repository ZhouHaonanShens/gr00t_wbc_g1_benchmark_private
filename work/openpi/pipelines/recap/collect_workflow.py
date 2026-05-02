from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


from work.openpi.eval.protocols.manifest import (  # noqa: E402
    EXPECTED_ALLOWED_VARIANTS,
    build_rollout_manifest_v21,
    write_rollout_manifest_v21,
)
from work.openpi.eval import OpenPIEvalApp, RolloutEvaluationScenario  # noqa: E402
from work.openpi.eval.scenarios import CheckpointSelection, TraceMetricConfig  # noqa: E402
from work.openpi.recap.dataset_aggregation import (  # noqa: E402
    CollectionBundle,
    compute_selection_source_hash,
    materialize_collection_bundle,
    read_json,
    read_jsonl,
    resolve_checkpoint_lineage,
)
from work.openpi.sources.libero_official.validate import build_source_prereq_report  # noqa: E402
from work.openpi.runtime.config import (  # noqa: E402
    DEFAULT_RUNTIME_BRIDGE_CONFIG,
    RuntimeBridgeConfig,
)


DEFAULT_VARIANT = "recap_only_relabel8d_v2"
DEFAULT_FIXEDADV_VARIANT = "fixedadv_relabel8d_control_v1"
MAX_ROLLOUT_COLLECTION_ATTEMPTS = 3


class _RolloutEvalMain(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


class _RolloutEvalCompatModule:
    def __init__(self, main: _RolloutEvalMain) -> None:
        self.main: _RolloutEvalMain = main


def _unpatched_rollout_eval_main(argv: list[str] | None = None) -> int:
    _ = argv
    raise RuntimeError(
        "libero_rollout_eval_v21.main compatibility seam was called without a test patch"
    )


libero_rollout_eval_v21 = _RolloutEvalCompatModule(_unpatched_rollout_eval_main)


@dataclass(frozen=True)
class CollectConfig:
    policy_checkpoint: Path
    critic_checkpoint: Path | None
    indicator_mode: str
    task_suite_name: str
    task_ids: tuple[int, ...]
    episodes: int
    output_dir: Path
    demo_dir: Path
    runtime: RuntimeBridgeConfig = DEFAULT_RUNTIME_BRIDGE_CONFIG

    @property
    def rollout_output_dir(self) -> Path:
        return self.output_dir / "rollout_eval_v21"

    @property
    def rollout_manifest_path(self) -> Path:
        return self.output_dir / "rollout_manifest_v21.json"


class CollectWorkflow:
    """Run the canonical RECAP collection stage and emit a collection bundle.

    The workflow validates the canonical source, materializes the rollout
    manifest, executes collection with the canonical eval surface, and writes
    the machine-readable collection artifacts consumed by merge and iteration.
    """

    def __init__(self, config: CollectConfig) -> None:
        self.config: CollectConfig = config

    def run(self) -> CollectionBundle:
        canonical_source_report = self._validate_canonical_source()
        policy_lineage = resolve_checkpoint_lineage(
            self.config.policy_checkpoint,
            explicit_critic_checkpoint=self.config.critic_checkpoint,
        )
        rollout_manifest_path = self._materialize_rollout_manifest()
        rollout_output_dir = self._run_rollout_collection(rollout_manifest_path)
        rollout_input_summary = read_json(
            rollout_output_dir / "_staging" / "rollout_input_summary.json"
        )
        trace_rows = read_jsonl(rollout_output_dir / "per_episode_trace.jsonl")
        return materialize_collection_bundle(
            output_dir=self.config.output_dir,
            canonical_source_report=canonical_source_report,
            policy_lineage=policy_lineage,
            task_suite_name=self.config.task_suite_name,
            task_ids=self.config.task_ids,
            episodes_requested=self.config.episodes,
            rollout_manifest_path=rollout_manifest_path,
            rollout_output_dir=rollout_output_dir,
            rollout_input_summary=rollout_input_summary,
            trace_rows=trace_rows,
        )

    def _validate_canonical_source(self) -> dict[str, object]:
        report = build_source_prereq_report(self.config.demo_dir)
        if report.get("status") != "ready":
            raise RuntimeError(f"canonical demo source blocked: {report}")
        return report

    def _materialize_rollout_manifest(self) -> Path:
        seed_plan = self._build_seed_plan()
        selection_payload = {
            "task_ids": [int(task_id) for task_id in self.config.task_ids],
            "episodes": int(self.config.episodes),
            "indicator_mode": self.config.indicator_mode,
            "seed_plan": {
                str(task_id): list(seeds) for task_id, seeds in seed_plan.items()
            },
        }
        manifest = build_rollout_manifest_v21(
            authority_id="fresh_rollout_v21_lite",
            manifest_name=f"task9_collect_{self.config.output_dir.name}",
            task_ids=self.config.task_ids,
            per_task_seed_manifest={
                str(task_id): tuple(seeds) for task_id, seeds in seed_plan.items()
            },
            num_trials_per_task=2,
            variant_scope=EXPECTED_ALLOWED_VARIANTS,
            selection_policy="task9_round_robin_seed_budget_v1",
            selection_source="task9_collect_request",
            selection_source_hash=compute_selection_source_hash(selection_payload),
        )
        return write_rollout_manifest_v21(self.config.rollout_manifest_path, manifest)

    def _run_rollout_collection(self, rollout_manifest_path: Path) -> Path:
        self.config.rollout_output_dir.mkdir(parents=True, exist_ok=True)
        exit_code = 1
        scenario = RolloutEvaluationScenario(
            checkpoint=CheckpointSelection(
                variant=self._resolve_rollout_variant(),
                checkpoint_ref=str(self.config.policy_checkpoint),
            ),
            manifest_path=rollout_manifest_path,
            metrics=TraceMetricConfig(metric_profile="budget_ladder_v1"),
            output_dir=self.config.rollout_output_dir,
            indicator_mode=self.config.indicator_mode,
            runtime=self.config.runtime,
        )
        for attempt_index in range(1, MAX_ROLLOUT_COLLECTION_ATTEMPTS + 1):
            compat_main = libero_rollout_eval_v21.main
            if compat_main is _unpatched_rollout_eval_main:
                exit_code = int(OpenPIEvalApp().run_rollout_evaluation(scenario))
            else:
                exit_code = int(
                    compat_main(
                        self._build_rollout_eval_compat_argv(
                            rollout_manifest_path=rollout_manifest_path,
                            scenario=scenario,
                        )
                    )
                )
            if exit_code == 0:
                break
            if attempt_index < MAX_ROLLOUT_COLLECTION_ATTEMPTS:
                print(
                    "[WARN] task9 collect retrying rollout_eval_v21 "
                    + f"attempt={attempt_index + 1}/{MAX_ROLLOUT_COLLECTION_ATTEMPTS}",
                    flush=True,
                )
        if exit_code != 0:
            raise RuntimeError(
                "libero_rollout_eval_v21 failed after retries "
                + f"exit_code={exit_code} attempts={MAX_ROLLOUT_COLLECTION_ATTEMPTS}"
            )
        return self.config.rollout_output_dir

    def _build_rollout_eval_compat_argv(
        self,
        *,
        rollout_manifest_path: Path,
        scenario: RolloutEvaluationScenario,
    ) -> list[str]:
        return [
            "--checkpoint-ref",
            str(self.config.policy_checkpoint),
            "--variant",
            str(scenario.checkpoint.variant),
            "--manifest-path",
            str(rollout_manifest_path),
            "--indicator-mode",
            str(scenario.indicator_mode),
            "--output-dir",
            str(self.config.rollout_output_dir),
        ]

    def _resolve_rollout_variant(self) -> str:
        if self.config.indicator_mode == "positive":
            return DEFAULT_FIXEDADV_VARIANT
        return DEFAULT_VARIANT

    def _build_seed_plan(self) -> dict[int, tuple[int, ...]]:
        required_seed_entries = max(
            len(self.config.task_ids),
            (int(self.config.episodes) + 1) // 2,
        )
        allocations: dict[int, list[int]] = {
            int(task_id): [] for task_id in self.config.task_ids
        }
        for index in range(required_seed_entries):
            task_id = int(self.config.task_ids[index % len(self.config.task_ids)])
            allocations[task_id].append(7000 + index)
        return {
            task_id: tuple(seeds) for task_id, seeds in allocations.items() if seeds
        }


def run_collection(config: CollectConfig) -> CollectionBundle:
    return CollectWorkflow(config).run()


__all__ = [
    "CollectConfig",
    "CollectWorkflow",
    "DEFAULT_FIXEDADV_VARIANT",
    "DEFAULT_VARIANT",
    "MAX_ROLLOUT_COLLECTION_ATTEMPTS",
    "libero_rollout_eval_v21",
    "run_collection",
]
