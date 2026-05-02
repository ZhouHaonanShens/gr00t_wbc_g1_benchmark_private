"""Canonical scenarios for OpenPI evaluation entrypoints.

Readers should start here when they want to understand which checkpoints,
manifests, runtime settings, and output roots the mainline workflows use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from work.openpi.runtime.config import DEFAULT_RUNTIME_BRIDGE_CONFIG, RuntimeBridgeConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CheckpointSelection:
    """Describe which checkpoint a workflow should evaluate."""

    variant: str
    checkpoint_ref: str | None = None
    checkpoint_source: str | None = None
    baseline_variant: str | None = None


@dataclass(frozen=True)
class ArtifactLayout:
    """Describe where tracked-gate artifacts should be written."""

    output_root: Path
    runtime_root: Path | None = None


@dataclass(frozen=True)
class TraceMetricConfig:
    """Select the metric profile for rollout trace aggregation."""

    metric_profile: str


@dataclass(frozen=True)
class TrackedGateEvaluationScenario:
    """Static configuration for the tracked-gate authority lane."""

    checkpoint: CheckpointSelection
    eval_manifest_path: Path
    artifacts: ArtifactLayout

    def to_cli_args(self) -> list[str]:
        if self.checkpoint.checkpoint_ref is None:
            raise ValueError("tracked gate evaluation requires checkpoint_ref")
        if self.checkpoint.baseline_variant is None:
            raise ValueError("tracked gate evaluation requires baseline_variant")
        return [
            "--variant",
            self.checkpoint.variant,
            "--checkpoint-dir",
            self.checkpoint.checkpoint_ref,
            "--eval-manifest",
            str(self.eval_manifest_path),
            "--baseline-variant",
            self.checkpoint.baseline_variant,
            "--output-root",
            str(self.artifacts.output_root),
        ]


@dataclass(frozen=True)
class RolloutEvaluationScenario:
    """Static configuration for the canonical rollout-evaluation workflow."""

    checkpoint: CheckpointSelection
    manifest_path: Path
    metrics: TraceMetricConfig
    output_dir: Path
    runtime: RuntimeBridgeConfig = DEFAULT_RUNTIME_BRIDGE_CONFIG
    indicator_mode: str = "cfg"
    canonical_source_dir: Path | None = None
    resolved_runtime_indicator_mode: str = ""
    resolved_runtime_indicator_source: str = ""
    resolved_runtime_consumer_mode: str = ""
    resolved_runtime_fixed_indicator_mode: str = ""
    resolved_runtime_critic_checkpoint_ref: str = ""

    def to_cli_args(self) -> list[str]:
        args: list[str] = [
            "--variant",
            self.checkpoint.variant,
            "--manifest",
            str(self.manifest_path),
            "--metric-profile",
            self.metrics.metric_profile,
            "--output-dir",
            str(self.output_dir),
            "--indicator-mode",
            self.indicator_mode,
        ]
        if self.checkpoint.checkpoint_source:
            args.extend(["--checkpoint-source", self.checkpoint.checkpoint_source])
        elif self.checkpoint.checkpoint_ref:
            args.extend(["--checkpoint-dir", self.checkpoint.checkpoint_ref])
        else:
            raise ValueError(
                "rollout evaluation requires either checkpoint_source or checkpoint_ref"
            )
        if self.resolved_runtime_indicator_mode:
            args.extend(
                [
                    "--resolved-runtime-indicator-mode",
                    self.resolved_runtime_indicator_mode,
                    "--resolved-runtime-indicator-source",
                    self.resolved_runtime_indicator_source,
                    "--resolved-runtime-consumer-mode",
                    self.resolved_runtime_consumer_mode,
                    "--resolved-runtime-critic-checkpoint-ref",
                    self.resolved_runtime_critic_checkpoint_ref,
                ]
            )
            if self.resolved_runtime_fixed_indicator_mode:
                args.extend(
                    [
                        "--resolved-runtime-fixed-indicator-mode",
                        self.resolved_runtime_fixed_indicator_mode,
                    ]
                )
        if self.canonical_source_dir is not None:
            args.extend(["--canonical-source-dir", str(self.canonical_source_dir)])
        return args


@dataclass(frozen=True)
class StockSmokeScenario:
    """Static configuration for the baseline stock smoke workflow."""

    checkpoint_ref: str
    task_suite_name: str
    task_id: int
    num_trials_per_task: int
    seed: int
    host: str
    port: int
    runtime: RuntimeBridgeConfig = DEFAULT_RUNTIME_BRIDGE_CONFIG
    indicator_mode: str = "cfg"


DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO = TrackedGateEvaluationScenario(
    checkpoint=CheckpointSelection(
        variant="stock_libero_ref_v1",
        checkpoint_ref="gs://openpi-assets/checkpoints/pi05_libero",
        baseline_variant="stock_libero_ref_v1",
    ),
    eval_manifest_path=REPO_ROOT
    / "work"
    / "openpi"
    / "eval"
    / "manifests"
    / "eval_manifest_rollout_lite_v2.json",
    artifacts=ArtifactLayout(
        output_root=REPO_ROOT / "agent" / "artifacts" / "openpi_libero_v2",
    ),
)

DEFAULT_STOCK_ROLLOUT_SCENARIO = RolloutEvaluationScenario(
    checkpoint=CheckpointSelection(
        variant="stock_libero_ref_v1",
        checkpoint_source="stock",
    ),
    manifest_path=REPO_ROOT
    / "work"
    / "openpi"
    / "eval"
    / "manifests"
    / "smoke_trace_v21.json",
    metrics=TraceMetricConfig(metric_profile="budget_ladder_v1"),
    output_dir=REPO_ROOT / "agent" / "artifacts" / "openpi_libero_v21" / "smoke_trace_default",
    indicator_mode="cfg",
)

DEFAULT_ROLLOUT_EVALUATION_SCENARIO = DEFAULT_STOCK_ROLLOUT_SCENARIO

DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO = RolloutEvaluationScenario(
    checkpoint=CheckpointSelection(
        variant="recap_only_relabel8d_v2",
        checkpoint_ref=str(
            REPO_ROOT
            / "agent"
            / "artifacts"
            / "openpi_recap_loop"
            / "iter0"
            / "policy_retrain"
            / "recap_informative"
            / "best"
        ),
    ),
    manifest_path=REPO_ROOT
    / "work"
    / "openpi"
    / "eval"
    / "manifests"
    / "smoke_trace_recap_v21.json",
    metrics=TraceMetricConfig(metric_profile="budget_ladder_v1"),
    output_dir=REPO_ROOT
    / "agent"
    / "artifacts"
    / "openpi_recap_loop"
    / "default_best_rollout",
    indicator_mode="positive",
)

DEFAULT_STOCK_SMOKE_SCENARIO = StockSmokeScenario(
    checkpoint_ref="gs://openpi-assets/checkpoints/pi05_libero",
    task_suite_name="libero_spatial",
    task_id=0,
    num_trials_per_task=1,
    seed=7,
    host="127.0.0.1",
    port=8000,
)


__all__ = [
    "ArtifactLayout",
    "CheckpointSelection",
    "DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO",
    "DEFAULT_ROLLOUT_EVALUATION_SCENARIO",
    "DEFAULT_STOCK_SMOKE_SCENARIO",
    "DEFAULT_STOCK_ROLLOUT_SCENARIO",
    "DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO",
    "RolloutEvaluationScenario",
    "StockSmokeScenario",
    "TraceMetricConfig",
    "TrackedGateEvaluationScenario",
]
