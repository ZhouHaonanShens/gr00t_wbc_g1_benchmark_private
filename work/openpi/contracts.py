from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalRoute:
    name: str
    help_text: str


@dataclass(frozen=True)
class V2AuthorityPaths:
    rollouts_root: Path
    artifact_dir: Path
    runtime_dir: Path
    log_path: Path
    per_episode: Path
    video_index: Path
    bootstrap_ci: Path
    paired_delta: Path
    eval_manifest: Path
    deviation_notes: Path
    summary: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "V2AuthorityPaths":
        return cls(
            rollouts_root=Path(str(payload["rollouts_root"])),
            artifact_dir=Path(str(payload["artifact_dir"])),
            runtime_dir=Path(str(payload["runtime_dir"])),
            log_path=Path(str(payload["log_path"])),
            per_episode=Path(str(payload["per_episode"])),
            video_index=Path(str(payload["video_index"])),
            bootstrap_ci=Path(str(payload["bootstrap_ci"])),
            paired_delta=Path(str(payload["paired_delta"])),
            eval_manifest=Path(str(payload["eval_manifest"])),
            deviation_notes=Path(str(payload["deviation_notes"])),
            summary=Path(str(payload["summary"])),
        )

    def as_impl_mapping(self) -> dict[str, Path]:
        return {
            "rollouts_root": self.rollouts_root,
            "artifact_dir": self.artifact_dir,
            "runtime_dir": self.runtime_dir,
            "log_path": self.log_path,
            "per_episode": self.per_episode,
            "video_index": self.video_index,
            "bootstrap_ci": self.bootstrap_ci,
            "paired_delta": self.paired_delta,
            "eval_manifest": self.eval_manifest,
            "deviation_notes": self.deviation_notes,
            "summary": self.summary,
        }

    def required_outputs(self) -> dict[str, str]:
        return {
            "per_episode_rollouts": str(self.per_episode),
            "video_index": str(self.video_index),
            "bootstrap_ci": str(self.bootstrap_ci),
            "paired_delta": str(self.paired_delta),
            "eval_manifest": str(self.eval_manifest),
            "deviation_notes": str(self.deviation_notes),
            "summary": str(self.summary),
        }


@dataclass(frozen=True)
class V2EvalRequest:
    variant: str
    baseline_variant: str
    checkpoint_ref: str
    eval_manifest: dict[str, object]
    eval_manifest_hash: str
    eval_manifest_id: str
    paths: V2AuthorityPaths


@dataclass(frozen=True)
class V21AuthorityPaths:
    per_episode_trace: Path
    metric_ladder_summary: Path
    bootstrap_ci: Path
    pairwise_delta: Path
    summary: Path
    eval_manifest: Path
    deviation_notes: Path

    @classmethod
    def from_output_dir(
        cls,
        output_dir: Path,
        *,
        trace_name: str,
        metric_ladder_name: str,
        bootstrap_name: str,
        pairwise_delta_name: str,
        summary_name: str,
        eval_manifest_name: str,
        deviation_notes_name: str,
    ) -> "V21AuthorityPaths":
        return cls(
            per_episode_trace=output_dir / trace_name,
            metric_ladder_summary=output_dir / metric_ladder_name,
            bootstrap_ci=output_dir / bootstrap_name,
            pairwise_delta=output_dir / pairwise_delta_name,
            summary=output_dir / summary_name,
            eval_manifest=output_dir / eval_manifest_name,
            deviation_notes=output_dir / deviation_notes_name,
        )

    def required_outputs(self) -> dict[str, str]:
        return {
            "per_episode_trace": str(self.per_episode_trace),
            "metric_ladder_summary": str(self.metric_ladder_summary),
            "bootstrap_ci": str(self.bootstrap_ci),
            "pairwise_delta": str(self.pairwise_delta),
            "summary": str(self.summary),
            "eval_manifest": str(self.eval_manifest),
            "deviation_notes": str(self.deviation_notes),
        }


@dataclass(frozen=True)
class V21EvalRequest:
    variant: str
    eval_manifest: dict[str, object]
    manifest_metric_profile: str
    checkpoint_ref: str
    checkpoint_mode: str
    raw_checkpoint_dir: str | None
    output_dir: Path
    eval_manifest_hash: str
    eval_manifest_id: str
    runtime_dir: Path
    log_path: Path
    indicator_mode_requested: str
    canonical_source_dir: Path | None
    output_paths: V21AuthorityPaths


@dataclass(frozen=True)
class RequestedRuntimeBinding:
    runtime_prompting: dict[str, str]
    effective_runtime_spec: dict[str, str]
    runtime_indicator_config: object
    prompt_surface_bundle: object


@dataclass(frozen=True)
class ExecutedRuntimeBinding:
    runtime_prompting: dict[str, str]
    effective_runtime_spec: dict[str, str]


@dataclass(frozen=True)
class TraceBuildOutputs:
    trace_rows: list[dict[str, object]]
    scope_audit: dict[str, object]
    metric_ladder_summary: dict[str, object]
    bootstrap_ci: dict[str, object]
    pairwise_delta: dict[str, object]
    deviation_notes: list[str]


@dataclass(frozen=True)
class RolloutSourceRows:
    source_dir: Path
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class EvalBundle:
    summary: dict[str, object]
    eval_manifest_payload: dict[str, object]
    payload_rows: Sequence[Mapping[str, object]]


@dataclass(frozen=True)
class OpenPIRuntimePaths:
    openpi_root: Path
    openpi_venv_python: Path
    serve_policy: Path
    libero_main: Path
    libero_submodule: Path
    config: Path
    runtime_dir: Path
    artifact_dir: Path
    evidence_path: Path | None = None


@dataclass(frozen=True)
class RuntimeServerSpec:
    host: str
    port: int
    checkpoint_dir: str
    server_ready_timeout_s: float
    client_timeout_s: float


@dataclass(frozen=True)
class StockEpisodeRequest:
    task_suite_name: str
    task_id: int
    seed: int
    trial_index: int
    video_path: Path
    host: str
    port: int


@dataclass(frozen=True)
class RuntimeEpisodeRequest:
    task_suite_name: str
    task_id: int
    seed: int
    trial_idx: int
    video_path: Path
    host: str
    port: int
    checkpoint_ref: str
    indicator_mode_requested: str
    runtime_indicator_config: Any


@dataclass(frozen=True)
class EpisodeExecutionResult:
    row: dict[str, object]
    client_log: Path | None = None


@dataclass(frozen=True)
class MaterializedRolloutSourceState:
    source_dir: Path
    rows: list[dict[str, object]]
    completed_keys: tuple[tuple[int, int, int], ...]
