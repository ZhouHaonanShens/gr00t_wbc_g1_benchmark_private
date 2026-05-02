from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from work.openpi.contracts import RuntimeServerSpec

from .spec import build_rollout_input_summary_v21


@dataclass(frozen=True)
class RolloutInputSummaryBuilder:
    schema_version: str
    binding_schema_version: str
    runtime_spec_schema_version: str
    key_files: tuple[Path, ...]

    def build(
        self,
        *,
        variant: str,
        checkpoint_ref: str,
        serve_checkpoint_ref: str,
        serve_checkpoint_mode: str,
        task_suite_name: str,
        task_seed_manifests: Sequence[tuple[int, tuple[int, ...]]],
        manifest: Mapping[str, object],
        num_trials_per_task: int,
        server_spec: RuntimeServerSpec,
        server_log: Path,
        harness_log: Path,
        episode_count: int,
        runtime_indicator_config: object,
        prompt_surface_bundle: object,
        observed_runtime_prompting: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return build_rollout_input_summary_v21(
            schema_version=self.schema_version,
            variant=variant,
            checkpoint_ref=checkpoint_ref,
            serve_checkpoint_ref=serve_checkpoint_ref,
            serve_checkpoint_mode=serve_checkpoint_mode,
            task_suite_name=task_suite_name,
            task_seed_manifests=task_seed_manifests,
            manifest=manifest,
            num_trials_per_task=num_trials_per_task,
            server_spec=server_spec,
            server_log=server_log,
            harness_log=harness_log,
            episode_count=episode_count,
            runtime_indicator_config=runtime_indicator_config,
            prompt_surface_bundle=prompt_surface_bundle,
            observed_runtime_prompting=observed_runtime_prompting,
            key_files=self.key_files,
            binding_schema_version=self.binding_schema_version,
            runtime_spec_schema_version=self.runtime_spec_schema_version,
        )
