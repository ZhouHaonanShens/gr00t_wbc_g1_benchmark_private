from __future__ import annotations

import argparse
from dataclasses import dataclass

from work.openpi.eval.scenarios import StockSmokeScenario
from work.openpi.runtime import RuntimePathsBuilder, run_stock_smoke_harness


@dataclass
class StockSmokeWorkflow:
    """Translate a stock-smoke scenario into the runtime harness contract.

    This workflow stays intentionally small so the default stock smoke remains a
    readable bridge from scenario defaults to the runtime harness inputs and
    artifact paths.
    """

    scenario: StockSmokeScenario

    def run(self) -> int:
        args = argparse.Namespace(
            task_suite_name=self.scenario.task_suite_name,
            task_id=self.scenario.task_id,
            num_trials_per_task=self.scenario.num_trials_per_task,
            seed=self.scenario.seed,
            checkpoint_dir=self.scenario.checkpoint_ref,
            host=self.scenario.runtime.host,
            port=self.scenario.runtime.port,
            indicator_mode=self.scenario.indicator_mode,
            server_ready_timeout_s=self.scenario.runtime.server_ready_timeout_s,
            client_timeout_s=self.scenario.runtime.client_timeout_s,
            video_fps=self.scenario.runtime.video_fps,
            internal_mode=None,
            probe_out="",
            client_summary_out="",
            client_video_out="",
            trial_index=0,
            resolved_runtime_indicator_mode="",
            resolved_runtime_indicator_source="",
            resolved_runtime_consumer_mode="",
            resolved_runtime_fixed_indicator_mode="",
            resolved_runtime_critic_checkpoint_ref="",
        )
        paths = RuntimePathsBuilder(
            artifact_root=self.scenario.runtime.artifact_root,
            runtime_root=self.scenario.runtime.runtime_root,
            evidence_path=self.scenario.runtime.evidence_path,
        ).build()
        return run_stock_smoke_harness(args, paths=paths)


__all__ = ["StockSmokeWorkflow"]
