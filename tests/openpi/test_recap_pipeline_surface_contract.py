from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import work.openpi.pipelines.recap.collect as collect_script  # noqa: E402
import work.openpi.pipelines.recap.collect_workflow as collect_workflow  # noqa: E402
import work.openpi.pipelines.recap.iteration as iteration_script  # noqa: E402
import work.openpi.pipelines.recap.iteration_workflow as iteration_workflow  # noqa: E402


def test_recap_facades_reexport_workflow_implementation_from_new_modules() -> None:
    assert (
        collect_script.CollectWorkflow.__module__
        == "work.openpi.pipelines.recap.collect_workflow"
    )
    assert (
        iteration_script.IterationWorkflow.__module__
        == "work.openpi.pipelines.recap.iteration_workflow"
    )
    assert collect_script.libero_rollout_eval_v21 is collect_workflow.libero_rollout_eval_v21
    assert iteration_script.train_recap_critic is iteration_workflow.train_recap_critic
    assert iteration_script.libero_recap_train is iteration_workflow.libero_recap_train
    assert iteration_script.libero_rollout_eval_v21 is iteration_workflow.libero_rollout_eval_v21


def test_collect_facade_main_keeps_cli_surface_and_calls_run_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, collect_script.CollectConfig] = {}

    def _fake_run_collection(
        config: collect_script.CollectConfig,
    ) -> object:
        observed["config"] = config
        return object()

    monkeypatch.setattr(collect_script, "run_collection", _fake_run_collection)

    exit_code = collect_script.main(
        [
            "--policy-checkpoint",
            str(tmp_path / "policy" / "best"),
            "--indicator-mode",
            "cfg",
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--episodes",
            "4",
            "--output-dir",
            str(tmp_path / "collect_out"),
            "--demo-dir",
            str(tmp_path / "demo"),
        ]
    )

    assert exit_code == 0
    config = observed["config"]
    assert isinstance(config, collect_script.CollectConfig)
    assert config.policy_checkpoint == (tmp_path / "policy" / "best").resolve()
    assert config.task_ids == (0, 1)
    assert config.episodes == 4


def test_iteration_facade_main_keeps_cli_surface_and_calls_run_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, iteration_script.IterationConfig] = {}

    def _fake_run_iteration(
        config: iteration_script.IterationConfig,
    ) -> dict[str, object]:
        observed["config"] = config
        return {"iter_id": config.iter_id}

    monkeypatch.setattr(iteration_script, "run_iteration", _fake_run_iteration)

    exit_code = iteration_script.main(
        [
            "--iter-id",
            "iter0",
            "--seed-policy-checkpoint",
            str(tmp_path / "policy" / "best"),
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--episodes",
            "6",
            "--output-dir",
            str(tmp_path / "iter_out"),
            "--demo-dir",
            str(tmp_path / "demo"),
        ]
    )

    assert exit_code == 0
    config = observed["config"]
    assert isinstance(config, iteration_script.IterationConfig)
    assert config.seed_policy_checkpoint == (tmp_path / "policy" / "best").resolve()
    assert config.demo_dir == (tmp_path / "demo").resolve()
    assert config.episodes == 6
