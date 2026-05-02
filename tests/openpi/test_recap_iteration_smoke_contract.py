from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
from typing import cast

from _pytest.monkeypatch import MonkeyPatch
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_ROOT = REPO_ROOT / "tests"
OPENPI_TESTS_ROOT = TESTS_ROOT / "openpi"


def _ensure_namespace_package_path(
    module_name: str, package_path: Path
) -> types.ModuleType:
    module = sys.modules.get(module_name)
    if not isinstance(module, types.ModuleType):
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
    path_attr = getattr(module, "__path__", None)
    normalized_package_path = str(package_path)
    normalized_paths = list(path_attr) if isinstance(path_attr, list) else []
    if normalized_package_path not in normalized_paths:
        normalized_paths.insert(0, normalized_package_path)
    module.__path__ = normalized_paths  # type: ignore[attr-defined]
    return module


tests_pkg = _ensure_namespace_package_path("tests", TESTS_ROOT)
openpi_tests_pkg = _ensure_namespace_package_path("tests.openpi", OPENPI_TESTS_ROOT)
setattr(tests_pkg, "openpi", openpi_tests_pkg)


from work.openpi.recap import dataset_aggregation  # noqa: E402
from work.openpi.recap import data_transforms  # noqa: E402
from work.openpi.recap import prompt_builder  # noqa: E402
from work.openpi.recap.protocol import build_train_runtime_dir  # noqa: E402
from work.openpi.recap.runtime_prompt import resolve_runtime_indicator_config  # noqa: E402
from work.openpi.recap.train_config import resolve_repaired_stage_config  # noqa: E402
from work.recap.lerobot_export import dataset_export  # noqa: E402
import work.openpi.pipelines.recap.iteration as iteration_script  # noqa: E402
import work.openpi.runtime.bridge as libero_native_smoke  # noqa: E402
import work.openpi.eval.workflows.rollout_support as libero_rollout_eval_v21  # noqa: E402
from tests.openpi.test_recap_dataset_aggregation import (  # noqa: E402
    write_recap_ready_demo_sibling,
)
from tests.openpi.test_recap_collection_schema import (  # noqa: E402
    patch_rollout_eval,
    write_demo_source,
    write_policy_checkpoint,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_task0_budget_slice_corrections(
    correction_dir: Path,
    *,
    policy_checkpoint_ref: str,
    critic_checkpoint_ref: str,
) -> Path:
    rows = [
        {
            "correction_id": "iter0_task0_budget_slice_7000_t0",
            "source_trial_id": "task0_seed7000_trial0",
            "task_suite_name": "libero_spatial",
            "task_id": 0,
            "prompt_text": "put the white mug on the left plate and put the yellow and white mug on the right plate",
            "start_step": 0,
            "end_step": 110,
            "policy_checkpoint_ref": policy_checkpoint_ref,
            "critic_checkpoint_ref": critic_checkpoint_ref,
        },
        {
            "correction_id": "iter0_task0_budget_slice_7002_t1_timeout",
            "source_trial_id": "task0_seed7002_trial1",
            "task_suite_name": "libero_spatial",
            "task_id": 0,
            "prompt_text": "put the white mug on the left plate and put the yellow and white mug on the right plate",
            "start_step": 0,
            "end_step": 110,
            "policy_checkpoint_ref": policy_checkpoint_ref,
            "critic_checkpoint_ref": critic_checkpoint_ref,
        },
        {
            "correction_id": "iter0_task0_budget_slice_7004_t1",
            "source_trial_id": "task0_seed7004_trial1",
            "task_suite_name": "libero_spatial",
            "task_id": 0,
            "prompt_text": "put the white mug on the left plate and put the yellow and white mug on the right plate",
            "start_step": 0,
            "end_step": 110,
            "policy_checkpoint_ref": policy_checkpoint_ref,
            "critic_checkpoint_ref": critic_checkpoint_ref,
        },
    ]
    return dataset_aggregation.write_jsonl(
        correction_dir / dataset_aggregation.CORRECTION_SEGMENTS_NAME,
        rows,
    )


def _parse_flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _parse_optional_flag(argv: list[str], name: str) -> str | None:
    if name not in argv:
        return None
    return _parse_flag(argv, name)


def _policy_metrics() -> dict[str, dict[str, float]]:
    return {
        "B1_fixed_positive_sft_v2": {
            "success_rate@0.50_budget": 0.60,
            "success_rate@0.75_budget": 0.80,
            "success_rate@1.00_budget": 0.90,
            "timeout_rate": 0.10,
            "median_first_success_step_fraction": 0.40,
            "throughput_like_score": 7.0,
        },
        "B0_omit_control_v2": {
            "success_rate@0.50_budget": 0.40,
            "success_rate@0.75_budget": 0.65,
            "success_rate@1.00_budget": 0.80,
            "timeout_rate": 0.20,
            "median_first_success_step_fraction": 0.45,
            "throughput_like_score": 5.0,
        },
        "X_shuffled_indicator_v2": {
            "success_rate@0.50_budget": 0.45,
            "success_rate@0.75_budget": 0.60,
            "success_rate@1.00_budget": 0.78,
            "timeout_rate": 0.25,
            "median_first_success_step_fraction": 0.44,
            "throughput_like_score": 4.5,
        },
        "C0_recap_informative_positiveinfer_v2": {
            "success_rate@0.50_budget": 0.60,
            "success_rate@0.75_budget": 0.80,
            "success_rate@1.00_budget": 0.92,
            "timeout_rate": 0.08,
            "median_first_success_step_fraction": 0.38,
            "throughput_like_score": 8.0,
        },
        "C1_recap_informative_cfg_v2": {
            "success_rate@0.50_budget": 0.60,
            "success_rate@0.75_budget": 0.80,
            "success_rate@1.00_budget": 0.92,
            "timeout_rate": 0.08,
            "median_first_success_step_fraction": 0.37,
            "throughput_like_score": 8.5,
        },
    }


def _iter0_blocked_policy_metrics() -> dict[str, dict[str, float]]:
    return {
        "B1_fixed_positive_sft_v2": {
            "success_rate@0.50_budget": 0.20,
            "success_rate@0.75_budget": 0.90,
            "success_rate@1.00_budget": 1.00,
            "timeout_rate": 0.00,
            "median_first_success_step_fraction": 0.5863636363636364,
            "throughput_like_score": 7.651109410864575,
        },
        "B0_omit_control_v2": {
            "success_rate@0.50_budget": 0.30,
            "success_rate@0.75_budget": 0.80,
            "success_rate@1.00_budget": 0.80,
            "timeout_rate": 0.20,
            "median_first_success_step_fraction": 0.60,
            "throughput_like_score": 5.559416261292564,
        },
        "X_shuffled_indicator_v2": {
            "success_rate@0.50_budget": 0.30,
            "success_rate@0.75_budget": 0.80,
            "success_rate@1.00_budget": 0.90,
            "timeout_rate": 0.10,
            "median_first_success_step_fraction": 0.6272727272727273,
            "throughput_like_score": 6.507592190889371,
        },
        "C0_recap_informative_positiveinfer_v2": {
            "success_rate@0.50_budget": 0.20,
            "success_rate@0.75_budget": 0.90,
            "success_rate@1.00_budget": 1.00,
            "timeout_rate": 0.00,
            "median_first_success_step_fraction": 0.5977272727272727,
            "throughput_like_score": 7.722007722007722,
        },
        "C1_recap_informative_cfg_v2": {
            "success_rate@0.50_budget": 0.20,
            "success_rate@0.75_budget": 0.90,
            "success_rate@1.00_budget": 1.00,
            "timeout_rate": 0.00,
            "median_first_success_step_fraction": 0.5977272727272727,
            "throughput_like_score": 7.722007722007722,
        },
    }


def _patch_training_and_eval(
    *,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    call_log: dict[str, list[str]] | None = None,
    metrics_by_variant: dict[str, dict[str, float]] | None = None,
) -> None:
    if metrics_by_variant is None:
        metrics_by_variant = _policy_metrics()
    critic_checkpoint_ref = str((tmp_path / "critic" / "best").resolve())

    def _fake_train_critic_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        dataset_dir = str(Path(_parse_flag(argv, "--dataset-dir")).resolve())
        if call_log is not None:
            call_log["critic_dataset_dir"] = [dataset_dir]
        output_dir = Path(_parse_flag(argv, "--output-dir")).resolve()
        checkpoint_dir = output_dir / "best"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "train_summary.json",
            {
                "checkpoint_dir": str(checkpoint_dir),
                "critic_dir": str(checkpoint_dir),
                "source_dataset_dir": dataset_dir,
                "metrics": {"val_loss": 0.12},
            },
        )
        return 0

    def _fake_policy_train_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        stage = _parse_flag(argv, "--stage")
        dataset_dir = str(Path(_parse_flag(argv, "--dataset-dir")).resolve())
        if call_log is not None:
            call_log.setdefault("policy_dataset_dirs", []).append(dataset_dir)
        output_dir = Path(_parse_flag(argv, "--output-dir")).resolve()
        checkpoint_dir = output_dir / "best"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = build_train_runtime_dir(output_dir, variant=stage)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        prepared_dataset_dir = tmp_path / "prepared_dataset" / stage
        fixed_indicator_mode = {
            "omit_control": "omit",
            "shuffled_indicator": "",
            "recap_informative": "",
        }[stage]
        consumer_mode = {
            "omit_control": "omit",
            "shuffled_indicator": "shuffled_adv_diag",
            "recap_informative": "informative",
        }[stage]
        _write_json(
            checkpoint_dir / "train_manifest.json",
            {
                "stage": stage,
                "training_route": {
                    "consumer_mode": consumer_mode,
                    "fixed_indicator_mode": fixed_indicator_mode,
                    "source_dataset_dir": dataset_dir,
                    "prepared_dataset_dir": str(prepared_dataset_dir.resolve()),
                },
            },
        )
        _write_json(
            checkpoint_dir / "checkpoint_provenance.json",
            {
                "stage": stage,
                "variant_derivation": {
                    "consumer_mode": consumer_mode,
                    "fixed_indicator_mode": fixed_indicator_mode,
                    "source_dataset_dir": dataset_dir,
                    "prepared_dataset_dir": str(prepared_dataset_dir.resolve()),
                },
            },
        )
        _ = (runtime_dir / "train.log").write_text("train log\n", encoding="utf-8")
        _ = (runtime_dir / "real_variant_training.log").write_text(
            "real variant log\n",
            encoding="utf-8",
        )
        _ = (runtime_dir / "real_variant_export" / "export_manifest.json").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        _write_json(
            runtime_dir / "real_variant_export" / "export_manifest.json",
            {
                "stage": stage,
                "export_dir": str((runtime_dir / "real_variant_export").resolve()),
            },
        )
        _ = (runtime_dir / "subprocess_cache" / "cache.txt").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        _ = (runtime_dir / "subprocess_cache" / "cache.txt").write_text(
            "cache\n",
            encoding="utf-8",
        )
        _ = (
            runtime_dir / "upstream_train_checkpoints" / "1" / "state.txt"
        ).parent.mkdir(parents=True, exist_ok=True)
        _ = (runtime_dir / "upstream_train_checkpoints" / "1" / "state.txt").write_text(
            "state\n",
            encoding="utf-8",
        )
        runtime_summary_path = runtime_dir / "summary.json"
        _write_json(
            runtime_summary_path,
            {
                "stage": stage,
                "output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
            },
        )
        return 0

    def _fake_eval_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        output_dir = Path(_parse_flag(argv, "--output-dir")).resolve()
        if output_dir.name == "rollout_eval_v21":
            staging_dir = output_dir / "_staging"
            output_dir.mkdir(parents=True, exist_ok=True)
            staging_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "variant": "fixedadv_relabel8d_control_v1",
                    "task_id": index % 2,
                    "seed": 7000 + index,
                    "trial_idx": 0,
                    "success": success,
                    "first_success_step": 18 if success else None,
                    "executed_steps": 40,
                    "max_steps_resolved": 80,
                    "success_within_50pct_budget": success,
                    "success_within_75pct_budget": success,
                    "timeout_flag": not success,
                    "deviation_notes": [] if success else ["timeout"],
                }
                for index, success in enumerate((True, False, True, True, False, True))
            ]
            _write_json(output_dir / "summary.json", {"collection_stub": True})
            _write_jsonl_path = output_dir / "per_episode_trace.jsonl"
            _write_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            _ = _write_jsonl_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            _write_json(
                staging_dir / "rollout_input_summary.json",
                {
                    "runtime_prompting": {
                        "indicator_mode_requested": _parse_flag(
                            argv, "--indicator-mode"
                        ),
                        "indicator_mode": _parse_flag(argv, "--indicator-mode"),
                        "indicator_source": "cli.indicator_mode",
                        "prompt_text_surface": "canonical_text_indicator",
                        "prompt_text": "put the bowl on the plate\nAdvantage: positive",
                        "critic_checkpoint_ref": critic_checkpoint_ref,
                    }
                },
            )
            return 0
        repaired_variant_id = output_dir.name
        metrics = metrics_by_variant[repaired_variant_id]
        explicit_runtime = {
            "B0_omit_control_v2": {
                "mode": "omit",
                "source": "cfg.fixed_indicator_mode",
                "consumer_mode": "omit",
                "fixed_indicator_mode": "omit",
            },
            "X_shuffled_indicator_v2": {
                "mode": "positive",
                "source": "cfg.consumer_mode.shuffled_adv_diag",
                "consumer_mode": "shuffled_adv_diag",
                "fixed_indicator_mode": "",
            },
            "C0_recap_informative_positiveinfer_v2": {
                "mode": "positive",
                "source": "cli.indicator_mode",
                "consumer_mode": "informative",
                "fixed_indicator_mode": "",
            },
            "C1_recap_informative_cfg_v2": {
                "mode": "positive",
                "source": "cfg.consumer_mode.informative",
                "consumer_mode": "informative",
                "fixed_indicator_mode": "",
            },
        }.get(repaired_variant_id)
        if explicit_runtime is not None:
            assert (
                _parse_flag(argv, "--resolved-runtime-indicator-mode")
                == explicit_runtime["mode"]
            )
            assert (
                _parse_flag(argv, "--resolved-runtime-indicator-source")
                == explicit_runtime["source"]
            )
            assert (
                _parse_flag(argv, "--resolved-runtime-consumer-mode")
                == explicit_runtime["consumer_mode"]
            )
            assert (
                _parse_flag(
                    argv,
                    "--resolved-runtime-critic-checkpoint-ref",
                )
                == "adapter_required"
            )
            if explicit_runtime["fixed_indicator_mode"]:
                assert (
                    _parse_flag(
                        argv,
                        "--resolved-runtime-fixed-indicator-mode",
                    )
                    == explicit_runtime["fixed_indicator_mode"]
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "summary.json",
            {
                "schema_version": "openpi_libero_rollout_eval_summary_v21",
                "variant": _parse_flag(argv, "--variant"),
                "checkpoint_ref": _parse_flag(argv, "--checkpoint-dir"),
                "output_dir": str(output_dir),
                "metric_ladder_summary": {
                    "metrics": {
                        metric_id: {"point_estimate": point_estimate}
                        for metric_id, point_estimate in metrics.items()
                    }
                },
                "required_outputs": {
                    "metric_ladder_summary": str(
                        output_dir / "metric_ladder_summary.json"
                    ),
                    "per_episode_trace": str(output_dir / "per_episode_trace.jsonl"),
                    "eval_manifest": str(output_dir / "eval_manifest.json"),
                },
                "runtime_prompting": {
                    "indicator_mode": _parse_flag(argv, "--indicator-mode"),
                    "indicator_source": "cli.indicator_mode",
                },
            },
        )
        return 0

    monkeypatch.setattr(
        iteration_script.train_recap_critic, "main", _fake_train_critic_main
    )
    monkeypatch.setattr(
        iteration_script.libero_recap_train, "main", _fake_policy_train_main
    )
    monkeypatch.setattr(
        iteration_script.libero_rollout_eval_v21, "main", _fake_eval_main
    )


def test_iteration_script_runs_full_task10_smoke_and_emits_tracked_outputs(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    demo_dir = write_demo_source(tmp_path)
    critic_checkpoint_ref = str((tmp_path / "critic" / "best").resolve())
    seed_policy_checkpoint = write_policy_checkpoint(
        tmp_path,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    patch_rollout_eval(
        monkeypatch=monkeypatch,
        critic_checkpoint_ref=critic_checkpoint_ref,
        success_pattern=(True, False, True, True, False, True),
    )
    call_log: dict[str, list[str]] = {}
    _patch_training_and_eval(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        call_log=call_log,
        metrics_by_variant=_iter0_blocked_policy_metrics(),
    )
    tracked_summary_path = (
        tmp_path / "exchange" / "openpi_recap_iteration_smoke_summary_v1.md"
    )
    output_dir = tmp_path / "iter0"
    merged_dataset_dir = str((output_dir / "dataset").resolve())

    exit_code = iteration_script.main(
        [
            "--iter-id",
            "0",
            "--seed-policy-checkpoint",
            str(seed_policy_checkpoint),
            "--critic-config",
            str(tmp_path / "critic.yaml"),
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--episodes",
            "10",
            "--output-dir",
            str(output_dir),
            "--demo-dir",
            str(demo_dir),
            "--tracked-summary-path",
            str(tracked_summary_path),
        ]
    )

    assert exit_code == 0
    manifest = dataset_aggregation.read_json(
        output_dir / dataset_aggregation.ITERATION_MANIFEST_NAME
    )
    critic_retrain = cast(dict[str, object], manifest["critic_retrain"])
    policy_retrain = cast(dict[str, object], manifest["policy_retrain"])
    iteration_eval = cast(dict[str, object], manifest["iteration_eval"])
    merged_surface = cast(dict[str, object], manifest["merged_dataset_surface"])
    critic_surface = cast(dict[str, object], critic_retrain["source_dataset_surface"])
    policy_surface = cast(dict[str, object], policy_retrain["source_dataset_surface"])
    stage_outputs = cast(list[dict[str, object]], policy_retrain["stage_outputs"])

    assert manifest["iter_id"] == "iter0"
    assert critic_retrain["route_id"] == iteration_script.CRITIC_RETRAIN_ROUTE_ID
    assert policy_retrain["route_id"] == iteration_script.POLICY_RETRAIN_ROUTE_ID
    assert iteration_eval["route_id"] == iteration_script.ITERATION_EVAL_ROUTE_ID
    assert manifest["merged_dataset_ref"] == merged_dataset_dir
    assert merged_surface["episodes_added"] == 6
    assert merged_surface["corrections_added"] == 0
    assert merged_surface["total_episodes"] == 8
    assert critic_retrain["source_dataset_ref"] == merged_dataset_dir
    assert critic_surface["total_episodes"] == 8
    assert critic_surface["episodes_added"] == 6
    assert policy_retrain["source_dataset_ref"] == merged_dataset_dir
    assert policy_surface["total_episodes"] == 8
    assert policy_surface["episodes_added"] == 6
    assert {row["stage"] for row in stage_outputs} == {
        "omit_control",
        "shuffled_indicator",
        "recap_informative",
    }
    assert iteration_eval["observed_variant_ids"] == [
        "B1_fixed_positive_sft_v2",
        "B0_omit_control_v2",
        "X_shuffled_indicator_v2",
        "C0_recap_informative_positiveinfer_v2",
        "C1_recap_informative_cfg_v2",
    ]
    assert call_log["critic_dataset_dir"] == [merged_dataset_dir]
    assert call_log["policy_dataset_dirs"] == [merged_dataset_dir] * 3

    for stage_output in stage_outputs:
        runtime_summary_path = Path(cast(str, stage_output["runtime_summary_ref"]))
        runtime_dir = runtime_summary_path.parent
        stage_manifest = dataset_aggregation.read_json(
            Path(cast(str, stage_output["train_manifest_ref"]))
        )
        training_route = cast(dict[str, object], stage_manifest["training_route"])
        assert training_route["source_dataset_dir"] == merged_dataset_dir

        checkpoint_provenance = dataset_aggregation.read_json(
            Path(cast(str, stage_output["checkpoint_provenance_ref"]))
        )
        variant_derivation = cast(
            dict[str, object], checkpoint_provenance["variant_derivation"]
        )
        assert variant_derivation["source_dataset_dir"] == merged_dataset_dir
        assert Path(cast(str, stage_output["checkpoint_dir"])).is_dir()
        assert runtime_summary_path.is_file()
        assert (runtime_dir / "train.log").is_file()
        assert (runtime_dir / "real_variant_training.log").is_file()
        assert (runtime_dir / "real_variant_export").is_dir()
        assert not (runtime_dir / "subprocess_cache").exists()
        assert not (runtime_dir / "upstream_train_checkpoints").exists()

    comparisons = dataset_aggregation.read_json(
        Path(cast(str, iteration_eval["repaired_comparisons_ref"]))
    )
    blocker_verdict = dataset_aggregation.read_json(
        Path(cast(str, iteration_eval["blocker_verdict_ref"]))
    )
    comparison_rows = cast(list[dict[str, object]], comparisons["results"])

    assert [row["comparison_id"] for row in comparison_rows] == [
        "C0_vs_B1",
        "C0_vs_X",
        "B1_vs_B0",
        "C1_vs_C0",
    ]
    assert [row["status"] for row in comparison_rows] == [
        "pass",
        "fail",
        "pass",
        "pass",
    ]
    assert blocker_verdict["blocking_gates"] == [
        row["gate"] for row in comparison_rows if row["status"] == "fail"
    ]
    assert blocker_verdict["ready_for_task11"] is False

    tracked_summary = tracked_summary_path.read_text(encoding="utf-8")
    assert "headline comparison 结论" in tracked_summary
    assert "task11_ready=false" in tracked_summary
    assert f"blocking_gates={blocker_verdict['blocking_gates']}" in tracked_summary
    assert (
        str(output_dir / dataset_aggregation.ITERATION_MANIFEST_NAME) in tracked_summary
    )
    assert "iter1_g2" not in tracked_summary
    assert "本轮 C0 在 budget-aware headline 上不落后于 B1。" in tracked_summary
    assert (
        "真实 indicator 还没有在 repaired headline 上压过 shuffled diagnostic。"
        in tracked_summary
    )


def test_iteration_script_preserves_corrections_in_manifest_lineage_and_training_surface(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    demo_dir = write_demo_source(tmp_path)
    critic_checkpoint_ref = str((tmp_path / "critic" / "best").resolve())
    seed_policy_checkpoint = write_policy_checkpoint(
        tmp_path,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    _ = write_recap_ready_demo_sibling(
        demo_dir,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    correction_dir = tmp_path / "corrections"
    correction_segments_path = _write_task0_budget_slice_corrections(
        correction_dir,
        policy_checkpoint_ref=str(seed_policy_checkpoint),
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    patch_rollout_eval(
        monkeypatch=monkeypatch,
        critic_checkpoint_ref=critic_checkpoint_ref,
        success_pattern=(True, False, True, True, False, True),
    )
    call_log: dict[str, list[str]] = {}
    _patch_training_and_eval(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        call_log=call_log,
    )
    tracked_summary_path = (
        tmp_path / "exchange" / "openpi_recap_iteration_smoke_summary_v1.md"
    )
    output_dir = tmp_path / "iter0_with_corrections"
    merged_dataset_dir = str((output_dir / "dataset").resolve())

    exit_code = iteration_script.main(
        [
            "--iter-id",
            "0",
            "--seed-policy-checkpoint",
            str(seed_policy_checkpoint),
            "--critic-config",
            str(tmp_path / "critic.yaml"),
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--episodes",
            "10",
            "--output-dir",
            str(output_dir),
            "--demo-dir",
            str(demo_dir),
            "--correction-dir",
            str(correction_dir),
            "--tracked-summary-path",
            str(tracked_summary_path),
        ]
    )

    assert exit_code == 0
    manifest = dataset_aggregation.read_json(
        output_dir / dataset_aggregation.ITERATION_MANIFEST_NAME
    )
    critic_retrain = cast(dict[str, object], manifest["critic_retrain"])
    policy_retrain = cast(dict[str, object], manifest["policy_retrain"])
    merged_surface = cast(dict[str, object], manifest["merged_dataset_surface"])
    critic_surface = cast(dict[str, object], critic_retrain["source_dataset_surface"])
    policy_surface = cast(dict[str, object], policy_retrain["source_dataset_surface"])
    merge_manifest = dataset_aggregation.read_json(
        output_dir / "dataset" / dataset_aggregation.MERGE_MANIFEST_NAME
    )
    raw_lineage_rows = dataset_aggregation.read_jsonl(
        output_dir
        / "dataset"
        / "meta"
        / dataset_aggregation.MERGED_EPISODE_LINEAGE_NAME
    )
    trainer_surface_dir = (
        output_dir / "dataset" / dataset_aggregation.MERGED_RECAP_READY_DATASET_DIRNAME
    )
    trainer_surface_info = dataset_aggregation.read_json(
        trainer_surface_dir / "meta" / "info.json"
    )
    trainer_surface_rows = dataset_aggregation.read_jsonl(
        trainer_surface_dir / "meta" / "episodes.jsonl"
    )

    raw_correction_rows = [
        row
        for row in raw_lineage_rows
        if row.get("source_kind") == "correction_segment"
    ]
    trainer_correction_rows = [
        row
        for row in trainer_surface_rows
        if row.get("source_kind") == "correction_segment"
    ]
    trainer_correction_frames = [
        pd.read_parquet(
            trainer_surface_dir
            / "data"
            / "chunk-000"
            / f"episode_{int(cast(int | str, row['episode_index'])):06d}.parquet"
        )
        for row in trainer_correction_rows
    ]

    assert manifest["merged_dataset_ref"] == merged_dataset_dir
    assert manifest["corrections_added"] == 3
    assert cast(dict[str, object], manifest["dataset_mix"])["correction"] == {
        "segments": 3,
        "forced_positive": True,
    }
    assert merged_surface["episodes_added"] == 6
    assert merged_surface["corrections_added"] == 3
    assert merged_surface["total_episodes"] == 11
    assert critic_surface["corrections_added"] == 3
    assert critic_surface["total_episodes"] == 11
    assert policy_surface["corrections_added"] == 3
    assert policy_surface["total_episodes"] == 11
    assert merge_manifest["corrections_added"] == 3
    assert merge_manifest["correction_segments_ref"] == str(correction_segments_path)
    assert trainer_surface_info["corrections_added"] == 3
    assert len(raw_correction_rows) == 3
    assert len(trainer_correction_rows) == 3
    assert {row["source_trial_id"] for row in raw_correction_rows} == {
        "task0_seed7000_trial0",
        "task0_seed7002_trial1",
        "task0_seed7004_trial1",
    }
    assert call_log["critic_dataset_dir"] == [merged_dataset_dir]
    assert call_log["policy_dataset_dirs"] == [merged_dataset_dir] * 3
    assert all(row["indicator_I"] == 1 for row in raw_correction_rows)
    assert all(row["is_correction"] is True for row in raw_correction_rows)
    assert all(row["forced_positive_indicator"] is True for row in raw_correction_rows)
    assert all(row["indicator_I"] == 1 for row in trainer_correction_rows)
    assert all(row["is_correction"] is True for row in trainer_correction_rows)
    assert all(
        row["forced_positive_indicator"] is True for row in trainer_correction_rows
    )
    assert all(
        row["human_correction_override_applied"] is True
        for row in trainer_correction_rows
    )
    assert all(
        int(frame.loc[0, "recap_m2.indicator_I"]) == 1
        for frame in trainer_correction_frames
    )
    assert all(
        float(frame.loc[0, "recap_m2.advantage_input"]) > 0.0
        for frame in trainer_correction_frames
    )
    assert all(
        str(frame.loc[0, "recap_m2.prompt_conditioned"]).endswith("Advantage: positive")
        for frame in trainer_correction_frames
    )


def test_iteration_policy_retrain_bypasses_rematerialization_for_prebuilt_merged_surface(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    demo_dir = write_demo_source(tmp_path)
    critic_checkpoint_ref = str((tmp_path / "critic" / "best").resolve())
    seed_policy_checkpoint = write_policy_checkpoint(
        tmp_path,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    _ = write_recap_ready_demo_sibling(
        demo_dir,
        critic_checkpoint_ref=critic_checkpoint_ref,
        omit_prompt_feature_keys=True,
    )
    patch_rollout_eval(
        monkeypatch=monkeypatch,
        critic_checkpoint_ref=critic_checkpoint_ref,
        success_pattern=(True, False, True, True, False, True),
    )
    _patch_training_and_eval(tmp_path=tmp_path, monkeypatch=monkeypatch)

    materialize_calls: list[int | None] = []

    def _recording_materialize_dataset(
        official_dataset_dir: str | Path,
        output_dir: str | Path,
        *,
        episode_limit: int | None = None,
        critic_checkpoint_dir: str | Path | None = None,
    ) -> dict[str, object]:
        materialize_calls.append(episode_limit)
        assert critic_checkpoint_dir is not None
        source_dir = Path(official_dataset_dir).resolve()
        prepared_dir = Path(output_dir).resolve()
        source_episodes = dataset_aggregation.read_jsonl(
            source_dir / "meta" / "episodes.jsonl"
        )
        selected_episodes = (
            source_episodes[:episode_limit]
            if episode_limit is not None
            else source_episodes
        )
        tasks = dataset_aggregation.read_jsonl(source_dir / "meta" / "tasks.jsonl")
        prepared_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            prepared_dir / "meta" / "info.json",
                {
                    "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
                    "route_id": "official_native_8d_recap_relabels_v1",
                    "source_dataset_dir": str(source_dir),
                    "source_dataset_name": source_dir.name,
                    "task_text_field": dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
                    "carrier_route": dataset_export.EXPORTER_CARRIER_ROUTE,
                    "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
                    "prompt_source_field": dataset_export.EXPORTER_PROMPT_SOURCE_FIELD,
                    "prompt_route": prompt_builder.PHASE1_PROMPT_ROUTE,
                    "conditioning_mode": prompt_builder.CONDITIONING_MODE,
                    "total_episodes": len(selected_episodes),
                    "total_frames": len(selected_episodes),
                    "total_tasks": len(tasks),
                    "total_videos": 0,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 10,
                "splits": {"train": f"0:{len(selected_episodes)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "recap_advantage_input_contract": {
                    "contract_version": "full_recap_continuous_adv_v2",
                    "critic_checkpoint_ref": str(Path(critic_checkpoint_dir).resolve()),
                    "epsilon_source": "per_task_quantile:prompt_raw:q=0.7",
                    "indicator_dropout_p": 0.3,
                    "human_correction_override": True,
                },
                "features": {
                    "observation.images.ego_view": {"dtype": "binary", "shape": [1]},
                    "observation.images.wrist_view": {"dtype": "binary", "shape": [1]},
                    "observation.state": {"dtype": "float32", "shape": [8]},
                    "action": {"dtype": "float32", "shape": [7]},
                    "timestamp": {"dtype": "float32", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "index": {"dtype": "int64", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                    "annotation.human.task_description": {
                        "dtype": "int64",
                        "shape": [1],
                    },
                    "annotation.human.action.task_description": {
                        "dtype": "int64",
                        "shape": [1],
                    },
                    "recap_m2.t": {"dtype": "int64", "shape": [1]},
                    "recap_m2.return_G": {"dtype": "float32", "shape": [1]},
                    "recap_m2.value_V": {"dtype": "float32", "shape": [1]},
                    "recap_m2.advantage_A": {"dtype": "float32", "shape": [1]},
                    "recap_m2.advantage_input": {"dtype": "float32", "shape": [1]},
                    "recap_m2.epsilon_l": {"dtype": "float32", "shape": [1]},
                    "recap_m2.indicator_I": {"dtype": "int64", "shape": [1]},
                    "recap_m2.prompt_raw": {"dtype": "string", "shape": [1]},
                    "recap_m2.prompt_conditioned": {"dtype": "string", "shape": [1]},
                },
            },
        )
        _write_json(
            prepared_dir / "meta" / "stats.json",
            {
                "observation.images.ego_view": {},
                "observation.images.wrist_view": {},
                "observation.state": {},
                "action": {},
                "timestamp": {},
                "frame_index": {},
                "episode_index": {},
                "index": {},
                "task_index": {},
            },
        )
        _write_json(
            prepared_dir / "meta" / "modality.json",
            {
                "video": {
                    "ego_view": {"original_key": "observation.images.ego_view"},
                    "wrist_view": {"original_key": "observation.images.wrist_view"},
                },
                "state": {
                    "libero_state": {
                        "start": 0,
                        "end": 8,
                        "original_key": "observation.state",
                    }
                },
                "action": {
                    "libero_action": {
                        "start": 0,
                        "end": 7,
                        "original_key": "action",
                    }
                },
                "annotation": {
                    "human.task_description": {
                        "original_key": "annotation.human.task_description"
                    },
                    "human.action.task_description": {
                        "original_key": "annotation.human.action.task_description"
                    },
                },
            },
        )
        dataset_aggregation.write_jsonl(prepared_dir / "meta" / "tasks.jsonl", tasks)
        dataset_aggregation.write_jsonl(
            prepared_dir / "meta" / "episodes.jsonl", selected_episodes
        )
        _write_json(
            prepared_dir / "materialization_report.json",
            {
                "schema_version": "openpi_libero_official_8d_recap_relabels_report_v1",
                "route_id": "official_native_8d_recap_relabels_v1",
                "final_status": "materialized",
            },
        )
        for row in selected_episodes:
            episode_index = int(cast(int | str, row["episode_index"]))
            tasks_raw = cast(list[object], row["tasks"])
            prompt_raw = str(tasks_raw[0])
            frame = pd.DataFrame(
                {
                    "action": [[0.1] * 7],
                    "episode_index": [episode_index],
                    "observation.state": [[float(episode_index)] * 8],
                    "observation.images.ego_view": [b"ego"],
                    "observation.images.wrist_view": [b"wrist"],
                    "timestamp": [0.0],
                    "frame_index": [0],
                    "index": [episode_index],
                    "task_index": [int(cast(int | str, row.get("task_id", 0)))],
                    "annotation.human.task_description": [0],
                    "annotation.human.action.task_description": [0],
                    "recap_m2.t": [0],
                    "recap_m2.return_G": [0.0],
                    "recap_m2.value_V": [0.0],
                    "recap_m2.advantage_A": [0.0],
                    "recap_m2.advantage_input": [0.0],
                    "recap_m2.epsilon_l": [0.0],
                    "recap_m2.indicator_I": [0],
                    "recap_m2.prompt_raw": [prompt_raw],
                    "recap_m2.prompt_conditioned": [prompt_raw],
                }
            )
            parquet_path = (
                prepared_dir
                / "data"
                / "chunk-000"
                / f"episode_{episode_index:06d}.parquet"
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(parquet_path, engine="pyarrow", index=False)
        return {"final_status": "materialized"}

    monkeypatch.setattr(
        data_transforms, "materialize_dataset", _recording_materialize_dataset
    )

    def _isolated_runtime_dir(output_dir: Path, *, variant: str) -> Path:
        resolved_output_dir = Path(output_dir).resolve()
        return tmp_path / "runtime_logs" / f"{variant}_{resolved_output_dir.name}_train"

    monkeypatch.setattr(
        iteration_script, "build_train_runtime_dir", _isolated_runtime_dir
    )

    def _fake_policy_train_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        stage = _parse_flag(argv, "--stage")
        dataset_dir = Path(_parse_flag(argv, "--dataset-dir")).resolve()
        critic_checkpoint_dir = Path(_parse_flag(argv, "--critic-checkpoint")).resolve()
        output_dir = Path(_parse_flag(argv, "--output-dir")).resolve()
        checkpoint_dir = output_dir / "best"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prepared = data_transforms.prepare_stage_training_dataset(
            dataset_dir=dataset_dir,
            stage_config=resolve_repaired_stage_config(stage),
            critic_checkpoint_dir=critic_checkpoint_dir,
        )
        runtime_dir = iteration_script.build_train_runtime_dir(
            output_dir, variant=stage
        )
        fixed_indicator_mode = {
            "omit_control": "omit",
            "shuffled_indicator": "",
            "recap_informative": "",
        }[stage]
        consumer_mode = {
            "omit_control": "omit",
            "shuffled_indicator": "shuffled_adv_diag",
            "recap_informative": "informative",
        }[stage]
        _write_json(
            checkpoint_dir / "train_manifest.json",
            {
                "stage": stage,
                "training_route": {
                    "consumer_mode": consumer_mode,
                    "fixed_indicator_mode": fixed_indicator_mode,
                    "critic_checkpoint_ref": "adapter_required",
                    "source_dataset_dir": str(dataset_dir),
                    "prepared_dataset_dir": str(prepared.dataset_dir),
                },
            },
        )
        _write_json(
            checkpoint_dir / "checkpoint_provenance.json",
            {
                "stage": stage,
                "variant_derivation": {
                    "consumer_mode": consumer_mode,
                    "fixed_indicator_mode": fixed_indicator_mode,
                    "critic_checkpoint_ref": "adapter_required",
                    "source_dataset_dir": str(dataset_dir),
                    "prepared_dataset_dir": str(prepared.dataset_dir),
                },
            },
        )
        _write_json(
            runtime_dir / "summary.json",
            {
                "stage": stage,
                "output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "source_dataset_dir": str(dataset_dir),
                "prepared_dataset_dir": str(prepared.dataset_dir),
            },
        )
        _ = (runtime_dir / "train.log").write_text("train log\n", encoding="utf-8")
        _ = (runtime_dir / "real_variant_training.log").write_text(
            "real variant log\n",
            encoding="utf-8",
        )
        _write_json(
            runtime_dir / "real_variant_export" / "export_manifest.json",
            {
                "stage": stage,
                "export_dir": str((runtime_dir / "real_variant_export").resolve()),
            },
        )
        return 0

    monkeypatch.setattr(
        iteration_script.libero_recap_train, "main", _fake_policy_train_main
    )

    tracked_summary_path = (
        tmp_path / "exchange" / "openpi_recap_iteration_smoke_summary_v1.md"
    )
    output_dir = tmp_path / "iter0_full_prepare_contract"
    exit_code = iteration_script.main(
        [
            "--iter-id",
            "0",
            "--seed-policy-checkpoint",
            str(seed_policy_checkpoint),
            "--critic-config",
            str(tmp_path / "critic.yaml"),
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--episodes",
            "10",
            "--output-dir",
            str(output_dir),
            "--demo-dir",
            str(demo_dir),
            "--tracked-summary-path",
            str(tracked_summary_path),
        ]
    )

    assert exit_code == 0
    manifest = dataset_aggregation.read_json(
        output_dir / dataset_aggregation.ITERATION_MANIFEST_NAME
    )
    merged_dataset_info = dataset_aggregation.read_json(
        output_dir / "dataset" / "meta" / "info.json"
    )
    trainer_surface_info = dataset_aggregation.read_json(
        output_dir
        / "dataset"
        / dataset_aggregation.MERGED_RECAP_READY_DATASET_DIRNAME
        / "meta"
        / "info.json"
    )
    policy_retrain = cast(dict[str, object], manifest["policy_retrain"])
    merged_dataset_dir = str((output_dir / "dataset").resolve())

    assert policy_retrain["source_dataset_ref"] == merged_dataset_dir
    assert merged_dataset_info[dataset_aggregation.MERGED_RECAP_READY_DATASET_REF_KEY]
    assert (
        cast(dict[str, object], trainer_surface_info["recap_advantage_input_contract"])[
            "epsilon_source"
        ]
        == "per_task_quantile:prompt_raw:q=0.7"
    )
    assert (
        cast(dict[str, object], trainer_surface_info["recap_advantage_input_contract"])[
            "indicator_dropout_p"
        ]
        == 0.3
    )
    assert (
        cast(dict[str, object], trainer_surface_info["recap_advantage_input_contract"])[
            "human_correction_override"
        ]
        is True
    )
    assert materialize_calls == []

    for stage_output in cast(list[dict[str, object]], policy_retrain["stage_outputs"]):
        stage = str(stage_output["stage"])
        train_manifest = dataset_aggregation.read_json(
            Path(cast(str, stage_output["train_manifest_ref"]))
        )
        runtime_summary = dataset_aggregation.read_json(
            Path(cast(str, stage_output["runtime_summary_ref"]))
        )
        prepared_dataset_dir = str(
            cast(dict[str, object], train_manifest["training_route"])[
                "prepared_dataset_dir"
            ]
        )
        expected_prepared_dataset_dir = (
            Path(merged_dataset_dir)
            / dataset_aggregation.MERGED_RECAP_READY_DATASET_DIRNAME
        )
        if stage == "recap_informative":
            expected_prepared_dataset_dir = expected_prepared_dataset_dir.with_name(
                expected_prepared_dataset_dir.name
                + data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DIR_SUFFIX
            )
        assert prepared_dataset_dir == str(expected_prepared_dataset_dir.resolve())
        assert "episodes_8" not in prepared_dataset_dir
        assert "episodes_full" not in prepared_dataset_dir
        assert runtime_summary["prepared_dataset_dir"] == prepared_dataset_dir


def test_client_runtime_config_prefers_explicit_resolved_fields_over_stock_cfg_defaults() -> (
    None
):
    args = libero_native_smoke._build_parser().parse_args(
        [
            "--internal-mode",
            "client",
            "--indicator-mode",
            "cfg",
            "--resolved-runtime-indicator-mode",
            "omit",
            "--resolved-runtime-indicator-source",
            "cfg.fixed_indicator_mode",
            "--resolved-runtime-consumer-mode",
            "omit",
            "--resolved-runtime-fixed-indicator-mode",
            "omit",
            "--resolved-runtime-critic-checkpoint-ref",
            "/tmp/critic/best",
            "--client-summary-out",
            "/tmp/client-summary.json",
            "--client-video-out",
            "/tmp/client-video.mp4",
        ]
    )

    config = libero_native_smoke._resolve_client_runtime_indicator_config(args)

    assert config.requested_indicator_mode == "cfg"
    assert config.indicator_mode == "omit"
    assert config.indicator_source == "cfg.fixed_indicator_mode"
    assert config.consumer_mode == "omit"
    assert config.fixed_indicator_mode == "omit"
    assert config.critic_checkpoint_ref == "/tmp/critic/best"


def test_rollout_source_runtime_mismatch_detector_catches_b0_and_x_regressions() -> (
    None
):
    b0_expected = resolve_runtime_indicator_config(
        requested_indicator_mode="cfg",
        variant="fixedadv_relabel8d_control_v1",
        checkpoint_provenance={
            "variant_derivation": {
                "consumer_mode": "omit",
                "fixed_indicator_mode": "omit",
            }
        },
    )
    b0_prompt = libero_rollout_eval_v21.build_runtime_prompt_bundle(
        "runtime prompt surface preview",
        config=b0_expected,
    )
    b0_bad_rows = [
        {
            "indicator_mode_requested": "cfg",
            "indicator_mode": "positive",
            "indicator_source": "cfg.consumer_mode.informative_adv",
            "prompt_text_surface": "canonical_text_indicator",
            "critic_checkpoint_ref": "not_applicable",
            "prompt_route": "recap_conditioned_prompt_token_v1",
            "conditioning_mode": "prompt_text_only",
            "source_prompt_field": "prompt_raw",
            "consumer_mode": "informative_adv",
            "fixed_indicator_mode": "",
        }
    ]
    b0_source_dir = Path(tempfile.mkdtemp(prefix="b0_runtime_mismatch_"))
    _write_jsonl_path = (
        b0_source_dir / libero_rollout_eval_v21.V2_INPUT_PER_EPISODE_NAME
    )
    _write_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    _ = _write_jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in b0_bad_rows) + "\n",
        encoding="utf-8",
    )
    _write_json(
        b0_source_dir / "rollout_input_summary.json",
        {"runtime_prompting": b0_bad_rows[0]},
    )

    assert libero_rollout_eval_v21._materialized_rollout_source_has_runtime_mismatch(
        source_dir=b0_source_dir,
        runtime_indicator_config=b0_expected,
        prompt_surface_bundle=b0_prompt,
        log_path=b0_source_dir / "eval.log",
    )

    x_expected = resolve_runtime_indicator_config(
        requested_indicator_mode="cfg",
        variant="recap_shuffledadv_diag_v1",
        checkpoint_provenance={"variant_derivation": {"consumer_mode": "shuffled"}},
    )
    x_prompt = libero_rollout_eval_v21.build_runtime_prompt_bundle(
        "runtime prompt surface preview",
        config=x_expected,
    )
    x_bad_rows = [
        {
            "indicator_mode_requested": "cfg",
            "indicator_mode": "positive",
            "indicator_source": "cfg.consumer_mode.informative_adv",
            "prompt_text_surface": "canonical_text_indicator",
            "critic_checkpoint_ref": "not_applicable",
            "prompt_route": "recap_conditioned_prompt_token_v1",
            "conditioning_mode": "prompt_text_only",
            "source_prompt_field": "prompt_raw",
            "consumer_mode": "informative_adv",
            "fixed_indicator_mode": "",
        }
    ]
    x_source_dir = Path(tempfile.mkdtemp(prefix="x_runtime_mismatch_"))
    _write_jsonl_path = x_source_dir / libero_rollout_eval_v21.V2_INPUT_PER_EPISODE_NAME
    _write_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    _ = _write_jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in x_bad_rows) + "\n",
        encoding="utf-8",
    )
    _write_json(
        x_source_dir / "rollout_input_summary.json",
        {"runtime_prompting": x_bad_rows[0]},
    )

    assert libero_rollout_eval_v21._materialized_rollout_source_has_runtime_mismatch(
        source_dir=x_source_dir,
        runtime_indicator_config=x_expected,
        prompt_surface_bundle=x_prompt,
        log_path=x_source_dir / "eval.log",
    )


def test_iteration_eval_reuses_same_effective_runtime_alias_source_for_c1_vs_c0(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    demo_dir = write_demo_source(tmp_path)
    critic_checkpoint_ref = str((tmp_path / "critic" / "best").resolve())
    seed_policy_checkpoint = write_policy_checkpoint(
        tmp_path,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    patch_rollout_eval(
        monkeypatch=monkeypatch,
        critic_checkpoint_ref=critic_checkpoint_ref,
        success_pattern=(True, False, True, True, False, True),
    )
    _patch_training_and_eval(tmp_path=tmp_path, monkeypatch=monkeypatch)

    eval_calls: dict[str, dict[str, object]] = {}
    alias_metrics_by_source: dict[str, dict[str, float]] = {}

    def _fake_alias_aware_eval_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        output_dir = Path(_parse_flag(argv, "--output-dir")).resolve()
        repaired_variant_id = output_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        if repaired_variant_id == "rollout_eval_v21":
            staging_dir = output_dir / "_staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "variant": "fixedadv_relabel8d_control_v1",
                    "task_id": index % 2,
                    "seed": 7000 + index,
                    "trial_idx": 0,
                    "success": success,
                    "first_success_step": 18 if success else None,
                    "executed_steps": 40,
                    "max_steps_resolved": 80,
                    "success_within_50pct_budget": success,
                    "success_within_75pct_budget": success,
                    "timeout_flag": not success,
                    "deviation_notes": [] if success else ["timeout"],
                }
                for index, success in enumerate((True, False, True, True, False, True))
            ]
            _write_json(output_dir / "summary.json", {"collection_stub": True})
            _write_jsonl_path = output_dir / "per_episode_trace.jsonl"
            _write_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            _ = _write_jsonl_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            _write_json(
                staging_dir / "rollout_input_summary.json",
                {
                    "runtime_prompting": {
                        "indicator_mode_requested": _parse_flag(
                            argv, "--indicator-mode"
                        ),
                        "indicator_mode": _parse_flag(argv, "--indicator-mode"),
                        "indicator_source": "cli.indicator_mode",
                        "prompt_text_surface": "canonical_text_indicator",
                        "prompt_text": (
                            "put the bowl on the plate\nAdvantage: positive"
                        ),
                        "critic_checkpoint_ref": critic_checkpoint_ref,
                    }
                },
            )
            return 0

        canonical_source_dir = _parse_optional_flag(argv, "--canonical-source-dir")
        eval_calls[repaired_variant_id] = {
            "canonical_source_dir": canonical_source_dir,
            "indicator_mode": _parse_flag(argv, "--indicator-mode"),
            "resolved_runtime_indicator_mode": _parse_flag(
                argv, "--resolved-runtime-indicator-mode"
            ),
        }
        metrics_map = {
            "B1_fixed_positive_sft_v2": {
                "success_rate@0.50_budget": 0.55,
                "success_rate@0.75_budget": 0.80,
                "success_rate@1.00_budget": 0.90,
                "timeout_rate": 0.10,
                "median_first_success_step_fraction": 0.40,
                "throughput_like_score": 7.0,
            },
            "B0_omit_control_v2": {
                "success_rate@0.50_budget": 0.35,
                "success_rate@0.75_budget": 0.60,
                "success_rate@1.00_budget": 0.75,
                "timeout_rate": 0.25,
                "median_first_success_step_fraction": 0.47,
                "throughput_like_score": 4.5,
            },
            "X_shuffled_indicator_v2": {
                "success_rate@0.50_budget": 0.40,
                "success_rate@0.75_budget": 0.62,
                "success_rate@1.00_budget": 0.78,
                "timeout_rate": 0.22,
                "median_first_success_step_fraction": 0.43,
                "throughput_like_score": 5.0,
            },
        }
        if repaired_variant_id in {
            "C0_recap_informative_positiveinfer_v2",
            "C1_recap_informative_cfg_v2",
        }:
            if canonical_source_dir is not None:
                metrics = alias_metrics_by_source.setdefault(
                    canonical_source_dir,
                    {
                        "success_rate@0.50_budget": 0.60,
                        "success_rate@0.75_budget": 0.80,
                        "success_rate@1.00_budget": 0.90,
                        "timeout_rate": 0.08,
                        "median_first_success_step_fraction": 0.38,
                        "throughput_like_score": 8.0,
                    },
                )
            elif repaired_variant_id == "C0_recap_informative_positiveinfer_v2":
                metrics = {
                    "success_rate@0.50_budget": 0.60,
                    "success_rate@0.75_budget": 0.80,
                    "success_rate@1.00_budget": 0.90,
                    "timeout_rate": 0.08,
                    "median_first_success_step_fraction": 0.38,
                    "throughput_like_score": 8.0,
                }
            else:
                metrics = {
                    "success_rate@0.50_budget": 0.20,
                    "success_rate@0.75_budget": 0.75,
                    "success_rate@1.00_budget": 0.80,
                    "timeout_rate": 0.20,
                    "median_first_success_step_fraction": 0.42,
                    "throughput_like_score": 6.0,
                }
        else:
            metrics = metrics_map[repaired_variant_id]

        _write_json(
            output_dir / "summary.json",
            {
                "schema_version": "openpi_libero_rollout_eval_summary_v21",
                "variant": _parse_flag(argv, "--variant"),
                "checkpoint_ref": _parse_flag(argv, "--checkpoint-dir"),
                "output_dir": str(output_dir),
                "source_rollout_dir": canonical_source_dir
                or str(output_dir / "_staging"),
                "metric_ladder_summary": {
                    "metrics": {
                        metric_id: {"point_estimate": point_estimate}
                        for metric_id, point_estimate in metrics.items()
                    }
                },
                "required_outputs": {
                    "metric_ladder_summary": str(
                        output_dir / "metric_ladder_summary.json"
                    ),
                    "per_episode_trace": str(output_dir / "per_episode_trace.jsonl"),
                    "eval_manifest": str(output_dir / "eval_manifest.json"),
                },
                "runtime_prompting": {
                    "indicator_mode_requested": "positive",
                    "indicator_mode": "positive",
                    "indicator_source": "cli.indicator_mode",
                    "prompt_text_surface": "canonical_text_indicator",
                    "critic_checkpoint_ref": "adapter_required",
                },
                "requested_runtime_prompting": {
                    "indicator_mode_requested": _parse_flag(argv, "--indicator-mode"),
                    "indicator_mode": _parse_flag(
                        argv, "--resolved-runtime-indicator-mode"
                    ),
                    "indicator_source": _parse_flag(
                        argv, "--resolved-runtime-indicator-source"
                    ),
                    "prompt_text_surface": "canonical_text_indicator",
                    "critic_checkpoint_ref": "adapter_required",
                },
                "rollout_source_binding": {
                    "source_selection_mode": (
                        "explicit_canonical_source_dir"
                        if canonical_source_dir is not None
                        else "variant_output_staging"
                    ),
                    "requested_runtime_prompting_matches_executed": False,
                    "effective_runtime_spec_matches_requested": True,
                },
            },
        )
        return 0

    monkeypatch.setattr(
        iteration_script.libero_rollout_eval_v21, "main", _fake_alias_aware_eval_main
    )

    tracked_summary_path = (
        tmp_path / "exchange" / "openpi_recap_iteration_smoke_summary_v1.md"
    )
    output_dir = tmp_path / "iter0_alias_fix"
    exit_code = iteration_script.main(
        [
            "--iter-id",
            "0",
            "--seed-policy-checkpoint",
            str(seed_policy_checkpoint),
            "--critic-config",
            str(tmp_path / "critic.yaml"),
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--episodes",
            "10",
            "--output-dir",
            str(output_dir),
            "--demo-dir",
            str(demo_dir),
            "--tracked-summary-path",
            str(tracked_summary_path),
        ]
    )

    assert exit_code == 0
    c0_call = eval_calls["C0_recap_informative_positiveinfer_v2"]
    c1_call = eval_calls["C1_recap_informative_cfg_v2"]
    assert c0_call["canonical_source_dir"] is not None
    assert c0_call["canonical_source_dir"] == c1_call["canonical_source_dir"]

    repaired_comparisons = dataset_aggregation.read_json(
        output_dir / "eval" / "repaired_headline_comparisons.json"
    )
    g4_row = next(
        row
        for row in cast(list[dict[str, object]], repaired_comparisons["results"])
        if row["comparison_id"] == "C1_vs_C0"
    )
    assert g4_row["status"] == "pass"

    eval_summary = dataset_aggregation.read_json(
        output_dir / "eval" / "eval_summary.json"
    )
    variant_results = {
        str(row["repaired_variant_id"]): row
        for row in cast(list[dict[str, object]], eval_summary["variant_results"])
    }
    c0_result = cast(
        dict[str, object], variant_results["C0_recap_informative_positiveinfer_v2"]
    )
    c1_result = cast(dict[str, object], variant_results["C1_recap_informative_cfg_v2"])
    assert c0_result["same_effective_runtime_aliases"] == [
        "C0_recap_informative_positiveinfer_v2",
        "C1_recap_informative_cfg_v2",
    ]
    assert c1_result["same_effective_runtime_aliases"] == [
        "C0_recap_informative_positiveinfer_v2",
        "C1_recap_informative_cfg_v2",
    ]
    assert (
        c0_result["effective_runtime_spec_hash"]
        == c1_result["effective_runtime_spec_hash"]
    )
    c0_binding = cast(dict[str, object], c0_result["rollout_source_binding"])
    c1_binding = cast(dict[str, object], c1_result["rollout_source_binding"])
    assert c0_binding["source_selection_mode"] == "explicit_canonical_source_dir"
    assert c1_binding["source_selection_mode"] == "explicit_canonical_source_dir"


def test_canonical_eval_source_changes_when_checkpoint_instance_changes_at_same_path(
    tmp_path: Path,
) -> None:
    checkpoint_dir = write_policy_checkpoint(
        tmp_path,
        critic_checkpoint_ref=str((tmp_path / "critic" / "best").resolve()),
    )
    checkpoint_payload_path = checkpoint_dir / "checkpoint.json"
    _write_json(
        checkpoint_payload_path,
        {
            "schema_version": "openpi_libero_recap_checkpoint_payload_v1",
            "created_at": "2026-04-07T18:29:53",
            "instance_token": "first",
        },
    )
    workflow = iteration_script.IterationWorkflow(
        iteration_script.IterationConfig(
            iter_id="0",
            seed_policy_checkpoint=checkpoint_dir,
            critic_checkpoint=None,
            indicator_mode="cfg",
            task_suite_name="libero_spatial",
            task_ids="0,1",
            episodes=10,
            output_dir=tmp_path / "iter0_binding_check",
            demo_dir=tmp_path / "demo",
            correction_dir=None,
            critic_config=None,
            repaired_matrix_summary_path=tmp_path / "repaired_matrix_summary.json",
            tracked_summary_path=tmp_path / "tracked_summary.md",
        )
    )
    raw_specs = (
        iteration_script.EvalVariantSpec(
            repaired_variant_id="C0_recap_informative_positiveinfer_v2",
            carrier_variant_id="recap_only_relabel8d_v2",
            checkpoint_dir=checkpoint_dir,
            indicator_mode="positive",
            output_dir=workflow.config.eval_dir
            / "C0_recap_informative_positiveinfer_v2",
        ),
        iteration_script.EvalVariantSpec(
            repaired_variant_id="C1_recap_informative_cfg_v2",
            carrier_variant_id="recap_only_relabel8d_v2",
            checkpoint_dir=checkpoint_dir,
            indicator_mode="cfg",
            output_dir=workflow.config.eval_dir / "C1_recap_informative_cfg_v2",
        ),
    )

    first_specs = workflow._canonicalize_eval_specs(raw_specs)

    _write_json(
        checkpoint_payload_path,
        {
            "schema_version": "openpi_libero_recap_checkpoint_payload_v1",
            "created_at": "2026-04-07T18:31:07",
            "instance_token": "second",
        },
    )

    second_specs = workflow._canonicalize_eval_specs(raw_specs)

    first_by_variant = {spec.repaired_variant_id: spec for spec in first_specs}
    second_by_variant = {spec.repaired_variant_id: spec for spec in second_specs}
    first_c0 = first_by_variant["C0_recap_informative_positiveinfer_v2"]
    first_c1 = first_by_variant["C1_recap_informative_cfg_v2"]
    second_c0 = second_by_variant["C0_recap_informative_positiveinfer_v2"]
    second_c1 = second_by_variant["C1_recap_informative_cfg_v2"]

    assert first_c0.effective_runtime_spec_hash == first_c1.effective_runtime_spec_hash
    assert (
        second_c0.effective_runtime_spec_hash == second_c1.effective_runtime_spec_hash
    )
    assert first_c0.canonical_source_dir == first_c1.canonical_source_dir
    assert second_c0.canonical_source_dir == second_c1.canonical_source_dir
    assert first_c0.canonical_source_dir is not None
    assert second_c0.canonical_source_dir is not None
    assert first_c0.effective_runtime_spec_hash != second_c0.effective_runtime_spec_hash
    assert first_c0.canonical_source_dir != second_c0.canonical_source_dir
