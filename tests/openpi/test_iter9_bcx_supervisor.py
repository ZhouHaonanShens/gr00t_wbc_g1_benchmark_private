from __future__ import annotations


def _snapshot_with_wrapper_abort(message: str) -> dict[str, object]:
    return {
        "lane": "B",
        "runtime_dir": "/tmp/runtime/B",
        "process_lines": [],
        "wrapper_launch_log": {
            "exists": True,
            "tail": f"prefix\n{message}\n",
        },
        "upstream_checkpoint": {
            "latest_step": 200,
        },
        "artifacts": {
            "loss_decomposition.jsonl": {"lines": None},
            "training_failure.json": {"exists": False},
            "training_timeout_report.json": {"exists": False},
            "driver_timeout_report_fallback.json": {"exists": False},
        },
    }


def _snapshot_with_driver_exit(exit_code: int) -> dict[str, object]:
    snapshot = _snapshot_with_wrapper_abort("")
    snapshot["wrapper_launch_log"] = {"exists": True, "tail": ""}
    snapshot["driver_exit"] = {
        "exists": True,
        "code": exit_code,
        "raw": str(exit_code),
    }
    return snapshot


def test_status_summary_marks_wrapper_preflight_abort_as_failed(monkeypatch, tmp_path) -> None:
    from work.openpi.pipelines.recap import iter9_bcx_supervisor as supervisor_mod

    monkeypatch.setattr(supervisor_mod, "REPO_ROOT", tmp_path)
    supervisor = supervisor_mod.Iter9BCXSupervisor(
        interval_s=60,
        auto_fix=True,
        root_dir_name="unit-supervisor",
    )

    summary = supervisor._status_summary(
        _snapshot_with_wrapper_abort(
            "BLOCK_INFRA: DISK_PRESSURE_ABORT free=60GB min=100GB"
        ),
        None,
    )

    assert summary["status"] == "failed"
    assert summary["failure_source"] == "wrapper_preflight"
    assert "DISK_PRESSURE_ABORT" in str(summary["failure_message"])


def test_should_autoheal_skips_wrapper_preflight_failures(monkeypatch, tmp_path) -> None:
    from work.openpi.pipelines.recap import iter9_bcx_supervisor as supervisor_mod

    monkeypatch.setattr(supervisor_mod, "REPO_ROOT", tmp_path)
    supervisor = supervisor_mod.Iter9BCXSupervisor(
        interval_s=60,
        auto_fix=True,
        root_dir_name="unit-supervisor",
    )

    assert (
        supervisor._should_autoheal(
            {
                "status": "failed",
                "failure_source": "wrapper_preflight",
            }
        )
        is False
    )
    assert (
        supervisor._should_autoheal(
            {
                "status": "failed",
                "failure_source": "training_artifact",
            }
        )
        is True
    )


def test_status_summary_marks_nonzero_driver_exit_as_failed(monkeypatch, tmp_path) -> None:
    from work.openpi.pipelines.recap import iter9_bcx_supervisor as supervisor_mod

    monkeypatch.setattr(supervisor_mod, "REPO_ROOT", tmp_path)
    supervisor = supervisor_mod.Iter9BCXSupervisor(
        interval_s=60,
        auto_fix=True,
        root_dir_name="unit-supervisor",
    )

    summary = supervisor._status_summary(_snapshot_with_driver_exit(4), None)

    assert summary["status"] == "failed"
    assert summary["failure_source"] == "driver_exit"
    assert summary["failure_message"] == "exit_code=4"


def test_status_summary_marks_wrapper_abnormal_exit_without_driver_exit(monkeypatch, tmp_path) -> None:
    from work.openpi.pipelines.recap import iter9_bcx_supervisor as supervisor_mod

    monkeypatch.setattr(supervisor_mod, "REPO_ROOT", tmp_path)
    supervisor = supervisor_mod.Iter9BCXSupervisor(
        interval_s=60,
        auto_fix=True,
        root_dir_name="unit-supervisor",
    )

    summary = supervisor._status_summary(
        {
            "lane": "B",
            "runtime_dir": "/tmp/runtime/B",
            "process_lines": [],
            "wrapper_launch_log": {
                "exists": True,
                "tail": "header\npython: command not found\n",
            },
            "driver_exit": {"exists": False, "code": None, "raw": ""},
            "upstream_checkpoint": {"latest_step": None},
            "artifacts": {
                "loss_decomposition.jsonl": {"lines": None},
                "training_failure.json": {"exists": False},
                "training_timeout_report.json": {"exists": False},
                "driver_timeout_report_fallback.json": {"exists": False},
            },
        },
        None,
    )

    assert summary["status"] == "failed"
    assert summary["failure_source"] == "wrapper_exit"
    assert summary["failure_message"] == "python: command not found"


def test_status_summary_does_not_mark_dead_lane_as_progressing(monkeypatch, tmp_path) -> None:
    from work.openpi.pipelines.recap import iter9_bcx_supervisor as supervisor_mod

    monkeypatch.setattr(supervisor_mod, "REPO_ROOT", tmp_path)
    supervisor = supervisor_mod.Iter9BCXSupervisor(
        interval_s=60,
        auto_fix=True,
        root_dir_name="unit-supervisor",
    )

    summary = supervisor._status_summary(
        {
            "lane": "B",
            "runtime_dir": "/tmp/runtime/B",
            "process_lines": [],
            "wrapper_launch_log": {"exists": True, "tail": ""},
            "upstream_checkpoint": {"latest_step": 200},
            "driver_exit": {"exists": False, "code": None, "raw": ""},
            "artifacts": {
                "loss_decomposition.jsonl": {"lines": 5},
                "training_failure.json": {"exists": False},
                "training_timeout_report.json": {"exists": False},
                "driver_timeout_report_fallback.json": {"exists": False},
            },
        },
        {"latest_step": 100, "loss_lines": 3},
    )

    assert summary["status"] == "idle"


def test_supervisor_restores_lane_state_from_existing_file(monkeypatch, tmp_path) -> None:
    import json

    from work.openpi.pipelines.recap import iter9_bcx_supervisor as supervisor_mod

    monkeypatch.setattr(supervisor_mod, "REPO_ROOT", tmp_path)
    root_dir = tmp_path / "agent" / "runtime_logs" / "unit-supervisor"
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "lane_state.json").write_text(
        json.dumps(
            {
                "schema_version": "iter9_bcx_supervisor_state_v1",
                "lanes": {
                    "B": {
                        "lane": "B",
                        "current_runtime_dir": "agent/runtime_logs/custom/B",
                        "restart_count": 11,
                        "last_action": "resume_restart",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    supervisor = supervisor_mod.Iter9BCXSupervisor(
        interval_s=60,
        auto_fix=True,
        root_dir_name="unit-supervisor",
    )

    assert supervisor.lanes["B"].current_runtime_dir == "agent/runtime_logs/custom/B"
    assert supervisor.lanes["B"].restart_count == 11
    assert supervisor.lanes["B"].last_action == "resume_restart"
