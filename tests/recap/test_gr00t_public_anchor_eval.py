from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_public_anchor_eval


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_runtime_layout(tmp_path: Path, mode: str) -> tuple[Path, Path, Path, Path]:
    runtime_dir = tmp_path / "runtime"
    videos_dir = tmp_path / "videos"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return (
        runtime_dir,
        videos_dir,
        runtime_dir / f"{mode}_00_server.log",
        runtime_dir / f"{mode}_10_client.log",
    )


def _fake_execution(
    config: gr00t_public_anchor_eval.EvalConfig,
    runtime_dir: Path,
    server_log: Path,
    client_log: Path,
    artifacts_videos: Path,
) -> dict[str, Any]:
    del artifacts_videos
    return {
        "runtime_status": "COMPLETED",
        "mode": config.mode,
        "requested_n_episodes": config.n_episodes,
        "requested_n_envs": config.n_envs,
        "requested_seed_list": list(config.seed_list),
        "completed_episodes": config.n_episodes,
        "successes": [True, False][: config.n_episodes]
        + [True] * max(0, config.n_episodes - 2),
        "success_count": min(config.n_episodes, max(1, config.n_episodes - 1)),
        "success_rate": float(
            min(config.n_episodes, max(1, config.n_episodes - 1)) / config.n_episodes
        ),
        "episode_summaries": [
            {
                "episode_index": index,
                "seed": seed,
                "env_slot": index % max(1, config.n_envs),
                "outer_steps": 3,
                "success": bool(index != 1),
                "motion_l2": 0.42,
                "controller_saturation_detected": False,
                "zero_motion_detected": False,
                "trajectory_fingerprint": f"fingerprint-{index}",
                "action_stats": {
                    "action.right_arm": {
                        "total_count": 21,
                        "mean_abs": 0.25,
                        "max_abs": 0.8,
                        "nonfinite_count": 0,
                        "zero_fraction": 0.0,
                        "saturated_fraction": 0.0,
                    }
                },
            }
            for index, seed in enumerate(config.seed_list)
        ],
        "global_action_stats": {
            "action.right_arm": {
                "total_count": 210,
                "mean_abs": 0.25,
                "max_abs": 0.8,
                "nonfinite_count": 0,
                "zero_fraction": 0.0,
                "saturated_fraction": 0.0,
            }
        },
        "scope_guard": gr00t_public_anchor_eval._build_scope_guard(config),
        "systemic_break_flags": [],
        "systemic_break_details": {},
        "modality_config_keys": ["action", "language", "state", "video"],
        "server_action_horizon": 30,
        "runtime_dir": str(runtime_dir),
        "server_log_path": str(server_log),
        "client_log_path": str(client_log),
        "video_dir": str(runtime_dir / "tmp_videos"),
        "archived_video_dir": str(runtime_dir / "archived_videos"),
        "server_reused_existing": False,
        "server_spawned_by_runner": True,
    }


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_public_anchor_eval.main(["--help"])
    assert exc_info.value.code == 0


def test_stack_batch_converts_object_string_arrays_to_fixed_string_dtype() -> None:
    value_a = np.asarray(["pick the apple"], dtype=object)
    value_b = np.asarray(["place on the plate"], dtype=object)

    stacked = gr00t_public_anchor_eval._stack_batch([value_a, value_b])

    assert isinstance(stacked, np.ndarray)
    assert stacked.dtype.kind in {"U", "S"}
    assert stacked.shape == (2, 1)
    assert stacked.tolist() == [["pick the apple"], ["place on the plate"]]


def test_normalize_policy_observation_casts_state_and_video_dtypes() -> None:
    obs = {
        "annotation.human.task_description": np.asarray(
            ["pick the apple"], dtype=object
        ),
        "state.left_leg": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "q": np.asarray([1, 2, 3], dtype=np.int64),
        "video.ego_view": np.asarray([[[[1, 2, 3]]]], dtype=np.int16),
        "ego_view_image": np.asarray([[[1, 2, 3]]], dtype=np.int16),
    }

    normalized = gr00t_public_anchor_eval._normalize_policy_observation(obs)

    prompt_value = normalized["annotation.human.task_description"]
    state_arr = np.asarray(normalized["state.left_leg"])
    q_arr = np.asarray(normalized["q"])
    video_arr = np.asarray(normalized["video.ego_view"])
    image_arr = np.asarray(normalized["ego_view_image"])

    assert isinstance(prompt_value, list)
    assert prompt_value == ["pick the apple"]
    assert state_arr.dtype == np.float32
    assert q_arr.dtype == np.float32
    assert video_arr.dtype == np.uint8
    assert image_arr.dtype == np.uint8


def test_ensure_server_ready_reclaims_stale_gr00t_port_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = gr00t_public_anchor_eval.EvalConfig(
        mode="smoke",
        output_dir=tmp_path,
        env_name=gr00t_public_anchor_eval.DEFAULT_ENV_NAME,
        model_path=gr00t_public_anchor_eval.DEFAULT_MODEL_PATH,
        embodiment_tag=gr00t_public_anchor_eval.DEFAULT_EMBODIMENT_TAG,
        server_host="127.0.0.1",
        server_port=5555,
        server_python="",
        mujoco_gl="",
        n_episodes=1,
        n_envs=1,
        max_episode_steps=50,
        n_action_steps=20,
        seed_list=(20000,),
        server_ready_timeout_s=30.0,
        server_ping_timeout_ms=1000,
        server_ping_interval_s=0.01,
        spawn_server_if_missing=True,
        kill_server_on_exit=True,
        total_timeout_s=60.0,
    )

    class _FakeProc:
        pid = 4321
        returncode = None

        def poll(self) -> None:
            return None

    client = object()
    ping_responses = iter([False, True])
    port_responses = iter([True, False])
    stale_kills: list[tuple[str, int]] = []

    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_make_policy_client",
        lambda host, port, timeout_ms: client,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_safe_ping",
        lambda bound_client, timeout_ms: next(ping_responses),
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_is_tcp_port_listening",
        lambda host, port, timeout_s=0.2: next(port_responses),
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_kill_stale_gr00t_server_on_port",
        lambda host, port: stale_kills.append((host, port)) or True,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_spawn_server_subprocess",
        lambda cmd, log_path, cwd, env: _FakeProc(),
    )
    monkeypatch.setattr(gr00t_public_anchor_eval.time, "sleep", lambda seconds: None)

    _, proc, started_by_me = gr00t_public_anchor_eval._ensure_server_ready(
        config,
        REPO_ROOT,
        tmp_path / "server.log",
    )

    assert proc is not None
    assert started_by_me is True
    assert stale_kills == [("127.0.0.1", 5555)]


def test_smoke_mode_emits_public_anchor_smoke_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_ensure_runtime_layout",
        lambda repo_root, mode: _fake_runtime_layout(tmp_path, mode),
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_maybe_reexec_into_wbc_venv",
        lambda repo_root: None,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_repo_root",
        lambda: REPO_ROOT,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_run_seeded_rollout",
        _fake_execution,
    )

    output_dir = tmp_path / "anchor"
    exit_code = gr00t_public_anchor_eval.main(
        ["--mode", "smoke", "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    smoke = _read_json(output_dir / gr00t_public_anchor_eval.SMOKE_JSON_NAME)
    assert smoke["artifact_kind"] == gr00t_public_anchor_eval.SMOKE_ARTIFACT_KIND
    assert smoke["mode"] == "smoke"
    assert smoke["protocol"]["n_episodes"] == 1
    assert smoke["protocol"]["n_envs"] == 1
    assert smoke["protocol"]["n_action_steps"] == 20
    assert smoke["public_anchor_scope"]["public_anchor_comparable"] is True
    assert (
        smoke["public_anchor_scope"]["new_embodiment_public_anchor_comparable"] is False
    )
    assert smoke["success_count"] >= 1
    assert smoke["ready_for_formal"] is True


def test_formal_mode_emits_formal_and_sanity_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_ensure_runtime_layout",
        lambda repo_root, mode: _fake_runtime_layout(tmp_path, mode),
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_maybe_reexec_into_wbc_venv",
        lambda repo_root: None,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_repo_root",
        lambda: REPO_ROOT,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_run_seeded_rollout",
        _fake_execution,
    )

    output_dir = tmp_path / "anchor"
    exit_code = gr00t_public_anchor_eval.main(
        ["--mode", "formal", "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    formal = _read_json(output_dir / gr00t_public_anchor_eval.FORMAL_JSON_NAME)
    gate = _read_json(output_dir / gr00t_public_anchor_eval.SANITY_GATE_JSON_NAME)
    assert formal["artifact_kind"] == gr00t_public_anchor_eval.FORMAL_ARTIFACT_KIND
    assert formal["formal_protocol"] == {
        "env_name": gr00t_public_anchor_eval.DEFAULT_ENV_NAME,
        "model_path": gr00t_public_anchor_eval.DEFAULT_MODEL_PATH,
        "embodiment_tag": gr00t_public_anchor_eval.DEFAULT_EMBODIMENT_TAG,
        "n_episodes": 10,
        "n_envs": 5,
        "max_episode_steps": 1440,
        "n_action_steps": 20,
        "seed_list": list(gr00t_public_anchor_eval.FORMAL_DEFAULT_SEED_LIST),
        "policy_horizon_expected": 30,
    }
    assert formal["public_anchor_scope"]["public_anchor_comparable"] is True
    assert (
        formal["public_anchor_scope"]["new_embodiment_public_anchor_comparable"]
        is False
    )
    assert gate["success_rate"] == formal["success_rate"]
    assert gate["success_count"] == formal["success_count"]
    assert gate["n_episodes"] == 10
    assert gate["seed_list"] == list(gr00t_public_anchor_eval.FORMAL_DEFAULT_SEED_LIST)
    assert gate["systemic_break_flags"] == []
    assert gate["sanity_status"] == "PASS"
    assert gate["continue_to_audit"] is True
    assert gate["new_embodiment_public_anchor_comparable"] is False
    assert gate["failure_note_path"] is None


def test_systemic_break_blocks_formal_audit_and_writes_failure_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _blocked_execution(
        config: gr00t_public_anchor_eval.EvalConfig,
        runtime_dir: Path,
        server_log: Path,
        client_log: Path,
        artifacts_videos: Path,
    ) -> dict[str, Any]:
        payload = _fake_execution(
            config,
            runtime_dir=runtime_dir,
            server_log=server_log,
            client_log=client_log,
            artifacts_videos=artifacts_videos,
        )
        payload["systemic_break_flags"] = [
            "all_identical_trajectories",
            "zero_motion_episodes",
        ]
        payload["systemic_break_details"] = {
            "all_identical_trajectory_fingerprint": "same",
            "zero_motion_episode_indices": [0, 1],
        }
        return payload

    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_ensure_runtime_layout",
        lambda repo_root, mode: _fake_runtime_layout(tmp_path, mode),
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_maybe_reexec_into_wbc_venv",
        lambda repo_root: None,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_repo_root",
        lambda: REPO_ROOT,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_run_seeded_rollout",
        _blocked_execution,
    )

    output_dir = tmp_path / "anchor"
    exit_code = gr00t_public_anchor_eval.main(
        ["--mode", "formal", "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    gate = _read_json(output_dir / gr00t_public_anchor_eval.SANITY_GATE_JSON_NAME)
    failure_note = (
        output_dir / gr00t_public_anchor_eval.FAILURE_NOTE_MARKDOWN_NAME
    ).read_text(encoding="utf-8")
    assert gate["sanity_status"] == "BLOCK"
    assert gate["continue_to_audit"] is False
    assert gate["systemic_break_flags"] == [
        "all_identical_trajectories",
        "zero_motion_episodes",
    ]
    assert gate["failure_note_path"] == str(
        output_dir / gr00t_public_anchor_eval.FAILURE_NOTE_MARKDOWN_NAME
    )
    assert "all_identical_trajectories" in failure_note
    assert "NEW_EMBODIMENT" in failure_note


def test_wrong_checkpoint_scope_is_not_publicly_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_ensure_runtime_layout",
        lambda repo_root, mode: _fake_runtime_layout(tmp_path, mode),
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_maybe_reexec_into_wbc_venv",
        lambda repo_root: None,
    )
    monkeypatch.setattr(
        gr00t_public_anchor_eval,
        "_repo_root",
        lambda: REPO_ROOT,
    )

    output_dir = tmp_path / "anchor"
    exit_code = gr00t_public_anchor_eval.main(
        [
            "--mode",
            "formal",
            "--output-dir",
            str(output_dir),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
        ]
    )

    assert exit_code == 0
    formal = _read_json(output_dir / gr00t_public_anchor_eval.FORMAL_JSON_NAME)
    gate = _read_json(output_dir / gr00t_public_anchor_eval.SANITY_GATE_JSON_NAME)
    assert formal["runtime_status"] == "SKIPPED_SCOPE_MISMATCH"
    assert formal["public_anchor_scope"]["public_anchor_comparable"] is False
    assert (
        formal["public_anchor_scope"]["new_embodiment_public_anchor_comparable"]
        is False
    )
    assert formal["systemic_break_flags"] == ["wrong_checkpoint_evaluation"]
    assert gate["sanity_status"] == "BLOCK"
    assert gate["continue_to_audit"] is False
