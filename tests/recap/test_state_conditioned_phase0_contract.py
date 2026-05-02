from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_conditioned_phase0_smoke


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_live_rollout_batch_and_unbatch_helpers_preserve_expected_shapes() -> None:
    np = pytest.importorskip("numpy")

    env_obs = {
        "q": np.zeros((1, 43), dtype=np.float64),
        "state.left_arm": np.zeros((1, 7), dtype=np.float64),
        "state.left_leg": np.zeros((1, 6), dtype=np.float64),
        "video.ego_view": np.zeros((1, 128, 128, 3), dtype=np.uint8),
        "annotation.human.task_description": "pick up the apple",
    }
    policy_obs = state_conditioned_phase0_smoke._batch_observation_for_policy(env_obs)
    assert getattr(policy_obs["q"], "shape", None) == (1, 1, 43)
    assert getattr(policy_obs["q"], "dtype", None) == np.float32
    assert getattr(policy_obs["state.left_arm"], "dtype", None) == np.float32
    assert getattr(policy_obs["state.left_leg"], "dtype", None) == np.float32
    assert getattr(policy_obs["video.ego_view"], "dtype", None) == np.uint8
    assert policy_obs["annotation.human.task_description"] == ["pick up the apple"]

    policy_action = {
        "action.left_arm": np.zeros((1, 30, 7), dtype=np.float32),
        "action.right_arm": np.zeros((1, 30, 7), dtype=np.float32),
    }
    env_action = state_conditioned_phase0_smoke._unbatch_policy_action(policy_action)
    assert getattr(env_action["action.left_arm"], "shape", None) == (30, 7)
    assert getattr(env_action["action.right_arm"], "shape", None) == (30, 7)


def test_make_live_env_inserts_wbc_wrapper_before_multistep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_log: dict[str, Any] = {}

    class FakeEnv:
        def __init__(self, **kwargs: Any) -> None:
            build_log["base_env_kwargs"] = dict(kwargs)

    class FakeBaseConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)

        def to_dict(self) -> dict[str, Any]:
            return dict(self.kwargs)

    class FakeWBCWrapper:
        def __init__(self, env: object, script_config: dict[str, Any]) -> None:
            self.env = env
            self.script_config = dict(script_config)

    class FakeMultiStepWrapper:
        def __init__(self, env: object, **kwargs: Any) -> None:
            self.env = env
            self.kwargs = dict(kwargs)

    real_import_module = state_conditioned_phase0_smoke.importlib.import_module

    def _fake_import_module(name: str) -> Any:
        if name == "gr00t_wbc.control.envs.robocasa.sync_env":
            return types.SimpleNamespace(GroundOnly_G1_gear_wbc=FakeEnv)
        if name == "gr00t_wbc.control.main.teleop.configs.configs":
            return types.SimpleNamespace(BaseConfig=FakeBaseConfig)
        if name == "gr00t_wbc.control.utils.n1_utils":
            return types.SimpleNamespace(WholeBodyControlWrapper=FakeWBCWrapper)
        if name == "gr00t.eval.sim.wrapper.multistep_wrapper":
            return types.SimpleNamespace(MultiStepWrapper=FakeMultiStepWrapper)
        return real_import_module(name)

    monkeypatch.setattr(
        state_conditioned_phase0_smoke.importlib,
        "import_module",
        _fake_import_module,
    )

    result = state_conditioned_phase0_smoke._make_live_env(
        resolved_env_name="gr00tlocomanip_g1_sim/GroundOnly_G1_gear_wbc",
        n_action_steps=30,
        max_episode_steps=60,
        video_delta_indices=[0],
        state_delta_indices=[0],
    )

    assert isinstance(result, FakeMultiStepWrapper)
    assert isinstance(result.env, FakeWBCWrapper)
    assert isinstance(result.env.env, FakeEnv)
    assert result.env.script_config["wbc_version"] == "gear_wbc"
    assert result.env.script_config["enable_waist"] is True
    assert result.kwargs["n_action_steps"] == 30


def _success_probe(*, repo_root: Path, add_live_paths: bool) -> dict[str, Any]:
    del repo_root, add_live_paths
    return {
        "ok": True,
        "blockers": [],
        "errors": {},
        "import_gr00t_ok": True,
        "runtime_probe": {
            "flash_attn_2_available": True,
            "gr00t_import_ok": True,
            "gr00t_wbc_import_ok": True,
            "gymnasium_import_ok": True,
            "numpy_import_ok": True,
            "policy_client_import_ok": True,
            "robocasa_import_ok": True,
            "rollout_policy_import_ok": True,
            "server_entrypoint_exists": True,
            "torch_bfloat16_cuda_ok": True,
            "torch_cuda_available": True,
            "torch_import_ok": True,
            "video_backend_probe_ok": True,
        },
        "sys_path_injected": ["/fake/live/root"],
    }


def _blocking_probe(*, repo_root: Path, add_live_paths: bool) -> dict[str, Any]:
    del repo_root, add_live_paths
    result = _success_probe(repo_root=REPO_ROOT, add_live_paths=False)
    result["ok"] = False
    result["import_gr00t_ok"] = False
    result["blockers"] = ["gr00t_import_ok"]
    runtime_probe = dict(result["runtime_probe"])
    runtime_probe["gr00t_import_ok"] = False
    result["runtime_probe"] = runtime_probe
    result["errors"] = {
        "gr00t_import_ok": "ModuleNotFoundError: No module named 'gr00t'"
    }
    return result


def _live_success(
    args: Any,
    *,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    del args, repo_root
    return {
        "policy_ping_ok": True,
        "get_action_ok": True,
        "rollout_episode_ok": True,
        "rollout_done_observed": True,
        "rollout_outer_steps": 1,
        "rollout_stop_reason": "env_done",
        "modality_config_keys": ["action", "language", "state", "video"],
        "action_horizon": 30,
        "server": {
            "host": "127.0.0.1",
            "port": 5555,
            "spawned": False,
            "reused_existing": True,
            "server_log": str(output_dir / "00_server.log"),
        },
    }


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_phase0_smoke.main(["--help"])
    assert exc_info.value.code == 0


def test_import_only_success_writes_machine_readable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        state_conditioned_phase0_smoke,
        "_collect_runtime_probe",
        _success_probe,
    )

    exit_code = state_conditioned_phase0_smoke.main(
        ["--mode", "import-only", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        tmp_path / state_conditioned_phase0_smoke.IMPORT_PROBE_JSON_NAME
    )

    assert exit_code == 0
    assert captured.err == ""
    assert payload["status"] == "PASS"
    assert payload["run_mode"] == "import-only"
    assert payload["import_gr00t_ok"] is True
    assert payload["failure"] is None
    assert payload["artifact_path"] == str(
        tmp_path / state_conditioned_phase0_smoke.IMPORT_PROBE_JSON_NAME
    )
    assert artifact == payload


def test_live_success_writes_server_smoke_ok_json_with_required_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        state_conditioned_phase0_smoke,
        "_collect_runtime_probe",
        _success_probe,
    )
    monkeypatch.setattr(
        state_conditioned_phase0_smoke,
        "_run_live_checks",
        _live_success,
    )

    exit_code = state_conditioned_phase0_smoke.main(
        ["--mode", "live", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact_path = tmp_path / state_conditioned_phase0_smoke.LIVE_SMOKE_OK_JSON_NAME
    artifact = _read_json(artifact_path)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["status"] == "PASS"
    assert payload["run_mode"] == "live"
    assert payload["import_gr00t_ok"] is True
    assert payload["policy_ping_ok"] is True
    assert payload["get_action_ok"] is True
    assert payload["rollout_episode_ok"] is True
    assert payload["live_checks_attempted"] is True
    assert payload["artifact_path"] == str(artifact_path)
    assert artifact == payload


def test_import_blocker_returns_machine_readable_fail_json_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        state_conditioned_phase0_smoke,
        "_collect_runtime_probe",
        _blocking_probe,
    )

    exit_code = state_conditioned_phase0_smoke.main(
        ["--mode", "import-only", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        tmp_path / state_conditioned_phase0_smoke.IMPORT_PROBE_JSON_NAME
    )

    assert exit_code == 1
    assert captured.err == ""
    assert "Traceback" not in captured.out
    assert payload["status"] == "FAIL"
    assert payload["failure"]["stage"] == "import_probe"
    assert payload["import_gr00t_ok"] is False
    assert payload["failure"]["blockers"] == ["gr00t_import_ok"]
    assert artifact == payload


def test_live_mode_fail_stops_before_live_checks_after_import_probe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    live_called = {"value": False}

    def _unexpected_live(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        live_called["value"] = True
        return {}

    monkeypatch.setattr(
        state_conditioned_phase0_smoke,
        "_collect_runtime_probe",
        _blocking_probe,
    )
    monkeypatch.setattr(
        state_conditioned_phase0_smoke,
        "_run_live_checks",
        _unexpected_live,
    )

    exit_code = state_conditioned_phase0_smoke.main(
        ["--mode", "live", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert live_called["value"] is False
    assert payload["status"] == "FAIL"
    assert payload["failure"]["stage"] == "import_probe"
    assert payload["live_checks_attempted"] is False


def test_live_env_lookup_failure_preserves_machine_readable_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _env_lookup_failure(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise state_conditioned_phase0_smoke.Phase0SmokeError(
            "env_lookup",
            "env resolution failed",
            detail={
                "code": "state_conditioned_env_unavailable",
                "logical_task": "apple_to_plate_g1",
                "requested_env_name": (
                    "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
                ),
                "available_close_matches": [
                    "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_g1_fixed_base_gear_wbc"
                ],
            },
        )

    monkeypatch.setattr(
        state_conditioned_phase0_smoke,
        "_collect_runtime_probe",
        _success_probe,
    )
    monkeypatch.setattr(
        state_conditioned_phase0_smoke,
        "_run_live_checks",
        _env_lookup_failure,
    )

    exit_code = state_conditioned_phase0_smoke.main(
        ["--mode", "live", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["failure"]["stage"] == "env_lookup"
    assert payload["failure"]["detail"]["code"] == "state_conditioned_env_unavailable"
    assert payload["failure"]["detail"]["logical_task"] == "apple_to_plate_g1"
