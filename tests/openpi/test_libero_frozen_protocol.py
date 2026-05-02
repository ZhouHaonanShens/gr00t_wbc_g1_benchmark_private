from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "agent/exchange/openpi_libero_frozen_protocol.md"
SERVE_POLICY = REPO_ROOT / "submodules/openpi/scripts/serve_policy.py"
TRAINING_CONFIG = REPO_ROOT / "submodules/openpi/src/openpi/training/config.py"
LIBERO_MAIN = REPO_ROOT / "submodules/openpi/examples/libero/main.py"
LIBERO_THIRD_PARTY = REPO_ROOT / "submodules/openpi/third_party/libero"


def test_frozen_protocol_doc_freezes_stock_baseline_constants_and_budgets() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO stock frozen protocol",
        "`config` | `pi05_libero`",
        "`checkpoint` | `gs://openpi-assets/checkpoints/pi05_libero`",
        "`action_horizon` | `10`",
        "`discrete_state_input` | `False`",
        "`extra_delta_transform` | `False`",
        "`replan_steps` | `5`",
        "`num_steps_wait` | `10`",
        "`task_suite_name=libero_spatial`",
        "`task_id=0`",
        "`num_trials_per_task=1`",
        "`seed=7`",
        "`task_ids=[0,1]`",
        "`seeds=[7,17]`",
        "`num_trials_per_task=2`",
    ]
    for item in required:
        assert item in text, f"missing frozen protocol item: {item}"


def test_frozen_protocol_doc_defines_preflight_commands_paths_schema_and_stop_condition() -> (
    None
):
    text = DOC.read_text(encoding="utf-8")
    required = [
        "python3 submodules/openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero",
        "python3 submodules/openpi/examples/libero/main.py --task-suite-name libero_spatial --num-trials-per-task 1 --seed 7 --replan-steps 5 --num-steps-wait 10",
        "`submodules/openpi/scripts/serve_policy.py`",
        "`submodules/openpi/src/openpi/training/config.py`",
        "`submodules/openpi/examples/libero/main.py`",
        "`submodules/openpi/third_party/libero/`",
        '"schema_version": "openpi_libero_stock_checkpoint_v1"',
        '"config": "pi05_libero"',
        '"checkpoint": "gs://openpi-assets/checkpoints/pi05_libero"',
        '"checkpoint_source": "upstream_openpi_default_or_explicit_cli"',
        '"env_mode": "LIBERO"',
        '"simulator": "MuJoCo"',
        '"action_horizon": 10',
        '"discrete_state_input": false',
        '"extra_delta_transform": false',
        '"replan_steps": 5',
        '"num_steps_wait": 10',
        "网络不可用",
        "fail-fast, do not continue to RECAP/state-token",
    ]
    for item in required:
        assert item in text, f"missing preflight gate item: {item}"


def test_frozen_protocol_doc_explicitly_excludes_experimental_or_g1_drift() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "stock baseline 不得把 `discrete_state_input=True` 写成默认值。",
        "stock baseline 不得混入任何 G1 env id、G1 seeds、`policy_horizon=30` 或其它旧 WBC frozen 值。",
        "不得“先继续做 RECAP 或 state-token 再回头补 baseline 证据”",
    ]
    for item in required:
        assert item in text, f"missing drift guard item: {item}"


def test_frozen_protocol_preflight_paths_exist() -> None:
    for path in [SERVE_POLICY, TRAINING_CONFIG, LIBERO_MAIN, LIBERO_THIRD_PARTY]:
        assert path.exists(), f"missing required preflight path: {path}"


def test_upstream_sources_still_match_frozen_stock_values() -> None:
    serve_text = SERVE_POLICY.read_text(encoding="utf-8")
    config_text = TRAINING_CONFIG.read_text(encoding="utf-8")
    client_text = LIBERO_MAIN.read_text(encoding="utf-8")

    assert "EnvMode.LIBERO: Checkpoint(" in serve_text
    assert 'config="pi05_libero"' in serve_text
    assert 'dir="gs://openpi-assets/checkpoints/pi05_libero"' in serve_text

    assert 'name="pi05_libero"' in config_text
    assert (
        "Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)"
        in config_text
    )
    assert "extra_delta_transform=False" in config_text

    assert "replan_steps: int = 5" in client_text
    assert "num_steps_wait: int = 10" in client_text
