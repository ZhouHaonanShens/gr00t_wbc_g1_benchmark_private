from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import work.openpi as openpi_root  # noqa: E402
import work.openpi.runtime as runtime_pkg  # noqa: E402
from work.openpi import rollout_runtime  # noqa: E402


def _workflow_text(name: str) -> str:
    return (
        REPO_ROOT / "work" / "openpi" / "eval" / "workflows" / f"{name}.py"
    ).read_text(encoding="utf-8")


def test_runtime_surface_exposes_public_api_without_private_bridge_helpers() -> None:
    package_public_names = {
        "DEFAULT_HOST",
        "DEFAULT_PORT",
        "LIBERO_NATIVE_SMOKE_ENTRY",
        "NUM_STEPS_WAIT",
        "FailFastError",
        "PolicyServerProcess",
        "RuntimeCleanup",
        "RuntimeEpisodeClient",
        "RuntimePathsBuilder",
        "build_runtime_paths",
        "max_steps_for_task_suite",
        "pick_free_port",
        "prepare_libero_config_dir",
        "run_stock_smoke_harness",
    }
    alias_public_names = {
        "DEFAULT_HOST",
        "DEFAULT_PORT",
        "LIBERO_NATIVE_SMOKE_ENTRY",
        "NUM_STEPS_WAIT",
        "FailFastError",
        "build_runtime_paths",
        "max_steps_for_task_suite",
        "pick_free_port",
        "prepare_libero_config_dir",
        "run_stock_smoke_harness",
    }

    assert rollout_runtime.__name__ == "work.openpi.runtime.api"
    assert openpi_root.rollout_runtime is rollout_runtime

    for name in package_public_names:
        assert hasattr(runtime_pkg, name), f"{runtime_pkg.__name__} missing {name}"
    for name in alias_public_names:
        assert hasattr(rollout_runtime, name), f"{rollout_runtime.__name__} missing {name}"

    for module in (runtime_pkg, rollout_runtime):
        assert not hasattr(module, "_required_paths")
        assert not hasattr(module, "_run_harness")


def test_eval_workflows_no_longer_reference_runtime_private_helpers() -> None:
    stock_smoke_text = _workflow_text("stock_smoke")
    rollout_support_text = _workflow_text("rollout_support")
    tracked_gate_text = _workflow_text("tracked_gate")

    assert "from work.openpi.runtime import _required_paths" not in stock_smoke_text
    assert "from work.openpi.runtime import _run_harness" not in stock_smoke_text
    assert "RuntimePathsBuilder" in stock_smoke_text
    assert "run_stock_smoke_harness" in stock_smoke_text

    assert "rollout_runtime._" not in rollout_support_text
    assert "libero_native_smoke._" not in tracked_gate_text
