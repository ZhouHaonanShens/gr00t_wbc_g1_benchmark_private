from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_run_module(filename: str, module_name: str):
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("shim_module_name", "core_module_name", "script_app_name"),
    [
        (
            "work.recap.script_apps.recap_collect_rollouts_app",
            "work.recap.collect_rollouts",
            "RecapCollectRolloutsScriptApp",
        ),
        (
            "work.recap.script_apps.recap_label_dataset_app",
            "work.recap.label_dataset",
            "RecapLabelDatasetScriptApp",
        ),
        (
            "work.recap.script_apps.recap_export_lerobot_with_video_app",
            "work.recap.lerobot_export.workflow",
            "RecapExportLeRobotWithVideoScriptApp",
        ),
        (
            "work.recap.script_apps.recap_finetune_full_app",
            "work.recap.finetune_full",
            "RecapFinetuneFullScriptApp",
        ),
    ],
)
def test_recap_script_app_shims_alias_core_modules(
    shim_module_name: str,
    core_module_name: str,
    script_app_name: str,
) -> None:
    shim_module = importlib.import_module(shim_module_name)
    core_module = importlib.import_module(core_module_name)

    assert shim_module.main is core_module.main
    assert getattr(shim_module, script_app_name) is getattr(
        core_module, script_app_name
    )


def test_39_script_app_run_keeps_wrapper_patch_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_module(
        "39_recap_export_lerobot_v2_with_video.py", "recap_39_script_app_sync"
    )
    observed: dict[str, bool] = {}

    def fake_repo_root() -> Path:
        return REPO_ROOT

    def fake_reexec(_repo_root: Path) -> None:
        return None

    def fake_tee_stdio(*_args, **_kwargs):
        return None

    def fake_install_alarm_timeout(_timeout_s: float | None) -> None:
        return None

    def fake_clear_alarm_timeout() -> None:
        return None

    def fake_main() -> int:
        observed["repo_root_synced"] = (
            module._app_module._repo_root is module._repo_root
        )
        observed["reexec_synced"] = (
            module._app_module._maybe_reexec_into_wbc_venv
            is module._maybe_reexec_into_wbc_venv
        )
        observed["tee_synced"] = module._app_module._tee_stdio is module._tee_stdio
        observed["alarm_set_synced"] = (
            module._app_module._install_alarm_timeout is module._install_alarm_timeout
        )
        observed["alarm_clear_synced"] = (
            module._app_module._clear_alarm_timeout is module._clear_alarm_timeout
        )
        return 39

    monkeypatch.setattr(module, "_repo_root", fake_repo_root)
    monkeypatch.setattr(module, "_maybe_reexec_into_wbc_venv", fake_reexec)
    monkeypatch.setattr(module, "_tee_stdio", fake_tee_stdio)
    monkeypatch.setattr(module, "_install_alarm_timeout", fake_install_alarm_timeout)
    monkeypatch.setattr(module, "_clear_alarm_timeout", fake_clear_alarm_timeout)
    monkeypatch.setattr(module._app_module, "main", fake_main)

    exit_code = module.RecapExportLeRobotWithVideoScriptApp().run()

    assert exit_code == 39
    assert observed == {
        "repo_root_synced": True,
        "reexec_synced": True,
        "tee_synced": True,
        "alarm_set_synced": True,
        "alarm_clear_synced": True,
    }


@pytest.mark.parametrize(
    ("filename", "module_name", "script_app_name", "expected_exit_code"),
    [
        (
            "31_recap_collect_rollouts.py",
            "recap_31_wrapper_exec",
            "RecapCollectRolloutsScriptApp",
            31,
        ),
        (
            "32_recap_label_dataset.py",
            "recap_32_wrapper_exec",
            "RecapLabelDatasetScriptApp",
            32,
        ),
    ],
)
def test_31_32_wrappers_execute_through_script_app(
    filename: str,
    module_name: str,
    script_app_name: str,
    expected_exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_module(filename, module_name)
    observed: dict[str, bool] = {}

    class _FakeScriptApp:
        def run(self) -> int:
            observed["run_called"] = True
            return expected_exit_code

    monkeypatch.setattr(module, script_app_name, _FakeScriptApp)

    script_app = module._script_app()
    exit_code = script_app.run()

    assert isinstance(script_app, _FakeScriptApp)
    assert exit_code == expected_exit_code
    assert observed == {"run_called": True}


@pytest.mark.parametrize(
    ("filename", "module_name"),
    [
        ("31_recap_collect_rollouts.py", "recap_31_wrapper_getattr"),
        ("32_recap_label_dataset.py", "recap_32_wrapper_getattr"),
    ],
)
def test_31_32_wrappers_forward_missing_attrs_via_getattr(
    filename: str,
    module_name: str,
) -> None:
    module = _load_run_module(filename, module_name)

    assert "main" not in module.__dict__
    assert getattr(module, "main") is module._app_module.main
