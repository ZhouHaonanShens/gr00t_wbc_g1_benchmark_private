from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LIVE_40_PATH = REPO_ROOT / "agent" / "run" / "40_recap_train_critic_dist_bins.py"
ARCHIVED_40_PATH = (
    REPO_ROOT
    / "agent"
    / "archive"
    / "recap_legacy_state_only_critic"
    / "40_recap_train_critic_dist_bins.py"
)


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_agent_run_40_wrapper_is_retired_but_archive_surface_still_exists() -> (
    None
):
    assert not LIVE_40_PATH.exists()
    assert ARCHIVED_40_PATH.is_file()


def test_archived_40_surface_keeps_help_and_bins_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(ARCHIVED_40_PATH, "critic_40_archived_surface_test")
    monkeypatch.setattr(module, "_repo_root", lambda: REPO_ROOT)
    monkeypatch.setattr(module, "_maybe_reexec_into_wbc_venv", lambda _repo_root: None)

    monkeypatch.setattr(
        module.sys,
        "argv",
        ["40_recap_train_critic_dist_bins.py", "--help"],
    )
    assert module.main() == 0

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "40_recap_train_critic_dist_bins.py",
            "--iter-tag",
            "task7_wrapper_bad",
            "--critic-tag",
            "critic_task7_wrapper_bad",
            "--bins",
            "1",
            "--seed",
            "0",
            "--max-epochs",
            "1",
            "--lr",
            "0.001",
            "--val-ratio",
            "0.1",
        ],
    )
    with pytest.raises(ValueError, match=r"--bins must be >=2, got 1"):
        module.main()


def test_multi_iter_workflow_currently_routes_critic_stage_to_archived_40_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = (
        REPO_ROOT / "work" / "recap" / "scripts" / "3A_recap_multi_iter_loop.py"
    )
    module = _load_module(module_path, "recap_3a_critic_surface_test")

    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_maybe_reexec_into_wbc_venv", lambda _repo_root: None)
    monkeypatch.setattr(
        module, "_git_head_and_dirty", lambda _repo_root: ("test-sha", False)
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "3A_recap_multi_iter_loop.py",
            "--run-id",
            "critic_surface_check",
            "--n-iterations",
            "1",
            "--dry-run",
            "--no-require-git-clean",
            "--no-write-repro-snapshot",
        ],
    )

    exit_code = module.main()
    manifest_path = (
        tmp_path
        / "agent"
        / "artifacts"
        / "p3A"
        / "critic_surface_check"
        / "manifest.json"
    )
    manifest = module.json.loads(manifest_path.read_text(encoding="utf-8"))
    critic_stage = next(
        stage
        for stage in manifest["iterations"][0]["stages"]
        if stage["name"] == "20_critic_cumulative"
    )

    assert exit_code == 0
    assert critic_stage["cmd"][1].endswith(
        "agent/archive/recap_legacy_state_only_critic/40_recap_train_critic_dist_bins.py"
    )
    assert not any(
        "agent/run/40_recap_train_critic_dist_bins.py" in part
        for part in critic_stage["cmd"]
    )
