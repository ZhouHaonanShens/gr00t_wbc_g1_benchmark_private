from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_export_module():
    module_path = (
        REPO_ROOT
        / "work"
        / "recap"
        / "scripts"
        / "39_recap_export_lerobot_v2_with_video.py"
    )
    spec = importlib.util.spec_from_file_location(
        "recap_export_wrapper_39", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_wrapper_parser_keeps_dual_task_text_aliases() -> None:
    module = _load_export_module()
    parser = module._build_parser()

    assert parser.prog == "39_recap_export_lerobot_v2_with_video.py"
    assert parser.parse_args([]).dual_task_text is True
    assert parser.parse_args(["--dual-task-text"]).dual_task_text is True
    assert parser.parse_args(["--no-dual-task-text"]).dual_task_text is False


def test_export_wrapper_help_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_export_module()
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["39_recap_export_lerobot_v2_with_video.py", "--help"],
    )

    assert module.main() == 0


def test_export_wrapper_delegates_selected_flags_to_canonical_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    captured: dict[str, object] = {}
    real_import_module = module.importlib.import_module

    def _fake_export(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_dataset_dir=tmp_path
            / "agent"
            / "artifacts"
            / "lerobot_datasets"
            / "wrapper_iter",
            total_videos=2,
            video_path_template="videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
            video_map_path=tmp_path
            / "agent"
            / "artifacts"
            / "lerobot_datasets"
            / "wrapper_iter"
            / "meta"
            / "videos.json",
        )

    def _fake_import_module(name: str):
        if name == "work.recap.lerobot_export.video_export":
            return SimpleNamespace(export_recap_to_lerobot_v2_with_video=_fake_export)
        return real_import_module(name)

    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_maybe_reexec_into_wbc_venv", lambda _repo_root: None)
    monkeypatch.setattr(
        module,
        "_tee_stdio",
        lambda _log_path, *, header: contextlib.nullcontext(),
    )
    monkeypatch.setattr(module, "_install_alarm_timeout", lambda _timeout_s: None)
    monkeypatch.setattr(module, "_clear_alarm_timeout", lambda: None)
    monkeypatch.setattr(module.importlib, "import_module", _fake_import_module)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "39_recap_export_lerobot_v2_with_video.py",
            "--iter-tag",
            "wrapper_iter",
            "--max-episodes",
            "2",
            "--no-dual-task-text",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0
    assert captured == {
        "iter_tag": "wrapper_iter",
        "repo_root": tmp_path,
        "input_recap_dataset_dir": "agent/artifacts/recap_datasets/wrapper_iter",
        "output_dataset_dir": "agent/artifacts/lerobot_datasets/wrapper_iter",
        "max_episodes": 2,
        "require_ffmpeg": False,
        "dual_task_text": False,
    }


def test_export_wrapper_rejects_non_positive_episode_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    monkeypatch.setattr(module, "_repo_root", lambda: REPO_ROOT)
    monkeypatch.setattr(module, "_maybe_reexec_into_wbc_venv", lambda _repo_root: None)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "39_recap_export_lerobot_v2_with_video.py",
            "--iter-tag",
            "wrapper_iter",
            "--max-episodes",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="max_episodes must be > 0"):
        module.main()
