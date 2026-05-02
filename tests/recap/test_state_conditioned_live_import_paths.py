from __future__ import annotations

import os
from pathlib import Path
import stat
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_conditioned_bucket_a_import
from work.demo_utils import paths as demo_paths


def _create_live_checkout_tree(repo_root: Path) -> None:
    for relative in (
        Path("submodules/Isaac-GR00T"),
        Path("submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"),
        Path(
            "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobosuite"
        ),
        Path(
            "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa"
        ),
        Path("submodules/Isaac-GR00T/external_dependencies/robocasa"),
    ):
        (repo_root / relative).mkdir(parents=True, exist_ok=True)


def test_build_live_pythonpath_prefers_wbc_dexmg_robocasa_before_generic_robocasa(
    tmp_path: Path,
) -> None:
    _create_live_checkout_tree(tmp_path)

    pythonpath = state_conditioned_bucket_a_import._build_live_pythonpath(tmp_path)

    assert pythonpath == demo_paths.wbc_checkout_pythonpath(tmp_path)
    assert pythonpath.index(
        str(
            tmp_path
            / "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa"
        )
    ) < pythonpath.index(
        str(tmp_path / "submodules/Isaac-GR00T/external_dependencies/robocasa")
    )


def test_maybe_reexec_into_wbc_venv_preserves_checkout_paths_in_pythonpath(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _create_live_checkout_tree(tmp_path)
    target = demo_paths.wbc_venv_python(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR)

    captured: dict[str, object] = {}

    def _fake_execv(path: str, argv: list[str]) -> None:
        captured["path"] = path
        captured["argv"] = list(argv)
        captured["pythonpath"] = os.environ.get("PYTHONPATH")

    monkeypatch.setattr(demo_paths.os, "execv", _fake_execv)
    monkeypatch.setattr(
        demo_paths.sys, "argv", ["work/recap/scripts/31_recap_collect_rollouts.py"]
    )
    monkeypatch.setattr(demo_paths.sys, "prefix", str(tmp_path / "host_prefix"))
    monkeypatch.setattr(demo_paths.sys, "executable", str(tmp_path / "host_python"))
    monkeypatch.setenv("PYTHONPATH", "/existing/site_a:/existing/site_b")

    demo_paths.maybe_reexec_into_wbc_venv(tmp_path)

    expected_prefix = demo_paths.wbc_checkout_pythonpath(tmp_path)
    assert captured["path"] == str(target)
    assert captured["argv"] == [
        str(target),
        "work/recap/scripts/31_recap_collect_rollouts.py",
    ]
    assert str(captured["pythonpath"]).split(os.pathsep) == [
        *expected_prefix,
        "/existing/site_a",
        "/existing/site_b",
    ]
