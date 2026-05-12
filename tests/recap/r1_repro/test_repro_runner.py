from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

import pytest

from work.recap.r1_repro import repro_runner


@dataclass(frozen=True)
class FakeProtocol:
    ckpt_root: Path
    driver_script: str
    driver_sha256: str
    env_name: str = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
    prompt: str = "pick up the apple, walk left and place the apple on the plate."
    seed_base: int = 20000
    episodes: int = 30
    max_episode_steps: int = 1440
    n_action_steps: int = 20
    cuda_visible_devices: str = "1"
    extra_cli_args: tuple[tuple[str, str], ...] = ()


def _protocol(tmp_path: Path) -> FakeProtocol:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    return FakeProtocol(
        ckpt_root=ckpt,
        driver_script="work/recap/scripts/gr00t_g3_formal_eval.py",
        driver_sha256="0" * 64,
    )


def test_construct_cli_propagates_cuda_pin_literal(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    argv = repro_runner._construct_cli(protocol, tmp_path / "out")
    assert argv[-2:] == ["--required-cuda-visible-devices", "1"]
    assert argv[argv.index("--checkpoint") + 1] == str(protocol.ckpt_root)
    assert argv[argv.index("--seed-base") + 1] == "20000"


def test_build_subprocess_env_strips_privilege_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUDO_USER", "root")
    monkeypatch.setenv("SUDO_COMMAND", "python")
    monkeypatch.setenv("KEEP_ME", "1")
    env = repro_runner._build_subprocess_env(_protocol(tmp_path))
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["KEEP_ME"] == "1"
    assert "SUDO_USER" not in env
    assert "SUDO_COMMAND" not in env


def test_assert_gpu_free_detects_memory_util_and_compute_apps(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if "--query-gpu=memory.used,utilization.gpu" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="2000, 0\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(repro_runner.subprocess, "run", fake_run)
    with pytest.raises(repro_runner.GpuTenantConflict, match="memory_mib=2000"):
        repro_runner._assert_gpu_free("1")
    assert calls


def test_assert_gpu_free_accepts_idle_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=memory.used,utilization.gpu" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="0, 0\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(repro_runner.subprocess, "run", fake_run)
    repro_runner._assert_gpu_free("1")


def test_t81_argparse_introspection_reports_required_drift(tmp_path: Path) -> None:
    driver = tmp_path / "t8_1_nav_postlift.py"
    driver.write_text(
        "import argparse\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--output-dir')\n"
        "    parser.add_argument('--base-checkpoint')\n"
        "    parser.add_argument('--seed-base')\n"
        "    return parser\n",
        encoding="utf-8",
    )
    protocol = FakeProtocol(
        ckpt_root=tmp_path,
        driver_script=str(driver),
        driver_sha256="0" * 64,
    )
    with pytest.raises(repro_runner.T81DriverCliDrift, match="episode-count"):
        repro_runner._construct_cli(protocol, tmp_path / "out")


def test_repro_runner_no_episode_loop() -> None:
    source = Path(repro_runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "seeds"
        )


def test_no_t8_calls_outside_r1_1_c_invocation_surface() -> None:
    package_root = Path(repro_runner.__file__).resolve().parent
    matches: list[Path] = []
    pattern = re.compile(r"subprocess.*t8_")
    for path in package_root.rglob("*.py"):
        if pattern.search(path.read_text(encoding="utf-8")):
            matches.append(path.relative_to(package_root))
    assert matches in ([], [Path("repro_runner.py")])
