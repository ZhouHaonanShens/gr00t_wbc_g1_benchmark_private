from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


LEGACY_PROJECT_ROOT = Path("/media/howard/Data/Projects/gr00t_wbc_g1_benchmark")
CANONICAL_PROJECT_ROOT = Path("/home/howard/Projects/gr00t_wbc_g1_benchmark")


DEFAULT_STAGE3_ITERATION_MANIFEST_REL = Path(
    "agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json"
)
DEFAULT_STAGE3_PREREQ_SMOKE_ARTIFACT_ROOT_REL = Path(
    "agent/artifacts/stage3_prereq_smoke"
)


def abspath_preserve_symlink(path: str | os.PathLike[str]) -> Path:
    return Path(
        os.path.abspath(str(remap_legacy_project_root(Path(path).expanduser())))
    )


def remap_legacy_project_root(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return candidate
    try:
        relative = candidate.relative_to(LEGACY_PROJECT_ROOT)
    except ValueError:
        return candidate
    if CANONICAL_PROJECT_ROOT.exists():
        return CANONICAL_PROJECT_ROOT / relative
    return candidate


def current_python_abspath_preserve_symlink() -> Path:
    return abspath_preserve_symlink(sys.executable)


def same_abspath_preserve_symlink(
    left: str | os.PathLike[str], right: str | os.PathLike[str]
) -> bool:
    return abspath_preserve_symlink(left) == abspath_preserve_symlink(right)


def stage3_iteration_manifest_path(repo_root: Path) -> Path:
    return abspath_preserve_symlink(repo_root / DEFAULT_STAGE3_ITERATION_MANIFEST_REL)


def stage3_prereq_smoke_artifact_root(repo_root: Path) -> Path:
    return abspath_preserve_symlink(
        repo_root / DEFAULT_STAGE3_PREREQ_SMOKE_ARTIFACT_ROOT_REL
    )


def _require_manifest_string(
    payload: dict[str, Any], *, field_name: str, manifest_path: Path
) -> str:
    raw = str(payload.get(field_name) or "").strip()
    if not raw:
        raise ValueError(
            f"iteration manifest {manifest_path} missing non-empty {field_name!r}"
        )
    return str(abspath_preserve_symlink(raw))


def load_stage3_training_python_contract(repo_root: Path) -> dict[str, str]:
    manifest_path = stage3_iteration_manifest_path(repo_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"stage3 iteration manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            f"stage3 iteration manifest must be a JSON object, got {type(payload).__name__}"
        )
    return {
        "manifest_path": str(manifest_path),
        "orchestrator_python": _require_manifest_string(
            payload,
            field_name="orchestrator_python",
            manifest_path=manifest_path,
        ),
        "delegate_runtime_python": _require_manifest_string(
            payload,
            field_name="delegate_runtime_python",
            manifest_path=manifest_path,
        ),
    }


def repo_root(from_path: str | os.PathLike[str] | None = None) -> Path:
    start: Path
    if from_path is None:
        start = Path.cwd().resolve()
    else:
        p = Path(from_path).resolve()
        start = p.parent if p.is_file() else p

    for cur in (start, *start.parents):
        if (cur / "AGENTS.md").is_file() and (cur / "agent").is_dir():
            return cur

    return start


def ensure_dirs(
    *,
    repo_root: Path,
    runtime_logs_rel: str,
    artifacts_videos_rel: str,
) -> tuple[Path, Path]:
    runtime_dir = repo_root / runtime_logs_rel
    artifacts_videos = repo_root / artifacts_videos_rel
    runtime_dir.mkdir(parents=True, exist_ok=True)
    artifacts_videos.mkdir(parents=True, exist_ok=True)
    return runtime_dir, artifacts_videos


def ensure_demo_live_dirs(
    repo_root: Path, video_archive_dir: str
) -> tuple[Path, Path, Path, Path]:
    runtime_dir = repo_root / "agent/runtime_logs/demo_live"
    artifacts_videos = repo_root / video_archive_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    artifacts_videos.mkdir(parents=True, exist_ok=True)
    server_log = runtime_dir / "00_server.log"
    client_log = runtime_dir / "01_client.log"
    return runtime_dir, artifacts_videos, server_log, client_log


def wbc_venv_python(repo_root: Path) -> Path:
    return repo_root / ".envs" / "wbc" / "bin" / "python"


def wbc_checkout_pythonpath(repo_root: Path) -> list[str]:
    gr00t_src = repo_root / "submodules" / "Isaac-GR00T"
    gr00t_wbc_src = gr00t_src / "external_dependencies" / "GR00T-WholeBodyControl"
    gr00t_wbc_robocasa_src = gr00t_wbc_src / "gr00t_wbc" / "dexmg" / "gr00trobocasa"
    gr00t_wbc_robosuite_src = gr00t_wbc_src / "gr00t_wbc" / "dexmg" / "gr00trobosuite"
    robocasa_src = gr00t_src / "external_dependencies" / "robocasa"
    active_paths: list[str] = []
    for source_path in (
        repo_root,
        gr00t_src,
        gr00t_wbc_src,
        gr00t_wbc_robosuite_src,
        gr00t_wbc_robocasa_src,
        robocasa_src,
    ):
        if source_path.exists():
            active_paths.append(str(source_path))
    return active_paths


def maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    target = wbc_venv_python(repo_root)
    try:
        skip_flag = str(os.environ.get("GR00T_SKIP_WBC_REEXEC", "")).strip().lower()
        if skip_flag in {"1", "true", "yes", "on"}:
            return
        if not target.is_file():
            return
        if not os.access(target, os.X_OK):
            return

        target_venv = target.parent.parent
        try:
            if Path(sys.prefix).resolve() == target_venv.resolve():
                return
        except Exception:
            pass

        try:
            exe_path = Path(sys.executable)
            bin_dir = target.parent
            if hasattr(exe_path, "is_relative_to"):
                if exe_path.is_relative_to(bin_dir):
                    return
            else:
                exe_str = os.path.abspath(str(exe_path))
                bin_str = os.path.abspath(str(bin_dir))
                if exe_str.startswith(bin_str.rstrip(os.sep) + os.sep):
                    return
        except Exception:
            pass

        argv = [str(target), *sys.argv]
        pythonpath_entries: list[str] = []
        for entry in [
            *wbc_checkout_pythonpath(repo_root),
            *str(os.environ.get("PYTHONPATH", "")).split(os.pathsep),
        ]:
            normalized = str(entry).strip()
            if normalized and normalized not in pythonpath_entries:
                pythonpath_entries.append(normalized)
        if pythonpath_entries:
            os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        else:
            os.environ.pop("PYTHONPATH", None)
        os.execv(str(target), argv)
    except Exception:
        return
