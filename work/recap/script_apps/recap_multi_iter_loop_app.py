#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from work.recap.scripts.recap_orchestration_common import git_head_and_dirty
from work.recap.scripts.recap_orchestration_common import repo_root as _shared_repo_root
from work.recap.scripts.recap_orchestration_common import require_port_free
from work.recap.scripts.recap_orchestration_common import run_cmd_tee
from work.recap.scripts.recap_orchestration_common import (
    select_latest_checkpoint,
)
from work.recap.scripts.recap_orchestration_common import (
    wbc_python as _shared_wbc_python,
)
from work.recap.scripts.recap_orchestration_common import write_json
from work.recap.multi_iter_loop import RecapMultiIterLoopWorkflow
from work.recap.multi_iter_loop import WorkflowDependencies
from work.recap.multi_iter_loop import build_workflow_config


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_START_POLICY_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT_BASE = 5800

DEFAULT_MUJOCO_GL = "egl"
DEFAULT_N_ACTION_STEPS_CONFIG = 30

DEFAULT_N_ITERATIONS = 3
DEFAULT_SEED = 0
DEFAULT_SEED_OFFSET_PER_ITER = 10000

DEFAULT_COLLECT_EPISODES = 40
DEFAULT_COLLECT_MAX_POLICY_STEPS = 10

DEFAULT_MIXDONE = True
DEFAULT_MIXDONE_SHORT_MAX_EPISODE_STEPS = 60
DEFAULT_MIXDONE_LONG_MAX_EPISODE_STEPS = 1440
DEFAULT_MIXDONE_LONG_SEED_OFFSET = 1000

DEFAULT_CRITIC_BINS = 201
DEFAULT_CRITIC_MAX_EPOCHS = 100
DEFAULT_CRITIC_PATIENCE = 10
DEFAULT_CRITIC_LR = 1e-3
DEFAULT_CRITIC_VAL_RATIO = 0.1
DEFAULT_CRITIC_DEVICE = "cuda"

DEFAULT_FINETUNE_MAX_STEPS = 100
DEFAULT_FINETUNE_SAVE_STEPS = 50
DEFAULT_FINETUNE_SAVE_TOTAL_LIMIT = 1

# Finetune tuning mode defaults.
DEFAULT_FINETUNE_TUNE_PROJECTOR = False
DEFAULT_FINETUNE_TUNE_DIFFUSION_MODEL = True

DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_PROMPT_PREFIX = "advantage positive "

DEFAULT_TIMEOUT_COLLECT_S = 4 * 3600
DEFAULT_TIMEOUT_CRITIC_S = 3 * 3600
DEFAULT_TIMEOUT_LABEL_S = 2 * 3600
DEFAULT_TIMEOUT_EXPORT_S = 2 * 3600
DEFAULT_TIMEOUT_FINETUNE_S = 6 * 3600
DEFAULT_TIMEOUT_EVAL_S = 4 * 3600

DEFAULT_MIN_FREE_GB = 40.0
DEFAULT_ARCHIVE_ROOT = (
    "/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_archives"
)
DEFAULT_KEEP_LAST_N_ITERS_LOCAL = 1
DEFAULT_TIMEOUT_ARCHIVE_S = 12 * 3600


_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class DiskBudgetError(RuntimeError):
    pass


def _repo_root() -> Path:
    return _shared_repo_root(__file__)


def _ensure_repo_root_on_syspath(repo_root: Path) -> None:
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    mod = __import__("work.demo_utils.paths", fromlist=["maybe_reexec_into_wbc_venv"])
    fn = getattr(mod, "maybe_reexec_into_wbc_venv")
    fn(Path(repo_root))


def _wbc_python(repo_root: Path) -> Path:
    return _shared_wbc_python(repo_root)


def _run_cmd_tee(
    cmd: list[str],
    *,
    log_path: Path,
    header: str,
    timeout_s: float,
    cwd: Path,
    env: dict[str, str],
) -> int:
    return run_cmd_tee(
        list(cmd),
        log_path=Path(log_path),
        header=str(header),
        timeout_s=float(timeout_s),
        cwd=Path(cwd),
        env=dict(env),
    )


def _write_json(path: Path, obj: Any) -> None:
    write_json(path, obj)


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _git_head_and_dirty(repo_root: Path) -> tuple[str, bool]:
    return git_head_and_dirty(repo_root)


def _is_tcp_port_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    host = str(host or "").strip() or "127.0.0.1"
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
            return True
    except OSError:
        return False


def _require_port_free(host: str, port: int, *, context: str) -> None:
    require_port_free(host, int(port), context=context)


def _wait_for_port_free(
    host: str, port: int, *, timeout_s: float = 15.0, poll_interval_s: float = 0.25
) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while time.monotonic() <= deadline:
        if not _is_tcp_port_listening(str(host), int(port)):
            return True
        time.sleep(max(0.05, float(poll_interval_s)))
    return not _is_tcp_port_listening(str(host), int(port))


def _validate_tag(tag: str, *, name: str) -> str:
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"{name} must be a non-empty str, got {tag!r}")
    if not _TAG_RE.match(tag):
        raise ValueError(
            f"{name} must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ (got {tag!r})"
        )
    if "/" in tag or "\\" in tag:
        raise ValueError(f"{name} must not contain path separators: {tag!r}")
    if tag in (".", ".."):
        raise ValueError(f"{name} must not be '.' or '..': {tag!r}")
    return str(tag)


def _select_latest_checkpoint(output_dir: Path) -> Path:
    return select_latest_checkpoint(output_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="3A_recap_multi_iter_loop.py",
        description=(
            "RECAP 3A multi-iteration orchestrator: collect(mixdone)->critic(cumulative)->label(critic)->"
            "export(with_video+dual_task_text[optional])->finetune(upstream)->eval(advpos)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--run-id", type=str, required=True)
    p.add_argument(
        "--start-policy-path", type=str, default=str(DEFAULT_START_POLICY_PATH)
    )
    p.add_argument("--n-iterations", type=int, default=int(DEFAULT_N_ITERATIONS))
    p.add_argument("--dry-run", action="store_true", default=False)

    p.add_argument("--env-name", type=str, default=str(DEFAULT_ENV_NAME))
    p.add_argument("--embodiment-tag", type=str, default=str(DEFAULT_EMBODIMENT_TAG))
    p.add_argument("--mujoco-gl", type=str, default=str(DEFAULT_MUJOCO_GL))
    p.add_argument(
        "--n-action-steps-config",
        type=int,
        default=int(DEFAULT_N_ACTION_STEPS_CONFIG),
        help="Pass-through to 31 collector (may be overridden by server modality config).",
    )

    p.add_argument("--server-host", type=str, default=str(DEFAULT_SERVER_HOST))
    p.add_argument("--server-port", type=int, default=int(DEFAULT_SERVER_PORT_BASE))

    p.add_argument("--seed", type=int, default=int(DEFAULT_SEED))
    p.add_argument(
        "--seed-offset-per-iter",
        type=int,
        default=int(DEFAULT_SEED_OFFSET_PER_ITER),
        help="Collector seed for iter k: seed + k*seed_offset_per_iter.",
    )
    p.add_argument(
        "--fixed-eval-seed",
        type=int,
        default=None,
        help=(
            "If set, force ALL eval stages (base + finetuned, across all iterations) to use this seed. "
            "Collection seeds remain seed + k*seed_offset_per_iter."
        ),
    )

    p.add_argument(
        "--collect-episodes", type=int, default=int(DEFAULT_COLLECT_EPISODES)
    )
    p.add_argument(
        "--collect-max-policy-steps",
        type=int,
        default=int(DEFAULT_COLLECT_MAX_POLICY_STEPS),
    )

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is None:
        g = p.add_mutually_exclusive_group(required=False)
        g.add_argument(
            "--mixdone",
            dest="mixdone",
            action="store_true",
            help=(
                "Use mixdone single-process two-phase collection (short then long) sharing one archived video dir."
            ),
        )
        g.add_argument(
            "--no-mixdone",
            dest="mixdone",
            action="store_false",
            help="Disable mixdone collection.",
        )
        p.set_defaults(mixdone=bool(DEFAULT_MIXDONE))

        g_dtt = p.add_mutually_exclusive_group(required=False)
        g_dtt.add_argument(
            "--dual-task-text",
            dest="dual_task_text",
            action="store_true",
            help="Enable dual task text in the export stage.",
        )
        g_dtt.add_argument(
            "--no-dual-task-text",
            dest="dual_task_text",
            action="store_false",
            help="Disable dual task text in the export stage.",
        )
        p.set_defaults(dual_task_text=True)
    else:
        p.add_argument(
            "--mixdone",
            action=bool_action,
            default=bool(DEFAULT_MIXDONE),
            help=(
                "Use mixdone single-process two-phase collection (short then long) sharing one archived video dir."
            ),
        )

        p.add_argument(
            "--dual-task-text",
            action=bool_action,
            default=True,
            help="Enable dual task text in the export stage.",
        )
    p.add_argument("--mixdone-short-episodes", type=int, default=None)
    p.add_argument("--mixdone-long-episodes", type=int, default=None)
    p.add_argument(
        "--mixdone-short-max-episode-steps",
        type=int,
        default=int(DEFAULT_MIXDONE_SHORT_MAX_EPISODE_STEPS),
    )
    p.add_argument(
        "--mixdone-long-max-episode-steps",
        type=int,
        default=int(DEFAULT_MIXDONE_LONG_MAX_EPISODE_STEPS),
    )
    p.add_argument(
        "--mixdone-long-seed-offset",
        type=int,
        default=int(DEFAULT_MIXDONE_LONG_SEED_OFFSET),
    )

    p.add_argument("--critic-bins", type=int, default=int(DEFAULT_CRITIC_BINS))
    p.add_argument(
        "--critic-max-epochs", type=int, default=int(DEFAULT_CRITIC_MAX_EPOCHS)
    )
    p.add_argument("--critic-patience", type=int, default=int(DEFAULT_CRITIC_PATIENCE))
    p.add_argument("--critic-lr", type=float, default=float(DEFAULT_CRITIC_LR))
    p.add_argument(
        "--critic-val-ratio", type=float, default=float(DEFAULT_CRITIC_VAL_RATIO)
    )
    p.add_argument(
        "--critic-device",
        type=str,
        default=str(DEFAULT_CRITIC_DEVICE),
        choices=["cpu", "cuda"],
    )

    p.add_argument(
        "--finetune-max-steps", type=int, default=int(DEFAULT_FINETUNE_MAX_STEPS)
    )
    p.add_argument(
        "--finetune-save-steps",
        type=int,
        default=None,
        help=(
            "Pass-through to upstream finetune --save-steps. If unset, defaults to --finetune-max-steps "
            "(avoid step-level checkpoint explosion)."
        ),
    )
    p.add_argument(
        "--finetune-save-total-limit",
        type=int,
        default=int(DEFAULT_FINETUNE_SAVE_TOTAL_LIMIT),
        help=(
            "Pass-through to upstream finetune --save-total-limit (HF Trainer). "
            "Used together with --finetune-save-steps/max-steps to bound disk usage."
        ),
    )

    p.add_argument(
        "--min-free-gb",
        type=float,
        default=float(DEFAULT_MIN_FREE_GB),
        help=(
            "Fail-fast before each stage execution when free disk under repo_root is below this threshold."
        ),
    )
    p.add_argument(
        "--archive-root",
        type=str,
        default=str(DEFAULT_ARCHIVE_ROOT),
        help="Archive root for moving older checkpoint dirs (outside repo).",
    )
    p.add_argument(
        "--keep-last-n-iters-local",
        type=int,
        default=int(DEFAULT_KEEP_LAST_N_ITERS_LOCAL),
        help=(
            "Keep last N iteration checkpoint dirs under agent/artifacts/checkpoints locally; "
            "older ones are archived then deleted."
        ),
    )
    p.add_argument(
        "--pin-checkpoint-dir",
        type=str,
        action="append",
        default=[],
        help=(
            "Iter tag checkpoint dir to keep local (skip archive/delete). Repeatable. "
            "Example: --pin-checkpoint-dir recap_<run_id>_k0"
        ),
    )

    if bool_action is None:
        g_ft_proj = p.add_mutually_exclusive_group(required=False)
        g_ft_proj.add_argument(
            "--finetune-tune-projector",
            dest="finetune_tune_projector",
            action="store_true",
            help="Fine-tune action head projector during upstream finetune.",
        )
        g_ft_proj.add_argument(
            "--no-finetune-tune-projector",
            dest="finetune_tune_projector",
            action="store_false",
            help="Do not fine-tune action head projector during upstream finetune.",
        )
        p.set_defaults(finetune_tune_projector=bool(DEFAULT_FINETUNE_TUNE_PROJECTOR))

        g_ft_diff = p.add_mutually_exclusive_group(required=False)
        g_ft_diff.add_argument(
            "--finetune-tune-diffusion-model",
            dest="finetune_tune_diffusion_model",
            action="store_true",
            help="Fine-tune diffusion action head during upstream finetune (VRAM heavy).",
        )
        g_ft_diff.add_argument(
            "--no-finetune-tune-diffusion-model",
            dest="finetune_tune_diffusion_model",
            action="store_false",
            help="Do not fine-tune diffusion action head during upstream finetune (OOM safer).",
        )
        p.set_defaults(
            finetune_tune_diffusion_model=bool(DEFAULT_FINETUNE_TUNE_DIFFUSION_MODEL)
        )
    else:
        p.add_argument(
            "--finetune-tune-projector",
            action=bool_action,
            default=bool(DEFAULT_FINETUNE_TUNE_PROJECTOR),
            help="Fine-tune action head projector during upstream finetune.",
        )
        p.add_argument(
            "--finetune-tune-diffusion-model",
            action=bool_action,
            default=bool(DEFAULT_FINETUNE_TUNE_DIFFUSION_MODEL),
            help="Fine-tune diffusion action head during upstream finetune (VRAM heavy).",
        )

    p.add_argument("--eval-episodes", type=int, default=int(DEFAULT_EVAL_EPISODES))
    p.add_argument(
        "--eval-max-policy-steps",
        type=int,
        default=None,
        help=(
            "Override max_policy_steps for eval stages only. "
            "If unset, eval stages use --collect-max-policy-steps."
        ),
    )
    p.add_argument(
        "--eval-policy-prompt-prefix",
        type=str,
        default=str(DEFAULT_EVAL_PROMPT_PREFIX),
    )

    p.add_argument(
        "--timeout-collect-s", type=float, default=float(DEFAULT_TIMEOUT_COLLECT_S)
    )
    p.add_argument(
        "--timeout-critic-s", type=float, default=float(DEFAULT_TIMEOUT_CRITIC_S)
    )
    p.add_argument(
        "--timeout-label-s", type=float, default=float(DEFAULT_TIMEOUT_LABEL_S)
    )
    p.add_argument(
        "--timeout-export-s", type=float, default=float(DEFAULT_TIMEOUT_EXPORT_S)
    )
    p.add_argument(
        "--timeout-finetune-s", type=float, default=float(DEFAULT_TIMEOUT_FINETUNE_S)
    )
    p.add_argument(
        "--timeout-eval-s", type=float, default=float(DEFAULT_TIMEOUT_EVAL_S)
    )

    if bool_action is None:
        g2 = p.add_mutually_exclusive_group(required=False)
        g2.add_argument(
            "--require-git-clean",
            dest="require_git_clean",
            action="store_true",
            help="Require git status --porcelain=v1 to be empty before running stages.",
        )
        g2.add_argument(
            "--no-require-git-clean",
            dest="require_git_clean",
            action="store_false",
            help="Allow dirty git workspace.",
        )
        p.set_defaults(require_git_clean=True)

        g3 = p.add_mutually_exclusive_group(required=False)
        g3.add_argument(
            "--write-repro-snapshot",
            dest="write_repro_snapshot",
            action="store_true",
            help="Write git repro snapshot under agent/artifacts/p3A/<run_id>/repro.",
        )
        g3.add_argument(
            "--no-write-repro-snapshot",
            dest="write_repro_snapshot",
            action="store_false",
            help="Disable repro snapshot writing.",
        )
        p.set_defaults(write_repro_snapshot=True)
    else:
        p.add_argument(
            "--require-git-clean",
            action=bool_action,
            default=True,
            help="Require git status --porcelain=v1 to be empty before running stages.",
        )
        p.add_argument(
            "--write-repro-snapshot",
            action=bool_action,
            default=True,
            help="Write git repro snapshot under agent/artifacts/p3A/<run_id>/repro.",
        )
    return p


def main() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        try:
            _build_parser().parse_args()
        except SystemExit as e:
            return int(getattr(e, "code", 0) or 0)
        return 0

    repo_root = _repo_root()
    _ensure_repo_root_on_syspath(repo_root)
    _maybe_reexec_into_wbc_venv(repo_root)

    args = _build_parser().parse_args()
    wbc_py = _wbc_python(repo_root)
    stage_py = (
        wbc_py
        if wbc_py.is_file() and os.access(wbc_py, os.X_OK)
        else Path(sys.executable)
    )
    git_sha, git_dirty = _git_head_and_dirty(repo_root)
    config = build_workflow_config(
        args,
        repo_root=repo_root,
        git_sha=str(git_sha),
        git_dirty=bool(git_dirty),
        stage_python=stage_py.as_posix(),
        validate_tag=lambda tag, name: _validate_tag(tag, name=name),
    )
    dependencies = WorkflowDependencies(
        now_iso=_now_iso,
        write_json=_write_json,
        run_cmd_tee=_run_cmd_tee,
        require_port_free=_require_port_free,
        select_latest_checkpoint=_select_latest_checkpoint,
    )
    return RecapMultiIterLoopWorkflow(config=config, dependencies=dependencies).run()


if __name__ == "__main__":
    raise SystemExit(main())


class RecapMultiIterLoopScriptApp:
    def run(self) -> int:
        return main()
