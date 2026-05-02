#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import subprocess
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


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_START_POLICY_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5800

DEFAULT_COLLECT_EPISODES = 10
DEFAULT_COLLECT_MAX_POLICY_STEPS = 10
DEFAULT_COLLECT_MAX_EPISODE_STEPS = 1440
DEFAULT_N_ACTION_STEPS_CONFIG = 30
DEFAULT_MUJOCO_GL = "egl"

DEFAULT_FINETUNE_MAX_STEPS = 20
DEFAULT_FINETUNE_SAVE_STEPS = 10

DEFAULT_EVAL_EPISODES = 10

DEFAULT_TIMEOUT_COLLECT_S = 3600
DEFAULT_TIMEOUT_LABEL_S = 1800
DEFAULT_TIMEOUT_EXPORT_S = 1800
DEFAULT_TIMEOUT_FINETUNE_S = 7200
DEFAULT_TIMEOUT_EVAL_S = 3600


def _repo_root() -> Path:
    return _REPO_ROOT_FOR_IMPORT


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


def _git_head_and_dirty(repo_root: Path) -> tuple[str, bool]:
    return git_head_and_dirty(repo_root, porcelain_mode="--porcelain")


def _is_tcp_port_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    host = str(host or "").strip() or "127.0.0.1"
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
            return True
    except OSError:
        return False


def _require_port_free(host: str, port: int, *, context: str) -> None:
    require_port_free(host, int(port), context=context)


def _select_latest_checkpoint(output_dir: Path) -> Path:
    return select_latest_checkpoint(output_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="38_recap_online_loop_iterate.py",
        description="RECAP online loop orchestrator: k0 collect->label->export(with video)->finetune->2x2 eval->k1 collect.",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument(
        "--start-policy-path", type=str, default=str(DEFAULT_START_POLICY_PATH)
    )
    p.add_argument("--dry-run", action="store_true", default=False)

    p.add_argument("--env-name", type=str, default=str(DEFAULT_ENV_NAME))
    p.add_argument("--embodiment-tag", type=str, default=str(DEFAULT_EMBODIMENT_TAG))
    p.add_argument("--server-host", type=str, default=str(DEFAULT_SERVER_HOST))
    p.add_argument("--server-port", type=int, default=int(DEFAULT_SERVER_PORT))
    p.add_argument("--mujoco-gl", type=str, default=str(DEFAULT_MUJOCO_GL))

    p.add_argument(
        "--collect-episodes", type=int, default=int(DEFAULT_COLLECT_EPISODES)
    )
    p.add_argument(
        "--collect-max-policy-steps",
        type=int,
        default=int(DEFAULT_COLLECT_MAX_POLICY_STEPS),
    )
    p.add_argument(
        "--collect-max-episode-steps",
        type=int,
        default=int(DEFAULT_COLLECT_MAX_EPISODE_STEPS),
    )
    p.add_argument(
        "--n-action-steps-config", type=int, default=int(DEFAULT_N_ACTION_STEPS_CONFIG)
    )

    p.add_argument(
        "--finetune-max-steps", type=int, default=int(DEFAULT_FINETUNE_MAX_STEPS)
    )
    p.add_argument(
        "--finetune-save-steps", type=int, default=int(DEFAULT_FINETUNE_SAVE_STEPS)
    )

    p.add_argument("--eval-episodes", type=int, default=int(DEFAULT_EVAL_EPISODES))

    p.add_argument(
        "--timeout-collect-s", type=float, default=float(DEFAULT_TIMEOUT_COLLECT_S)
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

    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--require-git-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require git status --porcelain=v1 to be empty before running stages",
    )
    p.add_argument(
        "--write-repro-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write git repro snapshot under agent/artifacts/p38/<run_id>/repro",
    )
    return p


class RecapOnlineLoopIterateWorkflow:
    def run(self) -> int:
        if any(a in ("-h", "--help") for a in sys.argv[1:]):
            try:
                _build_parser().parse_args()
            except SystemExit as e:
                return int(getattr(e, "code", 0) or 0)
            return 0

        args = _build_parser().parse_args()
        repo_root = _repo_root()
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        run_id = str(args.run_id)
        start_policy_path = str(args.start_policy_path)
        dry_run = bool(args.dry_run)
        require_git_clean = bool(getattr(args, "require_git_clean", True))
        write_repro_snapshot = bool(getattr(args, "write_repro_snapshot", True))

        iter_tag_k0 = f"recap_{run_id}_k0"
        iter_tag_k1 = f"recap_{run_id}_k1"

        runtime_dir = repo_root / "agent" / "runtime_logs" / "p38" / run_id
        artifacts_dir = repo_root / "agent" / "artifacts" / "p38" / run_id
        manifest_path = artifacts_dir / "manifest.json"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        wbc_py = _wbc_python(repo_root)
        stage_py = (
            wbc_py
            if wbc_py.is_file() and os.access(wbc_py, os.X_OK)
            else Path(sys.executable)
        )

        git_sha, git_dirty = _git_head_and_dirty(repo_root)
        if require_git_clean and git_dirty:
            raise RuntimeError(
                "git workspace is dirty but --require-git-clean is set; "
                "pass --no-require-git-clean to override for this run",
            )

        repro_snapshot_dir: Path | None = None
        if write_repro_snapshot:
            from work.demo_utils.repro_snapshot import create_repro_snapshot

            repro_snapshot_dir = artifacts_dir / "repro"
            create_repro_snapshot(repo_root, repro_snapshot_dir)

        header_log = runtime_dir / "00_header.log"
        with header_log.open("a", encoding="utf-8", buffering=1) as f:
            f.write(
                f"\n===== p38 header {_dt.datetime.now().isoformat(timespec='seconds')} =====\n"
            )
            f.write(f"run_id: {run_id}\n")
            f.write(f"repo_root: {repo_root.as_posix()}\n")
            f.write(f"git_sha: {git_sha}\n")
            f.write(f"git_dirty: {git_dirty}\n")
            f.write(f"sys.executable: {sys.executable}\n")
            f.write(f"stage_python: {stage_py.as_posix()}\n")
            f.write(f"start_policy_path: {start_policy_path}\n")
            f.write(f"env_name: {args.env_name}\n")
            f.write(f"embodiment_tag: {args.embodiment_tag}\n")
            f.write(f"server_host: {args.server_host}\n")
            f.write(f"server_port_base: {int(args.server_port)}\n")
            f.write(f"mujoco_gl: {args.mujoco_gl}\n")
            f.write(f"dry_run: {dry_run}\n")
            f.flush()

        isaac_root = repo_root / "submodules/Isaac-GR00T"
        wbc_ext_root = isaac_root / "external_dependencies/GR00T-WholeBodyControl"
        pp_parts = [str(isaac_root), str(wbc_ext_root)]
        existing_pp = os.environ.get("PYTHONPATH", "")
        if existing_pp:
            pp_parts.append(existing_pp)
        stage_env_base: dict[str, str] = {"PYTHONPATH": ":".join(pp_parts)}
        stage_cwd = repo_root

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "git": {"sha": git_sha, "dirty": bool(git_dirty)},
            "start_policy_path": start_policy_path,
            "iter_tags": {"k0": iter_tag_k0, "k1": iter_tag_k1},
            "ports": {
                "base": int(args.server_port),
                "collect_k0": int(args.server_port) + 0,
                "eval": int(args.server_port) + 1,
                "collect_k1": int(args.server_port) + 2,
            },
            "seed": int(args.seed),
            "stage_python": stage_py.as_posix(),
            "runtime_dir": runtime_dir.as_posix(),
            "artifacts_dir": artifacts_dir.as_posix(),
            "stages": [],
        }
        if repro_snapshot_dir is not None:
            manifest["repro_snapshot_dir"] = repro_snapshot_dir.as_posix()

        def write_manifest() -> None:
            _write_json(manifest_path, manifest)

        def stage(
            *,
            name: str,
            log_name: str,
            cmd: list[str],
            timeout_s: float,
            tags: dict[str, str],
            port: int | None = None,
            dry_skip: bool = False,
            env_overrides: dict[str, str] | None = None,
        ) -> None:
            log_path = runtime_dir / log_name
            rec: dict[str, Any] = {
                "name": name,
                "log_path": log_path.as_posix(),
                "cmd": [str(x) for x in cmd],
                "cwd": stage_cwd.as_posix(),
                "timeout_s": float(timeout_s),
                "tags": dict(tags),
                "port": int(port) if port is not None else None,
                "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "rc": None,
                "skipped": bool(dry_skip),
            }
            manifest["stages"].append(rec)
            write_manifest()

            if dry_skip:
                rec["ended_at"] = _dt.datetime.now().isoformat(timespec="seconds")
                write_manifest()
                return

            if port is not None:
                _require_port_free(str(args.server_host), int(port), context=name)

            env = dict(stage_env_base)
            if env_overrides:
                env.update(env_overrides)

            rc = _run_cmd_tee(
                cmd,
                log_path=log_path,
                header=f"p38:{name}",
                timeout_s=float(timeout_s),
                cwd=stage_cwd,
                env=env,
            )
            rec["rc"] = int(rc)
            rec["ended_at"] = _dt.datetime.now().isoformat(timespec="seconds")
            write_manifest()
            if int(rc) != 0:
                raise RuntimeError(f"Stage failed: {name} rc={rc} (see {log_path})")

        if dry_run:
            stage(
                name="10_collect_k0",
                log_name="10_collect_k0.log",
                cmd=["echo", "dry-run: collect k0"],
                timeout_s=float(args.timeout_collect_s),
                tags={"iter_tag": iter_tag_k0},
                port=int(args.server_port) + 0,
                dry_skip=True,
            )
            stage(
                name="20_label_k0",
                log_name="20_label_k0.log",
                cmd=["echo", "dry-run: label k0"],
                timeout_s=float(args.timeout_label_s),
                tags={"iter_tag": iter_tag_k0},
                dry_skip=True,
            )
            stage(
                name="30_export_with_video_k0",
                log_name="30_export_with_video_k0.log",
                cmd=["echo", "dry-run: export with video k0"],
                timeout_s=float(args.timeout_export_s),
                tags={"iter_tag": iter_tag_k0},
                dry_skip=True,
            )
            stage(
                name="40_finetune_k0",
                log_name="40_finetune_k0.log",
                cmd=["echo", "dry-run: finetune k0"],
                timeout_s=float(args.timeout_finetune_s),
                tags={"iter_tag": iter_tag_k0},
                dry_skip=True,
            )
            for nm, ln in (
                ("50_eval_k0_base_raw", "50_eval_k0_base_raw.log"),
                ("51_eval_k0_base_advpos", "51_eval_k0_base_advpos.log"),
                ("52_eval_k0_ft_raw", "52_eval_k0_ft_raw.log"),
                ("53_eval_k0_ft_advpos", "53_eval_k0_ft_advpos.log"),
            ):
                stage(
                    name=nm,
                    log_name=ln,
                    cmd=["echo", f"dry-run: {nm}"],
                    timeout_s=float(args.timeout_eval_s),
                    tags={"eval_tag": f"{run_id}_{nm}"},
                    port=int(args.server_port) + 1,
                    dry_skip=True,
                )
            stage(
                name="11_collect_k1",
                log_name="11_collect_k1.log",
                cmd=["echo", "dry-run: collect k1"],
                timeout_s=float(args.timeout_collect_s),
                tags={"iter_tag": iter_tag_k1},
                port=int(args.server_port) + 2,
                dry_skip=True,
            )

            _write_json(manifest_path, manifest)
            return 0

        stage_py_str = stage_py.as_posix()

        stage(
            name="preflight",
            log_name="01_preflight.log",
            cmd=[
                stage_py_str,
                "-c",
                "import gr00t, gr00t_wbc; import numpy, pandas, pyarrow; print('preflight ok')",
            ],
            timeout_s=60.0,
            tags={},
            dry_skip=False,
        )

        cmd_collect_k0 = [
            stage_py_str,
            str(repo_root / "work/recap/scripts/31_recap_collect_rollouts.py"),
            "--env-name",
            str(args.env_name),
            "--model-path",
            str(start_policy_path),
            "--embodiment-tag",
            str(args.embodiment_tag),
            "--server-host",
            str(args.server_host),
            "--server-port",
            str(int(args.server_port) + 0),
            "--iter-tag",
            str(iter_tag_k0),
            "--seed",
            str(int(args.seed)),
            "--n-episodes",
            str(int(args.collect_episodes)),
            "--max-policy-steps",
            str(int(args.collect_max_policy_steps)),
            "--max-episode-steps",
            str(int(args.collect_max_episode_steps)),
            "--n-action-steps-config",
            str(int(args.n_action_steps_config)),
            "--mujoco-gl",
            str(args.mujoco_gl),
            "--offscreen",
            "--kill-server-on-exit",
            "--total-timeout-s",
            str(float(args.timeout_collect_s)),
        ]
        stage(
            name="10_collect_k0",
            log_name="10_collect_k0.log",
            cmd=cmd_collect_k0,
            timeout_s=float(args.timeout_collect_s),
            tags={"iter_tag": iter_tag_k0, "model_path": start_policy_path},
            port=int(args.server_port) + 0,
        )

        cmd_label_k0 = [
            stage_py_str,
            str(repo_root / "work/recap/scripts/32_recap_label_dataset.py"),
            "--iter-tag",
            str(iter_tag_k0),
            "--total-timeout-s",
            str(float(args.timeout_label_s)),
        ]
        stage(
            name="20_label_k0",
            log_name="20_label_k0.log",
            cmd=cmd_label_k0,
            timeout_s=float(args.timeout_label_s),
            tags={"iter_tag": iter_tag_k0},
        )

        cmd_export_k0 = [
            stage_py_str,
            str(repo_root / "work/recap/scripts/39_recap_export_lerobot_v2_with_video.py"),
            "--iter-tag",
            str(iter_tag_k0),
            "--max-episodes",
            str(int(args.collect_episodes)),
            "--total-timeout-s",
            str(float(args.timeout_export_s)),
        ]
        stage(
            name="30_export_with_video_k0",
            log_name="30_export_with_video_k0.log",
            cmd=cmd_export_k0,
            timeout_s=float(args.timeout_export_s),
            tags={"iter_tag": iter_tag_k0},
        )

        out_dir = (
            repo_root
            / "agent"
            / "artifacts"
            / "checkpoints"
            / iter_tag_k0
            / "p38_finetune_k0"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        env_finetune: dict[str, str] = {}
        dataset_path = repo_root / "agent" / "artifacts" / "lerobot_datasets" / iter_tag_k0

        cmd_finetune = [
            stage_py_str,
            str(repo_root / "submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py"),
            "--base-model-path",
            str(start_policy_path),
            "--dataset-path",
            str(dataset_path),
            "--embodiment-tag",
            str(args.embodiment_tag),
            "--tune-diffusion-model",
            "--no-tune-projector",
            "--no-use-wandb",
            "--output-dir",
            str(out_dir),
            "--max-steps",
            str(int(args.finetune_max_steps)),
            "--save-steps",
            str(int(args.finetune_save_steps)),
            "--save-total-limit",
            "2",
            "--global-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--dataloader-num-workers",
            "0",
        ]
        stage(
            name="40_finetune_k0",
            log_name="40_finetune_k0.log",
            cmd=cmd_finetune,
            timeout_s=float(args.timeout_finetune_s),
            tags={"iter_tag": iter_tag_k0, "out_dir": out_dir.as_posix()},
            env_overrides=env_finetune,
        )

        selected_ckpt = _select_latest_checkpoint(out_dir)
        manifest["selected_checkpoint_path"] = selected_ckpt.as_posix()
        _write_json(manifest_path, manifest)
        with (runtime_dir / "40_finetune_k0.log").open(
            "a", encoding="utf-8", buffering=1
        ) as f:
            f.write(f"\n[INFO] selected_checkpoint_path: {selected_ckpt.as_posix()}\n")

        def eval_stage(name: str, log_name: str, *, model_path: str, prefix: str) -> None:
            eval_tag = f"recap_{run_id}_{name}"
            cmd_eval = [
                stage_py_str,
                str(repo_root / "work/recap/scripts/31_recap_collect_rollouts.py"),
                "--env-name",
                str(args.env_name),
                "--model-path",
                str(model_path),
                "--embodiment-tag",
                str(args.embodiment_tag),
                "--server-host",
                str(args.server_host),
                "--server-port",
                str(int(args.server_port) + 1),
                "--iter-tag",
                str(eval_tag),
                "--seed",
                str(int(args.seed)),
                "--n-episodes",
                str(int(args.eval_episodes)),
                "--max-policy-steps",
                str(int(args.collect_max_policy_steps)),
                "--max-episode-steps",
                str(int(args.collect_max_episode_steps)),
                "--n-action-steps-config",
                str(int(args.n_action_steps_config)),
                "--mujoco-gl",
                str(args.mujoco_gl),
                "--policy-prompt-prefix",
                str(prefix),
                "--offscreen",
                "--kill-server-on-exit",
                "--total-timeout-s",
                str(float(args.timeout_eval_s)),
            ]
            stage(
                name=name,
                log_name=log_name,
                cmd=cmd_eval,
                timeout_s=float(args.timeout_eval_s),
                tags={
                    "eval_tag": eval_tag,
                    "model_path": str(model_path),
                    "policy_prompt_prefix": str(prefix),
                },
                port=int(args.server_port) + 1,
            )

        eval_stage(
            "50_eval_k0_base_raw",
            "50_eval_k0_base_raw.log",
            model_path=str(start_policy_path),
            prefix="",
        )
        eval_stage(
            "51_eval_k0_base_advpos",
            "51_eval_k0_base_advpos.log",
            model_path=str(start_policy_path),
            prefix="advantage positive ",
        )
        eval_stage(
            "52_eval_k0_ft_raw",
            "52_eval_k0_ft_raw.log",
            model_path=str(selected_ckpt.as_posix()),
            prefix="",
        )
        eval_stage(
            "53_eval_k0_ft_advpos",
            "53_eval_k0_ft_advpos.log",
            model_path=str(selected_ckpt.as_posix()),
            prefix="advantage positive ",
        )

        cmd_collect_k1 = [
            stage_py_str,
            str(repo_root / "work/recap/scripts/31_recap_collect_rollouts.py"),
            "--env-name",
            str(args.env_name),
            "--model-path",
            str(selected_ckpt.as_posix()),
            "--embodiment-tag",
            str(args.embodiment_tag),
            "--server-host",
            str(args.server_host),
            "--server-port",
            str(int(args.server_port) + 2),
            "--iter-tag",
            str(iter_tag_k1),
            "--seed",
            str(int(args.seed)),
            "--n-episodes",
            str(int(args.collect_episodes)),
            "--max-policy-steps",
            str(int(args.collect_max_policy_steps)),
            "--max-episode-steps",
            str(int(args.collect_max_episode_steps)),
            "--n-action-steps-config",
            str(int(args.n_action_steps_config)),
            "--mujoco-gl",
            str(args.mujoco_gl),
            "--offscreen",
            "--kill-server-on-exit",
            "--total-timeout-s",
            str(float(args.timeout_collect_s)),
        ]
        stage(
            name="11_collect_k1",
            log_name="11_collect_k1.log",
            cmd=cmd_collect_k1,
            timeout_s=float(args.timeout_collect_s),
            tags={"iter_tag": iter_tag_k1, "model_path": selected_ckpt.as_posix()},
            port=int(args.server_port) + 2,
        )

        _write_json(manifest_path, manifest)
        return 0



def main() -> int:
    return RecapOnlineLoopIterateWorkflow().run()
if __name__ == "__main__":
    raise SystemExit(main())


class RecapOnlineLoopIterateScriptApp:
    def run(self) -> int:
        return main()
