from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_TIMEOUT_ARCHIVE_S = 12 * 3600


@dataclass(frozen=True)
class WorkflowDependencies:
    now_iso: Callable[[], str]
    write_json: Callable[[Path, Any], None]
    run_cmd_tee: Callable[..., int]
    require_port_free: Callable[..., None]
    select_latest_checkpoint: Callable[[Path], Path]


@dataclass(frozen=True)
class RecapMultiIterLoopConfig:
    repo_root: Path
    run_id: str
    start_policy_path: str
    n_iterations: int
    dry_run: bool
    require_git_clean: bool
    write_repro_snapshot: bool
    env_name: str
    embodiment_tag: str
    mujoco_gl: str
    n_action_steps_config: int
    server_host: str
    server_port_base: int
    seed: int
    seed_offset_per_iter: int
    fixed_eval_seed: int | None
    collect_episodes: int
    collect_max_policy_steps: int
    mixdone: bool
    dual_task_text: bool
    mixdone_short_episodes: int | None
    mixdone_long_episodes: int | None
    mixdone_short_max_episode_steps: int
    mixdone_long_max_episode_steps: int
    mixdone_long_seed_offset: int
    critic_bins: int
    critic_max_epochs: int
    critic_patience: int
    critic_lr: float
    critic_val_ratio: float
    critic_device: str
    finetune_max_steps: int
    finetune_save_steps: int
    finetune_save_total_limit: int
    finetune_tune_projector: bool
    finetune_tune_diffusion_model: bool
    eval_episodes: int
    eval_max_policy_steps: int | None
    eval_policy_prompt_prefix: str
    timeout_collect_s: float
    timeout_critic_s: float
    timeout_label_s: float
    timeout_export_s: float
    timeout_finetune_s: float
    timeout_eval_s: float
    min_free_gb: float
    archive_root: Path
    keep_last_n_iters_local: int
    pin_checkpoint_dirs: tuple[str, ...]
    git_sha: str
    git_dirty: bool
    stage_python: str

    @property
    def runtime_dir(self) -> Path:
        return self.repo_root / "agent" / "runtime_logs" / "p3A" / self.run_id

    @property
    def artifacts_dir(self) -> Path:
        return self.repo_root / "agent" / "artifacts" / "p3A" / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_dir / "manifest.json"

    @property
    def port_collect(self) -> int:
        return int(self.server_port_base)

    @property
    def port_eval(self) -> int:
        return int(self.server_port_base) + 1

    def build_stage_env_base(self) -> dict[str, str]:
        isaac_root = self.repo_root / "submodules/Isaac-GR00T"
        wbc_ext_root = isaac_root / "external_dependencies/GR00T-WholeBodyControl"
        pythonpath_parts = [str(isaac_root), str(wbc_ext_root)]
        existing_pp = os.environ.get("PYTHONPATH", "")
        if existing_pp:
            pythonpath_parts.append(existing_pp)
        return {"PYTHONPATH": ":".join(pythonpath_parts)}

    def manifest_params(self) -> dict[str, Any]:
        return {
            "n_iterations": int(self.n_iterations),
            "env_name": str(self.env_name),
            "embodiment_tag": str(self.embodiment_tag),
            "mujoco_gl": str(self.mujoco_gl),
            "n_action_steps_config": int(self.n_action_steps_config),
            "server_host": str(self.server_host),
            "server_port_base": int(self.server_port_base),
            "seed_offset_per_iter": int(self.seed_offset_per_iter),
            "fixed_eval_seed": self.fixed_eval_seed,
            "collect_episodes": int(self.collect_episodes),
            "collect_max_policy_steps": int(self.collect_max_policy_steps),
            "mixdone": bool(self.mixdone),
            "dual_task_text": bool(self.dual_task_text),
            "mixdone_short_episodes": self.mixdone_short_episodes,
            "mixdone_long_episodes": self.mixdone_long_episodes,
            "mixdone_short_max_episode_steps": int(
                self.mixdone_short_max_episode_steps
            ),
            "mixdone_long_max_episode_steps": int(self.mixdone_long_max_episode_steps),
            "mixdone_long_seed_offset": int(self.mixdone_long_seed_offset),
            "critic_bins": int(self.critic_bins),
            "critic_max_epochs": int(self.critic_max_epochs),
            "critic_patience": int(self.critic_patience),
            "critic_lr": float(self.critic_lr),
            "critic_val_ratio": float(self.critic_val_ratio),
            "critic_device": str(self.critic_device),
            "finetune_max_steps": int(self.finetune_max_steps),
            "finetune_save_steps": int(self.finetune_save_steps),
            "finetune_save_total_limit": int(self.finetune_save_total_limit),
            "min_free_gb": float(self.min_free_gb),
            "archive_root": self.archive_root.as_posix(),
            "keep_last_n_iters_local": int(self.keep_last_n_iters_local),
            "pin_checkpoint_dir": list(self.pin_checkpoint_dirs),
            "finetune_tune_projector": bool(self.finetune_tune_projector),
            "finetune_tune_diffusion_model": bool(self.finetune_tune_diffusion_model),
            "eval_episodes": int(self.eval_episodes),
            "eval_max_policy_steps": self.eval_max_policy_steps,
            "eval_policy_prompt_prefix": str(self.eval_policy_prompt_prefix),
            "timeout_collect_s": float(self.timeout_collect_s),
            "timeout_critic_s": float(self.timeout_critic_s),
            "timeout_label_s": float(self.timeout_label_s),
            "timeout_export_s": float(self.timeout_export_s),
            "timeout_finetune_s": float(self.timeout_finetune_s),
            "timeout_eval_s": float(self.timeout_eval_s),
            "require_git_clean": bool(self.require_git_clean),
            "write_repro_snapshot": bool(self.write_repro_snapshot),
            "dry_run": bool(self.dry_run),
        }


@dataclass(frozen=True)
class IterationPlan:
    index: int
    iter_tag: str
    critic_tag: str
    seed_k: int
    seed_eval_k: int
    recap_dataset_dir: Path
    critic_dir: Path
    lerobot_dataset_dir: Path
    finetune_out_dir: Path
    eval_iter_tag_ft_advpos: str
    eval_iter_tag_base_advpos: str | None

    def build_manifest_entry(self, *, policy_path_in: str) -> dict[str, Any]:
        return {
            "k": int(self.index),
            "iter_tag": str(self.iter_tag),
            "critic_tag": str(self.critic_tag),
            "started_at": None,
            "stages": [],
            "outputs": {
                "recap_dataset_dir": self.recap_dataset_dir.as_posix(),
                "critic_dir": self.critic_dir.as_posix(),
                "lerobot_dataset_dir": self.lerobot_dataset_dir.as_posix(),
                "finetune_out_dir": self.finetune_out_dir.as_posix(),
                "eval_iter_tag_ft_advpos": str(self.eval_iter_tag_ft_advpos),
                "eval_iter_tag_base_advpos": self.eval_iter_tag_base_advpos,
                "policy_path_used_for_collect": str(policy_path_in),
                "policy_path_after_finetune": None,
                "selected_checkpoint_path": None,
            },
            "iter_tags_for_critic": [],
            "policy_path_in": str(policy_path_in),
        }


@dataclass(frozen=True)
class StageSpec:
    name: str
    log_name: str
    cmd: list[str]
    timeout_s: float
    tags: dict[str, str]
    port: int | None
    non_fatal: bool = False


class DiskBudgetError(RuntimeError):
    pass


def build_workflow_config(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    git_sha: str,
    git_dirty: bool,
    stage_python: str,
    validate_tag: Callable[[str, str], str],
) -> RecapMultiIterLoopConfig:
    run_id = validate_tag(str(args.run_id), "run_id")
    n_iterations = int(args.n_iterations)
    if n_iterations <= 0:
        raise ValueError(f"--n-iterations must be > 0, got {n_iterations}")

    min_free_gb = float(getattr(args, "min_free_gb"))
    if not (min_free_gb > 0.0):
        raise ValueError(f"--min-free-gb must be > 0, got {min_free_gb}")

    keep_last_n_iters_local = int(getattr(args, "keep_last_n_iters_local"))
    if keep_last_n_iters_local < 0:
        raise ValueError(
            f"--keep-last-n-iters-local must be >= 0, got {keep_last_n_iters_local}"
        )

    pin_checkpoint_dirs_raw = list(getattr(args, "pin_checkpoint_dir", []) or [])
    pinned_iter_tags: list[str] = []
    seen: set[str] = set()
    for value in pin_checkpoint_dirs_raw:
        tag = validate_tag(str(value), "pin_checkpoint_dir")
        if tag in seen:
            continue
        seen.add(tag)
        pinned_iter_tags.append(tag)

    finetune_save_total_limit = int(getattr(args, "finetune_save_total_limit"))
    if finetune_save_total_limit <= 0:
        raise ValueError(
            f"--finetune-save-total-limit must be > 0, got {finetune_save_total_limit}"
        )

    finetune_save_steps_raw = getattr(args, "finetune_save_steps", None)
    finetune_save_steps = (
        int(args.finetune_max_steps)
        if finetune_save_steps_raw is None
        else int(finetune_save_steps_raw)
    )
    if finetune_save_steps <= 0:
        raise ValueError(
            f"--finetune-save-steps must be > 0 (or unset), got {finetune_save_steps_raw!r}"
        )

    fixed_eval_seed_raw = getattr(args, "fixed_eval_seed", None)
    fixed_eval_seed = None if fixed_eval_seed_raw is None else int(fixed_eval_seed_raw)

    return RecapMultiIterLoopConfig(
        repo_root=Path(repo_root),
        run_id=str(run_id),
        start_policy_path=str(args.start_policy_path),
        n_iterations=int(n_iterations),
        dry_run=bool(args.dry_run),
        require_git_clean=bool(getattr(args, "require_git_clean", True)),
        write_repro_snapshot=bool(getattr(args, "write_repro_snapshot", True)),
        env_name=str(args.env_name),
        embodiment_tag=str(args.embodiment_tag),
        mujoco_gl=str(args.mujoco_gl),
        n_action_steps_config=int(args.n_action_steps_config),
        server_host=str(args.server_host),
        server_port_base=int(args.server_port),
        seed=int(args.seed),
        seed_offset_per_iter=int(args.seed_offset_per_iter),
        fixed_eval_seed=fixed_eval_seed,
        collect_episodes=int(args.collect_episodes),
        collect_max_policy_steps=int(args.collect_max_policy_steps),
        mixdone=bool(args.mixdone),
        dual_task_text=bool(getattr(args, "dual_task_text", True)),
        mixdone_short_episodes=(
            None
            if args.mixdone_short_episodes is None
            else int(args.mixdone_short_episodes)
        ),
        mixdone_long_episodes=(
            None
            if args.mixdone_long_episodes is None
            else int(args.mixdone_long_episodes)
        ),
        mixdone_short_max_episode_steps=int(args.mixdone_short_max_episode_steps),
        mixdone_long_max_episode_steps=int(args.mixdone_long_max_episode_steps),
        mixdone_long_seed_offset=int(args.mixdone_long_seed_offset),
        critic_bins=int(args.critic_bins),
        critic_max_epochs=int(args.critic_max_epochs),
        critic_patience=int(args.critic_patience),
        critic_lr=float(args.critic_lr),
        critic_val_ratio=float(args.critic_val_ratio),
        critic_device=str(args.critic_device),
        finetune_max_steps=int(args.finetune_max_steps),
        finetune_save_steps=int(finetune_save_steps),
        finetune_save_total_limit=int(finetune_save_total_limit),
        finetune_tune_projector=bool(getattr(args, "finetune_tune_projector")),
        finetune_tune_diffusion_model=bool(
            getattr(args, "finetune_tune_diffusion_model")
        ),
        eval_episodes=int(args.eval_episodes),
        eval_max_policy_steps=(
            None
            if getattr(args, "eval_max_policy_steps", None) is None
            else int(getattr(args, "eval_max_policy_steps"))
        ),
        eval_policy_prompt_prefix=str(args.eval_policy_prompt_prefix),
        timeout_collect_s=float(args.timeout_collect_s),
        timeout_critic_s=float(args.timeout_critic_s),
        timeout_label_s=float(args.timeout_label_s),
        timeout_export_s=float(args.timeout_export_s),
        timeout_finetune_s=float(args.timeout_finetune_s),
        timeout_eval_s=float(args.timeout_eval_s),
        min_free_gb=float(min_free_gb),
        archive_root=Path(str(getattr(args, "archive_root"))).expanduser(),
        keep_last_n_iters_local=int(keep_last_n_iters_local),
        pin_checkpoint_dirs=tuple(pinned_iter_tags),
        git_sha=str(git_sha),
        git_dirty=bool(git_dirty),
        stage_python=str(stage_python),
    )


class RecapMultiIterLoopWorkflow:
    def __init__(
        self,
        config: RecapMultiIterLoopConfig,
        dependencies: WorkflowDependencies,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.stage_env_base = self.config.build_stage_env_base()
        self.manifest: dict[str, Any] = {}
        self.current_policy_path = str(self.config.start_policy_path)
        self.last_good_policy_path = str(self.config.start_policy_path)
        self.collected_iter_tags: list[str] = []
        self.disk_budget_abort: DiskBudgetError | None = None
        self.repro_snapshot_dir: Path | None = None

    def run(self) -> int:
        self._prepare_runtime_dirs()
        self._initialize_manifest()
        self._write_manifest()
        self._write_header_log()

        git_check_rc = self._ensure_git_clean_if_required()
        if git_check_rc is not None:
            return git_check_rc

        repro_snapshot_rc = self._write_repro_snapshot_if_enabled()
        if repro_snapshot_rc is not None:
            return repro_snapshot_rc

        self._configure_ports()
        self._preflight_ports()

        abort_rc = self._run_iterations()
        if abort_rc is not None:
            return abort_rc

        return self._finalize_success()

    def _prepare_runtime_dirs(self) -> None:
        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_manifest(self) -> None:
        self.manifest = {
            "run_id": self.config.run_id,
            "created_at": self._now_iso(),
            "git": {
                "sha": str(self.config.git_sha),
                "dirty": bool(self.config.git_dirty),
            },
            "seed": int(self.config.seed),
            "start_policy_path": str(self.config.start_policy_path),
            "runtime_dir": self.config.runtime_dir.as_posix(),
            "artifacts_dir": self.config.artifacts_dir.as_posix(),
            "manifest_path": self.config.manifest_path.as_posix(),
            "stage_python": str(self.config.stage_python),
            "params": self.config.manifest_params(),
            "disk_checks": [],
            "checkpoint_prune": [],
            "archive_moves": [],
            "iterations": [],
        }

    def _configure_ports(self) -> None:
        self.manifest["ports"] = {
            "base": int(self.config.server_port_base),
            "collect": int(self.config.port_collect),
            "eval": int(self.config.port_eval),
        }
        self._write_manifest()

    def _run_iterations(self) -> int | None:
        try:
            for k in range(int(self.config.n_iterations)):
                abort_rc = self._run_single_iteration(k)
                if abort_rc is not None:
                    return abort_rc
        except KeyboardInterrupt as exc:
            self._mark_running_stages_terminal(reason=f"interrupted:{exc}", rc=130)
            return self._abort_run(error=f"interrupted:{exc}", rc=130)
        except Exception as exc:
            self._mark_running_stages_terminal(
                reason=f"unhandled_exception:{exc}", rc=1
            )
            return self._abort_run(error=f"unhandled_exception:{exc}", rc=1)
        return None

    def _run_single_iteration(self, k: int) -> int | None:
        plan = self._build_iteration_plan(k)
        iteration_entry = plan.build_manifest_entry(
            policy_path_in=self.current_policy_path
        )
        iteration_entry["started_at"] = self._now_iso()
        self.manifest["iterations"].append(iteration_entry)
        self._write_manifest()

        stage_specs = self._build_stage_specs(plan)
        failed = False
        finetune_failed = False
        failure_reason: str | None = None

        for spec in stage_specs:
            if self.config.dry_run:
                self._run_stage(
                    iteration_entry=iteration_entry,
                    spec=spec,
                    cmd=list(spec.cmd),
                    tags=dict(spec.tags),
                    dry_skip=True,
                    skip_reason="dry_run",
                )
                continue

            if failed and not (finetune_failed and spec.name == "60_eval_ft_advpos"):
                self._run_stage(
                    iteration_entry=iteration_entry,
                    spec=spec,
                    cmd=list(spec.cmd),
                    tags=dict(spec.tags),
                    dry_skip=True,
                    skip_reason=f"upstream_failure:{failure_reason or 'unknown'}",
                )
                continue

            cmd, tags = self._resolve_stage_invocation(plan, iteration_entry, spec)
            rc, ok = self._run_stage(
                iteration_entry=iteration_entry,
                spec=spec,
                cmd=cmd,
                tags=tags,
                dry_skip=False,
                skip_reason=None,
            )
            if self.disk_budget_abort is not None:
                return self._handle_disk_budget_abort(
                    where=str(spec.name), err=self.disk_budget_abort
                )
            if not ok:
                failed = True
                failure_reason = f"{spec.name}:rc={rc}"
                if spec.name == "50_finetune_upstream":
                    finetune_failed = True

            if ok:
                post_failed, post_reason, post_finetune_failed = (
                    self._postprocess_success(
                        plan=plan,
                        iteration_entry=iteration_entry,
                        spec=spec,
                    )
                )
                if post_failed:
                    failed = True
                    failure_reason = post_reason
                    finetune_failed = finetune_failed or post_finetune_failed

        self._finalize_iteration(
            iteration_entry=iteration_entry,
            failed=failed,
            failure_reason=failure_reason,
        )
        return self._maybe_archive_previous_iteration(k)

    def _build_iteration_plan(self, k: int) -> IterationPlan:
        iter_tag = self._validate_tag(
            f"recap_{self.config.run_id}_k{k}", name="iter_tag"
        )
        critic_tag = self._validate_tag(
            f"critic_{self.config.run_id}_k{k}", name="critic_tag"
        )
        eval_iter_tag_ft_advpos = self._validate_tag(
            f"recap_{self.config.run_id}_k{k}_eval_ft_advpos", name="eval_iter_tag"
        )
        eval_iter_tag_base_advpos = (
            self._validate_tag(
                f"recap_{self.config.run_id}_k{k}_eval_base_advpos",
                name="eval_iter_tag",
            )
            if int(k) == 0
            else None
        )
        return IterationPlan(
            index=int(k),
            iter_tag=str(iter_tag),
            critic_tag=str(critic_tag),
            seed_k=int(self.config.seed)
            + int(k) * int(self.config.seed_offset_per_iter),
            seed_eval_k=(
                int(self.config.seed)
                + int(k) * int(self.config.seed_offset_per_iter)
                + 5000
            ),
            recap_dataset_dir=self.config.repo_root
            / "agent"
            / "artifacts"
            / "recap_datasets"
            / str(iter_tag),
            critic_dir=self.config.repo_root
            / "agent"
            / "artifacts"
            / "critics"
            / str(critic_tag),
            lerobot_dataset_dir=self.config.repo_root
            / "agent"
            / "artifacts"
            / "lerobot_datasets"
            / str(iter_tag),
            finetune_out_dir=self.config.repo_root
            / "agent"
            / "artifacts"
            / "checkpoints"
            / str(iter_tag)
            / f"p3A_finetune_k{k}",
            eval_iter_tag_ft_advpos=str(eval_iter_tag_ft_advpos),
            eval_iter_tag_base_advpos=eval_iter_tag_base_advpos,
        )

    def _build_stage_specs(self, plan: IterationPlan) -> list[StageSpec]:
        collect_cmd = self._build_collect_command(
            plan=plan,
            iter_tag_value=str(plan.iter_tag),
            model_path=str(self.current_policy_path),
            server_port=int(self.config.port_collect),
            n_episodes=int(self.config.collect_episodes),
            policy_prompt_prefix="",
            enable_mixdone=bool(self.config.mixdone),
        )
        critic_iter_tags = list(self.collected_iter_tags) + [str(plan.iter_tag)]
        critic_cmd = self._build_critic_command(
            plan=plan, critic_iter_tags=critic_iter_tags
        )
        label_cmd = self._build_label_command(plan=plan)
        export_cmd = self._build_export_command(plan=plan)
        finetune_cmd = self._build_finetune_command(plan=plan)
        eval_ft_cmd = self._build_collect_command(
            plan=plan,
            iter_tag_value=str(plan.eval_iter_tag_ft_advpos),
            model_path=str(self.current_policy_path),
            server_port=int(self.config.port_eval),
            n_episodes=int(self.config.eval_episodes),
            policy_prompt_prefix=str(self.config.eval_policy_prompt_prefix),
            enable_mixdone=False,
            seed_override=self.config.fixed_eval_seed,
            max_policy_steps_override=self._eval_max_policy_steps(),
        )

        specs = [
            StageSpec(
                name="10_collect",
                log_name=f"iter{plan.index:02d}_10_collect.log",
                cmd=collect_cmd,
                timeout_s=float(self.config.timeout_collect_s),
                tags={
                    "iter_tag": str(plan.iter_tag),
                    "model_path": str(self.current_policy_path),
                    "mixdone": str(bool(self.config.mixdone)),
                },
                port=int(self.config.port_collect),
            ),
            StageSpec(
                name="20_critic_cumulative",
                log_name=f"iter{plan.index:02d}_20_critic_cumulative.log",
                cmd=critic_cmd,
                timeout_s=float(self.config.timeout_critic_s),
                tags={
                    "critic_tag": str(plan.critic_tag),
                    "iter_tags": ",".join(critic_iter_tags),
                },
                port=None,
            ),
            StageSpec(
                name="30_label_value_source_critic",
                log_name=f"iter{plan.index:02d}_30_label_value_source_critic.log",
                cmd=label_cmd,
                timeout_s=float(self.config.timeout_label_s),
                tags={
                    "iter_tag": str(plan.iter_tag),
                    "critic_dir": plan.critic_dir.as_posix(),
                },
                port=None,
            ),
            StageSpec(
                name="40_export_with_video_dual_task_text",
                log_name=f"iter{plan.index:02d}_40_export_with_video_dual_task_text.log",
                cmd=export_cmd,
                timeout_s=float(self.config.timeout_export_s),
                tags={
                    "iter_tag": str(plan.iter_tag),
                    "require_ffmpeg": "true",
                    "dual_task_text": "true" if self.config.dual_task_text else "false",
                },
                port=None,
            ),
            StageSpec(
                name="50_finetune_upstream",
                log_name=f"iter{plan.index:02d}_50_finetune_upstream.log",
                cmd=finetune_cmd,
                timeout_s=float(self.config.timeout_finetune_s),
                tags={
                    "iter_tag": str(plan.iter_tag),
                    "base_model_path": str(self.config.start_policy_path),
                    "dataset_path": plan.lerobot_dataset_dir.as_posix(),
                    "output_dir": plan.finetune_out_dir.as_posix(),
                    "tune_projector": str(bool(self.config.finetune_tune_projector)),
                    "tune_diffusion_model": str(
                        bool(self.config.finetune_tune_diffusion_model)
                    ),
                },
                port=None,
            ),
            StageSpec(
                name="60_eval_ft_advpos",
                log_name=f"iter{plan.index:02d}_60_eval_ft_advpos.log",
                cmd=eval_ft_cmd,
                timeout_s=float(self.config.timeout_eval_s),
                tags={
                    "eval_iter_tag": str(plan.eval_iter_tag_ft_advpos),
                    "policy_prompt_prefix": str(self.config.eval_policy_prompt_prefix),
                },
                port=int(self.config.port_eval),
            ),
        ]

        if plan.eval_iter_tag_base_advpos is not None:
            specs.insert(
                0,
                StageSpec(
                    name="05_eval_base_advpos",
                    log_name=f"iter{plan.index:02d}_05_eval_base_advpos.log",
                    cmd=self._build_collect_command(
                        plan=plan,
                        iter_tag_value=str(plan.eval_iter_tag_base_advpos),
                        model_path=str(self.config.start_policy_path),
                        server_port=int(self.config.port_eval),
                        n_episodes=int(self.config.eval_episodes),
                        policy_prompt_prefix=str(self.config.eval_policy_prompt_prefix),
                        enable_mixdone=False,
                        seed_override=self.config.fixed_eval_seed,
                        max_policy_steps_override=self._eval_max_policy_steps(),
                    ),
                    timeout_s=float(self.config.timeout_eval_s),
                    tags={
                        "eval_iter_tag": str(plan.eval_iter_tag_base_advpos),
                        "model_path": str(self.config.start_policy_path),
                        "policy_prompt_prefix": str(
                            self.config.eval_policy_prompt_prefix
                        ),
                        "non_fatal": "true",
                    },
                    port=int(self.config.port_eval),
                    non_fatal=True,
                ),
            )

        iteration_entry = self.manifest["iterations"][-1]
        iteration_entry["iter_tags_for_critic"] = list(critic_iter_tags)
        return specs

    def _resolve_stage_invocation(
        self,
        plan: IterationPlan,
        iteration_entry: dict[str, Any],
        spec: StageSpec,
    ) -> tuple[list[str], dict[str, str]]:
        cmd = list(spec.cmd)
        tags = dict(spec.tags)
        if spec.name == "60_eval_ft_advpos":
            used_policy_path = str(self.current_policy_path)
            cmd = self._build_collect_command(
                plan=plan,
                iter_tag_value=str(plan.eval_iter_tag_ft_advpos),
                model_path=used_policy_path,
                server_port=int(self.config.port_eval),
                n_episodes=int(self.config.eval_episodes),
                policy_prompt_prefix=str(self.config.eval_policy_prompt_prefix),
                enable_mixdone=False,
                seed_override=self.config.fixed_eval_seed,
                max_policy_steps_override=self._eval_max_policy_steps(),
            )
            tags["model_path"] = str(used_policy_path)
            tags["eval_uses_finetuned"] = (
                "true"
                if str(used_policy_path) != str(iteration_entry.get("policy_path_in"))
                else "false"
            )
        return cmd, tags

    def _postprocess_success(
        self,
        *,
        plan: IterationPlan,
        iteration_entry: dict[str, Any],
        spec: StageSpec,
    ) -> tuple[bool, str | None, bool]:
        if spec.name == "30_label_value_source_critic":
            return self._write_advantage_contract(plan=plan, spec=spec)

        if spec.name == "10_collect":
            self.collected_iter_tags.append(str(plan.iter_tag))
            iteration_entry["outputs"]["policy_path_used_for_collect"] = str(
                self.current_policy_path
            )
            iteration_entry["iter_tags_for_critic"] = list(self.collected_iter_tags)
            self._write_manifest()
            return False, None, False

        if spec.name == "50_finetune_upstream":
            return self._select_finetuned_checkpoint(
                plan=plan, iteration_entry=iteration_entry, spec=spec
            )

        if spec.name == "60_eval_ft_advpos":
            iteration_entry["outputs"]["eval_policy_path"] = str(
                self.current_policy_path
            )
            self._write_manifest()
            return False, None, False

        return False, None, False

    def _write_advantage_contract(
        self,
        *,
        plan: IterationPlan,
        spec: StageSpec,
    ) -> tuple[bool, str | None, bool]:
        log_path = self.config.runtime_dir / str(spec.log_name)
        try:
            contract_path = self._write_continuous_advantage_contract(
                dataset_dir=plan.recap_dataset_dir,
                iter_tag=str(plan.iter_tag),
                critic_dir=plan.critic_dir,
            )
            with log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(
                    f"\n[INFO] continuous_advantage_contract_path: {contract_path.as_posix()}\n"
                )
            return False, None, False
        except Exception as exc:
            with log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(f"\n[ERROR] continuous_advantage_contract_failed: {exc}\n")
            return True, f"write_advantage_contract_failed:{exc}", False

    def _select_finetuned_checkpoint(
        self,
        *,
        plan: IterationPlan,
        iteration_entry: dict[str, Any],
        spec: StageSpec,
    ) -> tuple[bool, str | None, bool]:
        log_path = self.config.runtime_dir / str(spec.log_name)
        try:
            selected_ckpt = self.dependencies.select_latest_checkpoint(
                plan.finetune_out_dir
            )
            iteration_entry["outputs"]["selected_checkpoint_path"] = (
                selected_ckpt.as_posix()
            )
            iteration_entry["outputs"]["policy_path_after_finetune"] = (
                selected_ckpt.as_posix()
            )
            self.current_policy_path = selected_ckpt.as_posix()
            self.last_good_policy_path = str(self.current_policy_path)
            iteration_entry["policy_path_out"] = str(self.current_policy_path)
            self._write_manifest()
            with log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(
                    f"\n[INFO] selected_checkpoint_path: {selected_ckpt.as_posix()}\n"
                )
            self._prune_checkpoints_under(
                output_dir=plan.finetune_out_dir,
                keep_dir=selected_ckpt,
                stage_log_path=log_path,
                iteration_entry=iteration_entry,
            )
            return False, None, False
        except Exception as exc:
            iteration_entry["outputs"]["selected_checkpoint_path"] = None
            self._write_manifest()
            return True, f"select_checkpoint_failed:{exc}", True

    def _finalize_iteration(
        self,
        *,
        iteration_entry: dict[str, Any],
        failed: bool,
        failure_reason: str | None,
    ) -> None:
        if failed:
            iteration_entry["failed"] = True
            iteration_entry["failure_reason"] = str(failure_reason or "unknown")
            self.current_policy_path = str(self.last_good_policy_path)
        else:
            iteration_entry["failed"] = False
        iteration_entry["ended_at"] = self._now_iso()
        iteration_entry["policy_path_out"] = str(self.current_policy_path)
        self._write_manifest()

    def _maybe_archive_previous_iteration(self, k: int) -> int | None:
        if self.config.dry_run:
            return None
        archive_k = int(k) - int(self.config.keep_last_n_iters_local)
        if archive_k < 0:
            return None
        iter_tag_to_archive = self._validate_tag(
            f"recap_{self.config.run_id}_k{archive_k}", name="iter_tag_to_archive"
        )
        if iter_tag_to_archive in set(self.config.pin_checkpoint_dirs):
            return None
        try:
            self._archive_checkpoint_dir(iter_tag_value=str(iter_tag_to_archive))
        except DiskBudgetError as exc:
            return self._handle_disk_budget_abort(
                where=f"archive:{iter_tag_to_archive}", err=exc
            )
        return None

    def _run_stage(
        self,
        *,
        iteration_entry: dict[str, Any],
        spec: StageSpec,
        cmd: list[str],
        tags: dict[str, str],
        dry_skip: bool,
        skip_reason: str | None,
    ) -> tuple[int | None, bool]:
        log_path = self.config.runtime_dir / spec.log_name
        started_at = self._now_iso()
        if dry_skip:
            self._record_stage(
                iteration_entry=iteration_entry,
                spec=spec,
                cmd=cmd,
                log_path=log_path,
                tags=tags,
                skipped=True,
                skip_reason=skip_reason
                or ("dry_run" if self.config.dry_run else "skipped"),
                rc=0,
                started_at=started_at,
                ended_at=self._now_iso(),
            )
            self._write_manifest()
            return 0, True

        self._record_stage(
            iteration_entry=iteration_entry,
            spec=spec,
            cmd=cmd,
            log_path=log_path,
            tags=tags,
            skipped=False,
            skip_reason=None,
            rc=None,
            started_at=started_at,
            ended_at=None,
            status="running",
        )
        self._write_manifest()

        ended_at = started_at
        try:
            if spec.port is not None:
                self.dependencies.require_port_free(
                    str(self.config.server_host),
                    int(spec.port),
                    context=str(spec.name),
                )
            free_gb = self._disk_free_gb()
            self._append_disk_budget_line(
                log_path=log_path,
                stage=str(spec.name),
                free_gb=free_gb,
            )
            self._record_disk_check(stage_name=str(spec.name), free_gb=free_gb)
            env = dict(self.stage_env_base)
            rc = self.dependencies.run_cmd_tee(
                cmd,
                log_path=log_path,
                header=f"p3A:{spec.name}",
                timeout_s=float(spec.timeout_s),
                cwd=self.config.repo_root,
                env=env,
            )
            record_tags = dict(tags)
            if spec.port is not None:
                port_released = self._wait_for_port_free(
                    str(self.config.server_host), int(spec.port), timeout_s=15.0
                )
                record_tags["post_stage_port_released"] = (
                    "true" if bool(port_released) else "false"
                )
                if not port_released:
                    record_tags["post_stage_port_release_error"] = (
                        f"{self.config.server_host}:{int(spec.port)}"
                    )
                    if int(rc) == 0:
                        rc = 1
            ended_at = self._now_iso()
            self._record_stage(
                iteration_entry=iteration_entry,
                spec=spec,
                cmd=cmd,
                log_path=log_path,
                tags=record_tags,
                skipped=False,
                skip_reason=None,
                rc=int(rc),
                started_at=started_at,
                ended_at=ended_at,
                status="completed" if int(rc) == 0 else "failed",
                update_existing=True,
            )
            self._write_manifest()
            ok = int(rc) == 0
            if (not ok) and spec.non_fatal:
                return int(rc), True
            return int(rc), ok
        except DiskBudgetError as exc:
            ended_at = self._now_iso()
            self._append_exception_to_log(log_path, exc)
            self._record_stage(
                iteration_entry=iteration_entry,
                spec=spec,
                cmd=cmd,
                log_path=log_path,
                tags=dict(tags, exception=str(exc)),
                skipped=False,
                skip_reason=None,
                rc=1,
                started_at=started_at,
                ended_at=ended_at,
                status="failed",
                update_existing=True,
            )
            self._write_manifest()
            self.disk_budget_abort = exc
            return 1, False
        except Exception as exc:
            ended_at = self._now_iso()
            self._record_stage(
                iteration_entry=iteration_entry,
                spec=spec,
                cmd=cmd,
                log_path=log_path,
                tags=dict(tags, exception=str(exc)),
                skipped=False,
                skip_reason=None,
                rc=1,
                started_at=started_at,
                ended_at=ended_at,
                status="failed",
                update_existing=True,
            )
            self._write_manifest()
            if spec.non_fatal:
                return 1, True
            return 1, False

    def _record_stage(
        self,
        *,
        iteration_entry: dict[str, Any],
        spec: StageSpec,
        cmd: list[str],
        log_path: Path,
        tags: dict[str, str],
        skipped: bool,
        skip_reason: str | None,
        rc: int | None,
        started_at: str,
        ended_at: str | None,
        status: str | None = None,
        update_existing: bool = False,
    ) -> None:
        stage_status = str(
            status
            or (
                "skipped"
                if bool(skipped)
                else (
                    "running"
                    if rc is None
                    else ("completed" if int(rc) == 0 else "failed")
                )
            )
        )
        record = {
            "name": str(spec.name),
            "cmd": [str(x) for x in cmd],
            "log_path": log_path.as_posix(),
            "cwd": self.config.repo_root.as_posix(),
            "timeout_s": float(spec.timeout_s),
            "tags": dict(tags),
            "rc": int(rc) if rc is not None else None,
            "skipped": bool(skipped),
            "skip_reason": str(skip_reason) if skip_reason else None,
            "started_at": str(started_at),
            "ended_at": str(ended_at) if ended_at else None,
            "status": stage_status,
        }
        if update_existing:
            for idx in range(len(iteration_entry["stages"]) - 1, -1, -1):
                existing = iteration_entry["stages"][idx]
                if not isinstance(existing, dict):
                    continue
                if str(existing.get("name")) == str(spec.name) and str(
                    existing.get("started_at")
                ) == str(started_at):
                    iteration_entry["stages"][idx] = record
                    return
        iteration_entry["stages"].append(record)

    def _write_manifest(self) -> None:
        self._refresh_manifest_summary()
        self.dependencies.write_json(self.config.manifest_path, self.manifest)

    def _refresh_manifest_summary(self) -> dict[str, Any]:
        iteration_entries = self.manifest.get("iterations", [])
        failed_iterations: list[dict[str, Any]] = []
        skipped_stage_count = 0
        failed_stage_count = 0
        running_stage_count = 0
        if isinstance(iteration_entries, list):
            for entry in iteration_entries:
                if not isinstance(entry, dict):
                    continue
                stages = entry.get("stages", [])
                if isinstance(stages, list):
                    for stage in stages:
                        if not isinstance(stage, dict):
                            continue
                        if str(stage.get("status", "")).strip().lower() == "running":
                            running_stage_count += 1
                        if bool(stage.get("skipped")):
                            skipped_stage_count += 1
                        rc = stage.get("rc")
                        if (
                            (not bool(stage.get("skipped")))
                            and isinstance(rc, int)
                            and rc != 0
                        ):
                            failed_stage_count += 1
                if bool(entry.get("failed")):
                    failed_iterations.append(
                        {
                            "k": entry.get("k"),
                            "iter_tag": entry.get("iter_tag"),
                            "failure_reason": entry.get("failure_reason"),
                        }
                    )

        error_value = self.manifest.get("error")
        finalized = bool(self.manifest.get("ended_at"))
        if error_value:
            run_status = "BLOCKED"
        elif not finalized and running_stage_count > 0:
            run_status = "IN_PROGRESS"
        elif not finalized:
            run_status = "INCOMPLETE"
        elif failed_iterations:
            run_status = "COMPLETED_WITH_FAILURES"
        else:
            run_status = "COMPLETED"

        summary = {
            "run_status": str(run_status),
            "finalized": bool(finalized),
            "overall_ok": bool(finalized)
            and (not error_value)
            and (len(failed_iterations) == 0),
            "completed_with_failures": bool(finalized) and (len(failed_iterations) > 0),
            "iteration_count": len(iteration_entries)
            if isinstance(iteration_entries, list)
            else 0,
            "failed_iteration_count": len(failed_iterations),
            "failed_stage_count": int(failed_stage_count),
            "running_stage_count": int(running_stage_count),
            "skipped_stage_count": int(skipped_stage_count),
            "failed_iterations": failed_iterations,
        }
        self.manifest["summary"] = summary
        return summary

    def _write_header_log(self) -> None:
        header_log = self.config.runtime_dir / "00_header.log"
        with header_log.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(f"\n===== p3A header {self._now_iso()} =====\n")
            handle.write(f"run_id: {self.config.run_id}\n")
            handle.write(f"repo_root: {self.config.repo_root.as_posix()}\n")
            handle.write(f"git_sha: {self.config.git_sha}\n")
            handle.write(f"git_dirty: {self.config.git_dirty}\n")
            handle.write(f"stage_python: {self.config.stage_python}\n")
            handle.write(f"start_policy_path: {self.config.start_policy_path}\n")
            handle.write(f"env_name: {self.config.env_name}\n")
            handle.write(f"embodiment_tag: {self.config.embodiment_tag}\n")
            handle.write(f"server_host: {self.config.server_host}\n")
            handle.write(f"server_port_base: {self.config.server_port_base}\n")
            handle.write(f"min_free_gb: {self.config.min_free_gb}\n")
            handle.write(f"archive_root: {self.config.archive_root.as_posix()}\n")
            handle.write(
                f"keep_last_n_iters_local: {self.config.keep_last_n_iters_local}\n"
            )
            handle.write(
                "pin_checkpoint_dir: "
                + (
                    ",".join(self.config.pin_checkpoint_dirs)
                    if self.config.pin_checkpoint_dirs
                    else ""
                )
                + "\n"
            )
            handle.write(f"finetune_save_steps: {self.config.finetune_save_steps}\n")
            handle.write(
                f"finetune_save_total_limit: {self.config.finetune_save_total_limit}\n"
            )
            handle.write(f"dry_run: {self.config.dry_run}\n")

    def _ensure_git_clean_if_required(self) -> int | None:
        if self.config.require_git_clean and self.config.git_dirty:
            self.manifest["error"] = (
                "git workspace is dirty but --require-git-clean is set; "
                "pass --no-require-git-clean to override for this run"
            )
            self._write_manifest()
            return 2
        return None

    def _write_repro_snapshot_if_enabled(self) -> int | None:
        if not self.config.write_repro_snapshot:
            return None
        try:
            from work.demo_utils.repro_snapshot import create_repro_snapshot

            self.repro_snapshot_dir = self.config.artifacts_dir / "repro"
            snapshot_paths = create_repro_snapshot(
                self.config.repo_root, self.repro_snapshot_dir
            )
            self.manifest["repro_snapshot_dir"] = self.repro_snapshot_dir.as_posix()
            self.manifest["repro_snapshot_files"] = dict(snapshot_paths)
            self._write_manifest()
            return None
        except Exception as exc:
            self.manifest["error"] = f"repro_snapshot_failed: {exc}"
            self._write_manifest()
            return 3

    def _preflight_ports(self) -> None:
        if self.config.dry_run:
            return
        self.dependencies.require_port_free(
            str(self.config.server_host),
            int(self.config.port_collect),
            context="p3A:preflight:collect",
        )
        self.dependencies.require_port_free(
            str(self.config.server_host),
            int(self.config.port_eval),
            context="p3A:preflight:eval",
        )

    def _finalize_success(self) -> int:
        if not self.manifest.get("ended_at"):
            self.manifest["ended_at"] = self._now_iso()
        self.manifest["final_policy_path"] = str(self.current_policy_path)
        self.manifest["dry_run"] = bool(self.config.dry_run)
        if self.repro_snapshot_dir is not None:
            self.manifest["repro_snapshot_dir"] = self.repro_snapshot_dir.as_posix()
        self._write_manifest()
        print(f"[EVIDENCE] manifest_path={self.config.manifest_path.as_posix()}")
        print(f"[EVIDENCE] final_policy_path={self.current_policy_path}")
        print(f"[EVIDENCE] dry_run={bool(self.config.dry_run)}")
        summary = self._refresh_manifest_summary()
        print(f"[EVIDENCE] run_status={summary['run_status']}")
        print(f"[EVIDENCE] failed_iteration_count={summary['failed_iteration_count']}")
        return 0 if bool(summary.get("overall_ok")) else 1

    def _abort_run(self, *, error: str, rc: int) -> int:
        self.manifest["error"] = str(error)
        self.manifest["ended_at"] = self._now_iso()
        self.manifest["final_policy_path"] = str(self.current_policy_path)
        self.manifest["dry_run"] = bool(self.config.dry_run)
        self._write_manifest()
        print(f"[EVIDENCE] manifest_path={self.config.manifest_path.as_posix()}")
        print(f"[EVIDENCE] final_policy_path={self.current_policy_path}")
        print(f"[EVIDENCE] dry_run={bool(self.config.dry_run)}")
        return int(rc)

    def _handle_disk_budget_abort(self, *, where: str, err: DiskBudgetError) -> int:
        return self._abort_run(error=f"{where}: {err}", rc=11)

    def _mark_running_stages_terminal(self, *, reason: str, rc: int) -> None:
        ended_at = self._now_iso()
        for entry in self.manifest.get("iterations", []):
            if not isinstance(entry, dict):
                continue
            entry_failed = False
            stages = entry.get("stages", [])
            if isinstance(stages, list):
                for stage in stages:
                    if not isinstance(stage, dict):
                        continue
                    if str(stage.get("status", "")).strip().lower() != "running":
                        continue
                    stage["status"] = "failed"
                    stage["rc"] = int(rc)
                    stage["ended_at"] = str(ended_at)
                    tags = stage.get("tags")
                    if isinstance(tags, dict):
                        tags["forced_terminal_reason"] = str(reason)
                    else:
                        stage["tags"] = {"forced_terminal_reason": str(reason)}
                    entry_failed = True
            if entry_failed:
                entry["failed"] = True
                entry["failure_reason"] = str(reason)
                if not entry.get("ended_at"):
                    entry["ended_at"] = str(ended_at)

    def _build_collect_command(
        self,
        *,
        plan: IterationPlan,
        iter_tag_value: str,
        model_path: str,
        server_port: int,
        n_episodes: int,
        policy_prompt_prefix: str,
        enable_mixdone: bool,
        seed_override: int | None = None,
        max_policy_steps_override: int | None = None,
    ) -> list[str]:
        n_eps_cli = int(n_episodes)
        if (
            enable_mixdone
            and self.config.mixdone_short_episodes is not None
            and self.config.mixdone_long_episodes is not None
        ):
            n_eps_cli = int(self.config.mixdone_short_episodes) + int(
                self.config.mixdone_long_episodes
            )
        max_policy_steps_cli = (
            int(self.config.collect_max_policy_steps)
            if max_policy_steps_override is None
            else int(max_policy_steps_override)
        )
        seed_cli = (
            int(seed_override)
            if seed_override is not None
            else (
                int(plan.seed_k)
                if policy_prompt_prefix == ""
                else int(plan.seed_eval_k)
            )
        )
        cmd = [
            str(self.config.stage_python),
            str(
                self.config.repo_root
                / "work/recap/scripts/31_recap_collect_rollouts.py"
            ),
            "--env-name",
            str(self.config.env_name),
            "--model-path",
            str(model_path),
            "--embodiment-tag",
            str(self.config.embodiment_tag),
            "--server-host",
            str(self.config.server_host),
            "--server-port",
            str(int(server_port)),
            "--iter-tag",
            str(iter_tag_value),
            "--seed",
            str(int(seed_cli)),
            "--n-episodes",
            str(int(n_eps_cli)),
            "--max-policy-steps",
            str(int(max_policy_steps_cli)),
            "--n-action-steps-config",
            str(int(self.config.n_action_steps_config)),
            "--mujoco-gl",
            str(self.config.mujoco_gl),
            "--offscreen",
            "--kill-server-on-exit",
            "--total-timeout-s",
            str(
                float(
                    self.config.timeout_collect_s
                    if policy_prompt_prefix == ""
                    else self.config.timeout_eval_s
                )
            ),
        ]
        if policy_prompt_prefix:
            cmd.extend(["--policy-prompt-prefix", str(policy_prompt_prefix)])
        if enable_mixdone:
            cmd.append("--mixdone")
            if self.config.mixdone_short_episodes is not None:
                cmd.extend(
                    [
                        "--mixdone-short-episodes",
                        str(int(self.config.mixdone_short_episodes)),
                    ]
                )
            if self.config.mixdone_long_episodes is not None:
                cmd.extend(
                    [
                        "--mixdone-long-episodes",
                        str(int(self.config.mixdone_long_episodes)),
                    ]
                )
            cmd.extend(
                [
                    "--mixdone-short-max-episode-steps",
                    str(int(self.config.mixdone_short_max_episode_steps)),
                    "--mixdone-long-max-episode-steps",
                    str(int(self.config.mixdone_long_max_episode_steps)),
                    "--mixdone-long-seed-offset",
                    str(int(self.config.mixdone_long_seed_offset)),
                ]
            )
        else:
            cmd.append("--no-mixdone")
        return cmd

    def _build_critic_command(
        self, *, plan: IterationPlan, critic_iter_tags: list[str]
    ) -> list[str]:
        return [
            str(self.config.stage_python),
            str(
                self.config.repo_root
                / "agent/archive/recap_legacy_state_only_critic/40_recap_train_critic_dist_bins.py"
            ),
            "--iter-tags",
            ",".join([str(tag) for tag in critic_iter_tags]),
            "--critic-tag",
            str(plan.critic_tag),
            "--bins",
            str(int(self.config.critic_bins)),
            "--include-t",
            "--seed",
            str(int(self.config.seed)),
            "--max-epochs",
            str(int(self.config.critic_max_epochs)),
            "--patience",
            str(int(self.config.critic_patience)),
            "--lr",
            str(float(self.config.critic_lr)),
            "--val-ratio",
            str(float(self.config.critic_val_ratio)),
            "--device",
            str(self.config.critic_device),
        ]

    def _build_label_command(self, *, plan: IterationPlan) -> list[str]:
        return [
            str(self.config.stage_python),
            str(self.config.repo_root / "work/recap/scripts/32_recap_label_dataset.py"),
            "--iter-tag",
            str(plan.iter_tag),
            "--value-source",
            "critic",
            "--critic-dir",
            str(plan.critic_dir.as_posix()),
            "--total-timeout-s",
            str(float(self.config.timeout_label_s)),
        ]

    def _build_export_command(self, *, plan: IterationPlan) -> list[str]:
        cmd = [
            str(self.config.stage_python),
            str(
                self.config.repo_root
                / "work/recap/scripts/39_recap_export_lerobot_v2_with_video.py"
            ),
            "--iter-tag",
            str(plan.iter_tag),
            "--max-episodes",
            str(int(self.config.collect_episodes)),
            "--total-timeout-s",
            str(float(self.config.timeout_export_s)),
            "--require-ffmpeg",
        ]
        cmd.append(
            "--dual-task-text"
            if bool(self.config.dual_task_text)
            else "--no-dual-task-text"
        )
        return cmd

    def _build_finetune_command(self, *, plan: IterationPlan) -> list[str]:
        plan.finetune_out_dir.mkdir(parents=True, exist_ok=True)
        return [
            str(self.config.stage_python),
            str(
                self.config.repo_root
                / "submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py"
            ),
            "--base-model-path",
            str(self.config.start_policy_path),
            "--dataset-path",
            str(plan.lerobot_dataset_dir.as_posix()),
            "--embodiment-tag",
            str(self.config.embodiment_tag),
            (
                "--tune-projector"
                if self.config.finetune_tune_projector
                else "--no-tune-projector"
            ),
            (
                "--tune-diffusion-model"
                if self.config.finetune_tune_diffusion_model
                else "--no-tune-diffusion-model"
            ),
            "--no-use-wandb",
            "--output-dir",
            str(plan.finetune_out_dir.as_posix()),
            "--max-steps",
            str(int(self.config.finetune_max_steps)),
            "--save-steps",
            str(int(self.config.finetune_save_steps)),
            "--save-total-limit",
            str(int(self.config.finetune_save_total_limit)),
            "--global-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--dataloader-num-workers",
            "0",
        ]

    def _write_continuous_advantage_contract(
        self, *, dataset_dir: Path, iter_tag: str, critic_dir: Path
    ) -> Path:
        labels_path = dataset_dir / "m2_labels" / "labels.jsonl"
        if not labels_path.is_file():
            raise FileNotFoundError(
                f"labels.jsonl not found for advantage contract: {labels_path.as_posix()}"
            )
        label_rows: list[dict[str, Any]] = []
        with labels_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                raw = str(line).strip()
                if not raw:
                    continue
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"labels.jsonl line {idx} must be JSON object: {labels_path.as_posix()}"
                    )
                label_rows.append(record)
        if not label_rows:
            raise ValueError(
                f"labels.jsonl contains no records for advantage contract: {labels_path.as_posix()}"
            )

        advantage_mod = importlib.import_module("work.recap.advantage")
        compute_sign_scales = getattr(
            advantage_mod, "compute_sign_aware_advantage_scales"
        )
        build_contract = getattr(advantage_mod, "build_advantage_contract_metadata")
        advantage_values = [float(record["advantage_A"]) for record in label_rows]
        sign_scale_summary = compute_sign_scales(
            advantage_values,
            context=f"p3A.{iter_tag}.continuous_advantage_contract",
        )
        positive_scale = sign_scale_summary.get("positive_scale")
        negative_scale_abs = sign_scale_summary.get("negative_scale_abs")
        if positive_scale is None or negative_scale_abs is None:
            raise ValueError(
                "continuous advantage contract requires both positive and negative scales"
            )
        contract = build_contract(
            source_iter_tag=str(iter_tag),
            n_samples=len(label_rows),
            positive_scale=float(positive_scale),
            negative_scale_abs=float(negative_scale_abs),
            critic_dir=critic_dir.as_posix(),
            critic_include_t=True,
            advantage_stats={"value_source": "critic"},
            sign_scale_summary=dict(sign_scale_summary),
        )
        contract_path = dataset_dir / "m2_labels" / "continuous_advantage_contract.json"
        self.dependencies.write_json(contract_path, contract)
        return contract_path

    def _prune_checkpoints_under(
        self,
        *,
        output_dir: Path,
        keep_dir: Path,
        stage_log_path: Path,
        iteration_entry: dict[str, Any],
    ) -> None:
        if self.config.dry_run:
            return
        keep_r = keep_dir.resolve()
        to_delete: list[Path] = []
        for path in sorted(output_dir.glob("checkpoint-*")):
            if not path.is_dir():
                continue
            try:
                if path.resolve() == keep_r:
                    continue
            except Exception:
                if path.as_posix() == keep_dir.as_posix():
                    continue
            to_delete.append(path)

        if not to_delete:
            return

        deleted_paths: list[str] = []
        for path in to_delete:
            shutil.rmtree(path)
            deleted_paths.append(path.as_posix())

        self.manifest.setdefault("checkpoint_prune", []).extend(deleted_paths)
        iteration_entry.setdefault("checkpoint_prune", []).extend(deleted_paths)
        self._write_manifest()
        try:
            with stage_log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(f"\n[INFO] pruned_checkpoints_n={len(deleted_paths)}\n")
                for deleted_path in deleted_paths:
                    handle.write(f"[INFO] pruned_checkpoint: {deleted_path}\n")
        except Exception:
            pass

    def _archive_checkpoint_dir(self, *, iter_tag_value: str) -> None:
        if self.config.dry_run:
            return
        src = (
            self.config.repo_root
            / "agent"
            / "artifacts"
            / "checkpoints"
            / str(iter_tag_value)
        )
        if not src.is_dir():
            return
        current_policy_path = Path(str(self.current_policy_path))
        if self._path_is_under(current_policy_path, src):
            with (self.config.runtime_dir / "00_header.log").open(
                "a", encoding="utf-8", buffering=1
            ) as handle:
                handle.write(
                    f"[WARN] skip_archive_active_policy: iter_tag={iter_tag_value} current_policy_path={self.current_policy_path}\n"
                )
            return
        dst = (
            self.config.archive_root
            / "checkpoints"
            / self.config.run_id
            / str(iter_tag_value)
        )
        if dst.exists():
            with (self.config.runtime_dir / "00_header.log").open(
                "a", encoding="utf-8", buffering=1
            ) as handle:
                handle.write(
                    f"[WARN] skip_archive_dst_exists: src={src.as_posix()} dst={dst.as_posix()}\n"
                )
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.config.runtime_dir / f"archive_{iter_tag_value}.log"
        free_gb = self._disk_free_gb()
        self._append_disk_budget_line(
            log_path=log_path,
            stage=f"archive:{iter_tag_value}",
            free_gb=free_gb,
        )
        self._record_disk_check(stage_name=f"archive:{iter_tag_value}", free_gb=free_gb)
        rsync = shutil.which("rsync")
        verified = False
        if rsync:
            rc = self.dependencies.run_cmd_tee(
                [
                    str(rsync),
                    "-a",
                    f"{src.as_posix()}/",
                    f"{dst.as_posix()}/",
                ],
                log_path=log_path,
                header=f"p3A:archive:{iter_tag_value}",
                timeout_s=float(DEFAULT_TIMEOUT_ARCHIVE_S),
                cwd=self.config.repo_root,
                env=dict(self.stage_env_base),
            )
            if int(rc) != 0:
                raise RuntimeError(
                    f"archive_rsync_failed: rc={rc} src={src.as_posix()} dst={dst.as_posix()}"
                )
            verified = dst.is_dir() and any(dst.iterdir())
        else:
            shutil.copytree(src, dst)
            verified = dst.is_dir() and any(dst.iterdir())

        self.manifest.setdefault("archive_moves", []).append(
            {
                "src": src.as_posix(),
                "dst": dst.as_posix(),
                "verified": bool(verified),
                "bytes": None,
            }
        )
        self._write_manifest()
        if not verified:
            raise RuntimeError(
                f"archive_verify_failed: src={src.as_posix()} dst={dst.as_posix()}"
            )
        shutil.rmtree(src)
        self._write_manifest()

    def _disk_free_gb(self) -> float:
        _total_b, _used_b, free_b = shutil.disk_usage(str(self.config.repo_root))
        return float(free_b) / float(1024**3)

    def _append_disk_budget_line(
        self, *, log_path: Path, stage: str, free_gb: float
    ) -> None:
        if self.config.dry_run:
            return
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(
                    f"[INFO] free_gb={float(free_gb):.3f} min_free_gb={float(self.config.min_free_gb):.3f} repo_root={self.config.repo_root.as_posix()} stage={str(stage)}\n"
                )
        except Exception:
            pass

    def _record_disk_check(
        self, *, stage_name: str, free_gb: float | None = None
    ) -> dict[str, Any]:
        if free_gb is None:
            free_gb = self._disk_free_gb()
        record = {
            "stage": str(stage_name),
            "at": self._now_iso(),
            "free_gb": float(round(float(free_gb), 3)),
            "min_free_gb": float(self.config.min_free_gb),
        }
        self.manifest.setdefault("disk_checks", []).append(record)
        self._write_manifest()
        if float(free_gb) < float(self.config.min_free_gb):
            raise DiskBudgetError(
                f"disk_free_below_min: stage={stage_name} free_gb={float(free_gb):.2f} < min_free_gb={float(self.config.min_free_gb):.2f} (repo_root={self.config.repo_root.as_posix()})"
            )
        return record

    def _append_exception_to_log(self, log_path: Path, error: Exception) -> None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(f"\n[ERROR] {error}\n")
        except Exception:
            pass

    def _is_tcp_port_listening(
        self, host: str, port: int, timeout_s: float = 0.2
    ) -> bool:
        host = str(host or "").strip() or "127.0.0.1"
        try:
            with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
                return True
        except OSError:
            return False

    def _wait_for_port_free(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = 15.0,
        poll_interval_s: float = 0.25,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() <= deadline:
            if not self._is_tcp_port_listening(str(host), int(port)):
                return True
            time.sleep(max(0.05, float(poll_interval_s)))
        return not self._is_tcp_port_listening(str(host), int(port))

    def _path_is_under(self, child: Path, parent: Path) -> bool:
        try:
            child_r = child.resolve()
            parent_r = parent.resolve()
        except Exception:
            child_r = child
            parent_r = parent
        try:
            common = os.path.commonpath([str(child_r), str(parent_r)])
        except Exception:
            return False
        return common == str(parent_r)

    def _eval_max_policy_steps(self) -> int:
        return (
            int(self.config.collect_max_policy_steps)
            if self.config.eval_max_policy_steps is None
            else int(self.config.eval_max_policy_steps)
        )

    def _now_iso(self) -> str:
        return self.dependencies.now_iso()

    @staticmethod
    def _validate_tag(tag: str, *, name: str) -> str:
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"{name} must be a non-empty str, got {tag!r}")
        if "/" in tag or "\\" in tag:
            raise ValueError(f"{name} must not contain path separators: {tag!r}")
        if tag in (".", ".."):
            raise ValueError(f"{name} must not be '.' or '..': {tag!r}")
        return str(tag)
