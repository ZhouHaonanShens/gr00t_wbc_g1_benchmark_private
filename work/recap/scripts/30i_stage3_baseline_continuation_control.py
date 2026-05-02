#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.demo_utils import paths as demo_paths
from work.recap.finetune_full import COMPARABILITY_MANIFEST_FILENAME
from work.recap.finetune_full import emit_continuation_formal_lane_comparability_manifest
from work.recap.scope_experiment import build_v2_train_scope_shim_metadata
from work.recap.scripts.state_conditioned_common import write_json
from work.recap.train_scope_audit import add_scope_flag_argument
from work.recap.train_scope_audit import build_scope_summary


DEFAULT_CONTINUATION_CHECKPOINT = Path(
    "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/checkpoint-200"
)
DEFAULT_CONDITIONED_REFERENCE_SUMMARY = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "t13_advantage_full_update_1gpu/formal_run/formal_run_skipped.json"
)
DEFAULT_DATASET_PATH = Path("agent/artifacts/lerobot_datasets/recap_stage3_iter_002")
DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "t13_continuation_full_update_1gpu/formal_run"
)
DEFAULT_RUNTIME_LOG_DIR = Path(
    "agent/runtime_logs/recap_full_update_first/task12_continuation_formal_run"
)
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "control_summary.json"
DEFAULT_TRAINING_LAUNCHER = Path("work/recap/launch_finetune_use_ddp.py")
DEFAULT_MAX_STEPS = 200
DEFAULT_SAVE_STEPS = 50
DEFAULT_SAVE_TOTAL_LIMIT = 1
DEFAULT_GLOBAL_BATCH_SIZE = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_DATALOADER_NUM_WORKERS = 0
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_RECAP_TRAIN_SCOPE = "full_policy"
DEFAULT_NUM_GPUS = 1
DEFAULT_VISIBLE_DEVICE = "1"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_USE_WANDB = False
DEFAULT_TUNE_PROJECTOR = True
DEFAULT_TUNE_DIFFUSION_MODEL = True
SCHEMA_VERSION = "stage3_baseline_continuation_control_v1"
ARTIFACT_KIND = "stage3_baseline_continuation_control"
WRAPPER_NAME = "30i_stage3_baseline_continuation_control.py"


def _resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = demo_paths.remap_legacy_project_root(path)
    return path.resolve()


def _repo_relative(path: Path | str) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonify_cmd(cmd: list[str]) -> list[str]:
    return [str(part) for part in cmd]


def _selected_checkpoint_asset(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return None
    for candidate in (
        checkpoint_dir / "model.safetensors.index.json",
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "pytorch_model.bin.index.json",
        checkpoint_dir / "pytorch_model.bin",
    ):
        if candidate.is_file():
            return candidate
    return None


def _latest_checkpoint(output_dir: Path) -> Path | None:
    latest: tuple[int, float, Path] | None = None
    for path in sorted(output_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        suffix = path.name.split("checkpoint-", 1)[-1]
        if not suffix.isdigit():
            continue
        candidate = (int(suffix), float(path.stat().st_mtime), path)
        if latest is None or candidate[:2] > latest[:2]:
            latest = candidate
    return None if latest is None else latest[2]


def _effective_batch_geometry(
    *,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int,
) -> dict[str, int]:
    divisor = int(num_gpus) * int(gradient_accumulation_steps)
    if divisor <= 0:
        raise ValueError("num_gpus * gradient_accumulation_steps must be positive")
    if int(global_batch_size) % divisor != 0:
        raise ValueError(
            "global_batch_size must be evenly divisible by num_gpus * gradient_accumulation_steps"
        )
    per_device_batch_size = int(global_batch_size) // divisor
    return {
        "per_device_batch_size": int(per_device_batch_size),
        "effective_update_batch": int(per_device_batch_size * divisor),
    }


def _resolve_two_layer_python_contract(
    delegate_runtime_python_flag: str,
) -> dict[str, str | bool | None]:
    contract = demo_paths.load_stage3_training_python_contract(REPO_ROOT)
    launcher_python = str(demo_paths.current_python_abspath_preserve_symlink())
    orchestrator_python = str(contract["orchestrator_python"])
    if not demo_paths.same_abspath_preserve_symlink(launcher_python, orchestrator_python):
        raise ValueError(
            "two-layer interpreter contract violated: "
            f"expected orchestrator_python={orchestrator_python}, got {launcher_python}"
        )

    requested_delegate_runtime = str(delegate_runtime_python_flag or "").strip()
    delegate_runtime_python = str(contract["delegate_runtime_python"])
    if requested_delegate_runtime and not demo_paths.same_abspath_preserve_symlink(
        requested_delegate_runtime,
        delegate_runtime_python,
    ):
        raise ValueError(
            "two-layer interpreter contract violated: "
            f"expected delegate_runtime_python={delegate_runtime_python}, got {requested_delegate_runtime}"
        )

    return {
        "contract_manifest_path": str(contract["manifest_path"]),
        "launcher_python": launcher_python,
        "orchestrator_python": orchestrator_python,
        "delegate_runtime_python": delegate_runtime_python,
        "delegate_runtime_python_requested": (
            None if not requested_delegate_runtime else requested_delegate_runtime
        ),
        "orchestrator_python_exists": bool(Path(orchestrator_python).is_file()),
        "delegate_runtime_python_exists": bool(Path(delegate_runtime_python).is_file()),
    }


def _run_with_tee(
    cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stderr.write(line)
                log_fp.write(line)
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
            return 130
    return int(proc.returncode)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_flag_value(argv: object, *, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    values = [str(item) for item in argv]
    for index, token in enumerate(values):
        if token == flag:
            if index + 1 < len(values):
                return values[index + 1]
            return None
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _load_conditioned_reference(summary_path: Path) -> dict[str, Any]:
    payload = _read_json_if_exists(summary_path)
    if payload is None:
        return {
            "conditioned_reference_summary_path": str(summary_path),
            "conditioned_reference_exists": False,
            "same_dataset_path_as_conditioned": None,
            "same_max_steps_as_conditioned": None,
            "same_save_total_limit_as_conditioned": None,
            "same_batch_geometry_as_conditioned": None,
        }
    delegate_cmd = payload.get("delegate_cmd")
    conditioned_global_batch_size = payload.get("global_batch_size")
    if conditioned_global_batch_size is None:
        conditioned_global_batch_size = _read_flag_value(
            delegate_cmd, flag="--global-batch-size"
        )
    conditioned_gradient_accumulation_steps = payload.get(
        "gradient_accumulation_steps"
    )
    if conditioned_gradient_accumulation_steps is None:
        conditioned_gradient_accumulation_steps = _read_flag_value(
            delegate_cmd, flag="--gradient-accumulation-steps"
        )
    return {
        "conditioned_reference_summary_path": str(summary_path),
        "conditioned_reference_exists": True,
        "same_dataset_path_as_conditioned": None,
        "same_max_steps_as_conditioned": None,
        "same_save_total_limit_as_conditioned": None,
        "same_batch_geometry_as_conditioned": None,
        "conditioned_reference_dataset_path": str(payload.get("dataset_path") or ""),
        "conditioned_reference_max_steps": payload.get("max_steps"),
        "conditioned_reference_save_total_limit": payload.get("save_total_limit"),
        "conditioned_reference_global_batch_size": conditioned_global_batch_size,
        "conditioned_reference_gradient_accumulation_steps": conditioned_gradient_accumulation_steps,
        "conditioned_reference_numeric_advantage_path_active": payload.get(
            "numeric_advantage_path_active"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=WRAPPER_NAME,
        description=(
            "Thin wrapper for the stage3 single-GPU baseline continuation control lane. "
            "It warm-starts from the already-trained baseline checkpoint-200, keeps the same "
            "formal budget/data/batch geometry as the conditioned continuation run, and makes "
            "the numeric advantage path explicitly inactive."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="LeRobot dataset directory reused by the continuation control lane.",
    )
    parser.add_argument(
        "--continuation-checkpoint",
        type=Path,
        default=DEFAULT_CONTINUATION_CHECKPOINT,
        help="Baseline checkpoint-200 used as the warm-start source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Formal control output namespace.",
    )
    parser.add_argument(
        "--runtime-log-dir",
        type=Path,
        default=DEFAULT_RUNTIME_LOG_DIR,
        help="Runtime log namespace for the control lane.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Machine-readable control summary artifact.",
    )
    parser.add_argument(
        "--conditioned-reference-summary",
        type=Path,
        default=DEFAULT_CONDITIONED_REFERENCE_SUMMARY,
        help="Optional conditioned run summary used to prove matched budget/data geometry.",
    )
    parser.add_argument(
        "--training-launcher",
        type=Path,
        default=DEFAULT_TRAINING_LAUNCHER,
        help="Repo-local training launcher used for the unconditioned control lane.",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument(
        "--save-total-limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT
    )
    parser.add_argument(
        "--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=DEFAULT_DATALOADER_NUM_WORKERS,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    add_scope_flag_argument(
        parser,
        default=DEFAULT_RECAP_TRAIN_SCOPE,
        help_text=(
            "Public full-update-first scope taxonomy. This control wrapper records the "
            "requested v2 scope without renaming legacy S1/S2/S3 semantics."
        ),
    )
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    parser.add_argument(
        "--visible-device",
        type=str,
        default=DEFAULT_VISIBLE_DEVICE,
        help="CUDA_VISIBLE_DEVICES value for the single-GPU control lane.",
    )
    parser.add_argument(
        "--embodiment-tag", type=str, default=DEFAULT_EMBODIMENT_TAG
    )
    parser.add_argument(
        "--python",
        type=str,
        default="",
        help="Optional explicit delegate runtime python. Must match the stage3 contract.",
    )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_WANDB,
    )
    parser.add_argument(
        "--tune-projector",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TUNE_PROJECTOR,
    )
    parser.add_argument(
        "--tune-diffusion-model",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TUNE_DIFFUSION_MODEL,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the command and summary payload without launching training.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_steps) <= 0:
        raise ValueError(f"--max-steps must be > 0, got {args.max_steps!r}")
    if int(args.save_steps) <= 0:
        raise ValueError(f"--save-steps must be > 0, got {args.save_steps!r}")
    if int(args.save_total_limit) != int(DEFAULT_SAVE_TOTAL_LIMIT):
        raise ValueError(
            "This wrapper enforces single-checkpoint retention. "
            f"Expected --save-total-limit={DEFAULT_SAVE_TOTAL_LIMIT}, got {args.save_total_limit}."
        )
    if int(args.num_gpus) != int(DEFAULT_NUM_GPUS):
        raise ValueError(
            f"This control wrapper only supports --num-gpus={DEFAULT_NUM_GPUS}, got {args.num_gpus}."
        )
    if str(args.visible_device).strip() != DEFAULT_VISIBLE_DEVICE:
        raise ValueError(
            f"single_gpu_v1 control lane requires --visible-device {DEFAULT_VISIBLE_DEVICE}, got {args.visible_device!r}"
        )
    _ = _effective_batch_geometry(
        global_batch_size=int(args.global_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        num_gpus=int(args.num_gpus),
    )


def _build_training_cmd(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    continuation_checkpoint: Path,
    output_dir: Path,
    delegate_runtime_python: Path,
    training_launcher: Path,
) -> list[str]:
    cmd = [
        str(delegate_runtime_python),
        str(training_launcher),
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--base-model-path",
        str(continuation_checkpoint),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--max-steps",
        str(int(args.max_steps)),
        "--save-steps",
        str(int(args.save_steps)),
        "--save-total-limit",
        str(int(args.save_total_limit)),
        "--global-batch-size",
        str(int(args.global_batch_size)),
        "--gradient-accumulation-steps",
        str(int(args.gradient_accumulation_steps)),
        "--dataloader-num-workers",
        str(int(args.dataloader_num_workers)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--num-gpus",
        str(int(args.num_gpus)),
        "--tune-projector" if bool(args.tune_projector) else "--no-tune-projector",
        (
            "--tune-diffusion-model"
            if bool(args.tune_diffusion_model)
            else "--no-tune-diffusion-model"
        ),
        "--use-wandb" if bool(args.use_wandb) else "--no-use-wandb",
    ]
    return cmd


def build_continuation_comparability_manifest(
    *,
    output_dir: str | Path,
    continuation_checkpoint: str | Path,
    dataset_path: str | Path,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int,
    train_scope_requested: str,
    train_scope_effective: str | None = None,
    launch_family: str = "single_gpu_v1",
    seed_bundle_path: str | Path | None = None,
    policy_route: object | None = None,
    policy_indicator_mode: object | None = None,
) -> dict[str, Any]:
    return emit_continuation_formal_lane_comparability_manifest(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        warm_start_checkpoint=continuation_checkpoint,
        global_batch_size=int(global_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        num_gpus=int(num_gpus),
        dataset_path=dataset_path,
        launch_family=launch_family,
        train_scope_requested=train_scope_requested,
        train_scope_effective=(
            train_scope_requested
            if train_scope_effective is None
            else train_scope_effective
        ),
        seed_bundle_path=seed_bundle_path,
        policy_route=policy_route,
        policy_indicator_mode=policy_indicator_mode,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    dataset_path = _resolve_path(args.dataset_path)
    continuation_checkpoint = _resolve_path(args.continuation_checkpoint)
    output_dir = _resolve_path(args.output_dir)
    runtime_log_dir = _resolve_path(args.runtime_log_dir)
    summary_json = _resolve_path(args.summary_json)
    conditioned_reference_summary = _resolve_path(args.conditioned_reference_summary)
    training_launcher = _resolve_path(args.training_launcher)
    python_contract = _resolve_two_layer_python_contract(str(args.python).strip())
    delegate_runtime_python = Path(str(python_contract["delegate_runtime_python"]))
    geometry = _effective_batch_geometry(
        global_batch_size=int(args.global_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        num_gpus=int(args.num_gpus),
    )
    conditioned_reference = _load_conditioned_reference(conditioned_reference_summary)
    scope_summary = build_scope_summary(
        args.recap_train_scope,
        legacy_scope_bridge=build_v2_train_scope_shim_metadata(args.recap_train_scope),
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "wrapper": WRAPPER_NAME,
        "wrapper_status": "ok",
        "control_lane_kind": "baseline_continuation_control",
        "formal_mode": True,
        "numeric_advantage_path_active": False,
        "condition_focused_continuation": False,
        "training_semantics": "unconditioned_baseline_checkpoint_warm_start",
        "continuation_checkpoint_path": str(continuation_checkpoint),
        "continuation_checkpoint_exists": bool(continuation_checkpoint.is_dir()),
        "dataset_path": str(dataset_path),
        "dataset_exists": bool(dataset_path.is_dir()),
        "output_dir": str(output_dir),
        "runtime_log_dir": str(runtime_log_dir),
        "summary_json": str(summary_json),
        "training_launcher": str(training_launcher),
        "training_launcher_exists": bool(training_launcher.is_file()),
        "contract_manifest_path": str(python_contract["contract_manifest_path"]),
        "launcher_python": str(python_contract["launcher_python"]),
        "orchestrator_python": str(python_contract["orchestrator_python"]),
        "delegate_runtime_python": str(delegate_runtime_python),
        "delegate_runtime_python_requested": python_contract[
            "delegate_runtime_python_requested"
        ],
        "num_gpus": int(args.num_gpus),
        "use_ddp": False,
        "live_launch_family": "single_gpu_v1",
        "visible_devices_policy": "single_gpu_gpu1_only",
        "cuda_visible_devices": str(args.visible_device),
        "embodiment_tag": str(args.embodiment_tag),
        "max_steps": int(args.max_steps),
        "save_steps": int(args.save_steps),
        "save_total_limit": int(args.save_total_limit),
        "global_batch_size": int(args.global_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "learning_rate": float(args.learning_rate),
        "train_scope_requested": str(scope_summary["train_scope_requested"]),
        "scope_faithfulness": str(scope_summary["scope_faithfulness"]),
        "method_faithfulness": dict(scope_summary["method_faithfulness"]),
        "train_scope_taxonomy": dict(scope_summary),
        "tune_projector": bool(args.tune_projector),
        "tune_diffusion_model": bool(args.tune_diffusion_model),
        "use_wandb": bool(args.use_wandb),
        "per_device_batch_size": int(geometry["per_device_batch_size"]),
        "effective_update_batch": int(geometry["effective_update_batch"]),
        "training_budget_matches_task12": True,
        "data_path_matches_task12": True,
        "semantic_delta_vs_conditioned": {
            "numeric_advantage_path_active": {
                "conditioned": True,
                "control": False,
            },
            "model_path": {
                "conditioned": "work.recap.model.GR00TRecapModel + numeric-adv monkeypatch",
                "control": "standard launch_finetune_use_ddp baseline path",
            },
        },
        "delegate_cmd": None,
        "delegate_cmd_shell": None,
        "runtime_log_path": None,
        "upstream_returncode": None,
        "selected_checkpoint_path": None,
        "selected_checkpoint_exists": False,
        "selected_checkpoint_asset_path": None,
        "comparability_manifest_path": str(output_dir / COMPARABILITY_MANIFEST_FILENAME),
        "comparability_manifest_status": None,
        "comparability_manifest_blocker_code": None,
        "error": None,
        **conditioned_reference,
    }

    conditioned_dataset = payload.get("conditioned_reference_dataset_path")
    if isinstance(conditioned_dataset, str) and conditioned_dataset.strip():
        payload["same_dataset_path_as_conditioned"] = (
            _resolve_path(conditioned_dataset) == dataset_path
        )
    conditioned_steps = payload.get("conditioned_reference_max_steps")
    if isinstance(conditioned_steps, int):
        payload["same_max_steps_as_conditioned"] = conditioned_steps == int(args.max_steps)
    conditioned_save_total_limit = payload.get("conditioned_reference_save_total_limit")
    if isinstance(conditioned_save_total_limit, int):
        payload["same_save_total_limit_as_conditioned"] = conditioned_save_total_limit == int(
            args.save_total_limit
        )
    conditioned_gbs = payload.get("conditioned_reference_global_batch_size")
    conditioned_ga = payload.get("conditioned_reference_gradient_accumulation_steps")
    conditioned_gbs_int = None
    conditioned_ga_int = None
    try:
        if conditioned_gbs is not None:
            conditioned_gbs_int = int(conditioned_gbs)
        if conditioned_ga is not None:
            conditioned_ga_int = int(conditioned_ga)
    except (TypeError, ValueError):
        conditioned_gbs_int = None
        conditioned_ga_int = None
    if conditioned_gbs_int is not None and conditioned_ga_int is not None:
        payload["same_batch_geometry_as_conditioned"] = (
            conditioned_gbs_int == int(args.global_batch_size)
            and conditioned_ga_int == int(args.gradient_accumulation_steps)
        )

    try:
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        if not continuation_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Continuation checkpoint not found: {continuation_checkpoint}"
            )
        if _selected_checkpoint_asset(continuation_checkpoint) is None:
            raise FileNotFoundError(
                f"Continuation checkpoint missing model weights: {continuation_checkpoint}"
            )
        if not delegate_runtime_python.is_file():
            raise FileNotFoundError(
                f"Delegate runtime python not found: {delegate_runtime_python}"
            )
        if not training_launcher.is_file():
            raise FileNotFoundError(f"Training launcher not found: {training_launcher}")

        output_dir.mkdir(parents=True, exist_ok=True)
        runtime_log_dir.mkdir(parents=True, exist_ok=True)
        comparability_manifest = build_continuation_comparability_manifest(
            output_dir=output_dir,
            continuation_checkpoint=continuation_checkpoint,
            global_batch_size=int(args.global_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            num_gpus=int(args.num_gpus),
            dataset_path=dataset_path,
            train_scope_requested=payload["train_scope_requested"],
            train_scope_effective=payload["train_scope_requested"],
            launch_family=payload["live_launch_family"],
        )
        payload["comparability_manifest_status"] = comparability_manifest["status"]
        payload["comparability_manifest_blocker_code"] = comparability_manifest.get(
            "blocker_code"
        )
        runtime_log_path = runtime_log_dir / f"30i_stage3_baseline_continuation_control_{_timestamp()}.log"
        payload["runtime_log_path"] = str(runtime_log_path)

        cmd = _build_training_cmd(
            args=args,
            dataset_path=dataset_path,
            continuation_checkpoint=continuation_checkpoint,
            output_dir=output_dir,
            delegate_runtime_python=delegate_runtime_python,
            training_launcher=training_launcher,
        )
        payload["delegate_cmd"] = _jsonify_cmd(cmd)
        payload["delegate_cmd_shell"] = shlex.join(cmd)

        if bool(args.dry_run):
            write_json(summary_json, payload)
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
            return 0

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.visible_device)
        gr00t_root = str((REPO_ROOT / "submodules" / "Isaac-GR00T").resolve())
        old_pythonpath = str(env.get("PYTHONPATH", "")).strip()
        env["PYTHONPATH"] = gr00t_root + (os.pathsep + old_pythonpath if old_pythonpath else "")

        rc = _run_with_tee(cmd, cwd=REPO_ROOT, env=env, log_path=runtime_log_path)
        payload["upstream_returncode"] = int(rc)
        if rc != 0:
            payload["wrapper_status"] = "blocked"
            payload["error"] = f"delegated_training_failed: returncode={rc}"

        selected_checkpoint = _latest_checkpoint(output_dir)
        selected_checkpoint_asset = _selected_checkpoint_asset(selected_checkpoint)
        payload["selected_checkpoint_path"] = (
            str(selected_checkpoint)
            if selected_checkpoint is not None and selected_checkpoint_asset is not None
            else None
        )
        payload["selected_checkpoint_exists"] = bool(
            selected_checkpoint_asset is not None
        )
        payload["selected_checkpoint_asset_path"] = (
            str(selected_checkpoint_asset)
            if selected_checkpoint_asset is not None
            else None
        )
        if not bool(payload["selected_checkpoint_exists"]) and payload["error"] is None:
            payload["wrapper_status"] = "blocked"
            payload["error"] = "missing_checkpoint_asset"

        write_json(summary_json, payload)
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if payload["wrapper_status"] == "ok" else 1
    except Exception as exc:
        payload["wrapper_status"] = "blocked"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        write_json(summary_json, payload)
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
