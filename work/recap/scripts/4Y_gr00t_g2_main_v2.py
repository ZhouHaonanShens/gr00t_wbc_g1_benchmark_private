#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.demo_utils import paths as demo_paths
from work.recap import gr00t_main_recap


DEFAULT_WBC_PY_REL = ".envs/wbc/bin/python"
DEFAULT_VISIBLE_DEVICE = "1"
DEFAULT_INDICATOR_DROPOUT_P = 0.3
DEFAULT_TRAIN_SCOPE = "strict_full"
DEFAULT_FULL_STEPS = 2200
DEFAULT_PROBE_STEPS = 50
DEFAULT_SMOKE_STEPS = 1
DEFAULT_TIMEOUT_S = {
    "preflight": 120,
    "smoke": 3600,
    "probe": 7200,
    "full": 72000,
}


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return demo_paths.abspath_preserve_symlink(path)


def _default_python() -> str:
    contract_path = REPO_ROOT / DEFAULT_WBC_PY_REL
    if contract_path.is_file():
        return str(contract_path)
    return sys.executable


def _mode_slug(mode: str) -> str:
    if mode == "smoke":
        return "g2_main_v2_smoke_1step"
    if mode == "probe":
        return "g2_main_v2_throughput_probe_50step"
    if mode == "full":
        return "g2_main_v2_full_training"
    return "g2_main_v2_preflight"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="4Y_gr00t_g2_main_v2.py",
        description="GR00T G2-main-v2 launcher using binary text indicators and strict/full train scope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("preflight", "smoke", "probe", "full"), default="preflight")
    parser.add_argument("--dataset-path", default=str(gr00t_main_recap.DEFAULT_G2_MAIN_V2_DATASET_REL))
    parser.add_argument("--critic-dir", default=str(gr00t_main_recap.DEFAULT_G2_MAIN_V2_CRITIC_DIR_REL))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--runtime-log-dir", default="")
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--cuda-visible-devices", default=DEFAULT_VISIBLE_DEVICE)
    parser.add_argument("--indicator-dropout-p", type=float, default=DEFAULT_INDICATOR_DROPOUT_P)
    parser.add_argument("--train-scope", default=DEFAULT_TRAIN_SCOPE)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--timeout-s", type=int, default=0)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--tmux-session", default="RECAP-omx")
    parser.add_argument("--tmux-window", default="g2-main-v2-full")
    parser.add_argument("--launch-tmux", action="store_true")
    parser.add_argument("--print-command-only", action="store_true")
    return parser


def _default_output_dir(mode: str) -> Path:
    return gr00t_main_recap.DEFAULT_G2_MAIN_V2_OUTPUT_ROOT_REL / _mode_slug(mode)


def _default_runtime_log_dir(mode: str) -> Path:
    return gr00t_main_recap.DEFAULT_G2_MAIN_V2_RUNTIME_ROOT_REL / _mode_slug(mode)


def _steps_for_mode(args: argparse.Namespace) -> int:
    if int(args.max_steps) > 0:
        return int(args.max_steps)
    if args.mode == "smoke":
        return DEFAULT_SMOKE_STEPS
    if args.mode == "probe":
        return DEFAULT_PROBE_STEPS
    return DEFAULT_FULL_STEPS


def _save_steps_for_mode(args: argparse.Namespace) -> int:
    if int(args.save_steps) > 0:
        return int(args.save_steps)
    return _steps_for_mode(args)


def _timeout_for_mode(args: argparse.Namespace) -> int:
    if int(args.timeout_s) > 0:
        return int(args.timeout_s)
    return int(DEFAULT_TIMEOUT_S[str(args.mode)])


def _build_train_cmd(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_log_dir: Path,
    summary_json: Path,
) -> list[str]:
    train_script = REPO_ROOT / "work/recap/scripts/34b_recap_numeric_adv_smoke.py"
    return [
        str(_resolve_path(args.python)),
        str(train_script),
        "--dataset-path",
        str(_resolve_path(args.dataset_path)),
        "--output-dir",
        str(output_dir),
        "--runtime-log-dir",
        str(runtime_log_dir),
        "--runtime-log-prefix",
        _mode_slug(str(args.mode)),
        "--summary-json",
        str(summary_json),
        "--python",
        str(_resolve_path(args.python)),
        "--patched-out-root",
        "agent/artifacts/gr00t_recap_live/hf_patches",
        "--no-force-top-llm-layers-zero",
        "--conditioning-route",
        "text_indicator_v1",
        "--runtime-indicator-mode",
        "positive",
        "--indicator-dropout-p",
        str(float(args.indicator_dropout_p)),
        "--text-indicator-prompt-raw-column",
        "recap_m2.prompt_raw",
        "--text-indicator-step-text-fallback",
        "--bypass-scope-supervisor",
        "--recap-train-scope",
        str(args.train_scope),
        "--allow-downgrade",
        "false",
        "--max-steps",
        str(_steps_for_mode(args)),
        "--save-steps",
        str(_save_steps_for_mode(args)),
        "--save-total-limit",
        "1",
        "--global-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--dataloader-num-workers",
        "0",
        "--learning-rate",
        str(float(args.learning_rate)),
        "--condition-hot-lr-scale",
        "1.0",
        "--diffusion-trunk-lr-scale",
        "1.0",
        "--num-gpus",
        "1",
        "--seed",
        str(int(args.seed)),
        "--emit-optimizer-param-group-report",
        "--emit-in-memory-delta-report",
        "--emit-saved-checkpoint-delta-report",
        "--no-use-wandb",
    ]


def _write_preflight(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_log_dir: Path,
) -> dict[str, Any]:
    payload = gr00t_main_recap.build_g2_main_v2_preflight(
        dataset_path=_resolve_path(args.dataset_path),
        output_dir=output_dir,
        runtime_log_dir=runtime_log_dir,
        critic_dir=_resolve_path(args.critic_dir),
        indicator_dropout_p=float(args.indicator_dropout_p),
        seed=int(args.seed),
        train_scope=str(args.train_scope),
    )
    gr00t_main_recap.write_json_atomic(output_dir / "g2_main_v2_preflight.json", payload)
    return payload


def _ensure_tmux_session(session: str) -> None:
    has_session = subprocess.run(
        ["tmux", "has-session", "-t", session],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if has_session.returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", session], cwd=REPO_ROOT, check=True)


def _launch_tmux(
    *,
    args: argparse.Namespace,
    cmd: list[str],
    runtime_log_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _ensure_tmux_session(str(args.tmux_session))
    launcher_log = runtime_log_dir / "tmux_launcher.log"
    timeout_s = _timeout_for_mode(args)
    shell_cmd = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(args.cuda_visible_devices))} && "
        f"timeout {int(timeout_s)}s {shlex.join(cmd)} 2>&1 | tee {shlex.quote(str(launcher_log))}; "
        "echo; echo '[g2-main-v2 tmux command finished]'; exec bash"
    )
    subprocess.run(
        [
            "tmux",
            "new-window",
            "-t",
            str(args.tmux_session),
            "-n",
            str(args.tmux_window),
            shell_cmd,
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    payload = {
        "schema_version": "gr00t_g2_main_v2_tmux_launch_v1",
        "status": "started",
        "tmux_session": str(args.tmux_session),
        "tmux_window": str(args.tmux_window),
        "runtime_log_dir": str(runtime_log_dir),
        "launcher_log": str(launcher_log),
        "output_dir": str(output_dir),
        "timeout_s": timeout_s,
        "cmd": cmd,
        "cmd_shell": shlex.join(cmd),
    }
    gr00t_main_recap.write_json_atomic(output_dir / "tmux_launch_manifest.json", payload)
    return payload


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    output_dir = _resolve_path(args.output_dir) if str(args.output_dir).strip() else _resolve_path(_default_output_dir(str(args.mode)))
    runtime_log_dir = _resolve_path(args.runtime_log_dir) if str(args.runtime_log_dir).strip() else _resolve_path(_default_runtime_log_dir(str(args.mode)))
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    preflight = _write_preflight(args=args, output_dir=output_dir, runtime_log_dir=runtime_log_dir)
    if args.mode == "preflight":
        print(json.dumps(preflight, ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if preflight["launch_gate"]["g2_main_v2_launch_allowed"] else 1

    summary_json = (
        _resolve_path(args.summary_json)
        if str(args.summary_json).strip()
        else output_dir / "g2_main_v2_training_summary.json"
    )
    cmd = _build_train_cmd(
        args=args,
        output_dir=output_dir,
        runtime_log_dir=runtime_log_dir,
        summary_json=summary_json,
    )
    print(shlex.join(cmd))
    if bool(args.print_command_only):
        return 0
    if bool(args.launch_tmux):
        payload = _launch_tmux(
            args=args,
            cmd=cmd,
            runtime_log_dir=runtime_log_dir,
            output_dir=output_dir,
        )
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        timeout=_timeout_for_mode(args),
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
