#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.demo_utils import paths as demo_paths


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json"
)
DEFAULT_RUNTIME_LOG_DIR = (
    REPO_ROOT / "agent/runtime_logs/stage3_delegate_runtime_repair"
)
DEFAULT_FLASH_ATTN_VERSION = "2.8.3"

PROBE_SNIPPET = textwrap.dedent(
    """
    import json
    import platform
    import sys

    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch

        payload["torch_import_ok"] = True
        payload["torch_version"] = torch.__version__
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["torch_cuda_arch_list"] = list(torch.cuda.get_arch_list())
        if torch.cuda.is_available():
            payload["torch_device_name"] = torch.cuda.get_device_name(0)
            payload["torch_device_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        payload["torch_import_ok"] = False
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
        payload["torch_cuda_available"] = False
        payload["torch_cuda_arch_list"] = []

    try:
        import idna

        payload["idna_import_ok"] = True
        payload["idna_version"] = getattr(idna, "__version__", None)
    except Exception as exc:
        payload["idna_import_ok"] = False
        payload["idna_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import flash_attn

        payload["flash_attn_import_ok"] = True
        payload["flash_attn_version"] = getattr(flash_attn, "__version__", None)
    except Exception as exc:
        payload["flash_attn_import_ok"] = False
        payload["flash_attn_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from transformers.utils import is_flash_attn_2_available

        payload["transformers_probe_import_ok"] = True
        payload["flash_attn_2_available"] = bool(is_flash_attn_2_available())
    except Exception as exc:
        payload["transformers_probe_import_ok"] = False
        payload["flash_attn_2_available"] = False
        payload["transformers_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import flash_attn_2_cuda

        payload["flash_attn_2_cuda_import_ok"] = True
        payload["flash_attn_2_cuda_file"] = getattr(flash_attn_2_cuda, "__file__", None)
    except Exception as exc:
        payload["flash_attn_2_cuda_import_ok"] = False
        payload["flash_attn_2_cuda_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    """
).strip()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat(timespec="seconds")


def stamp_now() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def rel_or_abs(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def normalize_runtime_python(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return demo_paths.remap_legacy_project_root(path)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


class SessionLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def line(self, message: str = "") -> None:
        text = str(message)
        print(text)
        self._handle.write(text + "\n")
        self._handle.flush()

    def step(self, message: str) -> None:
        self.line(f"[{iso_now()}] {message}")


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    output: str
    timed_out: bool


def run_logged_command(
    command: list[str],
    *,
    logger: SessionLogger,
    timeout_s: int,
    description: str,
) -> CommandResult:
    logger.step(description)
    logger.line(f"$ {' '.join(shlex.quote(part) for part in command)}")
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": ""},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        output = completed.stdout or ""
        if output:
            for line in output.rstrip("\n").splitlines():
                logger.line(line)
        returncode = int(completed.returncode)
        timed_out = False
    except subprocess.TimeoutExpired:
        output = ""
        timeout_note = f"[TIMEOUT] command exceeded {timeout_s}s"
        logger.line(timeout_note)
        output += timeout_note + "\n"
        returncode = 124
        timed_out = True
    return CommandResult(
        command=list(command),
        returncode=int(returncode),
        output=output,
        timed_out=timed_out,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_probe(payload: dict[str, Any]) -> dict[str, Any]:
    arch_list = list(payload.get("torch_cuda_arch_list") or [])
    issues: list[str] = []
    if not payload.get("torch_import_ok"):
        issues.append("torch_import_failed")
    if not payload.get("idna_import_ok"):
        issues.append("idna_missing")
    if not payload.get("flash_attn_2_available"):
        issues.append("flash_attn_2_unavailable")
    if not payload.get("flash_attn_2_cuda_import_ok"):
        issues.append("flash_attn_2_cuda_import_failed")
    return {
        "healthy": not issues,
        "issues": issues,
        "torch_version": payload.get("torch_version"),
        "torch_cuda_available": bool(payload.get("torch_cuda_available")),
        "torch_cuda_arch_list": arch_list,
        "torch_arch_list_has_sm_120": (
            "sm_120" in arch_list or "compute_120" in arch_list
        ),
        "idna_import_ok": bool(payload.get("idna_import_ok")),
        "flash_attn_import_ok": bool(payload.get("flash_attn_import_ok")),
        "flash_attn_version": payload.get("flash_attn_version"),
        "flash_attn_2_available": bool(payload.get("flash_attn_2_available")),
        "flash_attn_2_cuda_import_ok": bool(payload.get("flash_attn_2_cuda_import_ok")),
    }


def probe_runtime(runtime_python: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        result = subprocess.run(
            [str(runtime_python), "-c", PROBE_SNIPPET],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": ""},
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        probe_record = {
            "command": [str(runtime_python), "-c", PROBE_SNIPPET],
            "returncode": 124,
            "stdout": "",
            "stderr": "delegate probe timed out after 120s",
            "parse_error": "delegate_probe_timeout",
        }
        return probe_record, {
            "healthy": False,
            "issues": ["delegate_probe_timeout"],
        }
    probe_record: dict[str, Any] = {
        "command": [str(runtime_python), "-c", PROBE_SNIPPET],
        "returncode": int(result.returncode),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        probe_record["parse_error"] = "delegate_probe_returncode_nonzero"
        return probe_record, {
            "healthy": False,
            "issues": ["delegate_probe_failed"],
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        probe_record["parse_error"] = f"JSONDecodeError: {exc}"
        return probe_record, {
            "healthy": False,
            "issues": ["delegate_probe_payload_invalid"],
        }
    probe_record["payload"] = payload
    return probe_record, summarize_probe(payload)


def repair_idna(runtime_python: Path, logger: SessionLogger) -> CommandResult:
    return run_logged_command(
        [str(runtime_python), "-m", "pip", "install", "idna"],
        logger=logger,
        timeout_s=900,
        description="检测到 idna 缺失，使用 delegate runtime 自带 pip 执行最小修复。",
    )


def repair_flash_attn(
    runtime_python: Path,
    *,
    logger: SessionLogger,
    flash_attn_version: str,
) -> CommandResult:
    return run_logged_command(
        [
            str(runtime_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            f"flash-attn=={flash_attn_version}",
        ],
        logger=logger,
        timeout_s=1800,
        description=(
            "检测到 flash-attn 相关探针未通过，优先尝试隔离在 delegate venv 内的可复现修复。"
        ),
    )


def build_fail_close_next_steps(
    *,
    health: dict[str, Any],
    flash_result: CommandResult | None,
) -> list[str]:
    steps: list[str] = []
    if not health.get("torch_arch_list_has_sm_120"):
        steps.append(
            "当前 torch CUDA binary 未证明支持 sm_120；请先按 agent/archive/prompts/11b_blackwell_sm120_pytorch.md 修复 torch，再回到本脚本。"
        )
    if flash_result is not None:
        output = flash_result.output
        if "nvcc" in output or "CUDA_HOME" in output:
            steps.append(
                "flash-attn 安装日志显示缺少 nvcc 或 CUDA_HOME；这属于系统级 prerequisite，需按 AGENTS.md 的 sudo gate 或 devel 容器路线补齐后再重跑。"
            )
        if "Building wheel for flash-attn" in output:
            steps.append(
                "本次未命中预编译 wheel，已进入源码构建路径；若当前机器不具备完整 CUDA toolkit，请不要假装修复成功。"
            )
    if not steps:
        steps.append(
            "delegate runtime 仍未满足 flash-attn 依赖，请先阅读 agent/archive/prompts/11a_smoke_eval_fix.md 和 11b_blackwell_sm120_pytorch.md，再根据最新日志选择 venv 内重装、devel 容器或系统级 toolkit 路线。"
        )
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stage3_delegate_runtime_repair.py",
        description=(
            "面向 iter_002 delegate runtime 的探针优先修复工具：默认从 manifest 读取 delegate_runtime_python，先检查 idna / flash-attn，再在确有缺失时执行最小修复。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--iteration-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="默认读取的 iter_002 manifest 路径。",
    )
    _ = parser.add_argument(
        "--runtime-python",
        type=Path,
        default=None,
        help="可选：手工覆盖目标解释器；未提供时以 manifest 中的 delegate_runtime_python 为准。",
    )
    _ = parser.add_argument(
        "--runtime-log-dir",
        type=Path,
        default=DEFAULT_RUNTIME_LOG_DIR,
        help="运行日志与 JSON summary 的输出目录。",
    )
    _ = parser.add_argument(
        "--probe-only",
        action="store_true",
        help="仅做探针并输出结论；若环境不健康则 fail-close，不执行任何修复。",
    )
    _ = parser.add_argument(
        "--flash-attn-version",
        default=DEFAULT_FLASH_ATTN_VERSION,
        help="需要执行隔离 venv 修复时默认安装的 flash-attn 版本。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.iteration_manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = load_manifest(manifest_path)
    runtime_python = (
        normalize_runtime_python(args.runtime_python)
        if args.runtime_python is not None
        else normalize_runtime_python(str(manifest["delegate_runtime_python"]))
    )
    if not runtime_python.is_file():
        raise SystemExit(f"Missing delegate runtime python: {runtime_python}")

    stamp = stamp_now()
    runtime_log_dir = Path(args.runtime_log_dir).resolve()
    session_log_path = runtime_log_dir / f"stage3_delegate_runtime_repair_{stamp}.log"
    summary_json_path = runtime_log_dir / f"stage3_delegate_runtime_repair_{stamp}.json"
    logger = SessionLogger(session_log_path)
    try:
        logger.step("开始执行 stage3 delegate runtime probe-first repair。")
        logger.line(f"manifest = {manifest_path}")
        logger.line(f"delegate_runtime_python = {runtime_python}")
        logger.line(f"probe_only = {bool(args.probe_only)}")

        initial_probe, initial_health = probe_runtime(runtime_python)
        logger.step("已完成初始探针。")
        logger.line(
            json.dumps(initial_probe, ensure_ascii=False, indent=2, sort_keys=True)
        )

        summary: dict[str, Any] = {
            "checked_at": iso_now(),
            "delegate_runtime_python": str(runtime_python),
            "delegate_runtime_python_realpath": str(runtime_python.resolve()),
            "iteration_manifest": rel_or_abs(manifest_path),
            "manifest_runtime_source": "delegate_runtime_python",
            "probe_only": bool(args.probe_only),
            "performed_mutation": False,
            "repair_attempted": False,
            "repair_actions": [],
            "initial_probe": initial_probe,
            "initial_health": initial_health,
            "final_probe": initial_probe,
            "final_health": initial_health,
            "artifacts": {
                "session_log": rel_or_abs(session_log_path),
                "summary_json": rel_or_abs(summary_json_path),
            },
            "notes": [],
            "next_steps": [],
        }

        if initial_health.get("healthy"):
            summary["status"] = "healthy_noop"
            summary["exit_code"] = 0
            summary["notes"].append(
                "当前 manifest 绑定 delegate runtime 已健康；本次仅记录探针结果，不执行任何 pip 变更。"
            )
            logger.step("探针确认环境健康，按 no-op 路径退出。")
            json_dump(summary_json_path, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.probe_only:
            summary["status"] = "unhealthy_probe_only"
            summary["exit_code"] = 1
            summary["notes"].append(
                "已按 --probe-only 只做探针；由于环境不健康，本次未进行任何修复。"
            )
            summary["next_steps"] = build_fail_close_next_steps(
                health=initial_health,
                flash_result=None,
            )
            logger.step("probe-only 模式下检测到环境不健康，fail-close 返回。")
            json_dump(summary_json_path, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 1

        current_probe = initial_probe
        current_health = initial_health
        flash_result: CommandResult | None = None

        if "idna_missing" in list(current_health.get("issues") or []):
            summary["repair_attempted"] = True
            idna_result = repair_idna(runtime_python, logger)
            summary["performed_mutation"] = True
            summary["repair_actions"].append(
                {
                    "action": "install_idna",
                    "command": idna_result.command,
                    "returncode": idna_result.returncode,
                    "timed_out": idna_result.timed_out,
                }
            )
            current_probe, current_health = probe_runtime(runtime_python)
            summary["final_probe"] = current_probe
            summary["final_health"] = current_health
            logger.step("已完成 idna 修复后的复探针。")
            logger.line(
                json.dumps(current_probe, ensure_ascii=False, indent=2, sort_keys=True)
            )
            if current_health.get("healthy"):
                summary["status"] = "repaired_idna"
                summary["exit_code"] = 0
                summary["notes"].append(
                    "缺失的 idna 已在 delegate venv 内修复，复探针通过。"
                )
                json_dump(summary_json_path, summary)
                print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
                return 0

        flash_related_failed = any(
            issue in list(current_health.get("issues") or [])
            for issue in ["flash_attn_2_unavailable", "flash_attn_2_cuda_import_failed"]
        )
        if flash_related_failed:
            if not current_health.get("torch_arch_list_has_sm_120"):
                summary["status"] = "repair_blocked_torch_arch"
                summary["exit_code"] = 1
                summary["next_steps"] = build_fail_close_next_steps(
                    health=current_health,
                    flash_result=None,
                )
                summary["notes"].append(
                    "未进入 flash-attn 安装步骤：当前 torch 还未证明支持 sm_120，继续盲修只会制造假阳性。"
                )
                logger.step("torch 架构不满足前置条件，按 fail-close 退出。")
                json_dump(summary_json_path, summary)
                print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
                return 1

            summary["repair_attempted"] = True
            flash_result = repair_flash_attn(
                runtime_python,
                logger=logger,
                flash_attn_version=str(args.flash_attn_version),
            )
            summary["performed_mutation"] = True
            summary["repair_actions"].append(
                {
                    "action": "repair_flash_attn",
                    "command": flash_result.command,
                    "returncode": flash_result.returncode,
                    "timed_out": flash_result.timed_out,
                    "version": str(args.flash_attn_version),
                }
            )
            current_probe, current_health = probe_runtime(runtime_python)
            summary["final_probe"] = current_probe
            summary["final_health"] = current_health
            logger.step("已完成 flash-attn 修复后的复探针。")
            logger.line(
                json.dumps(current_probe, ensure_ascii=False, indent=2, sort_keys=True)
            )
            if current_health.get("healthy"):
                summary["status"] = "repaired_flash_attn"
                summary["exit_code"] = 0
                summary["notes"].append(
                    "flash-attn 已在 delegate venv 内完成修复，复探针通过。"
                )
                json_dump(summary_json_path, summary)
                print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
                return 0

        summary["status"] = "repair_failed"
        summary["exit_code"] = 1
        summary["notes"].append(
            "脚本已按 probe-first 策略执行最小修复或前置条件检查，但 delegate runtime 仍未恢复到健康状态。"
        )
        summary["next_steps"] = build_fail_close_next_steps(
            health=current_health,
            flash_result=flash_result,
        )
        logger.step("修复后仍未健康，按 fail-close 退出并给出下一步。")
        json_dump(summary_json_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
