#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path


from work.demo_utils import paths as demo_paths
from work.recap.finetune_full import resolve_full_update_authority_output_dir


DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_UPSTREAM_SCRIPT_REL = "work/recap/launch_finetune_use_ddp.py"
DEFAULT_WBC_PY_REL = (
    "submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/"
    "GR00T-WholeBodyControl_uv/.venv/bin/python"
)
LIVE_LAUNCH_FAMILY = "single_gpu_v1"
HISTORICAL_DDP_LAUNCH_FAMILY = "task10_2gpu_ddp_diagnostic_v1"
DEFAULT_FORMAL_OUTPUT_DIR_REL = "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run"
DEFAULT_FORMAL_VISIBLE_DEVICE = "1"
MANUAL_FALLBACK_VISIBLE_DEVICE = "2"
ALLOWED_SINGLE_VISIBLE_DEVICES = frozenset({"1", "2"})
REQUIRED_DUAL_VISIBLE_DEVICES = "1,2"
MULTI_GPU_ENV_DEFAULTS = {
    "CUDA_VISIBLE_DEVICES": REQUIRED_DUAL_VISIBLE_DEVICES,
    "CUDA_LAUNCH_BLOCKING": "1",
    "NCCL_DEBUG": "INFO",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "work").is_dir() and (parent / "agent").is_dir():
            return parent
    return Path.cwd().resolve()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="34_recap_finetune_repro.py",
        description=(
            "RECAP finetune wrapper: patch HF base-model config (tune_top_llm_layers=0) "
            "via local HF cache scan, then launch the repo-local finetune launcher with a patched "
            "--base-model-path. NOTE: this wrapper enforces diffusion-model tuning unless explicitly "
            "overridden."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-model",
        type=str,
        default=str(DEFAULT_BASE_MODEL),
        help="HF repo_id of the base model (cached locally under ~/.cache/huggingface/hub by default).",
    )
    p.add_argument(
        "--base-model-revision",
        type=str,
        default="",
        help=(
            "Optional snapshot hash to pin under HF cache snapshots/. "
            "If empty, wrapper falls back to best-effort scan (mtime-latest)."
        ),
    )
    p.add_argument(
        "--hf-hub-cache-dir",
        type=str,
        default="",
        help="Optional HF hub cache root to scan (override default resolver).",
    )
    p.add_argument(
        "--patched-out-root",
        type=str,
        default="agent/artifacts/hf_patches",
        help="Repo-relative output root for patched base-model dirs.",
    )
    p.add_argument(
        "--python",
        type=str,
        default="",
        help=(
            "Python executable used to run the repo-local training launcher. "
            "Default priority: --python > env WBC_PY > fixed WBC venv path (M5)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate patched base-model dir and print evidence; do not launch finetune.",
    )
    return p


def _normalize_opt_str(v: str | None) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _resolve_python_exe(repo_root: Path, *, python_flag: str | None) -> Path:
    explicit = _normalize_opt_str(python_flag)
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = repo_root / p
        return demo_paths.abspath_preserve_symlink(p)

    env_p = _normalize_opt_str(os.environ.get("WBC_PY"))
    if env_p is not None:
        p = Path(env_p).expanduser()
        if not p.is_absolute():
            p = repo_root / p
        return demo_paths.abspath_preserve_symlink(p)

    try:
        contract = demo_paths.load_stage3_training_python_contract(repo_root)
        return demo_paths.abspath_preserve_symlink(contract["delegate_runtime_python"])
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return demo_paths.abspath_preserve_symlink(repo_root / DEFAULT_WBC_PY_REL)


def _is_executable_file(path: Path) -> bool:
    try:
        return path.is_file() and os.access(str(path), os.X_OK)
    except Exception:
        return False


def _has_flag(args: list[str], *, prefixes: tuple[str, ...]) -> bool:
    for a in args:
        for p in prefixes:
            if a == p or a.startswith(p + "="):
                return True
    return False


def _strip_flag_with_value(args: list[str], *, flag: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == flag:
            i += 1
            if i < len(args) and not args[i].startswith("-"):
                i += 1
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _read_int_flag(args: list[str], *, flag: str) -> int | None:
    i = 0
    while i < len(args):
        a = args[i]
        if a == flag:
            if i + 1 >= len(args):
                raise ValueError(
                    f"Expected integer value after {flag}, got end of args"
                )
            return int(args[i + 1])
        if a.startswith(flag + "="):
            return int(a.split("=", 1)[1])
        i += 1
    return None


def _read_str_flag(args: list[str], *, flag: str) -> str | None:
    i = 0
    while i < len(args):
        a = args[i]
        if a == flag:
            if i + 1 >= len(args):
                raise ValueError(f"Expected string value after {flag}, got end of args")
            return str(args[i + 1])
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        i += 1
    return None


def _is_default_formal_output_dir(raw: str | None) -> bool:
    normalized = _normalize_opt_str(raw)
    if normalized is None:
        return False
    normalized_path = Path(normalized).as_posix()
    return normalized_path.endswith(DEFAULT_FORMAL_OUTPUT_DIR_REL)


def _build_launch_policy_metadata(
    *,
    forwarded_args: list[str],
    num_gpus: int | None,
    validated_visible_devices: str | None,
) -> dict[str, int | str | bool | None]:
    output_dir = _read_str_flag(forwarded_args, flag="--output-dir")
    if output_dir is not None:
        resolve_full_update_authority_output_dir(
            _repo_root(),
            output_dir,
            require_v2_authority=False,
        )
    is_default_formal_path = _is_default_formal_output_dir(output_dir)
    if num_gpus is None:
        return {
            "live_launch_family": None,
            "visible_devices_policy": None,
            "torchrun_invoked": False,
            "formal_path_default_gpu1_authority": False,
            "output_dir": output_dir,
        }
    if int(num_gpus) > 1:
        return {
            "live_launch_family": HISTORICAL_DDP_LAUNCH_FAMILY,
            "visible_devices_policy": "torchrun_gpu1_gpu2_only",
            "torchrun_invoked": True,
            "formal_path_default_gpu1_authority": False,
            "output_dir": output_dir,
        }
    if is_default_formal_path and validated_visible_devices != DEFAULT_FORMAL_VISIBLE_DEVICE:
        raise ValueError(
            "single_gpu_v1 formal path requires CUDA_VISIBLE_DEVICES to be exactly '1'"
        )
    if validated_visible_devices == MANUAL_FALLBACK_VISIBLE_DEVICE:
        policy = "single_gpu_gpu2_manual_fallback"
    else:
        policy = "single_gpu_gpu1_only"
    return {
        "live_launch_family": LIVE_LAUNCH_FAMILY,
        "visible_devices_policy": policy,
        "torchrun_invoked": False,
        "formal_path_default_gpu1_authority": bool(is_default_formal_path),
        "output_dir": output_dir,
    }


def _pick_free_master_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _normalize_visible_devices(value: str | None) -> str | None:
    s = _normalize_opt_str(value)
    if s is None:
        return None
    parts = [part.strip() for part in s.split(",")]
    if any(part == "" for part in parts):
        return s
    return ",".join(parts)


def _validate_visible_devices_for_num_gpus(
    *, num_gpus: int | None, cuda_visible_devices: str | None
) -> str | None:
    normalized = _normalize_visible_devices(cuda_visible_devices)
    if num_gpus is None:
        return normalized
    if int(num_gpus) > 2:
        raise ValueError(
            "recap_finetune_repro_app.py only supports --num-gpus 1 or 2; "
            f"got {int(num_gpus)}"
        )
    if int(num_gpus) < 1:
        raise ValueError(f"--num-gpus must be >= 1; got {int(num_gpus)}")
    if int(num_gpus) == 1:
        if normalized is None:
            raise ValueError(
                "When --num-gpus 1, CUDA_VISIBLE_DEVICES must be set explicitly "
                "to a single allowed device: 1 or 2"
            )
        if normalized not in ALLOWED_SINGLE_VISIBLE_DEVICES:
            raise ValueError(
                "When --num-gpus 1, CUDA_VISIBLE_DEVICES must be exactly 1 or 2; "
                f"got {normalized!r}"
            )
        return normalized
    if normalized is None:
        raise ValueError(
            "When --num-gpus 2, CUDA_VISIBLE_DEVICES must be set explicitly to 1,2"
        )
    if normalized != REQUIRED_DUAL_VISIBLE_DEVICES:
        raise ValueError(
            "When --num-gpus 2, CUDA_VISIBLE_DEVICES must be exactly 1,2; "
            f"got {normalized!r}"
        )
    return normalized


def _build_training_launcher_cmd(
    *,
    python_exe: Path,
    upstream_script: Path,
    forwarded_args: list[str],
    cuda_visible_devices: str | None,
) -> tuple[list[str], dict[str, int | str | bool | None], dict[str, str]]:
    num_gpus = _read_int_flag(forwarded_args, flag="--num-gpus")
    validated_visible_devices = _validate_visible_devices_for_num_gpus(
        num_gpus=num_gpus,
        cuda_visible_devices=cuda_visible_devices,
    )
    launch_policy = _build_launch_policy_metadata(
        forwarded_args=forwarded_args,
        num_gpus=num_gpus,
        validated_visible_devices=validated_visible_devices,
    )
    if num_gpus is not None and int(num_gpus) > 1:
        master_port = _pick_free_master_port()
        cmd = [
            str(python_exe),
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(int(num_gpus)),
            "--master_port",
            str(int(master_port)),
            str(upstream_script),
            *forwarded_args,
        ]
        return (
            cmd,
            {
                "num_gpus": int(num_gpus),
                "uses_torchrun": True,
                "master_port": int(master_port),
                "visible_devices": REQUIRED_DUAL_VISIBLE_DEVICES,
                **launch_policy,
            },
            dict(MULTI_GPU_ENV_DEFAULTS),
        )

    single_env = (
        {}
        if validated_visible_devices is None
        else {"CUDA_VISIBLE_DEVICES": validated_visible_devices}
    )
    return [str(python_exe), str(upstream_script), *forwarded_args], {
        "num_gpus": None if num_gpus is None else int(num_gpus),
        "uses_torchrun": False,
        "master_port": None,
        "visible_devices": validated_visible_devices,
        **launch_policy,
    }, single_env


class RecapFinetuneReproWorkflow:
    def run(self) -> int:
        if any(a in ("-h", "--help") for a in sys.argv[1:]):
            try:
                _build_parser().parse_args()
            except SystemExit as e:
                return int(getattr(e, "code", 0) or 0)
            return 0

        parser = _build_parser()
        args, forwarded = parser.parse_known_args()

        repo_root = _repo_root()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from work.recap.hf_snapshot_patch import make_patched_base_model_dir

        base_model = str(getattr(args, "base_model", "") or DEFAULT_BASE_MODEL)
        revision = _normalize_opt_str(getattr(args, "base_model_revision", ""))
        hf_hub_cache_dir = _normalize_opt_str(getattr(args, "hf_hub_cache_dir", ""))
        patched_out_root = str(
            getattr(args, "patched_out_root", "") or "agent/artifacts/hf_patches"
        )

        patched_dir = make_patched_base_model_dir(
            repo_id=base_model,
            revision=revision,
            out_root=patched_out_root,
            overrides={},
            hf_hub_cache_dir=hf_hub_cache_dir,
            emit_evidence=True,
        )

        print(f"[EVIDENCE] patched_base_model_dir={patched_dir}")
        print("[EVIDENCE] tune_top_llm_layers_patched=0")

        fwd = list(map(str, forwarded))
        fwd = _strip_flag_with_value(fwd, flag="--base-model-path")
        fwd = _strip_flag_with_value(fwd, flag="--base_model_path")

        if not _has_flag(
            fwd, prefixes=("--tune-diffusion-model", "--no-tune-diffusion-model")
        ):
            fwd.append("--no-tune-diffusion-model")

        fwd.extend(["--base-model-path", str(patched_dir)])

        print(f"[EVIDENCE] forwarded_args={shlex.join(fwd)}")

        if bool(getattr(args, "dry_run", False)):
            return 0

        python_exe = _resolve_python_exe(
            repo_root, python_flag=_normalize_opt_str(getattr(args, "python", ""))
        )
        if not _is_executable_file(python_exe):
            raise FileNotFoundError(
                "Cannot find an executable python for upstream finetune. "
                f"Tried: {python_exe} (override via --python or env WBC_PY)"
            )

        upstream_script = (repo_root / DEFAULT_UPSTREAM_SCRIPT_REL).resolve()
        if not upstream_script.is_file():
            raise FileNotFoundError(
                f"Training launcher script not found: {upstream_script}"
            )

        env = dict(os.environ)
        gr00t_root = (repo_root / "submodules" / "Isaac-GR00T").resolve()
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(gr00t_root) + (os.pathsep + old_pp if old_pp else "")

        cmd, launch_meta, launch_env_overrides = _build_training_launcher_cmd(
            python_exe=python_exe,
            upstream_script=upstream_script,
            forwarded_args=fwd,
            cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
        )
        env.update(launch_env_overrides)
        print(f"[INFO] delegate_runtime_python={python_exe}")
        print(f"[INFO] training_launcher_python={python_exe}")
        print(f"[INFO] training_launcher_script={upstream_script}")
        print("[INFO] training_launcher_mode=repo_local_auto_use_ddp")
        print(f"[INFO] training_launcher_uses_torchrun={launch_meta['uses_torchrun']}")
        print(f"[INFO] live_launch_family={launch_meta['live_launch_family']}")
        print(f"[INFO] visible_devices_policy={launch_meta['visible_devices_policy']}")
        print(f"[INFO] torchrun_invoked={launch_meta['torchrun_invoked']}")
        if launch_meta["visible_devices"] is not None:
            print(f"[INFO] training_launcher_visible_devices={launch_meta['visible_devices']}")
        if launch_meta["master_port"] is not None:
            print(f"[INFO] training_launcher_master_port={launch_meta['master_port']}")
        print(f"[INFO] cmd={shlex.join(cmd)}")

        try:
            proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
        except KeyboardInterrupt:
            return 130
        return int(proc.returncode)


def main() -> int:
    return RecapFinetuneReproWorkflow().run()


if __name__ == "__main__":
    raise SystemExit(main())


class RecapFinetuneReproScriptApp:
    def run(self) -> int:
        return main()
