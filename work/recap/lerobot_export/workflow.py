#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import os
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

ITER_TAG = "recap_iter_000"
INPUT_DATASET_DIR_REL = "agent/artifacts/recap_datasets"
OUTPUT_DATASET_DIR_REL = "agent/artifacts/lerobot_datasets"

MAX_EPISODES = 1
TOTAL_TIMEOUT_S = 180
REQUIRE_FFMPEG = False


def _ensure_repo_root_on_syspath(repo_root_guess: Path) -> None:
    p = str(repo_root_guess)
    if p not in sys.path:
        sys.path.insert(0, p)


def _repo_root() -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "repo_root")
    return cast(Path, fn(from_path=__file__))


def _maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "maybe_reexec_into_wbc_venv")
    fn(repo_root)


@contextlib.contextmanager
def _tee_stdio(log_path: Path, *, header: str) -> Iterator[None]:
    mod = importlib.import_module("work.demo_utils.tee")
    fn = getattr(mod, "tee_stdio")
    with fn(Path(log_path), header=str(header)):
        yield


def _install_alarm_timeout(timeout_s: float | None) -> None:
    if timeout_s is None:
        return
    try:
        t = int(float(timeout_s))
    except Exception:
        return
    if t <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Timed out after {t}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(t)


def _clear_alarm_timeout() -> None:
    if hasattr(signal, "SIGALRM"):
        try:
            signal.alarm(0)
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="39_recap_export_lerobot_v2_with_video.py",
        description="RECAP M3 exporter (M1+M2 -> LeRobot v2) with archived videos injected.",
    )
    p.add_argument("--iter-tag", type=str, default=str(ITER_TAG))
    p.add_argument("--max-episodes", type=int, default=int(MAX_EPISODES))
    p.add_argument(
        "--total-timeout-s",
        type=float,
        default=float(TOTAL_TIMEOUT_S),
        help="Hard timeout fuse (best-effort) for the whole script.",
    )
    p.add_argument(
        "--require-ffmpeg",
        action="store_true",
        default=bool(REQUIRE_FFMPEG),
        help="Fail fast if ffmpeg/ffprobe are missing.",
    )

    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--dual-task-text",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Export task_text as a deterministic mix of raw/conditioned prompts.",
        )
    else:
        g = p.add_mutually_exclusive_group(required=False)
        g.add_argument(
            "--dual-task-text",
            dest="dual_task_text",
            action="store_true",
            help="Export task_text as a deterministic mix of raw/conditioned prompts.",
        )
        g.add_argument(
            "--no-dual-task-text",
            dest="dual_task_text",
            action="store_false",
            help="Disable dual task_text export; use single task_text mode.",
        )
        p.set_defaults(dual_task_text=True)
    return p


class LeRobotVideoExportWorkflow:
    def run(self) -> int:
        if any(a in ("-h", "--help") for a in sys.argv[1:]):
            try:
                _build_parser().parse_args()
            except SystemExit as e:
                return int(getattr(e, "code", 0) or 0)
            return 0

        args = _build_parser().parse_args()

        repo_root_guess = Path(__file__).resolve().parents[3]
        _ensure_repo_root_on_syspath(repo_root_guess)
        repo_root = _repo_root()
        _ensure_repo_root_on_syspath(repo_root)
        _maybe_reexec_into_wbc_venv(repo_root)

        iter_tag = str(getattr(args, "iter_tag", "") or ITER_TAG)
        runtime_dir = repo_root / "agent" / "runtime_logs" / iter_tag
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = runtime_dir / "m3_export_with_video.log"

        total_timeout_s = float(getattr(args, "total_timeout_s", 0.0) or 0.0)
        max_episodes = int(getattr(args, "max_episodes", 0) or 0)
        require_ffmpeg = bool(getattr(args, "require_ffmpeg", False))
        dual_task_text = bool(getattr(args, "dual_task_text", True))
        if max_episodes <= 0:
            raise ValueError(f"max_episodes must be > 0, got {max_episodes}")

        input_iter_dir_rel = str(Path(INPUT_DATASET_DIR_REL) / iter_tag)
        output_iter_dir_rel = str(Path(OUTPUT_DATASET_DIR_REL) / iter_tag)

        t0_total = time.monotonic()
        with _tee_stdio(log_path, header="39_recap_export_lerobot_v2_with_video"):
            _install_alarm_timeout(total_timeout_s or None)
            try:
                print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
                print("[INFO] python:", sys.version.replace("\n", " "))
                print("[INFO] sys.executable:", sys.executable)
                print("[INFO] iter_tag:", iter_tag)
                print("[INFO] input_iter_dir_rel:", input_iter_dir_rel)
                print("[INFO] output_iter_dir_rel:", output_iter_dir_rel)
                print("[INFO] runtime_dir:", str(runtime_dir))
                print("[INFO] log_path:", str(log_path))
                print("[INFO] require_ffmpeg:", bool(require_ffmpeg))
                print("[INFO] dual_task_text:", bool(dual_task_text))

                exp_mod = importlib.import_module(
                    "work.recap.lerobot_export.video_export"
                )
                export_fn = getattr(exp_mod, "export_recap_to_lerobot_v2_with_video")

                result: Any = export_fn(
                    iter_tag=str(iter_tag),
                    repo_root=repo_root,
                    input_recap_dataset_dir=str(input_iter_dir_rel),
                    output_dataset_dir=str(output_iter_dir_rel),
                    max_episodes=int(max_episodes),
                    require_ffmpeg=bool(require_ffmpeg),
                    dual_task_text=bool(dual_task_text),
                )

                output_dataset_dir = getattr(result, "output_dataset_dir", None)
                total_videos = int(getattr(result, "total_videos"))
                video_path_template = str(getattr(result, "video_path_template"))
                video_map_path = getattr(result, "video_map_path", None)

                print("[EVIDENCE] output.total_videos:", int(total_videos))
                print(
                    "[EVIDENCE] output.video_path_template:", str(video_path_template)
                )
                if output_dataset_dir is not None:
                    print("[EVIDENCE] output.dataset_dir:", str(output_dataset_dir))
                if video_map_path is not None:
                    print("[EVIDENCE] output.video_map_path:", str(video_map_path))

                elapsed = time.monotonic() - t0_total
                print("[INFO] done elapsed_s:", f"{elapsed:.2f}")
                return 0
            except KeyboardInterrupt:
                print("\n[INFO] KeyboardInterrupt -> stop early")
                return 130
            finally:
                _clear_alarm_timeout()


def main() -> int:
    return LeRobotVideoExportWorkflow().run()


if __name__ == "__main__":
    raise SystemExit(main())


class RecapExportLeRobotWithVideoScriptApp:
    def run(self) -> int:
        return main()
