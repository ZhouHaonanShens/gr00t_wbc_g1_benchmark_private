from __future__ import annotations

import argparse
from typing import cast
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "stage1_v22_full_training_eval_iter9_20260426T_nextZ"
COLD_ROOT = Path("/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_archives")


def _now_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _count_tree(root: Path) -> dict[str, int]:
    files = 0
    bytes_total = 0
    if not root.exists():
        return {"files": files, "bytes": bytes_total}
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            size = _file_size(path)
            if size is None:
                continue
            files += 1
            bytes_total += size
    except OSError:
        return {"files": files, "bytes": bytes_total}
    return {"files": files, "bytes": bytes_total}


def _latest_step(run_dir: Path) -> int | None:
    if not run_dir.is_dir():
        return None
    steps = [p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        return None
    return max(int(p.name) for p in steps)


def _line_count(path: Path) -> int | None:
    try:
        if not path.is_file():
            return None
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    try:
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


@dataclass(frozen=True)
class LaneConfig:
    lane: str
    variant_id: str
    gpu: str
    launch_script: str
    output_dir: str
    initial_runtime_dir: str
    training_timeout_s: int | None = None
    disk_min_gb: int | None = None
    disk_warn_gb: int | None = None
    active_no_progress_grace_s: int | None = None


LANES: dict[str, LaneConfig] = {
    "B": LaneConfig(
        lane="B",
        variant_id="control_no_recap_shuffled_adversarial_relabel",
        gpu="1",
        launch_script="agent/run/iter9_relaunch_b_real_training.sh",
        output_dir="agent/artifacts/stage1_v22_full_training_eval_iter9_20260426T_nextZ/openpi/v22_formal_training/B",
        initial_runtime_dir="agent/runtime_logs/iter9_B_real_training_resume_retry2_20260428_013049/B",
        training_timeout_s=230400,
        disk_min_gb=50,
        disk_warn_gb=100,
        active_no_progress_grace_s=3600,
    ),
    "C": LaneConfig(
        lane="C",
        variant_id="main_recap_method",
        gpu="2",
        launch_script="agent/run/iter9_relaunch_c_real_training.sh",
        output_dir="agent/artifacts/stage1_v22_full_training_eval_iter9_20260426T_nextZ/openpi/v22_formal_training/C",
        initial_runtime_dir="agent/runtime_logs/iter9_C_real_training_launch_20260428_000000/C",
        training_timeout_s=201600,
        disk_min_gb=50,
        disk_warn_gb=100,
    ),
    "X": LaneConfig(
        lane="X",
        variant_id="recap_variant_shuffle_diag",
        gpu="3",
        launch_script="agent/run/iter9_relaunch_x_real_training.sh",
        output_dir="agent/artifacts/stage1_v22_full_training_eval_iter9_20260426T_nextZ/openpi/v22_formal_training/X",
        initial_runtime_dir="agent/runtime_logs/iter9_x_real_training_relaunch_20260428_143122/X",
        training_timeout_s=201600,
        disk_min_gb=50,
        disk_warn_gb=100,
    ),
}


@dataclass
class LaneState:
    lane: str
    current_runtime_dir: str
    restart_count: int = 0
    last_action: str = "initial"


def _parse_header_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        if stripped.startswith("VARIANT=") and " GPU=" in stripped:
            variant_token, gpu_value = stripped.split(" GPU=", 1)
            fields["VARIANT"] = variant_token.split("=", 1)[1].strip()
            fields["GPU"] = gpu_value.strip()
            continue
        key, value = stripped.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _parse_utc_epoch(raw_value: object) -> float | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp()


def _process_line_pid(line: str) -> int | None:
    parts = line.strip().split(maxsplit=1)
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


class Iter9BCXSupervisor:
    def __init__(
        self,
        *,
        interval_s: int,
        auto_fix: bool,
        root_dir_name: str | None = None,
        b_active_no_progress_grace_s: int | None = None,
    ) -> None:
        self.interval_s = int(interval_s)
        self.auto_fix = bool(auto_fix)
        self.b_active_no_progress_grace_s = b_active_no_progress_grace_s
        self.root_dir_name = root_dir_name or f"iter9_bcx_supervisor_{_now_local()}"
        self.root_dir = REPO_ROOT / "agent" / "runtime_logs" / self.root_dir_name
        self.status_path = self.root_dir / "supervisor_status.json"
        self.lane_state_path = self.root_dir / "lane_state.json"
        self.snapshot_dir = self.root_dir / "snapshots"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.lanes = {
            lane: LaneState(lane=lane, current_runtime_dir=config.initial_runtime_dir)
            for lane, config in LANES.items()
        }
        self._load_saved_lane_state()

    def _load_saved_lane_state(self) -> None:
        if not self.lane_state_path.is_file():
            return
        try:
            payload = json.loads(self.lane_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        lanes_payload = payload.get("lanes")
        if not isinstance(lanes_payload, dict):
            return
        for lane, state_payload in lanes_payload.items():
            if lane not in self.lanes or not isinstance(state_payload, dict):
                continue
            current_runtime_dir = str(state_payload.get("current_runtime_dir", "")).strip()
            if current_runtime_dir:
                self.lanes[lane].current_runtime_dir = current_runtime_dir
            restart_count = state_payload.get("restart_count")
            if isinstance(restart_count, int):
                self.lanes[lane].restart_count = restart_count
            last_action = str(state_payload.get("last_action", "")).strip()
            if last_action:
                self.lanes[lane].last_action = last_action

    def save_state(self) -> None:
        _atomic_write_json(
            self.lane_state_path,
            {
                "schema_version": "iter9_bcx_supervisor_state_v1",
                "generated_at_local": _now_iso(),
                "root_dir": str(self.root_dir),
                "lanes": {lane: asdict(state) for lane, state in self.lanes.items()},
            },
        )

    def _gpu_snapshot(self) -> dict[str, dict[str, object]]:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        gpu: dict[str, dict[str, object]] = {}
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpu[parts[0]] = {
                "uuid": parts[1],
                "utilization_gpu": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_total_mib": int(parts[4]),
            }
        return gpu

    def _compute_apps(self) -> list[dict[str, object]]:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        rows: list[dict[str, object]] = []
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mib": int(parts[3]),
                }
            )
        return rows

    def _process_lines(self, lane: str) -> list[str]:
        config = LANES[lane]
        output_rel = config.output_dir
        runtime_rel = self.lanes[lane].current_runtime_dir
        ps = subprocess.run(
            ["ps", "-eo", "pid,ppid,stat,etime,pcpu,pmem,nlwp,cmd"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        results = []
        for line in ps:
            if config.variant_id in line or (runtime_rel in line) or (output_rel in line):
                results.append(line)
        return results

    def _lane_snapshot(self, lane: str) -> dict[str, object]:
        config = LANES[lane]
        state = self.lanes[lane]
        runtime = REPO_ROOT / state.current_runtime_dir
        runtime_root = runtime.parent
        output_dir = REPO_ROOT / config.output_dir
        export_runtime = runtime / "real_variant_export_runtime"
        upstream = export_runtime / "upstream_train_checkpoints" / "pi0_libero" / config.variant_id
        real_export = export_runtime / "real_variant_export"
        wrapper_launch_log = runtime_root / "wrapper.launch.log"
        driver_header_path = runtime / "driver.header"
        driver_header_tail = _tail_text(driver_header_path)
        driver_header_fields = _parse_header_fields(driver_header_tail)
        driver_exit_path = runtime / "driver.exit"
        driver_exit_raw = _tail_text(driver_exit_path, max_chars=100).strip()
        try:
            driver_exit_code = int(driver_exit_raw.splitlines()[-1]) if driver_exit_raw else None
        except ValueError:
            driver_exit_code = None
        traces = {
            name: output_dir / name
            for name in (
                "loss_decomposition.jsonl",
                "threshold_switch_trace.jsonl",
                "alpha_dual_loss_trace.jsonl",
                "training_run_manifest.json",
                "checkpoint_provenance.json",
                "training_failure.json",
                "training_timeout_report.json",
                "driver_timeout_report_fallback.json",
                "gradient_attestation.json",
                "SHA256SUMS",
                "shuffle_manifest.json",
                "deterministic_shuffle_provenance.json",
            )
        }
        snapshot = {
            "sample_time_local": _now_iso(),
            "lane": lane,
            "variant_id": config.variant_id,
            "gpu": config.gpu,
            "runtime_dir": str(runtime),
            "output_dir": str(output_dir),
            "process_lines": self._process_lines(lane),
            "gpu_snapshot": self._gpu_snapshot().get(config.gpu, {}),
            "compute_apps": self._compute_apps(),
            "wrapper_launch_log": {
                "exists": wrapper_launch_log.exists(),
                "size": _file_size(wrapper_launch_log),
                "tail": _tail_text(wrapper_launch_log),
            },
            "driver_log": {
                "exists": (runtime / "driver.log").exists(),
                "size": _file_size(runtime / "driver.log"),
                "tail": _tail_text(runtime / "driver.log"),
            },
            "driver_header": {
                "exists": driver_header_path.exists(),
                "path": str(driver_header_path),
                "size": _file_size(driver_header_path),
                "tail": driver_header_tail,
                "fields": driver_header_fields,
            },
            "driver_exit": {
                "exists": driver_exit_path.exists(),
                "code": driver_exit_code,
                "raw": driver_exit_raw,
            },
            "real_variant_training_log": {
                "exists": (export_runtime / "real_variant_training.log").exists(),
                "size": _file_size(export_runtime / "real_variant_training.log"),
                "tail": _tail_text(export_runtime / "real_variant_training.log"),
            },
            "upstream_checkpoint": {
                "exists": upstream.exists(),
                "tree": _count_tree(upstream),
                "latest_step": _latest_step(upstream),
            },
            "real_export": {
                "exists": real_export.exists(),
                "tree": _count_tree(real_export),
            },
            "output_tree": _count_tree(output_dir),
            "artifacts": {
                name: {
                    "exists": path.exists(),
                    "size": _file_size(path),
                    "lines": _line_count(path),
                }
                for name, path in traces.items()
            },
        }
        return cast(dict[str, object], snapshot)

    def _wrapper_preflight_failure(self, snapshot: dict[str, object]) -> str | None:
        wrapper_log = cast(dict[str, object], snapshot.get("wrapper_launch_log", {}))
        tail = str(wrapper_log.get("tail") or "")
        blocker_lines = [
            line.strip()
            for line in tail.splitlines()
            if "BLOCK_INFRA:" in line or "BLOCK_CONTRACT:" in line
        ]
        if not blocker_lines:
            return None
        return blocker_lines[-1]

    def _wrapper_abnormal_exit(self, snapshot: dict[str, object]) -> str | None:
        if snapshot.get("process_lines"):
            return None
        wrapper_log = cast(dict[str, object], snapshot.get("wrapper_launch_log", {}))
        if not wrapper_log.get("exists"):
            return None
        tail = str(wrapper_log.get("tail") or "").strip()
        if not tail:
            return None
        driver_exit = cast(dict[str, object], snapshot.get("driver_exit", {}))
        if driver_exit.get("exists"):
            return None
        art = cast(dict[str, dict[str, object]], snapshot.get("artifacts", {}))
        for artifact_name in (
            "training_failure.json",
            "training_timeout_report.json",
            "driver_timeout_report_fallback.json",
        ):
            if art.get(artifact_name, {}).get("exists"):
                return None
        last_line = tail.splitlines()[-1].strip()
        return last_line or "wrapper exited before driver.exit was written"

    def _failure_details(self, snapshot: dict[str, object]) -> tuple[str | None, str | None]:
        wrapper_preflight = self._wrapper_preflight_failure(snapshot)
        if wrapper_preflight is not None:
            return "wrapper_preflight", wrapper_preflight
        wrapper_abnormal = self._wrapper_abnormal_exit(snapshot)
        if wrapper_abnormal is not None:
            return "wrapper_exit", wrapper_abnormal
        driver_exit = cast(dict[str, object], snapshot.get("driver_exit", {}))
        driver_exit_code = driver_exit.get("code")
        if isinstance(driver_exit_code, int) and driver_exit_code != 0:
            return "driver_exit", f"exit_code={driver_exit_code}"
        art = cast(dict[str, dict[str, object]], snapshot.get("artifacts", {}))
        for artifact_name in (
            "training_failure.json",
            "training_timeout_report.json",
            "driver_timeout_report_fallback.json",
        ):
            artifact = art.get(artifact_name, {})
            if artifact.get("exists"):
                return "training_artifact", artifact_name
        return None, None

    def _active_no_progress_failure(self, snapshot: dict[str, object]) -> tuple[bool, dict[str, object]]:
        lane = str(snapshot.get("lane") or "")
        config = LANES[lane]
        grace_s = config.active_no_progress_grace_s
        if lane == "B" and self.b_active_no_progress_grace_s is not None:
            grace_s = self.b_active_no_progress_grace_s
        if grace_s is None or int(grace_s) <= 0:
            return False, {"enabled": False, "grace_s": grace_s}
        if not snapshot.get("process_lines"):
            return False, {"enabled": True, "grace_s": int(grace_s), "reason": "not_active"}
        upstream = cast(dict[str, object], snapshot["upstream_checkpoint"])
        artifacts = cast(dict[str, dict[str, object]], snapshot["artifacts"])
        latest_step = upstream.get("latest_step")
        loss_lines = artifacts["loss_decomposition.jsonl"].get("lines")
        if latest_step is not None or loss_lines:
            return False, {
                "enabled": True,
                "grace_s": int(grace_s),
                "reason": "progress_observed",
                "latest_step": latest_step,
                "loss_lines": loss_lines,
            }
        driver_header = cast(dict[str, object], snapshot.get("driver_header", {}))
        header_fields = cast(dict[str, object], driver_header.get("fields", {}))
        start_epoch = _parse_utc_epoch(header_fields.get("START_UTC"))
        start_source = "driver.header.START_UTC" if start_epoch is not None else None
        if start_epoch is None:
            runtime_dir = Path(str(snapshot["runtime_dir"]))
            if runtime_dir.exists():
                start_epoch = runtime_dir.stat().st_mtime
                start_source = "runtime_dir.mtime"
        if start_epoch is None:
            return False, {"enabled": True, "grace_s": int(grace_s), "reason": "start_time_unknown"}
        elapsed_s = max(0, int(time.time() - start_epoch))
        details = {
            "enabled": True,
            "grace_s": int(grace_s),
            "elapsed_s": elapsed_s,
            "start_source": start_source,
            "latest_step": latest_step,
            "loss_lines": loss_lines,
        }
        if elapsed_s < int(grace_s):
            details["reason"] = "within_grace"
            return False, details
        details["reason"] = "active_no_progress_grace_exceeded"
        return True, details

    def _should_autoheal(self, summary: dict[str, object]) -> bool:
        return bool(self.auto_fix) and summary.get("status") == "failed" and summary.get(
            "failure_source"
        ) != "wrapper_preflight"

    def _write_snapshot(self, snapshot: dict[str, object]) -> None:
        lane = str(snapshot["lane"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _atomic_write_json(self.snapshot_dir / f"{lane}_{ts}.json", snapshot)

    def _archive_output(self, lane: str, reason: str) -> tuple[Path, dict[str, int]]:
        output_dir = REPO_ROOT / LANES[lane].output_dir
        archive_dir = COLD_ROOT / f"stage1_v22_iter9_canonical_{lane}_supervisor_{reason}_{_now_local()}"
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        shutil.copytree(output_dir, archive_dir)
        return archive_dir, _count_tree(archive_dir)

    def _clear_output_dir(self, lane: str) -> None:
        output_dir = REPO_ROOT / LANES[lane].output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for child in list(output_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _seed_resume_checkpoint(self, lane: str, new_runtime_dir: Path) -> dict[str, object]:
        current_runtime = REPO_ROOT / self.lanes[lane].current_runtime_dir
        upstream = current_runtime / "real_variant_export_runtime" / "upstream_train_checkpoints" / "pi0_libero" / LANES[lane].variant_id
        latest = _latest_step(upstream)
        if latest is None:
            return {"used": False, "latest_step": None}
        src = upstream / str(latest)
        if lane == "B" and not (src / "train_state").is_dir():
            return {
                "used": False,
                "latest_step": latest,
                "source": str(src),
                "fallback_reason": "b_resume_seed_missing_train_state",
                "required_path": str(src / "train_state"),
            }
        dest = new_runtime_dir / "real_variant_export_runtime" / "upstream_train_checkpoints" / "pi0_libero" / LANES[lane].variant_id / str(latest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        return {
            "used": True,
            "latest_step": latest,
            "source": str(src),
            "dest": str(dest),
            "tree": _count_tree(dest),
        }

    def _launch_lane(self, lane: str, *, resume: bool) -> dict[str, object]:
        config = LANES[lane]
        ts = _now_local()
        runtime_root = REPO_ROOT / "agent" / "runtime_logs" / f"iter9_{lane}_supervisor_restart_{ts}"
        runtime_lane = runtime_root / lane
        runtime_root.mkdir(parents=True, exist_ok=True)
        seed_info = self._seed_resume_checkpoint(lane, runtime_lane) if resume else {"used": False, "latest_step": None}
        effective_resume = bool(resume and seed_info.get("used"))
        env = os.environ.copy()
        if effective_resume:
            env["ITER9_RESUME_REAL_EXPORT"] = "1"
        if config.training_timeout_s is not None:
            env["ITER9_TRAINING_TIMEOUT_S"] = str(int(config.training_timeout_s))
        if config.disk_min_gb is not None:
            env["ITER9_DISK_MIN_GB"] = str(int(config.disk_min_gb))
        if config.disk_warn_gb is not None:
            env["ITER9_DISK_WARN_GB"] = str(int(config.disk_warn_gb))
        launch_log = runtime_root / "wrapper.launch.log"
        with launch_log.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                ["bash", str(REPO_ROOT / config.launch_script), str(runtime_lane)],
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (runtime_root / "wrapper.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        self.lanes[lane].current_runtime_dir = str(runtime_lane.relative_to(REPO_ROOT))
        self.lanes[lane].restart_count += 1
        fallback_reason = str(seed_info.get("fallback_reason") or "").strip()
        if effective_resume:
            self.lanes[lane].last_action = "resume_restart"
        elif fallback_reason:
            self.lanes[lane].last_action = f"fresh_restart:{fallback_reason}"
        else:
            self.lanes[lane].last_action = "fresh_restart"
        self.save_state()
        return {
            "runtime_root": str(runtime_root),
            "runtime_lane": str(runtime_lane),
            "wrapper_pid": proc.pid,
            "resume_requested": bool(resume),
            "resume": effective_resume,
            "fallback_reason": fallback_reason or None,
            "last_action": self.lanes[lane].last_action,
            "seed": seed_info,
        }

    def _terminate_lane_processes(self, lane: str, snapshot: dict[str, object]) -> dict[str, object]:
        matched_pids = sorted(
            {
                pid
                for line in cast(list[str], snapshot.get("process_lines", []))
                for pid in [_process_line_pid(line)]
                if pid is not None and pid != os.getpid()
            }
        )
        if not matched_pids:
            return {"lane": lane, "matched_pids": [], "terminated": False}
        process_groups: set[int] = set()
        individual_pids: set[int] = set()
        current_pgid = os.getpgrp()
        for pid in matched_pids:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            if pgid == current_pgid:
                individual_pids.add(pid)
            else:
                process_groups.add(pgid)

        for pgid in sorted(process_groups):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for pid in sorted(individual_pids):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.time() + 30
        while time.time() < deadline:
            if all(not Path(f"/proc/{pid}").exists() for pid in matched_pids):
                break
            time.sleep(1)

        remaining = [pid for pid in matched_pids if Path(f"/proc/{pid}").exists()]
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        killed = [pid for pid in matched_pids if not Path(f"/proc/{pid}").exists()]
        return {
            "lane": lane,
            "matched_pids": matched_pids,
            "process_groups": sorted(process_groups),
            "individual_pids": sorted(individual_pids),
            "terminated": bool(killed),
            "terminated_pids": killed,
            "remaining_after_kill": [pid for pid in matched_pids if Path(f"/proc/{pid}").exists()],
        }

    def _write_action_markdown(self, lane: str, action: str, payload: dict[str, object]) -> None:
        ts = _now_local()
        log_path = REPO_ROOT / "agent" / "logs" / f"iter9_{lane}_supervisor_{action}_{ts}.md"
        evidence_path = REPO_ROOT / ".sisyphus" / "evidence" / f"iter9_{lane}_supervisor_{action}_{ts}.md"
        body = (
            f"# iter9_{lane}_supervisor_{action}_{ts}\n\n"
            f"- 时间：{_now_iso()}\n"
            f"- lane：{lane}\n"
            f"- 动作：{action}\n"
            f"- payload：\n\n```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```\n"
        )
        log_path.write_text(body, encoding="utf-8")
        evidence_path.write_text(body, encoding="utf-8")

    def _is_failure(self, snapshot: dict[str, object]) -> bool:
        procs = snapshot.get("process_lines", [])
        art = cast(dict[str, dict[str, object]], snapshot.get("artifacts", {}))
        failure = art.get("training_failure.json", {})
        timeout = art.get("training_timeout_report.json", {})
        driver_timeout = art.get("driver_timeout_report_fallback.json", {})
        driver_exit = cast(dict[str, object], snapshot.get("driver_exit", {}))
        driver_exit_code = driver_exit.get("code")
        wrapper_preflight = self._wrapper_preflight_failure(snapshot)
        wrapper_abnormal = self._wrapper_abnormal_exit(snapshot)
        return (not procs) and bool(
            wrapper_preflight
            or wrapper_abnormal
            or (isinstance(driver_exit_code, int) and driver_exit_code != 0)
            or failure.get("exists")
            or timeout.get("exists")
            or driver_timeout.get("exists")
        )

    def _status_summary(self, snapshot: dict[str, object], previous: dict[str, object] | None) -> dict[str, object]:
        upstream = cast(dict[str, object], snapshot["upstream_checkpoint"])
        artifacts = cast(dict[str, dict[str, object]], snapshot["artifacts"])
        latest_step = upstream.get("latest_step")
        loss_lines = artifacts["loss_decomposition.jsonl"].get("lines")
        status = "idle"
        if snapshot.get("process_lines"):
            status = "active"
        failure_source = None
        failure_message = None
        if self._is_failure(snapshot):
            status = "failed"
            failure_source, failure_message = self._failure_details(snapshot)
        active_no_progress_failed, active_no_progress = self._active_no_progress_failure(snapshot)
        if active_no_progress_failed:
            status = "failed"
            failure_source = "active_no_progress_stall"
            failure_message = (
                "active run exceeded no-progress grace; "
                f"elapsed_s={active_no_progress.get('elapsed_s')} "
                f"grace_s={active_no_progress.get('grace_s')}"
            )
        if previous is not None and status == "active":
            previous_latest_step = previous.get("latest_step")
            previous_loss_lines = previous.get("loss_lines")
            if latest_step is not None and latest_step != previous_latest_step:
                status = "progressing"
            elif loss_lines is not None and loss_lines != previous_loss_lines:
                status = "progressing"
        return {
            "lane": snapshot["lane"],
            "status": status,
            "latest_step": latest_step,
            "loss_lines": loss_lines,
            "runtime_dir": snapshot["runtime_dir"],
            "failure_source": failure_source,
            "failure_message": failure_message,
            "active_no_progress": active_no_progress,
        }

    def run_cycle(self) -> dict[str, object]:
        lane_entries: dict[str, dict[str, object]] = {}
        actions: list[dict[str, object]] = []
        previous_state: dict[str, dict[str, object]] = {}
        if self.status_path.exists():
            raw_previous = json.loads(self.status_path.read_text(encoding="utf-8")).get("lanes", {})
            if isinstance(raw_previous, dict):
                previous_state = cast(dict[str, dict[str, object]], raw_previous)
        for lane in LANES:
            snap = self._lane_snapshot(lane)
            prev = previous_state.get(lane)
            summary = self._status_summary(snap, prev)
            lane_entries[lane] = {"summary": summary, "snapshot": snap}
            self._write_snapshot(snap)
            if self._should_autoheal(summary):
                termination_payload = None
                if summary.get("failure_source") == "active_no_progress_stall":
                    termination_payload = self._terminate_lane_processes(lane, snap)
                archive_dir, archive_tree = self._archive_output(lane, "autoheal")
                self._clear_output_dir(lane)
                launch_payload = self._launch_lane(lane, resume=True)
                action_payload = {
                    "archive_dir": str(archive_dir),
                    "archive_tree": archive_tree,
                    "launch": launch_payload,
                    "failure_runtime": snap["runtime_dir"],
                    "termination": termination_payload,
                }
                actions.append({lane: action_payload})
                self._write_action_markdown(lane, "autoheal", action_payload)
        cycle: dict[str, object] = {
            "sample_time_local": _now_iso(),
            "lanes": lane_entries,
            "actions": actions,
        }
        _atomic_write_json(
            self.status_path,
            {
                "schema_version": "iter9_bcx_supervisor_status_v1",
                "generated_at_local": _now_iso(),
                "interval_s": self.interval_s,
                "root_dir": str(self.root_dir),
                "lanes": {lane: lane_entries[lane]["summary"] for lane in LANES},
                "actions": actions,
            },
        )
        self.save_state()
        return cycle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iter9_bcx_supervisor.py")
    parser.add_argument("--interval-s", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--no-auto-fix", action="store_true")
    parser.add_argument("--root-dir-name", default=None)
    parser.add_argument(
        "--b-active-no-progress-grace-s",
        type=int,
        default=None,
        help=(
            "Override B-only active stall grace in seconds. "
            "The default is the B lane config; C/X are unaffected."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    supervisor = Iter9BCXSupervisor(
        interval_s=int(args.interval_s),
        auto_fix=not bool(args.no_auto_fix),
        root_dir_name=args.root_dir_name,
        b_active_no_progress_grace_s=args.b_active_no_progress_grace_s,
    )
    cycle_count = 0
    while True:
        cycle = supervisor.run_cycle()
        print(json.dumps(cycle, indent=2, ensure_ascii=False))
        cycle_count += 1
        if bool(args.once):
            return 0
        if args.cycles is not None and cycle_count >= int(args.cycles):
            return 0
        time.sleep(max(int(args.interval_s), 1))


if __name__ == "__main__":
    raise SystemExit(main())
