#!/usr/bin/env python3
"""GR00T dual-track blocker push orchestration.

This script implements the GR00T/GPU1 slice of the dual-track plan:
formal comparability repair + P4 gate refresh, plus an exploratory-only
seed-wise signal artifact. It runs P5 only through the hard P4 formal gate
and never lets exploratory signals unlock formal status.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap import finetune_full  # noqa: E402

FORMAL_SCHEMA = "dual_track_formal_status_v1"
EXPLORATORY_SCHEMA = "dual_track_exploratory_signal_v1"
AUTHORITY_REL = Path("agent/artifacts/recap_min_loop/single_gpu_v2_full_update")
BASELINE_REL = Path("agent/artifacts/recap_min_loop/single_gpu_v1")
CONDITIONED_REL = AUTHORITY_REL / "t13_advantage_full_update_1gpu/formal_run"
CONTINUATION_REL = AUTHORITY_REL / "t13_continuation_full_update_1gpu/formal_run"
P4_REL = AUTHORITY_REL / "p4_loss_action_subgoal"
P4_TIMEOUT_SECONDS = 60 * 60
P5_TIMEOUT_SECONDS = 2 * 60 * 60
P5_SEED_START = 20260421
P5_SEED_END = 20260430


def _utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_state(path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "is_file": path.is_file(),
        "size_bytes": None if stat is None else stat.st_size,
        "mtime_ns": None if stat is None else stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _worker_name() -> str:
    raw = os.environ.get("OMX_TEAM_WORKER") or os.environ.get("OMX_WORKER") or ""
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return raw or "worker-2"


def _sudo_used(cmd: Sequence[str]) -> bool:
    return any(Path(str(part)).name == "sudo" for part in cmd)


def _forbidden_gpus_visible(cuda_visible_devices: str | None) -> bool:
    if cuda_visible_devices is None:
        return True
    tokens = [token.strip() for token in str(cuda_visible_devices).split(",") if token.strip()]
    return any(token in {"0", "3"} for token in tokens)


def _timeout_wrapped_cmd(cmd: Sequence[str], timeout_seconds: int) -> list[str]:
    if shutil.which("timeout") is None:
        return list(cmd)
    return ["timeout", f"{int(timeout_seconds)}s", *list(cmd)]


def _run(
    cmd: Sequence[str],
    *,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int = P4_TIMEOUT_SECONDS,
    lease_path: Path | None = None,
    artifacts: Sequence[Path] = (),
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    start_time = _iso_now()
    cuda_visible_devices = merged_env.get("CUDA_VISIBLE_DEVICES")
    effective_cmd = _timeout_wrapped_cmd(cmd, timeout_seconds)
    timed_out = False
    proc = subprocess.run(
        effective_cmd,
        cwd=str(REPO_ROOT),
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=None if effective_cmd[0] == "timeout" else timeout_seconds,
    )
    end_time = _iso_now()
    timed_out = proc.returncode == 124 and effective_cmd[0] == "timeout"
    header = "\n".join(
        [
            f"start_time={start_time}",
            f"end_time={end_time}",
            f"command={shlex.join(effective_cmd)}",
            f"raw_command={shlex.join(list(cmd))}",
            f"timeout_seconds={int(timeout_seconds)}",
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices or ''}",
            f"returncode={proc.returncode}",
            f"timed_out={str(timed_out).lower()}",
            "--- stdout/stderr ---",
            "",
        ]
    )
    log_path.write_text(header + proc.stdout, encoding="utf-8")
    lease = {
        "schema_version": "resource_lease_v1",
        "lane": "gr00t",
        "gpu": 1,
        "worker": _worker_name(),
        "command": effective_cmd,
        "raw_command": list(cmd),
        "command_shell": shlex.join(effective_cmd),
        "start_time": start_time,
        "end_time": end_time,
        "returncode": proc.returncode,
        "timeout_seconds": int(timeout_seconds),
        "timed_out": timed_out,
        "runtime_log": str(log_path),
        "artifacts": [str(path) for path in artifacts],
        "env": {"CUDA_VISIBLE_DEVICES": cuda_visible_devices},
        "forbidden_gpus_visible": _forbidden_gpus_visible(cuda_visible_devices),
        "sudo_used": _sudo_used(effective_cmd),
    }
    if lease_path is not None:
        _write_json(lease_path, lease)
    return {
        "cmd": effective_cmd,
        "raw_cmd": list(cmd),
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "lease_path": None if lease_path is None else str(lease_path),
        "lease": lease,
        "stdout_tail": (header + proc.stdout)[-4000:],
    }


def _nvidia_snapshot(gpu_index: int) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    if shutil.which("nvidia-smi") is None:
        return {"gpu_index": gpu_index, "ok": False, "reason": "nvidia-smi_not_found", "cmd": cmd}
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "gpu_index": gpu_index,
        "ok": proc.returncode == 0,
        "cmd": cmd,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def _du(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    proc = subprocess.run(["du", "-sh", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"path": str(path), "exists": True, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def _emit_conditioned_manifest_if_needed(*, artifact_repo_root: Path, audit_dir: Path) -> dict[str, Any]:
    conditioned_dir = artifact_repo_root / CONDITIONED_REL
    continuation_manifest_path = artifact_repo_root / CONTINUATION_REL / finetune_full.COMPARABILITY_MANIFEST_FILENAME
    conditioned_manifest_path = conditioned_dir / finetune_full.COMPARABILITY_MANIFEST_FILENAME
    pre_state = {
        "conditioned_manifest": _file_state(conditioned_manifest_path),
        "continuation_manifest": _file_state(continuation_manifest_path),
    }
    _write_json(audit_dir / "pre_manifest_state.json", pre_state)
    if conditioned_manifest_path.is_file():
        conditioned_manifest = _read_json(conditioned_manifest_path)
        action = "existing_conditioned_manifest_reused"
    else:
        continuation = _read_json(continuation_manifest_path)
        conditioned_manifest = finetune_full.emit_conditioned_formal_lane_comparability_manifest(
            repo_root=artifact_repo_root,
            output_dir=conditioned_dir,
            warm_start_checkpoint=continuation.get("warm_start_checkpoint"),
            global_batch_size=int(continuation["batch_geometry"]["global_batch_size"]),
            gradient_accumulation_steps=int(continuation["batch_geometry"]["gradient_accumulation_steps"]),
            num_gpus=int(continuation["batch_geometry"].get("num_gpus", 1)),
            dataset_fingerprint=continuation.get("dataset_fingerprint"),
            train_scope_requested=continuation.get("train_scope_requested"),
            train_scope_effective=continuation.get("train_scope_effective"),
            seed_bundle_path=continuation.get("seed_set_source_path"),
            policy_route=continuation.get("policy_route"),
            policy_indicator_mode=continuation.get("policy_route_freeze", {}).get("indicator_mode")
            if isinstance(continuation.get("policy_route_freeze"), Mapping)
            else None,
            launch_family=continuation.get("launch_family"),
        )
        action = "conditioned_manifest_emitted_with_finetune_full_builder"
    continuation_manifest = _read_json(continuation_manifest_path)
    validation = finetune_full.validate_full_update_comparability_manifests(conditioned_manifest, continuation_manifest)
    post_state = {
        "conditioned_manifest": _file_state(conditioned_manifest_path),
        "continuation_manifest": _file_state(continuation_manifest_path),
    }
    _write_json(audit_dir / "post_manifest_state.json", post_state)
    return {
        "action": action,
        "conditioned_manifest_path": str(conditioned_manifest_path),
        "continuation_manifest_path": str(continuation_manifest_path),
        "validation": validation,
        "pre_state_path": str(audit_dir / "pre_manifest_state.json"),
        "post_state_path": str(audit_dir / "post_manifest_state.json"),
    }


def _formal_status(
    *,
    lane_root: Path,
    p4_summary_path: Path,
    manifest_repair: Mapping[str, Any],
    p5_verdict_path: Path | None = None,
    p5_refresh: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    p4 = _read_json(p4_summary_path)
    p4_blockers = list(dict.fromkeys(str(x) for x in p4.get("blocking_reasons", []) if str(x)))
    if str(p4.get("status")).upper() != "PASS" and not p4_blockers:
        p4_blockers.append(f"p4_summary_status_{str(p4.get('status') or 'missing').lower()}")
    if str(p4.get("status")).upper() == "PASS" and p4.get("formal_claim_allowed") is not True:
        p4_blockers.append("p4_formal_claim_not_allowed")
    if not p4_blockers and p4.get("p5_formal_10ep_eligible") is not True:
        p4_blockers.append("p5_formal_10ep_ineligible")
    p4_blockers = list(dict.fromkeys(p4_blockers))

    p5_verdict: dict[str, Any] | None = None
    if p5_verdict_path is not None and p5_verdict_path.is_file():
        p5_verdict = _read_json(p5_verdict_path)
    p5_blockers: list[str] = []
    if not p4_blockers:
        if p5_verdict is None:
            p5_blockers.append("p5_gate_verdict_missing")
        elif str(p5_verdict.get("status")).upper() != "PASS":
            p5_blockers.extend(str(x) for x in p5_verdict.get("blocking_reasons", []) if str(x))
            blocker_reason = p5_verdict.get("blocker_reason")
            if isinstance(blocker_reason, str) and blocker_reason:
                p5_blockers.append(blocker_reason)
            if not p5_blockers:
                p5_blockers.append(f"p5_gate_status_{str(p5_verdict.get('status') or 'missing').lower()}")
    blockers = list(dict.fromkeys([*p4_blockers, *p5_blockers]))
    status = "PASS" if not blockers else "BLOCK"
    p5_formal_execution_attempted = bool(
        isinstance(p5_verdict, Mapping) and p5_verdict.get("formal_execution_attempted") is True
    )
    authority_inputs = [
        str(p4_summary_path),
        str(lane_root / "manifest_audit/pre_manifest_state.json"),
        str(lane_root / "manifest_audit/post_manifest_state.json"),
    ]
    if p5_verdict_path is not None:
        authority_inputs.append(str(p5_verdict_path))
    validator_outputs: list[Any] = [manifest_repair]
    if p5_refresh is not None:
        validator_outputs.append(dict(p5_refresh))
    return {
        "schema_version": FORMAL_SCHEMA,
        "lane": "gr00t",
        "track": "formal",
        "status": status,
        "formal_claim_allowed": status == "PASS",
        "blocking_reasons": [] if status == "PASS" else blockers,
        "authority_inputs": authority_inputs,
        "validator_outputs": validator_outputs,
        "entered_next_gate": p5_formal_execution_attempted,
        "next_gate_allowed": status == "PASS",
        "p4_status": p4.get("status"),
        "p4_blocking_reasons": list(p4.get("blocking_reasons") or []),
        "p5_status": None if p5_verdict is None else p5_verdict.get("status"),
        "p5_gate_mode": None if p5_verdict is None else p5_verdict.get("gate_mode"),
        "p5_formal_execution_attempted": p5_formal_execution_attempted,
        "notes": "GR00T formal status requires a clean P4 refresh and a gated P5 verdict; exploratory signals cannot unlock it.",
    }


def _exploratory_signal(*, subgoal_path: Path) -> dict[str, Any]:
    subgoal = _read_json(subgoal_path)
    pairs = list(subgoal.get("per_seed_pairs") or [])
    positive = [p for p in pairs if float(p.get("relative_improvement_min_dist_ee_to_apple", 0.0)) > 0.0]
    observed = {
        "source_subgoal_summary_path": str(subgoal_path),
        "selected_seeds": subgoal.get("selected_seeds", []),
        "positive_seed_count": len(positive),
        "paired_seed_improvement_count": subgoal.get("paired_seed_improvement_count"),
        "required_formal_improvement_count": 2,
        "mean_relative_improvement_min_dist_ee_to_apple": subgoal.get("mean_relative_improvement_min_dist_ee_to_apple"),
        "no_regression_on_contact_or_lift_proxy": subgoal.get("no_regression_on_contact_or_lift_proxy"),
        "seed_delta_table": pairs,
    }
    return {
        "schema_version": EXPLORATORY_SCHEMA,
        "lane": "gr00t",
        "track": "exploratory",
        "status": "SIGNAL" if positive else "NO_SIGNAL",
        "exploratory_only": True,
        "formal_claim_allowed": False,
        "must_not_unlock_formal_gate": True,
        "method": "other",
        "risk_label": "exploratory_not_formal",
        "inputs": [str(subgoal_path)],
        "outputs": [],
        "observed_signal": observed,
        "notes": "Seed-wise exploratory signal only; it cannot satisfy the formal 2-of-3 paired-seed gate or P5 eligibility.",
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-repo-root", default="/home/howard/Projects/gr00t_wbc_g1_benchmark")
    parser.add_argument("--timestamp", default=_utc_ts())
    parser.add_argument("--python", default="/home/howard/Projects/gr00t_wbc_g1_benchmark/.venv/bin/python")
    parser.add_argument("--p4-timeout-seconds", type=int, default=P4_TIMEOUT_SECONDS)
    parser.add_argument("--p5-timeout-seconds", type=int, default=P5_TIMEOUT_SECONDS)
    parser.add_argument("--p5-seed-start", type=int, default=P5_SEED_START)
    parser.add_argument("--p5-seed-end", type=int, default=P5_SEED_END)
    args = parser.parse_args(argv)

    artifact_repo_root = Path(args.artifact_repo_root).resolve()
    ts = args.timestamp
    lane_root = artifact_repo_root / AUTHORITY_REL / f"dual_track_{ts}" / "gr00t"
    runtime_log_dir = artifact_repo_root / "agent/runtime_logs" / f"gr00t_openpi_dual_track_{ts}" / "gr00t"
    audit_dir = lane_root / "manifest_audit"
    lane_root.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)

    resource_snapshot = {
        "schema_version": "gr00t_dual_track_resource_snapshot_v1",
        "lane": "gr00t",
        "gpu_boundary": "GPU1 only; CUDA_VISIBLE_DEVICES=1 for gate refresh command; GPU0/GPU3 not targeted",
        "sudo_boundary": "agent did not run sudo",
        "gpu1": _nvidia_snapshot(1),
        "disk": {
            "repo": _du(artifact_repo_root),
            "gr00t_authority_root": _du(artifact_repo_root / AUTHORITY_REL),
            "lerobot_datasets": _du(artifact_repo_root / "agent/artifacts/lerobot_datasets"),
        },
    }
    _write_json(lane_root / "resource_snapshot.json", resource_snapshot)

    manifest_repair = _emit_conditioned_manifest_if_needed(artifact_repo_root=artifact_repo_root, audit_dir=audit_dir)
    refresh_dir = lane_root / "p4_loss_action_subgoal"
    refresh = _run(
        [
            args.python,
            str(REPO_ROOT / "work/recap/scripts/35a_full_update_rollout_probe.py"),
            "--mode", "p4",
            "--baseline-authority-root", str(artifact_repo_root / BASELINE_REL),
            "--v2-authority-root", str(artifact_repo_root / AUTHORITY_REL),
            "--conditioned-run-root", str(artifact_repo_root / AUTHORITY_REL / "t13_advantage_full_update_1gpu"),
            "--continuation-run-root", str(artifact_repo_root / AUTHORITY_REL / "t13_continuation_full_update_1gpu"),
            "--output-dir", str(refresh_dir),
            "--baseline-v1-subgoal-override",
            str(artifact_repo_root / AUTHORITY_REL / "p5_gate_eval/baseline_first_subgoal_probe_v1.json"),
        ],
        log_path=runtime_log_dir / "35a_p4_gate_refresh.log",
        lease_path=lane_root / "resource_lease_p4.json",
        artifacts=[refresh_dir],
        env={"CUDA_VISIBLE_DEVICES": "1", "NO_ALBUMENTATIONS_UPDATE": "1"},
        timeout_seconds=int(args.p4_timeout_seconds),
    )
    if refresh["returncode"] != 0:
        raise SystemExit(json.dumps({"error": "35a_refresh_failed", "refresh": refresh}, indent=2))

    p4_summary_path = refresh_dir / "full_update_diagnostic_summary.json"
    p5_dir = lane_root / "p5_gate"
    p5_refresh = _run(
        [
            args.python,
            str(REPO_ROOT / "work/recap/scripts/35a_full_update_rollout_probe.py"),
            "--mode", "p5_gate",
            "--baseline-authority-root", str(artifact_repo_root / BASELINE_REL),
            "--v2-authority-root", str(lane_root),
            "--conditioned-run-root", str(artifact_repo_root / AUTHORITY_REL / "t13_advantage_full_update_1gpu"),
            "--continuation-run-root", str(artifact_repo_root / AUTHORITY_REL / "t13_continuation_full_update_1gpu"),
            "--output-dir", str(p5_dir),
            "--seed-start", str(int(args.p5_seed_start)),
            "--seed-end", str(int(args.p5_seed_end)),
        ],
        log_path=runtime_log_dir / "35a_p5_gate.log",
        lease_path=lane_root / "resource_lease_p5.json",
        artifacts=[p5_dir],
        env={"CUDA_VISIBLE_DEVICES": "1", "NO_ALBUMENTATIONS_UPDATE": "1"},
        timeout_seconds=int(args.p5_timeout_seconds),
    )
    if p5_refresh["returncode"] != 0:
        raise SystemExit(json.dumps({"error": "35a_p5_gate_failed", "p5_refresh": p5_refresh}, indent=2))

    p5_verdict_path = p5_dir / "min_loop_verdict.json"
    formal = _formal_status(
        lane_root=lane_root,
        p4_summary_path=p4_summary_path,
        manifest_repair=manifest_repair,
        p5_verdict_path=p5_verdict_path,
        p5_refresh=p5_refresh,
    )
    exploratory = _exploratory_signal(subgoal_path=refresh_dir / "subgoal_summary_3seed.json")
    _write_json(lane_root / "formal_status.json", formal)
    _write_json(lane_root / "exploratory_signal.json", exploratory)
    result = {
        "schema_version": "gr00t_dual_track_blocker_push_result_v1",
        "lane_root": str(lane_root),
        "runtime_log_dir": str(runtime_log_dir),
        "resource_snapshot_path": str(lane_root / "resource_snapshot.json"),
        "manifest_repair": manifest_repair,
        "refresh": refresh,
        "p5_refresh": p5_refresh,
        "formal_status_path": str(lane_root / "formal_status.json"),
        "exploratory_signal_path": str(lane_root / "exploratory_signal.json"),
        "p5_verdict_path": str(p5_verdict_path),
        "formal_status": formal["status"],
        "exploratory_status": exploratory["status"],
    }
    _write_json(lane_root / "run_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
