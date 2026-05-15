"""R2 authentic evaluation runner.

Sanctioned subprocess sites (3):
  #1 _capture_nvidia_smi_pre_run  — environment fingerprint
  #2 _git_commit_sha              — reproducibility provenance (V3-FIX-5)
  #3 _capture_failure_context     — post-failure diagnostics (V4-FIX-2)

All pure helpers live in _envelope.py.

LOC-cap deviation (team-lead approved): this module exceeds the default 200-LOC
cap because the 24-field R2CellResult dataclass + skip-path JSON reconstruction
helper materially expand the surface. Pure helpers were already split to
``_envelope.py`` per V4-FIX-1/2/4/6. The accepted cap for this module is
documented in ``test_no_t8_calls.LOC_CAP_EXCEPTIONS``; the v4 plan §10 will be
amended in follow-up to record the deviation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from work.recap.r1_repro.protocol import EvalProtocol, P0B_PROTOCOL, protocol_deterministic_sha
from work.recap.r1_repro.protocol_swap import swap_single_axis
from work.recap.r1_repro.repro_runner import (
    ARTIFACT_ROOT,
    REPO_ROOT,
    ReproCellResult as ReproResult,
    _git_diff_clean_outside_artifact_dir as git_diff_clean_outside_artifact_dir,
    run_protocol as r1_repro_run,
    validate_baseline_pass_marker,
)
from work.recap.r2_authentic_eval._envelope import (
    _capture_library_versions,
    _ckpt_pre_run_sha256,
    _eval_skip_decision,
    _r1_0_latest_mtime_utc,
    _r2_invocation_envelope_sha256,
    _r2_module_set_content_sha256,
    _utc_now,
    _write_cell,
)
from work.recap.r2_authentic_eval.delta_stats import (
    R2_BASELINE_N_DEFAULT,
    R2_BASELINE_SUCC_DEFAULT,
    R2_CELL_RESULT_SCHEMA_VERSION,
    newcombe_delta_ci_95,
    wilson_ci_95,
)
from work.recap.r2_authentic_eval.inventory import TrainedCheckpoint


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class R2EvalError(RuntimeError):
    """Raised on R2 evaluation setup / validation failure."""


class R2SourceCkptDriftBetweenPhases(RuntimeError):
    """Raised when checkpoint changes between R2.1 and R2.2 (B-IND7)."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthenticEvalRequest:
    checkpoint: TrainedCheckpoint
    search_root: Path
    strict_config: bool


@dataclass(frozen=True)
class R2CellResult:
    request: AuthenticEvalRequest
    success_count: int
    completed_episode_total: int
    rate: float
    wilson_ci_95: tuple[float, float]
    delta_vs_baseline: float
    newcombe_delta_ci_95: tuple[float, float]
    artifact_dir: Path
    formal_eval_summary_json: dict[str, Any]
    raw_repro_result: Any
    ckpt_pre_run_sha256: dict[str, str]
    r1_0_dir_present: bool
    r1_0_baseline_repro_latest_run_mtime_utc: str | None
    git_commit_sha: str
    nvidia_smi_pre_run_csv: str
    transformers_version: str
    torch_version: str
    python_version: str
    gr00t_version: str | None
    protocol_sha256: str
    r2_invocation_envelope_sha256: str
    git_commit_sha_fallback_reason: str | None = None
    r2_cell_result_schema_version: str = R2_CELL_RESULT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Sanctioned subprocess #1
# ---------------------------------------------------------------------------


def _capture_nvidia_smi_pre_run(devices: str = "1") -> str:
    c = subprocess.run(
        ["nvidia-smi", "-i", devices,
         "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    return c.stdout.strip() if c.returncode == 0 else f"ERROR: {c.stderr.strip()}"


# ---------------------------------------------------------------------------
# Sanctioned subprocess #2 (V3-FIX-5)
# ---------------------------------------------------------------------------


def _git_commit_sha() -> tuple[str, str | None]:
    c = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15, check=False,
    )
    if c.returncode == 0:
        return c.stdout.strip(), None
    return "", f"rev-parse failed: {c.stderr.strip()!r}"


# ---------------------------------------------------------------------------
# Sanctioned subprocess #3 (V4-FIX-2)
# ---------------------------------------------------------------------------


def _capture_failure_context(
    *,
    source_ckpt_root: Path,
    raw_hf_snapshot_root: Path,
    swap_root: Path,
    preserve_path: Path,
    audit_failure_class: str,
) -> dict[str, Any]:
    """12-field failure context for post-run diagnostics."""
    git_sha, git_fallback = _git_commit_sha()
    libs = _capture_library_versions()
    nvsmi = _capture_nvidia_smi_pre_run(str(P0B_PROTOCOL.cuda_visible_devices))
    return {
        "audit_failure_class": audit_failure_class,
        "captured_at_utc": _utc_now(),
        "git_commit_sha": git_sha,
        "git_commit_sha_fallback_reason": git_fallback,
        "source_ckpt_root": str(source_ckpt_root),
        "raw_hf_snapshot_root": str(raw_hf_snapshot_root),
        "swap_root": str(swap_root),
        "preserve_path": str(preserve_path),
        "nvidia_smi_csv": nvsmi,
        "transformers_version": libs.get("transformers"),
        "torch_version": libs.get("torch"),
        "gr00t_version": libs.get("gr00t"),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_baseline_marker_valid(protocol: EvalProtocol) -> dict[str, Any]:
    try:
        return validate_baseline_pass_marker(protocol)
    except Exception as exc:
        raise R2EvalError(f"baseline marker invalid: {exc}") from exc


def _cell_from_json(d: dict[str, Any], req: AuthenticEvalRequest) -> R2CellResult:
    """Reconstruct R2CellResult from stored JSON (skip path; raw_repro_result=None)."""
    return R2CellResult(
        request=req,
        success_count=int(d["success_count"]),
        completed_episode_total=int(d["completed_episode_total"]),
        rate=float(d["rate"]),
        wilson_ci_95=tuple(d["wilson_ci_95"]),
        delta_vs_baseline=float(d["delta_vs_baseline"]),
        newcombe_delta_ci_95=tuple(d["newcombe_delta_ci_95"]),
        artifact_dir=Path(d["artifact_dir"]),
        formal_eval_summary_json=dict(d.get("formal_eval_summary_json") or {}),
        raw_repro_result=None,
        ckpt_pre_run_sha256=dict(d.get("ckpt_pre_run_sha256") or {}),
        r1_0_dir_present=bool(d.get("r1_0_dir_present", False)),
        r1_0_baseline_repro_latest_run_mtime_utc=d.get("r1_0_baseline_repro_latest_run_mtime_utc"),
        git_commit_sha=str(d.get("git_commit_sha", "")),
        nvidia_smi_pre_run_csv=str(d.get("nvidia_smi_pre_run_csv", "")),
        transformers_version=str(d.get("transformers_version", "")),
        torch_version=str(d.get("torch_version", "")),
        python_version=str(d.get("python_version", "")),
        gr00t_version=d.get("gr00t_version"),
        protocol_sha256=str(d["protocol_sha256"]),
        r2_invocation_envelope_sha256=str(d["r2_invocation_envelope_sha256"]),
        git_commit_sha_fallback_reason=d.get("git_commit_sha_fallback_reason"),
        r2_cell_result_schema_version=str(
            d.get("r2_cell_result_schema_version", R2_CELL_RESULT_SCHEMA_VERSION)
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_r2_1_cell(req: AuthenticEvalRequest, out_dir: Path) -> R2CellResult:
    """Run one R2.1 eval cell (real GPU run, single ckpt_root swap from P0B_PROTOCOL)."""
    if not req.checkpoint.is_valid:
        raise R2EvalError(f"checkpoint not valid: {req.checkpoint.invalid_reason}")
    if not req.checkpoint.statistics_q99_matches_base:
        raise R2EvalError("checkpoint q99 does not match base fingerprint")

    raw_hf = P0B_PROTOCOL.ckpt_root
    mod_sha = _r2_module_set_content_sha256()
    inv_params: dict[str, Any] = {
        "search_root": str(req.search_root.resolve()),
        "strict_config": bool(req.strict_config),
        "raw_hf_snapshot": str(raw_hf.resolve()),
        "r2_module_set_content_sha": mod_sha,
    }
    protocol = swap_single_axis(P0B_PROTOCOL, "ckpt_root", req.checkpoint.abs_path, name="r2_1")
    proto_sha = protocol_deterministic_sha(protocol)
    env_sha = _r2_invocation_envelope_sha256(
        search_root=req.search_root,
        strict_config=req.strict_config,
        raw_hf_snapshot=raw_hf,
        r2_module_set_content_sha=mod_sha,
    )

    cell_json = out_dir / "cell_result.json"
    if cell_json.is_file():
        decision = _eval_skip_decision(cell_json, proto_sha, env_sha, inv_params)
        if decision["decided"]:
            return _cell_from_json(json.loads(cell_json.read_text(encoding="utf-8")), req)

    _assert_baseline_marker_valid(P0B_PROTOCOL)
    git_sha, git_fallback = _git_commit_sha()
    nvsmi = _capture_nvidia_smi_pre_run(str(protocol.cuda_visible_devices))
    libs = _capture_library_versions()
    pre_sha = _ckpt_pre_run_sha256(req.checkpoint.abs_path)

    repro: ReproResult = r1_repro_run(protocol, out_dir)
    n, total = repro.success_count, int(getattr(protocol, "episodes", 30))
    formal_json: dict[str, Any] = {}
    try:
        fp = out_dir / "formal_eval_summary.json"
        if fp.is_file():
            formal_json = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        pass

    cell = R2CellResult(
        request=req,
        success_count=n,
        completed_episode_total=total,
        rate=n / max(1, total),
        wilson_ci_95=wilson_ci_95(n, total),
        delta_vs_baseline=n / total - R2_BASELINE_SUCC_DEFAULT / R2_BASELINE_N_DEFAULT,
        newcombe_delta_ci_95=newcombe_delta_ci_95(n, R2_BASELINE_SUCC_DEFAULT, total, R2_BASELINE_N_DEFAULT),
        artifact_dir=out_dir,
        formal_eval_summary_json=formal_json,
        raw_repro_result=repro,
        ckpt_pre_run_sha256=pre_sha,
        r1_0_dir_present=(ARTIFACT_ROOT / "r1_0").is_dir(),
        r1_0_baseline_repro_latest_run_mtime_utc=_r1_0_latest_mtime_utc(ARTIFACT_ROOT / "r1_0"),
        git_commit_sha=git_sha,
        nvidia_smi_pre_run_csv=nvsmi,
        transformers_version=str(libs.get("transformers") or ""),
        torch_version=str(libs.get("torch") or ""),
        python_version=sys.version,
        gr00t_version=libs.get("gr00t"),
        protocol_sha256=proto_sha,
        r2_invocation_envelope_sha256=env_sha,
        git_commit_sha_fallback_reason=git_fallback,
    )
    _write_cell(cell, inv_params)
    return cell


def run_r2_2_swap_cell(r2_1_result: R2CellResult, out_dir: Path) -> R2CellResult:
    """Run R2.2 swap cell; re-check source ckpt sha for B-IND7 drift first."""
    req = r2_1_result.request
    post_sha = _ckpt_pre_run_sha256(req.checkpoint.abs_path)
    if post_sha != r2_1_result.ckpt_pre_run_sha256:
        raise R2SourceCkptDriftBetweenPhases(
            f"checkpoint sha drift between R2.1 and R2.2: {req.checkpoint.abs_path}"
        )
    return run_r2_1_cell(req, out_dir)


__all__ = [
    "AuthenticEvalRequest",
    "R2CellResult",
    "R2EvalError",
    "R2SourceCkptDriftBetweenPhases",
    "git_diff_clean_outside_artifact_dir",
    "run_r2_1_cell",
    "run_r2_2_swap_cell",
]
