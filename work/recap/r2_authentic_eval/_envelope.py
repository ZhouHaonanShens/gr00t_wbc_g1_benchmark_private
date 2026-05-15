"""Envelope helpers — pure (no subprocess, no GPU).

Provides: library version capture, R2 module-set content sha, invocation envelope sha,
checkpoint pre-run hash (B-IND7), UTC timestamp, eval skip decision (9-reason rule),
cell-JSON write helper, and r1_0 mtime probe.
All subprocess-using helpers live in eval_runner.py (sanctioned sites only).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Library version capture
# ---------------------------------------------------------------------------


def _capture_library_versions() -> dict[str, str | None]:
    """Best-effort version capture for reproducibility envelope.

    Returns dict with keys 'transformers', 'torch', 'gr00t'.
    On PackageNotFoundError or other import error, value is None.
    """
    result: dict[str, str | None] = {}
    for module in ("transformers", "torch", "gr00t"):
        try:
            result[module] = importlib.metadata.version(module)
        except Exception:
            result[module] = None
    return result


# ---------------------------------------------------------------------------
# File hash helper
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file via 1 MiB-chunked read."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint pre-run hash (B-IND7)
# ---------------------------------------------------------------------------


def _ckpt_pre_run_sha256(ckpt_root: Path) -> dict[str, str]:
    """Hash index.json and first .safetensors shard for B-IND7 drift detection."""
    result: dict[str, str] = {}
    index = ckpt_root / "index.json"
    if index.is_file():
        result["index.json"] = _sha256_file(index)
    shards = sorted(ckpt_root.glob("*.safetensors"))
    if shards:
        result[shards[0].name] = _sha256_file(shards[0])
    return result


# ---------------------------------------------------------------------------
# UTC timestamp
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string (Z suffix, no microseconds)."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# R2 module-set content sha (V4-FIX-4)
# ---------------------------------------------------------------------------


def _r2_module_set_content_sha256() -> str:
    """Deterministic content hash over every .py file under work/recap/r2_authentic_eval/.

    NOT a git operation; uses file-content hash to work in non-git contexts (CI tarballs, etc.).
    Walks the R2 module dir recursively via ``os.walk(followlinks=False)``; sorts by relative
    path; concatenates each file's bytes prefixed with its relpath; returns sha256 hex digest.

    **Symlink discipline (V4-FIX-4):** ``followlinks=False``. Symlinks are recorded by their
    **link metadata** (target path), not by the bytes of the target. Rationale: the R2 module
    set defines R2's behavior; if a `.py` file is symlinked, the symlink itself is the deployable
    artifact. This is asymmetric to ``_walk_recursive_sha_table`` in ``ckpt_config_swap.py``
    (which follows symlinks per V3-FIX-4): the swap-mechanic helper hashes byte-identity; this
    helper hashes deployment-identity. The asymmetry is intentional.
    """
    r2_dir = Path(__file__).parent
    entries: list[tuple[str, Path]] = []
    for dirpath, _, filenames in os.walk(str(r2_dir), followlinks=False):
        for fn in filenames:
            if fn.endswith(".py"):
                full = Path(dirpath) / fn
                rel = str(full.relative_to(r2_dir))
                entries.append((rel, full))
    digest = hashlib.sha256()
    for rel, full in sorted(entries):
        digest.update(rel.encode("utf-8"))
        digest.update(full.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# R2 invocation envelope sha (V3-FIX-2, V4-FIX-6)
# ---------------------------------------------------------------------------


def _r2_invocation_envelope_sha256(
    *,
    search_root: Path,
    strict_config: bool,
    raw_hf_snapshot: Path,
    r2_module_set_content_sha: str,
) -> str:
    """Hash over the R2-side invocation envelope.

    Distinct from ``protocol_sha256`` (which fingerprints EvalProtocol identity).
    Captures: which search root R2 walked, whether --strict-config was on,
    which raw HF snapshot anchored the swap, which R2 module-set content produced
    the cells.

    The ``r2_module_set_content_sha`` argument is conventionally produced by
    ``_r2_module_set_content_sha256()``. Renamed v4 (V4-FIX-6); the legacy
    parameter name (which embedded ``_git_`` rather than ``_content_``) is
    forbidden anywhere in this package by the test_no_t8_calls grep guard.
    """
    payload = json.dumps(
        {
            "search_root": str(search_root.resolve()),
            "strict_config": bool(strict_config),
            "raw_hf_snapshot": str(raw_hf_snapshot.resolve()),
            "r2_module_set_content_sha": str(r2_module_set_content_sha),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Eval skip decision (9-reason rule, V3-FIX-2, V4-FIX-3)
# ---------------------------------------------------------------------------


def _eval_skip_decision(
    cell_json_path: Path,
    proto_sha: str,
    env_sha: str,
    inv_params: dict[str, Any],
) -> dict[str, Any]:
    """Read stored cell JSON and decide whether to skip re-running (9-reason rule).

    Returns {'decided': bool, 'reason': str}.
    decided=True → skip (prior result valid); decided=False → must re-run.

    V4-FIX-3: missing ``r2_invocation_envelope_sha256`` in pre-v3 cells →
    ``envelope_mismatch_no_envelope_recorded`` (graceful, skip=False).
    """
    existing: dict[str, Any] = json.loads(cell_json_path.read_text(encoding="utf-8"))
    if existing.get("protocol_sha256") != proto_sha:
        return {"decided": False, "reason": "protocol_sha256_mismatch"}
    existing_env = existing.get("r2_invocation_envelope_sha256", "")
    if not existing_env:
        return {"decided": False, "reason": "envelope_mismatch_no_envelope_recorded"}
    if existing_env != env_sha:
        stored = existing.get("_r2_invocation_params", {})
        if stored.get("search_root") != inv_params.get("search_root"):
            return {"decided": False, "reason": "envelope_mismatch_search_root"}
        if stored.get("strict_config") != inv_params.get("strict_config"):
            return {"decided": False, "reason": "envelope_mismatch_strict_config"}
        if stored.get("raw_hf_snapshot") != inv_params.get("raw_hf_snapshot"):
            return {"decided": False, "reason": "envelope_mismatch_raw_hf_snapshot"}
        return {"decided": False, "reason": "envelope_mismatch_module_set_content_sha"}
    fstatus = existing.get("formal_eval_summary_json", {}).get("status", "")
    if fstatus != "PASS":
        return {"decided": False, "reason": "formal_eval_status_not_pass"}
    if existing.get("completed_episode_total", 0) != 30:
        return {"decided": False, "reason": "episode_count_not_30"}
    return {"decided": True, "reason": "all_match"}


# ---------------------------------------------------------------------------
# Cell-JSON write helper (B-IND3 sentinel; pure — no subprocess, no GPU)
# ---------------------------------------------------------------------------


def _write_cell(cell: Any, inv_params: dict[str, Any]) -> None:
    """Write cell JSON with _r2_invocation_params alongside (B-IND3 sentinel)."""
    out: Path = cell.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dataclasses.asdict(cell)
    payload["_r2_invocation_params"] = inv_params
    (out / "cell_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# R1.0 latest-run mtime probe (pure — file-stat only, no subprocess)
# ---------------------------------------------------------------------------


def _r1_0_latest_mtime_utc(r1_0_root: Path) -> str | None:
    """Return mtime of the most recent r1_0 run dir as UTC ISO-8601, or None."""
    if not r1_0_root.is_dir():
        return None
    dirs = [p for p in r1_0_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return (
        dt.datetime.fromtimestamp(latest.stat().st_mtime, tz=dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
