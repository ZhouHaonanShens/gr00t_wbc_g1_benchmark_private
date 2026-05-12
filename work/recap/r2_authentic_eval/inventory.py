"""R2.0 inventory: discover and classify RECAP-trained checkpoint candidates."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module-top constants (SCREAMING_CASE; no inline magic numbers)
# ---------------------------------------------------------------------------

REQUIRED_CKPT_FILES: tuple[str, ...] = (
    "config.json",
    "processor_config.json",
    "statistics.json",
)

BASE_RIGHT_HAND_Q99: tuple[float, ...] = (1.5, 1.5, 1.0, 1.5, 0.0, 0.0, 0.0)
"""Expected unitree_g1/action/right_hand/q99 fingerprint for RECAP-trained G1 checkpoints."""

DEFAULT_SEARCH_ROOT: Path = Path("agent/artifacts/gr00t_recap_live").resolve()
"""Root directory under which R2.0 scans for trained checkpoints."""

RECAP_PATH_INCLUDE_HINTS: tuple[str, ...] = ("recap",)
"""Forward-compat substring hints. NON-AUTHORITATIVE: classification is determined
by the RECAP_NEGATIVE_TOKENS blocklist + q99 fingerprint, not by these hints.
Preserved for logging/grep reference only; do NOT use as the classification signal."""

RECAP_NEGATIVE_TOKENS: tuple[str, ...] = (
    "pure_sft",
    "safe_sft",
    "safe_lora",
    "wi_rescue",
    "interpolate",
    "hf_patches/",
    "_sanity_check",
    "smoke",
    "throughput_probe",
    "preflight",
)
"""Negative-token blocklist (Choice P-A + B-IND8).

A checkpoint whose absolute path contains any of these substrings receives
label='OTHER'. The token ``_sanity_check`` (NOT bare ``sanity``) is intentional
per B-IND8: avoids false-rejecting ``g3_conditioned_continuation_after_sanity_*``
directories while still blocking synthetic sanity-check runs."""

Q99_TOLERANCE: float = 1e-6
"""Max per-dimension absolute deviation when comparing against BASE_RIGHT_HAND_Q99."""

R2_VALID_CELL_COUNT_EXPECTED: int = 5
"""Empirical bound (May 2026): R2.0 finds exactly 5 valid RECAP candidates.

This is a plan-time inventory expectation, NOT a runtime invariant. The closure
renderer uses runtime ``len(valid_cells_in_inventory)`` for SSOT discipline
(V4-FIX-1). Step 4 done-condition emits a WARNING (not an error) on drift —
the run continues with the true count.

If you find yourself updating this constant: check whether any closure-side
narrative or exponent fell out of date too. The renderer must NOT need touching
— that would indicate an SSOT regression.
"""

# Internal: JSON key path for the right-hand q99 fingerprint in statistics.json
_STATS_Q99_PATH: tuple[str, ...] = ("unitree_g1", "action", "right_hand", "q99")
_CHECKPOINT_STEP_RE = re.compile(r"^checkpoint-(\d+)$")


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class R2InventoryNoValidCandidates(RuntimeError):
    """Raised by pick_representative when no valid RECAP candidates are found."""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainedCheckpoint:
    """Immutable record for a single trained checkpoint from R2.0 inventory."""

    label: str
    abs_path: Path
    training_algo: str
    base_ckpt_at_training: str
    formalize_language: Optional[bool]
    statistics_q99_right_hand: tuple[float, ...]
    statistics_q99_matches_base: bool
    n_train_steps: int
    training_run_dir: Path
    config_json_sha256: str
    processor_config_json_sha256: str
    statistics_json_sha256: str
    is_valid: bool
    invalid_reason: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file via 1 MiB-chunked read."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_label(ckpt_path: Path) -> str:
    """Return 'RECAP' unless any RECAP_NEGATIVE_TOKENS substring is in the path string."""
    path_str = str(ckpt_path)
    for token in RECAP_NEGATIVE_TOKENS:
        if token in path_str:
            return "OTHER"
    return "RECAP"


def _extract_q99_right_hand(stats: dict) -> tuple[float, ...]:
    """Extract _STATS_Q99_PATH from a parsed statistics dict; empty tuple on miss."""
    try:
        node = stats
        for key in _STATS_Q99_PATH:
            node = node[key]
        return tuple(float(v) for v in node)
    except (KeyError, TypeError, ValueError):
        return ()


def _q99_matches_base(q99: tuple[float, ...]) -> bool:
    """True iff q99 matches BASE_RIGHT_HAND_Q99 element-wise within Q99_TOLERANCE."""
    if len(q99) != len(BASE_RIGHT_HAND_Q99):
        return False
    return all(abs(a - b) <= Q99_TOLERANCE for a, b in zip(q99, BASE_RIGHT_HAND_Q99))


def _parse_n_train_steps(name: str) -> int:
    """Parse step count from 'checkpoint-N'; return -1 if name doesn't match."""
    m = _CHECKPOINT_STEP_RE.match(name)
    return int(m.group(1)) if m else -1


def _build_checkpoint(ckpt_root: Path) -> TrainedCheckpoint:
    """Construct a TrainedCheckpoint from a directory known to pass is_checkpoint_dir."""
    config_path = ckpt_root / "config.json"
    proc_path = ckpt_root / "processor_config.json"
    stats_path = ckpt_root / "statistics.json"
    cfg: dict = json.loads(config_path.read_text(encoding="utf-8"))
    stats: dict = json.loads(stats_path.read_text(encoding="utf-8"))
    abs_path = ckpt_root.resolve()
    label = _classify_label(abs_path)
    q99 = _extract_q99_right_hand(stats)
    q99_matches = _q99_matches_base(q99)
    if label == "RECAP" and q99_matches:
        is_valid, invalid_reason = True, ""
    elif label != "RECAP":
        is_valid, invalid_reason = False, "not_recap"
    else:
        is_valid, invalid_reason = False, "stats_drift"
    return TrainedCheckpoint(
        label=label,
        abs_path=abs_path,
        training_algo=(cfg.get("architectures") or ["unknown"])[0],
        base_ckpt_at_training=str(cfg.get("base_model_name_or_path") or ""),
        formalize_language=cfg.get("formalize_language"),
        statistics_q99_right_hand=q99,
        statistics_q99_matches_base=q99_matches,
        n_train_steps=_parse_n_train_steps(ckpt_root.name),
        training_run_dir=ckpt_root.parent,
        config_json_sha256=_sha256_file(config_path),
        processor_config_json_sha256=_sha256_file(proc_path),
        statistics_json_sha256=_sha256_file(stats_path),
        is_valid=is_valid,
        invalid_reason=invalid_reason,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_checkpoint_dir(p: Path) -> bool:
    """True iff all REQUIRED_CKPT_FILES exist directly under p."""
    return all((p / f).is_file() for f in REQUIRED_CKPT_FILES)


def discover_recap_ckpts(
    search_root: Path = DEFAULT_SEARCH_ROOT,
) -> list[TrainedCheckpoint]:
    """Walk search_root and return all checkpoint dirs as TrainedCheckpoint records.

    Does not recurse inside a checkpoint directory.
    Sort tuple: (-n_train_steps, str(ckpt_root)) — canonical, no mtime tiebreak (A4).
    """
    results: list[TrainedCheckpoint] = []
    for dirpath, dirnames, _ in os.walk(search_root):
        p = Path(dirpath)
        if is_checkpoint_dir(p):
            results.append(_build_checkpoint(p))
            dirnames.clear()  # prune: do not recurse inside a checkpoint dir
    # Canonical sort: (-n_train_steps, str(ckpt_root)) — no mtime (A4)
    results.sort(key=lambda c: (-c.n_train_steps, str(c.abs_path)))
    return results


def filter_valid(ckpts: list[TrainedCheckpoint]) -> list[TrainedCheckpoint]:
    """Return only checkpoints with is_valid=True, preserving input order."""
    return [c for c in ckpts if c.is_valid]


def pick_representative(ckpts: list[TrainedCheckpoint]) -> TrainedCheckpoint:
    """Return the valid checkpoint with highest n_train_steps, then lex on abs_path.

    Sort tuple: (-n_train_steps, str(ckpt_root)) — canonical, consistent with
    discover_recap_ckpts (A4). No mtime tiebreak.

    Raises:
        R2InventoryNoValidCandidates: if no valid checkpoint is found in ckpts.
    """
    valid = [c for c in ckpts if c.is_valid]
    if not valid:
        raise R2InventoryNoValidCandidates(
            f"No valid RECAP candidates among {len(ckpts)} checkpoints."
        )
    # Canonical sort tuple: (-n_train_steps, str(abs_path)) — same as discover_recap_ckpts
    return min(valid, key=lambda c: (-c.n_train_steps, str(c.abs_path)))
