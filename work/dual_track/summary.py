"""Build and validate GR00T/OpenPI dual-track summary artifacts.

The dual-track blocker push deliberately separates formal gate evidence from
exploratory signal.  This module keeps that separation explicit when producing
``dual_track_summary.json`` so positive exploratory results cannot be promoted
into formal pass claims by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FORMAL_SCHEMA_VERSION = "dual_track_formal_status_v1"
EXPLORATORY_SCHEMA_VERSION = "dual_track_exploratory_signal_v1"
SUMMARY_SCHEMA_VERSION = "dual_track_summary_v1"
FORMAL_STATUSES = {"PASS", "BLOCK", "SKIPPED"}
EXPLORATORY_STATUSES = {"SIGNAL", "NO_SIGNAL", "SKIPPED", "FAILED"}
OPENPI_RUNTIME_LEVELS = {
    "materialization_ready",
    "none",
    "p0_loader_runtime_pass",
    "p1_one_step_pass",
    "p2_overfit_or_tiny_update_pass",
}
OPENPI_RUNTIME_LEVEL_RANK = {
    level: rank
    for rank, level in enumerate(
        (
            "none",
            "materialization_ready",
            "p0_loader_runtime_pass",
            "p1_one_step_pass",
            "p2_overfit_or_tiny_update_pass",
        )
    )
}
OPENPI_REQUIRED_RUNTIME_LEVELS = OPENPI_RUNTIME_LEVELS - {"none"}
FORBIDDEN_INFERENCES = [
    "exploratory signal != formal pass",
    "OpenPI exploratory dataset != formal materialized",
    "GR00T metric ablation/additional seed signal != P5 eligible",
]


class DualTrackSummaryError(ValueError):
    """Raised when a dual-track artifact would violate the test spec."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise DualTrackSummaryError(f"missing artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DualTrackSummaryError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DualTrackSummaryError(f"artifact must be a JSON object: {path}")
    return value


def validate_formal_status(obj: dict[str, Any], *, lane: str, path: Path | None = None) -> None:
    label = str(path) if path else lane
    if obj.get("schema_version") != FORMAL_SCHEMA_VERSION:
        raise DualTrackSummaryError(f"{label}: invalid formal schema_version")
    if obj.get("lane") != lane:
        raise DualTrackSummaryError(f"{label}: expected lane {lane!r}")
    if obj.get("track") != "formal":
        raise DualTrackSummaryError(f"{label}: formal artifact must have track='formal'")
    status = obj.get("status")
    if status not in FORMAL_STATUSES:
        raise DualTrackSummaryError(f"{label}: invalid formal status {status!r}")
    if isinstance(status, str) and ("(" in status or ")" in status):
        raise DualTrackSummaryError(f"{label}: compound status is forbidden")
    if not isinstance(obj.get("blocking_reasons"), list):
        raise DualTrackSummaryError(f"{label}: blocking_reasons must be a list")
    if obj.get("formal_claim_allowed") is True:
        if status != "PASS" or obj.get("next_gate_allowed") is not True:
            raise DualTrackSummaryError(
                f"{label}: formal_claim_allowed requires PASS and next_gate_allowed=true"
            )
    elif obj.get("formal_claim_allowed") is not False:
        raise DualTrackSummaryError(f"{label}: formal_claim_allowed must be boolean false or true")
    if lane == "openpi":
        runtime_level = obj.get("runtime_level")
        required_runtime_level = obj.get("required_runtime_level")
        runtime_level_valid = (
            runtime_level in OPENPI_RUNTIME_LEVELS
            or (isinstance(runtime_level, str) and runtime_level.startswith("blocked_"))
        )
        if not runtime_level_valid:
            raise DualTrackSummaryError(f"{label}: invalid OpenPI runtime_level")
        if required_runtime_level not in OPENPI_REQUIRED_RUNTIME_LEVELS:
            raise DualTrackSummaryError(
                f"{label}: invalid OpenPI required_runtime_level"
            )
        if not isinstance(obj.get("runtime_evidence"), list):
            raise DualTrackSummaryError(
                f"{label}: OpenPI runtime_evidence must be a list"
            )
        runtime_rank = OPENPI_RUNTIME_LEVEL_RANK.get(str(runtime_level), -1)
        if status == "PASS" and runtime_rank < OPENPI_RUNTIME_LEVEL_RANK[str(required_runtime_level)]:
            raise DualTrackSummaryError(
                f"{label}: OpenPI PASS cannot be below required_runtime_level"
            )


def validate_exploratory_signal(obj: dict[str, Any], *, lane: str, path: Path | None = None) -> None:
    label = str(path) if path else lane
    if obj.get("schema_version") != EXPLORATORY_SCHEMA_VERSION:
        raise DualTrackSummaryError(f"{label}: invalid exploratory schema_version")
    if obj.get("lane") != lane:
        raise DualTrackSummaryError(f"{label}: expected lane {lane!r}")
    if obj.get("track") != "exploratory":
        raise DualTrackSummaryError(f"{label}: exploratory artifact must have track='exploratory'")
    status = obj.get("status")
    if status not in EXPLORATORY_STATUSES:
        raise DualTrackSummaryError(f"{label}: invalid exploratory status {status!r}")
    if obj.get("exploratory_only") is not True:
        raise DualTrackSummaryError(f"{label}: exploratory_only must be true")
    if obj.get("formal_claim_allowed") is not False:
        raise DualTrackSummaryError(f"{label}: exploratory formal_claim_allowed must be false")
    if obj.get("must_not_unlock_formal_gate") is not True:
        raise DualTrackSummaryError(f"{label}: must_not_unlock_formal_gate must be true")
    if obj.get("risk_label") != "exploratory_not_formal":
        raise DualTrackSummaryError(f"{label}: risk_label must be exploratory_not_formal")


def _missing_formal(lane: str, artifact: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "lane": lane,
        "track": "formal",
        "status": "SKIPPED",
        "formal_claim_allowed": False,
        "blocking_reasons": ["pending_lane_artifact"],
        "authority_inputs": [],
        "validator_outputs": [],
        "entered_next_gate": False,
        "next_gate_allowed": False,
        "notes": f"No formal_status.json was present at summary build time: {artifact}",
    }
    if lane == "openpi":
        payload.update(
            {
                "runtime_level": "blocked_pending_lane_artifact",
                "required_runtime_level": "p1_one_step_pass",
                "runtime_evidence": [],
            }
        )
    return payload


def _missing_exploratory(lane: str, artifact: Path) -> dict[str, Any]:
    return {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "lane": lane,
        "track": "exploratory",
        "status": "SKIPPED",
        "exploratory_only": True,
        "formal_claim_allowed": False,
        "must_not_unlock_formal_gate": True,
        "method": "other",
        "risk_label": "exploratory_not_formal",
        "inputs": [],
        "outputs": [],
        "observed_signal": {},
        "notes": f"No exploratory_signal.json was present at summary build time: {artifact}",
    }


def _read_or_missing(path: Path, *, lane: str, track: str, allow_missing: bool) -> dict[str, Any]:
    if path.exists():
        return _load_json(path)
    if not allow_missing:
        raise DualTrackSummaryError(f"missing {lane} {track} artifact: {path}")
    return _missing_formal(lane, path) if track == "formal" else _missing_exploratory(lane, path)


def _summary_section(status_obj: dict[str, Any], artifact: Path, *, formal: bool) -> dict[str, Any]:
    section = {"status": status_obj["status"], "artifact": str(artifact)}
    if formal:
        section["formal_claim_allowed"] = bool(status_obj.get("formal_claim_allowed"))
        section["blocking_reasons"] = list(status_obj.get("blocking_reasons") or [])
        if "runtime_level" in status_obj:
            section["runtime_level"] = status_obj["runtime_level"]
        if "runtime_claims" in status_obj:
            section["runtime_claims"] = list(status_obj.get("runtime_claims") or [])
    return section


def build_summary(
    *,
    gr00t_root: Path,
    openpi_root: Path,
    allow_missing: bool = False,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a validated four-section summary from lane artifact roots."""

    artifact_paths = {
        "gr00t_formal": gr00t_root / "formal_status.json",
        "gr00t_exploratory": gr00t_root / "exploratory_signal.json",
        "openpi_formal": openpi_root / "formal_status.json",
        "openpi_exploratory": openpi_root / "exploratory_signal.json",
    }
    gr00t_formal = _read_or_missing(
        artifact_paths["gr00t_formal"], lane="gr00t", track="formal", allow_missing=allow_missing
    )
    gr00t_exploratory = _read_or_missing(
        artifact_paths["gr00t_exploratory"], lane="gr00t", track="exploratory", allow_missing=allow_missing
    )
    openpi_formal = _read_or_missing(
        artifact_paths["openpi_formal"], lane="openpi", track="formal", allow_missing=allow_missing
    )
    openpi_exploratory = _read_or_missing(
        artifact_paths["openpi_exploratory"], lane="openpi", track="exploratory", allow_missing=allow_missing
    )

    validate_formal_status(gr00t_formal, lane="gr00t", path=artifact_paths["gr00t_formal"])
    validate_exploratory_signal(
        gr00t_exploratory, lane="gr00t", path=artifact_paths["gr00t_exploratory"]
    )
    validate_formal_status(openpi_formal, lane="openpi", path=artifact_paths["openpi_formal"])
    validate_exploratory_signal(
        openpi_exploratory, lane="openpi", path=artifact_paths["openpi_exploratory"]
    )

    if gr00t_formal.get("formal_claim_allowed") and gr00t_exploratory["status"] == "SIGNAL":
        # Legal but worth making explicit in downstream review: the formal pass came
        # from the formal object only.  The section below never reads exploratory
        # status to compute formal_claim_allowed.
        pass

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "gr00t": {
            "formal": _summary_section(gr00t_formal, artifact_paths["gr00t_formal"], formal=True),
            "exploratory": _summary_section(
                gr00t_exploratory, artifact_paths["gr00t_exploratory"], formal=False
            ),
        },
        "openpi": {
            "formal": _summary_section(openpi_formal, artifact_paths["openpi_formal"], formal=True),
            "exploratory": _summary_section(
                openpi_exploratory, artifact_paths["openpi_exploratory"], formal=False
            ),
        },
        "next_actions": list(next_actions or []),
        "forbidden_inferences": list(FORBIDDEN_INFERENCES),
    }


def write_summary(summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-root", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="emit SKIPPED placeholder sections for lane artifacts that are not ready yet",
    )
    parser.add_argument("--next-action", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_summary(
        gr00t_root=args.gr00t_root,
        openpi_root=args.openpi_root,
        allow_missing=args.allow_missing,
        next_actions=args.next_action,
    )
    write_summary(summary, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
