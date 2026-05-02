import json
from pathlib import Path

import pytest

from work.dual_track.summary import DualTrackSummaryError, build_summary, main


def _formal(lane: str, status: str = "BLOCK") -> dict:
    payload = {
        "schema_version": "dual_track_formal_status_v1",
        "lane": lane,
        "track": "formal",
        "status": status,
        "formal_claim_allowed": False,
        "blocking_reasons": ["test_blocker"],
        "authority_inputs": [],
        "validator_outputs": [],
        "entered_next_gate": False,
        "next_gate_allowed": False,
        "notes": "unit test",
    }
    if lane == "openpi":
        payload["runtime_level"] = "blocked_policy_bridge_error"
        payload["required_runtime_level"] = "p1_one_step_pass"
        payload["runtime_evidence"] = []
        payload["runtime_claims"] = []
    return payload


def _exploratory(lane: str, status: str = "SIGNAL") -> dict:
    return {
        "schema_version": "dual_track_exploratory_signal_v1",
        "lane": lane,
        "track": "exploratory",
        "status": status,
        "exploratory_only": True,
        "formal_claim_allowed": False,
        "must_not_unlock_formal_gate": True,
        "method": "additional_seed",
        "risk_label": "exploratory_not_formal",
        "inputs": [],
        "outputs": [],
        "observed_signal": {"unit": True},
        "notes": "unit test",
    }


def _write(root: Path, formal: dict, exploratory: dict) -> None:
    root.mkdir(parents=True)
    (root / "formal_status.json").write_text(json.dumps(formal))
    (root / "exploratory_signal.json").write_text(json.dumps(exploratory))


def test_build_summary_keeps_exploratory_signal_out_of_formal_claim(tmp_path: Path) -> None:
    gr00t = tmp_path / "gr00t"
    openpi = tmp_path / "openpi"
    _write(gr00t, _formal("gr00t", "BLOCK"), _exploratory("gr00t", "SIGNAL"))
    _write(openpi, _formal("openpi", "BLOCK"), _exploratory("openpi", "SIGNAL"))

    summary = build_summary(gr00t_root=gr00t, openpi_root=openpi)

    assert summary["schema_version"] == "dual_track_summary_v1"
    assert summary["gr00t"]["formal"]["status"] == "BLOCK"
    assert summary["gr00t"]["formal"]["formal_claim_allowed"] is False
    assert summary["gr00t"]["formal"]["blocking_reasons"] == ["test_blocker"]
    assert summary["gr00t"]["exploratory"]["status"] == "SIGNAL"
    assert summary["openpi"]["formal"]["runtime_level"] == "blocked_policy_bridge_error"
    assert "exploratory signal != formal pass" in summary["forbidden_inferences"]


def test_rejects_compound_formal_status(tmp_path: Path) -> None:
    gr00t = tmp_path / "gr00t"
    openpi = tmp_path / "openpi"
    _write(gr00t, _formal("gr00t", "BLOCK(label_semantics_block)"), _exploratory("gr00t"))
    _write(openpi, _formal("openpi"), _exploratory("openpi"))

    with pytest.raises(DualTrackSummaryError, match="invalid formal status|compound status"):
        build_summary(gr00t_root=gr00t, openpi_root=openpi)


def test_rejects_exploratory_formal_claim(tmp_path: Path) -> None:
    gr00t = tmp_path / "gr00t"
    openpi = tmp_path / "openpi"
    bad_exploratory = _exploratory("gr00t")
    bad_exploratory["formal_claim_allowed"] = True
    _write(gr00t, _formal("gr00t"), bad_exploratory)
    _write(openpi, _formal("openpi"), _exploratory("openpi"))

    with pytest.raises(DualTrackSummaryError, match="exploratory formal_claim_allowed"):
        build_summary(gr00t_root=gr00t, openpi_root=openpi)


def test_cli_allow_missing_writes_skipped_summary(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"

    assert main([
        "--gr00t-root", str(tmp_path / "missing-gr00t"),
        "--openpi-root", str(tmp_path / "missing-openpi"),
        "--output", str(output),
        "--allow-missing",
        "--next-action", "wait for lane artifacts",
    ]) == 0

    summary = json.loads(output.read_text())
    assert summary["gr00t"]["formal"]["status"] == "SKIPPED"
    assert summary["openpi"]["formal"]["formal_claim_allowed"] is False
    assert summary["next_actions"] == ["wait for lane artifacts"]
