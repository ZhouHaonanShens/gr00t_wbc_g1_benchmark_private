"""Tests for work.recap.r2_authentic_eval.eval_runner (plan §3.1)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from work.recap.r2_authentic_eval._envelope import (
    _eval_skip_decision,
    _r2_invocation_envelope_sha256,
    _r2_module_set_content_sha256,
)
from work.recap.r2_authentic_eval.delta_stats import R2_CELL_RESULT_SCHEMA_VERSION
from work.recap.r2_authentic_eval.eval_runner import (
    AuthenticEvalRequest,
    R2CellResult,
    R2EvalError,
    R2SourceCkptDriftBetweenPhases,
    _capture_failure_context,
    _capture_nvidia_smi_pre_run,
    _git_commit_sha,
)
from work.recap.r2_authentic_eval.inventory import TrainedCheckpoint

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_PROTO_SHA = "proto_sha_000aaa"
_ENV_SHA = "env_sha_111bbb"
_INV_PARAMS: dict = {
    "search_root": "/search/root",
    "strict_config": False,
    "raw_hf_snapshot": "/hf/snapshot",
    "r2_module_set_content_sha": "mod_sha_222ccc",
}

_EXPECTED_FAILURE_CONTEXT_KEYS = frozenset(
    {
        "audit_failure_class",
        "captured_at_utc",
        "git_commit_sha",
        "git_commit_sha_fallback_reason",
        "source_ckpt_root",
        "raw_hf_snapshot_root",
        "swap_root",
        "preserve_path",
        "nvidia_smi_csv",
        "transformers_version",
        "torch_version",
        "gr00t_version",
    }
)


def _make_checkpoint(tmp_path: Path) -> TrainedCheckpoint:
    return TrainedCheckpoint(
        label="RECAP",
        abs_path=tmp_path,
        training_algo="GR00TN1Policy",
        base_ckpt_at_training="nvidia/GR00T-N1.6-G1",
        formalize_language=False,
        statistics_q99_right_hand=(1.5, 1.5, 1.0, 1.5, 0.0, 0.0, 0.0),
        statistics_q99_matches_base=True,
        n_train_steps=5000,
        training_run_dir=tmp_path.parent,
        config_json_sha256="abc123",
        processor_config_json_sha256="def456",
        statistics_json_sha256="ghi789",
        is_valid=True,
        invalid_reason="",
    )


def _make_request(tmp_path: Path) -> AuthenticEvalRequest:
    return AuthenticEvalRequest(
        checkpoint=_make_checkpoint(tmp_path),
        search_root=tmp_path,
        strict_config=False,
    )


def _make_cell(tmp_path: Path) -> R2CellResult:
    req = _make_request(tmp_path)
    return R2CellResult(
        request=req,
        success_count=17,
        completed_episode_total=30,
        rate=17 / 30,
        wilson_ci_95=(0.39, 0.73),
        delta_vs_baseline=0.0,
        newcombe_delta_ci_95=(-0.24, 0.24),
        artifact_dir=tmp_path,
        formal_eval_summary_json={"status": "PASS"},
        raw_repro_result=None,
        ckpt_pre_run_sha256={},
        r1_0_dir_present=False,
        r1_0_baseline_repro_latest_run_mtime_utc=None,
        git_commit_sha="abcdef1234567890abcdef1234567890abcdef12",
        nvidia_smi_pre_run_csv="0,A100,0,80000,0",
        transformers_version="4.40.0",
        torch_version="2.3.0",
        python_version="3.10.0",
        gr00t_version=None,
        protocol_sha256="proto_sha_abc",
        r2_invocation_envelope_sha256="env_sha_xyz",
    )


def _base_cell_payload() -> dict:
    return {
        "protocol_sha256": _PROTO_SHA,
        "r2_invocation_envelope_sha256": _ENV_SHA,
        "formal_eval_summary_json": {"status": "PASS"},
        "completed_episode_total": 30,
        "_r2_invocation_params": dict(_INV_PARAMS),
    }


def _write_cell_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# AuthenticEvalRequest: frozen dataclass
# ---------------------------------------------------------------------------


def test_authentic_eval_request_is_frozen(tmp_path: Path) -> None:
    req = _make_request(tmp_path)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        req.strict_config = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# R2CellResult: frozen + schema version default
# ---------------------------------------------------------------------------


def test_r2_cell_result_is_frozen(tmp_path: Path) -> None:
    cell = _make_cell(tmp_path)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cell.success_count = 999  # type: ignore[misc]


def test_r2_cell_result_schema_version_default(tmp_path: Path) -> None:
    cell = _make_cell(tmp_path)
    assert cell.r2_cell_result_schema_version == R2_CELL_RESULT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# _eval_skip_decision: all-match (decided=True)
# ---------------------------------------------------------------------------


def test_eval_skip_decision_matches_all(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    _write_cell_json(p, _base_cell_payload())
    result = _eval_skip_decision(p, _PROTO_SHA, _ENV_SHA, _INV_PARAMS)
    assert result["decided"] is True
    assert result["reason"] == "all_match"


# ---------------------------------------------------------------------------
# _eval_skip_decision: 8 failure reasons (decided=False)
# ---------------------------------------------------------------------------


def test_eval_skip_decision_protocol_sha_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    _write_cell_json(p, _base_cell_payload())
    result = _eval_skip_decision(p, "WRONG_PROTO_SHA", _ENV_SHA, _INV_PARAMS)
    assert result["decided"] is False
    assert result["reason"] == "protocol_sha256_mismatch"


def test_eval_skip_decision_no_envelope_recorded(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    payload = _base_cell_payload()
    del payload["r2_invocation_envelope_sha256"]
    _write_cell_json(p, payload)
    result = _eval_skip_decision(p, _PROTO_SHA, _ENV_SHA, _INV_PARAMS)
    assert result["decided"] is False
    assert result["reason"] == "envelope_mismatch_no_envelope_recorded"


def test_eval_skip_decision_envelope_mismatch_search_root(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    payload = _base_cell_payload()
    payload["r2_invocation_envelope_sha256"] = "OLD_ENV_SHA"
    payload["_r2_invocation_params"] = {**_INV_PARAMS, "search_root": "/different_root"}
    _write_cell_json(p, payload)
    result = _eval_skip_decision(p, _PROTO_SHA, "NEW_ENV_SHA", _INV_PARAMS)
    assert result["decided"] is False
    assert result["reason"] == "envelope_mismatch_search_root"


def test_eval_skip_decision_envelope_mismatch_strict_config(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    payload = _base_cell_payload()
    payload["r2_invocation_envelope_sha256"] = "OLD_ENV_SHA"
    # search_root matches; strict_config differs
    payload["_r2_invocation_params"] = {**_INV_PARAMS, "strict_config": True}
    _write_cell_json(p, payload)
    result = _eval_skip_decision(p, _PROTO_SHA, "NEW_ENV_SHA", _INV_PARAMS)
    assert result["decided"] is False
    assert result["reason"] == "envelope_mismatch_strict_config"


def test_eval_skip_decision_envelope_mismatch_raw_hf_snapshot(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    payload = _base_cell_payload()
    payload["r2_invocation_envelope_sha256"] = "OLD_ENV_SHA"
    # search_root + strict_config match; raw_hf_snapshot differs
    payload["_r2_invocation_params"] = {**_INV_PARAMS, "raw_hf_snapshot": "/other_hf"}
    _write_cell_json(p, payload)
    result = _eval_skip_decision(p, _PROTO_SHA, "NEW_ENV_SHA", _INV_PARAMS)
    assert result["decided"] is False
    assert result["reason"] == "envelope_mismatch_raw_hf_snapshot"


def test_eval_skip_decision_envelope_mismatch_module_set(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    payload = _base_cell_payload()
    payload["r2_invocation_envelope_sha256"] = "OLD_ENV_SHA"
    # all three checked fields match current → blame falls on module_set_content_sha
    payload["_r2_invocation_params"] = dict(_INV_PARAMS)
    _write_cell_json(p, payload)
    result = _eval_skip_decision(p, _PROTO_SHA, "NEW_ENV_SHA", _INV_PARAMS)
    assert result["decided"] is False
    assert result["reason"] == "envelope_mismatch_module_set_content_sha"


def test_eval_skip_decision_formal_eval_status_not_pass(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    payload = _base_cell_payload()
    payload["formal_eval_summary_json"] = {"status": "FAIL"}
    _write_cell_json(p, payload)
    result = _eval_skip_decision(p, _PROTO_SHA, _ENV_SHA, _INV_PARAMS)
    assert result["decided"] is False
    assert result["reason"] == "formal_eval_status_not_pass"


def test_eval_skip_decision_episode_count_not_30(tmp_path: Path) -> None:
    p = tmp_path / "cell_result.json"
    payload = _base_cell_payload()
    payload["completed_episode_total"] = 10
    _write_cell_json(p, payload)
    result = _eval_skip_decision(p, _PROTO_SHA, _ENV_SHA, _INV_PARAMS)
    assert result["decided"] is False
    assert result["reason"] == "episode_count_not_30"


# ---------------------------------------------------------------------------
# _git_commit_sha (subprocess #2, V3-FIX-5)
# ---------------------------------------------------------------------------


def test_git_commit_sha_returns_tuple() -> None:
    sha, fallback = _git_commit_sha()
    assert isinstance(sha, str)
    assert fallback is None or isinstance(fallback, str)
    if fallback is None:
        # On a real git repo the sha is a 40-char hex string
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


def test_git_commit_sha_fallback_on_no_git() -> None:
    with patch("work.recap.r2_authentic_eval.eval_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="not a git repo")
        sha, fallback = _git_commit_sha()
    assert sha == ""
    assert isinstance(fallback, str)
    assert fallback  # non-empty


# ---------------------------------------------------------------------------
# _capture_nvidia_smi_pre_run (subprocess #1)
# ---------------------------------------------------------------------------


def test_capture_nvidia_smi_pre_run_runs_subprocess() -> None:
    with patch("work.recap.r2_authentic_eval.eval_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="0,A100,0,80000,0\n")
        result = _capture_nvidia_smi_pre_run("1")
    assert isinstance(result, str)
    assert result == "0,A100,0,80000,0"


# ---------------------------------------------------------------------------
# _capture_failure_context: 12 fields (subprocess #3, V4-FIX-2)
# ---------------------------------------------------------------------------


def test_capture_failure_context_12_fields(tmp_path: Path) -> None:
    fake_sha = "a" * 40 + "\n"
    with patch("work.recap.r2_authentic_eval.eval_runner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=fake_sha, stderr="")
        ctx = _capture_failure_context(
            source_ckpt_root=tmp_path / "ckpt",
            raw_hf_snapshot_root=tmp_path / "hf",
            swap_root=tmp_path / "swap",
            preserve_path=tmp_path / "preserve",
            audit_failure_class="test_drift",
        )
    assert set(ctx.keys()) == _EXPECTED_FAILURE_CONTEXT_KEYS
    assert len(ctx) == 12
    assert ctx["audit_failure_class"] == "test_drift"
    assert ctx["source_ckpt_root"] == str(tmp_path / "ckpt")


# ---------------------------------------------------------------------------
# _envelope: invocation envelope sha changes with inputs
# ---------------------------------------------------------------------------


def test_r2_invocation_envelope_sha256_changes_with_inputs(tmp_path: Path) -> None:
    base_args: dict = dict(
        search_root=tmp_path,
        strict_config=False,
        raw_hf_snapshot=tmp_path / "hf",
        r2_module_set_content_sha="mod_abc",
    )
    sha_a = _r2_invocation_envelope_sha256(**base_args)
    sha_b = _r2_invocation_envelope_sha256(**{**base_args, "strict_config": True})
    sha_c = _r2_invocation_envelope_sha256(**{**base_args, "r2_module_set_content_sha": "mod_xyz"})
    assert sha_a != sha_b, "strict_config change must alter sha"
    assert sha_a != sha_c, "module_set_content_sha change must alter sha"
    assert sha_b != sha_c
    # Deterministic across two calls
    assert sha_a == _r2_invocation_envelope_sha256(**base_args)


# ---------------------------------------------------------------------------
# _envelope: module-set content sha is deterministic
# ---------------------------------------------------------------------------


def test_r2_module_set_content_sha256_is_deterministic() -> None:
    sha1 = _r2_module_set_content_sha256()
    sha2 = _r2_module_set_content_sha256()
    assert sha1 == sha2
    assert isinstance(sha1, str)
    assert len(sha1) == 64  # SHA-256 produces 64 hex chars
