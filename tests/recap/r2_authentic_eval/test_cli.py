"""Tests for the R2 CLI orchestrator (plan v4 §3.1 + §3.2 integration).

The GPU subcommands (``evaluate-all``, ``r2-run``) are not covered here — they
require a real GPU and a passing baseline marker, both reserved for team-lead
post-implementation. The covered surface is:
  - argparse plumbing (subcommand list, defaults).
  - ``inventory`` end-to-end on a real disk fake fixture.
  - ``build_statistical_regime`` SSOT propagation (V4-FIX-1 integration).
  - ``r2-summarise`` round-trip on stub cell_result.json files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from work.recap.r2_authentic_eval import _workflow as wf
from work.recap.r2_authentic_eval import cli
from work.recap.r2_authentic_eval.delta_stats import (
    R2_BASELINE_N_DEFAULT,
    R2_BASELINE_SUCC_DEFAULT,
    family_wise_error_rate_at_baseline,
)


# ---------------------------------------------------------------------------
# build_parser: subcommand list + defaults
# ---------------------------------------------------------------------------


def test_build_parser_lists_all_subcommands() -> None:
    parser = cli.build_parser()
    # Force-trigger the subparsers' metavar by parsing each subcommand --help
    expected = {
        "inventory",
        "evaluate-all",
        "config-swap",
        "r2-2-decompose-dry-run",
        "r2-summarise",
        "r2-run",
    }
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "build_parser must register the 'command' subparsers"
    found = set(actions[0].choices.keys())  # type: ignore[union-attr]
    missing = expected - found
    assert not missing, f"missing subcommands in cli.build_parser: {missing}"


def test_inventory_subparser_defaults_to_default_search_root() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["inventory"])
    assert args.command == "inventory"
    assert args.search_root  # non-empty default


def test_evaluate_all_has_skip_existing_cells_flag() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["evaluate-all", "--skip-existing-cells"])
    assert args.skip_existing_cells is True


def test_r2_summarise_requires_run_dir() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["r2-summarise"])


# ---------------------------------------------------------------------------
# build_statistical_regime: V4-FIX-1 SSOT propagation
# ---------------------------------------------------------------------------


def test_build_statistical_regime_propagates_n_valid_cells() -> None:
    marker = {"success_count": 17, "episode_count": 30}
    stat = wf.build_statistical_regime(baseline_marker=marker, n_valid_cells=4)
    assert stat["n_valid_cells"] == 4
    assert stat["baseline_succ"] == 17
    assert stat["baseline_total"] == 30
    expected_fwer = family_wise_error_rate_at_baseline(17, 30, n_cells=4)
    assert stat["family_wise_at_baseline"] == pytest.approx(expected_fwer)


def test_build_statistical_regime_falls_back_to_defaults_on_missing_marker_keys() -> None:
    stat = wf.build_statistical_regime(baseline_marker={}, n_valid_cells=5)
    assert stat["baseline_succ"] == R2_BASELINE_SUCC_DEFAULT
    assert stat["baseline_total"] == R2_BASELINE_N_DEFAULT
    assert stat["n_valid_cells"] == 5


def test_n_valid_cells_propagates_from_inventory_to_closure() -> None:
    """V4-FIX-1 integration: 4-cell inventory → closure narrative + exponent + FWER differ from 5."""
    from work.recap.r2_authentic_eval.reports import closure_report

    marker = {"success_count": 17, "episode_count": 30}
    stat4 = wf.build_statistical_regime(baseline_marker=marker, n_valid_cells=4)
    stat5 = wf.build_statistical_regime(baseline_marker=marker, n_valid_cells=5)
    md4 = closure_report.render(
        cells=[],
        statistical_regime=stat4,
        baseline_marker={"protocol_sha256": "abc", "timestamp_utc": "x"},
    )
    md5 = closure_report.render(
        cells=[],
        statistical_regime=stat5,
        baseline_marker={"protocol_sha256": "abc", "timestamp_utc": "x"},
    )
    assert "across 4 evidence-grade RECAP cells" in md4 and "^4" in md4
    assert "across 5 evidence-grade RECAP cells" in md5 and "^5" in md5
    assert f"{stat4['family_wise_at_baseline']:.3f}" in md4
    assert f"{stat5['family_wise_at_baseline']:.3f}" in md5


# ---------------------------------------------------------------------------
# inventory subcommand: end-to-end on a fixture search root with no candidates
# ---------------------------------------------------------------------------


def test_run_inventory_writes_artifacts_when_search_root_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    search_root = tmp_path / "fake_recap"
    search_root.mkdir()
    out_dir = tmp_path / "out"
    args = type(
        "A", (), {"search_root": str(search_root), "out": str(out_dir)}
    )()
    rc = wf.run_inventory(args)  # type: ignore[arg-type]
    assert rc == 0
    captured = capsys.readouterr()
    assert str(out_dir) in captured.out
    assert (out_dir / "inventory_report.md").is_file()
    done = json.loads((out_dir / "r2_0_done.json").read_text(encoding="utf-8"))
    assert done["valid_recap_count"] == 0
    assert done["total_recap_count"] == 0
    assert done["classification"] == "RECAP (negative-token rule)"


def test_run_inventory_emits_warning_on_count_drift(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """V4-FIX-1: drift from R2_VALID_CELL_COUNT_EXPECTED logs WARNING (not error)."""
    import logging
    search_root = tmp_path / "fake_recap"
    search_root.mkdir()
    out_dir = tmp_path / "out"
    args = type(
        "A", (), {"search_root": str(search_root), "out": str(out_dir)}
    )()
    with caplog.at_level(logging.WARNING, logger="r2_authentic_eval"):
        wf.run_inventory(args)  # type: ignore[arg-type]
    # 0 != 5 → warning fires
    assert any("R2.0 inventory found 0 valid cells" in r.message for r in caplog.records)


def test_run_config_delta_audit_writes_report_on_attention_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "config_delta"
    ckpt = type("C", (), {"abs_path": tmp_path / "ckpt"})()

    def fake_audit_inventory(*_args: Any, dossier_dir: Path, **_kwargs: Any) -> dict[str, Any]:
        dossier_dir.mkdir(parents=True, exist_ok=True)
        attention_md = dossier_dir / wf.ATTENTION_FILENAME
        attention_md.write_text("attention\n", encoding="utf-8")
        inventory = {
            "row_count": 1,
            "allowed_paths": ["config.json:formalize_language"],
            "summary": {
                "ONLY_FORMALIZE_LANGUAGE": 0,
                wf.ADDITIONAL_FIELDS_DIFFER: 1,
                "architectures_mismatch_count": 1,
            },
            "rows": [
                {
                    "ckpt_root": str(ckpt.abs_path),
                    "classification": wf.ADDITIONAL_FIELDS_DIFFER,
                    "outside_paths": ["config.json:hidden_size"],
                    "architectures_mismatch": True,
                }
            ],
            "attention": {
                "status": "pending",
                "attention_md": str(attention_md),
                "user_attention_md": str(dossier_dir / "r2_0_5_user_attention.md"),
                "acknowledgment_md": str(dossier_dir / wf.ACKNOWLEDGMENT_FILENAME),
            },
        }
        (dossier_dir / wf.INVENTORY_FILENAME).write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return inventory

    monkeypatch.setattr(wf, "discover_recap_ckpts", lambda _search_root: [ckpt])
    monkeypatch.setattr(wf, "filter_valid", lambda _inventory: [ckpt])
    monkeypatch.setattr(wf, "audit_inventory", fake_audit_inventory)

    args = type("A", (), {"search_root": str(tmp_path), "out": str(out_dir)})()
    rc = wf.run_config_delta_audit(args)  # type: ignore[arg-type]

    assert rc == 78
    report = (out_dir / "config_delta_report.md").read_text(encoding="utf-8")
    assert "allowed_paths" in report
    assert "pending" in report


# ---------------------------------------------------------------------------
# r2-summarise subcommand: writes summary_table.json from raw cell jsons
# ---------------------------------------------------------------------------


def _stub_cell_payload(
    rate: float,
    envelope_sha: str = "env_aaa",
    ckpt_abs_path: str = "/fake/g3/checkpoint-1",
) -> dict[str, Any]:
    return {
        "ckpt_abs_path": ckpt_abs_path,
        "success_count": int(round(rate * 30)),
        "completed_episode_total": 30,
        "rate": rate,
        "wilson_ci_95": [max(0.0, rate - 0.18), min(1.0, rate + 0.18)],
        "delta_vs_baseline": rate - 17 / 30,
        "newcombe_delta_ci_95": [-0.24, 0.24],
        "artifact_dir": "/fake",
        "formal_eval_summary_json": {"status": "PASS"},
        "ckpt_pre_run_sha256": {},
        "r1_0_dir_present": False,
        "r1_0_baseline_repro_latest_run_mtime_utc": None,
        "git_commit_sha": "a" * 40,
        "nvidia_smi_pre_run_csv": "",
        "transformers_version": "4.40.0",
        "torch_version": "2.3.0",
        "python_version": "3.10.12",
        "gr00t_version": None,
        "protocol_sha256": "proto_abc",
        "r2_invocation_envelope_sha256": envelope_sha,
        "git_commit_sha_fallback_reason": None,
        "r2_cell_result_schema_version": "1.0.0",
    }


def test_run_r2_summarise_writes_summary_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "ckpt-A").mkdir(parents=True)
    (run_dir / "ckpt-A" / "cell_result.json").write_text(
        json.dumps(_stub_cell_payload(0.6)), encoding="utf-8"
    )

    # Stub the baseline-marker validator so test does not need on-disk marker
    monkeypatch.setattr(
        wf, "validate_baseline_pass_marker",
        lambda protocol: {"success_count": 17, "episode_count": 30},
    )

    args = type("A", (), {"run_dir": str(run_dir)})()
    rc = wf.run_r2_summarise(args)  # type: ignore[arg-type]
    assert rc == 0
    payload = json.loads((run_dir / "summary_table.json").read_text(encoding="utf-8"))
    assert payload["r2_summary_table_schema_version"] == "1.0.0"
    assert payload["n_valid_cells"] == 1
    assert payload["baseline_succ"] == 17
    assert payload["baseline_total"] == 30
    assert "trigger_threshold" in payload
    assert len(payload["raw_cells"]) == 1


def test_run_r2_summarise_filters_excluded_cell_for_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    for slug, ckpt_abs_path in (
        ("raw-excluded", "/fake/g2_full_training/checkpoint-2200"),
        ("evidence", "/fake/g3_conditioned/checkpoint-6600"),
    ):
        (run_dir / slug).mkdir(parents=True)
        (run_dir / slug / "cell_result.json").write_text(
            json.dumps(_stub_cell_payload(0.6, ckpt_abs_path=ckpt_abs_path)),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        wf, "validate_baseline_pass_marker",
        lambda protocol: {"success_count": 17, "episode_count": 30},
    )

    rc = wf.run_r2_summarise(type("A", (), {"run_dir": str(run_dir)})())  # type: ignore[arg-type]
    payload = json.loads((run_dir / "summary_table.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["raw_observation_cell_count"] == 2
    assert payload["evidence_grade_cell_count"] == 1
    assert payload["n_valid_cells"] == 1
    assert len(payload["raw_cells"]) == 2


def test_run_r2_summarise_returns_nonzero_when_no_cells(tmp_path: Path) -> None:
    args = type("A", (), {"run_dir": str(tmp_path)})()
    rc = wf.run_r2_summarise(args)  # type: ignore[arg-type]
    assert rc != 0
