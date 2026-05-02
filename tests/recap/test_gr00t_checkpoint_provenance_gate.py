from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_checkpoint_provenance_gate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _build_happy_metadata(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint_dir = tmp_path / "checkpoint_C1_phase_mode" / "checkpoint-100"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_text(
        "dummy finetuned checkpoint bytes\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "state_conditioned_training_run_v1",
        "artifact_kind": "state_conditioned_training_run_metadata",
        "variant_key": "c1",
        "comparable_run_spec": {
            "stable_base": {
                "base_model": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
                "embodiment_tag": "UNITREE_G1",
            },
            "checkpoint_rule": {
                "save_total_limit": 1,
                "selected_checkpoint_path": str(checkpoint_dir),
            },
        },
        "evaluation_binding": {
            "eval_uses_finetuned": True,
            "server_load_mode": "model_path",
            "server_load_path": str(checkpoint_dir),
        },
    }
    metadata_path = _write_json(tmp_path / "run_metadata.json", metadata)
    return metadata_path, checkpoint_dir


def _build_historical_contamination_metadata(tmp_path: Path) -> Path:
    metadata = {
        "schema_version": "state_conditioned_training_run_v1",
        "artifact_kind": "state_conditioned_training_run_metadata",
        "variant_key": "c1",
        "comparable_run_spec": {
            "stable_base": {
                "base_model": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
                "embodiment_tag": "UNITREE_G1",
            },
            "checkpoint_rule": {
                "save_total_limit": 1,
                "selected_checkpoint_path": None,
            },
        },
        "evaluation_binding": {
            "eval_uses_finetuned": False,
            "server_load_mode": "model_path",
            "server_load_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
        },
        "finetune_returncode": 1,
        "finetune_failure_reason": "torch.OutOfMemoryError: CUDA out of memory",
    }
    return _write_json(tmp_path / "historical_contamination.json", metadata)


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_checkpoint_provenance_gate.main(["--help"])
    assert exc_info.value.code == 0


def test_happy_path_writes_machine_readable_allow_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_path, checkpoint_dir = _build_happy_metadata(tmp_path)
    output_dir = tmp_path / "provenance_report"

    exit_code = gr00t_checkpoint_provenance_gate.main(
        ["--metadata", str(metadata_path), "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        output_dir
        / gr00t_checkpoint_provenance_gate.CHECKPOINT_PROVENANCE_REPORT_JSON_NAME
    )

    assert exit_code == 0
    assert captured.err == ""
    assert payload["formal_eligibility"] == "ALLOW"
    assert payload["status"] == "PASS"
    assert payload["reason_code"] == "ok"
    assert payload["selected_checkpoint_path"] == str(checkpoint_dir)
    assert payload["server_load_path"] == str(checkpoint_dir)
    assert payload["base_model_path"] == "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
    assert payload["is_base_fallback"] is False
    assert payload["loadability_status"] == "LOADABLE_CHECKPOINT_CONFIRMED"
    assert payload["selected_checkpoint_metadata"]["selected_checkpoint_exists"] is True
    assert (
        payload["selected_checkpoint_metadata"]["selected_checkpoint_loadable"] is True
    )
    assert (
        payload["server_binding"]["server_load_path_matches_selected_checkpoint"]
        is True
    )
    assert payload["checksum_or_signature"].startswith("sha256:")
    assert artifact == payload


def test_historical_oom_base_fallback_blocks_with_required_reason_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_path = _build_historical_contamination_metadata(tmp_path)
    output_dir = tmp_path / "provenance_blocked"

    exit_code = gr00t_checkpoint_provenance_gate.main(
        ["--metadata", str(metadata_path), "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        output_dir
        / gr00t_checkpoint_provenance_gate.CHECKPOINT_PROVENANCE_REPORT_JSON_NAME
    )
    failure_note = (
        output_dir / gr00t_checkpoint_provenance_gate.FAILURE_NOTE_MARKDOWN_NAME
    ).read_text(encoding="utf-8")

    assert exit_code == 1
    assert captured.err == ""
    assert payload["formal_eligibility"] == "BLOCK"
    assert payload["status"] == "FAIL"
    assert payload["reason_code"] == "wrong_checkpoint_or_missing_finetune_artifact"
    assert payload["selected_checkpoint_path"] is None
    assert payload["server_load_path"] == "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
    assert payload["is_base_fallback"] is True
    assert payload["loadability_status"] == "BLOCKED_SELECTED_CHECKPOINT_MISSING"
    assert (
        payload["historical_regressions"]["historical_oom_contamination_pattern"]
        is True
    )
    assert "selected_checkpoint_path_missing" in payload["gate_reasons"]
    assert "eval_uses_finetuned=False" in payload["gate_reasons"]
    assert "wrong_checkpoint_or_missing_finetune_artifact" in failure_note
    assert artifact == payload


def test_repo_example_fixture_passes_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "repo_example_report"
    metadata_path = REPO_ROOT / "agent" / "artifacts" / "example_run_metadata.json"

    exit_code = gr00t_checkpoint_provenance_gate.main(
        ["--metadata", str(metadata_path), "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["formal_eligibility"] == "ALLOW"
    assert payload["selected_checkpoint_path"] == str(
        REPO_ROOT / "agent" / "artifacts" / "example_checkpoint" / "checkpoint-100"
    )
