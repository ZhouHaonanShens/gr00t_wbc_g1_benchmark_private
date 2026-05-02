from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import critic_promotion


def _load_script_module(script_name: str, module_name: str):
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _all_green_gates() -> dict[str, bool]:
    return {gate_name: True for gate_name in critic_promotion.GATES_A_F_ORDER}


def test_happy_path_promotes_critic_to_primary_relabel_source() -> None:
    payload = critic_promotion.build_critic_promotion_verdict(
        offline_audit_payload={
            "pass": True,
            "task": "T10 diagnostic relabel quality audit",
        },
        downstream_gate_payload={
            "gate_passed": True,
            "gate_status": "DIAGNOSTIC_PASS",
            "gate_name": "vlm_critic_downstream_diagnostic_gate",
        },
        gates_a_f_bundle=_all_green_gates(),
        evidence_paths={
            "offline_audit_json": "agent/artifacts/vlm_critic_relabel/relabel_quality_audit_v1.json",
            "downstream_gate_json": "agent/artifacts/vlm_critic_relabel/downstream_gate.json",
            "gates_a_f_json": "agent/artifacts/vlm_critic_relabel/gates_a_f.json",
        },
    )

    assert payload["promotion_allowed"] is True
    assert payload["promotion_status"] == critic_promotion.PROMOTION_STATUS_PASS
    assert payload["critic_role"] == critic_promotion.CRITIC_ROLE_PRIMARY_RELABEL_SOURCE
    assert payload["relabel_source"] == critic_promotion.RELABEL_SOURCE_CRITIC
    assert payload["gates_a_f_green"] is True
    assert payload["offline_gain_green"] is True
    assert payload["downstream_gain_green"] is True
    assert payload["failure_reasons"] == []


def test_offline_only_gain_is_rejected() -> None:
    payload = critic_promotion.build_critic_promotion_verdict(
        offline_audit_payload={"pass": True},
        downstream_gate_payload={
            "gate_passed": False,
            "gate_status": "DIAGNOSTIC_BLOCK",
        },
        gates_a_f_bundle=_all_green_gates(),
    )

    assert payload["promotion_allowed"] is False
    assert payload["promotion_status"] == critic_promotion.PROMOTION_STATUS_BLOCK
    assert payload["critic_role"] == critic_promotion.CRITIC_ROLE_REVIEW_ONLY
    assert payload["relabel_source"] == critic_promotion.RELABEL_SOURCE_DEFAULT_MAINLINE
    assert payload["offline_gain_green"] is True
    assert payload["downstream_gain_green"] is False
    assert "downstream_gain_not_green" in payload["failure_reasons"]


def test_downstream_only_gain_is_rejected() -> None:
    payload = critic_promotion.build_critic_promotion_verdict(
        offline_audit_payload={"pass": False},
        downstream_gate_payload={"gate_passed": True, "gate_status": "DIAGNOSTIC_PASS"},
        gates_a_f_bundle=_all_green_gates(),
    )

    assert payload["promotion_allowed"] is False
    assert payload["offline_gain_green"] is False
    assert payload["downstream_gain_green"] is True
    assert payload["critic_role"] == critic_promotion.CRITIC_ROLE_REVIEW_ONLY
    assert "offline_gain_not_green" in payload["failure_reasons"]


def test_gate_a_f_not_all_green_is_rejected() -> None:
    gates = _all_green_gates()
    gates["C"] = False

    payload = critic_promotion.build_critic_promotion_verdict(
        offline_audit_payload={"pass": True},
        downstream_gate_payload={"gate_passed": True, "gate_status": "DIAGNOSTIC_PASS"},
        gates_a_f_bundle=gates,
    )

    assert payload["promotion_allowed"] is False
    assert payload["gates_a_f_green"] is False
    assert payload["critic_role"] == critic_promotion.CRITIC_ROLE_REVIEW_ONLY
    assert payload["gates_a_f"]["C"] is False
    assert "gate_c_not_green" in payload["failure_reasons"]


def test_missing_inputs_default_to_review_only_diagnostic_only_behavior() -> None:
    payload = critic_promotion.build_critic_promotion_verdict()

    assert payload["promotion_allowed"] is False
    assert payload["promotion_status"] == critic_promotion.PROMOTION_STATUS_BLOCK
    assert payload["critic_role"] == critic_promotion.CRITIC_ROLE_REVIEW_ONLY
    assert payload["relabel_source"] == critic_promotion.RELABEL_SOURCE_DEFAULT_MAINLINE
    assert payload["gates_a_f_green"] is False
    assert payload["offline_gain_green"] is False
    assert payload["downstream_gain_green"] is False
    assert "gates_a_f_bundle_missing" in payload["failure_reasons"]
    assert "offline_audit_missing" in payload["failure_reasons"]
    assert "downstream_gate_missing" in payload["failure_reasons"]


def test_upgrade_gate_script_writes_machine_readable_sidecar(tmp_path: Path) -> None:
    gate_script = _load_script_module(
        "gr00t_critic_upgrade_gate.py", "critic_upgrade_gate_script"
    )
    offline_path = tmp_path / "offline.json"
    downstream_path = tmp_path / "downstream.json"
    gates_path = tmp_path / "gates.json"
    output_path = tmp_path / "critic_upgrade_gate.json"

    offline_path.write_text(json.dumps({"pass": True}) + "\n", encoding="utf-8")
    downstream_path.write_text(
        json.dumps({"gate_passed": True, "gate_status": "DIAGNOSTIC_PASS"}) + "\n",
        encoding="utf-8",
    )
    gates_path.write_text(json.dumps(_all_green_gates()) + "\n", encoding="utf-8")

    exit_code = gate_script.main(
        [
            "--offline-audit-json",
            str(offline_path),
            "--downstream-gate-json",
            str(downstream_path),
            "--gates-a-f-json",
            str(gates_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_payload["promotion_allowed"] is True
    assert (
        written_payload["critic_role"]
        == critic_promotion.CRITIC_ROLE_PRIMARY_RELABEL_SOURCE
    )
    assert written_payload["relabel_source"] == critic_promotion.RELABEL_SOURCE_CRITIC
