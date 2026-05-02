from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COORDINATOR = (
    REPO_ROOT / "agent/artifacts/stage1_v22_blind_calibration_iter8_20260426T_nextZ/coordinator"
)
INPUT_CONTRACT = COORDINATOR / "w6_iter8_input_contract.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar(path: Path) -> str:
    return path.with_name(f"{path.name}.sha256").read_text(encoding="utf-8").strip()


def test_w1_partial_state_gate_is_pass_before_w2() -> None:
    payload = json.loads((COORDINATOR / "w1_substage_status.json").read_text())

    assert payload["W1_A"]["status"] == "PASS"
    assert payload["W1_B"]["status"] == "PASS"
    assert payload["W1_C"]["status"] == "PASS"


def test_iter8_input_contract_sidecars_and_pins_match() -> None:
    from work.openpi.eval.v22_calibration_contracts import (
        load_a_stock_authority_manifest,
        load_input_contract,
        validate_iter8_input_contract,
    )

    assert _sidecar(INPUT_CONTRACT) == _sha256(INPUT_CONTRACT)
    contract = load_input_contract(INPUT_CONTRACT, _sidecar(INPUT_CONTRACT))

    assert validate_iter8_input_contract(contract) == []
    assert contract.schema_version == "w6_blind_calibration_input_contract_v3"
    assert contract.run_id == "stage1_v22_blind_calibration_iter8_20260426T_nextZ"
    assert contract.candidate_id_format == "matrix_verbatim"
    assert contract.calibration_variants == ("A",)
    assert contract.optional_control_variants == ("B",)
    assert contract.forbidden_selection_variants == ("C", "X")
    assert contract.formal_v22_execution_allowed is False
    assert _sha256(contract.canonical_blind_selection_rule_path) == (
        contract.canonical_blind_selection_rule_sha256
    )
    assert _sha256(contract.a_stock_authority_manifest_path) == (
        contract.a_stock_authority_manifest_sha256
    )

    manifest = load_a_stock_authority_manifest(COORDINATOR)
    assert manifest.schema_version == "a_stock_authority_manifest_iter8_v1"
    assert manifest.semantic_role == "stock_no_RECAP_baseline"
    assert manifest.openpi_install_mechanism == "editable_install"
    assert manifest.vendored_libero_install_mechanism == "editable_install"
    assert manifest.blocking_reasons == ()

