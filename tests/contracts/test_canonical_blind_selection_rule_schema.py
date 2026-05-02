from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "stage1_recap_longrun_iter5_5_contract_fix_20260425T_nextZ"
COORDINATOR = REPO_ROOT / "agent" / "artifacts" / RUN_ID / "coordinator"


def test_canonical_blind_selection_rule_schema_and_sidecar() -> None:
    rule_path = COORDINATOR / "canonical_blind_selection_rule.json"
    sidecar_path = COORDINATOR / "canonical_blind_selection_rule.sha256"

    assert rule_path.is_file()
    assert sidecar_path.is_file()

    payload = json.loads(rule_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "iter5p5_canonical_blind_selection_rule_v1"
    assert payload["run_id"] == RUN_ID
    assert payload["consumers"] == ["W3p5", "W5p5"]

    rule = payload["rule"]
    criteria = rule["selection_criteria"]
    assert criteria["calibration_target"] == "DESATURATED_FOUND"
    assert criteria["variant_codes_allowed_in_selection"] == ["A", "B"]
    assert criteria["variant_codes_forbidden_in_selection"] == ["C", "X"]
    assert "formal_v22_execution_allowed == false" in rule["anti_leakage_invariants"]

    expected_sha = hashlib.sha256(rule_path.read_bytes()).hexdigest()
    sidecar_sha, sidecar_rel = sidecar_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    assert sidecar_sha == expected_sha
    assert sidecar_rel == f"agent/artifacts/{RUN_ID}/coordinator/canonical_blind_selection_rule.json"


def test_w6_precondition_references_current_canonical_rule_hash() -> None:
    rule_path = COORDINATOR / "canonical_blind_selection_rule.json"
    sidecar_path = COORDINATOR / "canonical_blind_selection_rule.sha256"
    precondition_path = COORDINATOR / "w6_precondition_check.json"
    contract_path = COORDINATOR / "w6_input_contract.json"

    expected_sha = hashlib.sha256(rule_path.read_bytes()).hexdigest()
    sidecar_sha = sidecar_path.read_text(encoding="utf-8").split(maxsplit=1)[0]
    precondition = json.loads(precondition_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert sidecar_sha == expected_sha
    assert precondition["expected_sha256"] == expected_sha
    assert precondition["actual_sha256"] == expected_sha
    assert precondition["hash_match"] is True
    assert precondition["precondition_pass"] is True
    assert contract["canonical_rule_sha256"] == expected_sha
    assert contract["fallback_paths_allowed"] is False
