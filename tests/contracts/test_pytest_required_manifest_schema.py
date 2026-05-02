from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "stage1_recap_longrun_iter5_5_contract_fix_20260425T_nextZ"
VERIFIER = REPO_ROOT / "agent" / "artifacts" / RUN_ID / "verifier"

AUTHORIZED_ADDITIONS = [
    "tests/contracts/test_canonical_blind_selection_rule_schema.py",
    "tests/contracts/test_artifact_freeze_manifest_schema.py",
    "tests/contracts/test_pytest_required_manifest_schema.py",
    "tests/hooks/test_iter5p5_p6_bash_hook_self_test.py",
    "tests/hooks/test_iter5p5_claim_language_hook_jsonpointer.py",
]


def test_pytest_required_manifest_uses_runtime_formula_and_present_files() -> None:
    payload = json.loads((VERIFIER / "pytest_required_manifest.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == "iter5p5_pytest_required_manifest_v3"
    assert payload["expected_test_count"] >= 208
    assert payload["additions_authorized"] == AUTHORIZED_ADDITIONS
    assert payload["additions_authorized_count"] == 5
    assert payload["iter5p5_contract_test_files_present_count"] == 5
    assert payload["iter5p5_contract_test_files_present"] == AUTHORIZED_ADDITIONS
    assert payload["expected_total_count_formula"] == "expected_test_count + len(additions_authorized)"
    assert "expected_total_count" not in payload

    for rel_path in AUTHORIZED_ADDITIONS:
        assert (REPO_ROOT / rel_path).is_file()
