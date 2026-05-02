from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "stage1_recap_longrun_iter5_5_contract_fix_20260425T_nextZ"
COORDINATOR = REPO_ROOT / "agent" / "artifacts" / RUN_ID / "coordinator"


def test_artifact_freeze_manifest_pins_iter5_inputs_without_directory_chmod() -> None:
    manifest_path = COORDINATOR / "artifact_freeze_manifest.json"
    authority_index_path = COORDINATOR / "iter5p5_authority_index.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authority_index = json.loads(authority_index_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "iter5p5_artifact_freeze_manifest_v3"
    assert manifest["default_allow_overwrite"] is False
    assert manifest["chmod_policy"] == "per_file_only_directory_recursive_forbidden_H3"
    assert len(manifest["entries"]) >= 7

    for entry in manifest["entries"]:
        assert entry["allow_overwrite"] is False
        assert entry["chmod_a_w_scope"] == "per_file_only"
        path = REPO_ROOT / entry["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256_at_capture"]

    expected_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert authority_index["freeze_manifest_sha256"] == expected_sha
