from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HASH_LOCK = (
    REPO_ROOT
    / "agent/artifacts/stage1_v22_blind_calibration_iter8_20260426T_nextZ/openpi/v22_preregistration/v22_preregistration_hash_lock.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_hash_lock_loader_requires_byte_identical_sha(tmp_path: Path) -> None:
    from work.openpi.eval.v22_formal_eval_contracts import load_prereg_hash_lock

    copied_lock = tmp_path / "v22_preregistration_hash_lock.json"
    copied_lock.write_bytes(HASH_LOCK.read_bytes())
    expected_sha = _sha256(HASH_LOCK)

    lock = load_prereg_hash_lock(copied_lock, expected_sha256=expected_sha)
    assert lock.sha256 == expected_sha
    assert lock.n_per_variant == 192
    assert lock.variants == ("A", "B", "C", "X")

    payload = json.loads(copied_lock.read_text(encoding="utf-8"))
    payload["n_per_variant"] = 96
    _write_json(copied_lock, payload)

    with pytest.raises(ValueError, match="BLOCK_HASH_LOCK_SHA_MISMATCH"):
        load_prereg_hash_lock(copied_lock, expected_sha256=expected_sha)


def test_variant_authority_manifest_binds_hash_lock_sha(tmp_path: Path) -> None:
    from work.openpi.eval.v22_formal_eval_contracts import (
        load_prereg_hash_lock,
        load_variant_authority_manifest,
        validate_variant_authority_manifest,
    )

    lock = load_prereg_hash_lock(HASH_LOCK)
    manifest_path = tmp_path / "variant_authority_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "v22_variant_authority_manifest_v1",
            "formal_eval_allowed": True,
            "selected_protocol_compatible": True,
            "selected_protocol": dict(lock.selected_protocol),
            "variants_loaded": list(lock.variants),
            "hash_lock_sha256": "0" * 64,
        },
    )

    manifest = load_variant_authority_manifest(manifest_path)
    assert "BLOCK_VARIANT_AUTHORITY_HASH_LOCK_MISMATCH" in (
        validate_variant_authority_manifest(manifest, lock)
    )

