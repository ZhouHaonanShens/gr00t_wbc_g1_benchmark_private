from __future__ import annotations

import json
from pathlib import Path

import pytest

from work.recap.r1_repro.ckpt_variant_audit import (
    MISSING_VALUE,
    VariantAmbiguous,
    audit_variant,
    classify_risk,
    confirm_variant_uniqueness,
    inventory_symlinks,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_audit_variant_emits_keypath_tuples(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    variant_root = tmp_path / "variant"
    _write_json(raw_root / "config.json", {"same": True, "model": {"depth": 2}})
    _write_json(
        variant_root / "config.json",
        {"same": True, "model": {"depth": 3}, "new_key": "variant-only"},
    )

    diff = audit_variant(variant_root, raw_root, ["config.json"])

    assert diff["config.json"] == [
        ("model.depth", 3, 2),
        ("new_key", "variant-only", MISSING_VALUE),
    ]


def test_classify_risk_high_keys_table() -> None:
    diff = {
        "config.json": [
            ("language_model.hidden_size", 1, 2),
            ("n_action_steps", 20, 10),
            ("misc", "formalize_language=False", "raw"),
        ]
    }

    risk = classify_risk(diff)

    assert [entry[1] for entry in risk["HIGH"]] == [
        "language_model.hidden_size",
        "n_action_steps",
        "misc",
    ]


def test_classify_risk_medium_keys_table() -> None:
    diff = {
        "config.json": [
            ("transformers_version", "4.0", "4.1"),
            ("unknown_top.value", 1, 2),
        ]
    }

    risk = classify_risk(diff)

    assert [entry[1] for entry in risk["MEDIUM"]] == [
        "transformers_version",
        "unknown_top.value",
    ]


def test_classify_risk_low_keys_table() -> None:
    diff = {
        "processor_config.json": [
            ("_commit_hash", "a", "b"),
            ("telemetry_note", "old", "new"),
            ("comment", "old", "new"),
        ]
    }

    risk = classify_risk(diff)

    assert [entry[1] for entry in risk["LOW"]] == [
        "_commit_hash",
        "telemetry_note",
        "comment",
    ]


def test_inventory_symlinks_separates_local_overrides_from_symlinks(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    variant_root = tmp_path / "variant"
    raw_weights = raw_root / "model.safetensors"
    raw_weights.parent.mkdir(parents=True, exist_ok=True)
    raw_weights.write_text("weights\n", encoding="utf-8")
    variant_root.mkdir(parents=True, exist_ok=True)
    (variant_root / "model.safetensors").symlink_to(raw_weights)
    (variant_root / "config.json").write_text("{}", encoding="utf-8")

    inventory = inventory_symlinks(variant_root, raw_root)

    assert [entry["name"] for entry in inventory["symlinks"]] == [
        "model.safetensors"
    ]
    assert inventory["symlinks"][0]["target_sha256"]
    assert [entry["name"] for entry in inventory["non_symlink_overrides"]] == [
        "config.json"
    ]


def test_confirm_variant_uniqueness_raises_on_multiple(tmp_path: Path) -> None:
    first = tmp_path / "a" / "formalize_language=False"
    second = tmp_path / "b" / "formalize_language=False"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    with pytest.raises(VariantAmbiguous):
        confirm_variant_uniqueness([str(tmp_path / "**" / "formalize_language=False")])
