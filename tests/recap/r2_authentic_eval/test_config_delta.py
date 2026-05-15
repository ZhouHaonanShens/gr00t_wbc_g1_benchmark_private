"""Tests for R2.0.5 config-delta classification and acknowledgment gates."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from work.recap.r2_authentic_eval.config_delta import (
    ADDITIONAL_FIELDS_DIFFER,
    FORMALIZE_LANGUAGE_PATHS,
    ONLY_FORMALIZE_LANGUAGE,
    AcknowledgmentMissingError,
    audit_inventory,
    audit_one_ckpt,
    classify_config_delta,
    require_acknowledgment,
)
from work.recap.r2_authentic_eval.reports.config_delta_report import (
    render_config_delta_subsection,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _make_ckpt(
    root: Path,
    *,
    formalize_language: bool,
    arch: str = "Gr00tN1d6",
    hidden_size: int = 1024,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "config.json",
        {
            "architectures": [arch],
            "formalize_language": formalize_language,
            "hidden_size": hidden_size,
        },
    )
    _write_json(
        root / "processor_config.json",
        {"processor_kwargs": {"formalize_language": formalize_language}},
    )
    return root


def test_classify_config_delta_only_formalize_language(tmp_path: Path) -> None:
    source = _make_ckpt(tmp_path / "source", formalize_language=True)
    target = _make_ckpt(tmp_path / "target", formalize_language=False)

    result = classify_config_delta(source, target, allowed_paths=set(FORMALIZE_LANGUAGE_PATHS))

    assert result == ONLY_FORMALIZE_LANGUAGE


def test_classify_config_delta_additional_fields_differ(tmp_path: Path) -> None:
    source = _make_ckpt(tmp_path / "source", formalize_language=True, hidden_size=2048)
    target = _make_ckpt(tmp_path / "target", formalize_language=False, hidden_size=1024)

    result = classify_config_delta(source, target, allowed_paths=set(FORMALIZE_LANGUAGE_PATHS))

    assert result == ADDITIONAL_FIELDS_DIFFER


def test_classify_config_delta_returns_Literal_string(tmp_path: Path) -> None:
    source = _make_ckpt(tmp_path / "source", formalize_language=True)
    target = _make_ckpt(tmp_path / "target", formalize_language=False)

    result = classify_config_delta(source, target, allowed_paths=set(FORMALIZE_LANGUAGE_PATHS))

    assert result in (ONLY_FORMALIZE_LANGUAGE, ADDITIONAL_FIELDS_DIFFER)
    assert isinstance(result, str)


def test_audit_one_ckpt_returns_row_with_outside_paths(tmp_path: Path) -> None:
    source = _make_ckpt(tmp_path / "source", formalize_language=True, arch="GR00TRecapModel")
    target = _make_ckpt(tmp_path / "target", formalize_language=False, arch="Gr00tN1d6")

    row = audit_one_ckpt(source, allowed_paths=set(FORMALIZE_LANGUAGE_PATHS), target_ckpt=target)

    assert row["classification"] == ADDITIONAL_FIELDS_DIFFER
    assert row["architectures_mismatch"] is True
    assert "config.json:architectures" in row["outside_paths"]


def test_audit_inventory_aggregates_rows_into_inventory_dict(tmp_path: Path) -> None:
    source_a = _make_ckpt(tmp_path / "source_a", formalize_language=True)
    source_b = _make_ckpt(tmp_path / "source_b", formalize_language=True, hidden_size=4096)
    target = _make_ckpt(tmp_path / "target", formalize_language=False, hidden_size=1024)

    inventory = audit_inventory(
        [source_a, source_b],
        allowed_paths=set(FORMALIZE_LANGUAGE_PATHS),
        target_ckpt=target,
    )

    assert inventory["row_count"] == 2
    assert inventory["summary"][ONLY_FORMALIZE_LANGUAGE] == 1
    assert inventory["summary"][ADDITIONAL_FIELDS_DIFFER] == 1


def test_audit_inventory_writes_attention_md_on_additional_fields_differ(
    tmp_path: Path,
) -> None:
    source = _make_ckpt(tmp_path / "source", formalize_language=True, arch="GR00TRecapModel")
    target = _make_ckpt(tmp_path / "target", formalize_language=False, arch="Gr00tN1d6")
    dossier = tmp_path / "dossier"

    inventory = audit_inventory(
        [source],
        allowed_paths=set(FORMALIZE_LANGUAGE_PATHS),
        target_ckpt=target,
        dossier_dir=dossier,
    )

    assert inventory["attention"]["status"] == "pending"
    assert (dossier / "config_delta_inventory.json").is_file()
    assert (dossier / "whitelist_audit.attention.md").is_file()
    assert (dossier / "r2_0_5_user_attention.md").is_file()


def test_audit_inventory_preserves_attention_mtime_when_content_unchanged(
    tmp_path: Path,
) -> None:
    source = _make_ckpt(tmp_path / "source", formalize_language=True, arch="GR00TRecapModel")
    target = _make_ckpt(tmp_path / "target", formalize_language=False, arch="Gr00tN1d6")
    dossier = tmp_path / "dossier"

    audit_inventory(
        [source],
        allowed_paths=set(FORMALIZE_LANGUAGE_PATHS),
        target_ckpt=target,
        dossier_dir=dossier,
    )
    attention = dossier / "whitelist_audit.attention.md"
    os.utime(attention, (1000.0, 1000.0))

    audit_inventory(
        [source],
        allowed_paths=set(FORMALIZE_LANGUAGE_PATHS),
        target_ckpt=target,
        dossier_dir=dossier,
    )

    assert attention.stat().st_mtime == 1000.0


def test_render_config_delta_subsection_lists_allowed_paths_and_ack_status(
    tmp_path: Path,
) -> None:
    source = _make_ckpt(tmp_path / "source", formalize_language=True, arch="GR00TRecapModel")
    target = _make_ckpt(tmp_path / "target", formalize_language=False, arch="Gr00tN1d6")
    inventory = audit_inventory(
        [source],
        allowed_paths=set(FORMALIZE_LANGUAGE_PATHS),
        target_ckpt=target,
    )
    inventory["attention"] = {"status": "acknowledged@2026-05-11T00:00:00Z"}

    md = render_config_delta_subsection(inventory)

    assert "allowed_paths" in md
    assert "`config.json:formalize_language`" in md
    assert "acknowledged@2026-05-11T00:00:00Z" in md


def test_require_acknowledgment_accepts_with_margin_above_1s(tmp_path: Path) -> None:
    attention = tmp_path / "whitelist_audit.attention.md"
    ack = tmp_path / "whitelist_audit.acknowledged.md"
    attention.write_text("attention\n", encoding="utf-8")
    ack.write_text("ack\n", encoding="utf-8")
    os.utime(attention, (1000.0, 1000.0))
    os.utime(ack, (1001.1, 1001.1))

    require_acknowledgment(attention, ack)


def test_require_acknowledgment_rejects_with_margin_below_1s(tmp_path: Path) -> None:
    attention = tmp_path / "whitelist_audit.attention.md"
    ack = tmp_path / "whitelist_audit.acknowledged.md"
    attention.write_text("attention\n", encoding="utf-8")
    ack.write_text("ack\n", encoding="utf-8")
    os.utime(attention, (1000.0, 1000.0))
    os.utime(ack, (1001.0, 1001.0))

    with pytest.raises(AcknowledgmentMissingError):
        require_acknowledgment(attention, ack)


def test_require_acknowledgment_rejects_when_ack_missing(tmp_path: Path) -> None:
    attention = tmp_path / "whitelist_audit.attention.md"
    attention.write_text("attention\n", encoding="utf-8")

    with pytest.raises(AcknowledgmentMissingError):
        require_acknowledgment(attention, tmp_path / "whitelist_audit.acknowledged.md")


def test_workflow_never_writes_acknowledgment() -> None:
    repo_root = Path(__file__).parents[3]
    for rel in (
        "work/recap/r2_authentic_eval/_workflow.py",
        "work/recap/r2_authentic_eval/cli.py",
    ):
        tree = ast.parse((repo_root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_writes_acknowledgment(node):
                raise AssertionError(f"{rel} writes an acknowledgment file")


def test_acknowledgment_mtime_guard_with_ntfs3_quirks(tmp_path: Path) -> None:
    attention = tmp_path / "whitelist_audit.attention.md"
    ack = tmp_path / "whitelist_audit.acknowledged.md"
    attention.write_text("attention\n", encoding="utf-8")
    ack.write_text("ack\n", encoding="utf-8")
    real_stat = Path.stat

    def fake_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        values = list(result)
        if path == attention:
            values[8] = 1000.0
        elif path == ack:
            values[8] = 1001.0001
        return os.stat_result(values)

    with mock.patch.object(Path, "stat", fake_stat):
        require_acknowledgment(attention, ack)


def _call_writes_acknowledgment(node: ast.Call) -> bool:
    literal_args = [
        arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]
    if any(value.endswith(".acknowledged.md") for value in literal_args):
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "write_text",
            "write_bytes",
            "touch",
        }:
            return True
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            return any(mode in literal_args for mode in ("w", "wb", "a"))
    return False
