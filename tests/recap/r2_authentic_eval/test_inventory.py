"""Tests for work.recap.r2_authentic_eval.inventory (plan v4 §3.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import work.recap.r2_authentic_eval.inventory as _inv_mod
from work.recap.r2_authentic_eval.inventory import (
    BASE_RIGHT_HAND_Q99,
    Q99_TOLERANCE,
    R2InventoryNoValidCandidates,
    R2_VALID_CELL_COUNT_EXPECTED,
    RECAP_PATH_INCLUDE_HINTS,
    _classify_label,
    _q99_matches_base,
    _sha256_file,
    discover_recap_ckpts,
    filter_valid,
    is_checkpoint_dir,
    pick_representative,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _write_mock_ckpt(
    base: Path,
    *,
    rel: str,
    n_steps: int,
    q99: tuple[float, ...] = BASE_RIGHT_HAND_Q99,
    formalize_language: bool = True,
) -> Path:
    """Create a minimal checkpoint dir under base/rel/checkpoint-{n_steps}."""
    ckpt_dir = base / rel / f"checkpoint-{n_steps}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["GR00TRecapModel"],
        "formalize_language": formalize_language,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (ckpt_dir / "processor_config.json").write_text(
        json.dumps({"processor_class": "TestProc"}), encoding="utf-8"
    )
    stats = {"unitree_g1": {"action": {"right_hand": {"q99": list(q99)}}}}
    (ckpt_dir / "statistics.json").write_text(json.dumps(stats), encoding="utf-8")
    return ckpt_dir


# ---------------------------------------------------------------------------
# Label classification tests
# ---------------------------------------------------------------------------


def test_classify_recap_label_includes_recap_substring_paths() -> None:
    """A path with 'recap' and no negative tokens receives label='RECAP'."""
    p = Path("/artifacts/gr00t_recap_live/stage1/checkpoint-1000")
    assert _classify_label(p) == "RECAP"


def test_classify_recap_label_excludes_pure_sft_paths() -> None:
    """A path containing 'pure_sft' (P-A negative token) receives label='OTHER'."""
    p = Path("/artifacts/gr00t_recap_live/probes/probe_A_pure_sft_control/training_run/checkpoint-3300")
    assert _classify_label(p) == "OTHER"


def test_classify_recap_label_excludes_hf_patches_paths() -> None:
    """A path containing 'hf_patches/' (P-A negative token) receives label='OTHER'."""
    p = Path("/artifacts/gr00t_recap_live/hf_patches/snapshot/checkpoint-0")
    assert _classify_label(p) == "OTHER"


# ---------------------------------------------------------------------------
# Q99 fingerprint tests
# ---------------------------------------------------------------------------


def test_q99_tolerance_edges() -> None:
    """1e-6 boundary: BASE + 1e-6 is still valid; BASE + 1e-6 + epsilon is not."""
    exact_plus = tuple(b + Q99_TOLERANCE for b in BASE_RIGHT_HAND_Q99)
    assert _q99_matches_base(exact_plus), "BASE + Q99_TOLERANCE should match"
    beyond = tuple(b + Q99_TOLERANCE + 1e-12 for b in BASE_RIGHT_HAND_Q99)
    assert not _q99_matches_base(beyond), "BASE + Q99_TOLERANCE + epsilon should not match"


def test_q99_drift_marks_invalid(tmp_path: Path) -> None:
    """q99 ≈ 0.0117 (far from base) → is_valid=False, invalid_reason='stats_drift'."""
    drifted = tuple(0.0117 for _ in BASE_RIGHT_HAND_Q99)
    _write_mock_ckpt(tmp_path, rel="recap_run", n_steps=1000, q99=drifted)
    results = discover_recap_ckpts(tmp_path)
    assert len(results) == 1
    assert results[0].label == "RECAP"
    assert results[0].is_valid is False
    assert results[0].invalid_reason == "stats_drift"


def test_q99_matches_base_returns_true_at_exact_match() -> None:
    """Exact BASE_RIGHT_HAND_Q99 values return True."""
    assert _q99_matches_base(BASE_RIGHT_HAND_Q99)


# ---------------------------------------------------------------------------
# is_checkpoint_dir tests
# ---------------------------------------------------------------------------


def test_is_checkpoint_dir_requires_all_three_files(tmp_path: Path) -> None:
    """All three REQUIRED_CKPT_FILES must be present for is_checkpoint_dir=True."""
    d = tmp_path / "ckpt"
    d.mkdir()
    assert not is_checkpoint_dir(d)
    (d / "config.json").write_text("{}")
    assert not is_checkpoint_dir(d)
    (d / "processor_config.json").write_text("{}")
    assert not is_checkpoint_dir(d)
    (d / "statistics.json").write_text("{}")
    assert is_checkpoint_dir(d)


# ---------------------------------------------------------------------------
# SHA-256 stability
# ---------------------------------------------------------------------------


def test_sha256_stability(tmp_path: Path) -> None:
    """Same file produces the same 64-char hex digest on repeated calls."""
    f = tmp_path / "sample.bin"
    f.write_bytes(b'{"architecture": "GR00TRecapModel"}')
    h1 = _sha256_file(f)
    h2 = _sha256_file(f)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


# ---------------------------------------------------------------------------
# B-IND8 sanity token tests
# ---------------------------------------------------------------------------


def test_inventory_keeps_g3_conditioned_continuation_after_sanity(
    tmp_path: Path,
) -> None:
    """B-IND8: 'after_sanity' in path does NOT contain '_sanity_check' → kept as RECAP."""
    _write_mock_ckpt(
        tmp_path,
        rel="g3_conditioned_continuation_after_sanity_20260430_131809",
        n_steps=6600,
    )
    results = discover_recap_ckpts(tmp_path)
    assert len(results) == 1
    assert results[0].label == "RECAP"
    assert results[0].is_valid is True


def test_inventory_excludes_synthetic_sanity_check_dir(tmp_path: Path) -> None:
    """'g_test_sanity_check_run' contains '_sanity_check' → rejected as OTHER."""
    _write_mock_ckpt(tmp_path, rel="g_test_sanity_check_run", n_steps=1000)
    results = discover_recap_ckpts(tmp_path)
    assert len(results) == 1
    assert results[0].label == "OTHER"
    assert results[0].is_valid is False
    assert results[0].invalid_reason == "not_recap"


# ---------------------------------------------------------------------------
# Sort-order consistency (A4)
# ---------------------------------------------------------------------------


def test_inventory_and_pick_representative_sort_orders_agree(
    tmp_path: Path,
) -> None:
    """A4: discover_recap_ckpts()[0] == pick_representative(filter_valid(...))."""
    for name in ("alpha", "bravo", "charlie", "delta", "echo"):
        _write_mock_ckpt(tmp_path, rel=f"recap_{name}", n_steps=1000)
    all_ckpts = discover_recap_ckpts(tmp_path)
    assert len(all_ckpts) == 5
    assert all(c.is_valid for c in all_ckpts), "all mock ckpts should be valid"
    rep = pick_representative(filter_valid(all_ckpts))
    assert all_ckpts[0] == rep


def test_pick_representative_highest_n_train_steps_then_lex(
    tmp_path: Path,
) -> None:
    """pick_representative returns highest n_train_steps; no mtime tiebreak (A4)."""
    _write_mock_ckpt(tmp_path, rel="recap_zz", n_steps=1000)
    _write_mock_ckpt(tmp_path, rel="recap_aa", n_steps=2000)
    all_ckpts = discover_recap_ckpts(tmp_path)
    rep = pick_representative(filter_valid(all_ckpts))
    assert rep.n_train_steps == 2000


def test_pick_representative_raises_on_empty_valid() -> None:
    """pick_representative raises R2InventoryNoValidCandidates on empty input."""
    with pytest.raises(R2InventoryNoValidCandidates):
        pick_representative([])


# ---------------------------------------------------------------------------
# RECAP_PATH_INCLUDE_HINTS documentation check
# ---------------------------------------------------------------------------


def test_recap_path_include_hints_is_documented_as_non_authoritative() -> None:
    """RECAP_PATH_INCLUDE_HINTS docstring in inventory.py must say 'non-authoritative'."""
    src = Path(_inv_mod.__file__).read_text(encoding="utf-8")
    idx = src.find("RECAP_PATH_INCLUDE_HINTS")
    assert idx != -1, "RECAP_PATH_INCLUDE_HINTS constant not found in source"
    # Look at a generous window after the constant definition
    window = src[idx : idx + 500].lower()
    assert "non-authoritative" in window, (
        "RECAP_PATH_INCLUDE_HINTS docstring must contain 'non-authoritative'"
    )


# ---------------------------------------------------------------------------
# V4-FIX-1: R2_VALID_CELL_COUNT_EXPECTED constant
# ---------------------------------------------------------------------------


def test_R2_VALID_CELL_COUNT_EXPECTED_constant_present() -> None:
    """V4-FIX-1: constant == 5 and docstring contains plan-time intent phrases."""
    assert R2_VALID_CELL_COUNT_EXPECTED == 5
    src = Path(_inv_mod.__file__).read_text(encoding="utf-8")
    idx = src.find("R2_VALID_CELL_COUNT_EXPECTED")
    assert idx != -1
    # Check a generous window covering the variable + its docstring
    window = src[idx : idx + 700]
    assert "plan-time inventory expectation" in window, (
        "docstring must contain 'plan-time inventory expectation'"
    )
    assert "NOT a runtime invariant" in window, (
        "docstring must contain 'NOT a runtime invariant'"
    )
