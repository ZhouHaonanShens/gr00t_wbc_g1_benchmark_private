"""Tests for work.recap.r2_authentic_eval.ckpt_config_swap (plan §3.1, brief §5.3)."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from unittest import mock

import pytest

from work.recap.r2_authentic_eval.ckpt_config_swap import (
    ALLOW_LIST_TO_OVERRIDE,
    LINK_STRATEGY,
    PROTECTED_FILES,
    REQUIRED_CKPT_FILES,
    SWAP_PROVENANCE_FILENAME,
    WEIGHT_GLOB,
    CkptByteIdentityViolation,
    CkptRawHfMissingArtifact,
    CkptSourceMissingArtifact,
    CkptSrcMutatedDuringSwap,
    ConfigSwapResult,
    FieldSwapResult,
    materialise_field_targeted_swap,
    materialise_swap_ckpt,
    _walk_recursive_sha_table,
)
import work.recap.r2_authentic_eval.ckpt_config_swap as _swap_mod

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeCkpt:
    """Minimal duck-typed stand-in for TrainedCheckpoint in tests."""

    abs_path: Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_src_ckpt(base: Path) -> _FakeCkpt:
    """Build a synthetic sharded checkpoint dir under *base*."""
    src = base / "src_ckpt"
    src.mkdir(parents=True, exist_ok=True)
    # 3-shard synthetic safetensors — distinct byte content per shard
    for i in range(1, 4):
        (src / f"model-0000{i}-of-00003.safetensors").write_bytes(
            f"shard-content-{i}".encode()
        )
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 3}, "weight_map": {}}), encoding="utf-8"
    )
    (src / "embodiment_id.json").write_text(
        json.dumps({"embodiment": "unitree_g1"}), encoding="utf-8"
    )
    (src / "statistics.json").write_text(
        json.dumps({"unitree_g1": {"action": {"right_hand": {"q99": [1.5]}}}}),
        encoding="utf-8",
    )
    (src / "config.json").write_text(
        json.dumps({"model_type": "trained", "formalize_language": True}), encoding="utf-8"
    )
    (src / "processor_config.json").write_text(
        json.dumps({"processor_class": "trained_proc"}), encoding="utf-8"
    )
    (src / "experiment_cfg").mkdir()
    (src / "experiment_cfg" / "metadata.json").write_text(
        json.dumps({"run_id": "test-run-001"}), encoding="utf-8"
    )
    (src / "optimizer.pt").write_bytes(b"optimizer-stub")
    (src / "rng_state.pth").write_bytes(b"rng-stub")
    return _FakeCkpt(abs_path=src)


def _make_raw_hf(base: Path) -> Path:
    """Build a synthetic raw HF snapshot under *base* with distinct config content."""
    raw = base / "raw_hf"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "config.json").write_text(
        json.dumps({"model_type": "raw_hf", "formalize_language": False}), encoding="utf-8"
    )
    (raw / "processor_config.json").write_text(
        json.dumps({"processor_class": "raw_hf_proc"}), encoding="utf-8"
    )
    return raw


@pytest.fixture
def src_ckpt(tmp_path: Path) -> _FakeCkpt:
    return _make_src_ckpt(tmp_path)


@pytest.fixture
def raw_hf(tmp_path: Path) -> Path:
    return _make_raw_hf(tmp_path)


@pytest.fixture
def swap_root(tmp_path: Path) -> Path:
    p = tmp_path / "swap"
    p.mkdir()
    return p


@pytest.fixture
def result(src_ckpt: _FakeCkpt, raw_hf: Path, swap_root: Path) -> ConfigSwapResult:
    return materialise_swap_ckpt(src_ckpt, raw_hf, swap_root)


# ---------------------------------------------------------------------------
# Copy mechanic (not hardlink)
# ---------------------------------------------------------------------------


def test_materialise_swap_ckpt_uses_copytree_not_hardlink(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    """Swap files must have different inodes — copies, not hardlinks."""
    shard = "model-00001-of-00003.safetensors"
    src_ino = (src_ckpt.abs_path / shard).stat().st_ino
    swap_ino = (result.swap_dir / shard).stat().st_ino
    assert src_ino != swap_ino, "Shard must be a copy, not a hardlink"
    assert result.link_strategy == LINK_STRATEGY == "copytree"


# ---------------------------------------------------------------------------
# Shard and protected-file preservation (brief §5.3 step 4 + 6a/6b)
# ---------------------------------------------------------------------------


def test_materialise_swap_ckpt_preserves_all_shards(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    """All three shards must be present in swap dir with byte-identical content."""
    for i in range(1, 4):
        name = f"model-0000{i}-of-00003.safetensors"
        assert (result.swap_dir / name).read_bytes() == (src_ckpt.abs_path / name).read_bytes()


def test_materialise_swap_ckpt_preserves_safetensors_index(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    name = "model.safetensors.index.json"
    assert _sha((result.swap_dir / name).read_bytes()) == _sha((src_ckpt.abs_path / name).read_bytes())


def test_materialise_swap_ckpt_preserves_embodiment_id(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    name = "embodiment_id.json"
    assert _sha((result.swap_dir / name).read_bytes()) == _sha((src_ckpt.abs_path / name).read_bytes())


def test_materialise_swap_ckpt_preserves_statistics(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    name = "statistics.json"
    assert _sha((result.swap_dir / name).read_bytes()) == _sha((src_ckpt.abs_path / name).read_bytes())


# ---------------------------------------------------------------------------
# ALLOW_LIST_TO_OVERRIDE override (brief §5.3 step 5 + 6c)
# ---------------------------------------------------------------------------


def test_materialise_swap_ckpt_overwrites_config_from_raw_hf(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt, raw_hf: Path
) -> None:
    name = "config.json"
    swap_sha = _sha((result.swap_dir / name).read_bytes())
    raw_sha = _sha((raw_hf / name).read_bytes())
    src_sha = _sha((src_ckpt.abs_path / name).read_bytes())
    assert swap_sha == raw_sha, "swap/config.json must match raw_hf/config.json"
    assert swap_sha != src_sha, "swap/config.json must DIFFER from source/config.json"


def test_materialise_swap_ckpt_overwrites_processor_config_from_raw_hf(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt, raw_hf: Path
) -> None:
    name = "processor_config.json"
    swap_sha = _sha((result.swap_dir / name).read_bytes())
    raw_sha = _sha((raw_hf / name).read_bytes())
    src_sha = _sha((src_ckpt.abs_path / name).read_bytes())
    assert swap_sha == raw_sha
    assert swap_sha != src_sha


# ---------------------------------------------------------------------------
# Deep subtree preservation (experiment_cfg / optimizer / rng)
# ---------------------------------------------------------------------------


def test_materialise_swap_ckpt_preserves_experiment_cfg_subtree(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    """experiment_cfg/metadata.json must be byte-identical — proves copytree depth."""
    src_bytes = (src_ckpt.abs_path / "experiment_cfg" / "metadata.json").read_bytes()
    swap_bytes = (result.swap_dir / "experiment_cfg" / "metadata.json").read_bytes()
    assert swap_bytes == src_bytes


def test_materialise_swap_ckpt_preserves_optimizer_and_rng_state(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    for name in ("optimizer.pt", "rng_state.pth"):
        assert (result.swap_dir / name).exists(), f"{name} missing from swap dir"
        assert (result.swap_dir / name).read_bytes() == (src_ckpt.abs_path / name).read_bytes()


# ---------------------------------------------------------------------------
# Pre-swap snapshot coverage (brief §5.3 step 3)
# ---------------------------------------------------------------------------


def test_materialise_swap_ckpt_records_pre_swap_snapshot_of_every_src_file(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    """src_pre_swap_sha_table must include every file recursively under source."""
    expected = {
        str(p.relative_to(src_ckpt.abs_path))
        for p in src_ckpt.abs_path.rglob("*")
        if p.is_file()
    }
    assert set(result.src_pre_swap_sha_table.keys()) == expected


# ---------------------------------------------------------------------------
# Post-swap source re-hash (brief §5.3 step 7)
# ---------------------------------------------------------------------------


def test_materialise_swap_ckpt_re_hashes_src_post_swap_to_prove_untouched(
    result: ConfigSwapResult, src_ckpt: _FakeCkpt
) -> None:
    """Source must be untouched after a successful swap."""
    post_table = _walk_recursive_sha_table(src_ckpt.abs_path)
    assert post_table == result.src_pre_swap_sha_table


def test_materialise_swap_ckpt_raises_on_src_mutated_during_swap(
    tmp_path: Path,
) -> None:
    """CkptSrcMutatedDuringSwap raised when post-swap source re-hash mismatches."""
    ckpt = _make_src_ckpt(tmp_path / "src2")
    raw = _make_raw_hf(tmp_path / "raw2")
    swap_r = tmp_path / "swap2"
    swap_r.mkdir()

    _real_walk = _swap_mod._walk_recursive_sha_table
    _src_call_count: list[int] = [0]

    def _walk_inject_mismatch(root: Path) -> dict[str, str]:
        table = _real_walk(root)
        if root == ckpt.abs_path:
            _src_call_count[0] += 1
            if _src_call_count[0] == 2:  # second call = post-swap re-hash
                corrupted = dict(table)
                first_key = sorted(corrupted)[0]
                corrupted[first_key] = "a" * 64
                return corrupted
        return table

    with mock.patch.object(_swap_mod, "_walk_recursive_sha_table", side_effect=_walk_inject_mismatch):
        with pytest.raises(CkptSrcMutatedDuringSwap):
            materialise_swap_ckpt(ckpt, raw, swap_r)


# ---------------------------------------------------------------------------
# Audit-failure detection
# ---------------------------------------------------------------------------


def test_materialise_swap_ckpt_raises_on_post_copy_corruption(
    src_ckpt: _FakeCkpt, raw_hf: Path, swap_root: Path
) -> None:
    """CkptByteIdentityViolation raised when swap shard sha doesn't match source."""
    _real_sha = _swap_mod._sha256_hex
    shard_name = "model-00001-of-00003.safetensors"

    def _sha_inject_swap_mismatch(path: Path) -> str:
        # Return a fake sha when computing the shard in the swap dir (not source)
        if path.name == shard_name and path.parent != src_ckpt.abs_path:
            return "b" * 64
        return _real_sha(path)

    with mock.patch.object(_swap_mod, "_sha256_hex", side_effect=_sha_inject_swap_mismatch):
        with pytest.raises(CkptByteIdentityViolation):
            materialise_swap_ckpt(src_ckpt, raw_hf, swap_root)


def test_materialise_swap_ckpt_raises_when_required_ckpt_files_missing(
    tmp_path: Path, raw_hf: Path, swap_root: Path
) -> None:
    """CkptSourceMissingArtifact raised when a REQUIRED_CKPT_FILE is absent."""
    ckpt = _make_src_ckpt(tmp_path / "src_miss")
    (ckpt.abs_path / "embodiment_id.json").unlink()
    with pytest.raises(CkptSourceMissingArtifact):
        materialise_swap_ckpt(ckpt, raw_hf, swap_root)


def test_materialise_swap_ckpt_raises_when_allow_list_files_missing_in_raw_hf(
    src_ckpt: _FakeCkpt, tmp_path: Path, swap_root: Path
) -> None:
    """CkptRawHfMissingArtifact raised when ALLOW_LIST file absent from raw HF."""
    raw = _make_raw_hf(tmp_path / "raw_miss")
    (raw / "config.json").unlink()
    with pytest.raises(CkptRawHfMissingArtifact):
        materialise_swap_ckpt(src_ckpt, raw, swap_root)


# ---------------------------------------------------------------------------
# Provenance JSON
# ---------------------------------------------------------------------------


def test_materialise_swap_ckpt_writes_provenance_json(result: ConfigSwapResult) -> None:
    """_swap_provenance.json must exist with all required fields."""
    prov = result.swap_dir / SWAP_PROVENANCE_FILENAME
    assert prov.exists(), "_swap_provenance.json not written"
    payload = json.loads(prov.read_text(encoding="utf-8"))
    for key in (
        "source_ckpt_root",
        "raw_hf_root",
        "swap_dir",
        "materialised_at_utc",
        "src_pre_swap_sha_table",
        "swap_post_swap_sha_table",
        "weight_glob_sha_pairs",
        "protected_files_sha_pairs",
        "allow_list_files_sha_pairs",
        "link_strategy",
    ):
        assert key in payload, f"Missing key {key!r} in _swap_provenance.json"
    assert payload["link_strategy"] == "copytree"
    assert isinstance(payload["src_pre_swap_sha_table"], dict)
    assert len(payload["weight_glob_sha_pairs"]) == 3  # 3 shards


def test_materialise_swap_ckpt_swap_dir_inside_swap_root(
    result: ConfigSwapResult, swap_root: Path
) -> None:
    assert str(result.swap_dir).startswith(str(swap_root))


# ---------------------------------------------------------------------------
# Field-targeted config swap (R2.0.5 brief §4.4)
# ---------------------------------------------------------------------------


def _make_field_swap_pair(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "field_source"
    target = tmp_path / "field_target"
    shutil.copytree(_make_src_ckpt(tmp_path / "source_base").abs_path, source)
    shutil.copytree(_make_src_ckpt(tmp_path / "target_base").abs_path, target)
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["SourceModel"],
                "formalize_language": True,
                "nested": {"keep": "source", "copy_me": 7},
            }
        ),
        encoding="utf-8",
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TargetModel"],
                "formalize_language": False,
                "nested": {"keep": "target", "copy_me": 1},
            }
        ),
        encoding="utf-8",
    )
    (source / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"formalize_language": True, "other": "source"}}),
        encoding="utf-8",
    )
    (target / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"formalize_language": False, "other": "target"}}),
        encoding="utf-8",
    )
    return source, target


def test_materialise_field_targeted_swap_copies_only_listed_paths(tmp_path: Path) -> None:
    source, target = _make_field_swap_pair(tmp_path)

    result = materialise_field_targeted_swap(
        source,
        target,
        ["formalize_language"],
        swap_root=tmp_path / "swap",
    )

    swapped_config = json.loads((result.swap_dir / "config.json").read_text(encoding="utf-8"))
    assert swapped_config["formalize_language"] is True
    assert swapped_config["architectures"] == ["TargetModel"]
    assert swapped_config["nested"] == {"keep": "target", "copy_me": 1}


def test_materialise_field_targeted_swap_returns_FieldSwapResult_with_FieldChange_per_path(
    tmp_path: Path,
) -> None:
    source, target = _make_field_swap_pair(tmp_path)

    result = materialise_field_targeted_swap(
        source,
        target,
        ["formalize_language", "processor_config.json:processor_kwargs.formalize_language"],
        swap_root=tmp_path / "swap",
    )

    assert isinstance(result, FieldSwapResult)
    assert [change.path for change in result.changes] == [
        "config.json:formalize_language",
        "processor_config.json:processor_kwargs.formalize_language",
    ]


def test_materialise_field_targeted_swap_byte_identity_outside_listed_paths(
    tmp_path: Path,
) -> None:
    source, target = _make_field_swap_pair(tmp_path)

    result = materialise_field_targeted_swap(
        source,
        target,
        ["formalize_language"],
        swap_root=tmp_path / "swap",
    )

    for name in ("statistics.json", "embodiment_id.json", "model-00001-of-00003.safetensors"):
        assert (result.swap_dir / name).read_bytes() == (target / name).read_bytes()


def test_materialise_field_targeted_swap_rejects_missing_source_path(
    tmp_path: Path,
) -> None:
    source, target = _make_field_swap_pair(tmp_path)

    with pytest.raises(KeyError):
        materialise_field_targeted_swap(
            source,
            target,
            ["missing_path"],
            swap_root=tmp_path / "swap",
        )


def test_materialise_field_targeted_swap_rejects_unknown_target_path(
    tmp_path: Path,
) -> None:
    source, target = _make_field_swap_pair(tmp_path)
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    source_config["source_only"] = True
    (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")

    with pytest.raises(KeyError):
        materialise_field_targeted_swap(
            source,
            target,
            ["source_only"],
            swap_root=tmp_path / "swap",
        )


def test_materialise_field_targeted_swap_idempotent_on_equal_values(tmp_path: Path) -> None:
    source, target = _make_field_swap_pair(tmp_path)
    target_config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    target_config["formalize_language"] = True
    (target / "config.json").write_text(json.dumps(target_config), encoding="utf-8")

    result = materialise_field_targeted_swap(
        source,
        target,
        ["formalize_language"],
        swap_root=tmp_path / "swap",
    )

    assert result.changes[0].before is True
    assert result.changes[0].after is True


def test_materialise_field_targeted_swap_writes_FieldChange_with_before_after_values(
    tmp_path: Path,
) -> None:
    source, target = _make_field_swap_pair(tmp_path)

    result = materialise_field_targeted_swap(
        source,
        target,
        ["nested.copy_me"],
        swap_root=tmp_path / "swap",
    )

    assert result.changes[0].before == 1
    assert result.changes[0].after == 7
    swapped_config = json.loads((result.swap_dir / "config.json").read_text(encoding="utf-8"))
    assert swapped_config["nested"]["copy_me"] == 7


def test_materialise_field_targeted_swap_accepts_filename_keyed_field_overrides(
    tmp_path: Path,
) -> None:
    source, target = _make_field_swap_pair(tmp_path)

    result = materialise_field_targeted_swap(
        source,
        target,
        {
            "config.json": {"formalize_language": True},
            "processor_config.json": {
                "processor_kwargs.formalize_language": True,
            },
        },
        swap_root=tmp_path / "swap",
    )

    swapped_config = json.loads((result.swap_dir / "config.json").read_text(encoding="utf-8"))
    swapped_proc = json.loads(
        (result.swap_dir / "processor_config.json").read_text(encoding="utf-8")
    )
    assert swapped_config["formalize_language"] is True
    assert swapped_proc["processor_kwargs"]["formalize_language"] is True
    assert result.field_overrides == {
        "config.json": {"formalize_language": True},
        "processor_config.json": {"processor_kwargs.formalize_language": True},
    }


def test_materialise_field_targeted_swap_swap_root_isolation(tmp_path: Path) -> None:
    source, target = _make_field_swap_pair(tmp_path)
    source_before = _walk_recursive_sha_table(source)
    target_before = _walk_recursive_sha_table(target)
    swap_root = tmp_path / "swap"

    result = materialise_field_targeted_swap(
        source,
        target,
        ["formalize_language"],
        swap_root=swap_root,
    )

    assert result.swap_dir.resolve().is_relative_to(swap_root.resolve())
    assert _walk_recursive_sha_table(source) == source_before
    assert _walk_recursive_sha_table(target) == target_before


# ---------------------------------------------------------------------------
# _walk_recursive_sha_table symlink behaviour (V3-FIX-4)
# ---------------------------------------------------------------------------


def test_walk_recursive_sha_table_follows_symlinks(tmp_path: Path) -> None:
    """_walk_recursive_sha_table follows symlinks and hashes target content bytes."""
    real_file = tmp_path / "real.bin"
    real_file.write_bytes(b"symlink-target-content")
    link = tmp_path / "link.bin"
    link.symlink_to(real_file)

    table = _walk_recursive_sha_table(tmp_path)
    expected = hashlib.sha256(b"symlink-target-content").hexdigest()
    assert "real.bin" in table
    assert "link.bin" in table
    assert table["link.bin"] == expected, "symlink hash must equal target content hash"
    assert table["real.bin"] == expected


def test_walk_recursive_sha_table_broken_symlink_raises(tmp_path: Path) -> None:
    """Broken symlinks must raise FileNotFoundError per V3-FIX-4 spec."""
    broken = tmp_path / "broken.bin"
    broken.symlink_to(tmp_path / "does_not_exist.bin")
    with pytest.raises((FileNotFoundError, OSError)):
        _walk_recursive_sha_table(tmp_path)


# ---------------------------------------------------------------------------
# ConfigSwapResult is frozen
# ---------------------------------------------------------------------------


def test_config_swap_result_is_frozen(result: ConfigSwapResult) -> None:
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        result.link_strategy = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Source-scan guards
# ---------------------------------------------------------------------------


def test_no_t8_imports_in_ckpt_config_swap() -> None:
    src = (
        Path(__file__).parent.parent.parent.parent
        / "work" / "recap" / "r2_authentic_eval" / "ckpt_config_swap.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "safe_sft" not in node.module
            assert not node.module.startswith("t8_")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "safe_sft" not in alias.name
                assert not alias.name.startswith("t8_")


def test_no_os_link_calls_in_ckpt_config_swap() -> None:
    """ckpt_config_swap.py must not call os.link (no hardlinks — A3)."""
    src = (
        Path(__file__).parent.parent.parent.parent
        / "work" / "recap" / "r2_authentic_eval" / "ckpt_config_swap.py"
    )
    content = src.read_text(encoding="utf-8")
    assert "os.link" not in content, "os.link found — no hardlinks allowed (A3)"
