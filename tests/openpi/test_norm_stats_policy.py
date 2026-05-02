from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Callable, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "work/openpi/norm/policy.py"
SPEC = importlib.util.spec_from_file_location("openpi_norm_policy", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load norm policy module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["openpi_norm_policy"] = MODULE
SPEC.loader.exec_module(MODULE)

BuildNormPolicy: TypeAlias = Callable[[str | Path], object]
ValidateNormPolicy: TypeAlias = Callable[[object], object]
BuildNormProvenance: TypeAlias = Callable[[object], dict[str, str]]
NormPolicyCtor: TypeAlias = Callable[..., object]

NormPolicySpec = cast(NormPolicyCtor, getattr(MODULE, "NormPolicySpec"))

build_phase1_norm_policy = cast(
    BuildNormPolicy, getattr(MODULE, "build_phase1_norm_policy")
)
validate_phase1_norm_policy = cast(
    ValidateNormPolicy, getattr(MODULE, "validate_phase1_norm_policy")
)
build_phase1_norm_provenance = cast(
    BuildNormProvenance, getattr(MODULE, "build_phase1_norm_provenance")
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _make_dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    _write_json(
        dataset_dir / "meta/stats.json",
        {
            "observation.state": {"mean": [0.0], "std": [1.0]},
            "action": {"mean": [0.0], "std": [1.0]},
        },
    )
    return dataset_dir


def test_phase1_norm_policy_happy_path(tmp_path: Path) -> None:
    dataset_dir = _make_dataset_dir(tmp_path)
    spec = build_phase1_norm_policy(dataset_dir)
    validated = validate_phase1_norm_policy(spec)
    provenance = build_phase1_norm_provenance(validated)

    validated_obj = validated
    assert getattr(validated_obj, "policy_name") == "recompute_task_local_stats_primary"
    assert getattr(validated_obj, "norm_stats_source") == "dataset_meta_stats"
    assert getattr(validated_obj, "asset_id") == "task_local_recomputed"
    assert getattr(validated_obj, "reference_checkpoint_asset_id") == "droid"
    assert (
        getattr(validated_obj, "norm_stats_path") == dataset_dir / "meta" / "stats.json"
    )
    assert provenance["norm_stats_source"] == "dataset_meta_stats"
    assert provenance["norm_stats_path"].endswith("meta/stats.json")
    assert provenance["asset_id"] == "task_local_recomputed"


def test_phase1_norm_policy_requires_stats_file(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    try:
        _ = build_phase1_norm_policy(dataset_dir)
    except FileNotFoundError as exc:
        assert str(dataset_dir / "meta" / "stats.json") in str(exc)
    else:
        raise AssertionError("expected missing stats.json to fail")


def test_phase1_norm_policy_rejects_mismatched_runtime_spec(tmp_path: Path) -> None:
    dataset_dir = _make_dataset_dir(tmp_path)
    spec = build_phase1_norm_policy(dataset_dir)
    spec_obj = spec
    bad_spec = NormPolicySpec(
        policy_name=getattr(spec_obj, "policy_name"),
        norm_stats_source="checkpoint_assets",
        norm_stats_path=getattr(spec_obj, "norm_stats_path"),
        asset_id=getattr(spec_obj, "asset_id"),
        reference_checkpoint_asset_id=getattr(
            spec_obj, "reference_checkpoint_asset_id"
        ),
    )
    try:
        _ = validate_phase1_norm_policy(bad_spec)
    except ValueError as exc:
        assert "dataset_meta_stats" in str(exc)
    else:
        raise AssertionError("expected mismatched norm stats source to fail")
