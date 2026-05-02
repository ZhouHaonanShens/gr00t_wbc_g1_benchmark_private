from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.formal_branch_resolution import (  # noqa: E402
    BRANCH_RESOLUTION_JSON,
    EPISODES_JSONL,
    EXTERNAL_MANUAL_CORRECTION_BLOCKER,
    FormalBranchResolutionBlocked,
    SOURCE_DATASET_REF_JSON,
    TRANSITIONS_JSONL,
    import_external_manual_correction_bundle,
    maybe_reset_formal_nominal_dataset_dir,
    resolve_formal_collect_branch,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _build_repo_root(tmp_path: Path) -> Path:
    manifest_path = (
        tmp_path
        / "agent"
        / "artifacts"
        / "stage3_iteration"
        / "recap_stage3_iter_001"
        / "iteration_manifest.json"
    )
    _write_json(
        manifest_path,
        {
            "formal_iter_tag": "recap_stage3_iter_001",
            "train_iter_tag": "recap_stage3_iter_001_train",
            "env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
            "external_manual_correction_bundle_dir": "agent/artifacts/recap_corrections/recap_stage3_iter_001_manual",
        },
    )
    return tmp_path


def _episode_row(
    *, iter_tag: str, episode_id: str, success_episode: bool
) -> dict[str, Any]:
    return {
        "iter_tag": iter_tag,
        "episode_id": episode_id,
        "success_episode": bool(success_episode),
        "prompt_raw": "put apple on plate",
        "prompt_conditioned": "put apple on plate",
        "npz_path": f"arrays/{episode_id}.npz",
    }


def _transition_row(
    *, iter_tag: str, episode_id: str, success_step: bool
) -> dict[str, Any]:
    return {
        "iter_tag": iter_tag,
        "episode_id": episode_id,
        "t": 0,
        "n_action_steps_executed": 1,
        "inner_rewards": [1.0 if success_step else 0.0],
        "inner_dones": [bool(success_step)],
        "success_step": bool(success_step),
        "prompt_raw": "put apple on plate",
        "prompt_conditioned": "put apple on plate",
        "npz_path": f"arrays/{episode_id}.npz",
    }


def _build_dataset(
    dataset_dir: Path,
    *,
    iter_tag: str,
    success_flags: list[bool],
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = dataset_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for index, success in enumerate(success_flags, start=1):
        episode_id = f"{iter_tag}_ep{index:03d}"
        episodes.append(
            _episode_row(
                iter_tag=iter_tag, episode_id=episode_id, success_episode=success
            )
        )
        transitions.append(
            _transition_row(
                iter_tag=iter_tag, episode_id=episode_id, success_step=success
            )
        )
        np.savez_compressed(
            arrays_dir / f"{episode_id}.npz",
            **{
                "state/robot_state": np.asarray([float(index)], dtype=np.float32),
                "action/joint_action": np.asarray([float(index)], dtype=np.float32),
            },
        )

    _write_jsonl(dataset_dir / EPISODES_JSONL, episodes)
    _write_jsonl(dataset_dir / TRANSITIONS_JSONL, transitions)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_nominal_direct_branch_materializes_train_dataset(tmp_path: Path) -> None:
    repo_root = _build_repo_root(tmp_path)
    nominal_dataset_dir = (
        repo_root / "agent" / "artifacts" / "recap_datasets" / "recap_stage3_iter_001"
    )
    _build_dataset(
        nominal_dataset_dir,
        iter_tag="recap_stage3_iter_001",
        success_flags=[True, False, False],
    )

    resolution = resolve_formal_collect_branch(
        repo_root,
        iter_tag="recap_stage3_iter_001",
    )

    train_dir = (
        repo_root
        / "agent"
        / "artifacts"
        / "recap_datasets"
        / "recap_stage3_iter_001_train"
    )
    branch_resolution = _read_json(train_dir / BRANCH_RESOLUTION_JSON)
    train_episodes = [
        json.loads(line)
        for line in (train_dir / EPISODES_JSONL)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert resolution["decision"] == "nominal_direct"
    assert branch_resolution["decision"] == "nominal_direct"
    assert branch_resolution["nominal_batch"]["total_episodes"] == 3
    assert branch_resolution["nominal_batch"]["success_count"] == 1
    assert branch_resolution["nominal_batch"]["failure_count"] == 2
    assert branch_resolution["output"]["dataset_path"] == str(train_dir)
    assert branch_resolution["output"]["source_type"] == "nominal_dataset"
    assert all(
        row["iter_tag"] == "recap_stage3_iter_001_train" for row in train_episodes
    )
    assert (train_dir / "arrays").is_symlink()
    assert (train_dir / SOURCE_DATASET_REF_JSON).is_file()


def test_all_failure_imports_external_manual_bundle(tmp_path: Path) -> None:
    repo_root = _build_repo_root(tmp_path)
    nominal_dataset_dir = (
        repo_root / "agent" / "artifacts" / "recap_datasets" / "recap_stage3_iter_001"
    )
    manual_bundle_dir = (
        repo_root
        / "agent"
        / "artifacts"
        / "recap_corrections"
        / "recap_stage3_iter_001_manual"
        / "bundle_dataset"
    )
    _build_dataset(
        nominal_dataset_dir,
        iter_tag="recap_stage3_iter_001",
        success_flags=[False, False],
    )
    _build_dataset(
        manual_bundle_dir,
        iter_tag="manual_bundle_iter",
        success_flags=[True, True],
    )

    resolution = resolve_formal_collect_branch(
        repo_root,
        iter_tag="recap_stage3_iter_001",
    )

    train_dir = (
        repo_root
        / "agent"
        / "artifacts"
        / "recap_datasets"
        / "recap_stage3_iter_001_train"
    )
    branch_resolution = _read_json(train_dir / BRANCH_RESOLUTION_JSON)
    train_episodes = [
        json.loads(line)
        for line in (train_dir / EPISODES_JSONL)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert resolution["decision"] == "external_manual_correction"
    assert branch_resolution["decision"] == "external_manual_correction"
    assert branch_resolution["nominal_batch"]["success_count"] == 0
    assert (
        branch_resolution["output"]["source_type"]
        == "external_manual_correction_bundle"
    )
    assert all(
        row["iter_tag"] == "recap_stage3_iter_001_train" for row in train_episodes
    )


def test_all_failure_without_bundle_fails_closed_and_cleans_stale_payload(
    tmp_path: Path,
) -> None:
    repo_root = _build_repo_root(tmp_path)
    nominal_dataset_dir = (
        repo_root / "agent" / "artifacts" / "recap_datasets" / "recap_stage3_iter_001"
    )
    train_dir = (
        repo_root
        / "agent"
        / "artifacts"
        / "recap_datasets"
        / "recap_stage3_iter_001_train"
    )
    _build_dataset(
        nominal_dataset_dir,
        iter_tag="recap_stage3_iter_001",
        success_flags=[False, False],
    )
    _build_dataset(
        train_dir,
        iter_tag="stale_old_train",
        success_flags=[True],
    )

    with pytest.raises(FormalBranchResolutionBlocked) as exc_info:
        resolve_formal_collect_branch(repo_root, iter_tag="recap_stage3_iter_001")

    payload = exc_info.value.to_machine_payload()
    branch_resolution = _read_json(train_dir / BRANCH_RESOLUTION_JSON)

    assert payload["failure"]["blockers"] == [EXTERNAL_MANUAL_CORRECTION_BLOCKER]
    assert branch_resolution["decision"] == "blocked"
    assert branch_resolution["blocker"] == EXTERNAL_MANUAL_CORRECTION_BLOCKER
    assert not (train_dir / EPISODES_JSONL).exists()
    assert not (train_dir / TRANSITIONS_JSONL).exists()
    assert not (train_dir / "arrays").exists()
    assert not (train_dir / SOURCE_DATASET_REF_JSON).exists()


def test_import_cli_helper_materializes_train_dataset_from_bundle(
    tmp_path: Path,
) -> None:
    repo_root = _build_repo_root(tmp_path)
    bundle_dir = (
        repo_root
        / "agent"
        / "artifacts"
        / "recap_corrections"
        / "recap_stage3_iter_001_manual"
    )
    _build_dataset(
        bundle_dir,
        iter_tag="external_manual_bundle",
        success_flags=[True],
    )

    resolution = import_external_manual_correction_bundle(repo_root)

    train_dir = (
        repo_root
        / "agent"
        / "artifacts"
        / "recap_datasets"
        / "recap_stage3_iter_001_train"
    )
    branch_resolution = _read_json(train_dir / BRANCH_RESOLUTION_JSON)
    source_ref = _read_json(train_dir / SOURCE_DATASET_REF_JSON)

    assert resolution["decision"] == "external_manual_correction"
    assert branch_resolution["decision"] == "external_manual_correction"
    assert source_ref["source_type"] == "external_manual_correction_bundle"


def test_maybe_reset_formal_nominal_dataset_dir_only_resets_formal_tag(
    tmp_path: Path,
) -> None:
    repo_root = _build_repo_root(tmp_path)
    formal_dir = (
        repo_root / "agent" / "artifacts" / "recap_datasets" / "recap_stage3_iter_001"
    )
    other_dir = repo_root / "agent" / "artifacts" / "recap_datasets" / "other_iter"
    formal_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    (formal_dir / "stale.txt").write_text("stale\n", encoding="utf-8")
    (other_dir / "keep.txt").write_text("keep\n", encoding="utf-8")

    reset_dir = maybe_reset_formal_nominal_dataset_dir(
        repo_root,
        iter_tag="recap_stage3_iter_001",
    )
    skipped_dir = maybe_reset_formal_nominal_dataset_dir(
        repo_root, iter_tag="other_iter"
    )

    assert reset_dir == formal_dir
    assert skipped_dir is None
    assert formal_dir.is_dir()
    assert not (formal_dir / "stale.txt").exists()
    assert (other_dir / "keep.txt").is_file()
