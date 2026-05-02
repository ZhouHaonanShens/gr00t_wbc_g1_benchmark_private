from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.dataset_reader import read_m1_dataset  # noqa: E402
from work.recap.formal_branch_resolution import import_external_manual_correction_bundle  # noqa: E402
from work.recap.manual_correction_bundle import (  # noqa: E402
    build_manual_correction_bundle,
    scaffold_manual_correction_spec,
    validate_manual_correction_spec,
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
        "prompt_raw": "pick up the apple, walk left and place the apple on the plate.",
        "prompt_conditioned": "pick up the apple, walk left and place the apple on the plate.",
        "npz_path": f"arrays/{episode_id}.npz",
    }


def _transition_row(
    *,
    iter_tag: str,
    episode_id: str,
    t: int,
    success_step: bool,
) -> dict[str, Any]:
    return {
        "iter_tag": iter_tag,
        "episode_id": episode_id,
        "t": int(t),
        "n_action_steps_executed": 1,
        "inner_rewards": [1.0 if success_step else 0.0],
        "inner_dones": [bool(success_step)],
        "success_step": bool(success_step),
        "prompt_raw": "pick up the apple, walk left and place the apple on the plate.",
        "prompt_conditioned": "pick up the apple, walk left and place the apple on the plate.",
        "npz_path": f"arrays/{episode_id}.npz",
    }


def _build_dataset(
    dataset_dir: Path,
    *,
    iter_tag: str,
    success_flags: list[bool],
) -> list[str]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = dataset_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    for index, success in enumerate(success_flags, start=1):
        episode_id = f"{iter_tag}_ep{index:03d}"
        episode_ids.append(episode_id)
        episodes.append(
            _episode_row(
                iter_tag=iter_tag,
                episode_id=episode_id,
                success_episode=success,
            )
        )
        transitions.extend(
            [
                _transition_row(
                    iter_tag=iter_tag,
                    episode_id=episode_id,
                    t=0,
                    success_step=success,
                ),
                _transition_row(
                    iter_tag=iter_tag,
                    episode_id=episode_id,
                    t=1,
                    success_step=success,
                ),
            ]
        )
        np.savez_compressed(
            arrays_dir / f"{episode_id}.npz",
            state=np.asarray([float(index)], dtype=np.float32),
            action=np.asarray([float(index)], dtype=np.float32),
        )
    _write_jsonl(dataset_dir / "episodes.jsonl", episodes)
    _write_jsonl(dataset_dir / "transitions.jsonl", transitions)
    return episode_ids


def test_scaffold_manual_correction_spec_lists_failed_nominal_episodes(
    tmp_path: Path,
) -> None:
    repo_root = _build_repo_root(tmp_path)
    nominal_dataset_dir = (
        repo_root / "agent" / "artifacts" / "recap_datasets" / "recap_stage3_iter_001"
    )
    _build_dataset(
        nominal_dataset_dir,
        iter_tag="recap_stage3_iter_001",
        success_flags=[False, True, False],
    )
    spec_path = (
        repo_root / "agent" / "artifacts" / "recap_corrections" / "spec.template.json"
    )

    payload = scaffold_manual_correction_spec(repo_root, spec_path=spec_path)

    saved = json.loads(spec_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "recap_stage3_manual_correction_spec_v1"
    assert saved["bundle_iter_tag"] == "recap_stage3_iter_001_manual_bundle"
    assert len(saved["corrections"]) == 2
    assert (
        saved["corrections"][0]["nominal_episode_id"] == "recap_stage3_iter_001_ep001"
    )
    assert saved["corrections"][0]["corrected_source_dataset_dir"] == ""
    assert saved["corrections"][0]["nominal_t_end"] == 1


def test_validate_manual_correction_spec_rejects_nominal_failure_relabel(
    tmp_path: Path,
) -> None:
    repo_root = _build_repo_root(tmp_path)
    nominal_dataset_dir = (
        repo_root / "agent" / "artifacts" / "recap_datasets" / "recap_stage3_iter_001"
    )
    nominal_episode_ids = _build_dataset(
        nominal_dataset_dir,
        iter_tag="recap_stage3_iter_001",
        success_flags=[False],
    )
    spec_path = repo_root / "spec.json"
    _write_json(
        spec_path,
        {
            "schema_version": "recap_stage3_manual_correction_spec_v1",
            "formal_iter_tag": "recap_stage3_iter_001",
            "nominal_dataset_dir": str(nominal_dataset_dir),
            "bundle_iter_tag": "recap_stage3_iter_001_manual_bundle",
            "bundle_output_dir": str(
                repo_root
                / "agent"
                / "artifacts"
                / "recap_corrections"
                / "recap_stage3_iter_001_manual"
            ),
            "corrections": [
                {
                    "correction_id": "corr_001",
                    "nominal_episode_id": nominal_episode_ids[0],
                    "nominal_t_start": 0,
                    "nominal_t_end": 1,
                    "human_note": "this must fail closed",
                    "corrected_source_dataset_dir": str(nominal_dataset_dir),
                    "corrected_episode_id": nominal_episode_ids[0],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="must not equal the nominal dataset dir"):
        validate_manual_correction_spec(repo_root, spec_path=spec_path)


def test_validate_manual_correction_spec_rejects_inline_synthetic_fields(
    tmp_path: Path,
) -> None:
    repo_root = _build_repo_root(tmp_path)
    nominal_dataset_dir = (
        repo_root / "agent" / "artifacts" / "recap_datasets" / "recap_stage3_iter_001"
    )
    nominal_episode_ids = _build_dataset(
        nominal_dataset_dir,
        iter_tag="recap_stage3_iter_001",
        success_flags=[False],
    )
    corrected_source_dir = repo_root / "corrected_source"
    corrected_ids = _build_dataset(
        corrected_source_dir,
        iter_tag="external_manual_source",
        success_flags=[True],
    )
    spec_path = repo_root / "spec_with_inline_actions.json"
    _write_json(
        spec_path,
        {
            "schema_version": "recap_stage3_manual_correction_spec_v1",
            "formal_iter_tag": "recap_stage3_iter_001",
            "nominal_dataset_dir": str(nominal_dataset_dir),
            "bundle_iter_tag": "recap_stage3_iter_001_manual_bundle",
            "bundle_output_dir": str(repo_root / "bundle"),
            "corrections": [
                {
                    "correction_id": "corr_001",
                    "nominal_episode_id": nominal_episode_ids[0],
                    "nominal_t_start": 0,
                    "nominal_t_end": 1,
                    "human_note": "inline actions must fail closed",
                    "corrected_source_dataset_dir": str(corrected_source_dir),
                    "corrected_episode_id": corrected_ids[0],
                    "corrected_actions": [[0.0]],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="forbidden synthetic fields"):
        validate_manual_correction_spec(repo_root, spec_path=spec_path)


def test_build_manual_correction_bundle_materializes_importable_bundle(
    tmp_path: Path,
) -> None:
    repo_root = _build_repo_root(tmp_path)
    nominal_dataset_dir = (
        repo_root / "agent" / "artifacts" / "recap_datasets" / "recap_stage3_iter_001"
    )
    nominal_episode_ids = _build_dataset(
        nominal_dataset_dir,
        iter_tag="recap_stage3_iter_001",
        success_flags=[False],
    )
    corrected_source_dir = repo_root / "external_corrected_source"
    corrected_ids = _build_dataset(
        corrected_source_dir,
        iter_tag="external_manual_source",
        success_flags=[True],
    )
    spec_path = repo_root / "spec.json"
    bundle_dir = (
        repo_root
        / "agent"
        / "artifacts"
        / "recap_corrections"
        / "recap_stage3_iter_001_manual"
    )
    _write_json(
        spec_path,
        {
            "schema_version": "recap_stage3_manual_correction_spec_v1",
            "formal_iter_tag": "recap_stage3_iter_001",
            "nominal_dataset_dir": str(nominal_dataset_dir),
            "bundle_iter_tag": "recap_stage3_iter_001_manual_bundle",
            "bundle_output_dir": str(bundle_dir),
            "corrections": [
                {
                    "correction_id": "corr_001",
                    "nominal_episode_id": nominal_episode_ids[0],
                    "nominal_t_start": 0,
                    "nominal_t_end": 1,
                    "human_note": "real corrected source episode",
                    "corrected_source_dataset_dir": str(corrected_source_dir),
                    "corrected_episode_id": corrected_ids[0],
                }
            ],
        },
    )

    build_result = build_manual_correction_bundle(
        repo_root,
        spec_path=spec_path,
        bundle_dir=bundle_dir,
    )

    bundle_dataset = read_m1_dataset(bundle_dir, check_npz_keys=True)
    assert build_result["episode_count"] == 1
    assert build_result["transition_count"] == 2
    assert (bundle_dir / "episodes.jsonl").is_file()
    assert (bundle_dir / "transitions.jsonl").is_file()
    assert (bundle_dir / "correction_segments.jsonl").is_file()
    assert (bundle_dir / "arrays").is_dir()
    built_transitions = bundle_dataset["transitions_by_episode"]
    assert isinstance(built_transitions, dict)
    only_episode_id = next(iter(built_transitions.keys()))
    assert all(
        bool(row.get("is_correction")) for row in built_transitions[only_episode_id]
    )

    resolution = import_external_manual_correction_bundle(
        repo_root, bundle_dir=bundle_dir
    )
    train_dir = (
        repo_root
        / "agent"
        / "artifacts"
        / "recap_datasets"
        / "recap_stage3_iter_001_train"
    )
    train_dataset = read_m1_dataset(train_dir, check_npz_keys=True)
    train_transitions = train_dataset["transitions_by_episode"]
    assert resolution["decision"] == "external_manual_correction"
    assert isinstance(train_transitions, dict)
    train_episode_id = next(iter(train_transitions.keys()))
    assert train_episode_id.startswith("recap_stage3_iter_001_manual_bundle")
    assert all(
        str(row.get("iter_tag")) == "recap_stage3_iter_001_train"
        for row in train_transitions[train_episode_id]
    )
    assert all(
        bool(row.get("is_correction")) for row in train_transitions[train_episode_id]
    )
