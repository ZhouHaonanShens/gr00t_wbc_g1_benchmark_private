from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from work.recap.dataset_reader import read_m1_dataset


BRANCH_RESOLUTION_SCHEMA_VERSION = "recap_stage3_branch_resolution_v1"
SOURCE_DATASET_REF_SCHEMA_VERSION = "recap_stage3_source_dataset_ref_v1"
BLOCKED_EXIT_CODE = 2
EXTERNAL_MANUAL_CORRECTION_BLOCKER = "external_manual_correction_bundle_required"
ITERATION_MANIFEST_REL = Path(
    "agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json"
)
RECAP_DATASET_DIR_REL = Path("agent/artifacts/recap_datasets")
EPISODES_JSONL = "episodes.jsonl"
TRANSITIONS_JSONL = "transitions.jsonl"
ARRAYS_DIRNAME = "arrays"
BRANCH_RESOLUTION_JSON = "branch_resolution.json"
SOURCE_DATASET_REF_JSON = "source_dataset_ref.json"


@dataclass(frozen=True)
class Stage3IterationContract:
    repo_root: Path
    manifest_path: Path
    iteration_id: str
    formal_iter_tag: str
    train_iter_tag: str
    env_name: str
    external_manual_correction_bundle_dir: Path

    @property
    def nominal_dataset_dir(self) -> Path:
        return self.repo_root / RECAP_DATASET_DIR_REL / self.formal_iter_tag

    @property
    def train_dataset_dir(self) -> Path:
        return self.repo_root / RECAP_DATASET_DIR_REL / self.train_iter_tag


class FormalBranchResolutionBlocked(RuntimeError):
    def __init__(self, resolution: Mapping[str, Any]):
        self.resolution = dict(resolution)
        message = str(
            self.resolution.get("reason") or "formal_branch_resolution_blocked"
        )
        super().__init__(message)

    def to_machine_payload(self) -> dict[str, Any]:
        resolution = dict(self.resolution)
        blocker = str(resolution.get("blocker") or EXTERNAL_MANUAL_CORRECTION_BLOCKER)
        return {
            "status": "FAIL",
            "failure": {
                "stage": "formal_branch_resolution",
                "blockers": [blocker],
                "message": str(self),
                "detail": resolution,
            },
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object in {path}, got {type(payload).__name__}"
        )
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"expected JSON object in {path}:{line_number}, got {type(payload).__name__}"
                )
            rows.append(dict(payload))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    tmp_path.replace(path)
    return path


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _require_non_empty_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"iteration manifest missing non-empty {field_name}")
    return value.strip()


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def load_stage3_iteration_contract(
    repo_root: Path, *, manifest_path: Path | None = None
) -> Stage3IterationContract:
    resolved_repo_root = Path(repo_root).resolve()
    resolved_manifest_path = (
        _resolve_path(resolved_repo_root, manifest_path)
        if manifest_path is not None
        else (resolved_repo_root / ITERATION_MANIFEST_REL).resolve()
    )
    payload = _read_json(resolved_manifest_path)
    formal_iter_tag = _require_non_empty_string(payload, "formal_iter_tag")
    train_iter_tag = _require_non_empty_string(payload, "train_iter_tag")
    env_name = _require_non_empty_string(payload, "env_name")
    bundle_dir = _resolve_path(
        resolved_repo_root,
        _require_non_empty_string(payload, "external_manual_correction_bundle_dir"),
    )
    return Stage3IterationContract(
        repo_root=resolved_repo_root,
        manifest_path=resolved_manifest_path,
        iteration_id=formal_iter_tag,
        formal_iter_tag=formal_iter_tag,
        train_iter_tag=train_iter_tag,
        env_name=env_name,
        external_manual_correction_bundle_dir=bundle_dir,
    )


def maybe_reset_formal_nominal_dataset_dir(
    repo_root: Path, *, iter_tag: str
) -> Path | None:
    contract = load_stage3_iteration_contract(Path(repo_root))
    if str(iter_tag) != contract.formal_iter_tag:
        return None
    nominal_dataset_dir = contract.nominal_dataset_dir
    if nominal_dataset_dir.exists():
        shutil.rmtree(nominal_dataset_dir)
    nominal_dataset_dir.mkdir(parents=True, exist_ok=True)
    return nominal_dataset_dir


def _dataset_has_required_surface(dataset_dir: Path) -> bool:
    return (
        (dataset_dir / EPISODES_JSONL).is_file()
        and (dataset_dir / TRANSITIONS_JSONL).is_file()
        and (dataset_dir / ARRAYS_DIRNAME).is_dir()
    )


def _locate_external_bundle_dataset_dir(
    bundle_dir: Path,
) -> tuple[Path | None, str | None]:
    resolved_bundle_dir = Path(bundle_dir).expanduser().resolve()
    if not resolved_bundle_dir.exists():
        return None, f"bundle directory does not exist: {resolved_bundle_dir}"
    if not resolved_bundle_dir.is_dir():
        return None, f"bundle path is not a directory: {resolved_bundle_dir}"
    if _dataset_has_required_surface(resolved_bundle_dir):
        return resolved_bundle_dir, None

    candidate_dirs = [
        child
        for child in sorted(resolved_bundle_dir.iterdir())
        if child.is_dir() and _dataset_has_required_surface(child)
    ]
    if not candidate_dirs:
        return (
            None,
            "bundle directory does not contain a complete M1 dataset surface "
            f"(missing {EPISODES_JSONL}/{TRANSITIONS_JSONL}/{ARRAYS_DIRNAME}): {resolved_bundle_dir}",
        )
    if len(candidate_dirs) > 1:
        return (
            None,
            "bundle directory is ambiguous; expected exactly one nested M1 dataset directory, got "
            f"{len(candidate_dirs)} under {resolved_bundle_dir}",
        )
    return candidate_dirs[0], None


def _rewrite_jsonl_iter_tag(src: Path, dst: Path, *, iter_tag: str) -> None:
    rows = _read_jsonl(src)
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["iter_tag"] = str(iter_tag)
        rewritten.append(item)
    _write_jsonl(dst, rewritten)


def _reset_target_dataset_dir(target_dataset_dir: Path) -> None:
    if target_dataset_dir.exists() or target_dataset_dir.is_symlink():
        if target_dataset_dir.is_symlink() or target_dataset_dir.is_file():
            target_dataset_dir.unlink()
        else:
            shutil.rmtree(target_dataset_dir)
    target_dataset_dir.mkdir(parents=True, exist_ok=True)


def _clear_target_dataset_payload(target_dataset_dir: Path) -> None:
    if not target_dataset_dir.exists():
        target_dataset_dir.mkdir(parents=True, exist_ok=True)
        return
    for name in (
        EPISODES_JSONL,
        TRANSITIONS_JSONL,
        ARRAYS_DIRNAME,
        SOURCE_DATASET_REF_JSON,
    ):
        path = target_dataset_dir / name
        if not (path.exists() or path.is_symlink()):
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


def _materialize_target_dataset(
    *,
    contract: Stage3IterationContract,
    source_dataset_dir: Path,
    target_dataset_dir: Path,
    target_iter_tag: str,
    source_type: str,
    source_bundle_dir: Path | None = None,
) -> dict[str, Any]:
    read_m1_dataset(str(source_dataset_dir), check_npz_keys=True)
    source_episodes = source_dataset_dir / EPISODES_JSONL
    source_transitions = source_dataset_dir / TRANSITIONS_JSONL
    source_arrays = source_dataset_dir / ARRAYS_DIRNAME
    if not source_arrays.is_dir():
        raise FileNotFoundError(f"missing source arrays dir: {source_arrays}")

    _reset_target_dataset_dir(target_dataset_dir)
    _rewrite_jsonl_iter_tag(
        source_episodes, target_dataset_dir / EPISODES_JSONL, iter_tag=target_iter_tag
    )
    _rewrite_jsonl_iter_tag(
        source_transitions,
        target_dataset_dir / TRANSITIONS_JSONL,
        iter_tag=target_iter_tag,
    )

    arrays_dst = target_dataset_dir / ARRAYS_DIRNAME
    os.symlink(source_arrays, arrays_dst, target_is_directory=True)

    source_dataset_ref = {
        "schema_version": SOURCE_DATASET_REF_SCHEMA_VERSION,
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "iteration_id": contract.iteration_id,
        "iter_tag": str(target_iter_tag),
        "source_type": str(source_type),
        "source_dataset_dir": str(source_dataset_dir),
        "source_iter_tag": str(source_dataset_dir.name),
        "source_bundle_dir": str(source_bundle_dir)
        if source_bundle_dir is not None
        else None,
        "output_dataset_dir": str(target_dataset_dir),
        "arrays_path": str(arrays_dst),
    }
    _write_json(target_dataset_dir / SOURCE_DATASET_REF_JSON, source_dataset_ref)
    materialized_dataset = read_m1_dataset(str(target_dataset_dir), check_npz_keys=True)
    return {
        "dataset_path": str(target_dataset_dir),
        "source_type": str(source_type),
        "materialized": True,
        "episode_count": _require_int(
            materialized_dataset["n_episodes"],
            field_name="materialized_dataset.n_episodes",
        ),
        "transition_count": _require_int(
            materialized_dataset["n_transitions"],
            field_name="materialized_dataset.n_transitions",
        ),
        "source_dataset_ref_path": str(target_dataset_dir / SOURCE_DATASET_REF_JSON),
    }


def _build_resolution_payload(
    *,
    contract: Stage3IterationContract,
    nominal_batch: Mapping[str, Any],
    decision: str,
    reason: str,
    blocker: str | None,
    output: Mapping[str, Any],
    source_bundle_dir: Path | None = None,
    source_dataset_dir: Path | None = None,
) -> dict[str, Any]:
    target_dataset_dir = contract.train_dataset_dir
    return {
        "schema_version": BRANCH_RESOLUTION_SCHEMA_VERSION,
        "iteration_id": contract.iteration_id,
        "target_dataset": contract.train_iter_tag,
        "decision": str(decision),
        "reason": str(reason),
        "blocker": blocker,
        "nominal_batch": {
            "dataset_path": str(contract.nominal_dataset_dir),
            "total_episodes": int(nominal_batch.get("total_episodes", 0)),
            "success_count": int(nominal_batch.get("success_count", 0)),
            "failure_count": int(nominal_batch.get("failure_count", 0)),
        },
        "external_manual_correction": {
            "bundle_dir": str(contract.external_manual_correction_bundle_dir),
            "source_dataset_dir": str(source_dataset_dir)
            if source_dataset_dir is not None
            else None,
            "resolved": source_dataset_dir is not None,
            "requested": source_bundle_dir is not None,
        },
        "output": {
            "dataset_path": str(output.get("dataset_path") or target_dataset_dir),
            "source_type": str(output.get("source_type") or "none"),
            "materialized": bool(output.get("materialized", False)),
            "branch_resolution_path": str(target_dataset_dir / BRANCH_RESOLUTION_JSON),
            "source_dataset_ref_path": output.get("source_dataset_ref_path"),
        },
    }


def write_branch_resolution(
    contract: Stage3IterationContract, resolution: Mapping[str, Any]
) -> Path:
    path = contract.train_dataset_dir / BRANCH_RESOLUTION_JSON
    _write_json(path, resolution)
    return path


def resolve_formal_collect_branch(repo_root: Path, *, iter_tag: str) -> dict[str, Any]:
    contract = load_stage3_iteration_contract(Path(repo_root))
    if str(iter_tag) != contract.formal_iter_tag:
        raise ValueError(
            f"formal branch resolution only applies to {contract.formal_iter_tag}, got {iter_tag}"
        )

    nominal_dataset = read_m1_dataset(
        str(contract.nominal_dataset_dir), check_npz_keys=True
    )
    episodes = nominal_dataset["episodes"]
    if not isinstance(episodes, list):
        raise ValueError("read_m1_dataset returned invalid episodes payload")
    success_count = sum(
        1 for row in episodes if bool(dict(row).get("success_episode", False))
    )
    total_episodes = _require_int(
        nominal_dataset["n_episodes"], field_name="nominal_dataset.n_episodes"
    )
    failure_count = int(total_episodes - success_count)
    nominal_batch = {
        "total_episodes": total_episodes,
        "success_count": int(success_count),
        "failure_count": failure_count,
    }

    if success_count > 0:
        output = _materialize_target_dataset(
            contract=contract,
            source_dataset_dir=contract.nominal_dataset_dir,
            target_dataset_dir=contract.train_dataset_dir,
            target_iter_tag=contract.train_iter_tag,
            source_type="nominal_dataset",
        )
        resolution = _build_resolution_payload(
            contract=contract,
            nominal_batch=nominal_batch,
            decision="nominal_direct",
            reason="nominal batch contains at least one success episode",
            blocker=None,
            output=output,
        )
        write_branch_resolution(contract, resolution)
        return resolution

    source_dataset_dir, bundle_error = _locate_external_bundle_dataset_dir(
        contract.external_manual_correction_bundle_dir
    )
    if source_dataset_dir is not None:
        output = _materialize_target_dataset(
            contract=contract,
            source_dataset_dir=source_dataset_dir,
            target_dataset_dir=contract.train_dataset_dir,
            target_iter_tag=contract.train_iter_tag,
            source_type="external_manual_correction_bundle",
            source_bundle_dir=contract.external_manual_correction_bundle_dir,
        )
        resolution = _build_resolution_payload(
            contract=contract,
            nominal_batch=nominal_batch,
            decision="external_manual_correction",
            reason="nominal batch recorded zero success episodes; imported external manual correction bundle",
            blocker=None,
            output=output,
            source_bundle_dir=contract.external_manual_correction_bundle_dir,
            source_dataset_dir=source_dataset_dir,
        )
        write_branch_resolution(contract, resolution)
        return resolution

    _clear_target_dataset_payload(contract.train_dataset_dir)
    resolution = _build_resolution_payload(
        contract=contract,
        nominal_batch=nominal_batch,
        decision="blocked",
        reason=(
            "nominal batch recorded zero success episodes and no valid external manual correction bundle was found"
            + (f": {bundle_error}" if bundle_error else "")
        ),
        blocker=EXTERNAL_MANUAL_CORRECTION_BLOCKER,
        output={
            "dataset_path": str(contract.train_dataset_dir),
            "source_type": "none",
            "materialized": False,
            "source_dataset_ref_path": None,
        },
        source_bundle_dir=contract.external_manual_correction_bundle_dir,
        source_dataset_dir=None,
    )
    write_branch_resolution(contract, resolution)
    raise FormalBranchResolutionBlocked(resolution)


def import_external_manual_correction_bundle(
    repo_root: Path,
    *,
    bundle_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    contract = load_stage3_iteration_contract(
        Path(repo_root), manifest_path=manifest_path
    )
    source_bundle_dir = (
        _resolve_path(Path(repo_root), bundle_dir)
        if bundle_dir is not None
        else contract.external_manual_correction_bundle_dir
    )
    source_dataset_dir, bundle_error = _locate_external_bundle_dataset_dir(
        source_bundle_dir
    )
    if source_dataset_dir is None:
        _clear_target_dataset_payload(contract.train_dataset_dir)
        resolution = _build_resolution_payload(
            contract=contract,
            nominal_batch={
                "total_episodes": 0,
                "success_count": 0,
                "failure_count": 0,
            },
            decision="blocked",
            reason=(
                "external manual correction bundle is required but missing or invalid"
                + (f": {bundle_error}" if bundle_error else "")
            ),
            blocker=EXTERNAL_MANUAL_CORRECTION_BLOCKER,
            output={
                "dataset_path": str(contract.train_dataset_dir),
                "source_type": "none",
                "materialized": False,
                "source_dataset_ref_path": None,
            },
            source_bundle_dir=source_bundle_dir,
            source_dataset_dir=None,
        )
        write_branch_resolution(contract, resolution)
        raise FormalBranchResolutionBlocked(resolution)

    output = _materialize_target_dataset(
        contract=contract,
        source_dataset_dir=source_dataset_dir,
        target_dataset_dir=contract.train_dataset_dir,
        target_iter_tag=contract.train_iter_tag,
        source_type="external_manual_correction_bundle",
        source_bundle_dir=source_bundle_dir,
    )
    resolution = _build_resolution_payload(
        contract=contract,
        nominal_batch={
            "total_episodes": 0,
            "success_count": 0,
            "failure_count": 0,
        },
        decision="external_manual_correction",
        reason="manual correction bundle imported into stage3 train dataset",
        blocker=None,
        output=output,
        source_bundle_dir=source_bundle_dir,
        source_dataset_dir=source_dataset_dir,
    )
    write_branch_resolution(contract, resolution)
    return resolution


def maybe_resolve_formal_collect_branch(
    repo_root: Path, *, iter_tag: str
) -> dict[str, Any] | None:
    contract = load_stage3_iteration_contract(Path(repo_root))
    if str(iter_tag) != contract.formal_iter_tag:
        return None
    return resolve_formal_collect_branch(Path(repo_root), iter_tag=iter_tag)
