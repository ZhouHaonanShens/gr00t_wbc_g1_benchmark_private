from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_BUCKET_DIR = Path("agent/artifacts/state_conditioned_materialization/bucket_a")
DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_materialization/devbench")
DEFAULT_ORIGINAL_BASELINE_MODEL = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_MAX_EPISODE_STEPS = 240
DEFAULT_PAIRED_SEEDS: tuple[int, ...] = (
    31001,
    31002,
    31003,
    31004,
    31005,
    31006,
    31007,
    31008,
)
DEFAULT_STRATA_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "stratum_id": "nominal",
        "failure_injection_kind": "none",
        "apple_visibility": "visible",
    },
    {
        "stratum_id": "drop_during_transport",
        "failure_injection_kind": "drop_during_transport",
        "apple_visibility": "visible",
    },
    {
        "stratum_id": "failed_grasp_visible",
        "failure_injection_kind": "failed_grasp_visible",
        "apple_visibility": "visible",
    },
    {
        "stratum_id": "failed_grasp_occluded",
        "failure_injection_kind": "failed_grasp_occluded",
        "apple_visibility": "occluded",
    },
)
EXPECTED_STRATA_COUNTS: dict[str, int] = {
    "nominal": 8,
    "drop_during_transport": 8,
    "failed_grasp_visible": 8,
    "failed_grasp_occluded": 8,
}

FIXED_STRATA_DEFINITION_JSON_NAME = "fixed_strata_definition.json"
BASELINE_MANIFEST_JSON_NAME = "baseline_manifest.json"
BASELINE_DEV_SCORECARD_JSON_NAME = "baseline_dev_scorecard.json"
SCHEMA_VERSION = "g1_state_conditioned_dev_manifest_v1"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_bucket_a_sidecar


BaselineRunner = Callable[..., dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the dev-only paired state-conditioned manifest after canonical "
            "Bucket A Gate A and T5 sidecar artifacts are ready, then run the "
            "current baseline eval wrapper and emit a machine-readable scorecard."
        )
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=DEFAULT_BUCKET_DIR,
        help="Canonical Bucket A directory containing Gate A and T5 sidecar artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives fixed_strata_definition.json, baseline_manifest.json, and baseline_dev_scorecard.json.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_dir(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"{arg_name} directory does not exist: {resolved}")
    return resolved


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return state_conditioned_bucket_a_import._read_json(path)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    return state_conditioned_bucket_a_import._as_non_empty_string(
        value,
        field_name=field_name,
    )


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _accepted_canonical_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    accepted_entries: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for raw_entry in manifest.get("episodes", []):
        entry = dict(_as_mapping(raw_entry, field_name="manifest.episodes[]"))
        if not bool(entry.get("accepted", False)):
            continue
        if not bool(entry.get("fresh_nominal_recollection", False)):
            continue
        if bool(entry.get("debug_only", False)):
            continue
        if bool(entry.get("reused_existing_live_dataset", True)):
            continue
        episode_id = _as_non_empty_string(
            entry.get("episode_id"), field_name="episode_id"
        )
        if episode_id in seen_episode_ids:
            raise ValueError(
                f"duplicate canonical episode_id in manifest: {episode_id}"
            )
        seen_episode_ids.add(episode_id)
        accepted_entries.append(entry)
    return accepted_entries


def _load_bucket_preconditions(
    bucket_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    gate_path = bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME
    manifest_path = bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME
    sidecar_path = (
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_SIDECAR_JSON_NAME
    )
    join_coverage_path = (
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_JOIN_COVERAGE_JSON_NAME
    )
    exporter_manifest_path = (
        bucket_dir
        / state_conditioned_bucket_a_sidecar.BUCKET_A_EXPORTER_MANIFEST_JSON_NAME
    )
    required_paths = {
        "gate_path": gate_path,
        "manifest_path": manifest_path,
        "sidecar_path": sidecar_path,
        "join_coverage_path": join_coverage_path,
        "exporter_manifest_path": exporter_manifest_path,
    }
    for name, path in required_paths.items():
        if not path.is_file():
            raise ValueError(f"missing required T5 artifact {name}: {path}")

    gate = _read_json(gate_path)
    manifest = _read_json(manifest_path)
    if not bool(gate.get("ready", False)):
        raise ValueError(
            "state-conditioned dev manifest refuses to run until "
            "bucket_A_gate_a_ready.json.ready == true"
        )
    return gate, manifest, required_paths


def _normalize_strata_definitions(
    strata_definition: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, str]]:
    source = (
        DEFAULT_STRATA_DEFINITIONS if strata_definition is None else strata_definition
    )
    normalized: list[dict[str, str]] = []
    seen_stratum_ids: set[str] = set()
    for index, raw in enumerate(source):
        row = _as_mapping(raw, field_name=f"strata_definition[{index}]")
        stratum_id = _as_non_empty_string(
            row.get("stratum_id"), field_name=f"strata_definition[{index}].stratum_id"
        )
        if stratum_id in seen_stratum_ids:
            raise ValueError(f"duplicate stratum_id in strata_definition: {stratum_id}")
        seen_stratum_ids.add(stratum_id)
        normalized.append(
            {
                "stratum_id": stratum_id,
                "failure_injection_kind": _as_non_empty_string(
                    row.get("failure_injection_kind"),
                    field_name=f"strata_definition[{index}].failure_injection_kind",
                ),
                "apple_visibility": _as_non_empty_string(
                    row.get("apple_visibility"),
                    field_name=f"strata_definition[{index}].apple_visibility",
                ),
            }
        )
    return normalized


def _normalize_paired_seeds(paired_seeds: Sequence[int] | None) -> list[int]:
    seeds = list(DEFAULT_PAIRED_SEEDS if paired_seeds is None else paired_seeds)
    normalized: list[int] = []
    for index, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(
                f"paired_seeds[{index}] must be an int, got {type(seed).__name__}"
            )
        normalized.append(int(seed))
    return normalized


def _train_lineage(
    entries: Sequence[Mapping[str, Any]], *, manifest_path: Path
) -> dict[str, Any]:
    accepted_episode_ids: list[str] = []
    accepted_seed_values: list[int] = []
    source_dataset_dirs: list[str] = []
    for index, entry in enumerate(entries):
        accepted_episode_ids.append(
            _as_non_empty_string(
                entry.get("episode_id"), field_name=f"train[{index}].episode_id"
            )
        )
        accepted_seed_values.append(
            _as_int(entry.get("seed"), field_name=f"train[{index}].seed")
        )
        source_dataset_dirs.append(
            _as_non_empty_string(
                entry.get("source_dataset_dir"),
                field_name=f"train[{index}].source_dataset_dir",
            )
        )
    return {
        "source_manifest_path": str(manifest_path),
        "accepted_episode_count": int(len(accepted_episode_ids)),
        "accepted_episode_ids": accepted_episode_ids,
        "accepted_seed_values": sorted(set(accepted_seed_values)),
        "accepted_source_dataset_dirs": sorted(set(source_dataset_dirs)),
        "overlap_check_identity": "seed",
    }


def _build_fixed_strata_definition(
    *,
    bucket_dir: Path,
    artifact_paths: Mapping[str, Path],
    paired_seeds: Sequence[int],
    strata: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_dev_fixed_strata_definition",
        "bucket_dir": str(bucket_dir),
        "canonical_gate_path": str(artifact_paths["gate_path"]),
        "canonical_manifest_path": str(artifact_paths["manifest_path"]),
        "bucket_A_sidecar_path": str(artifact_paths["sidecar_path"]),
        "bucket_A_join_coverage_path": str(artifact_paths["join_coverage_path"]),
        "bucket_A_exporter_manifest_path": str(
            artifact_paths["exporter_manifest_path"]
        ),
        "paired_identity": ["seed", "stratum_id"],
        "paired_seed_values": [int(seed) for seed in paired_seeds],
        "paired_seed_count": int(len(paired_seeds)),
        "expected_total_entries": int(len(paired_seeds) * len(strata)),
        "strata": [
            {
                **dict(row),
                "paired_episode_count": int(len(paired_seeds)),
            }
            for row in strata
        ],
    }


def _build_manifest_entries(
    *,
    bucket_dir: Path,
    output_dir: Path,
    paired_seeds: Sequence[int],
    strata: Sequence[Mapping[str, str]],
    train_lineage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    train_seed_values = {
        int(seed) for seed in list(train_lineage.get("accepted_seed_values", []))
    }
    overlap_seed_values = sorted(
        train_seed_values & {int(seed) for seed in paired_seeds}
    )
    if overlap_seed_values:
        raise ValueError(
            "devbench overlaps canonical train lineage on seed values: "
            + ", ".join(str(seed) for seed in overlap_seed_values)
        )

    for stratum_index, row in enumerate(strata):
        stratum_id = str(row["stratum_id"])
        if not paired_seeds:
            raise ValueError(f"empty stratum is forbidden: {stratum_id}")
        for pair_index, seed in enumerate(paired_seeds):
            paired_key = f"seed={int(seed)}|stratum={stratum_id}"
            entries.append(
                {
                    "entry_id": f"dev_{stratum_id}_{int(seed)}",
                    "experiment_split": "dev_only",
                    "scope": "paired_devbench",
                    "seed": int(seed),
                    "pair_index": int(pair_index),
                    "stratum_index": int(stratum_index),
                    "stratum_id": stratum_id,
                    "failure_injection_kind": str(row["failure_injection_kind"]),
                    "apple_visibility": str(row["apple_visibility"]),
                    "paired_key": paired_key,
                    "paired_identity": {
                        "seed": int(seed),
                        "stratum_id": stratum_id,
                    },
                    "baseline_eval": {
                        "entrypoint": str(
                            REPO_ROOT / "agent" / "run" / "45d_vlm_critic_eval_smoke.py"
                        ),
                        "model_path": DEFAULT_ORIGINAL_BASELINE_MODEL,
                        "env_name": DEFAULT_ENV_NAME,
                        "advantage": "None",
                        "max_episode_steps": int(DEFAULT_MAX_EPISODE_STEPS),
                    },
                    "provenance": {
                        "bucket_dir": str(bucket_dir),
                        "output_dir": str(output_dir),
                        "train_lineage_manifest_path": str(
                            train_lineage["source_manifest_path"]
                        ),
                        "overlap_check_identity": str(
                            train_lineage["overlap_check_identity"]
                        ),
                        "overlap_seed_count": 0,
                    },
                }
            )
    return entries


def _validate_manifest_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if not entries:
        raise ValueError("baseline manifest must contain at least one entry")
    paired_keys: set[str] = set()
    counts: Counter[str] = Counter()
    for index, raw_entry in enumerate(entries):
        entry = _as_mapping(raw_entry, field_name=f"entries[{index}]")
        stratum_id = _as_non_empty_string(
            entry.get("stratum_id"), field_name=f"entries[{index}].stratum_id"
        )
        paired_key = _as_non_empty_string(
            entry.get("paired_key"), field_name=f"entries[{index}].paired_key"
        )
        if paired_key in paired_keys:
            raise ValueError(f"duplicate paired key: {paired_key}")
        paired_keys.add(paired_key)
        counts[stratum_id] += 1

    for stratum_id, count in counts.items():
        if int(count) <= 0:
            raise ValueError(f"empty stratum is forbidden: {stratum_id}")
    if dict(counts) != dict(EXPECTED_STRATA_COUNTS):
        raise ValueError(
            "fixed dev-only paired manifest must match expected strata counts: "
            + json.dumps(dict(EXPECTED_STRATA_COUNTS), sort_keys=True)
        )
    if len(entries) != int(sum(EXPECTED_STRATA_COUNTS.values())):
        raise ValueError(
            f"fixed dev-only paired manifest must contain exactly 32 entries, got {len(entries)}"
        )
    return {str(key): int(value) for key, value in sorted(counts.items())}


def _build_baseline_manifest(
    *,
    bucket_dir: Path,
    output_dir: Path,
    artifact_paths: Mapping[str, Path],
    paired_seeds: Sequence[int],
    strata: Sequence[Mapping[str, str]],
    train_lineage: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    stratum_counts: Mapping[str, int],
) -> dict[str, Any]:
    train_seed_values = list(train_lineage.get("accepted_seed_values", []))
    paired_seed_set = {int(seed) for seed in paired_seeds}
    overlap_seed_values = sorted(
        {int(seed) for seed in train_seed_values} & paired_seed_set
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_dev_baseline_manifest",
        "bucket_dir": str(bucket_dir),
        "output_dir": str(output_dir),
        "scope": "dev_only_paired",
        "pairing_identity": ["seed", "stratum_id"],
        "fixed_strata_definition_path": str(
            output_dir / FIXED_STRATA_DEFINITION_JSON_NAME
        ),
        "canonical_gate_path": str(artifact_paths["gate_path"]),
        "canonical_manifest_path": str(artifact_paths["manifest_path"]),
        "bucket_A_sidecar_path": str(artifact_paths["sidecar_path"]),
        "bucket_A_join_coverage_path": str(artifact_paths["join_coverage_path"]),
        "bucket_A_exporter_manifest_path": str(
            artifact_paths["exporter_manifest_path"]
        ),
        "baseline_policy": {
            "kind": "original_baseline",
            "model_path": DEFAULT_ORIGINAL_BASELINE_MODEL,
            "entrypoint": str(
                REPO_ROOT / "agent" / "run" / "45d_vlm_critic_eval_smoke.py"
            ),
            "eval_script": str(REPO_ROOT / "agent" / "run" / "3D_recap_eval.py"),
            "advantage": "None",
        },
        "counts": {
            "entries": int(len(entries)),
            "paired_seed_count": int(len(paired_seeds)),
            "per_stratum": {
                str(key): int(value) for key, value in stratum_counts.items()
            },
        },
        "train_lineage": {
            **dict(train_lineage),
            "overlap_seed_values": overlap_seed_values,
            "overlap_seed_count": int(len(overlap_seed_values)),
        },
        "strata": [dict(row) for row in strata],
        "entries": [dict(entry) for entry in entries],
    }


def _stratum_seed_block(
    entries: Sequence[Mapping[str, Any]], *, stratum_id: str
) -> list[int]:
    block = sorted(
        _as_int(entry.get("seed"), field_name="entry.seed")
        for entry in entries
        if str(entry.get("stratum_id")) == stratum_id
    )
    if not block:
        raise ValueError(f"empty stratum is forbidden: {stratum_id}")
    expected = list(range(block[0], block[0] + len(block)))
    if block != expected:
        raise ValueError(
            f"baseline runner requires contiguous paired seeds per stratum, got {block!r}"
        )
    return block


def _run_baseline_eval(
    *,
    output_dir: Path,
    manifest_path: Path,
    entries: Sequence[Mapping[str, Any]],
    stratum_counts: Mapping[str, int],
) -> dict[str, Any]:
    script_path = REPO_ROOT / "agent" / "run" / "45d_vlm_critic_eval_smoke.py"
    server_script = REPO_ROOT / "agent" / "run" / "3D_recap_run_adv_server.py"
    eval_script = REPO_ROOT / "agent" / "run" / "3D_recap_eval.py"
    runtime_log_dir = output_dir / "runtime_logs"
    artifact_dir = output_dir / "baseline_eval_artifacts"
    telemetry_dir = output_dir / "baseline_eval_telemetry"
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    per_stratum: dict[str, dict[str, Any]] = {}
    total_success_count = 0
    total_evaluated_episodes = 0
    total_requested = 0

    for stratum_id, requested_count in sorted(stratum_counts.items()):
        seed_block = _stratum_seed_block(entries, stratum_id=stratum_id)
        summary_json = output_dir / f"baseline_{stratum_id}.json"
        command = [
            str(Path(sys.executable).resolve()),
            str(script_path),
            "--main-repo-root",
            str(REPO_ROOT),
            "--python",
            str(Path(sys.executable).resolve()),
            "--server-script",
            str(server_script),
            "--eval-script",
            str(eval_script),
            "--summary-json",
            str(summary_json),
            "--runtime-log-dir",
            str(runtime_log_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--telemetry-dir",
            str(telemetry_dir),
            "--model-path",
            str(DEFAULT_ORIGINAL_BASELINE_MODEL),
            "--env-name",
            str(DEFAULT_ENV_NAME),
            "--advantage",
            "None",
            "--eval-label",
            f"state_conditioned_dev_{stratum_id}",
            "--n-episodes",
            str(int(requested_count)),
            "--max-episode-steps",
            str(int(DEFAULT_MAX_EPISODE_STEPS)),
            "--seed-base",
            str(int(seed_block[0])),
        ]
        commands.append(list(command))
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if not summary_json.is_file():
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"returncode={completed.returncode}"
            raise RuntimeError(
                f"baseline eval did not produce summary JSON for {stratum_id}: {detail}"
            )
        summary = _read_json(summary_json)
        success_count = int(summary.get("success_count", 0))
        evaluated_episodes = int(summary.get("episodes", 0))
        total_success_count += success_count
        total_evaluated_episodes += evaluated_episodes
        total_requested += int(requested_count)
        per_stratum[stratum_id] = {
            "requested_count": int(requested_count),
            "evaluated_episodes": int(evaluated_episodes),
            "success_count": int(success_count),
            "success_rate": float(summary.get("success_rate", 0.0)),
            "seed_base": int(seed_block[0]),
            "seed_values": seed_block,
            "summary_json": str(summary_json),
            "wrapper_status": summary.get("wrapper_status"),
            "error": summary.get("error"),
        }

    aggregate_success_rate = (
        float(total_success_count) / float(total_evaluated_episodes)
        if total_evaluated_episodes > 0
        else 0.0
    )
    return {
        "baseline_invocation": {
            "runner": str(script_path),
            "python": str(Path(sys.executable).resolve()),
            "main_repo_root": str(REPO_ROOT),
            "server_script": str(server_script),
            "eval_script": str(eval_script),
            "model_path": DEFAULT_ORIGINAL_BASELINE_MODEL,
            "advantage": "None",
            "manifest_path": str(manifest_path),
            "invocation_mode": "per_stratum_seed_block",
            "commands": commands,
        },
        "aggregate_metrics": {
            "requested_entries": int(total_requested),
            "evaluated_episodes": int(total_evaluated_episodes),
            "success_count": int(total_success_count),
            "success_rate": float(aggregate_success_rate),
        },
        "per_stratum": per_stratum,
    }


def materialize_state_conditioned_dev_manifest(
    *,
    bucket_dir: Path,
    output_dir: Path,
    baseline_runner: BaselineRunner | None = None,
    paired_seeds: Sequence[int] | None = None,
    strata_definition: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    bucket_dir = _validate_existing_dir(bucket_dir, arg_name="bucket-dir")
    output_dir = state_conditioned_bucket_a_import.validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate, manifest, artifact_paths = _load_bucket_preconditions(bucket_dir)
    accepted_entries = _accepted_canonical_entries(manifest)
    train_lineage = _train_lineage(
        accepted_entries,
        manifest_path=artifact_paths["manifest_path"],
    )
    seeds = _normalize_paired_seeds(paired_seeds)
    strata = _normalize_strata_definitions(strata_definition)

    fixed_strata_definition = _build_fixed_strata_definition(
        bucket_dir=bucket_dir,
        artifact_paths=artifact_paths,
        paired_seeds=seeds,
        strata=strata,
    )
    fixed_strata_definition_path = output_dir / FIXED_STRATA_DEFINITION_JSON_NAME
    _write_json(fixed_strata_definition_path, fixed_strata_definition)

    entries = _build_manifest_entries(
        bucket_dir=bucket_dir,
        output_dir=output_dir,
        paired_seeds=seeds,
        strata=strata,
        train_lineage=train_lineage,
    )
    stratum_counts = _validate_manifest_entries(entries)
    baseline_manifest = _build_baseline_manifest(
        bucket_dir=bucket_dir,
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        paired_seeds=seeds,
        strata=strata,
        train_lineage=train_lineage,
        entries=entries,
        stratum_counts=stratum_counts,
    )
    baseline_manifest_path = output_dir / BASELINE_MANIFEST_JSON_NAME
    _write_json(baseline_manifest_path, baseline_manifest)

    runner = _run_baseline_eval if baseline_runner is None else baseline_runner
    baseline_result = runner(
        output_dir=output_dir,
        manifest_path=baseline_manifest_path,
        entries=entries,
        stratum_counts=stratum_counts,
    )
    baseline_invocation = dict(
        _as_mapping(
            baseline_result.get("baseline_invocation"),
            field_name="baseline_result.baseline_invocation",
        )
    )
    aggregate_metrics = dict(
        _as_mapping(
            baseline_result.get("aggregate_metrics"),
            field_name="baseline_result.aggregate_metrics",
        )
    )
    per_stratum = dict(
        _as_mapping(
            baseline_result.get("per_stratum"),
            field_name="baseline_result.per_stratum",
        )
    )

    scorecard = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_dev_baseline_scorecard",
        "bucket_dir": str(bucket_dir),
        "manifest_path": str(baseline_manifest_path),
        "fixed_strata_definition_path": str(fixed_strata_definition_path),
        "canonical_gate_path": str(artifact_paths["gate_path"]),
        "baseline_invocation": baseline_invocation,
        "aggregate_metrics": aggregate_metrics,
        "per_stratum": per_stratum,
        "counts": {
            "requested_entries": int(len(entries)),
            "per_stratum": dict(stratum_counts),
        },
        "train_lineage": dict(baseline_manifest["train_lineage"]),
        "gate_snapshot": {
            "ready": bool(gate.get("ready", False)),
            "required_distinct_accepted_episode_count": int(
                gate.get("required_distinct_accepted_episode_count", 0)
            ),
            "accepted_episode_count": int(gate.get("accepted_episode_count", 0)),
            "distinct_accepted_episode_count": int(
                gate.get("distinct_accepted_episode_count", 0)
            ),
        },
    }
    scorecard_path = output_dir / BASELINE_DEV_SCORECARD_JSON_NAME
    _write_json(scorecard_path, scorecard)

    return {
        "fixed_strata_definition_path": str(fixed_strata_definition_path),
        "baseline_manifest_path": str(baseline_manifest_path),
        "baseline_dev_scorecard_path": str(scorecard_path),
        "entry_count": int(len(entries)),
        "per_stratum": dict(stratum_counts),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_state_conditioned_dev_manifest(
            bucket_dir=args.bucket_dir,
            output_dir=args.output_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BASELINE_DEV_SCORECARD_JSON_NAME",
    "BASELINE_MANIFEST_JSON_NAME",
    "DEFAULT_PAIRED_SEEDS",
    "DEFAULT_STRATA_DEFINITIONS",
    "EXPECTED_STRATA_COUNTS",
    "FIXED_STRATA_DEFINITION_JSON_NAME",
    "SCHEMA_VERSION",
    "build_parser",
    "materialize_state_conditioned_dev_manifest",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
