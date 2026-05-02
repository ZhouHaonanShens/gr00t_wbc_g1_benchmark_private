from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_BUCKET_DIR = Path("agent/artifacts/state_conditioned_materialization/bucket_a")
DEFAULT_DEV_DIR = Path("agent/artifacts/state_conditioned_materialization/devbench")
DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/state_conditioned_materialization/collection"
)
DEFAULT_BUCKET_B_TARGET = 16
DEFAULT_BUCKET_C_TARGET = 24
DEFAULT_EXPERIMENT_SPLIT = "devtrain"
DEFAULT_COLLECTION_TIMEOUT_S = 1500.0
DEFAULT_BUCKET_B_SEED_OFFSET = 1000

BUCKET_B_MANIFEST_JSON_NAME = "bucket_B_manifest.json"
BUCKET_C_MANIFEST_JSON_NAME = "bucket_C_manifest.json"
BUCKET_COLLECTION_SUMMARY_JSON_NAME = "bucket_collection_summary.json"
DATASET_SIDECAR_JSON_NAME = "state_conditioned_sidecar.jsonl"
SCHEMA_VERSION = "g1_state_conditioned_bucket_materialization_v1"
REQUIRED_FAILURE_FAMILIES: tuple[str, ...] = (
    "drop_during_transport",
    "failed_grasp_visible",
    "failed_grasp_occluded",
)
FAILURE_TRIGGER_T_BY_KIND: dict[str, int] = {
    "drop_during_transport": 3,
    "failed_grasp_visible": 1,
    "failed_grasp_occluded": 1,
}
INJECTION_METADATA_FIELDS = (
    "failure_injection_kind",
    "failure_injection_seed",
    "failure_injection_trigger_t",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_dev_manifest
from work.recap.scripts.state_conditioned_common import (
    exception_message as _exception_message,
)
from work.recap.scripts.state_conditioned_common import read_json as _read_json
from work.recap.scripts.state_conditioned_common import (
    read_jsonl_dicts as _read_jsonl_dicts,
)
from work.recap.scripts.state_conditioned_common import (
    validate_existing_dir as _validate_existing_dir,
)
from work.recap.scripts.state_conditioned_common import write_json as _write_json
from work.recap.scripts.state_conditioned_common import write_jsonl as _write_jsonl


CollectionRunner = Callable[..., dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize T7 Bucket B nominal expansion and Bucket C injected off-nominal "
            "episodes only after canonical Bucket A Gate A and T6 dev artifacts are ready."
        )
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=DEFAULT_BUCKET_DIR,
        help="Canonical Bucket A directory containing bucket_A_gate_a_ready.json.",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=DEFAULT_DEV_DIR,
        help=(
            "T6 devbench directory containing fixed_strata_definition.json, baseline_manifest.json, "
            "and baseline_dev_scorecard.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory that receives bucket_B_manifest.json, bucket_C_manifest.json, and "
            "bucket_collection_summary.json."
        ),
    )
    parser.add_argument(
        "--bucket-b-target",
        type=int,
        default=int(DEFAULT_BUCKET_B_TARGET),
        help="Canonical T7 Bucket B target episode count (fixed at 16).",
    )
    parser.add_argument(
        "--bucket-c-target",
        type=int,
        default=int(DEFAULT_BUCKET_C_TARGET),
        help="Canonical T7 Bucket C target episode count (fixed at 24 = 3 families x 8).",
    )
    return parser


def _group_transitions_by_episode(
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, raw_transition in enumerate(transitions):
        transition = dict(
            _as_mapping(raw_transition, field_name=f"transitions[{index}]")
        )
        episode_id = _as_non_empty_string(
            transition.get("episode_id"), field_name=f"transitions[{index}].episode_id"
        )
        grouped.setdefault(episode_id, []).append(transition)
    for episode_id in grouped:
        grouped[episode_id] = sorted(grouped[episode_id], key=lambda row: int(row["t"]))
    return grouped


def _default_phase_mode_for_dataset_episode(
    episode_record: Mapping[str, Any],
) -> tuple[str, str]:
    failure_kind = _optional_non_empty_string(
        episode_record.get("failure_injection_kind"),
        field_name="episode_record.failure_injection_kind",
    )
    if failure_kind == "drop_during_transport":
        return "TRANSPORT", "RECOVERY"
    if failure_kind in {"failed_grasp_visible", "failed_grasp_occluded"}:
        return "SEARCH", "RECOVERY"
    return "TRANSPORT", "NOMINAL"


def ensure_dataset_state_conditioned_sidecar(
    dataset_dir: Path,
) -> dict[str, Any]:
    dataset_dir = _validate_existing_dir(dataset_dir, arg_name="dataset-dir")
    episodes_path = dataset_dir / "episodes.jsonl"
    transitions_path = dataset_dir / "transitions.jsonl"
    if not episodes_path.is_file():
        raise ValueError(f"missing required dataset file: {episodes_path}")
    if not transitions_path.is_file():
        raise ValueError(f"missing required dataset file: {transitions_path}")

    episodes = _read_jsonl_dicts(episodes_path)
    transitions = _read_jsonl_dicts(transitions_path)
    transitions_by_episode = _group_transitions_by_episode(transitions)
    sidecar_rows: list[dict[str, Any]] = []
    for index, raw_episode in enumerate(episodes):
        episode_record = dict(
            _as_mapping(raw_episode, field_name=f"episodes.jsonl[{index}]")
        )
        episode_id = _as_non_empty_string(
            episode_record.get("episode_id"),
            field_name=f"episodes.jsonl[{index}].episode_id",
        )
        phase, mode = _default_phase_mode_for_dataset_episode(episode_record)
        episode_transitions = transitions_by_episode.get(episode_id, [])
        if not episode_transitions:
            raise ValueError(
                f"dataset {dataset_dir} is missing transitions for episode_id={episode_id!r}"
            )
        for transition in episode_transitions:
            t_value = _as_int(transition.get("t"), field_name="transition.t")
            row = state_conditioned_bucket_a_import._build_minimal_history_aware_sidecar_row(
                episode_id,
                int(t_value),
                transition=transition,
            )
            row["policy_condition.phase"] = phase
            row["policy_condition.mode"] = mode
            row["policy_condition_text"] = (
                state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
                    phase,
                    mode,
                )
            )
            sidecar_rows.append(row)

    sidecar_path = dataset_dir / DATASET_SIDECAR_JSON_NAME
    _write_jsonl(sidecar_path, sidecar_rows)
    validation = state_conditioned_bucket_a_import.validate_sidecar_round_trip(
        sidecar_path=sidecar_path,
        expected_join_keys=[
            [str(record["episode_id"]), int(record["t"])] for record in transitions
        ],
    )
    return {
        "dataset_dir": str(dataset_dir),
        "sidecar_path": str(sidecar_path),
        "record_count": _as_int(
            validation.get("record_count"), field_name="validation.record_count"
        ),
        "history_k": _as_int(
            validation.get("history_k"), field_name="validation.history_k"
        ),
        "history_stride": _as_int(
            validation.get("history_stride"), field_name="validation.history_stride"
        ),
    }


def backfill_state_conditioned_sidecars_from_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"manifest path does not exist: {manifest_path}")
    manifest = _read_json(manifest_path)
    episodes = [
        dict(_as_mapping(item, field_name=f"{manifest_path.name}.episodes[]"))
        for item in _as_list(
            manifest.get("episodes"), field_name=f"{manifest_path.name}.episodes"
        )
    ]
    results: list[dict[str, Any]] = []
    for episode in episodes:
        dataset_dir = (
            Path(
                _as_non_empty_string(
                    episode.get("dataset_dir"),
                    field_name="manifest episode.dataset_dir",
                )
            )
            .expanduser()
            .resolve()
        )
        results.append(ensure_dataset_state_conditioned_sidecar(dataset_dir))
    return {
        "manifest_path": str(manifest_path),
        "dataset_count": int(len(results)),
        "sidecar_paths": [str(item["sidecar_path"]) for item in results],
    }


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    return state_conditioned_bucket_a_import._as_non_empty_string(
        value,
        field_name=field_name,
    )


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _optional_non_empty_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    return normalized if normalized else None


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _load_t7_preconditions(
    bucket_dir: Path,
    dev_dir: Path,
) -> dict[str, Any]:
    gate_path = bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME
    canonical_manifest_path = (
        bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME
    )
    fixed_strata_definition_path = (
        dev_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME
    )
    baseline_manifest_path = (
        dev_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME
    )
    baseline_dev_scorecard_path = (
        dev_dir / state_conditioned_dev_manifest.BASELINE_DEV_SCORECARD_JSON_NAME
    )
    required_paths = {
        "gate_path": gate_path,
        "canonical_manifest_path": canonical_manifest_path,
        "fixed_strata_definition_path": fixed_strata_definition_path,
        "baseline_manifest_path": baseline_manifest_path,
        "baseline_dev_scorecard_path": baseline_dev_scorecard_path,
    }
    for name, path in required_paths.items():
        if not path.is_file():
            raise ValueError(f"missing required T6 artifact {name}: {path}")

    gate = _read_json(gate_path)
    if not bool(gate.get("ready", False)):
        raise ValueError(
            "state-conditioned bucket collection refuses to run until "
            "bucket_A_gate_a_ready.json.ready == true"
        )

    fixed_strata_definition = _read_json(fixed_strata_definition_path)
    baseline_manifest = _read_json(baseline_manifest_path)
    baseline_dev_scorecard = _read_json(baseline_dev_scorecard_path)

    paired_seed_values_raw = fixed_strata_definition.get("paired_seed_values")
    if not isinstance(paired_seed_values_raw, list):
        raise TypeError("fixed_strata_definition.paired_seed_values must be a list")
    paired_seed_values = [
        _as_int(seed, field_name=f"paired_seed_values[{index}]")
        for index, seed in enumerate(paired_seed_values_raw)
    ]
    if len(paired_seed_values) != 8:
        raise ValueError(
            "T6 fixed strata must carry exactly 8 paired_seed_values, got "
            + str(len(paired_seed_values))
        )
    paired_seed_count = _as_int(
        fixed_strata_definition.get("paired_seed_count"),
        field_name="fixed_strata_definition.paired_seed_count",
    )
    if paired_seed_count != len(paired_seed_values):
        raise ValueError(
            "fixed_strata_definition paired_seed_count mismatch: "
            + f"declared={paired_seed_count} actual={len(paired_seed_values)}"
        )

    counts = dict(
        _as_mapping(
            _as_mapping(
                baseline_manifest.get("counts"),
                field_name="baseline_manifest.counts",
            ).get("per_stratum"),
            field_name="baseline_manifest.counts.per_stratum",
        )
    )
    normalized_counts = {
        _as_non_empty_string(
            key, field_name="baseline_manifest.counts.per_stratum.key"
        ): _as_int(value, field_name=f"baseline_manifest.counts.per_stratum[{key}]")
        for key, value in counts.items()
    }
    if normalized_counts != dict(state_conditioned_dev_manifest.EXPECTED_STRATA_COUNTS):
        raise ValueError(
            "T6 baseline manifest per_stratum counts mismatch: "
            + json.dumps(dict(normalized_counts), ensure_ascii=True, sort_keys=True)
        )

    requested_entries = _as_int(
        _as_mapping(
            baseline_dev_scorecard.get("counts"),
            field_name="baseline_dev_scorecard.counts",
        ).get("requested_entries"),
        field_name="baseline_dev_scorecard.counts.requested_entries",
    )
    if requested_entries != int(
        sum(state_conditioned_dev_manifest.EXPECTED_STRATA_COUNTS.values())
    ):
        raise ValueError(
            "T6 baseline_dev_scorecard requested_entries mismatch: "
            + f"expected 32, got {requested_entries}"
        )

    stable_base_checkpoint_kind, stable_base_checkpoint_value = (
        _stable_base_checkpoint_reference(
            baseline_manifest=baseline_manifest,
            baseline_dev_scorecard=baseline_dev_scorecard,
        )
    )
    return {
        "gate": gate,
        "fixed_strata_definition": fixed_strata_definition,
        "baseline_manifest": baseline_manifest,
        "baseline_dev_scorecard": baseline_dev_scorecard,
        "paired_seed_values": paired_seed_values,
        "stable_base_checkpoint_kind": stable_base_checkpoint_kind,
        "stable_base_checkpoint_value": stable_base_checkpoint_value,
        **required_paths,
    }


def _stable_base_checkpoint_reference(
    *,
    baseline_manifest: Mapping[str, Any],
    baseline_dev_scorecard: Mapping[str, Any],
) -> tuple[str, str]:
    baseline_policy = dict(
        _as_mapping(
            baseline_manifest.get("baseline_policy"),
            field_name="baseline_manifest.baseline_policy",
        )
    )
    overlay_from = _optional_non_empty_string(
        baseline_policy.get("overlay_from"),
        field_name="baseline_manifest.baseline_policy.overlay_from",
    )
    if overlay_from is not None:
        return "overlay_from", overlay_from

    model_path = _optional_non_empty_string(
        baseline_policy.get("model_path"),
        field_name="baseline_manifest.baseline_policy.model_path",
    )
    if model_path is not None:
        return "model_path", model_path

    baseline_invocation = dict(
        _as_mapping(
            baseline_dev_scorecard.get("baseline_invocation"),
            field_name="baseline_dev_scorecard.baseline_invocation",
        )
    )
    fallback_model_path = _optional_non_empty_string(
        baseline_invocation.get("model_path"),
        field_name="baseline_dev_scorecard.baseline_invocation.model_path",
    )
    if fallback_model_path is not None:
        return "model_path", fallback_model_path

    raise ValueError(
        "T6 baseline artifacts are missing stable base checkpoint reference"
    )


def build_default_bucket_plans(
    *,
    paired_seed_values: Sequence[int],
    bucket_b_target: int,
    bucket_c_target: int,
) -> dict[str, list[dict[str, Any]]]:
    seeds = [
        _as_int(seed, field_name=f"paired_seed_values[{index}]")
        for index, seed in enumerate(paired_seed_values)
    ]
    if len(seeds) != 8:
        raise ValueError(
            f"bucket plans require exactly 8 paired seeds, got {len(seeds)}"
        )

    expected_bucket_b_target = len(seeds) * 2
    if int(bucket_b_target) != int(expected_bucket_b_target):
        raise ValueError(
            f"bucket-b-target must be exactly {expected_bucket_b_target}, got {bucket_b_target}"
        )
    expected_bucket_c_target = len(seeds) * len(REQUIRED_FAILURE_FAMILIES)
    if int(bucket_c_target) != int(expected_bucket_c_target):
        raise ValueError(
            f"bucket-c-target must be exactly {expected_bucket_c_target}, got {bucket_c_target}"
        )

    bucket_b: list[dict[str, Any]] = []
    for round_index in range(2):
        for seed in seeds:
            bucket_b.append(
                {
                    "bucket_key": "bucket_B",
                    "bucket_name": "Bucket B",
                    "collection_kind": "nominal_expansion",
                    "seed": int(seed) + int(round_index * DEFAULT_BUCKET_B_SEED_OFFSET),
                    "nominal_expansion_round": int(round_index),
                }
            )

    bucket_c: list[dict[str, Any]] = []
    for failure_injection_kind in REQUIRED_FAILURE_FAMILIES:
        trigger_t = FAILURE_TRIGGER_T_BY_KIND[failure_injection_kind]
        for seed in seeds:
            bucket_c.append(
                {
                    "bucket_key": "bucket_C",
                    "bucket_name": "Bucket C",
                    "collection_kind": "injected_off_nominal",
                    "seed": int(seed),
                    "failure_injection_kind": failure_injection_kind,
                    "failure_injection_seed": int(seed),
                    "failure_injection_trigger_t": int(trigger_t),
                }
            )
    return {"bucket_B": bucket_b, "bucket_C": bucket_c}


def _default_collection_runner(
    *,
    output_dir: Path,
    bucket_key: str,
    plan_index: int,
    plan_entry: Mapping[str, Any],
) -> dict[str, Any]:
    collect_script = (
        REPO_ROOT / state_conditioned_bucket_a_import.RECAP_COLLECT_SCRIPT_REL
    )
    if not collect_script.is_file():
        raise ValueError(f"missing collector script: {collect_script}")

    family_tag = state_conditioned_bucket_a_import._sanitize_tag_component(
        str(plan_entry.get("failure_injection_kind", "nominal"))
    )
    iter_tag = (
        f"state_conditioned_{str(bucket_key).lower()}_{family_tag}_"
        f"{int(plan_index):03d}_{state_conditioned_bucket_a_import._now_tag()}"
    )
    dataset_dir = (
        REPO_ROOT / state_conditioned_bucket_a_import.RECAP_DATASET_DIR_REL / iter_tag
    )
    bootstrap = (
        "import importlib, runpy, sys, types\n"
        "script = sys.argv[1]\n"
        "args = sys.argv[2:]\n"
        "try:\n"
        "    obj_utils = importlib.import_module('robocasa.utils.object_utils')\n"
        "    if not hasattr(obj_utils, 'check_obj_upright'):\n"
        "        obj_cos_fn = getattr(obj_utils, 'obj_cos', None)\n"
        "        def check_obj_upright(env, obj_name, threshold=0.8, symmetric=False):\n"
        "            if not callable(obj_cos_fn):\n"
        "                return False\n"
        "            try:\n"
        "                z_alignment = float(obj_cos_fn(env, obj_name=obj_name, ref=(0, 0, 1)))\n"
        "            except Exception:\n"
        "                return False\n"
        "            if bool(symmetric):\n"
        "                z_alignment = abs(z_alignment)\n"
        "            return bool(z_alignment > float(threshold))\n"
        "        setattr(obj_utils, 'check_obj_upright', check_obj_upright)\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    importlib.import_module('robocasa.utils.visuals_utls')\n"
        "except ModuleNotFoundError:\n"
        "    module_obj = types.ModuleType('robocasa.utils.visuals_utls')\n"
        "    class Gradient:\n"
        "        def __init__(self, *_args, **_kwargs):\n"
        "            return None\n"
        "    def randomize_materials_rgba(*_args, **_kwargs):\n"
        "        return None\n"
        "    setattr(module_obj, 'Gradient', Gradient)\n"
        "    setattr(module_obj, 'randomize_materials_rgba', randomize_materials_rgba)\n"
        "    sys.modules['robocasa.utils.visuals_utls'] = module_obj\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    importlib.import_module('robocasa.wrappers.ik_wrapper')\n"
        "except ModuleNotFoundError:\n"
        "    wrappers_mod = types.ModuleType('robocasa.wrappers')\n"
        "    ik_mod = types.ModuleType('robocasa.wrappers.ik_wrapper')\n"
        "    class IKWrapper:\n"
        "        def __init__(self, env, **_kwargs):\n"
        "            self.env = env\n"
        "        def __getattr__(self, name):\n"
        "            return getattr(self.env, name)\n"
        "    setattr(ik_mod, 'IKWrapper', IKWrapper)\n"
        "    sys.modules.setdefault('robocasa.wrappers', wrappers_mod)\n"
        "    sys.modules['robocasa.wrappers.ik_wrapper'] = ik_mod\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    robots_mod = importlib.import_module('robocasa.models.robots')\n"
        "    if not hasattr(robots_mod, 'GR00T_LOCOMANIP_ENVS_ROBOTS'):\n"
        "        setattr(robots_mod, 'GR00T_LOCOMANIP_ENVS_ROBOTS', {'G1': 'g1_sim'})\n"
        "    if not hasattr(robots_mod, 'remove_mimic_joints'):\n"
        "        def remove_mimic_joints(_gripper, action):\n"
        "            return action\n"
        "        setattr(robots_mod, 'remove_mimic_joints', remove_mimic_joints)\n"
        "except Exception:\n"
        "    pass\n"
        "sys.argv = [script, *args]\n"
        "runpy.run_path(script, run_name='__main__')\n"
    )
    command = [
        state_conditioned_bucket_a_import._preferred_live_python(REPO_ROOT),
        "-c",
        bootstrap,
        str(collect_script),
        "--iter-tag",
        iter_tag,
        "--seed",
        str(_as_int(plan_entry.get("seed"), field_name="plan_entry.seed")),
        "--n-episodes",
        "1",
        "--kill-server-on-exit",
        "--total-timeout-s",
        str(float(DEFAULT_COLLECTION_TIMEOUT_S)),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "GR00T_SKIP_WBC_REEXEC": "1",
            "PYTHONPATH": os.pathsep.join(
                state_conditioned_bucket_a_import._build_live_pythonpath(REPO_ROOT)
            ),
        },
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr if stderr else stdout
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RuntimeError(
            f"bucket collection failed rc={completed.returncode}: {detail or 'no output captured'}"
        )

    episode_index = state_conditioned_bucket_a_import._load_episode_index(dataset_dir)
    episode_order = list(episode_index["episode_order"])
    if len(episode_order) != 1:
        raise ValueError(
            "bucket collection requires exactly 1 episode per plan entry, got "
            + str(len(episode_order))
        )
    return {
        "iter_tag": iter_tag,
        "dataset_dir": str(dataset_dir),
        "episodes_path": str(episode_index["episodes_path"]),
        "transitions_path": str(dataset_dir / "transitions.jsonl"),
        "episode_order": episode_order,
        "episodes_by_id": dict(episode_index["episodes_by_id"]),
        "materialized_episode_count": int(len(episode_order)),
        "collected_episode_count": int(len(episode_order)),
        "collection_command": command,
        "runtime_log_path": str(
            REPO_ROOT / "agent" / "runtime_logs" / iter_tag / "collect.log"
        ),
        "materialization_mode": (
            "state_conditioned_bucket_B_nominal_expansion"
            if str(bucket_key) == "bucket_B"
            else "state_conditioned_bucket_C_injected_off_nominal_request"
        ),
        "reused_existing_live_dataset": False,
    }


def _normalize_bucket_plans(
    raw_bucket_plans: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    bucket_b_target: int,
    bucket_c_target: int,
) -> dict[str, list[dict[str, Any]]]:
    plans: dict[str, list[dict[str, Any]]] = {}
    for bucket_key in ("bucket_B", "bucket_C"):
        raw_entries = raw_bucket_plans.get(bucket_key)
        if raw_entries is None:
            raise ValueError(f"missing bucket plan entries for {bucket_key}")
        plans[bucket_key] = [
            dict(_as_mapping(entry, field_name=f"{bucket_key}.plan[]"))
            for entry in raw_entries
        ]
    if len(plans["bucket_B"]) != int(bucket_b_target):
        raise ValueError(
            f"Bucket B plan must contain exactly {bucket_b_target} entries, got {len(plans['bucket_B'])}"
        )
    if len(plans["bucket_C"]) != int(bucket_c_target):
        raise ValueError(
            f"Bucket C plan must contain exactly {bucket_c_target} entries, got {len(plans['bucket_C'])}"
        )
    return plans


def _single_episode_record(
    collection_result: Mapping[str, Any],
) -> tuple[str, dict[str, Any], Path]:
    episode_order = list(collection_result.get("episode_order", []))
    if len(episode_order) != 1:
        raise ValueError(
            "bucket materialization requires collection_result.episode_order to contain exactly 1 episode"
        )
    episode_id = _as_non_empty_string(
        episode_order[0], field_name="collection_result.episode_order[0]"
    )
    episodes_path = Path(
        _as_non_empty_string(
            collection_result.get("episodes_path"),
            field_name="collection_result.episodes_path",
        )
    )
    episodes = _read_jsonl_dicts(episodes_path)
    if len(episodes) != 1:
        raise ValueError(
            f"bucket materialization requires exactly 1 record in {episodes_path}, got {len(episodes)}"
        )
    record = dict(_as_mapping(episodes[0], field_name="episodes.jsonl[0]"))
    if str(record.get("episode_id")) != episode_id:
        raise ValueError(
            f"episode_id mismatch between collection_result and episodes.jsonl: {episode_id} != {record.get('episode_id')!r}"
        )
    return episode_id, record, episodes_path


def _build_episode_provenance(
    *,
    bucket_key: str,
    plan_entry: Mapping[str, Any],
    collection_result: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": (
            "state_conditioned_bucket_B_nominal_expansion"
            if str(bucket_key) == "bucket_B"
            else "state_conditioned_bucket_C_injected_off_nominal"
        ),
        "bucket_key": str(bucket_key),
        "source_dataset_dir": _as_non_empty_string(
            collection_result.get("dataset_dir"),
            field_name="collection_result.dataset_dir",
        ),
        "iter_tag": _optional_non_empty_string(
            collection_result.get("iter_tag"),
            field_name="collection_result.iter_tag",
        ),
        "materialization_mode": _optional_non_empty_string(
            collection_result.get("materialization_mode"),
            field_name="collection_result.materialization_mode",
        ),
        "reused_existing_live_dataset": bool(
            collection_result.get("reused_existing_live_dataset", False)
        ),
        "collection_request": {
            key: value
            for key, value in dict(plan_entry).items()
            if key
            not in {
                "bucket_key",
                "bucket_name",
            }
        },
        "canonical_gate_path": str(preconditions["gate_path"]),
        "canonical_manifest_path": str(preconditions["canonical_manifest_path"]),
        "fixed_strata_definition_path": str(
            preconditions["fixed_strata_definition_path"]
        ),
        "baseline_manifest_path": str(preconditions["baseline_manifest_path"]),
        "baseline_dev_scorecard_path": str(
            preconditions["baseline_dev_scorecard_path"]
        ),
        "collection_command": list(collection_result.get("collection_command", [])),
        "runtime_log_path": collection_result.get("runtime_log_path"),
    }


def _enrich_bucket_episode_record(
    *,
    bucket_key: str,
    episode_record: Mapping[str, Any],
    plan_entry: Mapping[str, Any],
    collection_result: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(episode_record)
    existing_injection_fields = [
        field for field in INJECTION_METADATA_FIELDS if field in record
    ]
    if str(bucket_key) == "bucket_B":
        if existing_injection_fields:
            raise ValueError(
                "Bucket B must not contain injection metadata: "
                + ", ".join(existing_injection_fields)
            )
        for field in INJECTION_METADATA_FIELDS:
            if field in plan_entry:
                raise ValueError(
                    "Bucket B plan must not request injection metadata: " + field
                )
    else:
        missing_injection_fields = [
            field for field in INJECTION_METADATA_FIELDS if field not in plan_entry
        ]
        if missing_injection_fields:
            raise ValueError(
                "Bucket C injected episode is missing injection metadata: "
                + ", ".join(missing_injection_fields)
            )

    provenance = _build_episode_provenance(
        bucket_key=bucket_key,
        plan_entry=plan_entry,
        collection_result=collection_result,
        preconditions=preconditions,
    )
    enriched = {
        **record,
        "experiment_split": str(DEFAULT_EXPERIMENT_SPLIT),
        "stable_base_checkpoint_kind": str(
            preconditions["stable_base_checkpoint_kind"]
        ),
        "stable_base_checkpoint_value": str(
            preconditions["stable_base_checkpoint_value"]
        ),
        "provenance": provenance,
    }
    if str(bucket_key) == "bucket_C":
        enriched["failure_injection_kind"] = _as_non_empty_string(
            plan_entry.get("failure_injection_kind"),
            field_name="plan_entry.failure_injection_kind",
        )
        enriched["failure_injection_seed"] = _as_int(
            plan_entry.get("failure_injection_seed"),
            field_name="plan_entry.failure_injection_seed",
        )
        enriched["failure_injection_trigger_t"] = _as_int(
            plan_entry.get("failure_injection_trigger_t"),
            field_name="plan_entry.failure_injection_trigger_t",
        )
    return enriched


def _write_back_single_episode_record(
    path: Path, episode_record: Mapping[str, Any]
) -> None:
    _write_jsonl(path, [episode_record])


def _ensure_sidecar_for_collection_result(
    collection_result: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_dir = (
        Path(
            _as_non_empty_string(
                collection_result.get("dataset_dir"),
                field_name="collection_result.dataset_dir",
            )
        )
        .expanduser()
        .resolve()
    )
    return ensure_dataset_state_conditioned_sidecar(dataset_dir)


def _build_manifest_entry(
    *,
    bucket_key: str,
    collection_result: Mapping[str, Any],
    episode_record: Mapping[str, Any],
) -> dict[str, Any]:
    entry = {
        "bucket_key": str(bucket_key),
        "episode_id": _as_non_empty_string(
            episode_record.get("episode_id"),
            field_name="episode_record.episode_id",
        ),
        "seed": episode_record.get("seed"),
        "success_episode": bool(episode_record.get("success_episode", False)),
        "experiment_split": _as_non_empty_string(
            episode_record.get("experiment_split"),
            field_name="episode_record.experiment_split",
        ),
        "stable_base_checkpoint_kind": _as_non_empty_string(
            episode_record.get("stable_base_checkpoint_kind"),
            field_name="episode_record.stable_base_checkpoint_kind",
        ),
        "stable_base_checkpoint_value": _as_non_empty_string(
            episode_record.get("stable_base_checkpoint_value"),
            field_name="episode_record.stable_base_checkpoint_value",
        ),
        "provenance": dict(
            _as_mapping(
                episode_record.get("provenance"),
                field_name="episode_record.provenance",
            )
        ),
        "dataset_dir": _as_non_empty_string(
            collection_result.get("dataset_dir"),
            field_name="collection_result.dataset_dir",
        ),
        "episodes_path": _as_non_empty_string(
            collection_result.get("episodes_path"),
            field_name="collection_result.episodes_path",
        ),
        "transitions_path": _as_non_empty_string(
            collection_result.get("transitions_path"),
            field_name="collection_result.transitions_path",
        ),
        "runtime_log_path": collection_result.get("runtime_log_path"),
        "collection_command": list(collection_result.get("collection_command", [])),
    }
    if str(bucket_key) == "bucket_C":
        for field in INJECTION_METADATA_FIELDS:
            entry[field] = episode_record[field]
    return entry


def _collect_bucket_entries(
    *,
    bucket_key: str,
    plans: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    output_dir: Path,
    collection_runner: CollectionRunner,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for plan_index, raw_plan_entry in enumerate(plans):
        plan_entry = dict(
            _as_mapping(raw_plan_entry, field_name=f"{bucket_key}.plans[{plan_index}]")
        )
        collection_result = collection_runner(
            output_dir=output_dir,
            bucket_key=bucket_key,
            plan_index=int(plan_index),
            plan_entry=plan_entry,
        )
        _episode_id, episode_record, episodes_path = _single_episode_record(
            collection_result
        )
        enriched_record = _enrich_bucket_episode_record(
            bucket_key=bucket_key,
            episode_record=episode_record,
            plan_entry=plan_entry,
            collection_result=collection_result,
            preconditions=preconditions,
        )
        _write_back_single_episode_record(episodes_path, enriched_record)
        _ensure_sidecar_for_collection_result(collection_result)
        entries.append(
            _build_manifest_entry(
                bucket_key=bucket_key,
                collection_result=collection_result,
                episode_record=enriched_record,
            )
        )
    return entries


def _validate_bucket_b_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(entries) != int(DEFAULT_BUCKET_B_TARGET):
        raise ValueError(
            f"Bucket B must contain exactly 16 episodes, got {len(entries)}"
        )
    for index, raw_entry in enumerate(entries):
        entry = dict(_as_mapping(raw_entry, field_name=f"bucket_B.entries[{index}]"))
        contamination = [field for field in INJECTION_METADATA_FIELDS if field in entry]
        if contamination:
            raise ValueError(
                "Bucket B must not contain injection metadata: "
                + ", ".join(contamination)
            )
    return {
        "episodes": int(len(entries)),
        "nominal_expansion_episodes": int(len(entries)),
        "injected_episode_count": 0,
    }


def _validate_bucket_c_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(entries) != int(DEFAULT_BUCKET_C_TARGET):
        raise ValueError(
            f"Bucket C must contain exactly 24 episodes, got {len(entries)}"
        )
    counts: Counter[str] = Counter()
    for index, raw_entry in enumerate(entries):
        entry = dict(_as_mapping(raw_entry, field_name=f"bucket_C.entries[{index}]"))
        missing = [field for field in INJECTION_METADATA_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                "Bucket C injected episode is missing injection metadata: "
                + ", ".join(missing)
            )
        family = _as_non_empty_string(
            entry.get("failure_injection_kind"),
            field_name=f"bucket_C.entries[{index}].failure_injection_kind",
        )
        _ = _as_int(
            entry.get("failure_injection_seed"),
            field_name=f"bucket_C.entries[{index}].failure_injection_seed",
        )
        _ = _as_int(
            entry.get("failure_injection_trigger_t"),
            field_name=f"bucket_C.entries[{index}].failure_injection_trigger_t",
        )
        counts[family] += 1
    normalized_counts = {str(key): int(value) for key, value in sorted(counts.items())}
    expected_counts = {family: 8 for family in REQUIRED_FAILURE_FAMILIES}
    if normalized_counts != expected_counts:
        raise ValueError(
            "Bucket C must contain exactly 3 failure families with 8 episodes each, got "
            + json.dumps(normalized_counts, ensure_ascii=True, sort_keys=True)
        )
    return {
        "episodes": int(len(entries)),
        "injected_episode_count": int(len(entries)),
        "per_failure_family": normalized_counts,
    }


def _build_bucket_manifest(
    *,
    bucket_key: str,
    entries: Sequence[Mapping[str, Any]],
    counts: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": f"state_conditioned_{bucket_key}_manifest",
        "bucket_key": str(bucket_key),
        "bucket_name": "Bucket B" if str(bucket_key) == "bucket_B" else "Bucket C",
        "output_dir": str(output_dir),
        "bucket_dir": str(preconditions["gate_path"].parent),
        "dev_dir": str(preconditions["baseline_manifest_path"].parent),
        "experiment_split": str(DEFAULT_EXPERIMENT_SPLIT),
        "stable_base_checkpoint_kind": str(
            preconditions["stable_base_checkpoint_kind"]
        ),
        "stable_base_checkpoint_value": str(
            preconditions["stable_base_checkpoint_value"]
        ),
        "canonical_gate_path": str(preconditions["gate_path"]),
        "canonical_manifest_path": str(preconditions["canonical_manifest_path"]),
        "fixed_strata_definition_path": str(
            preconditions["fixed_strata_definition_path"]
        ),
        "baseline_manifest_path": str(preconditions["baseline_manifest_path"]),
        "baseline_dev_scorecard_path": str(
            preconditions["baseline_dev_scorecard_path"]
        ),
        "counts": dict(counts),
        "episodes": [dict(entry) for entry in entries],
    }


@dataclass
class BucketPreconditionLoader:
    bucket_dir: Path
    dev_dir: Path

    def load(self) -> dict[str, Any]:
        return _load_t7_preconditions(self.bucket_dir, self.dev_dir)


@dataclass
class BucketPlanBuilder:
    preconditions: Mapping[str, Any]
    bucket_b_target: int
    bucket_c_target: int
    bucket_plans: Mapping[str, Sequence[Mapping[str, object]]] | None = None

    def build(self) -> dict[str, list[dict[str, object]]]:
        if self.bucket_plans is None:
            return build_default_bucket_plans(
                paired_seed_values=self.preconditions["paired_seed_values"],
                bucket_b_target=int(self.bucket_b_target),
                bucket_c_target=int(self.bucket_c_target),
            )
        return _normalize_bucket_plans(
            self.bucket_plans,
            bucket_b_target=int(self.bucket_b_target),
            bucket_c_target=int(self.bucket_c_target),
        )


@dataclass
class BucketManifestBuilder:
    preconditions: Mapping[str, Any]
    output_dir: Path

    def build(
        self, *, bucket_key: str, entries: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if bucket_key == "bucket_B":
            counts = _validate_bucket_b_entries(entries)
        elif bucket_key == "bucket_C":
            counts = _validate_bucket_c_entries(entries)
        else:
            raise ValueError(f"unsupported bucket_key: {bucket_key!r}")
        manifest = _build_bucket_manifest(
            bucket_key=bucket_key,
            entries=entries,
            counts=counts,
            preconditions=self.preconditions,
            output_dir=self.output_dir,
        )
        return manifest, counts


@dataclass
class StateConditionedBucketCollectionWorkflow:
    bucket_dir: Path
    dev_dir: Path
    output_dir: Path
    collection_runner: CollectionRunner | None = None
    bucket_b_target: int = DEFAULT_BUCKET_B_TARGET
    bucket_c_target: int = DEFAULT_BUCKET_C_TARGET
    bucket_plans: Mapping[str, Sequence[Mapping[str, object]]] | None = None
    preconditions: dict[str, Any] = field(init=False)
    normalized_bucket_plans: dict[str, list[dict[str, object]]] = field(init=False)
    runner: CollectionRunner = field(init=False)

    def __post_init__(self) -> None:
        self.bucket_dir = _validate_existing_dir(self.bucket_dir, arg_name="bucket-dir")
        self.dev_dir = _validate_existing_dir(self.dev_dir, arg_name="dev-dir")
        self.output_dir = state_conditioned_bucket_a_import.validate_output_dir(
            self.output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preconditions = BucketPreconditionLoader(
            bucket_dir=self.bucket_dir,
            dev_dir=self.dev_dir,
        ).load()
        self.normalized_bucket_plans = BucketPlanBuilder(
            preconditions=self.preconditions,
            bucket_b_target=int(self.bucket_b_target),
            bucket_c_target=int(self.bucket_c_target),
            bucket_plans=self.bucket_plans,
        ).build()
        self.runner = (
            _default_collection_runner
            if self.collection_runner is None
            else self.collection_runner
        )

    def collect_bucket(self, bucket_key: str) -> list[dict[str, Any]]:
        return _collect_bucket_entries(
            bucket_key=bucket_key,
            plans=self.normalized_bucket_plans[bucket_key],
            preconditions=self.preconditions,
            output_dir=self.output_dir,
            collection_runner=self.runner,
        )

    def build_summary(
        self,
        *,
        bucket_b_manifest_path: Path,
        bucket_c_manifest_path: Path,
        bucket_b_counts: Mapping[str, Any],
        bucket_c_counts: Mapping[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_bucket_collection_summary",
            "output_dir": str(self.output_dir),
            "bucket_dir": str(self.bucket_dir),
            "dev_dir": str(self.dev_dir),
            "experiment_split": str(DEFAULT_EXPERIMENT_SPLIT),
            "stable_base_checkpoint_kind": str(
                self.preconditions["stable_base_checkpoint_kind"]
            ),
            "stable_base_checkpoint_value": str(
                self.preconditions["stable_base_checkpoint_value"]
            ),
            "canonical_gate_path": str(self.preconditions["gate_path"]),
            "canonical_manifest_path": str(
                self.preconditions["canonical_manifest_path"]
            ),
            "fixed_strata_definition_path": str(
                self.preconditions["fixed_strata_definition_path"]
            ),
            "baseline_manifest_path": str(self.preconditions["baseline_manifest_path"]),
            "baseline_dev_scorecard_path": str(
                self.preconditions["baseline_dev_scorecard_path"]
            ),
            "bucket_B_manifest_path": str(bucket_b_manifest_path),
            "bucket_C_manifest_path": str(bucket_c_manifest_path),
            "counts": {
                "bucket_B": int(bucket_b_counts["episodes"]),
                "bucket_C": int(bucket_c_counts["episodes"]),
                "bucket_C_per_failure_family": dict(
                    bucket_c_counts["per_failure_family"]
                ),
            },
        }
        summary_path = self.output_dir / BUCKET_COLLECTION_SUMMARY_JSON_NAME
        _write_json(summary_path, summary)
        return summary_path, summary

    def execute(self) -> dict[str, Any]:
        bucket_b_entries = self.collect_bucket("bucket_B")
        bucket_c_entries = self.collect_bucket("bucket_C")
        manifest_builder = BucketManifestBuilder(
            preconditions=self.preconditions,
            output_dir=self.output_dir,
        )
        bucket_b_manifest, bucket_b_counts = manifest_builder.build(
            bucket_key="bucket_B",
            entries=bucket_b_entries,
        )
        bucket_c_manifest, bucket_c_counts = manifest_builder.build(
            bucket_key="bucket_C",
            entries=bucket_c_entries,
        )
        bucket_b_manifest_path = self.output_dir / BUCKET_B_MANIFEST_JSON_NAME
        bucket_c_manifest_path = self.output_dir / BUCKET_C_MANIFEST_JSON_NAME
        _write_json(bucket_b_manifest_path, bucket_b_manifest)
        _write_json(bucket_c_manifest_path, bucket_c_manifest)
        summary_path, _summary = self.build_summary(
            bucket_b_manifest_path=bucket_b_manifest_path,
            bucket_c_manifest_path=bucket_c_manifest_path,
            bucket_b_counts=bucket_b_counts,
            bucket_c_counts=bucket_c_counts,
        )
        return {
            "bucket_B_manifest_path": str(bucket_b_manifest_path),
            "bucket_C_manifest_path": str(bucket_c_manifest_path),
            "bucket_collection_summary_path": str(summary_path),
            "bucket_B_episode_count": int(bucket_b_counts["episodes"]),
            "bucket_C_episode_count": int(bucket_c_counts["episodes"]),
            "bucket_C_per_failure_family": dict(bucket_c_counts["per_failure_family"]),
        }


def materialize_state_conditioned_buckets(
    *,
    bucket_dir: Path,
    dev_dir: Path,
    output_dir: Path,
    collection_runner: CollectionRunner | None = None,
    bucket_b_target: int = DEFAULT_BUCKET_B_TARGET,
    bucket_c_target: int = DEFAULT_BUCKET_C_TARGET,
    bucket_plans: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> dict[str, Any]:
    return StateConditionedBucketCollectionWorkflow(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        output_dir=output_dir,
        collection_runner=collection_runner,
        bucket_b_target=bucket_b_target,
        bucket_c_target=bucket_c_target,
        bucket_plans=bucket_plans,
    ).execute()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_state_conditioned_buckets(
            bucket_dir=args.bucket_dir,
            dev_dir=args.dev_dir,
            output_dir=args.output_dir,
            bucket_b_target=int(args.bucket_b_target),
            bucket_c_target=int(args.bucket_c_target),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BUCKET_B_MANIFEST_JSON_NAME",
    "BUCKET_C_MANIFEST_JSON_NAME",
    "BUCKET_COLLECTION_SUMMARY_JSON_NAME",
    "DEFAULT_BUCKET_B_TARGET",
    "DEFAULT_BUCKET_C_TARGET",
    "INJECTION_METADATA_FIELDS",
    "REQUIRED_FAILURE_FAMILIES",
    "SCHEMA_VERSION",
    "build_default_bucket_plans",
    "build_parser",
    "materialize_state_conditioned_buckets",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())


class StateConditionedCollectBucketsScriptApp:
    def build_parser(self):
        return build_parser()

    def materialize_buckets(self, **kwargs):
        return StateConditionedBucketCollectionWorkflow(**kwargs).execute()

    def run(self, argv=None) -> int:
        return main(argv)
