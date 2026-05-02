#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any, TypedDict, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_RUNTIME_LOG = Path(
    "agent/runtime_logs/policy_modality_probe/00_smoke_eval_g1_once.log"
)
DEFAULT_OUTPUT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/controller_audit_unitree_g1.json"
)

REPORT_SCHEMA_VERSION = "gr00t_controller_audit_unitree_g1_v1"
REPORT_ARTIFACT_KIND = "gr00t_controller_audit_unitree_g1"
FAILURE_NOTE_MARKDOWN_NAME = "controller_audit_unitree_g1_failure_note.md"

EXPECTED_EMBODIMENT_TAG = "UNITREE_G1"
EXPECTED_N_ACTION_STEPS = 20
EXPECTED_SERVER_MODALITY_CONFIG_KEYS = ["action", "language", "state", "video"]
EXPECTED_ACTION_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "waist": 3,
    "base_height_command": 1,
    "navigate_command": 3,
}
EXPECTED_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
}

SUMMARY_ARRAY_RE = re.compile(
    r"(?P<key>[A-Za-z0-9._]+)=ndarray dtype=(?P<dtype>[A-Za-z0-9_<>]+) shape=\((?P<shape>[^)]*)\)"
)


class ArraySummary(TypedDict):
    dtype: str
    shape: list[int]


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ISAAC_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
if str(ISAAC_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_ROOT))


from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_controller_audit_unitree_g1.py",
        description=(
            "Audit UNITREE_G1 official controller / embodiment assumptions by aligning "
            "embodiment config, relative-action processor rules, rollout wrapper, and "
            "runtime rollout-side samples into one machine-readable report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--runtime-log",
        type=Path,
        default=DEFAULT_RUNTIME_LOG,
        help=(
            "Runtime log emitted by an actual rollout-side sample (reset obs + get_action "
            "summary + modality config summary)."
        ),
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Machine-readable UNITREE_G1 audit JSON output path.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f"output must be a file path, got existing directory: {resolved}"
        )
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"output must be a file path: {resolved}")
    if not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _rel_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _parse_shape_tuple(raw: str) -> tuple[int, ...]:
    stripped = raw.strip()
    if not stripped:
        return tuple()
    dims: list[int] = []
    for chunk in stripped.split(","):
        item = chunk.strip()
        if not item:
            continue
        dims.append(int(item))
    return tuple(dims)


def _parse_summary_arrays(summary_line: str) -> dict[str, ArraySummary]:
    parsed: dict[str, ArraySummary] = {}
    for match in SUMMARY_ARRAY_RE.finditer(summary_line):
        key = match.group("key")
        parsed[key] = {
            "dtype": match.group("dtype"),
            "shape": list(_parse_shape_tuple(match.group("shape"))),
        }
    return parsed


def _extract_prefixed_order(keys: list[str], *, prefix: str) -> list[str]:
    extracted: list[str] = []
    for key in keys:
        if key.startswith(prefix):
            extracted.append(key[len(prefix) :])
    return extracted


def _extract_line_value(text: str, prefix: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise ValueError(f"missing log line with prefix: {prefix}")


def _extract_optional_line_value(text: str, prefix: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _literal_list(text: str, *, field_name: str) -> list[str]:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {text!r}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must parse to a list")
    items = cast(list[object], value)
    return [str(item) for item in items]


def _load_unitree_g1_config() -> dict[str, Any]:
    embodiment_configs = importlib.import_module(
        "gr00t.configs.data.embodiment_configs"
    )
    data_types = importlib.import_module("gr00t.data.types")

    modality_configs_by_tag = cast(
        Mapping[str, Mapping[str, Any]], getattr(embodiment_configs, "MODALITY_CONFIGS")
    )
    action_representation = getattr(data_types, "ActionRepresentation")

    modality_configs = modality_configs_by_tag[EXPECTED_EMBODIMENT_TAG.lower()]
    action_modality = modality_configs["action"]
    state_modality = modality_configs["state"]

    action_order_expected = [str(key) for key in action_modality.modality_keys]
    state_order_expected = [str(key) for key in state_modality.modality_keys]
    action_configs = list(action_modality.action_configs or [])

    relative_action_keys: list[str] = []
    absolute_action_keys: list[str] = []
    relative_reference_state_keys: dict[str, str] = {}
    action_representations: dict[str, str] = {}
    for action_key, action_config in zip(action_order_expected, action_configs):
        rep_name = str(action_config.rep.value)
        action_representations[action_key] = rep_name
        if action_config.rep == action_representation.RELATIVE:
            relative_action_keys.append(action_key)
            relative_reference_state_keys[action_key] = str(
                action_config.state_key or action_key
            )
        else:
            absolute_action_keys.append(action_key)

    return {
        "state_order_expected": state_order_expected,
        "state_horizon_expected": len(state_modality.delta_indices),
        "action_order_expected": action_order_expected,
        "policy_horizon_expected": len(action_modality.delta_indices),
        "relative_action_keys": relative_action_keys,
        "absolute_action_keys": absolute_action_keys,
        "relative_reference_state_keys": relative_reference_state_keys,
        "action_representations": action_representations,
    }


def parse_runtime_sample(runtime_log: Path) -> dict[str, Any]:
    text = runtime_log.read_text(encoding="utf-8")
    server_modality_config_keys = _literal_list(
        _extract_line_value(text, "SERVER modality_config keys:"),
        field_name="SERVER modality_config keys",
    )
    server_action_keys = _literal_list(
        _extract_line_value(text, "SERVER action_keys:"),
        field_name="SERVER action_keys",
    )
    server_action_horizon = int(_extract_line_value(text, "SERVER action_horizon:"))
    reset_obs_keys = _literal_list(
        _extract_line_value(text, "RESET obs keys:"),
        field_name="RESET obs keys",
    )
    reset_obs_summary = _parse_summary_arrays(
        _extract_line_value(text, "RESET obs summary:")
    )
    action_keys_line = _literal_list(
        _extract_line_value(text, "ACTION[0] keys:"),
        field_name="ACTION[0] keys",
    )
    action_summary = _parse_summary_arrays(
        _extract_line_value(text, "ACTION[0] summary:")
    )

    reset_info_keys = _extract_optional_line_value(text, "RESET info keys:")
    action_info_keys = _extract_optional_line_value(text, "ACTION[0] info keys:")

    raw_state_keys = _extract_prefixed_order(reset_obs_keys, prefix="state.")
    raw_action_keys = _extract_prefixed_order(action_keys_line, prefix="action.")
    state_summaries = {
        key[len("state.") :]: value
        for key, value in reset_obs_summary.items()
        if key.startswith("state.")
    }
    action_summaries = {
        key[len("action.") :]: value
        for key, value in action_summary.items()
        if key.startswith("action.")
    }

    state_dims_runtime = {
        key: int(value["shape"][2])
        for key, value in state_summaries.items()
        if len(value["shape"]) == 3
    }
    state_horizons_runtime = {
        key: int(value["shape"][1])
        for key, value in state_summaries.items()
        if len(value["shape"]) == 3
    }
    action_dims_runtime = {
        key: int(value["shape"][2])
        for key, value in action_summaries.items()
        if len(value["shape"]) == 3
    }
    action_horizons_runtime = {
        key: int(value["shape"][1])
        for key, value in action_summaries.items()
        if len(value["shape"]) == 3
    }

    runtime_sim_policy_wrapper_detected = bool(action_keys_line) and all(
        key.startswith("action.") for key in action_keys_line
    )

    return {
        "runtime_log": _rel_repo(runtime_log),
        "server_modality_config_keys": server_modality_config_keys,
        "server_action_keys": server_action_keys,
        "server_action_horizon": server_action_horizon,
        "reset_obs_keys": reset_obs_keys,
        "reset_info_keys": _literal_list(reset_info_keys, field_name="RESET info keys")
        if reset_info_keys is not None
        else [],
        "action_info_keys": _literal_list(
            action_info_keys, field_name="ACTION[0] info keys"
        )
        if action_info_keys is not None
        else [],
        "reset_state_keys_raw": raw_state_keys,
        "action_keys_raw": raw_action_keys,
        "reset_state_summaries": state_summaries,
        "action_summaries": action_summaries,
        "state_dims_runtime": state_dims_runtime,
        "state_horizons_runtime": state_horizons_runtime,
        "action_dims_runtime": action_dims_runtime,
        "action_horizons_runtime": action_horizons_runtime,
        "sim_policy_wrapper_detected": runtime_sim_policy_wrapper_detected,
    }


def build_controller_provenance() -> dict[str, object]:
    return {
        "embodiment_tag": EXPECTED_EMBODIMENT_TAG,
        "official_env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
        "rollout_env_factory": {
            "function": "gr00t.eval.rollout_policy.get_groot_locomanip_env_fn",
            "wrapper_class": "gr00t_wbc.control.utils.n1_utils.WholeBodyControlWrapper",
            "wbc_config_ctor": "gr00t_wbc.control.main.teleop.configs.configs.BaseConfig(wbc_version='gear_wbc', enable_waist=True).to_dict()",
            "source_ref": "submodules/Isaac-GR00T/gr00t/eval/rollout_policy.py:95-117",
        },
        "policy_wrapper": {
            "class": "gr00t.policy.gr00t_policy.Gr00tSimPolicyWrapper",
            "flat_observation_prefixes": ["video.", "state."],
            "flat_action_prefix": "action.",
            "source_ref": "submodules/Isaac-GR00T/gr00t/policy/gr00t_policy.py:420-673",
        },
        "wbc_controller": {
            "wbc_version": "gear_wbc",
            "wbc_policy_class": "G1DecoupledWholeBodyPolicy",
            "onnx_assets": [
                "policy/GR00T-WholeBodyControl-Balance.onnx",
                "policy/GR00T-WholeBodyControl-Walk.onnx",
            ],
            "action_space_source_ref": "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/utils/n1_utils.py:81-124",
            "config_source_ref": "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/main/teleop/configs/configs.py:71-130",
        },
    }


def build_timebase(policy_horizon_expected: int) -> dict[str, object]:
    return {
        "sim_frequency_hz": 200,
        "sim_dt_s": 0.005,
        "control_frequency_hz": 50,
        "control_dt_s": 0.02,
        "sim_steps_per_control_step": 4,
        "policy_horizon_expected": int(policy_horizon_expected),
        "n_action_steps_expected": EXPECTED_N_ACTION_STEPS,
        "horizon_semantics": {
            "policy_horizon": "server get_action() chunk length",
            "execution_horizon": "rollout client sequentially executes the first n_action_steps entries from the returned chunk",
            "must_not_be_conflated": True,
        },
        "source_refs": [
            "submodules/Isaac-GR00T/examples/GR00T-WholeBodyControl/README.md:76-83",
            "submodules/Isaac-GR00T/gr00t/eval/rollout_policy.py:225-232",
            "submodules/Isaac-GR00T/gr00t/eval/sim/wrapper/multistep_wrapper.py:249-279",
            "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml:24-26",
            "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/main/teleop/configs/configs.py:33-45,98-103",
        ],
    }


def build_execution_surface_contract(expected: Mapping[str, Any]) -> dict[str, object]:
    return {
        "policy_horizon_expected": int(expected["policy_horizon_expected"]),
        "n_action_steps_expected": EXPECTED_N_ACTION_STEPS,
        "relative_action_keys": list(expected["relative_action_keys"]),
        "absolute_action_keys": list(expected["absolute_action_keys"]),
        "action_representation_by_key": dict(expected["action_representations"]),
        "relative_to_absolute_rule": {
            "enabled_for_relative_action_keys": True,
            "reference_state_timestep": "last",
            "reference_state_keys": dict(expected["relative_reference_state_keys"]),
        },
        "must_not_conflate_horizon_and_execution": True,
        "repo_local_formalization": {
            "field_names_are_repo_local": True,
            "upstream_policy_horizon_authority": "action.delta_indices",
            "upstream_execution_steps_authority": "rollout --n_action_steps",
            "note": (
                "This report freezes repo-local contract field names for comparability; "
                "they summarize upstream semantics but are not upstream official JSON field names."
            ),
        },
    }


def _mismatch_field(
    mismatches: list[str], *, name: str, expected: object, actual: object
) -> None:
    if expected != actual:
        mismatches.append(name)


def build_audit_report(*, runtime_log: Path) -> dict[str, Any]:
    expected = _load_unitree_g1_config()
    runtime = parse_runtime_sample(runtime_log)

    expected_state_order = list(expected["state_order_expected"])
    expected_action_order = list(expected["action_order_expected"])
    expected_policy_horizon = int(expected["policy_horizon_expected"])

    observed_state_key_set = set(runtime["reset_state_keys_raw"])
    state_order_runtime = [
        key for key in expected_state_order if key in observed_state_key_set
    ]
    state_order_missing = [
        key for key in expected_state_order if key not in observed_state_key_set
    ]
    state_order_extra = [
        key
        for key in runtime["reset_state_keys_raw"]
        if key not in expected_state_order
    ]

    action_order_runtime = list(runtime["server_action_keys"])
    action_keys_missing = [
        key for key in expected_action_order if key not in set(action_order_runtime)
    ]
    action_keys_extra = [
        key for key in action_order_runtime if key not in expected_action_order
    ]

    state_horizon_runtime = sorted(
        {int(value) for value in runtime["state_horizons_runtime"].values()}
    )
    action_horizon_runtime = sorted(
        {int(value) for value in runtime["action_horizons_runtime"].values()}
    )
    policy_horizon_runtime = int(runtime["server_action_horizon"])

    mismatches: list[str] = []
    _mismatch_field(
        mismatches,
        name="server_modality_config_keys",
        expected=EXPECTED_SERVER_MODALITY_CONFIG_KEYS,
        actual=runtime["server_modality_config_keys"],
    )
    _mismatch_field(
        mismatches,
        name="state_order_runtime",
        expected=expected_state_order,
        actual=state_order_runtime,
    )
    if state_order_missing:
        mismatches.append("state_order_missing")
    if state_order_extra:
        mismatches.append("state_order_extra")
    _mismatch_field(
        mismatches,
        name="action_order_runtime",
        expected=expected_action_order,
        actual=action_order_runtime,
    )
    if action_keys_missing:
        mismatches.append("action_keys_missing")
    if action_keys_extra:
        mismatches.append("action_keys_extra")
    _mismatch_field(
        mismatches,
        name="state_dims_runtime",
        expected=EXPECTED_STATE_DIMS,
        actual=runtime["state_dims_runtime"],
    )
    _mismatch_field(
        mismatches,
        name="action_dims_runtime",
        expected=EXPECTED_ACTION_DIMS,
        actual=runtime["action_dims_runtime"],
    )
    _mismatch_field(
        mismatches,
        name="policy_horizon_runtime",
        expected=expected_policy_horizon,
        actual=policy_horizon_runtime,
    )
    _mismatch_field(
        mismatches,
        name="action_chunk_horizon_runtime",
        expected=[expected_policy_horizon],
        actual=action_horizon_runtime,
    )
    _mismatch_field(
        mismatches,
        name="state_horizon_runtime",
        expected=[int(expected["state_horizon_expected"])],
        actual=state_horizon_runtime,
    )
    if not bool(runtime["sim_policy_wrapper_detected"]):
        mismatches.append("sim_policy_wrapper_detected")

    mismatch_fields = sorted(set(mismatches))
    equivalent = not mismatch_fields

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "embodiment_tag": EXPECTED_EMBODIMENT_TAG,
        "state_order_expected": expected_state_order,
        "state_order_runtime": state_order_runtime,
        "action_order_expected": expected_action_order,
        "action_order_runtime": action_order_runtime,
        "state_dims_expected": EXPECTED_STATE_DIMS,
        "state_dims_runtime": runtime["state_dims_runtime"],
        "action_dims_expected": EXPECTED_ACTION_DIMS,
        "action_dims_runtime": runtime["action_dims_runtime"],
        "relative_action_keys": expected["relative_action_keys"],
        "absolute_action_keys": expected["absolute_action_keys"],
        "action_representation_by_key": expected["action_representations"],
        "relative_to_absolute_processor": {
            "enabled_when_use_relative_action": True,
            "reference_state_timestep": "last",
            "reference_state_keys": expected["relative_reference_state_keys"],
            "source_ref": "submodules/Isaac-GR00T/gr00t/data/state_action/state_action_processor.py:434-487",
        },
        "policy_horizon_expected": expected_policy_horizon,
        "policy_horizon_runtime": policy_horizon_runtime,
        "n_action_steps_expected": EXPECTED_N_ACTION_STEPS,
        "execution_surface_contract": build_execution_surface_contract(expected),
        "timebase": build_timebase(expected_policy_horizon),
        "controller_provenance": build_controller_provenance(),
        "runtime_sampling": {
            "source": "actual_rollout_side_runtime_log",
            "runtime_log": runtime["runtime_log"],
            "server_modality_config_keys": runtime["server_modality_config_keys"],
            "server_action_keys": runtime["server_action_keys"],
            "reset_obs_keys": runtime["reset_obs_keys"],
            "reset_state_keys_raw": runtime["reset_state_keys_raw"],
            "action_keys_raw": runtime["action_keys_raw"],
            "reset_info_keys": runtime["reset_info_keys"],
            "action_info_keys": runtime["action_info_keys"],
            "state_horizons_runtime": runtime["state_horizons_runtime"],
            "action_horizons_runtime": runtime["action_horizons_runtime"],
            "sim_policy_wrapper_detected": runtime["sim_policy_wrapper_detected"],
        },
        "runtime_drift_details": {
            "state_order_missing": state_order_missing,
            "state_order_extra": state_order_extra,
            "action_keys_missing": action_keys_missing,
            "action_keys_extra": action_keys_extra,
            "state_horizon_runtime_unique": state_horizon_runtime,
            "action_horizon_runtime_unique": action_horizon_runtime,
        },
        "equivalent_to_official_unitree_g1": equivalent,
        "mismatch_fields": mismatch_fields,
        "source_refs": {
            "embodiment_config": "submodules/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py:13-90",
            "processor": "submodules/Isaac-GR00T/gr00t/data/state_action/state_action_processor.py:434-487",
            "rollout_env": "submodules/Isaac-GR00T/gr00t/eval/rollout_policy.py:95-117,188-233",
            "policy_io_contract": "agent/exchange/gr00t_policy_io.md:86-157",
            "wbc_env_contract": "agent/exchange/wbc_env_io.md:37-83",
            "runtime_probe_pattern": "agent/run/prompt_sensitivity_probe_g1.py:242-253,303-349",
        },
    }
    return report


def build_failure_note(report: Mapping[str, Any], *, output_path: Path) -> str:
    mismatch_fields = list(report.get("mismatch_fields", []))
    runtime_sampling = dict(report.get("runtime_sampling", {}))
    runtime_drift = dict(report.get("runtime_drift_details", {}))
    lines = [
        "# UNITREE_G1 controller audit failure note",
        "",
        f"- output: `{_rel_repo(output_path)}`",
        f"- runtime log: `{runtime_sampling.get('runtime_log', 'unknown')}`",
        f"- equivalent_to_official_unitree_g1: `{report.get('equivalent_to_official_unitree_g1')}`",
        f"- mismatch_fields: `{json.dumps(mismatch_fields, ensure_ascii=True)}`",
        "",
        "## Runtime drift details",
        "",
        "```json",
        json.dumps(runtime_drift, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        runtime_log = _validate_existing_file(
            cast(Path, args.runtime_log),
            arg_name="runtime-log",
        )
        output_path = _validate_output_path(cast(Path, args.output))

        report = build_audit_report(runtime_log=runtime_log)
        _ = _write_json(output_path, report)

        failure_note_path = output_path.with_name(FAILURE_NOTE_MARKDOWN_NAME)
        if not bool(report["equivalent_to_official_unitree_g1"]):
            _ = _write_text(
                failure_note_path,
                build_failure_note(report, output_path=output_path),
            )

        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
