#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any, NoReturn


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_CONTRACT_PATH = Path(
    "agent/exchange/state_structured_recap_analysis_only_schema_v1.md"
)

SPEC_START_MARKER = "<!-- STATE_STRUCTURED_RECAP_ANALYSIS_SCHEMA_SPEC_START -->"
SPEC_END_MARKER = "<!-- STATE_STRUCTURED_RECAP_ANALYSIS_SCHEMA_SPEC_END -->"
PASS_SENTINEL = "STATE_STRUCTURED_RECAP_ANALYSIS_ONLY_PASS"
FAIL_SENTINEL = "STATE_STRUCTURED_RECAP_ANALYSIS_ONLY_FAIL"
EXPECTED_SCHEMA_VERSION = "state_structured_recap_analysis_only_v1"
EXPECTED_CONTAINER_PATH = "analysis_only.state_structured_recap"
EXPECTED_LEAF_FIELDS: tuple[str, ...] = (
    "edge_id",
    "candidate_edge_mask",
    "recovery_family",
    "semantic_commit",
    "failed_edge_to_recovery_edge",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_build_training_set
from work.recap import state_conditioned_contract_gate
from work.recap.lerobot_export import dataset_export as lerobot_v2_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the analysis-only State-Structured RECAP schema document and "
            "prove that its fields are rejected by current deployable/train gates."
        )
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Markdown contract file that contains the embedded JSON spec block.",
    )
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="Print the parsed contract spec JSON after validation.",
    )
    return parser


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    print(FAIL_SENTINEL, file=sys.stderr)
    raise SystemExit(1)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"failed to read contract file {path}: {exc}")


def _extract_spec_json(markdown_text: str) -> dict[str, Any]:
    pattern = re.compile(
        re.escape(SPEC_START_MARKER)
        + r"\s*```json\s*(\{.*?\})\s*```\s*"
        + re.escape(SPEC_END_MARKER),
        re.DOTALL,
    )
    match = pattern.search(markdown_text)
    if match is None:
        _fail("contract markdown is missing the embedded JSON spec block")
    raw_json = match.group(1)
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        _fail(f"embedded contract spec JSON is invalid: {exc}")
    if not isinstance(payload, dict):
        _fail("embedded contract spec must be a JSON object")
    return dict(payload)


def _require_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be an object, got {type(value).__name__}")
    return dict(value)


def _require_list(
    value: Any, *, field_name: str, expected_len: int | None = None
) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field_name} must be a list, got {type(value).__name__}")
    result = list(value)
    if expected_len is not None and len(result) != int(expected_len):
        _fail(f"{field_name} must have length {expected_len}, got {len(result)}")
    return result


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        _fail(f"{field_name} must be a non-empty string")
    return normalized


def _require_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_string_list(value: Any, *, field_name: str) -> list[str]:
    items = _require_list(value, field_name=field_name)
    return [
        _require_non_empty_string(item, field_name=f"{field_name}[]") for item in items
    ]


def _check_markdown_coverage(markdown_text: str, spec: Mapping[str, Any]) -> None:
    required_phrases = (
        "analysis-only",
        "不进入当前 deployable/train gate",
        "state_conditioned_bucket_a_sidecar.py",
        "state_conditioned_collect_buckets.py",
        "failed_edge -> recovery_edge",
        "explicit non-goals",
        "sidecar / backfill",
    )
    for phrase in required_phrases:
        if phrase not in markdown_text:
            _fail(f"contract markdown is missing required phrase: {phrase!r}")

    for field_name in EXPECTED_LEAF_FIELDS:
        if f"`{field_name}`" not in markdown_text:
            _fail(
                f"contract markdown does not mention field {field_name!r} in backticks"
            )

    alias_map = _require_dict(spec.get("display_aliases"), field_name="display_aliases")
    if alias_map.get("failed_edge_to_recovery_edge") != "failed_edge -> recovery_edge":
        _fail("display_aliases.failed_edge_to_recovery_edge must stay frozen")


def load_contract_spec(
    contract_path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    markdown_text = _read_text(contract_path)
    spec_json = _extract_spec_json(markdown_text)
    _check_markdown_coverage(markdown_text, spec_json)
    return markdown_text, spec_json, _validate_contract_spec(spec_json)


def _validate_contract_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = _require_non_empty_string(
        spec.get("schema_version"), field_name="schema_version"
    )
    if schema_version != EXPECTED_SCHEMA_VERSION:
        _fail(f"schema_version mismatch: {schema_version!r}")
    if not _require_bool(spec.get("analysis_only"), field_name="analysis_only"):
        _fail("analysis_only must be true")

    container_path = _require_non_empty_string(
        spec.get("container_path"), field_name="container_path"
    )
    if container_path != EXPECTED_CONTAINER_PATH:
        _fail(f"container_path mismatch: {container_path!r}")

    source_inputs = _as_string_list(
        spec.get("source_inputs"), field_name="source_inputs"
    )
    if source_inputs != [
        "work/recap/scripts/state_conditioned_bucket_a_sidecar.py",
        "work/recap/scripts/state_conditioned_collect_buckets.py",
    ]:
        _fail("source_inputs must stay frozen to the two required reference scripts")

    leaf_field_order = _as_string_list(
        spec.get("leaf_field_order"), field_name="leaf_field_order"
    )
    if tuple(leaf_field_order) != EXPECTED_LEAF_FIELDS:
        _fail(f"leaf_field_order mismatch: {leaf_field_order!r}")

    fields = _require_dict(spec.get("fields"), field_name="fields")
    if tuple(fields.keys()) != EXPECTED_LEAF_FIELDS:
        _fail("fields keys must match leaf_field_order exactly")

    for field_name in EXPECTED_LEAF_FIELDS:
        field_spec = _require_dict(
            fields.get(field_name), field_name=f"fields.{field_name}"
        )
        _require_non_empty_string(
            field_spec.get("description"),
            field_name=f"fields.{field_name}.description",
        )
        _as_string_list(
            field_spec.get("source_notes"),
            field_name=f"fields.{field_name}.source_notes",
        )
        _require_non_empty_string(
            field_spec.get("future_use"), field_name=f"fields.{field_name}.future_use"
        )
        _require_non_empty_string(
            field_spec.get("non_goal"), field_name=f"fields.{field_name}.non_goal"
        )

    integration_notes = _as_string_list(
        spec.get("integration_notes"), field_name="integration_notes"
    )
    if len(integration_notes) < 4:
        _fail("integration_notes must contain at least 4 entries")

    explicit_non_goals = _as_string_list(
        spec.get("explicit_non_goals"), field_name="explicit_non_goals"
    )
    if len(explicit_non_goals) < 5:
        _fail("explicit_non_goals must contain at least 5 entries")

    current_context = _require_dict(
        spec.get("current_context"), field_name="current_context"
    )
    if not bool(
        current_context.get(
            "analysis_only_not_in_current_deployable_or_train_gate", False
        )
    ):
        _fail(
            "current_context.analysis_only_not_in_current_deployable_or_train_gate must be true"
        )

    gate_constraints = _require_dict(
        spec.get("current_gate_constraints"), field_name="current_gate_constraints"
    )
    if not bool(
        gate_constraints.get(
            "must_reject_if_mixed_into_current_deployable_or_train_payload", False
        )
    ):
        _fail(
            "current_gate_constraints.must_reject_if_mixed_into_current_deployable_or_train_payload must be true"
        )

    return {
        "schema_version": schema_version,
        "container_path": container_path,
        "leaf_field_order": leaf_field_order,
        "fields": fields,
        "integration_notes": integration_notes,
        "explicit_non_goals": explicit_non_goals,
    }


def validate_train_payload_field_names(
    field_names: Sequence[str], *, spec: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = [
        _require_non_empty_string(name, field_name="train_payload.field")
        for name in field_names
    ]
    allowed = set(state_conditioned_build_training_set.SAFE_LABEL_FIELD_ORDER)
    leaf_fields = set(spec["leaf_field_order"])
    container_path = str(spec["container_path"])
    extras = sorted(set(normalized) - allowed)
    if extras:
        leaks = [
            field_name
            for field_name in extras
            if field_name in leaf_fields
            or field_name == container_path
            or field_name.startswith(container_path + ".")
        ]
        if leaks:
            raise ValueError(
                "analysis-only field leaked into train payload: "
                + ", ".join(sorted(leaks))
            )
        raise ValueError("train payload field not allowlisted: " + ", ".join(extras))
    return {
        "status": "PASS",
        "field_count": int(len(normalized)),
    }


def run_leak_negative_checks(spec: Mapping[str, Any]) -> dict[str, Any]:
    leaf_fields = list(spec["leaf_field_order"])
    deployable_contract_gate: list[dict[str, str]] = []
    deployable_export_gate: list[dict[str, str]] = []
    train_payload_gate: list[dict[str, str]] = []

    base_candidate = copy.deepcopy(
        state_conditioned_contract_gate.build_reference_contract_example()
    )
    base_train_fields = list(
        state_conditioned_build_training_set.SAFE_LABEL_FIELD_ORDER
    )

    for field_name in leaf_fields:
        if field_name in set(
            state_conditioned_contract_gate.DEPLOYABLE_OBSERVATION_ALLOWLIST
        ):
            _fail(
                f"analysis-only field unexpectedly present in deployable allowlist: {field_name}"
            )
        if field_name in set(lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES):
            _fail(
                f"analysis-only field unexpectedly present in deployable_history allowlist: {field_name}"
            )
        if field_name in set(
            state_conditioned_build_training_set.SAFE_LABEL_FIELD_ORDER
        ):
            _fail(
                f"analysis-only field unexpectedly present in train payload allowlist: {field_name}"
            )

        candidate = copy.deepcopy(base_candidate)
        deployable_observation = _require_dict(
            candidate.get("deployable_observation"),
            field_name="candidate.deployable_observation",
        )
        deployable_observation[field_name] = {"probe": field_name}
        candidate["deployable_observation"] = deployable_observation
        try:
            state_conditioned_contract_gate.validate_contract_candidate(candidate)
        except ValueError as exc:
            deployable_contract_gate.append(
                {
                    "field": field_name,
                    "status": "REJECTED",
                    "message": str(exc),
                }
            )
        else:
            _fail(
                f"deployable contract gate accepted analysis-only field {field_name!r}"
            )

        field_groups = lerobot_v2_export.build_state_conditioned_field_groups()
        field_groups[lerobot_v2_export.DEPLOYABLE_HISTORY_GROUP_KEY].append(field_name)
        try:
            lerobot_v2_export.validate_state_conditioned_field_groups(field_groups)
        except ValueError as exc:
            deployable_export_gate.append(
                {
                    "field": field_name,
                    "status": "REJECTED",
                    "message": str(exc),
                }
            )
        else:
            _fail(f"deployable export gate accepted analysis-only field {field_name!r}")

        try:
            validate_train_payload_field_names(
                [*base_train_fields, field_name],
                spec=spec,
            )
        except ValueError as exc:
            train_payload_gate.append(
                {
                    "field": field_name,
                    "status": "REJECTED",
                    "message": str(exc),
                }
            )
        else:
            _fail(f"train payload gate accepted analysis-only field {field_name!r}")

    dotted_probe = str(spec["container_path"]) + ".edge_id"
    try:
        validate_train_payload_field_names(
            [*base_train_fields, dotted_probe], spec=spec
        )
    except ValueError as exc:
        train_container_probe = {
            "field": dotted_probe,
            "status": "REJECTED",
            "message": str(exc),
        }
    else:
        _fail(
            f"train payload gate accepted analysis-only container probe {dotted_probe!r}"
        )

    return {
        "status": "PASS",
        "field_count": int(len(leaf_fields)),
        "current_allowlists": {
            "deployable_observation_allowlist_size": int(
                len(state_conditioned_contract_gate.DEPLOYABLE_OBSERVATION_ALLOWLIST)
            ),
            "deployable_history_allowlist_size": int(
                len(lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES)
            ),
            "train_payload_allowlist_size": int(
                len(state_conditioned_build_training_set.SAFE_LABEL_FIELD_ORDER)
            ),
        },
        "deployable_contract_gate": deployable_contract_gate,
        "deployable_export_gate": deployable_export_gate,
        "train_payload_gate": train_payload_gate,
        "train_container_probe": train_container_probe,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _markdown_text, spec_json, spec = load_contract_spec(Path(args.contract))
    leak_negative = run_leak_negative_checks(spec)

    result: dict[str, Any] = {
        "status": "PASS",
        "schema_version": str(spec["schema_version"]),
        "field_count": len(spec["leaf_field_order"]),
        "leak_negative": leak_negative,
    }
    if args.print_spec:
        result["spec"] = spec_json

    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    print(PASS_SENTINEL)
    return 0


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "EXPECTED_CONTAINER_PATH",
    "EXPECTED_LEAF_FIELDS",
    "FAIL_SENTINEL",
    "PASS_SENTINEL",
    "build_parser",
    "load_contract_spec",
    "run_leak_negative_checks",
    "validate_train_payload_field_names",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
