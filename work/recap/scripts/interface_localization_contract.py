from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "interface_localization_sprint"
DEFAULT_CONTRACT_DOC_PATH = "agent/exchange/interface_localization_contract.md"

BASELINE_JSON_NAME = "baseline_tuple.json"
CONTRACT_JSON_NAME = "interface_localization_contract.json"

BASELINE_SCHEMA_VERSION = "interface_localization_baseline_tuple_v1"
CONTRACT_SCHEMA_VERSION = "interface_localization_contract_v1"
BASELINE_ARTIFACT_KIND = "interface_localization_baseline_tuple"
CONTRACT_ARTIFACT_KIND = "interface_localization_contract"

CONTRACT_SPEC_START_MARKER = "<!-- INTERFACE_LOCALIZATION_CONTRACT_SPEC_START -->"
CONTRACT_SPEC_END_MARKER = "<!-- INTERFACE_LOCALIZATION_CONTRACT_SPEC_END -->"

DEFERRED_PLACEHOLDER_PREFIX = "TASK1_SELECTION_DEFERRED__"

BASELINE_TUPLE_FIELD_ORDER: tuple[str, ...] = (
    "embodiment",
    "benchmark_env_name",
    "benchmark_task",
    "simulator",
    "controller_stack",
    "scene_motion_slice_identifier",
    "checkpoint_identifier",
    "seed_set_identifier",
    "condition_pair_identifier",
    "server_mode_identifier",
    "serving_path_axis_identifier",
    "task_text_surface_identifier",
    "replay_init_source_identifier",
    "right_arm_boundary_name",
    "right_hand_boundary_name",
)

BOUNDARY_ORDER: tuple[str, ...] = (
    "prompt_raw_source",
    "prompt_conditioned_write",
    "export_task_text_selection",
    "collector_policy_callsite",
    "server_policy_adapter",
    "policy_input_collation",
    "model_condition_injection",
    "policy_output_action",
    "action_semantics_adapter",
    "body_wrist_upper_limb_chain",
    "dex3_finger_hand_path",
)

STATUS_ORDER: tuple[str, ...] = (
    "survived",
    "died",
    "mutated",
    "rerouted",
    "bypassed",
    "blocked_missing_upstream",
)

PROVENANCE_CLASS_ORDER: tuple[str, ...] = (
    "static",
    "synthetic",
    "replay_live",
    "server_live",
)

FUTURE_ARTIFACT_MINIMUM_FIELDS: tuple[str, ...] = (
    "provenance_class",
    "generation_command",
    "input_baseline_summary",
    "backpointer",
)

PROTECTED_FIELD_PATHS: tuple[str, ...] = (
    "baseline_tuple_field_order",
    "baseline_tuple",
    "boundary_ontology.boundary_order",
    "boundary_ontology.boundaries",
    "boundary_ontology.never_merge_pairs",
    "status_ontology.status_order",
    "status_ontology.legal_statuses",
    "status_ontology.status_definitions",
    "status_ontology.blocked_but_proven_is_success",
    "provenance_classes.class_order",
    "provenance_classes.classes",
    "artifact_contract.output_root",
    "artifact_contract.required_files",
    "artifact_contract.future_artifact_minimum_fields",
    "artifact_contract.provenance_class_allowlist",
    "advantage_contract_facts",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap.advantage import ADVANTAGE_CONTRACT_VERSION
from work.recap.advantage import ADVANTAGE_INPUT_CLIP_RANGE
from work.recap.advantage import ADVANTAGE_INPUT_COLUMN
from work.recap.advantage import ADVANTAGE_RAW_COLUMN
from work.recap.advantage import ADVANTAGE_RETURN_COLUMN
from work.recap.advantage import ADVANTAGE_SCALE_EPS
from work.recap.advantage import ADVANTAGE_SCALE_QUANTILE
from work.recap.advantage import ADVANTAGE_SCALE_RULE
from work.recap.advantage import ADVANTAGE_VALUE_COLUMN
from work.recap.advantage import LEGACY_ADVANTAGE_CONTRACT_VERSION
from work.recap.advantage import MAINLINE_TASK_TEXT_FIELD


MISSING = object()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface_localization_contract.py",
        description=(
            "Freeze the Task 1 interface localization baseline tuple, ontology, "
            "and artifact schema into deterministic JSON artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, artifacts are written to "
            "<artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for generated JSON artifacts.",
    )
    _ = parser.add_argument(
        "--contract-doc",
        type=str,
        default=DEFAULT_CONTRACT_DOC_PATH,
        help="Markdown contract file that contains the embedded machine-readable freeze block.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "agent").is_dir() and (parent / "work").is_dir():
            return parent
    return Path.cwd().resolve()


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _deep_get(payload: Mapping[str, Any], field_path: str) -> object:
    current: object = payload
    for key in field_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return MISSING
        current = current[key]
    return current


def _value_for_report(value: object) -> object:
    if value is MISSING:
        return {"missing": True}
    return value


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _as_string_list(value: object, *, field_name: str) -> list[str]:
    items = _as_list(value, field_name=field_name)
    return [_as_non_empty_string(item, field_name=f"{field_name}[]") for item in items]


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _require_exact_keyset(
    mapping: Mapping[str, Any],
    *,
    expected_keys: Sequence[str],
    field_name: str,
) -> None:
    actual_keys = set(mapping.keys())
    expected_key_set = set(str(key) for key in expected_keys)
    if actual_keys != expected_key_set:
        missing = sorted(expected_key_set - actual_keys)
        extra = sorted(actual_keys - expected_key_set)
        raise ValueError(
            f"{field_name} keys mismatch: missing={missing!r}, extra={extra!r}"
        )


def _deferred(label: str) -> str:
    return f"{DEFERRED_PLACEHOLDER_PREFIX}{label}__repo_local_placeholder_v1"


def build_baseline_tuple() -> dict[str, Any]:
    return {
        "embodiment": "UNITREE_G1",
        "benchmark_env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
        "benchmark_task": "LMPnPAppleToPlate",
        "simulator": "MuJoCo",
        "controller_stack": "WholeBodyControl",
        "scene_motion_slice_identifier": _deferred("scene_motion_slice_identifier"),
        "checkpoint_identifier": _deferred("checkpoint_identifier"),
        "seed_set_identifier": _deferred("seed_set_identifier"),
        "condition_pair_identifier": _deferred("condition_pair_identifier"),
        "server_mode_identifier": "replay_sim_only",
        "serving_path_axis_identifier": "stock_serving_path_vs_custom_advantage_aware_path",
        "task_text_surface_identifier": "prompt_raw_vs_prompt_conditioned_vs_runtime_override",
        "replay_init_source_identifier": _deferred("replay_init_source_identifier"),
        "right_arm_boundary_name": "body_wrist_upper_limb_chain",
        "right_hand_boundary_name": "dex3_finger_hand_path",
    }


def build_baseline_tuple_artifact() -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "artifact_kind": BASELINE_ARTIFACT_KIND,
        "baseline_tuple_field_order": list(BASELINE_TUPLE_FIELD_ORDER),
        "baseline_tuple": build_baseline_tuple(),
    }


def _build_boundary_specs() -> dict[str, dict[str, Any]]:
    return {
        "prompt_raw_source": {
            "boundary_kind": "text_source",
            "lane": "text",
            "scope_summary": "Frozen source boundary for raw task text before RECAP conditioning or runtime override.",
            "blocked_rule": "If upstream source selection cannot be located, mark blocked_missing_upstream rather than guessing the raw text source.",
        },
        "prompt_conditioned_write": {
            "boundary_kind": "text_rewrite",
            "lane": "text",
            "scope_summary": "Boundary where conditioned task text is rewritten from raw task text and RECAP signal.",
            "blocked_rule": "If conditioned-text rewrite logic is absent or upstream-only, mark blocked_missing_upstream.",
        },
        "export_task_text_selection": {
            "boundary_kind": "text_export_selection",
            "lane": "text",
            "scope_summary": "Boundary that chooses which task-text field is exported into downstream artifacts or runtime payloads.",
            "blocked_rule": "If exporter text selection cannot be proven in repo or replay evidence, mark blocked_missing_upstream.",
        },
        "collector_policy_callsite": {
            "boundary_kind": "collector_callsite",
            "lane": "serving",
            "scope_summary": "Collector-side boundary that invokes policy calls with the chosen condition and task-text payload.",
            "blocked_rule": "If the collector callsite is only described indirectly and not inspectable, mark blocked_missing_upstream.",
        },
        "server_policy_adapter": {
            "boundary_kind": "server_adapter",
            "lane": "serving",
            "scope_summary": "Server-side adapter boundary that maps incoming request fields into the policy runtime contract.",
            "blocked_rule": "If the serving adapter lives only in missing upstream code, mark blocked_missing_upstream.",
        },
        "policy_input_collation": {
            "boundary_kind": "policy_input_collation",
            "lane": "serving",
            "scope_summary": "Boundary that collates observation and condition lanes into the model input payload.",
            "blocked_rule": "If the exact input collation logic cannot be observed or replayed, mark blocked_missing_upstream.",
        },
        "model_condition_injection": {
            "boundary_kind": "model_condition_injection",
            "lane": "numeric_text_conditioning",
            "scope_summary": "Boundary where the model consumes numeric/text condition signals after serving-path collation.",
            "blocked_rule": "If injection happens in opaque upstream code with no local proof surface, mark blocked_missing_upstream.",
        },
        "policy_output_action": {
            "boundary_kind": "policy_output",
            "lane": "action",
            "scope_summary": "Boundary where the policy emits raw action tensors or decoded action chunks.",
            "blocked_rule": "If raw action output cannot be observed due to missing upstream runtime hooks, mark blocked_missing_upstream.",
        },
        "action_semantics_adapter": {
            "boundary_kind": "action_adapter",
            "lane": "action",
            "scope_summary": "Boundary that maps policy output into canonical action semantics and controller-facing units.",
            "blocked_rule": "If the semantic adapter cannot be reconstructed from repo or replay evidence, mark blocked_missing_upstream.",
        },
        "body_wrist_upper_limb_chain": {
            "boundary_kind": "body_chain",
            "lane": "body",
            "body_side": "right",
            "distinct_from": "dex3_finger_hand_path",
            "scope_summary": "Right-side body, wrist, and upper-limb chain semantics up to the wrist boundary.",
            "blocked_rule": "If upstream only says right hand but does not disambiguate wrist versus fingers, mark blocked_missing_upstream instead of collapsing the split.",
        },
        "dex3_finger_hand_path": {
            "boundary_kind": "dex3_hand_path",
            "lane": "hand",
            "body_side": "right",
            "distinct_from": "body_wrist_upper_limb_chain",
            "scope_summary": "Right-side Dex3 palm, finger, fingertip, and hand actuation semantics below the wrist boundary.",
            "blocked_rule": "If upstream provides no explicit Dex3 or finger-level path evidence, use blocked_missing_upstream rather than inferring a synthetic finger path.",
        },
    }


def _build_status_definitions() -> dict[str, str]:
    return {
        "survived": "The signal reaches this boundary in the expected route and remains semantically intact.",
        "died": "The signal is no longer observable after this boundary even though it existed upstream.",
        "mutated": "The signal survives but changes representation, units, ordering, or semantics across this boundary.",
        "rerouted": "The signal remains alive but arrives through a different route than the expected primary path.",
        "bypassed": "A downstream effect appears without evidence that the expected boundary actually carried the signal.",
        "blocked_missing_upstream": "The needed upstream code, artifact, or public evidence is absent, so the boundary is blocked but still formally proven as a blocker.",
    }


def _build_provenance_class_definitions() -> dict[str, str]:
    return {
        "static": "Derived from repo-tracked code or markdown contracts without requiring a live runtime execution.",
        "synthetic": "Derived from deterministic synthetic fixtures or constructed negative/positive probes.",
        "replay_live": "Derived from replay or sim execution surfaces driven from repo-local harnesses.",
        "server_live": "Derived from a live serving path or server runtime interaction.",
    }


def build_interface_localization_contract() -> dict[str, Any]:
    output_root = str(Path(DEFAULT_ARTIFACT_DIR) / DEFAULT_OUTPUT_SUBDIR)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "artifact_kind": CONTRACT_ARTIFACT_KIND,
        "doc_language": "zh-CN",
        "baseline_tuple_field_order": list(BASELINE_TUPLE_FIELD_ORDER),
        "baseline_tuple": build_baseline_tuple(),
        "boundary_ontology": {
            "boundary_order": list(BOUNDARY_ORDER),
            "boundaries": _build_boundary_specs(),
            "never_merge_pairs": [
                {
                    "left": "body_wrist_upper_limb_chain",
                    "right": "dex3_finger_hand_path",
                    "rule": "must_remain_distinct_names_even_when_both_apply_to_the_right_side",
                }
            ],
        },
        "status_ontology": {
            "status_order": list(STATUS_ORDER),
            "legal_statuses": list(STATUS_ORDER),
            "status_definitions": _build_status_definitions(),
            "blocked_but_proven_is_success": True,
        },
        "provenance_classes": {
            "class_order": list(PROVENANCE_CLASS_ORDER),
            "classes": _build_provenance_class_definitions(),
        },
        "artifact_contract": {
            "output_root": output_root,
            "writer_script": "work/recap/scripts/interface_localization_contract.py",
            "required_files": [
                {
                    "relative_path": BASELINE_JSON_NAME,
                    "artifact_kind": BASELINE_ARTIFACT_KIND,
                    "schema_version": BASELINE_SCHEMA_VERSION,
                },
                {
                    "relative_path": CONTRACT_JSON_NAME,
                    "artifact_kind": CONTRACT_ARTIFACT_KIND,
                    "schema_version": CONTRACT_SCHEMA_VERSION,
                },
            ],
            "future_artifact_minimum_fields": list(FUTURE_ARTIFACT_MINIMUM_FIELDS),
            "provenance_class_allowlist": list(PROVENANCE_CLASS_ORDER),
            "json_determinism": {
                "ensure_ascii": True,
                "indent": 2,
                "sort_keys": True,
                "trailing_newline": True,
            },
        },
        "advantage_contract_facts": {
            "provenance_class": "static",
            "contract_version": ADVANTAGE_CONTRACT_VERSION,
            "legacy_contract_version": LEGACY_ADVANTAGE_CONTRACT_VERSION,
            "raw_column": ADVANTAGE_RAW_COLUMN,
            "input_column": ADVANTAGE_INPUT_COLUMN,
            "value_column": ADVANTAGE_VALUE_COLUMN,
            "return_column": ADVANTAGE_RETURN_COLUMN,
            "input_clip_range": float(ADVANTAGE_INPUT_CLIP_RANGE),
            "scale_eps": float(ADVANTAGE_SCALE_EPS),
            "scale_quantile": float(ADVANTAGE_SCALE_QUANTILE),
            "scale_rule": ADVANTAGE_SCALE_RULE,
            "mainline_task_text_field": MAINLINE_TASK_TEXT_FIELD,
        },
        "explicit_non_goals": [
            "Do not add replay, trace mining, numeric repair, or benchmark-fixing logic in Task 1.",
            "Do not collapse body_wrist_upper_limb_chain and dex3_finger_hand_path back into a single right_hand bucket.",
            "Do not mix analysis-only data into deployable or train payloads.",
            "Do not add new dependencies or unrelated prompt/log restoration work.",
        ],
    }


def _extract_embedded_contract_json(markdown_text: str) -> dict[str, Any]:
    pattern = re.compile(
        re.escape(CONTRACT_SPEC_START_MARKER)
        + r"\s*```json\s*(\{.*?\})\s*```\s*"
        + re.escape(CONTRACT_SPEC_END_MARKER),
        re.DOTALL,
    )
    match = pattern.search(markdown_text)
    if match is None:
        raise ValueError("contract markdown is missing the embedded JSON freeze block")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("embedded contract spec must be a JSON object")
    return dict(payload)


def _validate_baseline_tuple_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = _as_non_empty_string(
        payload.get("schema_version"), field_name="schema_version"
    )
    if schema_version != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"schema_version mismatch: {schema_version!r}")
    artifact_kind = _as_non_empty_string(
        payload.get("artifact_kind"), field_name="artifact_kind"
    )
    if artifact_kind != BASELINE_ARTIFACT_KIND:
        raise ValueError(f"artifact_kind mismatch: {artifact_kind!r}")
    field_order = _as_string_list(
        payload.get("baseline_tuple_field_order"),
        field_name="baseline_tuple_field_order",
    )
    if tuple(field_order) != BASELINE_TUPLE_FIELD_ORDER:
        raise ValueError(f"baseline_tuple_field_order mismatch: {field_order!r}")
    baseline_tuple = dict(
        _as_mapping(payload.get("baseline_tuple"), field_name="baseline_tuple")
    )
    _require_exact_keyset(
        baseline_tuple,
        expected_keys=BASELINE_TUPLE_FIELD_ORDER,
        field_name="baseline_tuple",
    )
    for field_name, expected_value in build_baseline_tuple().items():
        actual_value = _as_non_empty_string(
            baseline_tuple.get(field_name), field_name=f"baseline_tuple.{field_name}"
        )
        if actual_value != expected_value:
            raise ValueError(
                f"baseline_tuple.{field_name} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )
    return {
        "schema_version": schema_version,
        "artifact_kind": artifact_kind,
        "baseline_tuple_field_order": field_order,
        "baseline_tuple": baseline_tuple,
    }


def validate_interface_localization_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = _as_non_empty_string(
        payload.get("schema_version"), field_name="schema_version"
    )
    if schema_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"schema_version mismatch: {schema_version!r}")
    artifact_kind = _as_non_empty_string(
        payload.get("artifact_kind"), field_name="artifact_kind"
    )
    if artifact_kind != CONTRACT_ARTIFACT_KIND:
        raise ValueError(f"artifact_kind mismatch: {artifact_kind!r}")
    doc_language = _as_non_empty_string(
        payload.get("doc_language"), field_name="doc_language"
    )
    if doc_language != "zh-CN":
        raise ValueError(f"doc_language mismatch: {doc_language!r}")

    baseline = _validate_baseline_tuple_payload(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "artifact_kind": BASELINE_ARTIFACT_KIND,
            "baseline_tuple_field_order": payload.get("baseline_tuple_field_order"),
            "baseline_tuple": payload.get("baseline_tuple"),
        }
    )

    boundary_ontology = dict(
        _as_mapping(payload.get("boundary_ontology"), field_name="boundary_ontology")
    )
    boundary_order = _as_string_list(
        boundary_ontology.get("boundary_order"),
        field_name="boundary_ontology.boundary_order",
    )
    if tuple(boundary_order) != BOUNDARY_ORDER:
        raise ValueError(
            f"boundary_ontology.boundary_order mismatch: {boundary_order!r}"
        )
    boundaries = dict(
        _as_mapping(
            boundary_ontology.get("boundaries"),
            field_name="boundary_ontology.boundaries",
        )
    )
    _require_exact_keyset(
        boundaries,
        expected_keys=BOUNDARY_ORDER,
        field_name="boundary_ontology.boundaries",
    )
    for boundary_name in BOUNDARY_ORDER:
        boundary_spec = dict(
            _as_mapping(
                boundaries.get(boundary_name),
                field_name=f"boundary_ontology.boundaries.{boundary_name}",
            )
        )
        _ = _as_non_empty_string(
            boundary_spec.get("boundary_kind"),
            field_name=f"boundary_ontology.boundaries.{boundary_name}.boundary_kind",
        )
        _ = _as_non_empty_string(
            boundary_spec.get("lane"),
            field_name=f"boundary_ontology.boundaries.{boundary_name}.lane",
        )
        _ = _as_non_empty_string(
            boundary_spec.get("scope_summary"),
            field_name=f"boundary_ontology.boundaries.{boundary_name}.scope_summary",
        )
        blocked_rule = _as_non_empty_string(
            boundary_spec.get("blocked_rule"),
            field_name=f"boundary_ontology.boundaries.{boundary_name}.blocked_rule",
        )
        if "blocked_missing_upstream" not in blocked_rule:
            raise ValueError(
                f"boundary_ontology.boundaries.{boundary_name}.blocked_rule must mention blocked_missing_upstream"
            )
        if boundary_name in (
            "body_wrist_upper_limb_chain",
            "dex3_finger_hand_path",
        ):
            body_side = _as_non_empty_string(
                boundary_spec.get("body_side"),
                field_name=f"boundary_ontology.boundaries.{boundary_name}.body_side",
            )
            if body_side != "right":
                raise ValueError(
                    f"boundary_ontology.boundaries.{boundary_name}.body_side must be 'right'"
                )
            distinct_from = _as_non_empty_string(
                boundary_spec.get("distinct_from"),
                field_name=f"boundary_ontology.boundaries.{boundary_name}.distinct_from",
            )
            expected_other = (
                "dex3_finger_hand_path"
                if boundary_name == "body_wrist_upper_limb_chain"
                else "body_wrist_upper_limb_chain"
            )
            if distinct_from != expected_other:
                raise ValueError(
                    f"boundary_ontology.boundaries.{boundary_name}.distinct_from mismatch: {distinct_from!r}"
                )
    never_merge_pairs = _as_list(
        boundary_ontology.get("never_merge_pairs"),
        field_name="boundary_ontology.never_merge_pairs",
    )
    if len(never_merge_pairs) != 1:
        raise ValueError(
            "boundary_ontology.never_merge_pairs must contain one frozen pair"
        )
    merge_pair = dict(
        _as_mapping(
            never_merge_pairs[0],
            field_name="boundary_ontology.never_merge_pairs[0]",
        )
    )
    if merge_pair.get("left") != "body_wrist_upper_limb_chain":
        raise ValueError(
            "never_merge_pairs[0].left must stay body_wrist_upper_limb_chain"
        )
    if merge_pair.get("right") != "dex3_finger_hand_path":
        raise ValueError("never_merge_pairs[0].right must stay dex3_finger_hand_path")

    status_ontology = dict(
        _as_mapping(payload.get("status_ontology"), field_name="status_ontology")
    )
    status_order = _as_string_list(
        status_ontology.get("status_order"), field_name="status_ontology.status_order"
    )
    if tuple(status_order) != STATUS_ORDER:
        raise ValueError(f"status_ontology.status_order mismatch: {status_order!r}")
    legal_statuses = _as_string_list(
        status_ontology.get("legal_statuses"),
        field_name="status_ontology.legal_statuses",
    )
    if tuple(legal_statuses) != STATUS_ORDER:
        raise ValueError("status_ontology.legal_statuses must match status_order")
    status_definitions = dict(
        _as_mapping(
            status_ontology.get("status_definitions"),
            field_name="status_ontology.status_definitions",
        )
    )
    _require_exact_keyset(
        status_definitions,
        expected_keys=STATUS_ORDER,
        field_name="status_ontology.status_definitions",
    )
    for status_name, expected_value in _build_status_definitions().items():
        actual_value = _as_non_empty_string(
            status_definitions.get(status_name),
            field_name=f"status_ontology.status_definitions.{status_name}",
        )
        if actual_value != expected_value:
            raise ValueError(
                f"status_ontology.status_definitions.{status_name} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )
    if not _as_bool(
        status_ontology.get("blocked_but_proven_is_success"),
        field_name="status_ontology.blocked_but_proven_is_success",
    ):
        raise ValueError("status_ontology.blocked_but_proven_is_success must be true")

    provenance_classes = dict(
        _as_mapping(payload.get("provenance_classes"), field_name="provenance_classes")
    )
    class_order = _as_string_list(
        provenance_classes.get("class_order"),
        field_name="provenance_classes.class_order",
    )
    if tuple(class_order) != PROVENANCE_CLASS_ORDER:
        raise ValueError(f"provenance_classes.class_order mismatch: {class_order!r}")
    classes = dict(
        _as_mapping(
            provenance_classes.get("classes"), field_name="provenance_classes.classes"
        )
    )
    _require_exact_keyset(
        classes,
        expected_keys=PROVENANCE_CLASS_ORDER,
        field_name="provenance_classes.classes",
    )
    for class_name, expected_value in _build_provenance_class_definitions().items():
        actual_value = _as_non_empty_string(
            classes.get(class_name),
            field_name=f"provenance_classes.classes.{class_name}",
        )
        if actual_value != expected_value:
            raise ValueError(
                f"provenance_classes.classes.{class_name} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )

    artifact_contract = dict(
        _as_mapping(payload.get("artifact_contract"), field_name="artifact_contract")
    )
    output_root = _as_non_empty_string(
        artifact_contract.get("output_root"), field_name="artifact_contract.output_root"
    )
    if output_root != str(Path(DEFAULT_ARTIFACT_DIR) / DEFAULT_OUTPUT_SUBDIR):
        raise ValueError(f"artifact_contract.output_root mismatch: {output_root!r}")
    _ = _as_non_empty_string(
        artifact_contract.get("writer_script"),
        field_name="artifact_contract.writer_script",
    )
    required_files = _as_list(
        artifact_contract.get("required_files"),
        field_name="artifact_contract.required_files",
    )
    if len(required_files) != 2:
        raise ValueError(
            "artifact_contract.required_files must contain exactly two files"
        )
    required_paths: list[str] = []
    for index, file_payload in enumerate(required_files):
        entry = dict(
            _as_mapping(
                file_payload,
                field_name=f"artifact_contract.required_files[{index}]",
            )
        )
        required_paths.append(
            _as_non_empty_string(
                entry.get("relative_path"),
                field_name=f"artifact_contract.required_files[{index}].relative_path",
            )
        )
        _ = _as_non_empty_string(
            entry.get("artifact_kind"),
            field_name=f"artifact_contract.required_files[{index}].artifact_kind",
        )
        _ = _as_non_empty_string(
            entry.get("schema_version"),
            field_name=f"artifact_contract.required_files[{index}].schema_version",
        )
    if required_paths != [BASELINE_JSON_NAME, CONTRACT_JSON_NAME]:
        raise ValueError(
            f"artifact_contract.required_files relative paths mismatch: {required_paths!r}"
        )
    future_fields = _as_string_list(
        artifact_contract.get("future_artifact_minimum_fields"),
        field_name="artifact_contract.future_artifact_minimum_fields",
    )
    if tuple(future_fields) != FUTURE_ARTIFACT_MINIMUM_FIELDS:
        raise ValueError(
            f"artifact_contract.future_artifact_minimum_fields mismatch: {future_fields!r}"
        )
    provenance_allowlist = _as_string_list(
        artifact_contract.get("provenance_class_allowlist"),
        field_name="artifact_contract.provenance_class_allowlist",
    )
    if tuple(provenance_allowlist) != PROVENANCE_CLASS_ORDER:
        raise ValueError(
            f"artifact_contract.provenance_class_allowlist mismatch: {provenance_allowlist!r}"
        )
    json_determinism = dict(
        _as_mapping(
            artifact_contract.get("json_determinism"),
            field_name="artifact_contract.json_determinism",
        )
    )
    if not _as_bool(
        json_determinism.get("ensure_ascii"),
        field_name="artifact_contract.json_determinism.ensure_ascii",
    ):
        raise ValueError("artifact_contract.json_determinism.ensure_ascii must be true")
    if not _as_bool(
        json_determinism.get("sort_keys"),
        field_name="artifact_contract.json_determinism.sort_keys",
    ):
        raise ValueError("artifact_contract.json_determinism.sort_keys must be true")
    if not _as_bool(
        json_determinism.get("trailing_newline"),
        field_name="artifact_contract.json_determinism.trailing_newline",
    ):
        raise ValueError(
            "artifact_contract.json_determinism.trailing_newline must be true"
        )
    if (
        int(
            _as_number(
                json_determinism.get("indent"),
                field_name="artifact_contract.json_determinism.indent",
            )
        )
        != 2
    ):
        raise ValueError("artifact_contract.json_determinism.indent must equal 2")

    advantage_contract_facts = dict(
        _as_mapping(
            payload.get("advantage_contract_facts"),
            field_name="advantage_contract_facts",
        )
    )
    expected_advantage = {
        "provenance_class": "static",
        "contract_version": ADVANTAGE_CONTRACT_VERSION,
        "legacy_contract_version": LEGACY_ADVANTAGE_CONTRACT_VERSION,
        "raw_column": ADVANTAGE_RAW_COLUMN,
        "input_column": ADVANTAGE_INPUT_COLUMN,
        "value_column": ADVANTAGE_VALUE_COLUMN,
        "return_column": ADVANTAGE_RETURN_COLUMN,
        "input_clip_range": float(ADVANTAGE_INPUT_CLIP_RANGE),
        "scale_eps": float(ADVANTAGE_SCALE_EPS),
        "scale_quantile": float(ADVANTAGE_SCALE_QUANTILE),
        "scale_rule": ADVANTAGE_SCALE_RULE,
        "mainline_task_text_field": MAINLINE_TASK_TEXT_FIELD,
    }
    for field_name, expected_value in expected_advantage.items():
        actual_value = advantage_contract_facts.get(field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"advantage_contract_facts.{field_name} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )

    explicit_non_goals = _as_string_list(
        payload.get("explicit_non_goals"), field_name="explicit_non_goals"
    )
    if len(explicit_non_goals) < 4:
        raise ValueError("explicit_non_goals must contain at least four entries")
    return {
        "schema_version": schema_version,
        "artifact_kind": artifact_kind,
        "doc_language": doc_language,
        "baseline": baseline,
        "boundary_ontology": boundary_ontology,
        "status_ontology": status_ontology,
        "provenance_classes": provenance_classes,
        "artifact_contract": artifact_contract,
        "advantage_contract_facts": advantage_contract_facts,
        "explicit_non_goals": explicit_non_goals,
    }


def collect_contract_drifts(
    candidate: Mapping[str, Any],
    *,
    canonical: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    reference = (
        dict(canonical)
        if canonical is not None
        else build_interface_localization_contract()
    )
    drifts: list[dict[str, Any]] = []
    for field_path in PROTECTED_FIELD_PATHS:
        expected_value = _deep_get(reference, field_path)
        actual_value = _deep_get(candidate, field_path)
        if expected_value != actual_value:
            drifts.append(
                {
                    "field_path": field_path,
                    "expected": _value_for_report(expected_value),
                    "actual": _value_for_report(actual_value),
                    "reason": "interface localization contract drift is forbidden",
                }
            )
    return drifts


def assert_contract_matches_canonical(
    candidate: Mapping[str, Any],
    *,
    canonical: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated = validate_interface_localization_contract(candidate)
    drifts = collect_contract_drifts(candidate, canonical=canonical)
    if drifts:
        offending = ", ".join(drift["field_path"] for drift in drifts)
        raise ValueError("interface localization contract drift detected: " + offending)
    return validated


def load_markdown_contract_spec(
    contract_doc_path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    markdown_text = contract_doc_path.read_text(encoding="utf-8")
    required_phrases = (
        "body_wrist_upper_limb_chain",
        "dex3_finger_hand_path",
        "blocked_missing_upstream",
        "survived",
        "died",
        "mutated",
        "rerouted",
        "bypassed",
        "static",
        "synthetic",
        "replay_live",
        "server_live",
        "scene_motion_slice_identifier",
        "checkpoint_identifier",
        "seed_set_identifier",
        "condition_pair_identifier",
        "server_mode_identifier",
        "serving_path_axis_identifier",
        "task_text_surface_identifier",
        "replay_init_source_identifier",
        "baseline_tuple.json",
        "interface_localization_contract.json",
    )
    for phrase in required_phrases:
        if phrase not in markdown_text:
            raise ValueError(
                f"contract markdown is missing required phrase: {phrase!r}"
            )
    spec = _extract_embedded_contract_json(markdown_text)
    validated = assert_contract_matches_canonical(spec)
    return markdown_text, spec, validated


def resolve_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    raw_output_dir = str(args.output_dir).strip()
    if raw_output_dir:
        return _validate_output_dir(_resolve_path(repo_root, raw_output_dir))
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    return _validate_output_dir(artifact_dir / DEFAULT_OUTPUT_SUBDIR)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = _repo_root()
    try:
        output_dir = resolve_output_dir(repo_root, args)
        contract_doc_path = _resolve_path(repo_root, str(args.contract_doc))
        baseline_artifact = build_baseline_tuple_artifact()
        contract_artifact = build_interface_localization_contract()
        _ = _validate_baseline_tuple_payload(baseline_artifact)
        _ = assert_contract_matches_canonical(contract_artifact)
        _markdown_text, _raw_spec, _validated_spec = load_markdown_contract_spec(
            contract_doc_path
        )
        baseline_path = _write_json(output_dir / BASELINE_JSON_NAME, baseline_artifact)
        contract_path = _write_json(output_dir / CONTRACT_JSON_NAME, contract_artifact)
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "output_dir": str(output_dir),
                    "baseline_tuple_json": str(baseline_path),
                    "interface_localization_contract_json": str(contract_path),
                }
            ),
            end="",
        )
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
