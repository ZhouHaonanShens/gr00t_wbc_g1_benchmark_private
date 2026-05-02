from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import inspect
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "interface_localization_sprint"

TEXT_REWRITE_MAP_JSON_NAME = "recap_text_source_and_rewrite_map.json"
TEXT_REWRITE_MAP_SCHEMA_VERSION = (
    "interface_localization_recap_text_source_and_rewrite_map_v1"
)
TEXT_REWRITE_MAP_ARTIFACT_KIND = "recap_text_source_and_rewrite_map"

STAGE_ORDER: tuple[str, ...] = (
    "collector.prompt_raw",
    "collector.prompt_conditioned",
    "labeler.prompt_conditioned",
    "export.task_text_field",
    "export.dual_task_text",
    "runtime_override.annotation.human.task_description",
    "text_indicator_policy.prompt_raw",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import interface_localization_contract
from work.recap import state_conditioned_bucket_a_import
from work.recap.advantage import CONTINUOUS_ADVANTAGE_CONTRACT_DIAGNOSTIC_ROUTE
from work.recap.advantage import MAINLINE_TASK_TEXT_FIELD
from work.recap.advantage import NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE
from work.recap.advantage import build_diagnostic_surface_metadata
from work.recap.text_indicator import CANONICAL_NEGATIVE_LINE
from work.recap.text_indicator import CANONICAL_POSITIVE_LINE
from work.recap.text_indicator import RECAP_TEXT_INDICATOR_AUTHORITY_NAME
from work.recap.text_indicator import RECAP_TEXT_INDICATOR_CARRIER_FIELD
from work.recap.text_indicator import RECAP_TEXT_INDICATOR_SCHEMA_VERSION
from work.recap.text_indicator import RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD
from work.recap.lerobot_export.dataset_export import export_recap_to_lerobot_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface_localization_text_rewrite_map.py",
        description=(
            "Emit a deterministic text-lane rewrite map that shows which task-text "
            "surface is produced, selected, and actually consumed at each audited stage."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, the rewrite-map JSON is written "
            "to <artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for the generated rewrite-map JSON.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def resolve_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    raw_output_dir = str(args.output_dir).strip()
    if raw_output_dir:
        return state_conditioned_bucket_a_import.validate_output_dir(
            _resolve_path(repo_root, raw_output_dir)
        )
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    return state_conditioned_bucket_a_import.validate_output_dir(
        artifact_dir / DEFAULT_OUTPUT_SUBDIR
    )


def _relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _generation_command_for(output_dir: Path, repo_root: Path) -> str:
    display_output_dir = _relpath(repo_root, output_dir)
    return (
        "python3 work/recap/scripts/interface_localization_text_rewrite_map.py "
        f"--output-dir {display_output_dir}"
    )


def _load_contract_payload() -> dict[str, Any]:
    contract = interface_localization_contract.build_interface_localization_contract()
    _ = interface_localization_contract.assert_contract_matches_canonical(contract)
    return contract


def _baseline_summary(contract_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_schema_version": str(contract_payload["schema_version"]),
        "baseline_tuple": dict(contract_payload["baseline_tuple"]),
        "status_allowlist": list(contract_payload["status_ontology"]["legal_statuses"]),
        "provenance_class_allowlist": list(
            contract_payload["provenance_classes"]["class_order"]
        ),
        "mainline_task_text_field": str(
            contract_payload["advantage_contract_facts"]["mainline_task_text_field"]
        ),
    }


def _reference(
    *,
    file_path: str,
    function: str,
    reference_lines: str,
    evidence_role: str,
) -> dict[str, str]:
    return {
        "file_path": file_path,
        "function": function,
        "reference_lines": reference_lines,
        "evidence_role": evidence_role,
    }


def _stage_entry(
    *,
    stage_name: str,
    stage_category: str,
    references: Sequence[Mapping[str, str]],
    surface_authority_key: str,
    authority_status: str,
    mainline_authority: bool,
    diagnostic_only: bool,
    source_field: str,
    rewrite_rule: str,
    selected_field: str,
    effective_consumed_text_surface: str,
    effective_consumed_text: str,
    consumption_scope: str,
    live_model_consumption_proven: bool,
    notes: Sequence[str],
) -> dict[str, Any]:
    return {
        "stage_name": stage_name,
        "stage_category": stage_category,
        "status": "survived",
        "provenance_class": "static",
        "references": [dict(item) for item in references],
        "semantic_selection": {
            "surface_authority_key": surface_authority_key,
            "authority_status": authority_status,
            "mainline_authority": mainline_authority,
            "diagnostic_only": diagnostic_only,
            "source_field": source_field,
            "rewrite_rule": rewrite_rule,
            "selected_field": selected_field,
            "effective_consumed_text_surface": effective_consumed_text_surface,
            "effective_consumed_text": effective_consumed_text,
            "consumption_scope": consumption_scope,
            "live_model_consumption_proven": live_model_consumption_proven,
            "notes": list(notes),
        },
    }


def _export_defaults() -> tuple[str, bool]:
    signature = inspect.signature(export_recap_to_lerobot_v2)
    task_text_default = signature.parameters["task_text_field"].default
    dual_task_default = signature.parameters["dual_task_text"].default
    if not isinstance(task_text_default, str) or not task_text_default:
        raise TypeError(
            "export_recap_to_lerobot_v2.task_text_field default must be a non-empty str"
        )
    if not isinstance(dual_task_default, bool):
        raise TypeError(
            "export_recap_to_lerobot_v2.dual_task_text default must be a bool"
        )
    return task_text_default, dual_task_default


def _surface_authority_labels(
    *, exporter_default_field: str
) -> dict[str, dict[str, Any]]:
    numeric_advantage = build_diagnostic_surface_metadata(
        surface_route=CONTINUOUS_ADVANTAGE_CONTRACT_DIAGNOSTIC_ROUTE,
        authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
        surface_kind="numeric_advantage_lane",
    )
    numeric_advantage.update(
        {
            "surface_authority_key": "numeric_advantage",
            "label": "numeric-adv diagnostic-only lane",
            "notes": [
                "numeric-adv survives only as a diagnostic lane and must not be narrated as the active mainline authority",
            ],
        }
    )
    return {
        "carrier_text_v1": {
            "surface_authority_key": "carrier_text_v1",
            "label": "carrier_text_v1 mainline authority",
            "authority_status": "mainline_authority",
            "mainline_authority": True,
            "diagnostic_only": False,
            "authority_name": RECAP_TEXT_INDICATOR_AUTHORITY_NAME,
            "schema_version": RECAP_TEXT_INDICATOR_SCHEMA_VERSION,
            "carrier_field": RECAP_TEXT_INDICATOR_CARRIER_FIELD,
            "source_prompt_field": RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
            "exporter_default_field": exporter_default_field,
        },
        "policy_condition_metadata": {
            "surface_authority_key": "policy_condition_metadata",
            "label": "policy_condition metadata-side lane",
            "authority_status": "metadata_only",
            "mainline_authority": False,
            "diagnostic_only": False,
            "metadata_fields": [
                "policy_condition.phase",
                "policy_condition.mode",
                "policy_condition_text",
            ],
            "notes": [
                "phase/mode and policy_condition_text remain available as metadata-side surfaces and are not recap_text_indicator_v1 authority",
            ],
        },
        "prompt_conditioned": {
            "surface_authority_key": "prompt_conditioned",
            "label": "prompt_conditioned legacy non-authority lane",
            "authority_status": "legacy_non_authority",
            "mainline_authority": False,
            "diagnostic_only": False,
            "notes": [
                "prompt_conditioned may still appear in collector, labeler, and dual-text export surfaces, but it is legacy/non-authority after carrier_text_v1 freeze",
            ],
        },
        "numeric_advantage": numeric_advantage,
    }


def build_text_source_and_rewrite_map(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    contract_payload = _load_contract_payload()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR).resolve()
    )
    generation_command = _generation_command_for(resolved_output_dir, repo_root)
    exporter_default_field, exporter_dual_default = _export_defaults()
    surface_authority_labels = _surface_authority_labels(
        exporter_default_field=exporter_default_field
    )
    exporter_default_authority_status = (
        "mainline_authority"
        if exporter_default_field == RECAP_TEXT_INDICATOR_CARRIER_FIELD
        else "legacy_non_authority"
    )
    exporter_default_surface_authority_key = (
        "carrier_text_v1"
        if exporter_default_field == RECAP_TEXT_INDICATOR_CARRIER_FIELD
        else "prompt_conditioned"
    )
    exporter_default_summary_note = (
        "Exporter single-text default now aligns with carrier_text_v1 mainline authority while dual_task_text remains a legacy compatibility branch."
        if exporter_default_field == RECAP_TEXT_INDICATOR_CARRIER_FIELD
        else "Exporter single-text default still points at a non-authority lane and therefore remains a compatibility-only surface."
    )

    stages = {
        "collector.prompt_raw": _stage_entry(
            stage_name="collector.prompt_raw",
            stage_category="text_source",
            references=[
                _reference(
                    file_path="work/recap/collector.py",
                    function="collect_episode",
                    reference_lines="614-620",
                    evidence_role="prompt_raw is sourced from normalize_prompt(obs) before prefixing",
                )
            ],
            surface_authority_key="carrier_text_v1",
            authority_status="authority_source_field",
            mainline_authority=False,
            diagnostic_only=False,
            source_field="normalize_prompt(obs)",
            rewrite_rule="identity_after_normalize_prompt",
            selected_field="prompt_raw",
            effective_consumed_text_surface="collector.prompt_raw",
            effective_consumed_text="prompt_raw",
            consumption_scope="collector_local_source",
            live_model_consumption_proven=False,
            notes=(
                "This stage proves the collector-side raw text source, not the downstream winner across all serving paths.",
                "collector.prompt_conditioned and runtime override both derive from this prompt_raw value.",
            ),
        ),
        "collector.prompt_conditioned": _stage_entry(
            stage_name="collector.prompt_conditioned",
            stage_category="text_rewrite",
            references=[
                _reference(
                    file_path="work/recap/collector.py",
                    function="collect_episode",
                    reference_lines="614-665",
                    evidence_role="collector constructs prompt_conditioned and prefixes annotation.human.task_description before client.get_action()",
                )
            ],
            surface_authority_key="prompt_conditioned",
            authority_status="legacy_non_authority",
            mainline_authority=False,
            diagnostic_only=False,
            source_field="prompt_raw",
            rewrite_rule="prompt_conditioned = policy_prompt_prefix + prompt_raw if prefix else prompt_raw",
            selected_field="prompt_conditioned",
            effective_consumed_text_surface="obs_for_policy['annotation.human.task_description'] when policy_prompt_prefix is non-empty",
            effective_consumed_text="prompt_conditioned",
            consumption_scope="collector_runtime_request_surface",
            live_model_consumption_proven=True,
            notes=(
                "The collector definitely rewrites the runtime request text surface before client.get_action() when policy_prompt_prefix is non-empty.",
                "This proves a collector-side request mutation, not the authoritative recap_text_indicator_v1 carrier.",
                "prompt_conditioned remains a non-authority conditioned/request lane even when the collector writes it into annotation.human.task_description.",
            ),
        ),
        "labeler.prompt_conditioned": _stage_entry(
            stage_name="labeler.prompt_conditioned",
            stage_category="text_rewrite",
            references=[
                _reference(
                    file_path="work/recap/labeler.py",
                    function="finalize_m2_prelabels",
                    reference_lines="615-648",
                    evidence_role="labeler writes advantage-conditioned prompt text into prompt_conditioned",
                )
            ],
            surface_authority_key="prompt_conditioned",
            authority_status="legacy_non_authority",
            mainline_authority=False,
            diagnostic_only=False,
            source_field="prompt_raw",
            rewrite_rule="prompt_conditioned = ('advantage positive ' | 'advantage negative ') + prompt_raw",
            selected_field="prompt_conditioned",
            effective_consumed_text_surface="M2 label field prompt_conditioned",
            effective_consumed_text="prompt_conditioned",
            consumption_scope="offline_label_surface",
            live_model_consumption_proven=False,
            notes=(
                "The labeler proves conditioned text exists as an offline label field.",
                "prompt_conditioned is a legacy/non-authority conditioned text field rather than the recap_text_indicator_v1 authority carrier.",
            ),
        ),
        "export.task_text_field": _stage_entry(
            stage_name="export.task_text_field",
            stage_category="text_export_selection",
            references=[
                _reference(
                    file_path="work/recap/lerobot_export/dataset_export.py",
                    function="export_recap_to_lerobot_v2",
                    reference_lines="450-458",
                    evidence_role="exporter default argument fixes task_text_field to the current single-text mainline field",
                ),
                _reference(
                    file_path="work/recap/lerobot_export/dataset_export.py",
                    function="export_recap_to_lerobot_v2",
                    reference_lines="690-717",
                    evidence_role="single-text export branch selects _pick_task_text(label, field=task_text_field)",
                ),
            ],
            surface_authority_key=exporter_default_surface_authority_key,
            authority_status=exporter_default_authority_status,
            mainline_authority=(
                exporter_default_field == RECAP_TEXT_INDICATOR_CARRIER_FIELD
            ),
            diagnostic_only=False,
            source_field="label[task_text_field]",
            rewrite_rule=(
                "single-text branch uses _pick_task_text(field=task_text_field); the frozen mainline default expects carrier_text_v1 and fails closed instead of silently replacing authority with prompt_conditioned"
            ),
            selected_field=exporter_default_field,
            effective_consumed_text_surface="exported dataset single-task-text surface",
            effective_consumed_text=exporter_default_field,
            consumption_scope="dataset_export_single_text",
            live_model_consumption_proven=False,
            notes=(
                exporter_default_summary_note,
                "The legacy key summary.numeric_mainline_consumes is retained only as a compatibility alias for the same prompt_raw source field that feeds carrier_text_v1 authority.",
            ),
        ),
        "export.dual_task_text": _stage_entry(
            stage_name="export.dual_task_text",
            stage_category="text_export_selection",
            references=[
                _reference(
                    file_path="work/recap/lerobot_export/dataset_export.py",
                    function="export_recap_to_lerobot_v2",
                    reference_lines="690-717",
                    evidence_role="dual_task_text branch preserves both prompt_raw and prompt_conditioned when present",
                )
            ],
            surface_authority_key="prompt_conditioned",
            authority_status="legacy_non_authority",
            mainline_authority=False,
            diagnostic_only=False,
            source_field="label['prompt_raw'] + label['prompt_conditioned']",
            rewrite_rule="dual_task_text=True emits each non-empty text without picking a single winner",
            selected_field="dual_task_text",
            effective_consumed_text_surface="exported dataset dual-task-text surface",
            effective_consumed_text="prompt_raw + prompt_conditioned",
            consumption_scope="dataset_export_dual_text",
            live_model_consumption_proven=False,
            notes=(
                "The dual-text branch explicitly carries both surfaces forward instead of collapsing them.",
                "dual_task_text is explicitly non-authoritative because recap_text_indicator_v1 allows only one canonical text carrier.",
            ),
        ),
        "runtime_override.annotation.human.task_description": _stage_entry(
            stage_name="runtime_override.annotation.human.task_description",
            stage_category="runtime_override",
            references=[
                _reference(
                    file_path="work/recap/scripts/demo_g1_vla_live.py",
                    function="_override_task_prompt_in_obs",
                    reference_lines="517-522",
                    evidence_role="demo runtime override writes the selected task prompt into annotation.human.task_description",
                ),
                _reference(
                    file_path="work/recap/scripts/sandbox_g1_policy_prompt_dance.py",
                    function="_build_parser",
                    reference_lines="279-308",
                    evidence_role="sandbox live prompt entry surface exposes --task-prompt and prompt update controls",
                ),
            ],
            surface_authority_key="prompt_conditioned",
            authority_status="runtime_request_surface",
            mainline_authority=False,
            diagnostic_only=False,
            source_field="CLI/live prompt string",
            rewrite_rule="runtime override wraps the chosen prompt as [str(task_prompt)] and assigns it to annotation.human.task_description",
            selected_field="annotation.human.task_description",
            effective_consumed_text_surface="annotation.human.task_description",
            effective_consumed_text="annotation.human.task_description",
            consumption_scope="runtime_observation_override",
            live_model_consumption_proven=True,
            notes=(
                "This stage is the explicit runtime override surface required by the task.",
                "sandbox_g1_policy_prompt_dance also exposes the live prompt entry UI/CLI, while demo_g1_vla_live shows the direct observation write.",
            ),
        ),
        "text_indicator_policy.prompt_raw": _stage_entry(
            stage_name="text_indicator_policy.prompt_raw",
            stage_category="policy_text_consumption",
            references=[
                _reference(
                    file_path="work/recap/policy.py",
                    function="TextIndicatorGr00tPolicy._extract_prompt_raw / _get_action",
                    reference_lines="312-379",
                    evidence_role="text-indicator policy extracts prompt_raw from observation.language and rewrites canonical text before model.get_action()",
                )
            ],
            surface_authority_key="carrier_text_v1",
            authority_status="mainline_authority",
            mainline_authority=True,
            diagnostic_only=False,
            source_field="observation.language[self.language_key]",
            rewrite_rule=(
                "canonical_text = prompt_raw if indicator_mode='omit' else "
                + "prompt_raw + '\\n' + ('"
                + CANONICAL_NEGATIVE_LINE
                + "' | '"
                + CANONICAL_POSITIVE_LINE
                + "')"
            ),
            selected_field="prompt_raw",
            effective_consumed_text_surface="vla_step_data.text",
            effective_consumed_text="canonical_text_from_prompt_raw",
            consumption_scope="policy_local_model_input",
            live_model_consumption_proven=True,
            notes=(
                "This serving path is the authority anchor for recap_text_indicator_v1: carrier_text_v1 is rebuilt from prompt_raw rather than prompt_conditioned.",
                "The companion metadata in work.recap.text_indicator marks prompt_conditioned and policy_condition_text as non-authority roles.",
            ),
        ),
    }

    summary = {
        "authoritative_text_carrier_schema_version": RECAP_TEXT_INDICATOR_SCHEMA_VERSION,
        "authoritative_text_carrier_authority_name": RECAP_TEXT_INDICATOR_AUTHORITY_NAME,
        "authoritative_text_carrier_field": RECAP_TEXT_INDICATOR_CARRIER_FIELD,
        "authoritative_text_carrier_source_field": RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
        "authoritative_text_carrier_definition": "carrier_text_v1 = prompt_raw when indicator_mode=omit else prompt_raw + newline + canonical Advantage line",
        "numeric_mainline_consumes": MAINLINE_TASK_TEXT_FIELD,
        "numeric_mainline_consumes_compat_alias_for": "authoritative_text_carrier_source_field",
        "exporter_default_consumes": exporter_default_field,
        "exporter_default_authority_status": exporter_default_authority_status,
        "exporter_default_dual_task_text": exporter_dual_default,
        "runtime_override_surface": "annotation.human.task_description",
        "text_indicator_policy_source_field": "prompt_raw",
        "text_indicator_policy_effective_consumed_text": "canonical_text_from_prompt_raw",
        "prompt_conditioned_role": "legacy_conditioned_text_non_authority",
        "policy_condition_text_role": "separate_state_conditioned_lane_non_authority",
        "dual_task_text_role": "multi_text_legacy_non_authority",
        "surface_authority_labels": surface_authority_labels,
        "conditioned_text_live_model_consumption_proven": False,
        "conditioned_text_live_model_consumption_note": (
            "Conditioned text is definitely written by labeler/exporter and can be pushed onto the collector runtime request surface, "
            "but the audited authority contract still fixes prompt_raw as the source field for recap_text_indicator_v1/carrier_text_v1, and the text-indicator policy also derives "
            "model text from prompt_raw rather than prompt_conditioned or policy_condition_text."
        ),
    }

    relationships = [
        {
            "relationship_name": "numeric_mainline_vs_exporter_default",
            "display_relationship_name": "authoritative_source_field_vs_exporter_default",
            "compatibility_alias": True,
            "status": "survived",
            "provenance_class": "static",
            "left_surface": "numeric_mainline_consumes",
            "left_value": MAINLINE_TASK_TEXT_FIELD,
            "right_surface": "exporter_default_consumes",
            "right_value": exporter_default_field,
            "relationship": (
                "shared_mainline_authority_via_carrier_text_v1"
                if exporter_default_field == RECAP_TEXT_INDICATOR_CARRIER_FIELD
                else "legacy_non_authority_export_default"
            ),
            "explanation": (
                "work/recap/advantage.py still exposes MAINLINE_TASK_TEXT_FIELD=prompt_raw as the source-field compatibility alias, while "
                "work/recap/lerobot_export/dataset_export.py now defaults task_text_field to carrier_text_v1 so the export surface matches the frozen authority carrier without renaming the legacy summary key."
            ),
        },
        {
            "relationship_name": "collector_conditioned_write_vs_numeric_mainline",
            "status": "survived",
            "provenance_class": "static",
            "left_surface": "collector.prompt_conditioned",
            "left_value": "prompt_conditioned",
            "right_surface": "numeric_mainline_consumes",
            "right_value": MAINLINE_TASK_TEXT_FIELD,
            "relationship": "runtime_request_rewrite_without_mainline_field_swap",
            "explanation": (
                "The collector can rewrite annotation.human.task_description before client.get_action(), but this source path does not "
                "change the authoritative recap_text_indicator_v1 source field frozen in the advantage contract."
            ),
        },
        {
            "relationship_name": "text_indicator_policy_vs_labeler_prompt_conditioned",
            "status": "survived",
            "provenance_class": "static",
            "left_surface": "text_indicator_policy.prompt_raw",
            "left_value": "canonical_text_from_prompt_raw",
            "right_surface": "labeler.prompt_conditioned",
            "right_value": "prompt_conditioned",
            "relationship": "distinct_rewrite_families",
            "explanation": (
                "The text-indicator policy rebuilds the authoritative carrier_text_v1 from prompt_raw plus canonical Advantage lines, while the labeler writes a separate "
                "'advantage positive|negative ' + prompt_raw string into prompt_conditioned as a non-authority legacy rewrite family."
            ),
        },
    ]

    return {
        "schema_version": TEXT_REWRITE_MAP_SCHEMA_VERSION,
        "artifact_kind": TEXT_REWRITE_MAP_ARTIFACT_KIND,
        "provenance_class": "static",
        "generation_command": generation_command,
        "input_baseline_summary": _baseline_summary(contract_payload),
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_text_rewrite_map.py",
            "task1_contract_writer": "work/recap/scripts/interface_localization_contract.py",
            "task2_inventory_writer": "work/recap/scripts/interface_localization_surface_inventory.py",
            "expected_task1_contract_json": str(
                resolved_output_dir / interface_localization_contract.CONTRACT_JSON_NAME
            ),
            "expected_task2_inventory_json": str(
                resolved_output_dir
                / interface_localization_surface_inventory_json_name()
            ),
            "pytest_command": "python3 -m pytest tests/recap/test_prompt_rewrite_map.py -q",
        },
        "summary": summary,
        "surface_authority_labels": surface_authority_labels,
        "stage_order": list(STAGE_ORDER),
        "stages": stages,
        "relationships": relationships,
    }


def interface_localization_surface_inventory_json_name() -> str:
    return "replay_surface_inventory.json"


def write_artifact(*, output_dir: Path, payload: Mapping[str, Any]) -> Path:
    return state_conditioned_bucket_a_import._write_json(
        output_dir / TEXT_REWRITE_MAP_JSON_NAME,
        payload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output_dir = resolve_output_dir(REPO_ROOT, args)
        payload = build_text_source_and_rewrite_map(REPO_ROOT, output_dir=output_dir)
        artifact_path = write_artifact(output_dir=output_dir, payload=payload)
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "output_dir": str(output_dir),
                    "rewrite_map_json": str(artifact_path),
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
