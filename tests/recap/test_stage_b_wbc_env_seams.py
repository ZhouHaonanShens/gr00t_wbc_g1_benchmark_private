from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.stage_b.wbc_env_seams import (  # noqa: E402
    build_controller_output_definition_markdown,
    build_wbc_env_seam_map,
    get_controller_output_definition,
    get_identity_trace_fields,
    get_wbc_env_seams,
    write_artifacts,
)


def test_env_step_q_tau_is_terminal_proxy_not_true_controller_output() -> None:
    definition = get_controller_output_definition()

    assert definition.recommended_trace_name == "proxy:robocasa_sync_env.env_step_q_tau"
    assert definition.proxy_level == "env_step"
    assert definition.is_true_controller_output is False
    assert definition.may_support_equivalent_applied_actuator_command_claim is True
    assert "sync_env.py:244-284" in definition.source
    assert "post-actuator" in definition.pre_post_actuator_mapping


def test_wbc_env_seam_catalog_covers_controller_to_state_chain() -> None:
    seam_by_name = {seam.trace_name: seam for seam in get_wbc_env_seams()}

    expected = {
        "controller_input.wbc_goal",
        "wbc_internal_target.set_goal",
        "wbc_lower_body.body_action_cmd_q_dq_tau",
        "wbc_output.last_action_q",
        "g1_sync_env.post_safety_q",
        "env_applied_action.robocasa_q_tau",
        "robocasa_action_dict",
        "post_step_state",
    }
    assert expected <= seam_by_name.keys()

    lower_body = seam_by_name["wbc_lower_body.body_action_cmd_q_dq_tau"]
    assert lower_body.can_claim_true_controller_output is False
    assert lower_body.missing_stage_reason is not None
    assert "zeros" in lower_body.missing_stage_reason

    state = seam_by_name["post_step_state"]
    field_names = {field.name for field in state.trace_fields}
    assert {"q", "dq", "ddq", "tau_est", "wrist_pose"} <= field_names
    assert any(not field.required for field in state.trace_fields), (
        "object/contact/privileged fields must remain optional and missing-stage aware"
    )


def test_machine_readable_map_keeps_claim_boundary_wording() -> None:
    payload = build_wbc_env_seam_map()

    assert payload["schema_version"] == "stage_b_wbc_env_seam_map_v1"
    identity = payload["identity_contract"]
    assert "same policy->controller->env join id" in identity["chain_action_uuid"]
    assert "carry forward" in identity["upstream_action_content_hash"]
    assert "without overwriting" in identity["stage_payload_hash"]
    forbidden = "\n".join(payload["forbidden_mislabels"])
    assert "last_action.q true torque" in forbidden
    assert "cmd_tau learned WBC torque" in forbidden
    assert "tau_est an action command" in forbidden
    assert "missing object/contact" in forbidden


def test_wbc_env_events_preserve_upstream_identity_hashes() -> None:
    for seam in get_wbc_env_seams():
        field_by_name = {field.name: field for field in get_identity_trace_fields(seam.trace_name)}
        assert {
            "chain_action_uuid",
            "upstream_action_content_hash",
            "stage_payload_hash",
            "contrast_group_uuid",
        } <= field_by_name.keys()
        assert field_by_name["stage_payload_hash"].required is True

    payload = build_wbc_env_seam_map()
    for seam in payload["seams"]:
        identity_names = {field["name"] for field in seam["identity_fields"]}
        assert "chain_action_uuid" in identity_names
        assert "stage_payload_hash" in identity_names


def test_controller_output_markdown_names_proxy_and_denies_true_torque() -> None:
    text = build_controller_output_definition_markdown()

    assert "proxy:robocasa_sync_env.env_step_q_tau" in text
    assert "true_controller_output：`false`" in text
    assert "不得写成 true controller torque" in text
    assert "upstream_action_content_hash" in text
    assert "stage_payload_hash" in text


def test_artifact_writer_outputs_parseable_json_and_markdown(tmp_path: Path) -> None:
    seam_map_path, controller_md_path = write_artifacts(tmp_path)

    payload = json.loads(seam_map_path.read_text(encoding="utf-8"))
    text = controller_md_path.read_text(encoding="utf-8")

    assert payload["controller_output_definition"]["is_true_controller_output"] is False
    assert "env_applied_action.robocasa_q_tau" in text
