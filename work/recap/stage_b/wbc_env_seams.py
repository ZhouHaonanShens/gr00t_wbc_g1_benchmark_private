"""WBC/env seam catalog for Stage B controller-output diagnostics.

The helpers in this module are intentionally static and dependency-free.  They
do not import GR00T/WBC runtime modules, do not patch submodules, and do not
launch probes.  Their job is to freeze the terminology and hook locations that
Stage B instrumentation should use when tracing:

policy/controller input -> WBC internal target -> WBC output/proxy ->
post-safety/env-applied command -> post-step state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TraceField:
    """One traceable field at a seam."""

    name: str
    meaning: str
    required: bool = True


@dataclass(frozen=True)
class WbcEnvSeam:
    """Static description of a Stage B WBC/env instrumentation seam."""

    trace_name: str
    source: str
    hook_summary: str
    semantic_kind: str
    proxy_level: str
    joint_order: str
    units: str
    safety_position: str
    actuator_mapping_position: str
    gravity_compensation_position: str
    can_claim_true_controller_output: bool
    trace_fields: tuple[TraceField, ...]
    missing_stage_reason: str | None = None
    side_effect_risk: str = "low"

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["trace_fields"] = [asdict(field) for field in self.trace_fields]
        payload["identity_fields"] = [
            asdict(field) for field in get_identity_trace_fields(self.trace_name)
        ]
        return payload


@dataclass(frozen=True)
class ControllerOutputDefinition:
    """Stage B controller-output naming decision."""

    status: str
    recommended_trace_name: str
    semantic_kind: str
    proxy_level: str
    joint_order: str
    units: dict[str, str]
    pre_post_safety: str
    pre_post_actuator_mapping: str
    pre_post_gravity_compensation: str
    is_true_controller_output: bool
    may_support_equivalent_applied_actuator_command_claim: bool
    source: str
    guardrail: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def get_wbc_env_seams() -> tuple[WbcEnvSeam, ...]:
    """Return the frozen Stage B WBC/env seam catalog."""

    return (
        WbcEnvSeam(
            trace_name="controller_input.wbc_goal",
            source="gr00t_wbc/control/utils/n1_utils.py:35-48,127-149",
            hook_summary=(
                "WholeBodyControlWrapper.step after concat_action and before "
                "wbc_policy.set_goal"
            ),
            semantic_kind="policy_target / WBC goal",
            proxy_level="policy_output",
            joint_order="policy action groups converted to upper-body slice",
            units="joint q target plus navigate/base-height command units from WBC config",
            safety_position="pre-safety",
            actuator_mapping_position="pre-actuator-mapping",
            gravity_compensation_position="pre-gravity-compensation",
            can_claim_true_controller_output=False,
            trace_fields=(
                TraceField("target_upper_body_pose", "upper-body q target slice"),
                TraceField("navigate_cmd", "base/navigation command"),
                TraceField("base_height_command", "base height command"),
                TraceField("chain_action_uuid", "policy->controller->env join id"),
            ),
            missing_stage_reason=None,
            side_effect_risk="low",
        ),
        WbcEnvSeam(
            trace_name="wbc_internal_target.set_goal",
            source=(
                "gr00t_wbc/control/policy/"
                "g1_decoupled_whole_body_policy.py:34-90"
            ),
            hook_summary=(
                "G1DecoupledWholeBodyPolicy.set_goal after upper/lower goal split"
            ),
            semantic_kind="WBC internal target",
            proxy_level="wbc_internal",
            joint_order="split upper-body goal and lower-body toggle fields",
            units="target_upper_body_pose/base_height/navigate_cmd WBC units",
            safety_position="pre-safety",
            actuator_mapping_position="pre-actuator-mapping",
            gravity_compensation_position="pre-gravity-compensation",
            can_claim_true_controller_output=False,
            trace_fields=(
                TraceField("upper_body_goal", "goal passed to interpolation policy"),
                TraceField("lower_body_goal", "goal passed to lower-body policy"),
                TraceField(
                    "safe_default_navigate_cmd_injected",
                    "whether missing navigate_cmd was replaced by safe default",
                    required=False,
                ),
            ),
            side_effect_risk="low",
        ),
        WbcEnvSeam(
            trace_name="wbc_lower_body.body_action_cmd_q_dq_tau",
            source="gr00t_wbc/control/policy/g1_gear_wbc_policy.py:186-241",
            hook_summary="G1GearWbcPolicy.get_action after ONNX policy and cmd vectors",
            semantic_kind="pd_target / joint_position_target + zero feedforward torque",
            proxy_level="wbc_internal",
            joint_order="lower_body joint order",
            units="cmd_q q target; cmd_dq zero; cmd_tau zero",
            safety_position="pre-safety",
            actuator_mapping_position="pre-actuator-mapping",
            gravity_compensation_position="pre-gravity-compensation",
            can_claim_true_controller_output=False,
            trace_fields=(
                TraceField("self.action", "raw lower-body ONNX policy output"),
                TraceField("cmd_q", "lower-body joint-position target"),
                TraceField("cmd_dq", "zero lower-body velocity target"),
                TraceField("cmd_tau", "zero lower-body torque/feedforward target"),
            ),
            missing_stage_reason=(
                "cmd_tau is explicitly zeros in current code and is not learned "
                "true controller torque"
            ),
            side_effect_risk="medium_timing_sensitive",
        ),
        WbcEnvSeam(
            trace_name="wbc_output.last_action_q",
            source=(
                "gr00t_wbc/control/policy/"
                "g1_decoupled_whole_body_policy.py:91-147"
            ),
            hook_summary="G1DecoupledWholeBodyPolicy.get_action before return",
            semantic_kind="joint_position_target",
            proxy_level="wbc_internal / policy_output",
            joint_order="Pinocchio full robot joint order",
            units="full-body q target, normally radians for revolute joints",
            safety_position="pre-safety",
            actuator_mapping_position="pre-actuator-mapping",
            gravity_compensation_position="pre-gravity-compensation",
            can_claim_true_controller_output=False,
            trace_fields=(
                TraceField("q", "combined full-body joint-position target"),
                TraceField("upper_body_action", "interpolated upper-body action"),
                TraceField("lower_body_action", "lower-body body_action tuple"),
            ),
            missing_stage_reason="position target proxy; not true torque",
            side_effect_risk="medium_timing_sensitive",
        ),
        WbcEnvSeam(
            trace_name="g1_sync_env.post_safety_q",
            source="gr00t_wbc/control/envs/robocasa/sync_env.py:516-528",
            hook_summary="G1SyncEnv.queue_action before and after safety monitor",
            semantic_kind="post_safety_joint_position_target",
            proxy_level="post_safety",
            joint_order="Pinocchio full robot joint order",
            units="full-body q target",
            safety_position="post-safety",
            actuator_mapping_position="pre-actuator-mapping",
            gravity_compensation_position="pre-gravity-compensation",
            can_claim_true_controller_output=False,
            trace_fields=(
                TraceField("pre_safety_q", "input q before safety monitor"),
                TraceField("post_safety_q", "q after safety monitor action rewrite"),
                TraceField("last_safety_ok", "shutdown/safety status"),
            ),
            missing_stage_reason="post-safety q target proxy; not final env action",
            side_effect_risk="low",
        ),
        WbcEnvSeam(
            trace_name="env_applied_action.robocasa_q_tau",
            source="gr00t_wbc/control/envs/robocasa/sync_env.py:244-284",
            hook_summary=(
                "SyncEnv.queue_action immediately before self.env.step({'q','tau'})"
            ),
            semantic_kind="sim_action / applied_actuator_command_proxy",
            proxy_level="env_step",
            joint_order="actuator order after convert_q_to_actuated_joint_order",
            units=(
                "q target; tau feedforward/torque vector, zero unless gravity "
                "compensation is enabled"
            ),
            safety_position=(
                "post-safety when routed through G1SyncEnv.queue_action; parent "
                "SyncEnv alone is pre-safety"
            ),
            actuator_mapping_position="post-actuator-mapping",
            gravity_compensation_position="tau post-gravity-compensation if enabled",
            can_claim_true_controller_output=False,
            trace_fields=(
                TraceField("action_q", "actuator-order q passed to RoboCasa env"),
                TraceField("tau_q", "actuator-order tau passed to RoboCasa env"),
                TraceField(
                    "enable_gravity_compensation",
                    "whether tau includes gravity compensation",
                ),
                TraceField(
                    "gravity_compensation_joints",
                    "joint groups used for gravity compensation",
                    required=False,
                ),
            ),
            missing_stage_reason=(
                "strongest sim terminal proxy, but still not measured true torque"
            ),
            side_effect_risk="medium_timing_sensitive",
        ),
        WbcEnvSeam(
            trace_name="robocasa_action_dict",
            source="gr00t_wbc/control/envs/robocasa/utils/robocasa_env.py:236-265",
            hook_summary="Gr00tLocomanipRoboCasaEnv.step before super().step",
            semantic_kind="robosuite_action_dict",
            proxy_level="env_internal",
            joint_order="RoboCasa controller-key order",
            units="controller-specific q and *_tau fields",
            safety_position="post-safety if called through G1SyncEnv",
            actuator_mapping_position="post-RoboCasa-conversion",
            gravity_compensation_position="post-gravity-compensation",
            can_claim_true_controller_output=False,
            trace_fields=(
                TraceField("action_dict_keys", "RoboCasa action dict key set"),
                TraceField("q_fields", "q-like controller fields"),
                TraceField("tau_fields", "tau fields generated from action['tau']"),
            ),
            missing_stage_reason="env-internal command proxy, not measured state",
            side_effect_risk="medium_timing_sensitive",
        ),
        WbcEnvSeam(
            trace_name="post_step_state",
            source=(
                "gr00t_wbc/control/envs/robocasa/sync_env.py:144-199; "
                "gr00t_wbc/control/envs/robocasa/utils/robocasa_env.py:360-443"
            ),
            hook_summary="SyncEnv.observe / Gr00tLocomanipRoboCasaEnv.get_gr00t_observation",
            semantic_kind="measured_state",
            proxy_level="post_env_step",
            joint_order="GR00T observation joint order after conversion",
            units="q/dq/ddq/tau_est/time/wrist/privileged observations",
            safety_position="not-applicable",
            actuator_mapping_position="not-applicable",
            gravity_compensation_position="measured actuator_force/tau_est",
            can_claim_true_controller_output=False,
            trace_fields=(
                TraceField("q", "post-step joint positions"),
                TraceField("dq", "post-step joint velocities"),
                TraceField("ddq", "post-step joint accelerations"),
                TraceField("tau_est", "post-step measured/estimated actuator force"),
                TraceField("wrist_pose", "left/right wrist FK pose"),
                TraceField(
                    "privileged_obs_keys",
                    "task-specific object/contact keys if exposed",
                    required=False,
                ),
                TraceField("success_info", "diagnostic success dict", required=False),
            ),
            missing_stage_reason="post-step state result; not an action command",
            side_effect_risk="low",
        ),
    )


def get_identity_trace_fields(trace_name: str) -> tuple[TraceField, ...]:
    """Return identity/hash fields required for a WBC/env seam event.

    Worker-3 owns the policy/action UUID contract.  WBC/env stages must keep the
    same chain id and upstream action-content hash, then add their own
    stage-payload hash for transformed q/tau/state payloads.  This prevents
    same-observation comparisons from being broken by the transformed content
    itself while still preserving per-stage payload identity.
    """

    post_state = trace_name == "post_step_state"
    return (
        TraceField("chain_action_uuid", "policy->controller->env join id"),
        TraceField(
            "upstream_action_content_hash",
            "unchanged hash of upstream policy/controller action content",
            required=not post_state,
        ),
        TraceField(
            "stage_payload_hash",
            "hash of this seam's transformed payload; never overwrites upstream hash",
        ),
        TraceField(
            "contrast_group_uuid",
            "same-observation paired comparison id excluding action content",
            required=False,
        ),
    )


def get_controller_output_definition() -> ControllerOutputDefinition:
    """Return the Stage B naming decision for controller_output."""

    return ControllerOutputDefinition(
        status="APPLIED_ACTUATOR_COMMAND_PROXY_OBSERVED_WHEN_SYNC_ENV_Q_TAU_TRACED",
        recommended_trace_name="proxy:robocasa_sync_env.env_step_q_tau",
        semantic_kind="sim_action / applied_actuator_command_proxy",
        proxy_level="env_step",
        joint_order="actuator order after convert_q_to_actuated_joint_order",
        units={
            "q": "robot_model q target units; revolute joints normally radians",
            "tau": (
                "torque/feedforward units; zero unless gravity compensation is "
                "enabled"
            ),
        },
        pre_post_safety=(
            "post-safety when routed through G1SyncEnv.queue_action; parent "
            "SyncEnv alone is pre-safety"
        ),
        pre_post_actuator_mapping="post-actuator-mapping",
        pre_post_gravity_compensation=(
            "tau is post-gravity-compensation if enabled; q is unaffected"
        ),
        is_true_controller_output=False,
        may_support_equivalent_applied_actuator_command_claim=True,
        source="gr00t_wbc/control/envs/robocasa/sync_env.py:244-284",
        guardrail=(
            "Do not call this true torque or learned WBC controller output unless "
            "a later probe proves command-sender/actuator equivalence."
        ),
    )


def build_wbc_env_seam_map() -> dict[str, object]:
    """Build a machine-readable WBC/env seam map."""

    return {
        "schema_version": "stage_b_wbc_env_seam_map_v1",
        "controller_output_definition": get_controller_output_definition().to_jsonable(),
        "identity_contract": {
            "chain_action_uuid": (
                "same policy->controller->env join id from worker-3/worker-2 "
                "contracts; do not derive from transformed action content"
            ),
            "upstream_action_content_hash": (
                "carry forward the upstream action content hash unchanged"
            ),
            "stage_payload_hash": (
                "hash each transformed seam payload, e.g. actuator-order q/tau, "
                "without overwriting upstream_action_content_hash"
            ),
            "contrast_group_uuid": (
                "same-observation comparison id; excludes indicator_mode and "
                "action content"
            ),
        },
        "seams": [seam.to_jsonable() for seam in get_wbc_env_seams()],
        "forbidden_mislabels": [
            "Do not call last_action.q true torque",
            "Do not call cmd_tau learned WBC torque; current code sets it to zeros",
            "Do not call tau_est an action command; it is a post-step readout",
            "Do not treat missing object/contact privileged keys as zero delta",
        ],
    }


def build_controller_output_definition_markdown() -> str:
    """Render the controller-output definition in user-readable Chinese."""

    definition = get_controller_output_definition()
    seams = get_wbc_env_seams()
    rows = "\n".join(
        "| {trace_name} | {semantic_kind} | {proxy_level} | {joint_order} | {safety} | {mapping} | {gcomp} | {true} |".format(
            trace_name=seam.trace_name,
            semantic_kind=seam.semantic_kind,
            proxy_level=seam.proxy_level,
            joint_order=seam.joint_order,
            safety=seam.safety_position,
            mapping=seam.actuator_mapping_position,
            gcomp=seam.gravity_compensation_position,
            true="yes" if seam.can_claim_true_controller_output else "no",
        )
        for seam in seams
    )
    return f"""# Stage B controller_output / applied actuator command definition v1

## 结论

当前推荐终端 action seam：`{definition.recommended_trace_name}`。

- semantic_kind：`{definition.semantic_kind}`
- proxy_level：`{definition.proxy_level}`
- joint_order：`{definition.joint_order}`
- source：`{definition.source}`
- true_controller_output：`{str(definition.is_true_controller_output).lower()}`
- guardrail：{definition.guardrail}

该 seam 可支持 `equivalent_applied_actuator_command` 的后续证明，但在未证明
command-sender / actuator equivalence 前，不得写成 true controller torque。

## Identity / hash rule

- `chain_action_uuid` 必须沿用同一次 policy call 的上游 join id。
- `upstream_action_content_hash` 必须原样保留，不得被 actuator-order `{{q,tau}}` 或
  post-step state 的 hash 覆盖。
- 每个 WBC/env 转换层另外写 `stage_payload_hash`，例如
  `env_applied_action.robocasa_q_tau` 对 actuator-order `action_q/tau_q` 单独 hash。
- `contrast_group_uuid` 继续用于 same-observation paired comparison，不能包含 action
  content hash。

## Seam table

| trace_name | semantic_kind | proxy_level | joint_order | safety | actuator_mapping | gravity_compensation | true_controller_output |
|---|---|---|---|---|---|---|---|
{rows}
"""


def write_artifacts(output_dir: Path) -> tuple[Path, Path]:
    """Write the static seam map JSON and controller-output markdown."""

    output_dir.mkdir(parents=True, exist_ok=True)
    seam_map_path = output_dir / "wbc_env_seam_map_v1.json"
    controller_md_path = output_dir / "controller_output_definition.md"
    seam_map_path.write_text(
        json.dumps(build_wbc_env_seam_map(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    controller_md_path.write_text(
        build_controller_output_definition_markdown(),
        encoding="utf-8",
    )
    return seam_map_path, controller_md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-artifacts", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.write_artifacts is not None:
        seam_map_path, controller_md_path = write_artifacts(args.write_artifacts)
        print(json.dumps({"seam_map": str(seam_map_path), "controller_definition": str(controller_md_path)}))
        return 0

    payload = build_wbc_env_seam_map()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(build_controller_output_definition_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
