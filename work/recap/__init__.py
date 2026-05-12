from __future__ import annotations

from importlib import import_module
import sys

from .advantage import (
    ADVANTAGE_CONTRACT_VERSION,
    ADVANTAGE_INPUT_COLUMN,
    ADVANTAGE_RAW_COLUMN,
)
from .episode_writer import EpisodeWriter, summarize_value, to_jsonable_list

_OBJECT_ALIASES = {
    "GR00TRecapActionHead": ("work.recap.model", "GR00TRecapActionHead"),
    "GR00TRecapModel": ("work.recap.model", "GR00TRecapModel"),
    "AdvantageAwareGr00tPolicy": ("work.recap.policy", "AdvantageAwareGr00tPolicy"),
    "TextIndicatorGr00tPolicy": ("work.recap.policy", "TextIndicatorGr00tPolicy"),
}


_SCRIPT_ALIASES = {
    "31_recap_collect_rollouts",
    "32_recap_label_dataset",
    "33_recap_export_lerobot_v2_dataset",
    "34_recap_finetune_repro",
    "34b_recap_numeric_adv_smoke",
    "38_recap_online_loop_iterate",
    "39_recap_export_lerobot_v2_with_video",
    "3A_recap_multi_iter_loop",
    "3A_recap_summarize_results",
    "3D_recap_eval",
    "3D_recap_finetune_full",
    "3D_recap_run_adv_server",
    "41_vlm_critic_contract_check",
    "41b_vlm_critic_split_manifest",
    "41c_vlm_critic_public_warmstart_manifest",
    "42_vlm_critic_dataset_build",
    "43_vlm_critic_train",
    "43b_vlm_critic_artifact_smoke",
    "43c_vlm_critic_sign_audit",
    "44_vlm_critic_offline_gate",
    "44b_vlm_critic_ablation_gate",
    "44c_vlm_critic_postmortem",
    "45_recap_label_dataset_vlm_backend",
    "45b_vlm_critic_relabel_audit",
    "45c_vlm_critic_finetune_smoke",
    "45d_vlm_critic_eval_smoke",
    "45e_vlm_critic_downstream_gate",
    "45f_vlm_critic_pilot_eval_wrapper",
    "46d_vlm_critic_fullsize_relabel",
    "demo_g1_vla_live",
    "gr00t_action_chain_telemetry",
    "gr00t_checkpoint_provenance_gate",
    "gr00t_condition_flip_probe",
    "gr00t_controller_audit_new_embodiment",
    "gr00t_controller_audit_unitree_g1",
    "gr00t_d_ladder_new_embodiment",
    "gr00t_d_ladder_policy_gate",
    "gr00t_d_ladder_unitree_g1",
    "gr00t_dual_branch_scorecard",
    "gr00t_eval_contract_gate",
    "gr00t_ladder_policy_gate",
    "gr00t_p_ladder_new_embodiment",
    "gr00t_p_ladder_unitree_g1",
    "gr00t_public_anchor_eval",
    "gr00t_recap_attribution_pack",
    "gr00t_teacher_reachability_gate",
    "gr00t_teacher_student_gap",
    "gr00t_wbc_preflight_gate",
    "interface_localization_action_roundtrip",
    "interface_localization_contract",
    "interface_localization_numeric_gap",
    "interface_localization_pack",
    "interface_localization_right_hand_split",
    "interface_localization_surface_inventory",
    "interface_localization_text_rewrite_map",
    "interface_localization_trace",
    "prompt_sensitivity_probe_g1",
    "sandbox_g1_policy_prompt_dance",
    "state_conditioned_bucket_a_import",
    "state_conditioned_bucket_a_sidecar",
    "state_conditioned_build_training_set",
    "state_conditioned_collect_buckets",
    "state_conditioned_contract_gate",
    "state_conditioned_dev_manifest",
    "state_conditioned_micro_overfit_sanity",
    "state_conditioned_offline_sanity",
    "state_conditioned_open_loop_agreement",
    "state_conditioned_oracle_eval",
    "state_conditioned_phase0_smoke",
    "state_conditioned_pseudodemo_audit_pack",
    "state_conditioned_pseudodemo_label_audit",
    "state_conditioned_pseudodemo_taxonomy_checker",
    "state_conditioned_snapshot_harvest",
    "state_conditioned_teacher_upper_bound_sanity",
    "state_conditioned_train",
    "state_conditioned_wave_freeze_manifest",
    "state_structured_recap_analysis_checker",
}

_MODULE_ALIASES = {
    "export_lerobot_with_video": "work.recap.lerobot_export.workflow",
    "lerobot_v2_export": "work.recap.lerobot_export.dataset_export",
    "lerobot_v2_export_with_video": "work.recap.lerobot_export.video_export",
}


def __getattr__(name: str):
    if name in _OBJECT_ALIASES:
        module_name, attr_name = _OBJECT_ALIASES[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    if name in _MODULE_ALIASES:
        module = import_module(_MODULE_ALIASES[name])
        globals()[name] = module
        sys.modules[f"{__name__}.{name}"] = module
        return module
    if name not in _SCRIPT_ALIASES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"work.recap.scripts.{name}")
    globals()[name] = module
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SCRIPT_ALIASES | set(_MODULE_ALIASES) | set(_OBJECT_ALIASES))


__all__ = [
    "ADVANTAGE_CONTRACT_VERSION",
    "ADVANTAGE_INPUT_COLUMN",
    "ADVANTAGE_RAW_COLUMN",
    "AdvantageAwareGr00tPolicy",
    "EpisodeWriter",
    "GR00TRecapActionHead",
    "GR00TRecapModel",
    "TextIndicatorGr00tPolicy",
    "export_lerobot_with_video",
    "lerobot_v2_export",
    "lerobot_v2_export_with_video",
    "summarize_value",
    "to_jsonable_list",
    *sorted(_SCRIPT_ALIASES),
]
