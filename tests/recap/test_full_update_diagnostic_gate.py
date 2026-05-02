from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from work.recap import policy as POLICY


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


LAUNCH = _load_module(
    "launch_finetune_use_ddp_for_diagnostic_gate_tests",
    "work/recap/launch_finetune_use_ddp.py",
)
FINETUNE_FULL = _load_module(
    "finetune_full_for_diagnostic_gate_tests",
    "work/recap/finetune_full.py",
)
CONTINUATION_CONTROL = _load_module(
    "continuation_control_for_diagnostic_gate_tests",
    "work/recap/scripts/30i_stage3_baseline_continuation_control.py",
)
SUPERVISOR = _load_module(
    "full_update_scope_supervisor_for_diagnostic_gate_tests",
    "work/recap/scripts/34c_full_update_scope_supervisor.py",
)
SMOKE34B = _load_module(
    "recap_numeric_adv_smoke_for_diagnostic_gate_tests",
    "work/recap/scripts/34b_recap_numeric_adv_smoke.py",
)
ROLLOUT35A = _load_module(
    "full_update_rollout_probe_for_diagnostic_gate_tests",
    "work/recap/scripts/35a_full_update_rollout_probe.py",
)
REPORT35B = _load_module(
    "full_update_report_for_diagnostic_gate_tests",
    "work/recap/scripts/35b_full_update_report.py",
)


class _FakeCuda:
    @staticmethod
    def get_arch_list() -> list[str]:
        return ["sm_120"]


class _FakeTorch:
    __version__ = "2.8.0-test"
    cuda = _FakeCuda()


class _FakeSmokeModule:
    def __init__(
        self,
        *,
        output_dir: Path,
        static_by_scope: dict[str, dict[str, object]],
        preflight_by_scope: dict[str, dict[str, object]],
        runtime_by_scope: dict[str, tuple[int, dict[str, object]]],
    ) -> None:
        self.output_dir = output_dir
        self.static_by_scope = static_by_scope
        self.preflight_by_scope = preflight_by_scope
        self.runtime_by_scope = runtime_by_scope

    def _resolve_full_update_output_dir(self, repo_root: Path, raw: str) -> Path:
        del repo_root, raw
        return self.output_dir

    def run_numeric_adv_static_scope_audit(
        self,
        args: SimpleNamespace,
        *,
        requested_scope_override: str | None = None,
    ) -> dict[str, object]:
        del args
        assert requested_scope_override is not None
        static_audit = dict(self.static_by_scope[requested_scope_override])
        runtime_preflight = dict(self.preflight_by_scope[requested_scope_override])

        metadata_dir = self.output_dir / "repo_local_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        static_audit_path = metadata_dir / f"{requested_scope_override}_static_audit.json"
        static_audit_path.write_text(
            json.dumps(static_audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        preflight_path = self.output_dir / "p0_scope_audit" / "runtime_preflight.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(
            json.dumps(runtime_preflight, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "train_scope_requested": requested_scope_override,
            "output_dir": str(self.output_dir),
            "metadata_dir": str(metadata_dir),
            "static_audit_path": str(static_audit_path),
            "static_audit": static_audit,
            "runtime_preflight_path": str(preflight_path),
        }

    def run_numeric_adv_single_scope(
        self,
        args: SimpleNamespace,
        *,
        requested_scope_override: str | None = None,
        summary_json_override: object = None,
        emit_summary: bool = False,
    ) -> tuple[int, dict[str, object]]:
        del args, summary_json_override, emit_summary
        assert requested_scope_override is not None
        return self.runtime_by_scope[requested_scope_override]

    def load_task11_best_scope_authority(
        self,
        repo_root: Path,
        *,
        best_scope_audit: str = "",
    ) -> dict[str, object]:
        return SMOKE34B.load_task11_best_scope_authority(
            repo_root,
            best_scope_audit=best_scope_audit,
        )

    def load_task11_preformal_gate_summary(
        self,
        repo_root: Path,
        *,
        gate_summary: str = "",
    ) -> dict[str, object]:
        return SMOKE34B.load_task11_preformal_gate_summary(
            repo_root,
            gate_summary=gate_summary,
        )

    def resolve_task11_conditioned_warm_start_checkpoint(
        self,
        repo_root: Path,
        *,
        gate_summary_payload: dict[str, object] | None,
        continuation_checkpoint_path: str = "",
    ) -> Path:
        return SMOKE34B.resolve_task11_conditioned_warm_start_checkpoint(
            repo_root,
            gate_summary_payload=gate_summary_payload,
            continuation_checkpoint_path=continuation_checkpoint_path,
        )


def _make_args(
    tmp_path: Path,
    requested_scope: str,
    **overrides: object,
) -> SimpleNamespace:
    payload: dict[str, object] = {
        "recap_train_scope": requested_scope,
        "output_dir": str(tmp_path / "authority_out"),
        "summary_json": "",
        "entrypoint": "",
        "require_p3_formal_eligible": False,
        "gate_summary": "",
        "best_scope_audit": "",
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _static_audit(
    *,
    verdict: str = "PASS",
    fits_available_memory: bool | None = True,
) -> dict[str, object]:
    return {
        "static_verdict": verdict,
        "static_block_reasons": [],
        "memory_feasibility": {
            "fits_available_memory": fits_available_memory,
            "estimated_total_bytes": 1024,
        },
    }


def _runtime_payload(output_dir: Path) -> dict[str, object]:
    runtime_log_path = output_dir / "runtime.log"
    runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_log_path.write_text("runtime ok\n", encoding="utf-8")
    return {
        "wrapper_status": "ok",
        "runtime_log_path": str(runtime_log_path),
        "delegate_cmd": ["python", "34b"],
        "delegate_cmd_shell": "python 34b",
        "checkpoint_load_report_path": str(output_dir / "checkpoint_report.json"),
        "selected_checkpoint_path": str(output_dir / "checkpoint-10"),
        "selected_checkpoint_asset_path": str(output_dir / "checkpoint-10" / "model.safetensors"),
        "selected_checkpoint_exists": True,
        "trainer_global_step": 1,
        "advantage_embedding_keys_present": True,
        "advantage_embedding_missing_keys": [],
        "grad_probe_after_backward": {"status": "ok"},
        "param_delta_after_step": {"status": "ok"},
        "all_major_grad_norms": {
            "diffusion_trunk": 0.1,
            "advantage_embedding": 0.01,
        },
        "all_major_param_delta": {
            "diffusion_trunk": 0.2,
            "advantage_embedding": 0.02,
        },
        "error": None,
    }


def _read_attempts(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "repo_local_metadata" / "downgrade_attempts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_dynamic_audit(output_dir: Path) -> dict[str, object]:
    path = output_dir / "repo_local_metadata" / "full_update_scope_audit_dynamic.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _conditioning_probe_payload(
    *,
    loss_sensitivity_gate_pass: bool = True,
    include_train_subset: bool = True,
    include_heldout: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "loss_sensitivity_gate_pass": loss_sensitivity_gate_pass,
    }
    if include_train_subset:
        payload["train_subset_loss_probe"] = {
            "loss_sensitive": loss_sensitivity_gate_pass,
        }
    if include_heldout:
        payload["heldout_loss_probe"] = {
            "loss_sensitive": loss_sensitivity_gate_pass,
        }
    return payload


def _paired_probe_payload(
    *,
    action_sensitivity_gate_pass: bool = True,
    instrumentation_incomplete: bool = False,
) -> dict[str, object]:
    layers: dict[str, dict[str, object]] = {
        "raw_normalized_action_delta": {
            "available": True,
            "delta_l2": 0.11,
        },
        "decoded_action_delta": {
            "available": True,
            "delta_l2": 0.12,
        },
        "postprocessed_action_delta": {
            "available": True,
            "delta_l2": 0.13,
        },
        "controller_input_delta": {
            "available": True,
            "delta_l2": 0.14,
        },
    }
    if not action_sensitivity_gate_pass:
        for layer in layers.values():
            layer["delta_l2"] = 0.0
    if instrumentation_incomplete:
        layers["controller_input_delta"] = {
            "available": False,
            "reason": "controller_input_seam_not_available",
        }
    return {"action_delta_layers": layers}


def _label_semantics_payload(
    *,
    label_semantics_gate_pass: bool = True,
    shuffled_advantage_negative_control_pass: bool = True,
) -> dict[str, object]:
    effective_label_semantics_gate_pass = bool(
        label_semantics_gate_pass and shuffled_advantage_negative_control_pass
    )
    blocking_reasons: list[str] = []
    if not effective_label_semantics_gate_pass:
        blocking_reasons.append("label_semantics_gate_block")
    return {
        "label_semantics_gate_pass": effective_label_semantics_gate_pass,
        "blocking_reasons": blocking_reasons,
        "shuffled_advantage_negative_control": {
            "negative_control_pass": shuffled_advantage_negative_control_pass,
            "blocking_reasons": (
                []
                if shuffled_advantage_negative_control_pass
                else ["train_subset_shuffled_control_not_worse_than_true"]
            ),
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _route_freeze_payload(*, frozen: bool = True) -> dict[str, object]:
    return {
        "authority_owner": "training_entrypoint",
        "carrier_route": None,
        "carrier_schema_version": None,
        "diagnostic_only": True,
        "frozen": frozen,
        "indicator_mode": None,
        "mainline_authority": False,
        "policy_class_name": POLICY.DIAGNOSTIC_NUMERIC_ADV_POLICY_CLASS_NAME,
        "route": POLICY.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
        "runtime_indicator_mode_required": False,
        "runtime_supported_indicator_modes": [],
        "schema_version": "repo_local_route_freeze_v1",
    }


def _write_task13_scope_and_route_authority(
    tmp_path: Path,
    *,
    resolution_status: str = "PASS",
    best_scope_authority: bool = True,
    route_frozen: bool = True,
) -> None:
    _write_json(
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/p1_one_step/repo_local_metadata/full_update_scope_audit_dynamic.json",
        {
            "artifact_kind": "recap_full_update_scope_audit_dynamic",
            "best_scope_authority": best_scope_authority,
            "resolution_status": resolution_status,
        },
    )
    _write_json(
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/p2_full_update_overfit20/repo_local_metadata/version_surface.json",
        {
            "route_freeze": _route_freeze_payload(frozen=route_frozen),
        },
    )


def _write_task13_task8_artifacts(
    tmp_path: Path,
    *,
    conditioning_probe: dict[str, object] | None = None,
    paired_probe: dict[str, object] | None = None,
    label_semantics_audit: dict[str, object] | None = None,
) -> None:
    v2_root = (
        tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    )
    if conditioning_probe is not None:
        _write_json(
            v2_root / "p2_full_update_overfit20/conditioning_functional_probe_step20.json",
            conditioning_probe,
        )
    if paired_probe is not None:
        _write_json(
            v2_root / "p2_full_update_overfit20/paired_action_probe_step20.json",
            paired_probe,
        )
    if label_semantics_audit is not None:
        _write_json(
            v2_root / "p2_5_label_semantics/label_semantics_audit.json",
            label_semantics_audit,
        )


def _write_task13_baseline_authority(tmp_path: Path) -> Path:
    baseline_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v1"
    _write_json(
        baseline_root / "eval_seed_set.json",
        {
            "schema_version": "recap_eval_seed_set_v1",
            "formal_eval_episodes": 10,
            "seed_base": 20260421,
            "episode_indices": list(range(10)),
            "seeds": [20260421, 20260422, 20260423, 20260424, 20260425],
            "same_seed_set_required_for": [
                "baseline_eval",
                "numeric_advantage_conditioned_eval",
            ],
        },
    )
    _write_json(
        baseline_root / "min_loop_verdict.json",
        {
            "artifact_kind": "recap_min_loop_comparative_verdict",
            "claim_level": "fail",
        },
    )
    for relative in (
        "eval_numeric_advantage_conditioned/eval_summary.json",
        "eval_baseline_continuation_control/eval_summary.json",
        "t5_baseline_formal_eval/eval_summary.json",
    ):
        _write_json(
            baseline_root / relative,
            {
                "episodes": 10,
                "success_rate": 0.0,
                "success_count": 0,
            },
        )
    return baseline_root


def _build_eval_summary_payload(
    *,
    seeds: list[int],
    label: str,
    success_count: int = 0,
    success_rate: float = 0.0,
    advantage: float | None = None,
) -> dict[str, object]:
    return {
        "episodes": len(seeds),
        "success_count": success_count,
        "success_rate": success_rate,
        "seed_base": seeds[0],
        "episode_results": [
            {
                "seed": seed,
                "success": False,
                "episode_index": index,
            }
            for index, seed in enumerate(seeds)
        ],
        "advantage": advantage,
        "advantage_mode": "explicit_positive" if advantage is not None else "unconditional",
        "server_provenance": {
            "policy_model_path": f"agent/artifacts/mock/{label}/checkpoint-200",
        },
    }


def _write_task14_baseline_authority(tmp_path: Path, *, seeds: list[int]) -> Path:
    baseline_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v1"
    _write_json(
        baseline_root / "eval_seed_set.json",
        {
            "schema_version": "recap_eval_seed_set_v1",
            "formal_eval_episodes": len(seeds),
            "seed_base": seeds[0],
            "episode_indices": list(range(len(seeds))),
            "seeds": seeds,
            "same_seed_set_required_for": [
                "baseline_eval",
                "numeric_advantage_conditioned_eval",
                "state_conditioned_eval_if_run",
            ],
        },
    )
    _write_json(
        baseline_root / "min_loop_verdict.json",
        {
            "artifact_kind": "recap_min_loop_comparative_verdict",
            "claim_level": "fail",
        },
    )
    _write_json(
        baseline_root / "t5_baseline_formal_eval/eval_summary.json",
        _build_eval_summary_payload(seeds=seeds, label="baseline"),
    )
    _write_json(
        baseline_root / "eval_baseline_continuation_control/eval_summary.json",
        _build_eval_summary_payload(seeds=seeds, label="continuation_control"),
    )
    _write_json(
        baseline_root / "eval_numeric_advantage_conditioned/eval_summary.json",
        _build_eval_summary_payload(seeds=seeds, label="conditioned", advantage=1.0),
    )
    return baseline_root


def _write_task14_gate_summary(
    v2_root: Path,
    *,
    eligible: bool,
    blocking_reasons: list[str] | None = None,
    status: str | None = None,
    formal_claim_allowed: bool | None = None,
    p5_formal_10ep_eligible: bool | None = None,
) -> Path:
    reasons = list(blocking_reasons or [])
    summary_status = status or ("PASS" if eligible else "BLOCK")
    payload: dict[str, object] = {
        "artifact_kind": "preformal_gate_decision",
        "status": summary_status,
        "formal_claim_allowed": bool(eligible)
        if formal_claim_allowed is None
        else bool(formal_claim_allowed),
        "p3_formal_training_eligible": bool(eligible),
        "p3_skip_reason": None if eligible else (reasons[0] if reasons else "p5_blocked"),
        "p5_probe_eligible": bool(eligible),
        "p5_formal_10ep_eligible": bool(eligible)
        if p5_formal_10ep_eligible is None
        else bool(p5_formal_10ep_eligible),
        "p6_branch_eligible": False,
        "comparability_manifest_pass": bool(eligible),
        "route_freeze_ok": True,
        "blocking_reasons": reasons,
    }
    return _write_json(
        v2_root
        / "p4_loss_action_subgoal"
        / ROLLOUT35A.FULL_UPDATE_DIAGNOSTIC_SUMMARY_FILENAME,
        payload,
    )


def _write_task14_checkpoint_lane_root(tmp_path: Path, lane_name: str) -> Path:
    lane_root = (
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
        / lane_name
    )
    checkpoint_dir = lane_root / "formal_run" / "checkpoint-200"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"test")
    return lane_root


def _write_task15_report_authority(
    tmp_path: Path,
    *,
    p5_gate_mode: str = "skipped",
    p5_status: str = "SKIPPED",
    p5_blocker_reason: str = "p5_formal_10ep_ineligible",
    p5_blocking_reasons: list[str] | None = None,
) -> tuple[Path, Path]:
    v2_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    metadata_root = v2_root / "p1_one_step" / "repo_local_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        metadata_root / "full_update_scope_audit_dynamic.json",
        {
            "artifact_kind": "recap_full_update_scope_audit_dynamic",
            "resolution_status": "PASS",
            "train_scope_effective": "strict_full",
            "attempt_count": 1,
            "strict_full_runtime_attempted": True,
            "output_dir": str(v2_root / "p1_one_step"),
        },
    )
    _write_json(
        metadata_root / "full_update_scope_audit.json",
        {
            "artifact_kind": "recap_full_update_scope_audit",
            "train_scope_requested": "strict_full",
            "static_verdict": "PASS",
            "scope_faithfulness": "full_update_equivalent",
            "total_parameter_rows": 128,
            "trainable_parameter_rows": 128,
            "summary": {
                "total_parameter_rows": 128,
                "trainable_parameter_rows": 128,
            },
            "memory_feasibility": {
                "estimated_total_bytes": 1073741824,
                "trainable_numel": 2048,
            },
            "parameter_coverage": {
                "total_parameter_rows": 128,
                "trainable_parameter_rows": 128,
            },
            "method_faithfulness": {
                "paper_equivalent": True,
                "paper_method_gap": [],
            },
        },
    )
    _write_json(
        metadata_root / "first_backward_grad_probe_rank0.json",
        {
            "scopes": {
                "advantage_embedding": {"grad_l2_norm": 0.01},
                "diffusion_trunk": {"grad_l2_norm": 0.1},
            }
        },
    )
    _write_json(
        metadata_root / "first_optimizer_step_param_delta_rank0.json",
        {
            "scopes": {
                "advantage_embedding": {"delta_l2_norm": 1.0e-4},
                "diffusion_trunk": {"delta_l2_norm": 2.5e-1},
            }
        },
    )
    _write_json(
        metadata_root / "optimizer_state_rank0.json",
        {
            "optimizer_class_name": "AdamW",
            "param_group_count": 1,
            "param_groups_preview": [
                {"lr": 1e-4, "weight_decay": 0.01, "params": ["p0", "p1"]}
            ],
        },
    )
    (metadata_root / "downgrade_attempts.jsonl").write_text(
        json.dumps(
            {
                "candidate_scope": "strict_full",
                "status": "PASS",
                "runtime_returncode": 0,
                "runtime_preflight_status": "PASS",
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        metadata_root / "version_surface.json",
        {
            "route_freeze": {
                "frozen": True,
                "route": POLICY.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
                "diagnostic_only": True,
            }
        },
    )

    _write_json(
        v2_root / "p2_full_update_overfit20" / "conditioning_functional_probe_step20.json",
        {
            "loss_sensitivity_gate_pass": True,
            "train_subset_loss_probe": {"loss_span": 0.25},
            "heldout_loss_probe": {"loss_span": 0.2},
        },
    )
    _write_json(
        v2_root / "p2_full_update_overfit20" / "paired_action_probe_step20.json",
        {
            "action_sensitivity_gate_pass": True,
            "action_delta_layers": {
                "raw_normalized_action_delta": {
                    "available": True,
                    "delta_l2": 0.11,
                },
                "decoded_action_delta": {
                    "available": True,
                    "delta_l2": 0.12,
                },
                "controller_input_delta": {
                    "available": False,
                    "reason": "controller_input_seam_not_available",
                },
            },
        },
    )
    _write_json(
        v2_root / "p2_5_label_semantics" / "label_semantics_audit.json",
        {
            "label_semantics_gate_pass": False,
            "positive_success_rate": 0.3,
            "blocking_reasons": ["label_semantics_gate_block"],
            "shuffled_advantage_negative_control": {
                "negative_control_pass": False,
                "per_subset": {
                    "train_subset": {"shuffled_minus_true_loss": 0.0},
                    "heldout": {"shuffled_minus_true_loss": 0.0},
                },
            },
        },
    )
    _write_json(
        v2_root / "p2_5_label_semantics" / "preformal_gate_decision.json",
        {
            "artifact_kind": "preformal_gate_decision",
            "p3_formal_training_eligible": False,
            "p3_skip_reason": "label_semantics_gate_block",
            "blocking_reasons": ["label_semantics_gate_block"],
        },
    )
    _write_json(
        v2_root / "p4_loss_action_subgoal" / ROLLOUT35A.FULL_UPDATE_DIAGNOSTIC_SUMMARY_FILENAME,
        {
            "artifact_kind": "full_update_diagnostic_summary",
            "status": "BLOCK",
            "loss_sensitivity_gate_pass": True,
            "blocking_reasons": [
                "paired_action_instrumentation_incomplete",
                "label_semantics_gate_block",
            ],
            "loss_probe": {
                "status": "PASS",
                "loss_sensitivity_gate_pass": True,
            },
            "paired_action_probe": {
                "status": "BLOCK",
                "instrumentation_incomplete": True,
            },
            "first_subgoal_probe": {
                "status": "BLOCK",
                "strong_subgoal_progress_gate_pass": False,
                "paired_seed_improvement_count": 0,
                "mean_relative_improvement_min_dist_ee_to_apple": 0.0,
                "no_regression_on_contact_or_lift_proxy": False,
                "blocking_reasons": ["comparability_manifest_block"],
            },
        },
    )

    blocking_reasons = list(p5_blocking_reasons or ["paired_action_instrumentation_incomplete"])
    _write_json(
        v2_root / "p5_gate_eval" / "min_loop_verdict.json",
        {
            "artifact_kind": "recap_min_loop_comparative_verdict",
            "status": p5_status,
            "gate_mode": p5_gate_mode,
            "formal_execution_attempted": False,
            "p5_formal_10ep_eligible": False,
            "seed_set_source": "inherit_from_v1",
            "blocker_reason": p5_blocker_reason,
            "blocking_reasons": blocking_reasons,
        },
    )
    _write_json(
        v2_root / "p5_gate_eval" / "p5_gate_blocker_summary.json",
        {
            "artifact_kind": "full_update_p5_gate_blocker_summary",
            "gate_summary_status": "BLOCK",
            "blocking_reasons": blocking_reasons,
            "blocker_reason": p5_blocker_reason,
        },
    )

    output_dir = tmp_path / "report_output"
    return v2_root, output_dir


def _write_task13_lane_root(
    tmp_path: Path,
    lane_name: str,
    *,
    comparability_manifest: dict[str, object] | None = None,
    skip_manifest: dict[str, object] | None = None,
    first_subgoal_probe: dict[str, object] | None = None,
) -> Path:
    lane_root = (
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
        / lane_name
    )
    formal_root = lane_root / "formal_run"
    formal_root.mkdir(parents=True, exist_ok=True)
    if comparability_manifest is not None:
        _write_json(
            formal_root / FINETUNE_FULL.COMPARABILITY_MANIFEST_FILENAME,
            comparability_manifest,
        )
        checkpoint_dir = formal_root / "checkpoint-200"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "model.safetensors").write_bytes(b"test")
    if skip_manifest is not None:
        _write_json(formal_root / "formal_run_skipped.json", skip_manifest)
    if first_subgoal_probe is not None:
        _write_json(formal_root / "first_subgoal_probe.json", first_subgoal_probe)
    return lane_root


def _valid_first_subgoal_probe(
    *,
    seeds: list[int],
    distances: list[float],
    contact_or_lift: list[float],
) -> dict[str, object]:
    return {
        "status": "PASS",
        "seed_metrics": [
            {
                "seed": seed,
                "min_dist_ee_to_apple": distance,
                "contact_or_lift_proxy": contact,
            }
            for seed, distance, contact in zip(
                seeds,
                distances,
                contact_or_lift,
                strict=True,
            )
        ],
    }


def _make_task13_environment(
    tmp_path: Path,
    *,
    conditioning_probe: dict[str, object] | None,
    paired_probe: dict[str, object] | None,
    label_semantics_audit: dict[str, object] | None,
    conditioned_manifest: dict[str, object] | None,
    continuation_manifest: dict[str, object] | None,
    conditioned_first_subgoal: dict[str, object] | None,
    continuation_first_subgoal: dict[str, object] | None,
    baseline_first_subgoal: dict[str, object] | None,
) -> tuple[Path, Path, Path, Path, Path]:
    baseline_root = _write_task13_baseline_authority(tmp_path)
    if baseline_first_subgoal is not None:
        _write_json(
            baseline_root / "first_subgoal_probe.json",
            baseline_first_subgoal,
        )
    _write_task13_scope_and_route_authority(tmp_path)
    _write_task13_task8_artifacts(
        tmp_path,
        conditioning_probe=conditioning_probe,
        paired_probe=paired_probe,
        label_semantics_audit=label_semantics_audit,
    )
    conditioned_root = _write_task13_lane_root(
        tmp_path,
        "t13_advantage_full_update_1gpu",
        comparability_manifest=conditioned_manifest,
        first_subgoal_probe=conditioned_first_subgoal,
    )
    continuation_root = _write_task13_lane_root(
        tmp_path,
        "t13_continuation_full_update_1gpu",
        comparability_manifest=continuation_manifest,
        first_subgoal_probe=continuation_first_subgoal,
    )
    output_dir = (
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/p4_loss_action_subgoal"
    )
    return (
        baseline_root,
        tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update",
        conditioned_root,
        continuation_root,
        output_dir,
    )


def test_p4_missing_probe_block(tmp_path: Path) -> None:
    conditioned_manifest = _build_conditioned_manifest(tmp_path)
    continuation_manifest = _build_continuation_manifest(tmp_path)
    seeds = [20260421, 20260422, 20260423]
    baseline_root, v2_root, conditioned_root, continuation_root, output_dir = (
        _make_task13_environment(
            tmp_path,
            conditioning_probe=None,
            paired_probe=_paired_probe_payload(),
            label_semantics_audit=_label_semantics_payload(),
            conditioned_manifest=conditioned_manifest,
            continuation_manifest=continuation_manifest,
            conditioned_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.30, 0.29, 0.28],
                contact_or_lift=[0.1, 0.1, 0.1],
            ),
            continuation_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.32, 0.31, 0.30],
                contact_or_lift=[0.1, 0.1, 0.1],
            ),
            baseline_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.33, 0.32, 0.31],
                contact_or_lift=[0.1, 0.1, 0.1],
            ),
        )
    )

    result = ROLLOUT35A.run_p4_diagnostics(
        baseline_authority_root=baseline_root,
        v2_authority_root=v2_root,
        conditioned_run_root=conditioned_root,
        continuation_run_root=continuation_root,
        output_dir=output_dir,
    )

    summary = json.loads(
        Path(result["full_update_diagnostic_summary_path"]).read_text(encoding="utf-8")
    )
    assert summary["status"] == "BLOCK"
    assert summary["loss_probe"]["status"] == "BLOCK"
    assert "missing_train_subset_loss_probe" in summary["loss_probe"]["blocking_reasons"]


def test_p4_comparability_block(tmp_path: Path) -> None:
    conditioned_manifest = _build_conditioned_manifest(tmp_path)
    continuation_manifest = _build_continuation_manifest(
        tmp_path,
        policy_route=POLICY.MAINLINE_RUNTIME_ROUTE,
        policy_indicator_mode="omit",
    )
    seeds = [20260421, 20260422, 20260423]
    baseline_root, v2_root, conditioned_root, continuation_root, output_dir = (
        _make_task13_environment(
            tmp_path,
            conditioning_probe=_conditioning_probe_payload(),
            paired_probe=_paired_probe_payload(),
            label_semantics_audit=_label_semantics_payload(),
            conditioned_manifest=conditioned_manifest,
            continuation_manifest=continuation_manifest,
            conditioned_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.30, 0.29, 0.28],
                contact_or_lift=[0.1, 0.1, 0.1],
            ),
            continuation_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.32, 0.31, 0.30],
                contact_or_lift=[0.1, 0.1, 0.1],
            ),
            baseline_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.33, 0.32, 0.31],
                contact_or_lift=[0.1, 0.1, 0.1],
            ),
        )
    )

    result = ROLLOUT35A.run_p4_diagnostics(
        baseline_authority_root=baseline_root,
        v2_authority_root=v2_root,
        conditioned_run_root=conditioned_root,
        continuation_run_root=continuation_root,
        output_dir=output_dir,
    )

    summary = json.loads(
        Path(result["full_update_diagnostic_summary_path"]).read_text(encoding="utf-8")
    )
    assert summary["status"] == "BLOCK"
    assert summary["artifact_kind"] == "full_update_diagnostic_summary"
    assert summary["schema_version"] == ROLLOUT35A.SCHEMA_VERSION
    assert summary["comparability_manifest_pass"] is False
    assert summary["comparability_blocker_reason"] == "route_mismatch_block"
    assert summary["full_update_scope_gate_pass"] is True
    assert (
        summary["scope_gate_path"]
        == str(v2_root / "p1_one_step/repo_local_metadata/full_update_scope_audit_dynamic.json")
    )
    assert summary["scope_gate_resolution_status"] == "PASS"
    assert summary["scope_gate_best_scope_authority"] is True
    assert "full_update_scope_gate_path" not in summary
    assert "full_update_scope_gate_reason" not in summary


def test_weak_distance_only_blocks_formal_p4(tmp_path: Path) -> None:
    conditioned_manifest = _build_conditioned_manifest(tmp_path)
    continuation_manifest = _build_continuation_manifest(tmp_path)
    seeds = [20260421, 20260422, 20260423]
    baseline_root, v2_root, conditioned_root, continuation_root, output_dir = (
        _make_task13_environment(
            tmp_path,
            conditioning_probe=_conditioning_probe_payload(),
            paired_probe=_paired_probe_payload(),
            label_semantics_audit=_label_semantics_payload(),
            conditioned_manifest=conditioned_manifest,
            continuation_manifest=continuation_manifest,
            conditioned_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.24, 0.23, 0.22],
                contact_or_lift=[0.0, 0.0, 0.0],
            ),
            continuation_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.30, 0.29, 0.28],
                contact_or_lift=[0.0, 0.0, 0.0],
            ),
            baseline_first_subgoal=_valid_first_subgoal_probe(
                seeds=seeds,
                distances=[0.32, 0.31, 0.30],
                contact_or_lift=[0.0, 0.0, 0.0],
            ),
        )
    )

    result = ROLLOUT35A.run_p4_diagnostics(
        baseline_authority_root=baseline_root,
        v2_authority_root=v2_root,
        conditioned_run_root=conditioned_root,
        continuation_run_root=continuation_root,
        output_dir=output_dir,
    )

    summary = json.loads(
        Path(result["full_update_diagnostic_summary_path"]).read_text(encoding="utf-8")
    )
    assert summary["status"] == "BLOCK"
    assert summary["first_subgoal_probe"]["status"] == "BLOCK"
    assert summary["strong_subgoal_progress_gate_pass"] is False
    assert summary["weak_distance_only"] is True
    assert "contact_or_lift_proxy_uninformative_all_zero" in summary["blocking_reasons"]


def _run_supervisor(
    tmp_path: Path,
    monkeypatch,
    *,
    requested_scope: str,
    static_by_scope: dict[str, dict[str, object]],
    preflight_by_scope: dict[str, dict[str, object]],
    runtime_by_scope: dict[str, tuple[int, dict[str, object]]],
    arg_overrides: dict[str, object] | None = None,
) -> tuple[int, Path]:
    output_dir = tmp_path / "authority_out"
    fake_smoke = _FakeSmokeModule(
        output_dir=output_dir,
        static_by_scope=static_by_scope,
        preflight_by_scope=preflight_by_scope,
        runtime_by_scope=runtime_by_scope,
    )
    monkeypatch.setattr(SUPERVISOR, "_load_smoke_module", lambda: fake_smoke)
    monkeypatch.setattr(SUPERVISOR, "_repo_root", lambda: tmp_path)
    rc = SUPERVISOR.run_numeric_adv_scope_supervisor(
        _make_args(
            tmp_path,
            requested_scope,
            **({} if arg_overrides is None else arg_overrides),
        )
    )
    return rc, output_dir


def _write_task11_best_scope_authority(
    tmp_path: Path,
    *,
    effective_scope: str,
) -> Path:
    path = (
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/p1_one_step/repo_local_metadata/full_update_scope_audit_dynamic.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "recap_full_update_scope_audit_dynamic",
                "best_scope_authority": True,
                "train_scope_effective": effective_scope,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_task11_gate_summary(
    tmp_path: Path,
    *,
    eligible: bool,
    skip_reason: str | None,
) -> Path:
    conditioning_probe_path = (
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/p2_full_update_overfit20/conditioning_functional_probe_step20.json"
    )
    conditioning_probe_path.parent.mkdir(parents=True, exist_ok=True)
    conditioning_probe_path.write_text(
        json.dumps({"status": "ok"}, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path = (
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/p2_5_label_semantics/preformal_gate_decision.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    blocking_reasons = [] if eligible else [str(skip_reason or "p3_formal_training_ineligible")]
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "preformal_gate_decision",
                "conditioning_probe_path": str(conditioning_probe_path),
                "p3_formal_training_eligible": bool(eligible),
                "p3_skip_reason": None if eligible else str(skip_reason),
                "blocking_reasons": blocking_reasons,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_task11_conditioned_peer_manifest(
    tmp_path: Path,
    *,
    warm_start_checkpoint: str | Path,
) -> Path:
    output_dir = (
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/t13_advantage_full_update_1gpu/formal_run"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    FINETUNE_FULL.emit_conditioned_formal_lane_comparability_manifest(
        repo_root=tmp_path,
        output_dir=output_dir,
        warm_start_checkpoint=warm_start_checkpoint,
        global_batch_size=4,
        gradient_accumulation_steps=4,
        num_gpus=1,
        dataset_path="agent/artifacts/lerobot_datasets/recap_stage3_iter_002",
        train_scope_requested="strict_full",
        train_scope_effective="strict_full",
    )
    return output_dir / FINETUNE_FULL.COMPARABILITY_MANIFEST_FILENAME


def _build_conditioned_manifest(
    tmp_path: Path,
    *,
    warm_start_checkpoint: str | Path = "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/checkpoint-200",
    policy_route: object | None = None,
    policy_indicator_mode: object | None = None,
    seed_bundle_path: str | Path | None = None,
) -> dict[str, object]:
    output_dir = tmp_path / "conditioned"
    return FINETUNE_FULL.emit_full_update_comparability_manifest(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        warm_start_checkpoint=warm_start_checkpoint,
        global_batch_size=4,
        gradient_accumulation_steps=4,
        num_gpus=1,
        dataset_path="agent/artifacts/lerobot_datasets/recap_stage3_iter_002",
        launch_family="single_gpu_v1",
        train_scope_requested="strict_full",
        train_scope_effective="full_policy",
        advantage_consumed=True,
        seed_bundle_path=seed_bundle_path,
        policy_route=policy_route,
        policy_indicator_mode=policy_indicator_mode,
    )


def _build_continuation_manifest(
    tmp_path: Path,
    *,
    continuation_checkpoint: str | Path = "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/checkpoint-200",
    policy_route: object | None = None,
    policy_indicator_mode: object | None = None,
    seed_bundle_path: str | Path | None = None,
) -> dict[str, object]:
    output_dir = tmp_path / "continuation"
    return CONTINUATION_CONTROL.build_continuation_comparability_manifest(
        output_dir=output_dir,
        continuation_checkpoint=continuation_checkpoint,
        dataset_path="agent/artifacts/lerobot_datasets/recap_stage3_iter_002",
        global_batch_size=4,
        gradient_accumulation_steps=4,
        num_gpus=1,
        train_scope_requested="strict_full",
        train_scope_effective="full_policy",
        launch_family="single_gpu_v1",
        seed_bundle_path=seed_bundle_path,
        policy_route=policy_route,
        policy_indicator_mode=policy_indicator_mode,
    )


def test_comparability_manifest_freezes_shared_factors(tmp_path: Path) -> None:
    conditioned = _build_conditioned_manifest(tmp_path)
    continuation = _build_continuation_manifest(tmp_path)

    conditioned_path = tmp_path / "conditioned" / FINETUNE_FULL.COMPARABILITY_MANIFEST_FILENAME
    continuation_path = tmp_path / "continuation" / FINETUNE_FULL.COMPARABILITY_MANIFEST_FILENAME
    assert conditioned_path.is_file()
    assert continuation_path.is_file()
    assert json.loads(conditioned_path.read_text(encoding="utf-8")) == conditioned
    assert json.loads(continuation_path.read_text(encoding="utf-8")) == continuation

    for field_name in FINETUNE_FULL.COMPARABILITY_SHARED_FIELDS:
        assert conditioned[field_name] == continuation[field_name]

    assert conditioned["policy_route"] == POLICY.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE
    assert conditioned["policy_route"] == continuation["policy_route"]
    assert conditioned["seed_set_source"] == "inherit_from_v1"
    assert continuation["seed_set_source"] == "inherit_from_v1"
    assert conditioned["advantage_consumed"] is True
    assert continuation["advantage_consumed"] is False

    verdict = FINETUNE_FULL.validate_full_update_comparability_manifests(
        conditioned,
        continuation,
    )
    assert verdict["status"] == "pass"
    assert set(verdict["differing_fields"]) == {
        "advantage_consumed",
        "output_dir",
        "comparability_manifest_path",
    }
    assert verdict["unexpected_diff_fields"] == []


def test_route_mismatch_block(tmp_path: Path) -> None:
    conditioned = _build_conditioned_manifest(tmp_path)
    continuation = _build_continuation_manifest(
        tmp_path,
        policy_route=POLICY.MAINLINE_RUNTIME_ROUTE,
        policy_indicator_mode="omit",
    )

    verdict = FINETUNE_FULL.validate_full_update_comparability_manifests(
        conditioned,
        continuation,
    )

    assert verdict["status"] == "blocked"
    assert verdict["blocker_code"] == "route_mismatch_block"


def test_warm_start_mismatch_block(tmp_path: Path) -> None:
    conditioned = _build_conditioned_manifest(tmp_path)
    continuation = _build_continuation_manifest(
        tmp_path,
        continuation_checkpoint="agent/artifacts/stage3_t10_baseline_continuation_control_1gpu/formal_run/checkpoint-200",
    )

    verdict = FINETUNE_FULL.validate_full_update_comparability_manifests(
        conditioned,
        continuation,
    )

    assert verdict["status"] == "blocked"
    assert verdict["blocker_code"] == "warm_start_mismatch_block"


def test_seed_bundle_missing_block(tmp_path: Path) -> None:
    missing_seed_bundle = tmp_path / "missing_eval_seed_set.json"
    conditioned = _build_conditioned_manifest(
        tmp_path,
        seed_bundle_path=missing_seed_bundle,
    )
    continuation = _build_continuation_manifest(tmp_path)

    assert conditioned["status"] == "blocked"
    assert conditioned["blocker_code"] == "seed_bundle_missing_block"
    assert conditioned["seed_set"] is None
    assert not (tmp_path / "conditioned" / "new_v2_bundle").exists()

    verdict = FINETUNE_FULL.validate_full_update_comparability_manifests(
        conditioned,
        continuation,
    )
    assert verdict["status"] == "blocked"
    assert verdict["blocker_code"] == "seed_bundle_missing_block"


def test_runtime_preflight_block(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={"strict_full": _static_audit()},
        preflight_by_scope={
            "strict_full": {
                "status": "BLOCK",
                "memory_feasibility_estimate": {"risk": "LOW"},
                "hard_block_reasons": ["flash_attn_2_unavailable"],
            }
        },
        runtime_by_scope={},
    )

    assert rc == 1
    attempts = _read_attempts(output_dir)
    dynamic_audit = _read_dynamic_audit(output_dir)
    assert attempts[0]["status"] == "BLOCK"
    assert attempts[0]["runtime_attempted"] is False
    assert attempts[0]["runtime_preflight_status"] == "BLOCK"
    assert attempts[0]["runtime_preflight_block_reasons"] == ["flash_attn_2_unavailable"]
    assert dynamic_audit["failure_reason"] == "RUNTIME_PREFLIGHT_BLOCK"


def test_flash_attn_block(tmp_path: Path, monkeypatch) -> None:
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        LAUNCH,
        "probe_repo_local_transformers_flash_attn",
        lambda: {
            "transformers_flash_attn_import_ok": True,
            "flash_attn_2_available": False,
        },
    )
    monkeypatch.setattr(
        LAUNCH,
        "probe_repo_local_nvidia_smi_gpu_snapshot",
        lambda gpu_index=1: {
            "ok": True,
            "gpu_index": gpu_index,
            "row": {"memory_total_mb": "1000", "memory_used_mb": "100"},
            "available_memory_bytes": 900 * 1024 * 1024,
        },
    )

    payload = LAUNCH.build_repo_local_runtime_preflight_payload(
        output_dir=tmp_path / "authority_out",
        python_path=python_path,
        requested_num_gpus=1,
        requested_scope="full_action",
        static_audit={"parameter_coverage": {"parameter_rows": []}},
        torch_module=_FakeTorch(),
    )

    assert payload["status"] == "BLOCK"
    assert payload["flash_attn_2_available"] is False
    assert payload["transformers_flash_attn_import_ok"] is True
    assert payload["cuda_visible_devices_expected"] == "1"
    assert payload["hard_block_reasons"] == ["flash_attn_2_unavailable"]


def test_venv_symlink_block(tmp_path: Path, monkeypatch) -> None:
    broken_python = tmp_path / "broken_python"
    os.symlink(tmp_path / "missing_python_target", broken_python)
    monkeypatch.setattr(
        LAUNCH,
        "probe_repo_local_transformers_flash_attn",
        lambda: {
            "transformers_flash_attn_import_ok": True,
            "flash_attn_2_available": True,
        },
    )
    monkeypatch.setattr(
        LAUNCH,
        "probe_repo_local_nvidia_smi_gpu_snapshot",
        lambda gpu_index=1: {
            "ok": True,
            "gpu_index": gpu_index,
            "row": {"memory_total_mb": "1000", "memory_used_mb": "100"},
            "available_memory_bytes": 900 * 1024 * 1024,
        },
    )

    payload = LAUNCH.build_repo_local_runtime_preflight_payload(
        output_dir=tmp_path / "authority_out",
        python_path=broken_python,
        requested_num_gpus=1,
        requested_scope="full_action",
        static_audit={"parameter_coverage": {"parameter_rows": []}},
        torch_module=_FakeTorch(),
    )

    assert payload["venv_symlink_valid"] is False
    assert payload["status"] == "BLOCK"
    assert payload["hard_block_reasons"] == ["venv_symlink_invalid"]


def test_memory_estimator_block(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={
            "strict_full": _static_audit(fits_available_memory=True),
            "full_policy": _static_audit(fits_available_memory=True),
        },
        preflight_by_scope={
            "strict_full": {
                "status": "PASS",
                "strict_full_runtime_skipped_reason": "memory_estimator_block",
                "memory_feasibility_estimate": {"risk": "BLOCK"},
            },
            "full_policy": {
                "status": "PASS",
                "memory_feasibility_estimate": {"risk": "LOW"},
            },
        },
        runtime_by_scope={
            "full_policy": (0, _runtime_payload(tmp_path / "authority_out")),
        },
    )

    assert rc == 0
    attempts = _read_attempts(output_dir)
    dynamic_audit = _read_dynamic_audit(output_dir)
    assert [attempt["status"] for attempt in attempts] == [
        "MEMORY_ESTIMATOR_BLOCK",
        "PASS",
    ]
    assert attempts[0]["runtime_attempted"] is False
    assert attempts[0]["strict_full_runtime_skipped_reason"] == "memory_estimator_block"
    assert dynamic_audit["resolution_status"] == "DEGRADE"
    assert dynamic_audit["train_scope_effective"] == "full_policy"


def test_p3_scope_drift_block(tmp_path: Path, monkeypatch) -> None:
    _write_task11_best_scope_authority(tmp_path, effective_scope="strict_full")
    _write_task11_gate_summary(
        tmp_path,
        eligible=True,
        skip_reason=None,
    )

    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="full_policy",
        static_by_scope={},
        preflight_by_scope={},
        runtime_by_scope={},
        arg_overrides={
            "entrypoint": "conditioned",
            "require_p3_formal_eligible": True,
        },
    )

    assert rc == 1
    dynamic_audit = _read_dynamic_audit(output_dir)
    assert dynamic_audit["failure_reason"] == "P3_SCOPE_DRIFT_BLOCK"
    assert dynamic_audit["best_scope_authority_requested"] == "strict_full"


def test_continuation_parity_block(tmp_path: Path, monkeypatch) -> None:
    _write_task11_best_scope_authority(tmp_path, effective_scope="strict_full")
    gate_summary_path = _write_task11_gate_summary(
        tmp_path,
        eligible=True,
        skip_reason=None,
    )
    warm_start_checkpoint = gate_summary_path.parent.parent / "p2_full_update_overfit20/checkpoint-20"
    warm_start_checkpoint.mkdir(parents=True, exist_ok=True)
    (warm_start_checkpoint / "model.safetensors").write_bytes(b"test-warm-start")
    _write_task11_conditioned_peer_manifest(
        tmp_path,
        warm_start_checkpoint="agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/checkpoint-200",
    )

    def _fake_continuation_runtime(args: SimpleNamespace, *, requested_scope_override: str):
        del args, requested_scope_override
        return 0, _runtime_payload(tmp_path / "authority_out")

    monkeypatch.setattr(
        SUPERVISOR,
        "_run_continuation_single_scope",
        _fake_continuation_runtime,
    )

    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={"strict_full": _static_audit()},
        preflight_by_scope={"strict_full": {"status": "PASS"}},
        runtime_by_scope={},
        arg_overrides={
            "entrypoint": "continuation",
            "require_p3_formal_eligible": True,
            "dataset_path": "agent/artifacts/lerobot_datasets/recap_stage3_iter_002",
            "global_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_gpus": 1,
        },
    )

    assert rc == 1
    attempts = _read_attempts(output_dir)
    dynamic_audit = _read_dynamic_audit(output_dir)
    assert attempts[0]["failure_reason"] == "CONTINUATION_PARITY_BLOCK"
    assert dynamic_audit["failure_reason"] == "CONTINUATION_PARITY_BLOCK"
    assert dynamic_audit["formal_entrypoint"] == "continuation"


@pytest.mark.parametrize(
    ("entrypoint", "expected_artifact_kind"),
    [
        ("conditioned", "conditioned_formal_run_skipped"),
        ("continuation", "continuation_formal_run_skipped"),
    ],
)
def test_p3_ineligible_skip(
    tmp_path: Path,
    monkeypatch,
    entrypoint: str,
    expected_artifact_kind: str,
) -> None:
    _write_task11_best_scope_authority(tmp_path, effective_scope="strict_full")
    _write_task14_baseline_authority(
        tmp_path,
        seeds=list(range(20260421, 20260431)),
    )
    _write_task11_gate_summary(
        tmp_path,
        eligible=False,
        skip_reason="label_semantics_gate_block",
    )
    warm_start_checkpoint = (
        tmp_path
        / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/p2_full_update_overfit20/checkpoint-20"
    )
    warm_start_checkpoint.mkdir(parents=True, exist_ok=True)
    (warm_start_checkpoint / "model.safetensors").write_bytes(b"test-warm-start")

    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={},
        preflight_by_scope={},
        runtime_by_scope={},
        arg_overrides={
            "entrypoint": entrypoint,
            "require_p3_formal_eligible": True,
            "dataset_path": "agent/artifacts/lerobot_datasets/recap_stage3_iter_002",
            "global_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_gpus": 1,
        },
    )

    assert rc == 0
    comparability_manifest_path = output_dir / FINETUNE_FULL.COMPARABILITY_MANIFEST_FILENAME
    comparability_manifest = json.loads(
        comparability_manifest_path.read_text(encoding="utf-8")
    )
    skipped_manifest = json.loads(
        (output_dir / "formal_run_skipped.json").read_text(encoding="utf-8")
    )
    assert comparability_manifest["status"] == "ok"
    assert comparability_manifest["seed_set_source"] == "inherit_from_v1"
    assert comparability_manifest["advantage_consumed"] is (entrypoint == "conditioned")
    assert skipped_manifest["status"] == "skipped"
    assert skipped_manifest["artifact_kind"] == expected_artifact_kind
    assert skipped_manifest["p3_formal_training_eligible"] is False
    assert skipped_manifest["p3_skip_reason"] == "label_semantics_gate_block"
    assert skipped_manifest["comparability_manifest_path"] == str(
        comparability_manifest_path
    )


def test_missing_action_delta_layers_marks_instrumentation_incomplete() -> None:
    verdict = SMOKE34B._evaluate_paired_action_probe_contract(
        {
            "action_delta_layers": {
                "raw_normalized_action_delta": {
                    "available": True,
                    "delta_l2": 0.1,
                },
                "decoded_action_delta": {
                    "available": True,
                    "delta_l2": 0.2,
                },
                "controller_input_delta": {
                    "available": False,
                    "reason": "missing_local_seam",
                },
            }
        }
    )

    assert verdict["instrumentation_incomplete"] is True
    assert verdict["action_sensitivity_gate_pass"] is False
    assert verdict["missing_layers"] == ["postprocessed_action_delta"]
    assert "controller_input_delta" in verdict["unavailable_layers"]


def test_label_semantics_gate_blocks_without_negative_control_pass() -> None:
    payload = {
        "value_is_constant": False,
        "all_returns_negative": False,
        "positive_subgoal_evidence_available": True,
        "shuffled_advantage_negative_control": {
            "negative_control_pass": False,
            "blocking_reasons": ["heldout_shuffled_control_not_worse_than_true"],
        },
    }

    verdict = SMOKE34B._evaluate_label_semantics_gate(payload)

    assert verdict["label_semantics_gate_pass"] is False
    assert verdict["formal_claim_allowed"] is False
    assert "shuffled_advantage_negative_control_failed" in verdict["blocking_reasons"]


def test_shuffled_negative_control_requires_true_labels_to_outperform_shuffle() -> None:
    verdict = SMOKE34B._evaluate_shuffled_negative_control(
        {
            "per_subset": {
                "train_subset": {
                    "shuffled_minus_true_loss": 0.05,
                },
                "heldout": {
                    "shuffled_minus_true_loss": -0.01,
                },
            }
        }
    )

    assert verdict["negative_control_pass"] is False
    assert verdict["blocking_reasons"] == [
        "heldout_shuffled_control_not_worse_than_true"
    ]


def test_diagnostic_gate_contract_reports_machine_readable_statuses(
    tmp_path: Path,
) -> None:
    verdict = SMOKE34B._task8_build_preformal_gate_decision(
        conditioning_probe=_conditioning_probe_payload(),
        paired_probe=_paired_probe_payload(),
        label_semantics_audit=_label_semantics_payload(),
        output_dir=tmp_path / "p2_full_update_overfit20",
        label_semantics_output_dir=tmp_path / "p2_5_label_semantics",
        full_update_scope_gate_pass=True,
        comparability_manifest_pass=True,
        comparability_blocker_reason=None,
    )

    assert verdict["loss_probe"]["status"] == "PASS"
    assert verdict["paired_action_probe"]["status"] == "PASS"
    assert verdict["first_subgoal_probe"]["status"] == "SKIPPED"
    assert verdict["full_update_scope_gate_pass"] is True
    assert verdict["comparability_manifest_pass"] is True
    assert verdict["instrumentation_incomplete"] is False
    assert verdict["label_semantics_gate_pass"] is True
    assert verdict["shuffled_advantage_negative_control_pass"] is True
    assert verdict["p3_formal_training_eligible"] is True
    assert verdict["p3_skip_reason"] is None
    assert verdict["p5_probe_eligible"] is True
    assert verdict["p5_formal_10ep_eligible"] is True
    assert verdict["p6_branch_eligible"] is False
    assert verdict["routing_decision"] == "route_p3_formal_training"
    assert verdict["routing_reasons"] == []


def test_p3_gate_requires_scope_instrumentation_and_negative_control(
    tmp_path: Path,
) -> None:
    verdict = SMOKE34B._task8_build_preformal_gate_decision(
        conditioning_probe=_conditioning_probe_payload(),
        paired_probe=_paired_probe_payload(),
        label_semantics_audit=_label_semantics_payload(
            shuffled_advantage_negative_control_pass=False,
        ),
        output_dir=tmp_path / "p2_full_update_overfit20",
        label_semantics_output_dir=tmp_path / "p2_5_label_semantics",
        full_update_scope_gate_pass=True,
        comparability_manifest_pass=True,
        comparability_blocker_reason=None,
    )

    assert verdict["p3_formal_training_eligible"] is False
    assert verdict["p3_skip_reason"] == "shuffled_advantage_negative_control_block"
    assert verdict["p5_probe_eligible"] is True
    assert verdict["p5_formal_10ep_eligible"] is False
    assert verdict["routing_decision"] == "route_p6_semantic_branch"
    assert "shuffled_advantage_negative_control_block" in verdict["routing_reasons"]


def test_p5_formal_gate_requires_comparability_even_when_p3_is_green(
    tmp_path: Path,
) -> None:
    verdict = SMOKE34B._task8_build_preformal_gate_decision(
        conditioning_probe=_conditioning_probe_payload(),
        paired_probe=_paired_probe_payload(),
        label_semantics_audit=_label_semantics_payload(),
        output_dir=tmp_path / "p2_full_update_overfit20",
        label_semantics_output_dir=tmp_path / "p2_5_label_semantics",
        full_update_scope_gate_pass=True,
        comparability_manifest_pass=False,
        comparability_blocker_reason="comparability_manifest_block",
    )

    assert verdict["p3_formal_training_eligible"] is True
    assert verdict["p5_probe_eligible"] is False
    assert verdict["p5_formal_10ep_eligible"] is False
    assert verdict["p6_branch_eligible"] is False
    assert verdict["routing_decision"] == "route_p3_formal_training"
    assert "comparability_manifest_block" in verdict["routing_reasons"]


def test_p6_trigger_matrix_requires_legal_semantic_insufficiency_without_blockers(
    tmp_path: Path,
) -> None:
    verdict = SMOKE34B._task8_build_preformal_gate_decision(
        conditioning_probe=_conditioning_probe_payload(),
        paired_probe=_paired_probe_payload(),
        label_semantics_audit=_label_semantics_payload(
            label_semantics_gate_pass=False,
        ),
        output_dir=tmp_path / "p2_full_update_overfit20",
        label_semantics_output_dir=tmp_path / "p2_5_label_semantics",
        full_update_scope_gate_pass=False,
        comparability_manifest_pass=True,
        comparability_blocker_reason=None,
        continuous_numeric_advantage_dead_after_full_update=True,
    )

    assert verdict["p3_formal_training_eligible"] is False
    assert verdict["p5_probe_eligible"] is False
    assert verdict["p5_formal_10ep_eligible"] is False
    assert verdict["p6_branch_eligible"] is False
    assert verdict["routing_decision"] == "block_downstream"

    verdict = SMOKE34B._task8_build_preformal_gate_decision(
        conditioning_probe=_conditioning_probe_payload(),
        paired_probe=_paired_probe_payload(),
        label_semantics_audit=_label_semantics_payload(
            label_semantics_gate_pass=False,
        ),
        output_dir=tmp_path / "p2_full_update_overfit20_green",
        label_semantics_output_dir=tmp_path / "p2_5_label_semantics_green",
        full_update_scope_gate_pass=True,
        comparability_manifest_pass=True,
        comparability_blocker_reason=None,
        continuous_numeric_advantage_dead_after_full_update=True,
    )

    assert verdict["p3_formal_training_eligible"] is True
    assert verdict["p5_probe_eligible"] is True
    assert verdict["p5_formal_10ep_eligible"] is False
    assert verdict["p6_branch_eligible"] is True
    assert verdict["routing_decision"] == "route_p3_formal_training"
    assert "label_semantics_gate_block" in verdict["routing_reasons"]
    assert (
        "continuous_numeric_advantage_dead_after_full_update"
        in verdict["routing_reasons"]
    )


def test_instrumentation_incomplete_block_suppresses_p3_and_p6(
    tmp_path: Path,
) -> None:
    verdict = SMOKE34B._task8_build_preformal_gate_decision(
        conditioning_probe=_conditioning_probe_payload(),
        paired_probe=_paired_probe_payload(instrumentation_incomplete=True),
        label_semantics_audit=_label_semantics_payload(),
        output_dir=tmp_path / "p2_full_update_overfit20",
        label_semantics_output_dir=tmp_path / "p2_5_label_semantics",
        full_update_scope_gate_pass=True,
        comparability_manifest_pass=True,
        comparability_blocker_reason=None,
    )

    assert verdict["paired_action_probe"]["status"] == "BLOCK"
    assert verdict["instrumentation_incomplete"] is True
    assert verdict["p3_formal_training_eligible"] is False
    assert verdict["p3_skip_reason"] == "instrumentation_incomplete"
    assert verdict["p5_formal_10ep_eligible"] is False
    assert verdict["p6_branch_eligible"] is False
    assert verdict["routing_decision"] == "block_downstream"
    assert "paired_action_instrumentation_incomplete" in verdict["routing_reasons"]


def test_comparability_block_suppresses_downstream_branches_only(
    tmp_path: Path,
) -> None:
    verdict = SMOKE34B._task8_build_preformal_gate_decision(
        conditioning_probe=_conditioning_probe_payload(),
        paired_probe=_paired_probe_payload(),
        label_semantics_audit=_label_semantics_payload(
            label_semantics_gate_pass=False,
        ),
        output_dir=tmp_path / "p2_full_update_overfit20",
        label_semantics_output_dir=tmp_path / "p2_5_label_semantics",
        full_update_scope_gate_pass=True,
        comparability_manifest_pass=False,
        comparability_blocker_reason="comparability_manifest_block",
        continuous_numeric_advantage_dead_after_full_update=True,
    )

    assert verdict["p3_formal_training_eligible"] is True
    assert verdict["p5_probe_eligible"] is False
    assert verdict["p5_formal_10ep_eligible"] is False
    assert verdict["p6_branch_eligible"] is False
    assert verdict["routing_decision"] == "route_p3_formal_training"
    assert "comparability_manifest_block" in verdict["routing_reasons"]


def test_label_semantics_gate_arms_p6_without_mislabelling_p5_failure(
    tmp_path: Path,
) -> None:
    verdict = SMOKE34B._task8_build_preformal_gate_decision(
        conditioning_probe=_conditioning_probe_payload(),
        paired_probe=_paired_probe_payload(),
        label_semantics_audit=_label_semantics_payload(
            label_semantics_gate_pass=False,
        ),
        output_dir=tmp_path / "p2_full_update_overfit20",
        label_semantics_output_dir=tmp_path / "p2_5_label_semantics",
        full_update_scope_gate_pass=True,
        comparability_manifest_pass=True,
        comparability_blocker_reason=None,
    )

    assert verdict["label_semantics_gate_pass"] is False
    assert verdict["p5_probe_eligible"] is True
    assert verdict["p5_formal_10ep_eligible"] is False
    assert verdict["p6_branch_eligible"] is True
    assert verdict["routing_decision"] == "route_p3_formal_training"
    assert "label_semantics_gate_block" in verdict["routing_reasons"]


def test_p5_ineligible_skip(tmp_path: Path, monkeypatch) -> None:
    seeds = list(range(20260421, 20260431))
    baseline_root = _write_task14_baseline_authority(tmp_path, seeds=seeds)
    v2_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    _write_task14_gate_summary(
        v2_root,
        eligible=False,
        blocking_reasons=["paired_action_instrumentation_incomplete"],
    )
    conditioned_root = v2_root / "t13_advantage_full_update_1gpu"
    continuation_root = v2_root / "t13_continuation_full_update_1gpu"
    output_dir = v2_root / "p5_gate_eval"

    def _unexpected_lane_state(*args, **kwargs):
        raise AssertionError("skip path must not inspect formal lane state")

    monkeypatch.setattr(ROLLOUT35A, "_resolve_lane_state", _unexpected_lane_state)

    result = ROLLOUT35A.run_p5_gate(
        baseline_authority_root=baseline_root,
        v2_authority_root=v2_root,
        conditioned_run_root=conditioned_root,
        continuation_run_root=continuation_root,
        output_dir=output_dir,
        seed_start=20260421,
        seed_end=20260430,
    )

    verdict = json.loads(
        Path(result["min_loop_verdict_path"]).read_text(encoding="utf-8")
    )
    blocker_summary = json.loads(
        Path(result["blocker_summary_path"]).read_text(encoding="utf-8")
    )
    assert verdict["status"] == "SKIPPED"
    assert verdict["gate_mode"] == "skipped"
    assert verdict["blocker_reason"] == "p5_formal_gate_not_passed"
    assert "p5_formal_10ep_ineligible" in verdict["blocking_reasons"]
    assert verdict["seed_set_source"] == "inherit_from_v1"
    assert verdict["seed_set"] == seeds
    assert verdict["requested_seed_set"] == seeds
    assert verdict["formal_execution_attempted"] is False
    assert "paired_action_instrumentation_incomplete" in verdict["blocking_reasons"]
    assert blocker_summary["artifact_kind"] == "full_update_p5_gate_blocker_summary"


def test_p5_gate_blocks_blocked_p4_summary_even_when_raw_p5_flag_is_true(
    tmp_path: Path, monkeypatch
) -> None:
    seeds = list(range(20260421, 20260431))
    baseline_root = _write_task14_baseline_authority(tmp_path, seeds=seeds)
    v2_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    _write_task14_gate_summary(
        v2_root,
        eligible=True,
        status="BLOCK",
        formal_claim_allowed=False,
        p5_formal_10ep_eligible=True,
        blocking_reasons=["paired_seed_improvement_count_below_2_of_3"],
    )
    output_dir = v2_root / "p5_gate_eval"

    def _unexpected_lane_state(*args, **kwargs):
        raise AssertionError("blocked P4 summary must skip before formal lane lookup")

    monkeypatch.setattr(ROLLOUT35A, "_resolve_lane_state", _unexpected_lane_state)

    result = ROLLOUT35A.run_p5_gate(
        baseline_authority_root=baseline_root,
        v2_authority_root=v2_root,
        conditioned_run_root=v2_root / "t13_advantage_full_update_1gpu",
        continuation_run_root=v2_root / "t13_continuation_full_update_1gpu",
        output_dir=output_dir,
        seed_start=20260421,
        seed_end=20260430,
    )

    verdict = json.loads(
        Path(result["min_loop_verdict_path"]).read_text(encoding="utf-8")
    )
    assert verdict["status"] == "SKIPPED"
    assert verdict["gate_mode"] == "skipped"
    assert verdict["blocker_reason"] == "p5_formal_gate_not_passed"
    assert verdict["p5_formal_10ep_eligible"] is True
    assert verdict["formal_execution_attempted"] is False
    assert "p4_summary_status_block" in verdict["blocking_reasons"]
    assert "p4_formal_claim_not_allowed" in verdict["blocking_reasons"]
    assert "paired_seed_improvement_count_below_2_of_3" in verdict["blocking_reasons"]


def test_p5_gate_blocks_pass_status_with_nonempty_p4_blocking_reasons(
    tmp_path: Path, monkeypatch
) -> None:
    seeds = list(range(20260421, 20260431))
    baseline_root = _write_task14_baseline_authority(tmp_path, seeds=seeds)
    v2_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    _write_task14_gate_summary(
        v2_root,
        eligible=True,
        status="PASS",
        formal_claim_allowed=True,
        p5_formal_10ep_eligible=True,
        blocking_reasons=["late_gate_blocker"],
    )
    output_dir = v2_root / "p5_gate_eval"

    def _unexpected_lane_state(*args, **kwargs):
        raise AssertionError("nonempty P4 blocking_reasons must skip before lane lookup")

    monkeypatch.setattr(ROLLOUT35A, "_resolve_lane_state", _unexpected_lane_state)

    result = ROLLOUT35A.run_p5_gate(
        baseline_authority_root=baseline_root,
        v2_authority_root=v2_root,
        conditioned_run_root=v2_root / "t13_advantage_full_update_1gpu",
        continuation_run_root=v2_root / "t13_continuation_full_update_1gpu",
        output_dir=output_dir,
        seed_start=20260421,
        seed_end=20260430,
    )

    verdict = json.loads(
        Path(result["min_loop_verdict_path"]).read_text(encoding="utf-8")
    )
    assert verdict["status"] == "SKIPPED"
    assert verdict["blocker_reason"] == "p5_formal_gate_not_passed"
    assert verdict["formal_execution_attempted"] is False
    assert "p4_blocking_reasons_present" in verdict["blocking_reasons"]
    assert "late_gate_blocker" in verdict["blocking_reasons"]


def test_missing_v1_seed_bundle(tmp_path: Path, monkeypatch) -> None:
    baseline_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v1"
    v2_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    _write_task14_gate_summary(v2_root, eligible=True)
    conditioned_root = v2_root / "t13_advantage_full_update_1gpu"
    continuation_root = v2_root / "t13_continuation_full_update_1gpu"
    output_dir = v2_root / "p5_gate_eval"

    def _unexpected_lane_state(*args, **kwargs):
        raise AssertionError("missing v1 seed bundle must skip before formal lane lookup")

    monkeypatch.setattr(ROLLOUT35A, "_resolve_lane_state", _unexpected_lane_state)

    result = ROLLOUT35A.run_p5_gate(
        baseline_authority_root=baseline_root,
        v2_authority_root=v2_root,
        conditioned_run_root=conditioned_root,
        continuation_run_root=continuation_root,
        output_dir=output_dir,
        seed_start=20260421,
        seed_end=20260430,
    )

    verdict = json.loads(
        Path(result["min_loop_verdict_path"]).read_text(encoding="utf-8")
    )
    assert verdict["status"] == "SKIPPED"
    assert verdict["gate_mode"] == "skipped"
    assert verdict["blocker_reason"] == "missing_v1_seed_bundle"
    assert verdict["seed_set_source"] == "inherit_from_v1"
    assert verdict["seed_bundle_status"] == "blocked"
    assert verdict["seed_bundle_blocker_code"] == "seed_bundle_missing_block"
    assert verdict["formal_execution_attempted"] is False


def test_p5_seed_bundle_identity(tmp_path: Path, monkeypatch) -> None:
    seeds = list(range(20260421, 20260431))
    baseline_root = _write_task14_baseline_authority(tmp_path, seeds=seeds)
    v2_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    _write_task14_gate_summary(v2_root, eligible=True)
    conditioned_root = _write_task14_checkpoint_lane_root(
        tmp_path,
        "t13_advantage_full_update_1gpu",
    )
    continuation_root = _write_task14_checkpoint_lane_root(
        tmp_path,
        "t13_continuation_full_update_1gpu",
    )
    output_dir = v2_root / "p5_gate_eval"

    def _fake_run_p5_eval_lane(
        *,
        lane_name: str,
        lane_state: dict[str, object],
        run_root: Path,
        output_dir: Path,
        requested_seed_set: list[int],
    ) -> dict[str, object]:
        del lane_state, run_root
        summary_payload = _build_eval_summary_payload(
            seeds=list(requested_seed_set),
            label=lane_name,
            success_count=1,
            success_rate=0.1,
            advantage=1.0 if lane_name == "conditioned" else None,
        )
        output_path = ROLLOUT35A._copy_eval_summary(
            output_dir=output_dir,
            lane_name=lane_name,
            payload=summary_payload,
        )
        return {
            "lane_name": lane_name,
            "status": "PASS",
            "source_summary_path": None,
            "output_summary_path": ROLLOUT35A._safe_relpath(output_path),
            "checkpoint_path": None,
            "checkpoint_asset_path": None,
            "success_count": 1,
            "success_rate": 0.1,
            "episodes": len(requested_seed_set),
            "seed_base": requested_seed_set[0],
            "episode_seeds": list(requested_seed_set),
        }

    monkeypatch.setattr(ROLLOUT35A, "_run_p5_eval_lane", _fake_run_p5_eval_lane)

    result = ROLLOUT35A.run_p5_gate(
        baseline_authority_root=baseline_root,
        v2_authority_root=v2_root,
        conditioned_run_root=conditioned_root,
        continuation_run_root=continuation_root,
        output_dir=output_dir,
        seed_start=20260421,
        seed_end=20260430,
    )

    verdict = json.loads(
        Path(result["min_loop_verdict_path"]).read_text(encoding="utf-8")
    )
    baseline_eval = json.loads(
        (output_dir / "baseline" / "eval_summary.json").read_text(encoding="utf-8")
    )
    conditioned_eval = json.loads(
        (output_dir / "conditioned" / "eval_summary.json").read_text(encoding="utf-8")
    )
    continuation_eval = json.loads(
        (output_dir / "continuation" / "eval_summary.json").read_text(encoding="utf-8")
    )
    assert verdict["status"] == "PASS"
    assert verdict["gate_mode"] == "executed"
    assert verdict["seed_set_source"] == "inherit_from_v1"
    assert verdict["seed_set"] == seeds
    assert verdict["requested_seed_set"] == seeds
    assert verdict["seed_bundle_identity_pass"] is True
    assert verdict["formal_execution_attempted"] is True
    assert verdict["lane_eval_outputs"]["baseline"]["episode_seeds"] == seeds
    assert verdict["lane_eval_outputs"]["conditioned"]["episode_seeds"] == seeds
    assert verdict["lane_eval_outputs"]["continuation"]["episode_seeds"] == seeds
    assert [entry["seed"] for entry in baseline_eval["episode_results"]] == seeds
    assert [entry["seed"] for entry in conditioned_eval["episode_results"]] == seeds
    assert [entry["seed"] for entry in continuation_eval["episode_results"]] == seeds


def test_report_sections_generate_required_report_set_and_sections(tmp_path: Path) -> None:
    v2_root, output_dir = _write_task15_report_authority(tmp_path)

    outputs = REPORT35B.generate_report_set(v2_root, output_dir)

    assert set(outputs) == {"report", "readme", "plan_snapshot"}
    assert outputs["report"] == output_dir / REPORT35B.REPORT_FILENAME
    assert outputs["readme"] == output_dir / REPORT35B.README_FILENAME
    assert outputs["plan_snapshot"] == output_dir / REPORT35B.PLAN_SNAPSHOT_FILENAME
    for path in outputs.values():
        assert path.is_file()

    report_text = outputs["report"].read_text(encoding="utf-8")
    readme_text = outputs["readme"].read_text(encoding="utf-8")
    plan_snapshot_text = outputs["plan_snapshot"].read_text(encoding="utf-8")

    assert "## 2. Authority Inputs" in report_text
    assert "## 11. P5 result / formal gate" in report_text
    assert "## 13. Plain answers" in report_text
    assert REPORT35B._safe_relpath(v2_root / "p5_gate_eval" / "min_loop_verdict.json") in report_text
    assert REPORT35B._safe_relpath(v2_root / "p4_loss_action_subgoal" / ROLLOUT35A.FULL_UPDATE_DIAGNOSTIC_SUMMARY_FILENAME) in report_text
    assert f"./{REPORT35B.REPORT_FILENAME}" in readme_text
    assert f"./{REPORT35B.PLAN_SNAPSHOT_FILENAME}" in readme_text
    assert "## Frozen Task 15 excerpt" in plan_snapshot_text
    assert REPORT35B._safe_relpath(REPORT35B.PLAN_PATH) in plan_snapshot_text


def test_skipped_gate_reason_report_includes_blocker_rationale_when_p5_skipped(tmp_path: Path) -> None:
    v2_root, output_dir = _write_task15_report_authority(
        tmp_path,
        p5_gate_mode="skipped",
        p5_status="SKIPPED",
        p5_blocker_reason="p5_formal_10ep_ineligible",
        p5_blocking_reasons=[
            "paired_action_instrumentation_incomplete",
            "label_semantics_gate_block",
        ],
    )

    outputs = REPORT35B.generate_report_set(v2_root, output_dir)
    report_text = outputs["report"].read_text(encoding="utf-8")

    assert "- skipped gate rationale comes from the authority verdict and must remain explicit here." in report_text
    assert "- blocker_reason: `p5_formal_10ep_ineligible`" in report_text
    assert "- blocker summary path: " in report_text
    assert REPORT35B._safe_relpath(v2_root / "p5_gate_eval" / "p5_gate_blocker_summary.json") in report_text
    assert "- gate_summary_status: `BLOCK`" in report_text
    assert "- `paired_action_instrumentation_incomplete`" in report_text
    assert "- `label_semantics_gate_block`" in report_text
