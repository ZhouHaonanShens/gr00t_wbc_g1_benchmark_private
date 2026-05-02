from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from work.recap import finetune_full
from work.recap import launch_finetune_use_ddp
from work.recap import policy as recap_policy
from work.recap import run_manifest
from work.recap import scope_experiment
from work.recap.script_apps import recap_finetune_repro_app
from work.recap.train_scope_audit import PAPER_METHOD_GAP
from work.recap.train_scope_audit import RECAP_TRAIN_SCOPE_CHOICES
from work.recap.train_scope_audit import RUNTIME_RESOLUTION_STATUS_NOT_ATTEMPTED
from work.recap.train_scope_audit import STATIC_SCOPE_AUDIT_ARTIFACT_KIND
from work.recap.train_scope_audit import TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY
from work.recap.train_scope_audit import build_scope_summary
from work.recap.train_scope_audit import compute_static_scope_audit
from work.recap.train_scope_audit import emit_scope_audit_json
from work.recap.train_scope_audit import parse_scope_flag


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_ROOT_REL = "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"


def _load_script_module(module_name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_checkpoint(root: Path, name: str) -> Path:
    checkpoint_dir = root / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"test-checkpoint")
    return checkpoint_dir


def _relative_absolute_action_contract() -> dict[str, object]:
    return {
        "relative_action_keys": ["left_arm", "right_arm"],
        "absolute_action_keys": ["left_hand", "right_hand", "waist"],
        "action_representation_by_key": {
            "left_arm": "RELATIVE",
            "right_arm": "RELATIVE",
            "left_hand": "ABSOLUTE",
            "right_hand": "ABSOLUTE",
            "waist": "ABSOLUTE",
        },
        "must_not_conflate_horizon_and_execution": True,
    }


class _FakeAuditParameter:
    def __init__(
        self,
        *,
        requires_grad: bool,
        numel: int,
        shape: tuple[int, ...] = (1,),
        dtype: str = "torch.float32",
        device: str = "cpu",
        is_meta: bool = False,
    ) -> None:
        self.requires_grad = bool(requires_grad)
        self._numel = int(numel)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.is_meta = bool(is_meta)

    def numel(self) -> int:
        return self._numel


class _FakeAuditModel:
    def __init__(self, named_parameters: list[tuple[str, _FakeAuditParameter]]) -> None:
        self._named_parameters = list(named_parameters)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> list[tuple[str, _FakeAuditParameter]]:
        del prefix, recurse
        if not remove_duplicate:
            return list(self._named_parameters)
        deduped: list[tuple[str, _FakeAuditParameter]] = []
        seen: set[int] = set()
        for name, param in self._named_parameters:
            if id(param) in seen:
                continue
            seen.add(id(param))
            deduped.append((name, param))
        return deduped


class _FakeAuditOptimizer:
    def __init__(self, param_groups: list[dict[str, object]]) -> None:
        self.param_groups = param_groups


def _make_static_audit_fixture() -> tuple[_FakeAuditModel, _FakeAuditOptimizer]:
    advantage = _FakeAuditParameter(requires_grad=True, numel=4, shape=(2, 2))
    vlln = _FakeAuditParameter(requires_grad=True, numel=6, shape=(3, 2))
    diffusion = _FakeAuditParameter(requires_grad=False, numel=8, shape=(2, 4))
    projector = _FakeAuditParameter(requires_grad=True, numel=2, shape=(2,))
    model = _FakeAuditModel(
        [
            ("action_head.advantage_embedding.weight", advantage),
            ("action_head.vlln.weight", vlln),
            ("action_head.model.weight", diffusion),
            ("projector.weight", projector),
        ]
    )
    optimizer = _FakeAuditOptimizer(
        [
            {"params": [advantage, vlln], "lr": 3e-5, "weight_decay": 0.1},
            {"params": [projector], "lr": 1e-5, "weight_decay": 0.0},
        ]
    )
    return model, optimizer


def test_authority_root_routes_v2_output_and_repo_local_metadata() -> None:
    output_dir_rel = f"{AUTHORITY_ROOT_REL}/p1_one_step"
    resolved_output_dir = finetune_full.resolve_full_update_authority_output_dir(
        REPO_ROOT,
        output_dir_rel,
        require_v2_authority=True,
    )

    contract = finetune_full._build_deterministic_artifact_contract(
        REPO_ROOT,
        output_dir=resolved_output_dir,
        num_gpus=1,
    )
    assert contract["direct_output_contract_path"] == output_dir_rel
    assert contract["direct_output_contract_status"] == "live_authority"
    assert contract["version_surface_path"] == str(
        resolved_output_dir / "repo_local_metadata/version_surface.json"
    )
    assert contract["census_after_model_build_paths"] == [
        str(resolved_output_dir / "repo_local_metadata/census_after_model_build_rank0.json")
    ]

    _cmd, meta, env_overrides = recap_finetune_repro_app._build_training_launcher_cmd(
        python_exe=Path("/tmp/python"),
        upstream_script=Path("/tmp/launch.py"),
        forwarded_args=[
            "--dataset-path",
            "/tmp/data",
            "--output-dir",
            output_dir_rel,
            "--num-gpus",
            "1",
        ],
        cuda_visible_devices="1",
    )
    assert meta["output_dir"] == output_dir_rel
    assert env_overrides == {"CUDA_VISIBLE_DEVICES": "1"}

    config = SimpleNamespace(
        training=SimpleNamespace(output_dir=output_dir_rel, experiment_name="formal_run")
    )
    effective_output_dir = launch_finetune_use_ddp._compute_effective_output_dir(config)
    metadata_dir = launch_finetune_use_ddp._repo_local_metadata_dir(config)
    assert effective_output_dir == resolved_output_dir / "formal_run"
    assert metadata_dir == effective_output_dir / "repo_local_metadata"


@pytest.mark.parametrize(
    ("attempted_output_dir", "blocker_code", "matched_prefix"),
    [
        (
            "agent/artifacts/recap_min_loop/single_gpu_v1/t5_baseline_formal_eval",
            "readonly_baseline_root_blocked",
            None,
        ),
        (
            "agent/artifacts/stage3_t10_advantage_1gpu/formal_run",
            "historical_stage3_output_root_blocked",
            "stage3_t10",
        ),
        (
            "agent/artifacts/stage3_t11_scope_probe/formal_run",
            "historical_stage3_output_root_blocked",
            "stage3_t11",
        ),
        (
            "agent/artifacts/stage3_t12_scope_probe/formal_run",
            "historical_stage3_output_root_blocked",
            "stage3_t12",
        ),
        (
            "agent/artifacts/stage3_t13_advantage_full_update_1gpu/formal_run",
            "historical_stage3_output_root_blocked",
            "stage3_t13",
        ),
    ],
)
def test_legacy_root_block_emits_machine_readable_blocker(
    attempted_output_dir: str,
    blocker_code: str,
    matched_prefix: str | None,
) -> None:
    with pytest.raises(finetune_full.AuthorityRootBlocker) as excinfo:
        recap_finetune_repro_app._build_training_launcher_cmd(
            python_exe=Path("/tmp/python"),
            upstream_script=Path("/tmp/launch.py"),
            forwarded_args=[
                "--dataset-path",
                "/tmp/data",
                "--output-dir",
                attempted_output_dir,
                "--num-gpus",
                "1",
            ],
            cuda_visible_devices="1",
        )

    payload = json.loads(str(excinfo.value))
    assert payload["schema_version"] == finetune_full.AUTHORITY_ROOT_BLOCKER_SCHEMA_VERSION
    assert payload["artifact_kind"] == "recap_full_update_authority_root_blocker"
    assert payload["status"] == "blocked"
    assert payload["blocker_code"] == blocker_code
    assert payload["attempted_output_dir"] == attempted_output_dir
    assert payload["required_authority_root"] == AUTHORITY_ROOT_REL
    assert payload["matched_legacy_root_prefix"] == matched_prefix


def test_scope_taxonomy_maps_public_contract_exactly() -> None:
    expected_labels = {
        "current_partial": "legacy_partial_control",
        "full_action": "maximal_feasible_action_scope_candidate",
        "full_policy": "maximal_feasible_policy_scope_candidate",
        "strict_full": "strict_full_scope_candidate",
    }

    summaries = {
        scope: build_scope_summary(
            scope,
            legacy_scope_bridge=scope_experiment.build_v2_train_scope_shim_metadata(scope),
        )
        for scope in RECAP_TRAIN_SCOPE_CHOICES
    }

    assert tuple(summaries) == RECAP_TRAIN_SCOPE_CHOICES
    for scope, summary in summaries.items():
        assert summary["train_scope_requested"] == scope
        assert summary["scope_faithfulness"] == expected_labels[scope]
        assert summary["method_faithfulness"]["recap_method_contract"] == (
            "continuous_numeric_advantage_v2"
        )
        assert summary["method_faithfulness"]["paper_equivalent"] is False
        assert summary["method_faithfulness"]["paper_method_gap"] == list(
            PAPER_METHOD_GAP
        )
        assert summary["legacy_scope_shim"]["legacy_scope_semantics_preserved"] is True

    assert summaries["current_partial"]["required_trainable_families"] != summaries[
        "strict_full"
    ]["required_trainable_families"]
    assert summaries["strict_full"]["scope_faithfulness"] == "strict_full_scope_candidate"


def test_task8_bool_groups_accept_explicit_true_values() -> None:
    module = _load_script_module(
        "recap_numeric_adv_smoke_task8_cli",
        "work/recap/scripts/34b_recap_numeric_adv_smoke.py",
    )
    parser = module._build_parser()

    args = parser.parse_args(
        [
            "--dataset-path",
            "/tmp/data",
            "--output-dir",
            "/tmp/out",
            "--balanced-advantage-batches",
            "true",
            "--write-conditioning-functional-probe",
            "true",
            "--write-paired-action-probe",
            "true",
            "--write-label-semantics-audit",
            "true",
            "--write-shuffled-advantage-negative-control",
            "true",
        ]
    )

    assert args.balanced_advantage_batches is True
    assert args.write_conditioning_functional_probe is True
    assert args.write_paired_action_probe is True
    assert args.write_label_semantics_audit is True
    assert args.write_shuffled_advantage_negative_control is True


def test_task8_bool_groups_preserve_flag_and_no_flag_behavior() -> None:
    module = _load_script_module(
        "recap_numeric_adv_smoke_task8_cli_toggle",
        "work/recap/scripts/34b_recap_numeric_adv_smoke.py",
    )
    parser = module._build_parser()

    enabled = parser.parse_args(
        [
            "--dataset-path",
            "/tmp/data",
            "--output-dir",
            "/tmp/out",
            "--write-conditioning-functional-probe",
        ]
    )
    disabled = parser.parse_args(
        [
            "--dataset-path",
            "/tmp/data",
            "--output-dir",
            "/tmp/out",
            "--no-write-conditioning-functional-probe",
        ]
    )

    assert enabled.write_conditioning_functional_probe is True
    assert disabled.write_conditioning_functional_probe is False


def test_task8_tensor_l2_accepts_cuda_tensor() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        return

    module = _load_script_module(
        "recap_numeric_adv_smoke_task8_tensor_l2",
        "work/recap/scripts/34b_recap_numeric_adv_smoke.py",
    )

    value = torch.tensor([3.0, 4.0], device="cuda")

    assert abs(float(module._task8_tensor_l2(value)) - 5.0) < 1e-6


def test_scope_summary_cli_contract_and_manifest_bridge(tmp_path: Path) -> None:
    smoke_script = _load_script_module(
        "task2_recap_numeric_adv_smoke",
        "work/recap/scripts/34b_recap_numeric_adv_smoke.py",
    )
    control_script = _load_script_module(
        "task2_stage3_baseline_control",
        "work/recap/scripts/30i_stage3_baseline_continuation_control.py",
    )

    smoke_parser = smoke_script._build_parser()
    control_parser = control_script.build_parser()
    smoke_scope_arg = next(
        action for action in smoke_parser._actions if action.dest == "recap_train_scope"
    )
    control_scope_arg = next(
        action for action in control_parser._actions if action.dest == "recap_train_scope"
    )

    assert tuple(smoke_scope_arg.choices) == RECAP_TRAIN_SCOPE_CHOICES
    assert tuple(control_scope_arg.choices) == RECAP_TRAIN_SCOPE_CHOICES

    smoke_args = smoke_parser.parse_args(
        [
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--output-dir",
            f"{AUTHORITY_ROOT_REL}/p1_one_step",
            "--recap-train-scope",
            "strict_full",
        ]
    )
    control_args = control_parser.parse_args(
        [
            "--recap-train-scope",
            "full_policy",
        ]
    )
    assert smoke_args.recap_train_scope == "strict_full"
    assert control_args.recap_train_scope == "full_policy"

    checkpoint_dir = _make_checkpoint(tmp_path, "checkpoint-100")
    scope_extension = scope_experiment.build_scope_experiment_extension("S2")
    scope_summary = build_scope_summary(
        "strict_full",
        legacy_scope_bridge=scope_experiment.build_v2_train_scope_shim_metadata(
            "strict_full"
        ),
    )
    manifest = run_manifest.build_run_manifest_from_sources(
        finetune_summary={
            "branch": "UNITREE_G1",
            "commit": "abc123def456",
            "dataset_fingerprint": "dataset-fingerprint-123",
            "checkpoint_selected": str(checkpoint_dir),
            "checkpoint_loaded": str(checkpoint_dir),
            "trainable_module_regex": scope_extension["derived_core_fields"][
                "trainable_module_regex"
            ],
            "eval_overlay_regex": scope_extension["derived_core_fields"][
                "eval_overlay_regex"
            ],
            "policy_horizon": 30,
            "n_action_steps": 20,
            "relative_absolute_action_contract": _relative_absolute_action_contract(),
            TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY: scope_summary,
        },
        eval_summary={
            "evaluation_binding": {
                "eval_uses_finetuned": True,
                "server_load_mode": "model_path",
                "server_load_path": str(checkpoint_dir),
                "base_model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            }
        },
        extensions={
            scope_experiment.SCOPE_EXPERIMENT_EXTENSION_KEY: {"preset_id": "S2"}
        },
        branch="UNITREE_G1",
        commit="abc123def456",
    )

    validation = run_manifest.validate_run_manifest(manifest, repo_root=REPO_ROOT)
    normalized_scope = validation["normalized_manifest"]["extensions"][
        TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY
    ]

    assert validation["formal_eligibility"] == "ALLOW"
    assert normalized_scope["train_scope_requested"] == "strict_full"
    assert normalized_scope["scope_faithfulness"] == "strict_full_scope_candidate"
    assert normalized_scope["method_faithfulness"]["paper_equivalent"] is False
    assert normalized_scope["method_faithfulness"]["paper_method_gap"] == list(
        PAPER_METHOD_GAP
    )


def test_invalid_scope_is_rejected_before_training_starts(tmp_path: Path) -> None:
    assert "not_a_scope" not in RECAP_TRAIN_SCOPE_CHOICES
    with pytest.raises(ValueError, match="--recap-train-scope"):
        parse_scope_flag("not_a_scope")

    smoke_script = _load_script_module(
        "task2_recap_numeric_adv_smoke_invalid",
        "work/recap/scripts/34b_recap_numeric_adv_smoke.py",
    )
    control_script = _load_script_module(
        "task2_stage3_baseline_control_invalid",
        "work/recap/scripts/30i_stage3_baseline_continuation_control.py",
    )

    with pytest.raises(SystemExit):
        smoke_script._build_parser().parse_args(
            [
                "--dataset-path",
                str(tmp_path / "dataset"),
                "--output-dir",
                f"{AUTHORITY_ROOT_REL}/p1_one_step",
                "--recap-train-scope",
                "not_a_scope",
            ]
        )
    with pytest.raises(SystemExit):
        control_script.build_parser().parse_args(["--recap-train-scope", "not_a_scope"])


def test_scope_flag_30i_matches_public_contract(tmp_path: Path) -> None:
    control_script = _load_script_module(
        "task12_stage3_baseline_control_scope_flag",
        "work/recap/scripts/30i_stage3_baseline_continuation_control.py",
    )

    parser = control_script.build_parser()
    scope_arg = next(
        action for action in parser._actions if action.dest == "recap_train_scope"
    )

    assert tuple(scope_arg.choices) == RECAP_TRAIN_SCOPE_CHOICES
    args = parser.parse_args(
        [
            "--recap-train-scope",
            "strict_full",
            "--summary-json",
            str(tmp_path / "control_summary.json"),
        ]
    )
    assert args.recap_train_scope == "strict_full"

    with pytest.raises(SystemExit):
        parser.parse_args(["--recap-train-scope", "not_a_scope"])


@pytest.mark.parametrize(
    ("requested_scope", "expected_flags"),
    [
        (
            "full_action",
            {
                "tune_llm": False,
                "tune_visual": False,
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_top_llm_layers": 0,
                "tune_vlln": True,
            },
        ),
        (
            "full_policy",
            {
                "tune_llm": False,
                "tune_visual": False,
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_top_llm_layers": 0,
                "tune_vlln": True,
            },
        ),
        (
            "strict_full",
            {
                "tune_llm": True,
                "tune_visual": True,
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_top_llm_layers": 0,
                "tune_vlln": True,
            },
        ),
    ],
)
def test_smoke_wrapper_effective_tuning_flags_follow_requested_scope(
    tmp_path: Path,
    requested_scope: str,
    expected_flags: dict[str, bool | int],
) -> None:
    smoke_script = _load_script_module(
        f"task7_recap_numeric_adv_smoke_{requested_scope}",
        "work/recap/scripts/34b_recap_numeric_adv_smoke.py",
    )
    args = smoke_script._build_parser().parse_args(
        [
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--output-dir",
            f"{AUTHORITY_ROOT_REL}/p1_one_step",
            "--recap-train-scope",
            requested_scope,
            "--no-tune-projector",
            "--no-tune-diffusion-model",
            "--no-tune-vlln",
        ]
    )

    flags = smoke_script._effective_tuning_flags(args)
    for key, expected_value in expected_flags.items():
        assert flags[key] == expected_value
    assert flags["requested_scope"] == requested_scope
    assert flags["scope_authority_override_active"] is True


def test_one_step_verifier_training_overrides_only_apply_to_full_update_one_step(
    tmp_path: Path,
) -> None:
    smoke_script = _load_script_module(
        "task7_recap_numeric_adv_smoke_one_step_override",
        "work/recap/scripts/34b_recap_numeric_adv_smoke.py",
    )
    parser = smoke_script._build_parser()

    active_args = parser.parse_args(
        [
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--output-dir",
            f"{AUTHORITY_ROOT_REL}/p1_one_step",
            "--recap-train-scope",
            "strict_full",
            "--max-steps",
            "1",
            "--gradient-accumulation-steps",
            "4",
        ]
    )
    inactive_scope_args = parser.parse_args(
        [
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--output-dir",
            f"{AUTHORITY_ROOT_REL}/p1_one_step",
            "--recap-train-scope",
            "current_partial",
            "--max-steps",
            "1",
            "--gradient-accumulation-steps",
            "4",
        ]
    )
    inactive_steps_args = parser.parse_args(
        [
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--output-dir",
            f"{AUTHORITY_ROOT_REL}/p1_one_step",
            "--recap-train-scope",
            "strict_full",
            "--max-steps",
            "2",
            "--gradient-accumulation-steps",
            "4",
        ]
    )

    assert smoke_script._resolve_one_step_verifier_training_overrides(
        active_args,
        requested_additional_steps=1,
    ) == {
        "override_active": True,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
    }
    assert smoke_script._resolve_one_step_verifier_training_overrides(
        inactive_scope_args,
        requested_additional_steps=1,
    ) == {
        "override_active": False,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": None,
        "warmup_ratio": None,
    }
    assert smoke_script._resolve_one_step_verifier_training_overrides(
        inactive_steps_args,
        requested_additional_steps=2,
    ) == {
        "override_active": False,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": None,
        "warmup_ratio": None,
    }


def test_conflicting_scope_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting_scope_override"):
        launch_finetune_use_ddp.build_repo_local_trainability_authority(
            requested_scope="full_action",
            scope_summary=build_scope_summary(
                "strict_full",
                legacy_scope_bridge=scope_experiment.build_v2_train_scope_shim_metadata(
                    "strict_full"
                ),
            ),
            condition_focused_continuation=True,
            condition_hot_lr_scale=3.0,
            diffusion_trunk_lr_scale=0.0,
            route=recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
        )


def test_static_audit_export(tmp_path: Path) -> None:
    model, optimizer = _make_static_audit_fixture()

    audit = compute_static_scope_audit(
        model=model,
        optimizer=optimizer,
        scope_requested=build_scope_summary("current_partial"),
    )
    dest = tmp_path / "full_update_scope_audit.json"
    written = emit_scope_audit_json(dest, audit)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == STATIC_SCOPE_AUDIT_ARTIFACT_KIND
    assert payload["audit_phase"] == "static"
    assert payload["train_scope_requested"] == "current_partial"
    assert payload["strict_full_runtime_attempted"] is False
    assert payload["runtime_resolution_status"] == RUNTIME_RESOLUTION_STATUS_NOT_ATTEMPTED
    assert payload["scope_faithfulness"] == "legacy_partial_control"
    assert payload["static_verdict"] == "PASS"
    assert payload["candidate_scope_coverage"]["required_trainable_names_missing"] == []
    assert payload["candidate_scope_coverage"]["forbidden_trainable_names"] == []
    assert payload["optimizer_integrity"]["trainable_params_missing_from_optimizer"] == []
    assert payload["optimizer_integrity"]["duplicate_optimizer_params"] == []
    assert "train_scope_effective" not in payload
    assert "resolution_status" not in payload
    assert "grad_probe_after_backward" not in payload
    assert "param_delta_after_step" not in payload


def test_optimizer_coverage() -> None:
    shared = _FakeAuditParameter(requires_grad=True, numel=12, shape=(3, 4))
    unique = _FakeAuditParameter(requires_grad=True, numel=6, shape=(2, 3))
    model = _FakeAuditModel(
        [
            ("action_head.model.shared_a", shared),
            ("action_head.model.shared_b", shared),
            ("action_head.model.unique", unique),
        ]
    )
    optimizer = _FakeAuditOptimizer(
        [{"params": [shared, unique], "lr": 1e-5, "weight_decay": 0.1}]
    )

    audit = compute_static_scope_audit(
        model=model,
        optimizer=optimizer,
        scope_requested=build_scope_summary("strict_full"),
    )

    rows = audit["parameter_coverage"]["parameter_rows"]
    shared_rows = [row for row in rows if row["name"].startswith("action_head.model.shared_")]
    assert audit["static_verdict"] == "PASS"
    assert audit["optimizer_integrity"]["trainable_params_missing_from_optimizer"] == []
    assert audit["optimizer_integrity"]["duplicate_optimizer_params"] == []
    assert len(shared_rows) == 2
    assert shared_rows[0]["param_identity"] == shared_rows[1]["param_identity"]
    assert shared_rows[0]["optimizer_group_indices"] == [0]
    assert shared_rows[1]["optimizer_group_indices"] == [0]


def test_duplicate_bucket_block() -> None:
    shared = _FakeAuditParameter(requires_grad=True, numel=4, shape=(2, 2))
    other = _FakeAuditParameter(requires_grad=True, numel=6, shape=(3, 2))
    model = _FakeAuditModel(
        [
            ("action_head.model.shared", shared),
            ("action_head.model.other", other),
        ]
    )
    optimizer = _FakeAuditOptimizer(
        [
            {"params": [shared], "lr": 1e-5, "weight_decay": 0.1},
            {"params": [shared, other], "lr": 1e-5, "weight_decay": 0.0},
        ]
    )

    audit = compute_static_scope_audit(
        model=model,
        optimizer=optimizer,
        scope_requested=build_scope_summary("strict_full"),
    )

    duplicates = audit["optimizer_integrity"]["duplicate_optimizer_params"]
    assert audit["static_verdict"] == "BLOCK"
    assert "DUPLICATE_OPTIMIZER_PARAM" in audit["static_block_reasons"]
    assert len(duplicates) == 1
    assert duplicates[0]["duplicate_kind"] == "cross_group"
    assert duplicates[0]["group_indices"] == [0, 1]


def test_zero_lr_trainable_block() -> None:
    trainable = _FakeAuditParameter(requires_grad=True, numel=4, shape=(2, 2))
    model = _FakeAuditModel([("action_head.model.weight", trainable)])
    optimizer = _FakeAuditOptimizer(
        [{"params": [trainable], "lr": 0.0, "weight_decay": 0.1}]
    )

    audit = compute_static_scope_audit(
        model=model,
        optimizer=optimizer,
        scope_requested=build_scope_summary("strict_full"),
    )

    zero_lr_groups = audit["optimizer_integrity"]["zero_lr_trainable_param_groups"]
    assert audit["static_verdict"] == "BLOCK"
    assert "ZERO_LR_TRAINABLE_PARAM_GROUP" in audit["static_block_reasons"]
    assert len(zero_lr_groups) == 1
    assert zero_lr_groups[0]["group_index"] == 0
    assert zero_lr_groups[0]["zero_lr_trainable_names"] == ["action_head.model.weight"]
