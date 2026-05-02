from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import types
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import finetune_full
from work.recap import launch_finetune_use_ddp
from work.recap import policy as recap_policy
from work.recap.script_apps import recap_finetune_repro_app
from work.recap.train_scope_audit import build_scope_summary


def _dummy_ft_config(*, num_gpus: int) -> SimpleNamespace:
    return SimpleNamespace(
        base_model_path="/tmp/base-model",
        dataset_path="/tmp/dataset",
        embodiment_tag=SimpleNamespace(value="UNITREE_G1"),
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        state_dropout_prob=0.0,
        random_rotation_angle=None,
        color_jitter_params=None,
        global_batch_size=4,
        dataloader_num_workers=0,
        learning_rate=1e-5,
        gradient_accumulation_steps=1,
        output_dir="/tmp/output",
        save_steps=4,
        save_total_limit=1,
        num_gpus=num_gpus,
        use_wandb=False,
        max_steps=4,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        shard_size=1024,
        episode_sampling_rate=0.1,
        num_shards_per_epoch=1000,
    )


def _dummy_runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(),
        training=SimpleNamespace(),
        data=SimpleNamespace(),
    )


def test_apply_finetune_overrides_keeps_single_gpu_behavior() -> None:
    config = _dummy_runtime_config()
    ft_config = _dummy_ft_config(num_gpus=1)

    use_ddp = launch_finetune_use_ddp.apply_finetune_overrides(
        config=config,
        ft_config=ft_config,
    )

    assert use_ddp is False
    assert config.training.num_gpus == 1
    assert config.training.use_ddp is False
    assert config.training.start_from_checkpoint == "/tmp/base-model"
    assert config.training.output_dir == "/tmp/output"
    assert config.data.shard_size == 1024
    assert config.data.episode_sampling_rate == 0.1


def test_apply_finetune_overrides_enables_ddp_for_multi_gpu() -> None:
    config = _dummy_runtime_config()
    ft_config = _dummy_ft_config(num_gpus=2)

    use_ddp = launch_finetune_use_ddp.apply_finetune_overrides(
        config=config,
        ft_config=ft_config,
    )

    assert use_ddp is True
    assert config.training.num_gpus == 2
    assert config.training.use_ddp is True
    assert launch_finetune_use_ddp.resolve_repo_local_use_ddp(2) is True


def test_apply_finetune_overrides_disables_grad_clipping_for_full_update_one_step_probe() -> None:
    config = _dummy_runtime_config()
    ft_config = _dummy_ft_config(num_gpus=1)
    authority = launch_finetune_use_ddp.build_repo_local_trainability_authority(
        requested_scope="strict_full",
        scope_summary=build_scope_summary("strict_full"),
        condition_focused_continuation=False,
        condition_hot_lr_scale=3.0,
        diffusion_trunk_lr_scale=0.0,
        route=recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
    )
    setattr(
        ft_config,
        launch_finetune_use_ddp.REPO_LOCAL_TRAINABILITY_AUTHORITY_FIELD,
        authority,
    )
    ft_config.max_steps = 1

    launch_finetune_use_ddp.apply_finetune_overrides(
        config=config,
        ft_config=ft_config,
    )

    assert config.training.max_grad_norm == 0.0


def test_repo_local_wrappers_point_to_use_ddp_launcher() -> None:
    assert (
        recap_finetune_repro_app.DEFAULT_UPSTREAM_SCRIPT_REL
        == "work/recap/launch_finetune_use_ddp.py"
    )
    assert (
        finetune_full.DEFAULT_REAL_LAUNCHER_REL
        == "work/recap/launch_finetune_use_ddp.py"
    )
    assert finetune_full._auto_use_ddp(1) is False
    assert finetune_full._auto_use_ddp(2) is True


def test_effective_batch_geometry_records_single_gpu_formal_values() -> None:
    assert finetune_full._effective_batch_geometry(
        global_batch_size=4,
        gradient_accumulation_steps=4,
        num_gpus=1,
    ) == {
        "per_device_batch_size": 1,
        "effective_update_batch": 4,
    }


def test_build_training_launcher_cmd_keeps_single_process_for_one_gpu() -> None:
    cmd, meta, env_overrides = recap_finetune_repro_app._build_training_launcher_cmd(
        python_exe=Path("/tmp/python"),
        upstream_script=Path("/tmp/launch.py"),
        forwarded_args=[
            "--dataset-path",
            "/tmp/data",
            "--output-dir",
            "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run",
            "--num-gpus",
            "1",
        ],
        cuda_visible_devices="1",
    )

    assert cmd == [
        "/tmp/python",
        "/tmp/launch.py",
        "--dataset-path",
        "/tmp/data",
        "--output-dir",
        "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run",
        "--num-gpus",
        "1",
    ]
    assert meta == {
        "num_gpus": 1,
        "uses_torchrun": False,
        "master_port": None,
        "visible_devices": "1",
        "live_launch_family": "single_gpu_v1",
        "visible_devices_policy": "single_gpu_gpu1_only",
        "torchrun_invoked": False,
        "formal_path_default_gpu1_authority": True,
        "output_dir": "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run",
    }
    assert env_overrides == {"CUDA_VISIBLE_DEVICES": "1"}


def test_build_training_launcher_cmd_uses_torchrun_for_multi_gpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        recap_finetune_repro_app, "_pick_free_master_port", lambda: 29517
    )

    cmd, meta, env_overrides = recap_finetune_repro_app._build_training_launcher_cmd(
        python_exe=Path("/tmp/python"),
        upstream_script=Path("/tmp/launch.py"),
        forwarded_args=["--dataset-path", "/tmp/data", "--num-gpus", "2"],
        cuda_visible_devices="1,2",
    )

    assert cmd == [
        "/tmp/python",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--master_port",
        "29517",
        "/tmp/launch.py",
        "--dataset-path",
        "/tmp/data",
        "--num-gpus",
        "2",
    ]
    assert meta == {
        "num_gpus": 2,
        "uses_torchrun": True,
        "master_port": 29517,
        "visible_devices": "1,2",
        "live_launch_family": "task10_2gpu_ddp_diagnostic_v1",
        "visible_devices_policy": "torchrun_gpu1_gpu2_only",
        "torchrun_invoked": True,
        "formal_path_default_gpu1_authority": False,
        "output_dir": None,
    }
    assert env_overrides == recap_finetune_repro_app.MULTI_GPU_ENV_DEFAULTS


@pytest.mark.parametrize(
    ("num_gpus", "cuda_visible_devices", "error_message"),
    [
        (1, None, "CUDA_VISIBLE_DEVICES must be set explicitly"),
        (1, "1,2", "CUDA_VISIBLE_DEVICES must be exactly 1 or 2"),
        (2, "2,1", "CUDA_VISIBLE_DEVICES must be exactly 1,2"),
        (3, "1,2", "only supports --num-gpus 1 or 2"),
    ],
)
def test_build_training_launcher_cmd_rejects_invalid_gpu_visibility_contract(
    num_gpus: int,
    cuda_visible_devices: str | None,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        recap_finetune_repro_app._build_training_launcher_cmd(
            python_exe=Path("/tmp/python"),
            upstream_script=Path("/tmp/launch.py"),
            forwarded_args=["--dataset-path", "/tmp/data", "--num-gpus", str(num_gpus)],
            cuda_visible_devices=cuda_visible_devices,
        )


def test_single_gpu_gpu1_formal_path_rejects_gpu2_as_default_authority() -> None:
    with pytest.raises(
        ValueError,
        match="single_gpu_v1 formal path requires CUDA_VISIBLE_DEVICES to be exactly '1'",
    ):
        recap_finetune_repro_app._build_training_launcher_cmd(
            python_exe=Path("/tmp/python"),
            upstream_script=Path("/tmp/launch.py"),
            forwarded_args=[
                "--dataset-path",
                "/tmp/data",
                "--output-dir",
                "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run",
                "--num-gpus",
                "1",
            ],
            cuda_visible_devices="2",
        )


def test_pre_bind_and_collect_runtime_surface_rejects_single_process_multi_gpu(
    monkeypatch,
) -> None:
    class _FakeCuda:
        def __init__(self) -> None:
            self.set_device_calls: list[int] = []

        def set_device(self, device: int) -> None:
            self.set_device_calls.append(int(device))

        def current_device(self) -> int:
            return 0

    fake_torch = SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(RuntimeError, match="WORLD_SIZE > 1"):
        launch_finetune_use_ddp._pre_bind_and_collect_runtime_surface(
            requested_num_gpus=2
        )

    assert fake_torch.cuda.set_device_calls == []


def test_write_version_surface_persists_repo_local_metadata_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        launch_finetune_use_ddp,
        "_load_python_contract",
        lambda: {
            "orchestrator_python": "/tmp/orchestrator-python",
            "delegate_runtime_python": "/tmp/delegate-python",
        },
    )
    monkeypatch.setattr(
        launch_finetune_use_ddp,
        "_read_git_commit",
        lambda _repo_path: "deadbeef",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    config = SimpleNamespace(
        training=SimpleNamespace(
            output_dir=str(tmp_path / "formal_output"),
            experiment_name="formal_run",
        )
    )
    fake_torch = SimpleNamespace(
        __version__="2.8.0",
        version=SimpleNamespace(cuda="12.8"),
    )

    version_surface_path = launch_finetune_use_ddp._write_version_surface(
        config=config,
        torch_module=fake_torch,
        current_device=1,
        requested_num_gpus=2,
    )

    assert version_surface_path == (
        tmp_path
        / "formal_output"
        / "formal_run"
        / "repo_local_metadata"
        / "version_surface.json"
    )

    payload = json.loads(version_surface_path.read_text(encoding="utf-8"))
    assert payload["repo_local_launcher"].endswith(
        "work/recap/launch_finetune_use_ddp.py"
    )
    assert payload["orchestrator_python"] == "/tmp/orchestrator-python"
    assert payload["delegate_runtime_python"] == "/tmp/delegate-python"
    assert payload["cuda_visible_devices"] == "1,2"
    assert payload["rank_env"] == {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}
    assert payload["requested_num_gpus"] == 2
    assert payload["torch_cuda_current_device"] == 1
    assert payload["isaac_gr00t_commit"] == "deadbeef"


class _FakeGradParameter:
    def __init__(
        self,
        *,
        requires_grad: bool,
        numel: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.requires_grad = bool(requires_grad)
        self._numel = int(numel)
        self.data = torch.zeros((int(numel),), dtype=dtype)

    def requires_grad_(self, value: bool) -> "_FakeGradParameter":
        self.requires_grad = bool(value)
        return self

    def numel(self) -> int:
        return self._numel

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype


class _FakeNamedParameterModel:
    def __init__(self, named_parameters: list[tuple[str, _FakeGradParameter]]) -> None:
        self._named_parameters = list(named_parameters)

    def named_parameters(self):
        return iter(self._named_parameters)


class _FakeTrainerForOptimizerPlan:
    def __init__(
        self,
        *,
        model: _FakeNamedParameterModel,
        learning_rate: float,
        weight_decay: float,
        decay_parameter_names: set[str],
    ) -> None:
        self.model = model
        self.model_wrapped = None
        self.optimizer = None
        self.args = SimpleNamespace(
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self._decay_parameter_names = set(decay_parameter_names)

    def get_decay_parameter_names(self, _model: object) -> set[str]:
        return set(self._decay_parameter_names)


class _ProbeModel:
    def __init__(self) -> None:
        self._named_parameters = [
            (
                "action_head.advantage_embedding.weight",
                torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32)),
            ),
            (
                "action_head.model.weight",
                torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32)),
            ),
        ]

    def named_parameters(self):
        return iter(self._named_parameters)


class _ProbeOptimizer:
    def __init__(self, model: _ProbeModel) -> None:
        self._model = model
        self.param_groups = [{"params": [param for _, param in model.named_parameters()]}]
        self._apply_update = False
        self.gradient_state = SimpleNamespace(sync_gradients=False)

    def step(self) -> None:
        if not self._apply_update:
            return
        for _name, param in self._model.named_parameters():
            param.data = param.data.add(torch.tensor([1.0], dtype=param.dtype))


def test_optimizer_step_wrapper_preserves_bound_method_semantics_for_scheduler(
    monkeypatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        model=model,
        model_wrapped=None,
        state=SimpleNamespace(global_step=0),
        args=SimpleNamespace(max_steps=2),
    )
    initial_weight = model.weight.detach().clone()

    monkeypatch.setitem(
        launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE,
        "first_optimizer_step_written",
        True,
    )
    monkeypatch.setitem(
        launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE,
        "first_optimizer_step_pending_snapshot",
        None,
    )

    wrapped = launch_finetune_use_ddp._maybe_wrap_repo_local_optimizer_first_step_probe(
        trainer=trainer,
        optimizer=optimizer,
    )

    assert wrapped is optimizer
    assert inspect.ismethod(optimizer.step)
    assert optimizer.step.__self__ is optimizer
    assert callable(optimizer.step.__func__)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    loss = model(torch.ones((1, 1), dtype=torch.float32)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    assert not torch.equal(model.weight.detach(), initial_weight)
    assert scheduler.last_epoch == 1


def test_route_freeze_is_machine_readable_in_version_surface(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        launch_finetune_use_ddp,
        "_load_python_contract",
        lambda: {
            "orchestrator_python": "/tmp/orchestrator-python",
            "delegate_runtime_python": "/tmp/delegate-python",
        },
    )
    monkeypatch.setattr(
        launch_finetune_use_ddp,
        "_read_git_commit",
        lambda _repo_path: "deadbeef",
    )
    authority = launch_finetune_use_ddp.build_repo_local_trainability_authority(
        requested_scope="full_action",
        scope_summary=build_scope_summary("full_action"),
        condition_focused_continuation=True,
        condition_hot_lr_scale=3.0,
        diffusion_trunk_lr_scale=0.0,
        route=recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            output_dir=str(tmp_path / "formal_output"),
            experiment_name="formal_run",
        )
    )
    launch_finetune_use_ddp.attach_repo_local_trainability_authority(
        config=config,
        authority=authority,
    )
    fake_torch = SimpleNamespace(
        __version__="2.8.0",
        version=SimpleNamespace(cuda="12.8"),
    )

    version_surface_path = launch_finetune_use_ddp._write_version_surface(
        config=config,
        torch_module=fake_torch,
        current_device=1,
        requested_num_gpus=1,
    )

    payload = json.loads(version_surface_path.read_text(encoding="utf-8"))
    assert payload["trainability_authority"]["authority_owner"] == "training_entrypoint"
    assert payload["trainability_authority"]["requested_scope"] == "full_action"
    assert payload["route_freeze"]["route"] == recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE
    assert payload["route_freeze"]["policy_class_name"] == recap_policy.DIAGNOSTIC_NUMERIC_ADV_POLICY_CLASS_NAME
    assert payload["route_freeze"]["diagnostic_only"] is True


def test_trainability_authority_controls_requires_grad_and_optimizer_plan() -> None:
    authority = launch_finetune_use_ddp.build_repo_local_trainability_authority(
        requested_scope="current_partial",
        scope_summary=build_scope_summary("current_partial"),
        condition_focused_continuation=True,
        condition_hot_lr_scale=3.0,
        diffusion_trunk_lr_scale=0.0,
        route=recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
    )
    named_parameters = [
        (
            "action_head.advantage_embedding.weight",
            _FakeGradParameter(requires_grad=False, numel=4),
        ),
        ("action_head.vlln.weight", _FakeGradParameter(requires_grad=False, numel=6)),
        ("action_head.model.weight", _FakeGradParameter(requires_grad=True, numel=8)),
        ("action_head.projector.weight", _FakeGradParameter(requires_grad=True, numel=2)),
    ]
    model = _FakeNamedParameterModel(named_parameters)

    summary = launch_finetune_use_ddp.apply_repo_local_trainability_authority(
        model=model,
        authority=authority,
    )
    updated = dict(model.named_parameters())
    assert updated["action_head.advantage_embedding.weight"].requires_grad is True
    assert updated["action_head.vlln.weight"].requires_grad is True
    assert updated["action_head.model.weight"].requires_grad is False
    assert summary["forced_trainable"]["tensors"] == 2
    assert summary["forced_frozen"]["tensors"] == 1
    assert summary["route_freeze"]["route"] == recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE

    trainer = _FakeTrainerForOptimizerPlan(
        model=model,
        learning_rate=1e-5,
        weight_decay=0.1,
        decay_parameter_names={
            "action_head.advantage_embedding.weight",
            "action_head.vlln.weight",
            "action_head.projector.weight",
        },
    )
    plan = launch_finetune_use_ddp.build_repo_local_optimizer_group_plan(
        trainer=trainer,
        authority=authority,
    )

    assert plan is not None
    summaries = plan["logical_group_summaries"]
    assert summaries["condition_hot"]["mode"] == "boosted"
    assert summaries["condition_hot"]["tensors"] == 2
    assert abs(float(summaries["condition_hot"]["lr"]) - 3e-5) < 1e-12
    assert summaries["diffusion_trunk_cold"]["mode"] == "frozen"
    assert summaries["diffusion_trunk_cold"]["tensors"] == 1
    assert summaries["default"]["tensors"] == 1
    assert summaries["default"]["sample_names"] == ["action_head.projector.weight"]


@pytest.mark.parametrize(
    (
        "requested_scope",
        "expected_requires_grad",
        "expected_forced_trainable_tensors",
        "expected_forced_frozen_tensors",
    ),
    [
        (
            "full_action",
            {
                "action_head.advantage_embedding.weight": True,
                "action_head.model.weight": True,
                "action_head.action_decoder.weight": True,
                "projector.weight": False,
                "vla_action_interface.weight": False,
                "backbone.layer.weight": False,
            },
            3,
            1,
        ),
        (
            "full_policy",
            {
                "action_head.advantage_embedding.weight": True,
                "action_head.model.weight": True,
                "action_head.action_decoder.weight": True,
                "projector.weight": True,
                "vla_action_interface.weight": True,
                "backbone.layer.weight": False,
            },
            5,
            1,
        ),
        (
            "strict_full",
            {
                "action_head.advantage_embedding.weight": True,
                "action_head.model.weight": True,
                "action_head.action_decoder.weight": True,
                "projector.weight": True,
                "vla_action_interface.weight": True,
                "backbone.layer.weight": True,
            },
            5,
            0,
        ),
    ],
)
def test_full_scope_authority_maps_requested_scope_to_effective_trainability(
    requested_scope: str,
    expected_requires_grad: dict[str, bool],
    expected_forced_trainable_tensors: int,
    expected_forced_frozen_tensors: int,
) -> None:
    authority = launch_finetune_use_ddp.build_repo_local_trainability_authority(
        requested_scope=requested_scope,
        scope_summary=build_scope_summary(requested_scope),
        condition_focused_continuation=False,
        condition_hot_lr_scale=3.0,
        diffusion_trunk_lr_scale=0.0,
        route=recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
    )
    model = _FakeNamedParameterModel(
        [
            (
                "action_head.advantage_embedding.weight",
                _FakeGradParameter(requires_grad=False, numel=4),
            ),
            ("action_head.model.weight", _FakeGradParameter(requires_grad=False, numel=6)),
            (
                "action_head.action_decoder.weight",
                _FakeGradParameter(requires_grad=False, numel=8),
            ),
            ("projector.weight", _FakeGradParameter(requires_grad=False, numel=2)),
            (
                "vla_action_interface.weight",
                _FakeGradParameter(requires_grad=False, numel=2),
            ),
            ("backbone.layer.weight", _FakeGradParameter(requires_grad=True, numel=10)),
        ]
    )

    summary = launch_finetune_use_ddp.apply_repo_local_trainability_authority(
        model=model,
        authority=authority,
    )

    updated = dict(model.named_parameters())
    assert summary["effective_requested_scope"] == requested_scope
    assert summary["scope_authority_override_active"] is True
    assert summary["forced_trainable"]["tensors"] == expected_forced_trainable_tensors
    assert summary["forced_frozen"]["tensors"] == expected_forced_frozen_tensors
    for name, expected_value in expected_requires_grad.items():
        assert updated[name].requires_grad is expected_value


def test_apply_trainability_authority_promotes_success_probe_trainable_params_to_fp32() -> None:
    authority = launch_finetune_use_ddp.build_repo_local_trainability_authority(
        requested_scope="full_action",
        scope_summary=build_scope_summary("full_action"),
        condition_focused_continuation=False,
        condition_hot_lr_scale=3.0,
        diffusion_trunk_lr_scale=0.0,
        route=recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
    )
    model = _FakeNamedParameterModel(
        [
            (
                "action_head.advantage_embedding.weight",
                _FakeGradParameter(
                    requires_grad=False,
                    numel=4,
                    dtype=torch.bfloat16,
                ),
            ),
            (
                "action_head.model.weight",
                _FakeGradParameter(
                    requires_grad=False,
                    numel=6,
                    dtype=torch.bfloat16,
                ),
            ),
            (
                "action_head.action_decoder.weight",
                _FakeGradParameter(
                    requires_grad=False,
                    numel=8,
                    dtype=torch.bfloat16,
                ),
            ),
            (
                "projector.weight",
                _FakeGradParameter(
                    requires_grad=False,
                    numel=2,
                    dtype=torch.bfloat16,
                ),
            ),
        ]
    )

    summary = launch_finetune_use_ddp.apply_repo_local_trainability_authority(
        model=model,
        authority=authority,
    )

    updated = dict(model.named_parameters())
    assert updated["action_head.advantage_embedding.weight"].dtype == torch.float32
    assert updated["action_head.model.weight"].dtype == torch.float32
    assert updated["action_head.action_decoder.weight"].dtype == torch.bfloat16
    fp32_summary = summary["success_probe_trainable_params_fp32"]
    assert fp32_summary["converted"]["tensors"] == 2
    assert fp32_summary["converted"]["sample_names"] == [
        "action_head.advantage_embedding.weight",
        "action_head.model.weight",
    ]


def test_first_optimizer_step_probe_waits_for_synced_gradient_step(monkeypatch) -> None:
    model = _ProbeModel()
    optimizer = _ProbeOptimizer(model)
    trainer = SimpleNamespace(
        model=model,
        model_wrapped=None,
        state=SimpleNamespace(global_step=0),
    )
    captured: dict[str, object] = {}

    def _capture_probe(*, trainer, optimizer, pre_step_snapshot, payload_override=None):
        payload = (
            launch_finetune_use_ddp._build_rank0_first_optimizer_step_param_delta_payload(
                trainer=trainer,
                optimizer=optimizer,
                pre_step_snapshot=pre_step_snapshot,
            )
            if payload_override is None
            else dict(payload_override)
        )
        captured["payload"] = payload
        return Path("/tmp/repo_local_probe.json")

    monkeypatch.setattr(
        launch_finetune_use_ddp,
        "_write_rank0_first_optimizer_step_param_delta_probe",
        _capture_probe,
    )
    launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] = False

    wrapped = launch_finetune_use_ddp._maybe_wrap_repo_local_optimizer_first_step_probe(
        trainer=trainer,
        optimizer=optimizer,
    )
    wrapped.step()

    assert "payload" not in captured
    assert launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] is False

    optimizer.gradient_state.sync_gradients = True
    optimizer._apply_update = True
    wrapped.step()

    payload = captured["payload"]
    assert isinstance(payload, dict)
    scopes = payload["scopes"]
    assert scopes["advantage_embedding"]["delta_l2_norm"] > 0.0
    assert scopes["diffusion_trunk"]["delta_l2_norm"] > 0.0
    assert launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] is True


def test_first_optimizer_step_probe_defers_one_step_authority_until_train_end(
    monkeypatch,
) -> None:
    model = _ProbeModel()
    optimizer = _ProbeOptimizer(model)
    trainer = SimpleNamespace(
        model=model,
        model_wrapped=None,
        optimizer=optimizer,
        state=SimpleNamespace(global_step=0),
        args=SimpleNamespace(max_steps=1),
    )
    captured: dict[str, object] = {}

    def _capture_probe(*, trainer, optimizer, pre_step_snapshot, payload_override=None):
        payload = (
            launch_finetune_use_ddp._build_rank0_first_optimizer_step_param_delta_payload(
                trainer=trainer,
                optimizer=optimizer,
                pre_step_snapshot=pre_step_snapshot,
            )
            if payload_override is None
            else dict(payload_override)
        )
        captured["payload"] = payload
        return Path("/tmp/repo_local_probe.json")

    monkeypatch.setattr(
        launch_finetune_use_ddp,
        "_write_rank0_first_optimizer_step_param_delta_probe",
        _capture_probe,
    )
    launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] = False
    launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE[
        "first_optimizer_step_pending_snapshot"
    ] = None

    wrapped = launch_finetune_use_ddp._maybe_wrap_repo_local_optimizer_first_step_probe(
        trainer=trainer,
        optimizer=optimizer,
    )
    optimizer.gradient_state.sync_gradients = True
    optimizer._apply_update = True
    wrapped.step()

    assert "payload" not in captured
    assert launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] is False
    assert isinstance(
        launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE[
            "first_optimizer_step_pending_snapshot"
        ],
        dict,
    )

    trainer.state.global_step = 1
    probe_path = (
        launch_finetune_use_ddp._maybe_finalize_repo_local_first_optimizer_step_probe_at_train_end(
            trainer=trainer,
        )
    )

    assert probe_path == Path("/tmp/repo_local_probe.json")
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["trainer_global_step"] == 1
    scopes = payload["scopes"]
    assert scopes["advantage_embedding"]["delta_l2_norm"] > 0.0
    assert scopes["diffusion_trunk"]["delta_l2_norm"] > 0.0
    assert launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] is True
    assert (
        launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE[
            "first_optimizer_step_pending_snapshot"
        ]
        is None
    )


def test_train_end_finalize_falls_back_to_trainer_model_when_optimizer_view_is_stale(
    monkeypatch,
) -> None:
    class _TorchProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_head = torch.nn.Module()
            self.action_head.model = torch.nn.Linear(2, 2, bias=False)
            self.action_head.advantage_embedding = torch.nn.Linear(1, 2)

    stale_optimizer_model = _TorchProbeModel()
    live_model = _TorchProbeModel()
    live_model.load_state_dict(stale_optimizer_model.state_dict())
    optimizer = torch.optim.SGD(stale_optimizer_model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        model=live_model,
        model_wrapped=stale_optimizer_model,
        optimizer=optimizer,
        state=SimpleNamespace(global_step=1),
        args=SimpleNamespace(max_steps=1),
    )
    captured: dict[str, object] = {}

    def _capture_probe(*, trainer, optimizer, pre_step_snapshot, payload_override=None):
        payload = (
            launch_finetune_use_ddp._build_rank0_first_optimizer_step_param_delta_payload(
                trainer=trainer,
                optimizer=optimizer,
                pre_step_snapshot=pre_step_snapshot,
            )
            if payload_override is None
            else dict(payload_override)
        )
        captured["payload"] = payload
        return Path("/tmp/repo_local_probe.json")

    monkeypatch.setattr(
        launch_finetune_use_ddp,
        "_write_rank0_first_optimizer_step_param_delta_probe",
        _capture_probe,
    )
    launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] = False
    launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE[
        "first_optimizer_step_pending_snapshot"
    ] = launch_finetune_use_ddp._capture_repo_local_first_step_parameter_snapshot(
        trainer=trainer,
        optimizer=optimizer,
    )

    for _name, param in live_model.named_parameters():
        param.data = param.data.add(torch.tensor([1.0], dtype=param.dtype))

    probe_path = (
        launch_finetune_use_ddp._maybe_finalize_repo_local_first_optimizer_step_probe_at_train_end(
            trainer=trainer,
        )
    )

    assert probe_path == Path("/tmp/repo_local_probe.json")
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["trainer_global_step"] == 1
    scopes = payload["scopes"]
    assert scopes["advantage_embedding"]["delta_l2_norm"] > 0.0
    assert scopes["diffusion_trunk"]["delta_l2_norm"] > 0.0


def test_train_end_finalize_falls_back_to_checkpoint_when_runtime_views_are_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file

    class _TorchProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_head = torch.nn.Module()
            self.action_head.model = torch.nn.Linear(2, 2, bias=False)
            self.action_head.advantage_embedding = torch.nn.Linear(1, 2)

    live_model = _TorchProbeModel()
    checkpoint_model = _TorchProbeModel()
    checkpoint_model.load_state_dict(live_model.state_dict())
    optimizer = torch.optim.SGD(live_model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        model=live_model,
        model_wrapped=None,
        optimizer=optimizer,
        state=SimpleNamespace(global_step=1),
        args=SimpleNamespace(max_steps=1),
    )
    output_dir = tmp_path / "authority_out"
    metadata_dir = output_dir / "repo_local_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["metadata_dir"] = metadata_dir
    captured: dict[str, object] = {}

    def _capture_probe(*, trainer, optimizer, pre_step_snapshot, payload_override=None):
        payload = (
            launch_finetune_use_ddp._build_rank0_first_optimizer_step_param_delta_payload(
                trainer=trainer,
                optimizer=optimizer,
                pre_step_snapshot=pre_step_snapshot,
            )
            if payload_override is None
            else dict(payload_override)
        )
        captured["payload"] = payload
        return Path("/tmp/repo_local_probe.json")

    monkeypatch.setattr(
        launch_finetune_use_ddp,
        "_write_rank0_first_optimizer_step_param_delta_probe",
        _capture_probe,
    )
    launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] = False
    launch_finetune_use_ddp.REPO_LOCAL_CENSUS_HOOK_STATE[
        "first_optimizer_step_pending_snapshot"
    ] = launch_finetune_use_ddp._capture_repo_local_first_step_parameter_snapshot(
        trainer=trainer,
        optimizer=optimizer,
    )

    for _name, param in checkpoint_model.named_parameters():
        param.data = param.data.add(torch.tensor([1.0], dtype=param.dtype))
    save_file(checkpoint_model.state_dict(), str(checkpoint_dir / "model.safetensors"))

    probe_path = (
        launch_finetune_use_ddp._maybe_finalize_repo_local_first_optimizer_step_probe_at_train_end(
            trainer=trainer,
        )
    )

    assert probe_path == Path("/tmp/repo_local_probe.json")
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["trainer_global_step"] == 1
    scopes = payload["scopes"]
    assert scopes["advantage_embedding"]["delta_l2_norm"] > 0.0
    assert scopes["diffusion_trunk"]["delta_l2_norm"] > 0.0


def test_build_deterministic_artifact_contract_tracks_version_and_census_paths() -> None:
    output_dir = (
        REPO_ROOT / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run"
    )

    contract = finetune_full._build_deterministic_artifact_contract(
        REPO_ROOT,
        output_dir=output_dir,
        num_gpus=1,
    )

    assert contract["live_launch_family"] == "single_gpu_v1"
    assert contract["direct_output_contract_path"] == "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run"
    assert contract["direct_output_contract_status"] == "live_authority"
    assert contract["visible_devices_policy"] == "single_gpu_gpu1_only"
    assert contract["torchrun_invoked"] is False
    assert contract["version_surface_path"].endswith(
        "repo_local_metadata/version_surface.json"
    )
    assert contract["nvidia_smi_snapshot_path"].endswith(
        "repo_local_metadata/nvidia_smi_snapshot.json"
    )
    assert contract["census_after_model_build_paths"] == [
        str(output_dir / "repo_local_metadata/census_after_model_build_rank0.json"),
    ]
    assert contract["census_before_first_forward_paths"] == [
        str(output_dir / "repo_local_metadata/census_before_first_forward_rank0.json"),
    ]
    assert "green_smoke_gate_status" in contract


class _FakeTensor:
    def __init__(self, *, device: str, dtype: str):
        self.device = device
        self.dtype = dtype


class _FakeModel:
    def __init__(
        self,
        *,
        parameters: list[_FakeTensor] | None = None,
        former_parameters: dict[str, _FakeTensor] | None = None,
        buffers: list[_FakeTensor] | None = None,
    ) -> None:
        self._parameters = list(parameters or [])
        self._former_parameters = former_parameters
        self._buffers = list(buffers or [])

    def parameters(self):
        return iter(self._parameters)

    def buffers(self):
        return iter(self._buffers)


class _FakeTorchModule:
    float32 = "torch.float32"

    @staticmethod
    def device(spec: str) -> str:
        return f"device<{spec}>"

    @staticmethod
    def get_default_dtype() -> str:
        return "torch.float32"


def _write_patch_b2_candidate(
    tmp_path: Path,
    *,
    device_unknown_rank0: bool = True,
    device_unknown_rank1: bool = True,
    illegal_memory_access: bool = True,
    child_failed_error: bool = True,
) -> Path:
    candidate_path = tmp_path / "green_smoke_candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "pass": False,
                "candidate_status": "blocked",
                "log_scan": {
                    "forbidden_tokens": {
                        "illegal_memory_access": {"present": illegal_memory_access},
                        "ChildFailedError": {"present": child_failed_error},
                    },
                    "extra_findings": {
                        "device_unknown_rank0": {"present": device_unknown_rank0},
                        "device_unknown_rank1": {"present": device_unknown_rank1},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return candidate_path


def test_resolve_repo_local_model_device_dtype_prefers_parameters() -> None:
    model = _FakeModel(
        parameters=[_FakeTensor(device="cuda:1", dtype="torch.bfloat16")],
        former_parameters={"former": _FakeTensor(device="cuda:7", dtype="torch.float16")},
        buffers=[_FakeTensor(device="cuda:9", dtype="torch.float32")],
    )

    device, device_source = launch_finetune_use_ddp.resolve_repo_local_model_device(
        model,
        torch_module=_FakeTorchModule(),
    )
    dtype, dtype_source = launch_finetune_use_ddp.resolve_repo_local_model_dtype(
        model,
        torch_module=_FakeTorchModule(),
    )

    assert device == "cuda:1"
    assert device_source == "parameters"
    assert dtype == "torch.bfloat16"
    assert dtype_source == "parameters"


def test_resolve_repo_local_model_device_dtype_falls_back_to_former_parameters() -> None:
    model = _FakeModel(
        former_parameters={"former": _FakeTensor(device="cuda:2", dtype="torch.float16")},
        buffers=[_FakeTensor(device="cuda:3", dtype="torch.float32")],
    )

    device, device_source = launch_finetune_use_ddp.resolve_repo_local_model_device(
        model,
        torch_module=_FakeTorchModule(),
    )
    dtype, dtype_source = launch_finetune_use_ddp.resolve_repo_local_model_dtype(
        model,
        torch_module=_FakeTorchModule(),
    )

    assert device == "cuda:2"
    assert device_source == "_former_parameters"
    assert dtype == "torch.float16"
    assert dtype_source == "_former_parameters"


def test_resolve_repo_local_model_device_dtype_falls_back_to_buffers() -> None:
    model = _FakeModel(
        buffers=[_FakeTensor(device="cuda:4", dtype="torch.float32")],
    )

    device, device_source = launch_finetune_use_ddp.resolve_repo_local_model_device(
        model,
        torch_module=_FakeTorchModule(),
    )
    dtype, dtype_source = launch_finetune_use_ddp.resolve_repo_local_model_dtype(
        model,
        torch_module=_FakeTorchModule(),
    )

    assert device == "cuda:4"
    assert device_source == "buffers"
    assert dtype == "torch.float32"
    assert dtype_source == "buffers"


def test_resolve_repo_local_model_device_dtype_falls_back_to_local_rank_and_safe_dtype(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "7")
    model = _FakeModel()

    device, device_source = launch_finetune_use_ddp.resolve_repo_local_model_device(
        model,
        torch_module=_FakeTorchModule(),
    )
    dtype, dtype_source = launch_finetune_use_ddp.resolve_repo_local_model_dtype(
        model,
        torch_module=_FakeTorchModule(),
    )

    assert device == "device<cuda:7>"
    assert device_source == "local_rank_fallback"
    assert dtype == "torch.float32"
    assert dtype_source == "safe_dtype_fallback"


def test_detect_patch_b1_trigger_uses_task8_candidate_and_does_not_skip() -> None:
    trigger = launch_finetune_use_ddp.detect_patch_b1_trigger()

    assert trigger["candidate_path"].endswith(
        "agent/artifacts/stage3_ddp_smoke/run_c_gpu12_attempt01/green_smoke_candidate.json"
    )
    assert trigger["activate"] is True
    assert trigger["skip_reason"] is None
    assert len(trigger["triggered_paths"]) == 2


def test_maybe_install_patch_b1_patches_gr00t_device_dtype_properties(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_path = tmp_path / "green_smoke_candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "census_checks": {
                    "files": [
                        {
                            "phase": "after_model_build",
                            "path": "/tmp/rank0.json",
                            "exists": True,
                            "former_parameters_present": True,
                        },
                        {
                            "phase": "after_model_build",
                            "path": "/tmp/rank1.json",
                            "exists": True,
                            "former_parameters_present": True,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REPO_LOCAL_B1_TRIGGER_CANDIDATE_PATH", str(candidate_path))

    class _FakeGr00t(_FakeModel):
        pass

    class _FakeActionHead(_FakeModel):
        pass

    monkeypatch.setitem(sys.modules, "gr00t", types.ModuleType("gr00t"))
    monkeypatch.setitem(sys.modules, "gr00t.model", types.ModuleType("gr00t.model"))
    monkeypatch.setitem(
        sys.modules,
        "gr00t.model.gr00t_n1d6",
        types.ModuleType("gr00t.model.gr00t_n1d6"),
    )
    fake_gr00t_module = types.ModuleType("gr00t.model.gr00t_n1d6.gr00t_n1d6")
    setattr(fake_gr00t_module, "Gr00tN1d6", _FakeGr00t)
    setattr(fake_gr00t_module, "Gr00tN1d6ActionHead", _FakeActionHead)
    monkeypatch.setitem(
        sys.modules,
        "gr00t.model.gr00t_n1d6.gr00t_n1d6",
        fake_gr00t_module,
    )

    result = launch_finetune_use_ddp.maybe_install_patch_b1(
        torch_module=_FakeTorchModule(),
    )

    assert result["activate"] is True
    assert result["skip_reason"] is None
    assert result["patched_classes"] == ["Gr00tN1d6", "Gr00tN1d6ActionHead"]
    fake_gr00t_instance = _FakeGr00t(
        former_parameters={"former": _FakeTensor(device="cuda:5", dtype="torch.float16")}
    )
    fake_action_head_instance = _FakeActionHead(
        buffers=[_FakeTensor(device="cuda:6", dtype="torch.bfloat16")]
    )
    assert getattr(fake_gr00t_instance, "device") == "cuda:5"
    assert getattr(fake_action_head_instance, "dtype") == "torch.bfloat16"


def _fake_init_process_group_with_device_id(
    *, backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name="", pg_options=None, device_id=None
):
    return {
        "backend": backend,
        "device_id": device_id,
    }


def _fake_init_process_group_without_device_id(
    *, backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name="", pg_options=None
):
    return {
        "backend": backend,
    }


def test_detect_patch_b2_trigger_activates_only_for_blocked_device_unknown_candidate(
    tmp_path: Path,
) -> None:
    candidate_path = _write_patch_b2_candidate(tmp_path)
    fake_dist = SimpleNamespace(
        init_process_group=_fake_init_process_group_with_device_id,
    )

    trigger = launch_finetune_use_ddp.detect_patch_b2_trigger(
        candidate_path=candidate_path,
        torch_dist_module=fake_dist,
    )

    assert trigger["activate"] is True
    assert trigger["skip_reason"] is None
    assert trigger["has_device_id"] is True
    assert trigger["device_unknown_tokens"] == [
        "device_unknown_rank0",
        "device_unknown_rank1",
    ]
    assert trigger["failure_tokens"] == [
        "illegal_memory_access",
        "ChildFailedError",
    ]


def test_detect_patch_b2_trigger_skips_when_device_id_is_unsupported(
    tmp_path: Path,
) -> None:
    candidate_path = _write_patch_b2_candidate(tmp_path)
    fake_dist = SimpleNamespace(
        init_process_group=_fake_init_process_group_without_device_id,
    )

    trigger = launch_finetune_use_ddp.detect_patch_b2_trigger(
        candidate_path=candidate_path,
        torch_dist_module=fake_dist,
    )

    assert trigger["activate"] is False
    assert trigger["has_device_id"] is False
    assert trigger["skip_reason"] == "init_process_group_missing_device_id_parameter"


def test_detect_patch_b2_trigger_skips_without_device_unknown_warning(
    tmp_path: Path,
) -> None:
    candidate_path = _write_patch_b2_candidate(tmp_path, device_unknown_rank1=False)
    fake_dist = SimpleNamespace(
        init_process_group=_fake_init_process_group_with_device_id,
    )

    trigger = launch_finetune_use_ddp.detect_patch_b2_trigger(
        candidate_path=candidate_path,
        torch_dist_module=fake_dist,
    )

    assert trigger["activate"] is False
    assert trigger["skip_reason"] == "candidate_missing_device_unknown_warning"


def test_maybe_install_patch_b2_wraps_existing_init_process_group_minimally(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_path = _write_patch_b2_candidate(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    calls: list[dict[str, object]] = []

    def fake_init_process_group(
        *,
        backend=None,
        init_method=None,
        timeout=None,
        world_size=-1,
        rank=-1,
        store=None,
        group_name="",
        pg_options=None,
        device_id=None,
    ):
        calls.append(
            {
                "args": (),
                "kwargs": {
                    "backend": backend,
                    "init_method": init_method,
                    "timeout": timeout,
                    "world_size": world_size,
                    "rank": rank,
                    "store": store,
                    "group_name": group_name,
                    "pg_options": pg_options,
                    "device_id": device_id,
                },
            }
        )
        return "ok"

    fake_dist = SimpleNamespace(init_process_group=fake_init_process_group)

    result = launch_finetune_use_ddp.maybe_install_patch_b2(
        torch_module=_FakeTorchModule(),
        torch_dist_module=fake_dist,
        current_device=1,
        candidate_path=candidate_path,
    )

    assert result["activate"] is True
    assert result["patched"] is True
    assert result["injected_device_id"] == "device<cuda:1>"
    assert result["device_id_source"] == "current_device"

    assert fake_dist.init_process_group(backend="nccl") == "ok"
    assert calls[0] == {
        "args": (),
        "kwargs": {
            "backend": "nccl",
            "init_method": None,
            "timeout": None,
            "world_size": -1,
            "rank": -1,
            "store": None,
            "group_name": "",
            "pg_options": None,
            "device_id": "device<cuda:1>",
        },
    }

    assert fake_dist.init_process_group(backend="nccl", device_id="device<cuda:9>") == "ok"
    assert calls[1] == {
        "args": (),
        "kwargs": {
            "backend": "nccl",
            "init_method": None,
            "timeout": None,
            "world_size": -1,
            "rank": -1,
            "store": None,
            "group_name": "",
            "pg_options": None,
            "device_id": "device<cuda:9>",
        },
    }
