from __future__ import annotations

# pyright: reportAny=false, reportMissingParameterType=false, reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportUnusedParameter=false

import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_lock_payload() -> dict[str, object]:
    return {
        "schema_version": "v22_preregistration_hash_lock_v1",
        "run_id": "stage1_v22_blind_calibration_iter8_20260426T_nextZ",
        "n_per_variant": 192,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "iter5_r2_sha256": "0f4a1b74152a8e4f88c7259b0033e0f262257a1222e1a6fb5436f084547e7e69",
        "iter5_r4_sha256": "c08da923e96c6d2d6f1f6b2522219eee7f3ab5b6851f880e7502b3fadd7af965",
        "variants": ["A", "B", "C", "X"],
        "selected_protocol": {
            "suite": "libero_spatial",
            "budget": 0.5,
            "cell_id": "libero_spatial__budget_0_50",
            "step_cap": 110,
            "max_steps_full": 220,
            "tasks": ["all_tasks_round_robin_episode_index_modulo_10"],
        },
    }


def _anchor_payload() -> dict[str, object]:
    return {
        "schema_version": "v22_canonical_training_anchor_v1",
        "run_id": "stage1_v22_full_training_eval_iter9_20260426T_nextZ",
        "source_config_name": "pi0_libero",
        "anchor_values": {
            "num_train_steps": 30000,
            "batch_size": 32,
            "seed": 42,
            "num_workers": 2,
            "log_interval_canonical": 100,
            "log_interval_v22_real_training_override": 1,
            "save_interval": 1000,
            "keep_period": 5000,
        },
    }


def _base_args(tmp_path: Path) -> list[str]:
    hash_lock = tmp_path / "hash_lock.json"
    anchor = tmp_path / "canonical_training_anchor.json"
    dataset = tmp_path / "dataset"
    warm_start = tmp_path / "warm_start"
    _write_json(hash_lock, _hash_lock_payload())
    _write_json(anchor, _anchor_payload())
    dataset.mkdir()
    (warm_start / "params").mkdir(parents=True)
    return [
        "--variant",
        "B",
        "--variant-id",
        "control_no_recap_shuffled_adversarial_relabel",
        "--prereg-hash-lock",
        str(hash_lock),
        "--canonical-anchor",
        str(anchor),
        "--output-dir",
        str(tmp_path / "B"),
        "--runtime-log-dir",
        str(tmp_path / "logs"),
        "--warm-start-checkpoint",
        str(warm_start),
        "--dataset-dir",
        str(dataset),
        "--num-train-steps",
        "2",
        "--batch-size",
        "1",
        "--enable-r2-phase-threshold-switching",
        "--emit-loss-decomposition",
        "--emit-threshold-trace",
        "--emit-gradient-attestation",
        "--emit-control-signal-absence-attestation",
        "--emit-sha256sums",
        "--no-sudo",
        "--cuda-visible-devices",
        "1",
    ]


def test_cli_surface_exposes_real_training_flags() -> None:
    from work.openpi.pipelines.recap import v22_train_variants_real as trainer

    help_text = trainer.build_parser().format_help()

    for flag in (
        "--canonical-anchor",
        "--num-train-steps",
        "--batch-size",
        "--preregistration-skeleton",
        "--emit-loss-decomposition",
        "--emit-threshold-trace",
        "--emit-alpha-dual-trace",
        "--save-interval-override",
        "--cuda-visible-devices",
    ):
        assert flag in help_text


def test_explicit_save_interval_override_resolves_runtime_only(
    tmp_path: Path,
) -> None:
    from work.openpi.pipelines.recap import v22_train_variants_real as trainer

    argv = _base_args(tmp_path) + ["--save-interval-override", "37"]
    args = trainer.build_parser().parse_args(argv)
    request = trainer.request_from_args(args)
    anchor = trainer._load_canonical_anchor(request.canonical_anchor)
    resources = trainer._resolve_resources(request, anchor)
    export_request = trainer._build_real_export_request(request, resources)

    assert anchor.save_interval == 1000
    assert request.save_interval_override == 37
    assert resources.save_interval == 37
    assert resources.save_interval_source == "real_training_runtime_override"
    assert export_request.default_save_interval == 37


def test_graceful_timeout_writes_incomplete_manifest_and_checkpoint_evidence(
    tmp_path: Path,
) -> None:
    from work.openpi.pipelines.recap import v22_train_variants_real as trainer

    args = trainer.build_parser().parse_args(_base_args(tmp_path))
    request = trainer.request_from_args(args)
    request.training.output_dir.mkdir(parents=True, exist_ok=True)
    anchor = trainer._load_canonical_anchor(request.canonical_anchor)
    resources = trainer._resolve_resources(request, anchor)
    precondition = trainer._build_real_precondition(request, anchor, resources)
    checkpoint_run_dir = (
        request.training.runtime_log_dir
        / "real_variant_export_runtime"
        / "upstream_train_checkpoints"
        / trainer.TRAIN_CONFIG_NAME
        / request.training.variant_id
    )
    for step in (200, 400):
        (checkpoint_run_dir / str(step) / "params").mkdir(parents=True)
        (checkpoint_run_dir / str(step) / "params" / "_METADATA").write_text(
            f"checkpoint {step}\n",
            encoding="utf-8",
        )

    trainer._write_graceful_timeout_report(
        request,
        anchor,
        resources,
        precondition,
        signum=signal.SIGTERM,
    )

    output_dir = tmp_path / "B"
    report = json.loads((output_dir / "training_timeout_report.json").read_text())
    manifest = json.loads((output_dir / "training_run_manifest.json").read_text())

    assert report["schema_version"] == "v22_variant_real_train_timeout_report_v1"
    assert report["status"] == "GRACEFUL_TIMEOUT"
    assert report["completion_status"] == "INCOMPLETE"
    assert report["terminal_reason"] == "graceful_timeout_sigterm"
    assert report["signal"] == signal.SIGTERM
    assert report["return_code"] == trainer.GRACEFUL_TIMEOUT_RETURN_CODE
    assert report["save_interval"] == trainer.DEFAULT_REAL_SAVE_INTERVAL_OVERRIDE
    assert report["save_interval_source"] == "real_training_runtime_override"
    assert report["save_interval_override_default"] == 200
    assert report["checkpoint_run_dir_exists"] is True
    assert report["last_step"] == 400
    expected_checkpoint_suffix = (
        "upstream_train_checkpoints/pi0_libero/"
        + "control_no_recap_shuffled_adversarial_relabel/400"
    )
    assert str(report["last_checkpoint"]).endswith(expected_checkpoint_suffix)
    assert report["last_checkpoint_path"] == report["last_checkpoint"]
    assert report["last_checkpoint_tree_sha256"]
    assert manifest["schema_version"] == "v22_variant_real_train_manifest_v1"
    assert manifest["status"] == "GRACEFUL_TIMEOUT"
    assert manifest["completion_status"] == "INCOMPLETE"
    assert manifest["terminal_reason"] == "graceful_timeout_sigterm"
    assert manifest["last_step"] == 400
    assert manifest["last_checkpoint"] == report["last_checkpoint"]
    assert manifest["save_interval"] == trainer.DEFAULT_REAL_SAVE_INTERVAL_OVERRIDE
    assert manifest["save_interval_source"] == "real_training_runtime_override"


def test_run_training_handles_real_sigterm_path_and_returns_timeout_code(
    monkeypatch, tmp_path: Path
) -> None:
    from work.openpi.pipelines.recap import v22_train_variants_real as trainer

    def fake_export(request):
        checkpoint_run_dir = (
            request.runtime_dir
            / "upstream_train_checkpoints"
            / trainer.TRAIN_CONFIG_NAME
            / request.variant_name
        )
        for step in (200, 400):
            (checkpoint_run_dir / str(step) / "params").mkdir(parents=True)
            (checkpoint_run_dir / str(step) / "params" / "_METADATA").write_text(
                f"checkpoint {step}\n",
                encoding="utf-8",
            )
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("unreachable after SIGTERM")

    monkeypatch.setattr(trainer, "run_real_variant_training_export", fake_export)

    args = trainer.build_parser().parse_args(_base_args(tmp_path))
    request = trainer.request_from_args(args)

    rc = trainer.run_training(request)

    output_dir = tmp_path / "B"
    report = json.loads((output_dir / "training_timeout_report.json").read_text())
    manifest = json.loads((output_dir / "training_run_manifest.json").read_text())
    runtime_log = (tmp_path / "logs" / "v22_train_variants_real.log").read_text(
        encoding="utf-8"
    )

    assert rc == trainer.GRACEFUL_TIMEOUT_RETURN_CODE
    assert report["status"] == "GRACEFUL_TIMEOUT"
    assert manifest["status"] == "GRACEFUL_TIMEOUT"
    assert report["last_step"] == 400
    assert manifest["last_step"] == 400
    assert "return_code=124" in runtime_log


def test_patch_ordering_installs_jsonl_appender_and_restores(tmp_path: Path) -> None:
    from work.openpi.recap import real_variant_export as export_mod

    calls: list[tuple[dict[str, float], int | None]] = []

    def original_train_step() -> None:
        return None

    def original_loss_fn() -> None:
        return None

    def original_log(data: dict[str, float], *, step: int | None = None, **_: object) -> None:
        calls.append((data, step))

    train_main_mod = SimpleNamespace(
        train_step=original_train_step,
        loss_fn=original_loss_fn,
        wandb=SimpleNamespace(log=original_log),
        at=SimpleNamespace(typecheck=lambda fn: fn),
    )
    validate, finalize = export_mod._install_v22_real_training_hooks(
        train_main_mod,
        trace_config=export_mod.V22RealTraceConfig(
            trace_dir=tmp_path,
            run_id="run",
            variant="B",
            emit_loss_decomposition=True,
            emit_threshold_trace=True,
            emit_alpha_dual_trace=False,
            enable_r2_phase_threshold_switching=True,
            enable_r4_alpha_dual_loss=False,
            phase_threshold_step=1,
            alpha_pre_phase=0.0,
            alpha_post_phase=1.0,
        ),
    )

    validate()
    assert train_main_mod.train_step is not original_train_step
    assert train_main_mod.loss_fn is not original_loss_fn
    assert train_main_mod.wandb.log is not original_log

    train_main_mod.wandb.log(
        {
            "loss": 2.0,
            "grad_norm": 3.0,
            "param_norm": 4.0,
            "flow_loss": 1.0,
            "discrete_action_ce": 0.5,
            "text_ce": 0.25,
        },
        step=0,
    )

    finalize()
    assert train_main_mod.train_step is original_train_step
    assert train_main_mod.loss_fn is original_loss_fn
    assert train_main_mod.wandb.log is original_log
    assert calls
    rows = [
        json.loads(line)
        for line in (tmp_path / "loss_decomposition.jsonl").read_text().splitlines()
    ]
    assert rows[0]["schema_version"] == "v22_variant_loss_decomposition_real_v1"
    assert rows[0]["flow_loss"] == 1.0


def test_jsonl_appender_survives_wandb_init_replacing_log(tmp_path: Path) -> None:
    from work.openpi.recap import real_variant_export as export_mod

    calls: list[tuple[str, dict[str, float], int | None]] = []

    def original_train_step() -> None:
        return None

    def original_loss_fn() -> None:
        return None

    def preinit_log(
        data: dict[str, float],
        *,
        step: int | None = None,
        **_: object,
    ) -> None:
        calls.append(("preinit", data, step))

    def postinit_log(
        data: dict[str, float],
        *,
        step: int | None = None,
        **_: object,
    ) -> None:
        calls.append(("postinit", data, step))

    def init_wandb(*_: object, **__: object) -> str:
        train_main_mod.wandb.log = postinit_log
        return "disabled-run"

    train_main_mod = SimpleNamespace(
        train_step=original_train_step,
        loss_fn=original_loss_fn,
        init_wandb=init_wandb,
        wandb=SimpleNamespace(log=preinit_log),
        at=SimpleNamespace(typecheck=lambda fn: fn),
    )
    validate, finalize = export_mod._install_v22_real_training_hooks(
        train_main_mod,
        trace_config=export_mod.V22RealTraceConfig(
            trace_dir=tmp_path,
            run_id="run",
            variant="B",
            emit_loss_decomposition=True,
            emit_threshold_trace=True,
            emit_alpha_dual_trace=False,
            enable_r2_phase_threshold_switching=False,
            enable_r4_alpha_dual_loss=False,
            phase_threshold_step=1,
            alpha_pre_phase=0.0,
            alpha_post_phase=1.0,
        ),
    )

    validate()
    assert train_main_mod.init_wandb is not init_wandb
    assert train_main_mod.init_wandb() == "disabled-run"
    assert train_main_mod.wandb.log is not postinit_log

    payload = {
        "loss": 2.0,
        "grad_norm": 3.0,
        "param_norm": 4.0,
        "flow_loss": 1.0,
        "discrete_action_ce": 0.5,
        "text_ce": 0.25,
    }
    train_main_mod.wandb.log(payload, step=7)

    finalize()
    assert train_main_mod.init_wandb is init_wandb
    assert train_main_mod.wandb.log is postinit_log
    assert calls == [("postinit", payload, 7)]
    rows = [
        json.loads(line)
        for line in (tmp_path / "loss_decomposition.jsonl").read_text().splitlines()
    ]
    assert rows[0]["schema_version"] == "v22_variant_loss_decomposition_real_v1"
    assert rows[0]["step"] == 7


def test_alpha_trace_emits_for_x_when_requested(tmp_path: Path) -> None:
    from work.openpi.recap import real_variant_export as export_mod

    export_mod._append_v22_real_trace_rows(
        export_mod.V22RealTraceConfig(
            trace_dir=tmp_path,
            run_id="run",
            variant="X",
            emit_loss_decomposition=False,
            emit_threshold_trace=False,
            emit_alpha_dual_trace=True,
            enable_r2_phase_threshold_switching=True,
            enable_r4_alpha_dual_loss=True,
            phase_threshold_step=1,
            alpha_pre_phase=0.0,
            alpha_post_phase=1.0,
        ),
        {
            "loss": 2.0,
            "grad_norm": 3.0,
            "param_norm": 4.0,
            "flow_loss": 1.0,
            "discrete_action_ce": 0.5,
            "text_ce": 0.25,
        },
        step=0,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "alpha_dual_loss_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows == [
        {
            "schema_version": "v22_variant_alpha_dual_loss_trace_real_v1",
            "run_id": "run",
            "variant": "X",
            "step": 0,
            "alpha": 0.0,
            "dual_loss_components": {
                "flow_loss": 1.0,
                "discrete_action_ce": 0.5,
                "text_ce": 0.25,
            },
            "total_alpha_dual_loss": 2.0,
        }
    ]


def test_real_wrapper_smoke_records_two_real_trace_rows(
    monkeypatch, tmp_path: Path
) -> None:
    from work.openpi.pipelines.recap import v22_train_variants_real as trainer
    from work.openpi.recap.real_variant_export import RealVariantExportBundle

    captured_env: dict[str, str] = {}

    def fake_export(request):
        captured_env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        captured_env["OPENPI_VARIANT_TRAIN_NUM_STEPS"] = os.environ[
            "OPENPI_VARIANT_TRAIN_NUM_STEPS"
        ]
        captured_env["OPENPI_VARIANT_TRAIN_SAVE_INTERVAL"] = os.environ[
            "OPENPI_VARIANT_TRAIN_SAVE_INTERVAL"
        ]
        captured_env["OPENPI_VARIANT_TRAIN_SAVE_INTERVAL_SOURCE"] = os.environ[
            "OPENPI_VARIANT_TRAIN_SAVE_INTERVAL_SOURCE"
        ]
        assert request.train_config_name == "pi0_libero"
        assert request.default_num_train_steps == 2
        assert request.default_save_interval == 200
        assert request.log_interval == 1
        assert request.v22_trace_dir is not None
        trace_dir = Path(request.v22_trace_dir)
        loss_rows = [
            {
                "schema_version": "v22_variant_loss_decomposition_real_v1",
                "run_id": "stage1_v22_full_training_eval_iter9_20260426T_nextZ",
                "variant": "B",
                "step": 0,
                "loss": 2.0,
                "grad_norm": 1.0,
                "param_norm": 5.0,
                "flow_loss": 1.0,
                "discrete_action_ce": 0.5,
                "text_ce": 0.25,
            },
            {
                "schema_version": "v22_variant_loss_decomposition_real_v1",
                "run_id": "stage1_v22_full_training_eval_iter9_20260426T_nextZ",
                "variant": "B",
                "step": 1,
                "loss": 1.5,
                "grad_norm": 1.1,
                "param_norm": 5.1,
                "flow_loss": 0.75,
                "discrete_action_ce": 0.4,
                "text_ce": 0.2,
            },
        ]
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "loss_decomposition.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in loss_rows),
            encoding="utf-8",
        )
        (trace_dir / "threshold_switch_trace.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": "v22_variant_threshold_switch_trace_real_v1",
                    "run_id": "stage1_v22_full_training_eval_iter9_20260426T_nextZ",
                    "variant": "B",
                    "step": 0,
                    "phase": "pre_threshold",
                    "threshold_value": 1,
                    "switch_event": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        export_dir = request.runtime_dir / "fake_export"
        (export_dir / "params").mkdir(parents=True)
        (export_dir / "assets" / "physical-intelligence" / "libero").mkdir(
            parents=True
        )
        (export_dir / "params" / "_METADATA").write_text("state", encoding="utf-8")
        (export_dir / "params" / "array").write_text("nonempty", encoding="utf-8")
        runtime_log = request.runtime_dir / "real_variant_training.log"
        runtime_log.parent.mkdir(parents=True, exist_ok=True)
        runtime_log.write_text("fake real export\n", encoding="utf-8")
        return RealVariantExportBundle(export_dir=export_dir, runtime_log_path=runtime_log)

    monkeypatch.setattr(trainer, "run_real_variant_training_export", fake_export)

    rc = trainer.main(_base_args(tmp_path))

    output_dir = tmp_path / "B"
    rows = [
        json.loads(line)
        for line in (output_dir / "loss_decomposition.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((output_dir / "training_run_manifest.json").read_text())
    gradient = json.loads((output_dir / "gradient_attestation.json").read_text())

    assert rc == 0
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured_env["OPENPI_VARIANT_TRAIN_NUM_STEPS"] == "2"
    assert captured_env["OPENPI_VARIANT_TRAIN_SAVE_INTERVAL"] == "200"
    assert (
        captured_env["OPENPI_VARIANT_TRAIN_SAVE_INTERVAL_SOURCE"]
        == "real_training_runtime_override"
    )
    assert manifest["num_train_steps"] == 2
    assert manifest["batch_size"] == 1
    assert manifest["num_train_steps_source"] == "explicit_cli_override"
    assert manifest["canonical_anchor_save_interval"] == 1000
    assert manifest["save_interval"] == 200
    assert manifest["save_interval_source"] == "real_training_runtime_override"
    assert manifest["save_interval_override"] == 200
    assert manifest["save_interval_override_default"] == 200
    assert manifest["total_step_count_in_loss_decomposition_jsonl"] == 2
    assert len(rows) == 2
    assert rows[0]["flow_loss"] != rows[1]["flow_loss"], "loss must change across steps"
    assert rows[0]["step"] == 0 and rows[1]["step"] == 1, "step counter monotonic"
    assert {
        rows[0]["schema_version"],
        rows[1]["schema_version"],
    } == {"v22_variant_loss_decomposition_real_v1"}
    assert all(
        isinstance(rows[0][key], float)
        for key in ("loss", "flow_loss", "discrete_action_ce", "text_ce")
    )
    assert (output_dir / "checkpoint" / "params" / "_METADATA").stat().st_size > 0
    assert gradient["placeholder"] is False
    assert gradient["gradient_sha256"]
    assert (output_dir / "SHA256SUMS").is_file()
