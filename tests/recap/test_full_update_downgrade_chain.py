from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
_UNSET = object()


def _load_script_module(module_name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERVISOR = _load_script_module(
    "full_update_scope_supervisor_34c",
    "work/recap/scripts/34c_full_update_scope_supervisor.py",
)
LAUNCH_FINETUNE = _load_script_module(
    "launch_finetune_use_ddp",
    "work/recap/launch_finetune_use_ddp.py",
)


class _FakeSmokeModule:
    def __init__(
        self,
        *,
        output_dir: Path,
        static_by_scope: dict[str, dict[str, object]],
        runtime_by_scope: dict[str, tuple[int, dict[str, object]]],
    ) -> None:
        self.output_dir = output_dir
        self.static_by_scope = static_by_scope
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
        audit = dict(self.static_by_scope[requested_scope_override])
        metadata_dir = self.output_dir / "repo_local_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        audit_path = metadata_dir / f"{requested_scope_override}_static_audit.json"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "train_scope_requested": requested_scope_override,
            "output_dir": str(self.output_dir),
            "metadata_dir": str(metadata_dir),
            "static_audit_path": str(audit_path),
            "static_audit": audit,
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
        rc, payload = self.runtime_by_scope[requested_scope_override]
        return rc, dict(payload)


def _make_args(tmp_path: Path, requested_scope: str) -> SimpleNamespace:
    return SimpleNamespace(
        recap_train_scope=requested_scope,
        output_dir=str(tmp_path / "authority_out"),
        summary_json="",
        allow_downgrade=True,
    )


def _static_audit(
    *,
    verdict: str = "PASS",
    block_reasons: list[str] | None = None,
    fits_available_memory: bool | None = True,
) -> dict[str, object]:
    return {
        "static_verdict": verdict,
        "static_block_reasons": [] if block_reasons is None else list(block_reasons),
        "memory_feasibility": {
            "fits_available_memory": fits_available_memory,
            "estimated_total_bytes": 1024,
        },
    }


def _runtime_payload(
    *,
    output_dir: Path,
    wrapper_status: str = "ok",
    error: str | None = None,
    selected_checkpoint_exists: bool = True,
    trainer_global_step: int | None = 1,
    grad_probe_after_backward: object = _UNSET,
    param_delta_after_step: object = _UNSET,
    all_major_grad_norms: object = _UNSET,
    all_major_param_delta: object = _UNSET,
) -> dict[str, object]:
    runtime_log_path = output_dir / "runtime.log"
    runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_log_path.write_text("runtime log\n", encoding="utf-8")
    if all_major_grad_norms is _UNSET:
        all_major_grad_norms = {
            "diffusion_trunk": 1.25,
            "advantage_embedding": 0.75,
        }
    if all_major_param_delta is _UNSET:
        all_major_param_delta = {
            "diffusion_trunk": 0.25,
            "advantage_embedding": 0.125,
        }
    if grad_probe_after_backward is _UNSET:
        grad_probe_after_backward = {
            "available": True,
            "trainer_global_step": 0,
            "scopes": {
                "diffusion_trunk": {"grad_l2_norm": 1.25, "grad_abs_max": 0.5},
                "advantage_embedding": {"grad_l2_norm": 0.75, "grad_abs_max": 0.25},
            },
        }
    if param_delta_after_step is _UNSET:
        param_delta_after_step = {
            "available": True,
            "trainer_global_step": trainer_global_step,
            "scopes": {
                "diffusion_trunk": {"delta_l2_norm": 0.25, "delta_abs_max": 0.125},
                "advantage_embedding": {"delta_l2_norm": 0.125, "delta_abs_max": 0.0625},
            },
        }
    return {
        "wrapper_status": wrapper_status,
        "runtime_log_path": str(runtime_log_path),
        "delegate_cmd": ["python", "34b"],
        "delegate_cmd_shell": "python 34b",
        "checkpoint_load_report_path": str(output_dir / "checkpoint_report.json"),
        "selected_checkpoint_path": str(output_dir / "checkpoint-10"),
        "selected_checkpoint_asset_path": str(output_dir / "checkpoint-10" / "model.safetensors"),
        "selected_checkpoint_exists": selected_checkpoint_exists,
        "trainer_global_step": trainer_global_step,
        "advantage_embedding_keys_present": True,
        "advantage_embedding_missing_keys": [],
        "grad_probe_after_backward": grad_probe_after_backward,
        "param_delta_after_step": param_delta_after_step,
        "all_major_grad_norms": all_major_grad_norms,
        "all_major_param_delta": all_major_param_delta,
        "error": error,
    }


def _read_dynamic_audit(output_dir: Path) -> dict[str, object]:
    path = output_dir / "repo_local_metadata" / "full_update_scope_audit_dynamic.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_attempts(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "repo_local_metadata" / "downgrade_attempts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_supervisor(
    tmp_path: Path,
    monkeypatch,
    *,
    requested_scope: str,
    static_by_scope: dict[str, dict[str, object]],
    runtime_by_scope: dict[str, tuple[int, dict[str, object]]],
    allow_downgrade: bool = True,
) -> tuple[int, Path]:
    output_dir = tmp_path / "authority_out"
    fake_smoke = _FakeSmokeModule(
        output_dir=output_dir,
        static_by_scope=static_by_scope,
        runtime_by_scope=runtime_by_scope,
    )
    monkeypatch.setattr(SUPERVISOR, "_load_smoke_module", lambda: fake_smoke)
    monkeypatch.setattr(SUPERVISOR, "_repo_root", lambda: tmp_path)
    args = _make_args(tmp_path, requested_scope)
    args.allow_downgrade = allow_downgrade
    rc = SUPERVISOR.run_numeric_adv_scope_supervisor(args)
    return rc, output_dir


def test_explicit_degrade_chain(tmp_path: Path, monkeypatch) -> None:
    oom_log = tmp_path / "authority_out" / "strict_full_oom.log"
    oom_log.parent.mkdir(parents=True, exist_ok=True)
    oom_log.write_text("CUDA out of memory\n", encoding="utf-8")
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={
            "strict_full": _static_audit(),
            "full_policy": _static_audit(),
        },
        runtime_by_scope={
            "strict_full": (
                1,
                {
                    **_runtime_payload(output_dir=tmp_path / "authority_out", wrapper_status="blocked"),
                    "runtime_log_path": str(oom_log),
                    "error": "CUDA out of memory",
                },
            ),
            "full_policy": (0, _runtime_payload(output_dir=tmp_path / "authority_out")),
        },
    )

    assert rc == 0
    dynamic_audit = _read_dynamic_audit(output_dir)
    attempts = _read_attempts(output_dir)
    assert dynamic_audit["resolution_status"] == "DEGRADE"
    assert dynamic_audit["train_scope_effective"] == "full_policy"
    assert dynamic_audit["runtime_supervisor_used"] is True
    assert dynamic_audit["strict_full_runtime_attempted"] is True
    assert dynamic_audit["best_scope_authority"] is True
    assert [attempt["candidate_scope"] for attempt in attempts] == ["strict_full", "full_policy"]
    assert [attempt["status"] for attempt in attempts] == ["OOM", "PASS"]


def test_allow_downgrade_false_blocks_after_first_failure(tmp_path: Path, monkeypatch) -> None:
    oom_log = tmp_path / "authority_out" / "strict_full_no_downgrade.log"
    oom_log.parent.mkdir(parents=True, exist_ok=True)
    oom_log.write_text("CUDA out of memory\n", encoding="utf-8")
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        allow_downgrade=False,
        static_by_scope={
            "strict_full": _static_audit(),
            "full_policy": _static_audit(),
        },
        runtime_by_scope={
            "strict_full": (
                1,
                {
                    **_runtime_payload(output_dir=tmp_path / "authority_out", wrapper_status="blocked"),
                    "runtime_log_path": str(oom_log),
                    "error": "CUDA out of memory",
                },
            ),
            "full_policy": (0, _runtime_payload(output_dir=tmp_path / "authority_out")),
        },
    )

    assert rc == 1
    dynamic_audit = _read_dynamic_audit(output_dir)
    attempts = _read_attempts(output_dir)
    assert dynamic_audit["resolution_status"] == "BLOCK"
    assert dynamic_audit["failure_reason"] == "EXHAUSTED_DOWNGRADE_CHAIN"
    assert [attempt["candidate_scope"] for attempt in attempts] == ["strict_full"]
    assert attempts[0]["status"] == "OOM"


def test_success_path_requires_complete_probe_evidence(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="full_action",
        static_by_scope={"full_action": _static_audit()},
        runtime_by_scope={
            "full_action": (0, _runtime_payload(output_dir=tmp_path / "authority_out"))
        },
    )

    assert rc == 0
    dynamic_audit = _read_dynamic_audit(output_dir)
    attempts = _read_attempts(output_dir)
    assert dynamic_audit["resolution_status"] == "PASS"
    assert dynamic_audit["train_scope_effective"] == "full_action"
    assert dynamic_audit["runtime_supervisor_used"] is True
    assert dynamic_audit["strict_full_runtime_attempted"] is False
    assert dynamic_audit["best_scope_authority"] is True
    assert attempts[0]["status"] == "PASS"
    assert attempts[0]["trainer_global_step"] == 1


def test_missing_success_probe_evidence_blocks_even_if_runtime_returns_ok(
    tmp_path: Path, monkeypatch
) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="full_action",
        static_by_scope={"full_action": _static_audit()},
        runtime_by_scope={
            "full_action": (
                0,
                _runtime_payload(
                    output_dir=tmp_path / "authority_out",
                    grad_probe_after_backward=None,
                    all_major_grad_norms=None,
                ),
            )
        },
    )

    assert rc == 1
    dynamic_audit = _read_dynamic_audit(output_dir)
    attempts = _read_attempts(output_dir)
    assert dynamic_audit["resolution_status"] == "BLOCK"
    assert attempts[0]["status"] == "BLOCK"
    dynamic_block_reasons = attempts[0]["dynamic_block_reasons"]
    assert isinstance(dynamic_block_reasons, list)
    assert "MISSING_GRAD_EVIDENCE_PROBE" in dynamic_block_reasons


def test_no_fallback_to_current_partial(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={
            "strict_full": _static_audit(),
            "full_policy": _static_audit(),
            "full_action": _static_audit(),
        },
        runtime_by_scope={
            "strict_full": (1, {**_runtime_payload(output_dir=tmp_path / "authority_out", wrapper_status="blocked"), "error": "cuda out of memory"}),
            "full_policy": (1, {**_runtime_payload(output_dir=tmp_path / "authority_out", wrapper_status="blocked"), "error": "cuda out of memory"}),
            "full_action": (1, {**_runtime_payload(output_dir=tmp_path / "authority_out", wrapper_status="blocked"), "error": "cuda out of memory"}),
        },
    )

    assert rc == 1
    attempts = _read_attempts(output_dir)
    assert [attempt["candidate_scope"] for attempt in attempts] == [
        "strict_full",
        "full_policy",
        "full_action",
    ]
    assert all(attempt["candidate_scope"] != "current_partial" for attempt in attempts)


def test_zero_lr_block(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={
            "strict_full": _static_audit(
                verdict="BLOCK",
                block_reasons=["ZERO_LR_TRAINABLE_PARAM_GROUP"],
            )
        },
        runtime_by_scope={},
    )

    assert rc == 1
    dynamic_audit = _read_dynamic_audit(output_dir)
    attempts = _read_attempts(output_dir)
    assert dynamic_audit["resolution_status"] == "BLOCK"
    assert attempts[0]["status"] == "BLOCK"
    assert attempts[0]["runtime_attempted"] is False
    assert (output_dir / "p0_scope_audit" / "p0_block_report.md").is_file()
    assert (tmp_path / ".sisyphus" / "evidence" / "task-p0-block.md").is_file()


def test_missing_param_block(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="full_policy",
        static_by_scope={
            "full_policy": _static_audit(
                verdict="BLOCK",
                block_reasons=["TRAINABLE_MISSING_FROM_OPTIMIZER"],
            )
        },
        runtime_by_scope={},
    )

    assert rc == 1
    attempts = _read_attempts(output_dir)
    assert attempts[0]["status"] == "BLOCK"
    assert attempts[0]["static_block_reasons"] == ["TRAINABLE_MISSING_FROM_OPTIMIZER"]


def test_advantage_embedding_missing_from_optimizer_block(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="full_action",
        static_by_scope={
            "full_action": _static_audit(
                verdict="BLOCK",
                block_reasons=["ADVANTAGE_EMBEDDING_NOT_IN_OPTIMIZER"],
            )
        },
        runtime_by_scope={},
    )

    assert rc == 1
    attempts = _read_attempts(output_dir)
    assert attempts[0]["status"] == "BLOCK"
    assert attempts[0]["static_block_reasons"] == ["ADVANTAGE_EMBEDDING_NOT_IN_OPTIMIZER"]


def test_all_zero_grad_block(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="full_action",
        static_by_scope={"full_action": _static_audit()},
        runtime_by_scope={
            "full_action": (
                1,
                _runtime_payload(
                    output_dir=tmp_path / "authority_out",
                    wrapper_status="blocked",
                    error="all zero grad",
                    all_major_grad_norms={
                        "diffusion_trunk": 0.0,
                        "advantage_embedding": 0.0,
                    },
                ),
            )
        },
    )

    assert rc == 1
    dynamic_audit = _read_dynamic_audit(output_dir)
    attempts = _read_attempts(output_dir)
    assert dynamic_audit["failure_reason"] == "DYNAMIC_RUNTIME_BLOCK"
    assert attempts[0]["status"] == "BLOCK"
    assert attempts[0]["dynamic_block_reasons"] == ["ALL_ZERO_GRAD"]


def test_all_zero_delta_block(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="full_action",
        static_by_scope={"full_action": _static_audit()},
        runtime_by_scope={
            "full_action": (
                1,
                _runtime_payload(
                    output_dir=tmp_path / "authority_out",
                    wrapper_status="blocked",
                    error="all zero delta",
                    all_major_param_delta={
                        "diffusion_trunk": 0.0,
                        "advantage_embedding": 0.0,
                    },
                ),
            )
        },
    )

    assert rc == 1
    attempts = _read_attempts(output_dir)
    assert attempts[0]["status"] == "BLOCK"
    assert attempts[0]["dynamic_block_reasons"] == ["ALL_ZERO_PARAM_DELTA"]


def test_param_delta_probe_uses_optimizer_owned_parameters_when_model_wrapped_is_stale() -> None:
    class _ProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_head = torch.nn.Module()
            self.action_head.model = torch.nn.Linear(2, 2, bias=False)
            self.action_head.advantage_embedding = torch.nn.Linear(1, 2)

    live_model = _ProbeModel()
    stale_wrapped_model = _ProbeModel()
    stale_wrapped_model.load_state_dict(live_model.state_dict())
    stale_before = {
        name: param.detach().clone()
        for name, param in stale_wrapped_model.named_parameters()
    }
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=0),
        model_wrapped=stale_wrapped_model,
        model=live_model,
    )
    optimizer = torch.optim.SGD(live_model.parameters(), lr=0.1)

    pre_step_snapshot = LAUNCH_FINETUNE._capture_repo_local_first_step_parameter_snapshot(
        trainer=trainer,
        optimizer=optimizer,
    )
    for _, param in live_model.named_parameters():
        param.grad = torch.ones_like(param)

    optimizer.step()

    payload = LAUNCH_FINETUNE._build_rank0_first_optimizer_step_param_delta_payload(
        trainer=trainer,
        optimizer=optimizer,
        pre_step_snapshot=pre_step_snapshot,
    )

    assert payload["available"] is True
    assert payload["trainer_global_step"] == 1
    diffusion_trunk = payload["scopes"]["diffusion_trunk"]
    advantage_embedding = payload["scopes"]["advantage_embedding"]
    assert diffusion_trunk["snapshot_tensor_count"] == 1
    assert advantage_embedding["snapshot_tensor_count"] == 2
    assert diffusion_trunk["delta_l2_norm"] > 0.0
    assert advantage_embedding["delta_l2_norm"] > 0.0
    assert diffusion_trunk["nonzero_delta_tensor_count"] == 1
    assert advantage_embedding["nonzero_delta_tensor_count"] == 2
    for name, param in stale_wrapped_model.named_parameters():
        assert torch.equal(param.detach(), stale_before[name])


def test_memory_estimator_block(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={
            "strict_full": _static_audit(fits_available_memory=False),
            "full_policy": _static_audit(),
        },
        runtime_by_scope={
            "full_policy": (0, _runtime_payload(output_dir=tmp_path / "authority_out"))
        },
    )

    assert rc == 0
    dynamic_audit = _read_dynamic_audit(output_dir)
    attempts = _read_attempts(output_dir)
    assert dynamic_audit["resolution_status"] == "DEGRADE"
    assert attempts[0]["status"] == "MEMORY_ESTIMATOR_BLOCK"
    assert attempts[0]["runtime_attempted"] is False
    assert attempts[1]["status"] == "PASS"


def test_static_block_propagates_to_p0_report_path(tmp_path: Path, monkeypatch) -> None:
    rc, output_dir = _run_supervisor(
        tmp_path,
        monkeypatch,
        requested_scope="strict_full",
        static_by_scope={
            "strict_full": _static_audit(
                verdict="BLOCK",
                block_reasons=["DUPLICATE_OPTIMIZER_PARAM"],
            )
        },
        runtime_by_scope={},
    )

    assert rc == 1
    dynamic_audit = _read_dynamic_audit(output_dir)
    report_path = output_dir / "p0_scope_audit" / "p0_block_report.md"
    evidence_path = tmp_path / ".sisyphus" / "evidence" / "task-p0-block.md"
    assert dynamic_audit["p0_block_report_path"] == str(report_path)
    assert dynamic_audit["p0_block_evidence_path"] == str(evidence_path)
    assert "DUPLICATE_OPTIMIZER_PARAM" in report_path.read_text(encoding="utf-8")
