from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    _ = sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.real_variant_export import (  # noqa: E402
    RealVariantExportBundle,
)
from work.openpi.scripts import openpi_recap_one_step_probe  # noqa: E402


def _write_p0_manifest(path: Path, *, materialized: bool) -> None:
    payload = {
        "dataset_join_root": str((path.parent / "joined_dataset").resolve()),
        "dataset_join_root_status": {
            "materialized": materialized,
            "blocking_reason": "" if materialized else "dataset_not_materialized",
            "loader_exception_summary": "" if materialized else "materialized=False, sample_file_count=0",
            "status": "materialized=true" if materialized else "BLOCKED(dataset_not_materialized)",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_main_writes_blocked_probe_verdict_before_runtime_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "p0_scope_audit" / "scope_audit_manifest.json"
    output_root = tmp_path / "openpi_libero_recap_v2_full_update"
    _write_p0_manifest(manifest_path, materialized=False)

    def _unexpected_run(_: object) -> object:
        raise AssertionError("real one-step path must not run when dataset is not materialized")

    monkeypatch.setattr(
        openpi_recap_one_step_probe,
        "run_real_variant_training_export",
        _unexpected_run,
    )

    exit_code = openpi_recap_one_step_probe.main(
        [
            "--output-root",
            str(output_root),
            "--p0-scope-audit-manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 0
    payload = cast(
        dict[str, Any],
        json.loads(
            (output_root / "p1_one_step" / "one_step_probe.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    assert payload["status"] == "BLOCKED(dataset_not_materialized)"
    assert payload["verdict"] == "BLOCKED(dataset_not_materialized)"
    assert payload["skip_before_execute"] is True
    assert payload["runtime_started"] is False
    assert payload["probe_pass"] is False
    assert payload["blocking_reasons"] == ["dataset_not_materialized"]
    assert payload["loss_values"] == []
    assert payload["any_grad_nonzero"] is False
    assert payload["any_param_delta_nonzero"] is False
    assert payload["gpu"] is None
    assert payload["cuda_visible_devices"] is None
    assert payload["resolved_egl_device_pci_bus_id"] is None
    assert payload["command_context"]["gpu"] is None
    assert payload["command_context"]["planned_runtime_env"] == {
        "CUDA_VISIBLE_DEVICES": None,
        "JAX_PLATFORMS": None,
        "JAX_PLATFORM_NAME": None,
    }
    assert payload["command_context"]["expected_openpi_python"] == str(
        openpi_recap_one_step_probe.OPENPI_VENV_PYTHON
    )


def test_main_writes_probe_metrics_when_gate_is_green(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "p0_scope_audit" / "scope_audit_manifest.json"
    output_root = tmp_path / "openpi_libero_recap_v2_full_update"
    _write_p0_manifest(manifest_path, materialized=True)

    def _fake_run(request):
        assert request.probe_metrics_path is not None
        request.runtime_dir.mkdir(parents=True, exist_ok=True)
        request.probe_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        _ = request.probe_metrics_path.write_text(
            json.dumps(
                {
                    "loss_values": [1.25],
                    "any_grad_nonzero": True,
                    "any_param_delta_nonzero": True,
                    "probe_pass": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        runtime_log_path = request.runtime_dir / "real_variant_training.log"
        _ = runtime_log_path.write_text("ok\n", encoding="utf-8")
        export_dir = request.runtime_dir / "real_variant_export"
        export_dir.mkdir(parents=True, exist_ok=True)
        return RealVariantExportBundle(
            export_dir=export_dir,
            runtime_log_path=runtime_log_path,
        )

    monkeypatch.setattr(
        openpi_recap_one_step_probe,
        "run_real_variant_training_export",
        _fake_run,
    )

    exit_code = openpi_recap_one_step_probe.main(
        [
            "--output-root",
            str(output_root),
            "--p0-scope-audit-manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 0
    payload = cast(
        dict[str, Any],
        json.loads(
            (output_root / "p1_one_step" / "one_step_probe.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    assert payload["status"] == "PASS"
    assert payload["verdict"] == "PASS"
    assert payload["skip_before_execute"] is False
    assert payload["runtime_started"] is True
    assert payload["probe_pass"] is True
    assert payload["blocking_reasons"] == []
    assert payload["loss_values"] == [1.25]
    assert payload["any_grad_nonzero"] is True
    assert payload["any_param_delta_nonzero"] is True
    assert payload["gpu"] == 2
    assert payload["runtime_output"]["probe_metrics_path"].endswith(
        "p1_one_step/probe_metrics.json"
    )
