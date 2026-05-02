from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import cast

from _pytest.monkeypatch import MonkeyPatch
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.checkpoint import load_provenance_pair  # noqa: E402
from work.openpi.contracts import RuntimeServerSpec  # noqa: E402
from work.openpi.recap.runtime_prompt import (  # noqa: E402
    build_runtime_prompt_bundle,
    resolve_runtime_indicator_config,
)
from work.openpi.model import (  # noqa: E402
    build_effective_runtime_spec,
    build_rollout_input_summary_v21,
    effective_runtime_spec_hash,
)
from work.openpi.runtime.bridge import (  # noqa: E402
    _build_explicit_infer_element,
    _run_runtime_episode_subprocess,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def test_load_provenance_pair_keeps_parent_compat_lookup(tmp_path: Path) -> None:
    source_dir = tmp_path / "bundle" / "_staging"
    source_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = {
        "schema_version": "train_manifest_v1",
        "base_checkpoint_id": "pi05_libero_anchor",
    }
    checkpoint_provenance = {
        "schema_version": "checkpoint_provenance_v1",
        "base_checkpoint_id": "pi05_libero_anchor",
    }
    _write_json(source_dir.parent / "train_manifest.json", train_manifest)
    _write_json(source_dir.parent / "checkpoint_provenance.json", checkpoint_provenance)

    observed_train_manifest, observed_checkpoint_provenance = load_provenance_pair(
        source_dir
    )

    assert observed_train_manifest == train_manifest
    assert observed_checkpoint_provenance == checkpoint_provenance


def test_model_builds_root_runtime_summary_and_hash(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "policy" / "best"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        checkpoint_dir / "checkpoint.json",
        {
            "schema_version": "openpi_libero_recap_checkpoint_payload_v1",
            "instance_token": "root-model-test",
        },
    )
    config = resolve_runtime_indicator_config(
        requested_indicator_mode="cfg",
        variant="recap_only_relabel8d_v2",
        checkpoint_provenance={
            "variant_derivation": {"consumer_mode": "informative_adv"}
        },
    )
    prompt_bundle = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=config,
    )
    spec = build_effective_runtime_spec(
        variant="recap_only_relabel8d_v2",
        checkpoint_ref=str(checkpoint_dir),
        runtime_indicator_config=config,
        prompt_surface_bundle=prompt_bundle,
        key_files=(Path("checkpoint.json"),),
        binding_schema_version="openpi_libero_checkpoint_instance_binding_v1",
        runtime_spec_schema_version="openpi_libero_effective_runtime_spec_v1",
    )
    summary = build_rollout_input_summary_v21(
        schema_version="openpi_libero_rollout_eval_v21_input_v2",
        variant="recap_only_relabel8d_v2",
        checkpoint_ref=str(checkpoint_dir),
        serve_checkpoint_ref=str(checkpoint_dir),
        serve_checkpoint_mode="local_orbax_checkpoint",
        task_suite_name="libero_spatial",
        task_seed_manifests=((0, (7,)),),
        manifest={"seed_manifest": [7]},
        num_trials_per_task=1,
        server_spec=RuntimeServerSpec(
            host="127.0.0.1",
            port=8000,
            checkpoint_dir=str(checkpoint_dir),
            server_ready_timeout_s=150.0,
            client_timeout_s=80.0,
        ),
        server_log=tmp_path / "server.log",
        harness_log=tmp_path / "harness.log",
        episode_count=1,
        runtime_indicator_config=config,
        prompt_surface_bundle=prompt_bundle,
        key_files=(Path("checkpoint.json"),),
        binding_schema_version="openpi_libero_checkpoint_instance_binding_v1",
        runtime_spec_schema_version="openpi_libero_effective_runtime_spec_v1",
    )

    assert summary["effective_runtime_spec"] == spec
    assert summary["effective_runtime_spec_hash"] == effective_runtime_spec_hash(spec)


def test_run_runtime_episode_subprocess_reads_client_summary(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    def _fake_run(command: list[str], **_: object) -> SimpleNamespace:
        summary_path = Path(command[command.index("--client-summary-out") + 1])
        _write_json(
            summary_path,
            {
                "episode_results": [
                    {
                        "success": True,
                        "steps_observed": 42,
                        "inference_calls": 9,
                        "error": "",
                    }
                ],
                "runtime_prompting": {
                    "indicator_mode_requested": "cfg",
                    "indicator_mode": "positive",
                    "indicator_source": "cfg.consumer_mode.informative_adv",
                    "prompt_text_surface": "canonical_text_indicator",
                    "prompt_route": "recap_conditioned_prompt_token_v1",
                    "conditioning_mode": "prompt_text_only",
                    "consumer_mode": "informative_adv",
                    "fixed_indicator_mode": "",
                    "critic_checkpoint_ref": "adapter_required",
                    "source_prompt_field": "prompt_raw",
                    "prompt_text": "put the bowl on the plate\nAdvantage: positive",
                },
            },
        )
        return SimpleNamespace(
            returncode=0,
            stdout="LIBERO_NATIVE_CLIENT_DONE\n",
            stderr="",
        )

    monkeypatch.setattr(
        "work.openpi.runtime.bridge.subprocess.run",
        _fake_run,
    )

    result = _run_runtime_episode_subprocess(
        task_suite_name="libero_spatial",
        task_id=0,
        seed=7,
        trial_idx=0,
        video_path=tmp_path / "video.mp4",
        host="127.0.0.1",
        port=8000,
        venv_python=tmp_path / "python",
        openpi_root=tmp_path / "openpi_root",
        libero_config_dir=tmp_path / "libero_config",
        runtime_dir=tmp_path / "runtime",
        timeout_s=15.0,
        checkpoint_ref=str(tmp_path / "policy" / "best"),
        indicator_mode_requested="cfg",
        runtime_indicator_config=SimpleNamespace(
            indicator_mode="positive",
            indicator_source="cfg.consumer_mode.informative_adv",
            consumer_mode="informative_adv",
            fixed_indicator_mode=None,
            critic_checkpoint_ref="adapter_required",
        ),
        cli_entry=tmp_path / "libero_native_smoke.py",
    )

    assert result["success"] is True
    assert result["steps_observed"] == 42
    assert result["indicator_mode"] == "positive"
    assert result["prompt_text_surface"] == "canonical_text_indicator"
    assert cast(str, result["client_log"]).endswith("client.log")


def test_build_explicit_infer_element_keeps_runtime_bridge_explicit_input_only() -> (
    None
):
    element = _build_explicit_infer_element(
        image=[[1]],
        wrist_image=[[2]],
        state=[0.1, 0.2],
        prompt="put the bowl on the plate",
    )

    assert set(element) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert element["prompt"] == "put the bowl on the plate"
