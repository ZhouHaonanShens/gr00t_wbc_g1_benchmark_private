from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import stage3_contract_precondition_gate


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return dict(payload)


def _load_script_module(script_name: str, module_name: str):
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_repo_fixture(
    tmp_path: Path, *, checkpoint_rel: str | None
) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    (repo_root / "agent").mkdir(parents=True, exist_ok=True)
    (repo_root / "AGENTS.md").write_text("fixture\n", encoding="utf-8")
    manifest_path = _write_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json",
        {
            "schema_version": "stage3_iteration_manifest_v3",
            "artifact_root": "agent/artifacts/stage3_iteration/recap_stage3_iter_002/",
            "collect_policy_ckpt_path": checkpoint_rel,
        },
    )
    return repo_root, manifest_path


def _checkpoint_index(
    repo_root: Path, checkpoint_rel: str, *, with_adv_pair: bool
) -> Path:
    checkpoint_dir = repo_root / checkpoint_rel
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {
        "action_head.output_proj.weight": "model-00001-of-00001.safetensors",
    }
    if with_adv_pair:
        weight_map.update(
            {
                "action_head.advantage_embedding.weight": "model-00001-of-00001.safetensors",
                "action_head.advantage_embedding.bias": "model-00001-of-00001.safetensors",
            }
        )
    _write_json(
        checkpoint_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 1},
            "weight_map": weight_map,
        },
    )
    return checkpoint_dir


def test_contract_gate_selects_adv_server_for_checkpoint_with_advantage_embedding(
    tmp_path: Path,
) -> None:
    checkpoint_rel = "agent/artifacts/checkpoints/iter002_candidate/checkpoint-8"
    repo_root, manifest_path = _make_repo_fixture(
        tmp_path, checkpoint_rel=checkpoint_rel
    )
    _checkpoint_index(repo_root, checkpoint_rel, with_adv_pair=True)

    result = stage3_contract_precondition_gate.run_stage3_contract_precondition_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    gate_payload = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/contract_precondition_gate.json"
    )
    manifest = _read_json(manifest_path)
    assert result["pass"] is True
    assert gate_payload["pass"] is True
    assert gate_payload["prelim_eval_surface"]["mode"] == "adv_server_required"
    assert gate_payload["require_advantage_embedding"] is True
    assert gate_payload["allow_baseline_default_advantage_embedding_init"] is False
    assert (
        gate_payload["checkpoint_weight_map_features"]["has_advantage_embedding_pair"]
        is True
    )
    assert manifest["prelim_eval_surface"]["mode"] == "adv_server_required"
    assert manifest["prelim_eval_require_advantage_embedding"] is True
    assert (
        manifest["prelim_eval_allow_baseline_default_advantage_embedding_init"] is False
    )
    assert manifest["contract_precondition_gate"]["pass"] is True


def test_contract_gate_selects_baseline_default_init_for_explicit_baseline_path(
    tmp_path: Path,
) -> None:
    checkpoint_rel = (
        "agent/artifacts/stage3_iteration/recap_stage3_iter_002/"
        "baseline_train_attempt_001/finetune_output/checkpoint-4"
    )
    repo_root, manifest_path = _make_repo_fixture(
        tmp_path, checkpoint_rel=checkpoint_rel
    )
    _checkpoint_index(repo_root, checkpoint_rel, with_adv_pair=False)

    result = stage3_contract_precondition_gate.run_stage3_contract_precondition_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    gate_payload = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/contract_precondition_gate.json"
    )
    manifest = _read_json(manifest_path)
    assert result["pass"] is True
    assert gate_payload["prelim_eval_surface"]["mode"] == "baseline_default_adv_init"
    assert gate_payload["require_advantage_embedding"] is True
    assert gate_payload["allow_baseline_default_advantage_embedding_init"] is True
    assert (
        gate_payload["checkpoint_weight_map_features"]["has_advantage_embedding_pair"]
        is False
    )
    assert gate_payload["checkpoint_weight_map_features"]["baseline_like_path"] is True
    assert manifest["prelim_eval_surface"]["mode"] == "baseline_default_adv_init"
    assert manifest["contract_precondition_gate"]["pass"] is True


def test_contract_gate_blocks_incompatible_checkpoint_as_inconclusive_contract_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_rel = "agent/artifacts/checkpoints/formal_eval_candidate/checkpoint-12"
    repo_root, manifest_path = _make_repo_fixture(
        tmp_path, checkpoint_rel=checkpoint_rel
    )
    _checkpoint_index(repo_root, checkpoint_rel, with_adv_pair=False)

    result = stage3_contract_precondition_gate.run_stage3_contract_precondition_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    gate_payload = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/contract_precondition_gate.json"
    )
    manifest = _read_json(manifest_path)
    assert result["pass"] is False
    assert result["status"] == "inconclusive_contract_mismatch"
    assert gate_payload["failure_status_if_blocked"] == "inconclusive_contract_mismatch"
    assert (
        gate_payload["prelim_eval_surface"]["mode"] == "incompatible_checkpoint_surface"
    )
    assert gate_payload["require_advantage_embedding"] is False
    assert gate_payload["allow_baseline_default_advantage_embedding_init"] is False
    assert gate_payload["failure_reason_codes"] == [
        "checkpoint_missing_advantage_embedding_for_selected_surface"
    ]
    assert manifest["contract_precondition_gate"]["pass"] is False


def test_contract_gate_cli_exits_zero_for_unbound_manifest(
    tmp_path: Path, capsys: Any
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path, checkpoint_rel=None)
    script = _load_script_module(
        "30c_stage3_contract_precondition_gate.py",
        "stage3_contract_precondition_gate_cli_test",
    )
    old_repo_root = getattr(script, "REPO_ROOT")
    setattr(script, "REPO_ROOT", repo_root)
    try:
        exit_code = script.main(["--iteration-manifest", str(manifest_path)])
    finally:
        setattr(script, "REPO_ROOT", old_repo_root)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["pass"] is False
    assert payload["status"] == "inconclusive_contract_mismatch"
    assert payload["prelim_eval_surface"]["mode"] == "checkpoint_binding_missing"
