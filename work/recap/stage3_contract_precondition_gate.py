from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts.state_conditioned_common import read_json
from work.recap.scripts.state_conditioned_common import write_json


DEFAULT_STAGE3_ITER_002_MANIFEST_REL = Path(
    "agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json"
)
CONTRACT_PRECONDITION_GATE_JSON_NAME = "contract_precondition_gate.json"
STAGE3_ITERATION_MANIFEST_V3 = "stage3_iteration_manifest_v3"
CONTRACT_PRECONDITION_GATE_SCHEMA_VERSION = "stage3_contract_precondition_gate_v1"
CONTRACT_PRECONDITION_GATE_ARTIFACT_KIND = "stage3_contract_precondition_gate"
ADV_SERVER_REQUIRED = "adv_server_required"
BASELINE_DEFAULT_ADV_INIT = "baseline_default_adv_init"
CHECKPOINT_BINDING_MISSING = "checkpoint_binding_missing"
CHECKPOINT_PATH_MISSING = "checkpoint_path_missing"
CHECKPOINT_ASSET_MISSING = "checkpoint_asset_missing"
CHECKPOINT_KEYS_UNINSPECTABLE = "checkpoint_keys_uninspectable"
INCOMPATIBLE_CHECKPOINT_SURFACE = "incompatible_checkpoint_surface"
FAILURE_STATUS_INCONCLUSIVE_CONTRACT_MISMATCH = "inconclusive_contract_mismatch"
CONTINUE_STATUS = "continue"
ADVANTAGE_WEIGHT_KEY = "action_head.advantage_embedding.weight"
ADVANTAGE_BIAS_KEY = "action_head.advantage_embedding.bias"
BASELINE_PATH_TOKENS = (
    "baseline",
    "baseline_train",
    "baseline-trained",
    "baseline_trained",
    "smoke",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_relative_path(repo_root: Path, path: Path | str) -> str:
    resolved = _resolve_path(repo_root, path)
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _manifest_path(repo_root: Path, manifest_path: Path | None) -> Path:
    if manifest_path is not None:
        return _resolve_path(repo_root, manifest_path)
    return _resolve_path(repo_root, DEFAULT_STAGE3_ITER_002_MANIFEST_REL)


def _artifact_root(
    repo_root: Path, manifest_payload: Mapping[str, Any], manifest_path: Path
) -> Path:
    artifact_root_raw = str(manifest_payload.get("artifact_root") or "").strip()
    if artifact_root_raw:
        return _resolve_path(repo_root, artifact_root_raw)
    return manifest_path.parent.resolve()


def _selected_checkpoint_path(
    repo_root: Path, manifest_payload: Mapping[str, Any]
) -> tuple[Path | None, str | None]:
    raw = str(manifest_payload.get("collect_policy_ckpt_path") or "").strip()
    if not raw:
        return None, None
    return _resolve_path(repo_root, raw), "collect_policy_ckpt_path"


def _selected_checkpoint_asset(
    checkpoint_path: Path | None,
) -> tuple[Path | None, str | None]:
    if checkpoint_path is None:
        return None, None
    if checkpoint_path.is_file():
        suffixes = checkpoint_path.suffixes
        if checkpoint_path.name.endswith(".safetensors"):
            return checkpoint_path, "safetensors"
        if checkpoint_path.name.endswith(".bin"):
            return checkpoint_path, "pytorch_bin"
        if suffixes[-2:] == [".index", ".json"]:
            stem = checkpoint_path.name.removesuffix(".index.json")
            if stem.endswith(".safetensors"):
                return checkpoint_path, "safetensors_index_json"
            if stem.endswith(".bin"):
                return checkpoint_path, "pytorch_bin_index_json"
        return checkpoint_path, "unknown_file"
    if not checkpoint_path.is_dir():
        return None, None
    candidates: list[tuple[Path, str]] = [
        (checkpoint_path / "model.safetensors.index.json", "safetensors_index_json"),
        (checkpoint_path / "model.safetensors", "safetensors"),
        (checkpoint_path / "pytorch_model.bin.index.json", "pytorch_bin_index_json"),
        (checkpoint_path / "pytorch_model.bin", "pytorch_bin"),
    ]
    for candidate_path, asset_kind in candidates:
        if candidate_path.is_file():
            return candidate_path, asset_kind
    return None, None


def _load_checkpoint_key_names(asset_path: Path, asset_kind: str) -> list[str]:
    if asset_kind in {"safetensors_index_json", "pytorch_bin_index_json"}:
        payload = json.loads(asset_path.read_text(encoding="utf-8"))
        weight_map_raw = payload.get("weight_map")
        if not isinstance(weight_map_raw, Mapping):
            raise TypeError(
                f"checkpoint index at {asset_path} missing object weight_map"
            )
        return sorted(str(key) for key in weight_map_raw.keys())

    if asset_kind == "safetensors":
        from safetensors import safe_open

        with safe_open(str(asset_path), framework="pt", device="cpu") as handle:
            return sorted(str(key) for key in handle.keys())

    if asset_kind == "pytorch_bin":
        import torch

        raw_payload = torch.load(str(asset_path), map_location="cpu")
        if isinstance(raw_payload, Mapping):
            state_dict_raw = raw_payload.get("state_dict")
            state_dict = (
                state_dict_raw if isinstance(state_dict_raw, Mapping) else raw_payload
            )
            return sorted(str(key) for key in state_dict.keys())
        raise TypeError(
            f"unsupported pytorch checkpoint payload type at {asset_path}: {type(raw_payload).__name__}"
        )

    raise ValueError(f"unsupported checkpoint asset kind: {asset_kind}")


def _baseline_like_reasons(
    manifest_payload: Mapping[str, Any], checkpoint_path: Path | None
) -> list[str]:
    reasons: list[str] = []
    decision = str(manifest_payload.get("collect_policy_ckpt_decision") or "").strip()
    if decision in {"baseline_trained", "baseline_train_required"}:
        reasons.append(f"manifest_decision:{decision}")
    if checkpoint_path is not None:
        lowered = str(checkpoint_path).lower()
        for token in BASELINE_PATH_TOKENS:
            if token in lowered:
                reasons.append(f"path_token:{token}")
    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def _inspect_checkpoint_weight_map_features(
    *,
    repo_root: Path,
    manifest_payload: Mapping[str, Any],
    checkpoint_path: Path | None,
    checkpoint_source_field: str | None,
) -> dict[str, Any]:
    baseline_reasons = _baseline_like_reasons(manifest_payload, checkpoint_path)
    features: dict[str, Any] = {
        "checkpoint_source_field": checkpoint_source_field,
        "checkpoint_path": (
            _repo_relative_path(repo_root, checkpoint_path)
            if checkpoint_path is not None
            else None
        ),
        "checkpoint_bound": checkpoint_path is not None,
        "checkpoint_exists": bool(
            checkpoint_path is not None and checkpoint_path.exists()
        ),
        "checkpoint_path_kind": (
            "directory"
            if checkpoint_path is not None and checkpoint_path.is_dir()
            else "file"
            if checkpoint_path is not None and checkpoint_path.is_file()
            else "missing"
        ),
        "checkpoint_asset_path": None,
        "checkpoint_asset_kind": None,
        "checkpoint_asset_exists": False,
        "inspectable": False,
        "inspection_error": None,
        "key_count": 0,
        "has_advantage_embedding_weight": False,
        "has_advantage_embedding_bias": False,
        "has_advantage_embedding_pair": False,
        "baseline_like_path": bool(baseline_reasons),
        "baseline_like_reasons": baseline_reasons,
    }
    if checkpoint_path is None:
        return features
    if not checkpoint_path.exists():
        return features
    asset_path, asset_kind = _selected_checkpoint_asset(checkpoint_path)
    if asset_path is None or asset_kind is None:
        return features
    features["checkpoint_asset_path"] = _repo_relative_path(repo_root, asset_path)
    features["checkpoint_asset_kind"] = asset_kind
    features["checkpoint_asset_exists"] = True
    try:
        key_names = _load_checkpoint_key_names(asset_path, asset_kind)
    except Exception as exc:
        features["inspection_error"] = f"{type(exc).__name__}: {exc}"
        return features
    features["inspectable"] = True
    features["key_count"] = int(len(key_names))
    has_weight = ADVANTAGE_WEIGHT_KEY in key_names
    has_bias = ADVANTAGE_BIAS_KEY in key_names
    features["has_advantage_embedding_weight"] = has_weight
    features["has_advantage_embedding_bias"] = has_bias
    features["has_advantage_embedding_pair"] = bool(has_weight and has_bias)
    return features


def _blocked_surface(*, mode: str, selection_reason: str) -> dict[str, Any]:
    return {
        "mode": str(mode),
        "selection_reason": str(selection_reason),
        "explicit_machine_readable": True,
        "compatible_with_checkpoint": False,
        "require_advantage_embedding": False,
        "allow_baseline_default_advantage_embedding_init": False,
        "server_script": "work/recap/scripts/3D_recap_run_adv_server.py",
        "eval_wrapper_script": "work/recap/scripts/45d_vlm_critic_eval_smoke.py",
    }


def _select_prelim_eval_surface(
    *, features: Mapping[str, Any]
) -> tuple[dict[str, Any], bool, list[str]]:
    if not bool(features.get("checkpoint_bound")):
        return (
            _blocked_surface(
                mode=CHECKPOINT_BINDING_MISSING,
                selection_reason="manifest has no collect_policy_ckpt_path yet",
            ),
            False,
            ["manifest_bound_checkpoint_missing"],
        )
    if not bool(features.get("checkpoint_exists")):
        return (
            _blocked_surface(
                mode=CHECKPOINT_PATH_MISSING,
                selection_reason="manifest-bound checkpoint path does not exist locally",
            ),
            False,
            ["manifest_bound_checkpoint_missing_on_disk"],
        )
    if not bool(features.get("checkpoint_asset_exists")):
        return (
            _blocked_surface(
                mode=CHECKPOINT_ASSET_MISSING,
                selection_reason="checkpoint does not expose an inspectable weight asset",
            ),
            False,
            ["checkpoint_asset_missing"],
        )
    if not bool(features.get("inspectable")):
        return (
            _blocked_surface(
                mode=CHECKPOINT_KEYS_UNINSPECTABLE,
                selection_reason="checkpoint weight keys could not be inspected",
            ),
            False,
            ["checkpoint_weight_keys_uninspectable"],
        )
    if bool(features.get("has_advantage_embedding_pair")):
        surface = {
            "mode": ADV_SERVER_REQUIRED,
            "selection_reason": "checkpoint ships action_head.advantage_embedding.{weight,bias}",
            "explicit_machine_readable": True,
            "compatible_with_checkpoint": True,
            "require_advantage_embedding": True,
            "allow_baseline_default_advantage_embedding_init": False,
            "server_script": "work/recap/scripts/3D_recap_run_adv_server.py",
            "eval_wrapper_script": "work/recap/scripts/45d_vlm_critic_eval_smoke.py",
        }
        return surface, True, []
    if bool(features.get("baseline_like_path")):
        surface = {
            "mode": BASELINE_DEFAULT_ADV_INIT,
            "selection_reason": (
                "checkpoint path is explicitly baseline/smoke-like and does not ship "
                "action_head.advantage_embedding.{weight,bias}"
            ),
            "explicit_machine_readable": True,
            "compatible_with_checkpoint": True,
            "require_advantage_embedding": True,
            "allow_baseline_default_advantage_embedding_init": True,
            "server_script": "work/recap/scripts/3D_recap_run_adv_server.py",
            "eval_wrapper_script": "work/recap/scripts/45d_vlm_critic_eval_smoke.py",
        }
        return surface, True, []
    return (
        _blocked_surface(
            mode=INCOMPATIBLE_CHECKPOINT_SURFACE,
            selection_reason=(
                "checkpoint lacks action_head.advantage_embedding.{weight,bias} and is not an explicit baseline/smoke path"
            ),
        ),
        False,
        ["checkpoint_missing_advantage_embedding_for_selected_surface"],
    )


def _build_gate_payload(
    *,
    repo_root: Path,
    manifest_path: Path,
    gate_path: Path,
    manifest_payload: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path, checkpoint_source_field = _selected_checkpoint_path(
        repo_root, manifest_payload
    )
    checkpoint_weight_map_features = _inspect_checkpoint_weight_map_features(
        repo_root=repo_root,
        manifest_payload=manifest_payload,
        checkpoint_path=checkpoint_path,
        checkpoint_source_field=checkpoint_source_field,
    )
    prelim_eval_surface, gate_pass, failure_reason_codes = _select_prelim_eval_surface(
        features=checkpoint_weight_map_features
    )
    require_advantage_embedding = bool(
        prelim_eval_surface.get("require_advantage_embedding")
    )
    allow_baseline_default_advantage_embedding_init = bool(
        prelim_eval_surface.get("allow_baseline_default_advantage_embedding_init")
    )
    return {
        "schema_version": CONTRACT_PRECONDITION_GATE_SCHEMA_VERSION,
        "artifact_kind": CONTRACT_PRECONDITION_GATE_ARTIFACT_KIND,
        "generated_at": _now_iso(),
        "manifest_path": _repo_relative_path(repo_root, manifest_path),
        "gate_json_path": _repo_relative_path(repo_root, gate_path),
        "manifest_schema_version": str(manifest_payload.get("schema_version") or ""),
        "checkpoint_weight_map_features": checkpoint_weight_map_features,
        "prelim_eval_surface": prelim_eval_surface,
        "require_advantage_embedding": require_advantage_embedding,
        "allow_baseline_default_advantage_embedding_init": (
            allow_baseline_default_advantage_embedding_init
        ),
        "pass": bool(gate_pass),
        "status": CONTINUE_STATUS
        if gate_pass
        else FAILURE_STATUS_INCONCLUSIVE_CONTRACT_MISMATCH,
        "failure_status_if_blocked": FAILURE_STATUS_INCONCLUSIVE_CONTRACT_MISMATCH,
        "failure_reason_codes": list(failure_reason_codes),
        "exit_code": 0,
    }


def _upgrade_manifest_payload(
    *,
    repo_root: Path,
    manifest_path: Path,
    gate_path: Path,
    manifest_payload: Mapping[str, Any],
    gate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    upgraded = dict(manifest_payload)
    upgraded["schema_version"] = STAGE3_ITERATION_MANIFEST_V3
    upgraded["checkpoint_weight_map_features"] = dict(
        gate_payload.get("checkpoint_weight_map_features") or {}
    )
    upgraded["prelim_eval_surface"] = dict(
        gate_payload.get("prelim_eval_surface") or {}
    )
    upgraded["prelim_eval_require_advantage_embedding"] = bool(
        gate_payload.get("require_advantage_embedding")
    )
    upgraded["prelim_eval_allow_baseline_default_advantage_embedding_init"] = bool(
        gate_payload.get("allow_baseline_default_advantage_embedding_init")
    )
    upgraded["contract_precondition_gate"] = {
        "json_path": _repo_relative_path(repo_root, gate_path),
        "pass": bool(gate_payload.get("pass")),
        "status": str(gate_payload.get("status") or "").strip(),
        "failure_status_if_blocked": str(
            gate_payload.get("failure_status_if_blocked") or ""
        ).strip(),
        "failure_reason_codes": list(gate_payload.get("failure_reason_codes") or []),
        "selected_at": _now_iso(),
    }
    return upgraded


def run_stage3_contract_precondition_gate(
    *, repo_root: Path, manifest_path: Path | None = None
) -> dict[str, Any]:
    resolved_repo_root = Path(repo_root).resolve()
    resolved_manifest_path = _manifest_path(resolved_repo_root, manifest_path)
    manifest_payload = _read_json_object(resolved_manifest_path)
    artifact_root = _artifact_root(
        resolved_repo_root, manifest_payload, resolved_manifest_path
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    gate_path = artifact_root / CONTRACT_PRECONDITION_GATE_JSON_NAME
    gate_payload = _build_gate_payload(
        repo_root=resolved_repo_root,
        manifest_path=resolved_manifest_path,
        gate_path=gate_path,
        manifest_payload=manifest_payload,
    )
    upgraded_manifest = _upgrade_manifest_payload(
        repo_root=resolved_repo_root,
        manifest_path=resolved_manifest_path,
        gate_path=gate_path,
        manifest_payload=manifest_payload,
        gate_payload=gate_payload,
    )
    _ = write_json(gate_path, gate_payload)
    _ = write_json(resolved_manifest_path, upgraded_manifest)
    result = dict(gate_payload)
    result["artifact_root"] = _repo_relative_path(resolved_repo_root, artifact_root)
    return result


__all__ = [
    "ADV_SERVER_REQUIRED",
    "BASELINE_DEFAULT_ADV_INIT",
    "CONTRACT_PRECONDITION_GATE_JSON_NAME",
    "FAILURE_STATUS_INCONCLUSIVE_CONTRACT_MISMATCH",
    "run_stage3_contract_precondition_gate",
]
