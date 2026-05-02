from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from work.openpi.pipelines.recap.blind_calibration_runtime import (
    read_json_object,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "stage1_v22_blind_calibration_iter8_20260426T_nextZ"
ITER8_INPUT_CONTRACT_SCHEMA = "w6_blind_calibration_input_contract_v3"
A_STOCK_AUTHORITY_SCHEMA = "a_stock_authority_manifest_iter8_v1"
R2_R4_PIN_SCHEMA = "r2_r4_closure_pin_v1"
ITER6_MATRIX_SHA256 = "533042bfc05c9178fc2538331ae45448303b062b6e05c404cee83767b4af6407"


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value))


def _sidecar_hex(path: Path) -> str | None:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        return None
    return sidecar.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class Iter8InputContract:
    path: Path
    sha256: str
    schema_version: str
    run_id: str
    canonical_blind_selection_rule_path: Path
    canonical_blind_selection_rule_sha256: str
    candidate_space_matrix_path: Path
    candidate_space_matrix_sha256: str
    candidate_id_format: str
    early_stop_policy_path: Path
    early_stop_policy_sha256: str
    a_stock_authority_manifest_path: Path
    a_stock_authority_manifest_sha256: str | None
    r2_r4_closure_pin_path: Path
    r2_r4_closure_pin_sha256: str
    episode_policy: Mapping[str, object]
    b_scan_policy: str
    gpu2_memory_threshold_mib: int
    forbidden_selection_variants: tuple[str, ...]
    calibration_variants: tuple[str, ...]
    optional_control_variants: tuple[str, ...]
    formal_v22_execution_allowed: bool
    raw: Mapping[str, object]


@dataclass(frozen=True)
class AStockAuthorityManifest:
    path: Path
    sha256: str
    schema_version: str
    run_id: str
    semantic_role: str
    checkpoint_id: str
    gs_path: str
    loader_entrypoint: str
    openpi_commit: str
    openpi_install_mechanism: str | None
    vendored_libero_install_mechanism: str | None
    local_resolved_path: Path | None
    local_checkpoint_sha256: str | None
    fallback_inspection_log: tuple[object, ...]
    blocking_reasons: tuple[str, ...]
    raw: Mapping[str, object]


@dataclass(frozen=True)
class EpisodePolicy:
    episodes_per_cell_A: int
    episodes_per_cell_B: int
    episodes_per_cell_smoke: int
    b_scan_policy: str


@dataclass(frozen=True)
class R2R4Pin:
    path: Path
    sha256: str
    schema_version: str
    r2_status_path: Path
    r2_status: str
    r4_status_path: Path
    r4_status: str
    raw: Mapping[str, object]


def load_input_contract(path: Path, expected_sha256: str) -> Iter8InputContract:
    if not path.is_file():
        raise FileNotFoundError(f"BLOCK_INPUT_CONTRACT_PATH_MISSING:{path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"BLOCK_INPUT_CONTRACT_SHA_MISMATCH expected={expected_sha256} actual={actual_sha}"
        )
    payload = read_json_object(path)
    episode_policy = payload.get("episode_policy")
    if not isinstance(episode_policy, Mapping):
        episode_policy = {}
    return Iter8InputContract(
        path=path,
        sha256=actual_sha,
        schema_version=str(payload.get("schema_version") or ""),
        run_id=str(payload.get("run_id") or ""),
        canonical_blind_selection_rule_path=_repo_path(
            str(payload.get("canonical_blind_selection_rule_path") or "")
        ),
        canonical_blind_selection_rule_sha256=str(
            payload.get("canonical_blind_selection_rule_sha256") or ""
        ),
        candidate_space_matrix_path=_repo_path(
            str(payload.get("candidate_space_matrix_path") or "")
        ),
        candidate_space_matrix_sha256=str(payload.get("candidate_space_matrix_sha256") or ""),
        candidate_id_format=str(payload.get("candidate_id_format") or ""),
        early_stop_policy_path=_repo_path(str(payload.get("early_stop_policy_path") or "")),
        early_stop_policy_sha256=str(payload.get("early_stop_policy_sha256") or ""),
        a_stock_authority_manifest_path=_repo_path(
            str(payload.get("a_stock_authority_manifest_path") or "")
        ),
        a_stock_authority_manifest_sha256=(
            str(payload.get("a_stock_authority_manifest_sha256"))
            if payload.get("a_stock_authority_manifest_sha256") is not None
            else None
        ),
        r2_r4_closure_pin_path=_repo_path(str(payload.get("r2_r4_closure_pin_path") or "")),
        r2_r4_closure_pin_sha256=str(payload.get("r2_r4_closure_pin_sha256") or ""),
        episode_policy=episode_policy,
        b_scan_policy=str(payload.get("b_scan_policy") or ""),
        gpu2_memory_threshold_mib=int(payload.get("gpu2_memory_threshold_mib") or 0),
        forbidden_selection_variants=_string_tuple(payload.get("forbidden_selection_variants")),
        calibration_variants=_string_tuple(payload.get("calibration_variants")),
        optional_control_variants=_string_tuple(payload.get("optional_control_variants")),
        formal_v22_execution_allowed=bool(payload.get("formal_v22_execution_allowed")),
        raw=payload,
    )


def load_a_stock_authority_manifest(coordinator_dir: Path) -> AStockAuthorityManifest:
    path = coordinator_dir / "a_stock_authority_manifest_iter8.json"
    if not path.is_file():
        raise FileNotFoundError(f"BLOCK_A_STOCK_AUTHORITY_MISSING:{path}")
    actual_sha = sha256_file(path)
    payload = read_json_object(path)
    resolved_raw = payload.get("local_resolved_path")
    resolved_path = _repo_path(str(resolved_raw)) if resolved_raw else None
    return AStockAuthorityManifest(
        path=path,
        sha256=actual_sha,
        schema_version=str(payload.get("schema_version") or ""),
        run_id=str(payload.get("run_id") or ""),
        semantic_role=str(payload.get("semantic_role") or ""),
        checkpoint_id=str(payload.get("checkpoint_id") or ""),
        gs_path=str(payload.get("gs_path") or ""),
        loader_entrypoint=str(payload.get("loader_entrypoint") or ""),
        openpi_commit=str(payload.get("openpi_commit") or ""),
        openpi_install_mechanism=(
            str(payload.get("openpi_install_mechanism"))
            if payload.get("openpi_install_mechanism") is not None
            else None
        ),
        vendored_libero_install_mechanism=(
            str(payload.get("vendored_libero_install_mechanism"))
            if payload.get("vendored_libero_install_mechanism") is not None
            else None
        ),
        local_resolved_path=resolved_path,
        local_checkpoint_sha256=(
            str(payload.get("local_checkpoint_sha256"))
            if payload.get("local_checkpoint_sha256") is not None
            else None
        ),
        fallback_inspection_log=_sequence(payload.get("fallback_inspection_log")),
        blocking_reasons=tuple(str(item) for item in _sequence(payload.get("blocking_reasons"))),
        raw=payload,
    )


def validate_iter8_input_contract(contract: Iter8InputContract) -> list[str]:
    reasons: list[str] = []
    if contract.schema_version != ITER8_INPUT_CONTRACT_SCHEMA:
        reasons.append("BLOCK_INPUT_CONTRACT_SCHEMA_MISMATCH")
    if contract.run_id != RUN_ID:
        reasons.append("BLOCK_INPUT_CONTRACT_RUN_ID_MISMATCH")
    if contract.formal_v22_execution_allowed:
        reasons.append("BLOCK_FORMAL_V22_EXECUTION_NOT_ALLOWED_VIOLATED")
    if contract.candidate_space_matrix_sha256 != ITER6_MATRIX_SHA256:
        reasons.append("BLOCK_MATRIX_SHA_MISMATCH")
    if contract.candidate_id_format != "matrix_verbatim":
        reasons.append("BLOCK_CANDIDATE_ID_FORMAT")
    if contract.b_scan_policy not in {"all_cells", "headroom_eligible_only", "none"}:
        reasons.append("BLOCK_B_SCAN_POLICY_INVALID")
    if contract.gpu2_memory_threshold_mib <= 0:
        reasons.append("BLOCK_GPU2_MEMORY_THRESHOLD_INVALID")
    if {"C", "X"} - set(contract.forbidden_selection_variants):
        reasons.append("BLOCK_C_X_LEAKAGE")
    if any(variant in {"C", "X"} for variant in contract.calibration_variants):
        reasons.append("BLOCK_C_X_LEAKAGE")
    if any(variant in {"C", "X"} for variant in contract.optional_control_variants):
        reasons.append("BLOCK_C_X_LEAKAGE")
    episode_policy = contract.episode_policy
    for key in ("episodes_per_cell_A", "episodes_per_cell_B", "episodes_per_cell_smoke"):
        if int(episode_policy.get(key) or 0) <= 0:
            reasons.append(f"BLOCK_EPISODE_POLICY_INVALID:{key}")
    return reasons


def assert_candidate_id_format(matrix_path: Path, expected: str = "matrix_verbatim") -> None:
    matrix = read_json_object(matrix_path)
    explicit = matrix.get("candidate_id_format")
    if explicit is not None and explicit != expected:
        raise ValueError("BLOCK_CANDIDATE_ID_FORMAT")
    rows = matrix.get("candidate_cells")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("BLOCK_CANDIDATE_ID_FORMAT")
    candidate_ids = [
        str(row.get("candidate_id"))
        for row in rows
        if isinstance(row, Mapping) and row.get("candidate_id")
    ]
    if len(candidate_ids) != len(rows):
        raise ValueError("BLOCK_CANDIDATE_ID_FORMAT")
    if expected == "matrix_verbatim":
        bad = [
            candidate_id
            for candidate_id in candidate_ids
            if "__budget_0_" not in candidate_id or "__taskset_" in candidate_id or "0p" in candidate_id
        ]
        if bad:
            raise ValueError("BLOCK_CANDIDATE_ID_FORMAT:" + ",".join(bad[:3]))


def assert_no_c_x_leakage(
    *,
    calibration_variants: Sequence[str],
    optional_control_variants: Sequence[str],
    forbidden_selection_variants: Sequence[str],
) -> None:
    forbidden = {str(item) for item in forbidden_selection_variants}
    if {"C", "X"} - forbidden:
        raise ValueError("BLOCK_C_X_LEAKAGE")
    selected = {str(item) for item in calibration_variants} | {
        str(item) for item in optional_control_variants
    }
    leaked = sorted(selected & forbidden)
    if leaked:
        raise ValueError("BLOCK_C_X_LEAKAGE:" + ",".join(leaked))


def coerce_episode_policy(args: Any) -> EpisodePolicy:
    episodes_a = getattr(args, "episodes_per_cell_A", None)
    episodes_b = getattr(args, "episodes_per_cell_B", None)
    episodes_smoke = getattr(args, "episodes_per_cell_smoke", None)
    return EpisodePolicy(
        episodes_per_cell_A=max(1, int(episodes_a or 12)),
        episodes_per_cell_B=max(1, int(episodes_b or 12)),
        episodes_per_cell_smoke=max(1, int(episodes_smoke or 2)),
        b_scan_policy=str(getattr(args, "b_scan_policy", "headroom_eligible_only")),
    )


def pin_iter5_r2_r4_closure(coordinator_dir: Path) -> R2R4Pin:
    path = coordinator_dir / "r2_r4_closure_pin.json"
    if not path.is_file():
        path = (
            REPO_ROOT
            / "agent/artifacts/stage1_v22_blind_calibration_iter8_20260426T_nextZ/coordinator/r2_r4_closure_pin.json"
        )
    if not path.is_file():
        raise FileNotFoundError(f"BLOCK_R2_R4_CLOSURE_PIN_MISSING:{path}")
    actual_sha = sha256_file(path)
    payload = read_json_object(path)
    return R2R4Pin(
        path=path,
        sha256=actual_sha,
        schema_version=str(payload.get("schema_version") or ""),
        r2_status_path=_repo_path(str(payload.get("r2_status_path") or "")),
        r2_status=str(payload.get("r2_status") or ""),
        r4_status_path=_repo_path(str(payload.get("r4_status_path") or "")),
        r4_status=str(payload.get("r4_status") or ""),
        raw=payload,
    )


def sidecar_matches(path: Path) -> bool:
    expected = _sidecar_hex(path)
    return expected is not None and expected == sha256_file(path)
