"""Stage B P0 eval-protocol gated-ladder artifact builder.

The P0 lane is a pre-check, not a Stage B runtime probe.  This module freezes
the documented seed table, materializes the target ``n_envs`` ladder, and
records the strict gate that prevents P0 execution before P1 loader audit is
complete.  It intentionally does not import GR00T, launch policy servers, run
rollouts, train, tune checkpoints, or patch submodules.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


P0_SCHEMA_VERSION = "stage_b_p0_eval_protocol_v1"
P0_ARTIFACT_KIND = "stage_b_p0_eval_protocol_gated_ladder"
P0_GATE_ARTIFACT_KIND = "stage_b_p0_eval_protocol_gate_decision"
P0_SEED_TABLE_NAME = "p0_seed_table_30ep.csv"
P0_MATRIX_JSON_NAME = "p0_eval_matrix.json"
P0_MATRIX_MD_NAME = "p0_eval_matrix.md"
P0_GATE_JSON_NAME = "p0_gate_decision.json"
P0_GATE_MD_NAME = "p0_gate_decision.md"
PRECHECK_SHARED_SEED_TABLE_NAME = "seed_table_30ep.csv"

DEFAULT_STAGE_A_DIR = Path("agent/artifacts/stage_A_baseline_freeze_20260501T014232Z")
DEFAULT_STAGE_B_DIR = Path("agent/artifacts/stage_B_controller_seam_20260501T045404Z")
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_PROMPT_RAW = "pick up the apple, walk left and place the apple on the plate."
DEFAULT_MODE = "positive"
DEFAULT_N_ACTION_STEPS = 20
DEFAULT_MAX_EPISODE_STEPS = 1440
DEFAULT_TIMEOUT_SECONDS = 7200
DEFAULT_GPU = 1
FORMAL_30_SEED_COUNT = 30
P0_SUCCESS_THRESHOLD_COUNT = 9
P0_SUCCESS_THRESHOLD_RATE = 0.30
TARGET_N_ENVS: tuple[int, ...] = (1, 5, 30, 50)


@dataclass(frozen=True)
class SeedRow:
    """One frozen Stage A pre-registration seed row selected for P0."""

    p0_seed_index: int
    seed_value: int
    source_seed_role: str
    source_seed_index: int
    formal_lite: bool
    formal_30: bool
    high_variance_50: bool
    notes: str

    def to_csv_row(self, source_path: Path) -> dict[str, str]:
        return {
            "p0_seed_index": str(self.p0_seed_index),
            "seed_value": str(self.seed_value),
            "source_seed_role": self.source_seed_role,
            "source_seed_index": str(self.source_seed_index),
            "formal_lite": _bool_text(self.formal_lite),
            "formal_30": _bool_text(self.formal_30),
            "high_variance_50": _bool_text(self.high_variance_50),
            "source_path": str(source_path),
            "notes": self.notes,
        }

    def to_jsonable(self) -> dict[str, object]:
        return {
            "p0_seed_index": self.p0_seed_index,
            "seed_value": self.seed_value,
            "source_seed_role": self.source_seed_role,
            "source_seed_index": self.source_seed_index,
            "formal_lite": self.formal_lite,
            "formal_30": self.formal_30,
            "high_variance_50": self.high_variance_50,
            "notes": self.notes,
        }


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_seed_source(stage_a_dir: Path) -> Path:
    return stage_a_dir / "pre_registration_seed_table_v1.csv"


def read_formal_30_seed_rows(stage_a_dir: str | Path) -> tuple[SeedRow, Path]:
    """Read and validate the Stage A documented 30-episode seed set."""

    stage_a_path = Path(stage_a_dir)
    seed_source = _resolve_seed_source(stage_a_path)
    if not seed_source.is_file():
        raise FileNotFoundError(f"Stage A seed table not found: {seed_source}")

    with seed_source.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    selected: list[SeedRow] = []
    for raw in raw_rows:
        if not _truthy(raw.get("formal_30")):
            continue
        try:
            source_seed_index = int(str(raw.get("seed_index", "")).strip())
            seed_value = int(str(raw.get("seed_value", "")).strip())
        except ValueError as exc:
            raise ValueError(f"invalid seed row in {seed_source}: {raw}") from exc
        selected.append(
            SeedRow(
                p0_seed_index=len(selected),
                seed_value=seed_value,
                source_seed_role=str(raw.get("seed_role", "")).strip() or "base",
                source_seed_index=source_seed_index,
                formal_lite=_truthy(raw.get("formal_lite")),
                formal_30=True,
                high_variance_50=_truthy(raw.get("high_variance_50")),
                notes=str(raw.get("notes", "")).strip(),
            )
        )

    if len(selected) != FORMAL_30_SEED_COUNT:
        raise ValueError(
            f"expected {FORMAL_30_SEED_COUNT} formal_30 seeds in {seed_source}, "
            f"found {len(selected)}"
        )

    seen = [row.seed_value for row in selected]
    if len(set(seen)) != len(seen):
        raise ValueError(f"duplicate formal_30 seed values in {seed_source}: {seen}")

    return tuple(selected), seed_source


def _write_seed_table(path: Path, rows: Sequence[SeedRow], source_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "p0_seed_index",
        "seed_value",
        "source_seed_role",
        "source_seed_index",
        "formal_lite",
        "formal_30",
        "high_variance_50",
        "source_path",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row(source_path))
    return path


def _stage_a_manifest(stage_a_dir: Path) -> dict[str, Any]:
    return _read_json(stage_a_dir / "baseline_manifest_v1.json")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _checkpoint_descriptors(stage_a_dir: Path) -> dict[str, dict[str, object]]:
    manifest = _stage_a_manifest(stage_a_dir)
    internal = _mapping(manifest.get("internal_baseline"))
    public = _mapping(manifest.get("public_baseline"))
    public_repro = _mapping(manifest.get("public_reproduction"))
    return {
        "post_recap_g3_checkpoint_6600": {
            "checkpoint_role": "post_recap_g3_checkpoint_6600",
            "label": "post-RECAP / G3 checkpoint-6600",
            "model_path": internal.get("checkpoint_abs_path")
            or internal.get("checkpoint_path")
            or "UNRESOLVED_POST_RECAP_CHECKPOINT",
            "model_path_source": "Stage A baseline_manifest_v1.internal_baseline",
            "reference_status": "RESOLVED" if internal.get("checkpoint_exists") else "UNRESOLVED",
            "expected_loader": "local checkpoint directory via run_gr00t_server.py + --use_sim_policy_wrapper",
        },
        "base_reference": {
            "checkpoint_role": "base_reference",
            "label": "base/reference public checkpoint",
            "model_path": public.get("model_repo") or public.get("base_model") or "REFERENCE_UNRESOLVED",
            "model_path_source": "Stage A baseline_manifest_v1.public_baseline",
            "reference_status": (
                "PUBLIC_REFERENCE_LEVEL0_BLOCKED"
                if public_repro.get("status") == "failed"
                else "REFERENCE_REQUIRES_P1_AUDIT"
            ),
            "expected_loader": "public model path; must not be treated as resolved 50% reference until loader/eval evidence exists",
        },
    }


def _build_rollout_command(
    *,
    model_path: object,
    n_envs: int,
    effective_episodes: int,
    output_dir: Path,
) -> list[str]:
    return [
        "timeout",
        str(DEFAULT_TIMEOUT_SECONDS),
        "env",
        f"CUDA_VISIBLE_DEVICES={DEFAULT_GPU}",
        "MUJOCO_GL=egl",
        "python3",
        "submodules/Isaac-GR00T/gr00t/eval/rollout_policy.py",
        "--model_path",
        str(model_path),
        "--env_name",
        DEFAULT_ENV_NAME,
        "--n_episodes",
        str(effective_episodes),
        "--n_envs",
        str(n_envs),
        "--max_episode_steps",
        str(DEFAULT_MAX_EPISODE_STEPS),
        "--n_action_steps",
        str(DEFAULT_N_ACTION_STEPS),
    ]


def _cell_id(checkpoint_role: str, n_envs: int) -> str:
    prefix = "post_recap" if checkpoint_role.startswith("post_recap") else "base_reference"
    if n_envs == 1 and prefix == "post_recap":
        rung = "P0a"
    elif n_envs == 1:
        rung = "P0b"
    elif n_envs in {5, 30}:
        rung = "P0c"
    else:
        rung = "P0d"
    return f"{rung}_{prefix}_nenvs_{n_envs}"


def build_p0_eval_matrix(
    *,
    stage_a_dir: str | Path,
    stage_b_dir: str | Path,
    p1_status: str,
) -> dict[str, Any]:
    """Build the P0 target matrix without running any eval cell."""

    stage_a_path = Path(stage_a_dir)
    stage_b_path = Path(stage_b_dir)
    seed_rows, seed_source = read_formal_30_seed_rows(stage_a_path)
    descriptors = _checkpoint_descriptors(stage_a_path)
    p0_dir = stage_b_path / "prechecks" / "P0_eval_protocol_determinism"
    cells: list[dict[str, object]] = []
    strict_order_blocks = str(p1_status) != "P1_PASS"
    for checkpoint_role in (
        "post_recap_g3_checkpoint_6600",
        "base_reference",
    ):
        descriptor = descriptors[checkpoint_role]
        for n_envs in TARGET_N_ENVS:
            effective_episodes = max(FORMAL_30_SEED_COUNT, int(n_envs))
            cell_output_dir = p0_dir / "cells" / _cell_id(checkpoint_role, int(n_envs))
            runner_notes = [
                "P0 must bind every executed cell to the same p0_seed_table_30ep.csv.",
                "upstream rollout_policy.py supports --n_envs but internally sets n_episodes=max(n_episodes,n_envs).",
                "upstream rollout_policy.py does not expose an explicit seed-table argument; execution must record that runner seed semantics are unresolved or use a reviewed seed wrapper.",
            ]
            if n_envs == 1:
                runner_notes.append(
                    "Existing gr00t_g3_formal_eval.py can replay seed_base/episode_count with total_n_envs=1, but it must not be misreported as an n_envs ladder runner."
                )
            status = "PENDING_P1_GATE" if strict_order_blocks else "READY_FOR_GATED_EXECUTION"
            if descriptor.get("reference_status") == "PUBLIC_REFERENCE_LEVEL0_BLOCKED":
                status = "REFERENCE_PROVENANCE_BLOCKED" if not strict_order_blocks else status
            cells.append(
                {
                    "cell_id": _cell_id(checkpoint_role, int(n_envs)),
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_label": descriptor["label"],
                    "model_path": descriptor["model_path"],
                    "model_path_source": descriptor["model_path_source"],
                    "reference_status": descriptor["reference_status"],
                    "expected_loader": descriptor["expected_loader"],
                    "n_envs": int(n_envs),
                    "requested_seed_count": FORMAL_30_SEED_COUNT,
                    "requested_episode_count": FORMAL_30_SEED_COUNT,
                    "effective_min_episode_count": effective_episodes,
                    "seed_values": [row.seed_value for row in seed_rows],
                    "seed_table_path": str(p0_dir / P0_SEED_TABLE_NAME),
                    "primary_metric": "official_task_success",
                    "success_threshold_count": P0_SUCCESS_THRESHOLD_COUNT,
                    "success_threshold_rate": P0_SUCCESS_THRESHOLD_RATE,
                    "status": status,
                    "execution_blocker": (
                        "P1 loader audit is not P1_PASS; strict P1→P0 order blocks this cell."
                        if strict_order_blocks
                        else None
                    ),
                    "runner_semantics": {
                        "wraps_upstream_rollout_policy_n_envs": True,
                        "does_not_assume_gr00t_g3_formal_eval_supports_n_envs": True,
                        "rollout_policy_effective_episode_rule": "n_episodes=max(requested_n_episodes,n_envs)",
                        "explicit_seed_table_replay_status": "UNRESOLVED_FOR_UPSTREAM_ROLLOUT_POLICY",
                        "notes": runner_notes,
                    },
                    "representative_command": _build_rollout_command(
                        model_path=descriptor["model_path"],
                        n_envs=int(n_envs),
                        effective_episodes=effective_episodes,
                        output_dir=cell_output_dir,
                    ),
                    "output_dir": str(cell_output_dir),
                }
            )

    return {
        "schema_version": P0_SCHEMA_VERSION,
        "artifact_kind": P0_ARTIFACT_KIND,
        "created_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "stage": "Stage B",
        "precheck": "P0_eval_protocol_determinism",
        "diagnostic_only": True,
        "formal_benchmark": False,
        "training_allowed": False,
        "checkpoint_update_allowed": False,
        "method_claim_allowed": False,
        "p1_status": str(p1_status),
        "strict_order": "P1_loader_audit -> P0_eval_protocol_determinism -> P2_inference_unconditional_swap",
        "stage_a_dir": str(stage_a_path),
        "stage_b_dir": str(stage_b_path),
        "seed_source_path": str(seed_source),
        "seed_source_sha256": _sha256_file(seed_source),
        "seed_count": len(seed_rows),
        "seed_values": [row.seed_value for row in seed_rows],
        "target_n_envs": list(TARGET_N_ENVS),
        "target_matrix_notes": [
            "post-RECAP n_envs=1 is P0a and stops runtime probes if success >=9/30.",
            "base/reference n_envs=1 is P0b and must resolve reference provenance before any 50% claim.",
            "n_envs={5,30} are P0c and run only after n_envs=1 cannot explain collapse.",
            "n_envs=50 is P0d and must record rollout_policy n_episodes=max(n_episodes,n_envs).",
        ],
        "cells": cells,
    }


def classify_p0_gate(
    *,
    p1_status: str,
    cell_results: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Classify P0 gate status from completed cell results."""

    if str(p1_status) != "P1_PASS":
        return {
            "decision": "P0_BLOCKED",
            "reason": "P1 loader audit is not P1_PASS; strict Stage B order blocks P0 execution.",
            "blocked_by": "P1_loader_audit",
            "continue_to_p2": False,
            "continue_to_runtime_probes": False,
            "fix_route": "finish_or_fix_P1_loader_audit_before_eval_protocol_ladder",
        }

    results = list(cell_results or [])
    if not results:
        return {
            "decision": "P0_BLOCKED",
            "reason": "No executed P0 cells have been recorded yet.",
            "blocked_by": "P0_eval_cells_not_executed",
            "continue_to_p2": False,
            "continue_to_runtime_probes": False,
            "fix_route": "execute_P0a_then_follow_gated_ladder",
        }

    by_cell = {str(result.get("cell_id")): result for result in results}

    def _success_count(result: Mapping[str, object] | None) -> int | None:
        if result is None:
            return None
        raw = result.get("success_count")
        try:
            return int(str(raw))
        except (TypeError, ValueError):
            return None

    post_n1 = _success_count(by_cell.get("P0a_post_recap_nenvs_1"))
    if post_n1 is not None and post_n1 >= P0_SUCCESS_THRESHOLD_COUNT:
        return {
            "decision": "STOP_EVAL_PROTOCOL",
            "reason": "post-RECAP recovered to >=9/30 under exact n_envs=1 protocol.",
            "continue_to_p2": False,
            "continue_to_runtime_probes": False,
            "fix_route": "eval_harness_determinism_or_env_drift_fix",
        }

    higher_post = [
        (cell_id, _success_count(result))
        for cell_id, result in by_cell.items()
        if cell_id.startswith("P0c_post_recap") or cell_id.startswith("P0d_post_recap")
    ]
    recovered_higher = [
        (cell_id, count)
        for cell_id, count in higher_post
        if count is not None and count >= P0_SUCCESS_THRESHOLD_COUNT
    ]
    if (post_n1 is not None and post_n1 < P0_SUCCESS_THRESHOLD_COUNT) and recovered_higher:
        return {
            "decision": "STOP_N_ENVS_VECTOR_BUG",
            "reason": "post-RECAP stayed low at n_envs=1 but recovered at a higher n_envs cell.",
            "recovered_cells": [
                {"cell_id": cell_id, "success_count": count}
                for cell_id, count in recovered_higher
            ],
            "continue_to_p2": False,
            "continue_to_runtime_probes": False,
            "fix_route": "vector_env_seed_semantics_or_async_scheduling_fix",
        }

    base_n1 = _success_count(by_cell.get("P0b_base_reference_nenvs_1"))
    if base_n1 is not None and base_n1 < P0_SUCCESS_THRESHOLD_COUNT:
        return {
            "decision": "P0_BASE_UNSTABLE",
            "reason": "base/reference did not reach >=9/30 at n_envs=1; current eval harness or reference provenance is not trustworthy.",
            "continue_to_p2": False,
            "continue_to_runtime_probes": False,
            "fix_route": "resolve_reference_loader_eval_protocol_before_method_diagnosis",
        }

    if post_n1 is not None and base_n1 is not None:
        return {
            "decision": "P0_NEGATIVE",
            "reason": "Executed n_envs=1 ladder did not recover post-RECAP and base/reference was not unstable.",
            "continue_to_p2": True,
            "continue_to_runtime_probes": False,
            "fix_route": "continue_to_P2_before_any_runtime_probe",
        }

    return {
        "decision": "P0_BLOCKED",
        "reason": "P0 cell results are incomplete for a gate decision.",
        "blocked_by": "incomplete_p0_cell_results",
        "continue_to_p2": False,
        "continue_to_runtime_probes": False,
        "fix_route": "complete_the_next_gated_P0_cell_or_record_skip_reason",
    }


def build_p0_gate_decision(
    *,
    p1_status: str,
    matrix: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    classification = classify_p0_gate(p1_status=p1_status, cell_results=cell_results)
    return {
        "schema_version": P0_SCHEMA_VERSION,
        "artifact_kind": P0_GATE_ARTIFACT_KIND,
        "created_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "stage": "Stage B",
        "precheck": "P0_eval_protocol_determinism",
        "diagnostic_only": True,
        "training_allowed": False,
        "method_claim_allowed": False,
        "success_threshold_count": P0_SUCCESS_THRESHOLD_COUNT,
        "success_threshold_rate": P0_SUCCESS_THRESHOLD_RATE,
        "seed_table_path": str(
            Path(str(matrix.get("stage_b_dir", "")))
            / "prechecks"
            / "P0_eval_protocol_determinism"
            / P0_SEED_TABLE_NAME
        ),
        "matrix_path": str(
            Path(str(matrix.get("stage_b_dir", "")))
            / "prechecks"
            / "P0_eval_protocol_determinism"
            / P0_MATRIX_JSON_NAME
        ),
        **classification,
        "claim_boundary": [
            "P0 recovery is an eval-protocol sanity signal, not Stage B PASS.",
            "P0 negative/blocked is not a RECAP method success/failure claim.",
            "Do not run P2 or runtime probes when this decision is STOP or BLOCKED.",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def render_matrix_markdown(matrix: Mapping[str, Any]) -> str:
    rows = []
    for cell in matrix.get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        rows.append(
            "| {cell_id} | {checkpoint_label} | {n_envs} | {effective_min_episode_count} | {status} | {reference_status} |".format(
                **cell
            )
        )
    table = "\n".join(rows)
    return f"""# Stage B P0 eval-protocol determinism ladder

本文件冻结 P0 的目标矩阵与 gated ladder。P0 是 pre-check，不是 benchmark，不训练、不调 checkpoint，也不允许据此声明 RECAP 成败。

- schema: `{matrix.get("schema_version")}`
- P1 状态: `{matrix.get("p1_status")}`
- seed source: `{matrix.get("seed_source_path")}`
- seed sha256: `{matrix.get("seed_source_sha256")}`
- seed count: `{matrix.get("seed_count")}`
- success threshold: `{P0_SUCCESS_THRESHOLD_COUNT}/30`

## Runner 语义

- `rollout_policy.py` 支持 `--n_envs`，但内部会执行 `n_episodes=max(n_episodes,n_envs)`。
- `rollout_policy.py` 当前没有显式 seed-table 参数；执行任何 cell 时必须记录该 runner 语义，或先使用经过审阅的 seed wrapper。
- 现有 `gr00t_g3_formal_eval.py` 可用于 `n_envs=1` 的 seed_base/episode_count 复核，但不能被伪称为 `n_envs` ladder runner。

## Target matrix

| cell | checkpoint | n_envs | effective min episodes | status | reference status |
|---|---|---:|---:|---|---|
{table}
"""


def render_gate_markdown(gate: Mapping[str, Any]) -> str:
    return f"""# Stage B P0 gate decision

- decision: `{gate.get("decision")}`
- reason: {gate.get("reason")}
- continue_to_p2: `{gate.get("continue_to_p2")}`
- continue_to_runtime_probes: `{gate.get("continue_to_runtime_probes")}`
- fix_route: `{gate.get("fix_route")}`

## Claim boundary

- P0 只是 eval-protocol / vector-env / seed 语义 sanity pre-check。
- STOP/BLOCKED 时不得继续 P2 或 runtime probes，除非计划被显式修订。
- 任何恢复 ≥9/30 只说明需要先排查 eval harness / vector-env，不是 Stage B PASS 或方法成功。
"""


def infer_p1_status(stage_b_dir: str | Path) -> str:
    """Infer current P1 status from the Stage B artifact root, if present."""

    gate_path = (
        Path(stage_b_dir)
        / "prechecks"
        / "P1_loader_audit"
        / "p1_gate_decision.json"
    )
    gate = _read_json(gate_path)
    for key in ("decision", "status", "gate_status"):
        value = gate.get(key)
        if value:
            return str(value)
    return "P1_PENDING"


def write_p0_artifacts(
    *,
    stage_a_dir: str | Path,
    stage_b_dir: str | Path,
    p1_status: str | None = None,
) -> dict[str, Path]:
    """Write seed table, matrix, and gate artifacts for P0."""

    stage_a_path = Path(stage_a_dir)
    stage_b_path = Path(stage_b_dir)
    resolved_p1_status = p1_status or infer_p1_status(stage_b_path)
    seed_rows, seed_source = read_formal_30_seed_rows(stage_a_path)
    p0_dir = stage_b_path / "prechecks" / "P0_eval_protocol_determinism"
    log_dir = p0_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    seed_path = _write_seed_table(p0_dir / P0_SEED_TABLE_NAME, seed_rows, seed_source)
    shared_seed_path = _write_seed_table(
        stage_b_path / "prechecks" / PRECHECK_SHARED_SEED_TABLE_NAME,
        seed_rows,
        seed_source,
    )
    matrix = build_p0_eval_matrix(
        stage_a_dir=stage_a_path,
        stage_b_dir=stage_b_path,
        p1_status=resolved_p1_status,
    )
    matrix_json_path = _write_json(p0_dir / P0_MATRIX_JSON_NAME, matrix)
    matrix_md_path = p0_dir / P0_MATRIX_MD_NAME
    matrix_md_path.write_text(render_matrix_markdown(matrix), encoding="utf-8")
    gate = build_p0_gate_decision(p1_status=resolved_p1_status, matrix=matrix)
    gate_json_path = _write_json(p0_dir / P0_GATE_JSON_NAME, gate)
    gate_md_path = p0_dir / P0_GATE_MD_NAME
    gate_md_path.write_text(render_gate_markdown(gate), encoding="utf-8")
    static_log_path = log_dir / "p0_static_artifact_generation.log"
    static_log_path.write_text(
        "\n".join(
            [
                f"created_at_utc={matrix['created_at_utc']}",
                f"stage_a_dir={stage_a_path}",
                f"stage_b_dir={stage_b_path}",
                f"p1_status={resolved_p1_status}",
                f"seed_source={seed_source}",
                f"seed_table={seed_path}",
                f"matrix={matrix_json_path}",
                f"gate={gate_json_path}",
                "execution_started=false",
                "reason=static P0 gated-ladder artifacts only; no rollout/training/full-long-run launched",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "seed_table": seed_path,
        "shared_seed_table": shared_seed_path,
        "matrix_json": matrix_json_path,
        "matrix_md": matrix_md_path,
        "gate_json": gate_json_path,
        "gate_md": gate_md_path,
        "static_log": static_log_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-dir", type=Path, default=DEFAULT_STAGE_A_DIR)
    parser.add_argument("--stage-b-dir", type=Path, default=DEFAULT_STAGE_B_DIR)
    parser.add_argument(
        "--p1-status",
        default="",
        help="Override inferred P1 status; defaults to reading p1_gate_decision.json.",
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Write P0 seed table, matrix, and gate decision artifacts.",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print the matrix JSON to stdout instead of writing artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    p1_status = str(args.p1_status).strip() or infer_p1_status(args.stage_b_dir)
    if args.print_matrix:
        matrix = build_p0_eval_matrix(
            stage_a_dir=args.stage_a_dir,
            stage_b_dir=args.stage_b_dir,
            p1_status=p1_status,
        )
        print(json.dumps(matrix, ensure_ascii=False, indent=2))
        return 0
    if args.write_artifacts:
        paths = write_p0_artifacts(
            stage_a_dir=args.stage_a_dir,
            stage_b_dir=args.stage_b_dir,
            p1_status=p1_status,
        )
        for name, path in paths.items():
            print(f"{name}={path}")
        return 0
    _build_parser().print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
