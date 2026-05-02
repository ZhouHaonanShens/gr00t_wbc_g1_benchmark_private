"""P2 temporary inference adapter contract for Stage B pre-checks.

This module is intentionally a lightweight, opt-in diagnostic kernel.  It does
not import GR00T runtime code, does not mutate checkpoints, and does not launch
rollouts.  Runtime integration code can call the same contract checks before it
attempts a dual-model unconditional swap.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from .array_summary import summarize_array, to_numpy_array
from .execution_boundaries import require_stage_b_safe_command


P2_SCHEMA_VERSION = "stage_b_p2_inference_adapter_v1"
P2_DEFAULT_WEIGHTS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)

P2_READY = "P2_READY"
P2_SKIPPED_P1_NOT_PASS = "P2_SKIPPED_P1_NOT_PASS"
P2_SKIPPED_P0_EXPLAINS_OR_BLOCKS = "P2_SKIPPED_P0_EXPLAINS_OR_BLOCKS"
P2_BLOCKED_UNSAFE = "P2_BLOCKED_UNSAFE"
P2_NO_ENV_SANITY_PASS = "P2_NO_ENV_ACTION_SANITY_PASS"

P1_PASS_STATUSES = frozenset({"PASS", "P1_PASS", "LOADER_AUDIT_PASS"})
P0_ALLOW_P2_STATUSES = frozenset({"P0_NEGATIVE", "NEGATIVE", "PASS_NO_RECOVERY"})
P0_STOP_OR_BLOCK_STATUSES = frozenset(
    {
        "STOP_EVAL_PROTOCOL",
        "STOP_N_ENVS_VECTOR_BUG",
        "P0_BASE_UNSTABLE",
        "P0_BLOCKED",
        "BLOCKED",
        "PENDING",
        "UNKNOWN",
    }
)


def _normalize_status(value: object) -> str:
    return str(value or "UNKNOWN").strip().upper()


def _finite_non_negative_weight(value: float) -> float:
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"guidance weight must be finite and non-negative: {value!r}")
    return weight


def _has_grad_attached(value: Any) -> bool:
    return bool(getattr(value, "requires_grad", False))


@dataclass(frozen=True)
class P2ReadinessDecision:
    """Whether the P2 adapter is allowed by the P1/P0 pre-check gate."""

    status: str
    allowed: bool
    p1_status: str
    p0_status: str
    reason: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionContract:
    """Shape/order/noise/timestep surface for one no-env model prediction."""

    label: str
    shape: tuple[int, ...]
    dtype: str
    action_order: tuple[str, ...]
    normalization_id: str
    timestep_schedule_hash: str
    initial_noise_hash: str
    value_hash: str
    finite: bool

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["action_order"] = list(self.action_order)
        return payload


@dataclass(frozen=True)
class NoEnvActionSanityResult:
    """Result of the required same-noise/same-timestep P2 safety check."""

    status: str
    safe_to_eval: bool
    reasons: tuple[str, ...]
    fine_tuned_contract: PredictionContract
    frozen_unconditional_contract: PredictionContract

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema_version": P2_SCHEMA_VERSION,
            "status": self.status,
            "safe_to_eval": self.safe_to_eval,
            "reasons": list(self.reasons),
            "fine_tuned_contract": self.fine_tuned_contract.to_jsonable(),
            "frozen_unconditional_contract": (
                self.frozen_unconditional_contract.to_jsonable()
            ),
        }


@dataclass(frozen=True)
class WeightSweepPlan:
    """Static P2 sweep contract; it does not report benchmark outcomes."""

    weights: tuple[float, ...] = P2_DEFAULT_WEIGHTS
    n_envs: int = 1
    documented_seed_count: int = 30
    success_threshold_count: int = 9

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema_version": P2_SCHEMA_VERSION,
            "weights": [float(weight) for weight in self.weights],
            "n_envs": self.n_envs,
            "documented_seed_count": self.documented_seed_count,
            "success_threshold_count": self.success_threshold_count,
            "success_threshold": ">=9/30 (>=30%)",
            "same_seed_table_required": True,
            "w0_recovery_rule": (
                "If w=0 reaches >=30%, treat as positive conditional / "
                "loader / eval-path recovery, not unconditional replacement."
            ),
            "positive_unconditional_rule": (
                "Only w>0 recovery with w=0 still below threshold supports "
                "the unconditional-pathway-collapse hypothesis."
            ),
        }


def evaluate_p2_readiness(
    *,
    p1_status: object,
    p0_status: object,
) -> P2ReadinessDecision:
    """Return the P1/P0 gate decision for attempting P2.

    P2 is allowed only after the loader audit passes and P0 has failed to
    explain/recover the collapse.  Pending or blocked inputs deliberately block
    P2 rather than being treated as a negative finding.
    """

    normalized_p1 = _normalize_status(p1_status)
    normalized_p0 = _normalize_status(p0_status)
    if normalized_p1 not in P1_PASS_STATUSES:
        return P2ReadinessDecision(
            status=P2_SKIPPED_P1_NOT_PASS,
            allowed=False,
            p1_status=normalized_p1,
            p0_status=normalized_p0,
            reason="P2 requires P1 loader audit PASS before any adapter work.",
        )
    if normalized_p0 not in P0_ALLOW_P2_STATUSES:
        status = P2_SKIPPED_P0_EXPLAINS_OR_BLOCKS
        if normalized_p0 not in P0_STOP_OR_BLOCK_STATUSES:
            status = P2_SKIPPED_P0_EXPLAINS_OR_BLOCKS
        return P2ReadinessDecision(
            status=status,
            allowed=False,
            p1_status=normalized_p1,
            p0_status=normalized_p0,
            reason=(
                "P2 is gated until P0 is NEGATIVE/no-recovery; STOP, BLOCKED, "
                "pending, or unknown P0 states must not be treated as negative."
            ),
        )
    return P2ReadinessDecision(
        status=P2_READY,
        allowed=True,
        p1_status=normalized_p1,
        p0_status=normalized_p0,
        reason="P1 PASS and P0 negative/no-recovery; P2 no-env sanity may run.",
    )


def build_prediction_contract(
    *,
    label: str,
    prediction: Any,
    action_order: Sequence[str],
    normalization_id: str,
    timestep_schedule_hash: str,
    initial_noise_hash: str,
) -> PredictionContract:
    """Summarize one model prediction without retaining the raw action."""

    summary = summarize_array(prediction)
    return PredictionContract(
        label=str(label),
        shape=tuple(int(item) for item in summary["shape"]),
        dtype=str(summary["dtype"]),
        action_order=tuple(str(item) for item in action_order),
        normalization_id=str(normalization_id),
        timestep_schedule_hash=str(timestep_schedule_hash),
        initial_noise_hash=str(initial_noise_hash),
        value_hash=str(summary["sha256"]),
        finite=int(summary.get("nan_count", 0)) == 0
        and int(summary.get("inf_count", 0)) == 0,
    )


def compare_no_env_action_sanity(
    *,
    fine_tuned_contract: PredictionContract,
    frozen_unconditional_contract: PredictionContract,
) -> NoEnvActionSanityResult:
    """Check shape/order/dtype/noise/timestep compatibility before P2 eval."""

    reasons: list[str] = []
    if fine_tuned_contract.shape != frozen_unconditional_contract.shape:
        reasons.append(
            "shape_mismatch:"
            f"{fine_tuned_contract.shape}!={frozen_unconditional_contract.shape}"
        )
    if fine_tuned_contract.dtype != frozen_unconditional_contract.dtype:
        reasons.append(
            "dtype_mismatch:"
            f"{fine_tuned_contract.dtype}!={frozen_unconditional_contract.dtype}"
        )
    if fine_tuned_contract.action_order != frozen_unconditional_contract.action_order:
        reasons.append("action_order_mismatch")
    if (
        fine_tuned_contract.normalization_id
        != frozen_unconditional_contract.normalization_id
    ):
        reasons.append("normalization_mismatch")
    if (
        fine_tuned_contract.timestep_schedule_hash
        != frozen_unconditional_contract.timestep_schedule_hash
    ):
        reasons.append("timestep_schedule_mismatch")
    if fine_tuned_contract.initial_noise_hash != frozen_unconditional_contract.initial_noise_hash:
        reasons.append("initial_noise_mismatch")
    if not fine_tuned_contract.finite:
        reasons.append("fine_tuned_prediction_non_finite")
    if not frozen_unconditional_contract.finite:
        reasons.append("frozen_unconditional_prediction_non_finite")

    safe = not reasons
    return NoEnvActionSanityResult(
        status=P2_NO_ENV_SANITY_PASS if safe else P2_BLOCKED_UNSAFE,
        safe_to_eval=safe,
        reasons=tuple(reasons or ["contracts_match"]),
        fine_tuned_contract=fine_tuned_contract,
        frozen_unconditional_contract=frozen_unconditional_contract,
    )


def run_no_env_action_sanity(
    *,
    fine_tuned_prediction: Any,
    frozen_unconditional_prediction: Any,
    action_order: Sequence[str],
    normalization_id: str,
    timestep_schedule_hash: str,
    initial_noise_hash: str,
) -> NoEnvActionSanityResult:
    """Build and compare both prediction contracts for a no-env sanity check."""

    return compare_no_env_action_sanity(
        fine_tuned_contract=build_prediction_contract(
            label="fine_tuned_positive",
            prediction=fine_tuned_prediction,
            action_order=action_order,
            normalization_id=normalization_id,
            timestep_schedule_hash=timestep_schedule_hash,
            initial_noise_hash=initial_noise_hash,
        ),
        frozen_unconditional_contract=build_prediction_contract(
            label="frozen_unconditional",
            prediction=frozen_unconditional_prediction,
            action_order=action_order,
            normalization_id=normalization_id,
            timestep_schedule_hash=timestep_schedule_hash,
            initial_noise_hash=initial_noise_hash,
        ),
    )


def blend_unconditional_swap_prediction(
    *,
    fine_tuned_conditional_prediction: Any,
    frozen_unconditional_prediction: Any,
    weight: float,
) -> Any:
    """Apply the P2 diagnostic formula to two compatible predictions.

    The result is a numpy array for lightweight contract tests.  Runtime hooks
    that need framework-native tensors should preserve this formula inside
    their own inference-only call site after passing the same contract checks.
    """

    guidance_weight = _finite_non_negative_weight(weight)
    if _has_grad_attached(fine_tuned_conditional_prediction) or _has_grad_attached(
        frozen_unconditional_prediction
    ):
        raise ValueError(
            "P2 adapter received tensors with gradients attached; run under "
            "torch.inference_mode() and model.eval()."
        )

    fine_tuned = to_numpy_array(fine_tuned_conditional_prediction)
    frozen = to_numpy_array(frozen_unconditional_prediction)
    if fine_tuned.shape != frozen.shape:
        raise ValueError(f"P2 prediction shape mismatch: {fine_tuned.shape}!={frozen.shape}")
    return (1.0 + guidance_weight) * fine_tuned - guidance_weight * frozen


def build_weight_sweep_smoke(
    *,
    fine_tuned_conditional_prediction: Any,
    frozen_unconditional_prediction: Any,
    weights: Sequence[float] = P2_DEFAULT_WEIGHTS,
) -> dict[str, object]:
    """Return shape/hash summaries for the local synthetic P2 weight sweep."""

    sweep: list[dict[str, object]] = []
    for weight in weights:
        guided = blend_unconditional_swap_prediction(
            fine_tuned_conditional_prediction=fine_tuned_conditional_prediction,
            frozen_unconditional_prediction=frozen_unconditional_prediction,
            weight=float(weight),
        )
        summary = summarize_array(guided)
        sweep.append(
            {
                "weight": float(weight),
                "shape": summary["shape"],
                "dtype": summary["dtype"],
                "sha256": summary["sha256"],
                "finite": int(summary.get("nan_count", 0)) == 0
                and int(summary.get("inf_count", 0)) == 0,
            }
        )
    return {
        "schema_version": P2_SCHEMA_VERSION,
        "diagnostic_only": True,
        "formal_benchmark": False,
        "training_allowed": False,
        "checkpoint_update_allowed": False,
        "weights": [float(weight) for weight in weights],
        "sweep": sweep,
        "runtime_eval_status": "NOT_RUN_SYNTHETIC_SMOKE_ONLY",
    }


def build_weight_sweep_skip_summary(
    *,
    readiness: P2ReadinessDecision,
    sanity_result: NoEnvActionSanityResult,
) -> dict[str, object]:
    """Return an explicit non-execution record when P2 eval is still gated."""

    return {
        "schema_version": P2_SCHEMA_VERSION,
        "diagnostic_only": True,
        "formal_benchmark": False,
        "training_allowed": False,
        "checkpoint_update_allowed": False,
        "weights": [float(weight) for weight in P2_DEFAULT_WEIGHTS],
        "sweep": [],
        "runtime_eval_status": "SKIPPED_P1_P0_GATE_OR_SANITY_NOT_READY",
        "skip_reason": readiness.reason
        if not readiness.allowed
        else "no-env action sanity did not pass; P2 eval is unsafe.",
        "p1_status": readiness.p1_status,
        "p0_status": readiness.p0_status,
        "no_env_action_sanity_status": sanity_result.status,
        "safe_to_eval": sanity_result.safe_to_eval,
    }


def build_adapter_contract_markdown(
    *,
    readiness: P2ReadinessDecision | None = None,
) -> str:
    """Render the P2 temporary adapter contract in Chinese."""

    readiness_block = ""
    if readiness is not None:
        readiness_block = f"""
## 当前 P1/P0 gate

- status: `{readiness.status}`
- allowed: `{readiness.allowed}`
- P1: `{readiness.p1_status}`
- P0: `{readiness.p0_status}`
- reason: {readiness.reason}
"""

    weights = ", ".join(str(weight) for weight in P2_DEFAULT_WEIGHTS)
    return f"""# P2 temporary inference adapter contract

本文件冻结 Stage B P2 的临时推理 adapter 合同。P2 只是 pre-check sanity gate，不是 benchmark，也不是方法成功/失败结论。

## 允许范围

- 仅在 `P1 PASS` 且 `P0 NEGATIVE / no recovery` 后运行。
- 只做 inference-only diagnostic；必须使用 `model.eval()` 与 `torch.inference_mode()` 或等价机制。
- 不允许 optimizer、backward、checkpoint save、LoRA/SFT、训练或 GR00T full long-run。
- small eval 前必须先通过 no-env action sanity。

## 诊断公式

`epsilon_tilde = (1 + w) * epsilon_finetuned_positive - w * epsilon_frozen_unconditional`

计划权重：`{weights}`。

## no-env action sanity 必须证明

- same observation；
- same initial action noise；
- same timestep schedule；
- shape / dtype / action order 一致；
- normalization source 一致；
- prediction finite。

若任一条件不满足，状态必须为 `P2_BLOCKED_UNSAFE`，不得把 blocked 当作 negative。
{readiness_block}
"""


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, title: str, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )


def write_p2_artifacts(
    *,
    output_dir: str | Path,
    p1_status: object,
    p0_status: object,
    sanity_result: NoEnvActionSanityResult,
    sweep_summary: Mapping[str, object],
) -> dict[str, str]:
    """Write the Stage B P2 contract and safety/smoke artifacts."""

    output = Path(output_dir)
    readiness = evaluate_p2_readiness(p1_status=p1_status, p0_status=p0_status)
    effective_sweep_summary = (
        dict(sweep_summary)
        if readiness.allowed and sanity_result.safe_to_eval
        else build_weight_sweep_skip_summary(
            readiness=readiness,
            sanity_result=sanity_result,
        )
    )
    gate_payload = {
        "schema_version": P2_SCHEMA_VERSION,
        "gate": readiness.to_jsonable(),
        "no_env_action_sanity_status": sanity_result.status,
        "weight_sweep_runtime_status": effective_sweep_summary.get(
            "runtime_eval_status"
        ),
        "diagnostic_only": True,
        "formal_benchmark": False,
        "method_claim_allowed": False,
        "training_allowed": False,
        "checkpoint_update_allowed": False,
        "next_step": (
            "Run real P2 small eval only if gate.allowed and sanity.safe_to_eval "
            "are both true, using documented 30 seeds and n_envs=1."
        ),
    }
    paths = {
        "adapter_contract_md": str(output / "p2_adapter_contract.md"),
        "no_env_action_sanity_json": str(output / "p2_no_env_action_sanity.json"),
        "no_env_action_sanity_md": str(output / "p2_no_env_action_sanity.md"),
        "w_sweep_summary_json": str(output / "p2_w_sweep_summary.json"),
        "w_sweep_summary_md": str(output / "p2_w_sweep_summary.md"),
        "gate_decision_json": str(output / "p2_gate_decision.json"),
        "gate_decision_md": str(output / "p2_gate_decision.md"),
    }
    output.mkdir(parents=True, exist_ok=True)
    Path(paths["adapter_contract_md"]).write_text(
        build_adapter_contract_markdown(readiness=readiness),
        encoding="utf-8",
    )
    _write_json(Path(paths["no_env_action_sanity_json"]), sanity_result.to_jsonable())
    _write_markdown(
        Path(paths["no_env_action_sanity_md"]),
        "P2 no-env action sanity",
        sanity_result.to_jsonable(),
    )
    _write_json(Path(paths["w_sweep_summary_json"]), effective_sweep_summary)
    _write_markdown(
        Path(paths["w_sweep_summary_md"]),
        "P2 weight sweep summary",
        effective_sweep_summary,
    )
    _write_json(Path(paths["gate_decision_json"]), gate_payload)
    _write_markdown(Path(paths["gate_decision_md"]), "P2 gate decision", gate_payload)
    return paths


def run_synthetic_self_test(
    output_dir: str | Path | None = None,
    *,
    p1_status: object = "PENDING",
    p0_status: object = "PENDING",
) -> dict[str, object]:
    """Run a tiny no-env adapter smoke without loading GR00T."""

    fine_tuned = [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]]
    frozen = [[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]
    readiness = evaluate_p2_readiness(p1_status=p1_status, p0_status=p0_status)
    sanity = run_no_env_action_sanity(
        fine_tuned_prediction=fine_tuned,
        frozen_unconditional_prediction=frozen,
        action_order=("right_arm", "left_arm", "waist"),
        normalization_id="synthetic_norm_v1",
        timestep_schedule_hash="sha256:synthetic_timestep",
        initial_noise_hash="sha256:synthetic_noise",
    )
    sweep = build_weight_sweep_smoke(
        fine_tuned_conditional_prediction=fine_tuned,
        frozen_unconditional_prediction=frozen,
    )
    effective_sweep = (
        sweep
        if readiness.allowed and sanity.safe_to_eval
        else build_weight_sweep_skip_summary(
            readiness=readiness,
            sanity_result=sanity,
        )
    )
    payload: dict[str, object] = {
        "schema_version": P2_SCHEMA_VERSION,
        "gate": readiness.to_jsonable(),
        "sanity": sanity.to_jsonable(),
        "sweep": effective_sweep,
        "sweep_plan": WeightSweepPlan().to_jsonable(),
    }
    if output_dir is not None:
        paths = write_p2_artifacts(
            output_dir=output_dir,
            p1_status=p1_status,
            p0_status=p0_status,
            sanity_result=sanity,
            sweep_summary=effective_sweep,
        )
        payload["artifact_paths"] = paths
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--p1-status", default="PENDING")
    parser.add_argument("--p0-status", default="PENDING")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    require_stage_b_safe_command(["python3", "-m", __name__, "--self-test"])

    if not args.self_test:
        parser.print_help()
        return 0

    payload = run_synthetic_self_test(
        output_dir=args.output_dir,
        p1_status=args.p1_status,
        p0_status=args.p0_status,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("P2 synthetic self-test PASS")
        if args.output_dir is not None:
            print(f"artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
