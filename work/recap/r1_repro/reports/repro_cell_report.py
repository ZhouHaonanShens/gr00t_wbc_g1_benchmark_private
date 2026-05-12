from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from ..gates import Verdict
from ..gates import newcombe_ci_on_delta


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Verdict):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_repro_cell(
    result: Any,
    verdict: Verdict,
    baseline_rate: float | None,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = getattr(result, "protocol", None)
    episode_count = len(getattr(result, "per_episode", []) or [])
    success_count = int(getattr(result, "success_count", 0))
    payload: dict[str, Any] = {
        "verdict": verdict.value,
        "success_count": success_count,
        "episode_count": episode_count,
        "wilson_ci_on_rate": getattr(result, "wilson_ci_on_rate", None),
        "formal_eval_summary_status": getattr(result, "formal_eval_summary_status", None),
        "stdout_path": getattr(result, "stdout_path", None),
        "stderr_path": getattr(result, "stderr_path", None),
        "run_manifest_path": getattr(result, "run_manifest_path", None),
        "per_episode": getattr(result, "per_episode", []),
        "protocol": protocol,
        "cuda_pin_literal": getattr(protocol, "cuda_visible_devices", None),
        "driver_sha256": getattr(protocol, "driver_sha256", None),
    }
    if baseline_rate is not None and episode_count > 0:
        baseline_count = round(float(baseline_rate) * episode_count)
        delta_ci = newcombe_ci_on_delta(success_count, baseline_count, episode_count, episode_count)
        payload.update(
            {
                "baseline_rate": float(baseline_rate),
                "observed_delta": float((success_count / episode_count) - baseline_rate),
                "newcombe_ci_on_delta": delta_ci,
                "delta_ci_excludes_zero": bool(delta_ci[0] > 0.0 or delta_ci[1] < 0.0),
            }
        )
    _write_json(out_dir / "repro_cell_report.json", payload)

    lines = [
        "# R1 Repro Cell Report",
        "",
        f"- verdict: {verdict.value}",
        f"- success_count: {success_count}",
        f"- episode_count: {episode_count}",
        f"- formal_eval_summary_status: {payload['formal_eval_summary_status']}",
        f"- cuda_pin_literal: {payload['cuda_pin_literal']}",
    ]
    if "observed_delta" in payload:
        lines.extend(
            [
                f"- observed_delta: {payload['observed_delta']:.6f}",
                f"- newcombe_ci_on_delta: {payload['newcombe_ci_on_delta']}",
                f"- delta_ci_excludes_zero: {payload['delta_ci_excludes_zero']}",
            ]
        )
    (out_dir / "repro_cell_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
