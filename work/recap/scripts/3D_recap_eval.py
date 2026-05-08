#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import json
import os
import signal
import sys
import time
import traceback
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from agent.run import state_conditioned_env_resolution
from work.recap.advantage import (
    ADVANTAGE_CONTRACT_VERSION,
    GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS,
    MAINLINE_TASK_TEXT_FIELD,
    NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
    NUMERIC_ADVANTAGE_EVAL_DIAGNOSTIC_ROUTE,
    build_diagnostic_surface_metadata,
)
from work.recap.identity import validate_preflight_report_for_entrypoint


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


DEFAULT_ENV_NAME = (
    state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_N_EPISODES = 10
DEFAULT_MAX_EPISODE_STEPS = 1440
DEFAULT_RUNTIME_LOG_DIR = "agent/runtime_logs/recap_3D"
DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_TELEMETRY_DIR = "agent/artifacts/recap_eval_telemetry"
DEFAULT_CONNECT_TIMEOUT_S = 300
DEFAULT_TOTAL_TIMEOUT_S = 1800
DEFAULT_G1_EXECUTION_N_ACTION_STEPS = 20
DEFAULT_G1_PHASE_A_N_ACTION_STEPS = DEFAULT_G1_EXECUTION_N_ACTION_STEPS


SUCCESS_KEYS: tuple[str, ...] = (
    "success",
    "is_success",
    "task_success",
    "episode_success",
    "success_episode",
    "goal_achieved",
    "task_complete",
    "completed",
)


class _TeeStream:
    def __init__(self, *targets: TextIO):
        self._targets = targets

    def write(self, s: str) -> int:
        for t in self._targets:
            t.write(s)
        return len(s)

    def flush(self) -> None:
        for t in self._targets:
            t.flush()


@contextlib.contextmanager
def _tee_stdio(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        stdout = _TeeStream(sys.stdout, f)
        stderr = _TeeStream(sys.stderr, f)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "agent").is_dir() and (parent / "work").is_dir():
            return parent
    return Path.cwd().resolve()


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _add_import_roots(repo_root: Path) -> None:
    submodule_root = repo_root / "submodules" / "Isaac-GR00T"
    wbc_ext_root = submodule_root / "external_dependencies" / "GR00T-WholeBodyControl"
    wbc_robocasa_root = wbc_ext_root / "gr00t_wbc" / "dexmg" / "gr00trobocasa"
    wbc_robosuite_root = wbc_ext_root / "gr00t_wbc" / "dexmg" / "gr00trobosuite"
    robocasa_ext_root = submodule_root / "external_dependencies" / "robocasa"
    for p in (
        repo_root,
        submodule_root,
        wbc_ext_root,
        wbc_robosuite_root,
        robocasa_ext_root,
        wbc_robocasa_root,
    ):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


def _timestamp() -> str:
    now = _dt.datetime.now()
    ms = int(now.microsecond // 1000)
    return now.strftime("%Y%m%d_%H%M%S") + f"_{ms:03d}_pid{os.getpid()}"


def _parse_advantage(raw: str) -> float | None:
    s = str(raw).strip()
    if s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid --advantage value: {raw!r}. Expected 'None' or a float. ({type(e).__name__}: {e})"
        )


def _advantage_mode(value: float | None) -> str:
    if value is None:
        return "unconditional"
    if float(value) == 0.0:
        return "explicit_neutral"
    if float(value) > 0.0:
        return "explicit_positive"
    return "explicit_negative"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="3D_recap_eval.py",
        description=(
            "RECAP evaluation script with explicit advantage conditioning options passthrough."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME)
    p.add_argument("--host", type=str, default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    p.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    p.add_argument(
        "--advantage",
        type=_parse_advantage,
        default=None,
        help=(
            "Policy conditioning value (float). Use 'None' to omit. "
            "Canonical values: 0.0/1.0; ablations may also use -1.0."
        ),
    )
    p.add_argument("--runtime-log-dir", type=str, default=DEFAULT_RUNTIME_LOG_DIR)
    p.add_argument("--artifact-dir", type=str, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument(
        "--telemetry-dir",
        type=str,
        default=DEFAULT_TELEMETRY_DIR,
        help=(
            "Directory for richer per-step / per-episode telemetry JSONL files. "
            "Ignored when --no-save-telemetry is used."
        ),
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional summary JSON output path. If empty, write to agent/artifacts.",
    )
    p.add_argument(
        "--canonical-preflight-report",
        type=str,
        default="",
        help=(
            "Required Phase-1 canonical identity STRICT_PROMOTION PASS report. "
            "Eval fails before server/env startup when missing or non-PASS."
        ),
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate --canonical-preflight-report and exit before server/env startup.",
    )
    p.add_argument("--connect-timeout-s", type=float, default=DEFAULT_CONNECT_TIMEOUT_S)
    p.add_argument("--total-timeout-s", type=float, default=DEFAULT_TOTAL_TIMEOUT_S)
    p.add_argument(
        "--save-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write richer per-step and per-episode telemetry JSONL artifacts.",
    )
    p.add_argument(
        "--debug-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print success-related info keys and value previews per episode.",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=0,
        help=(
            "Override executed action steps per policy call. If <=0, G1 loco-manip envs default to the "
            "formal execution-window value 20; other envs infer from server modality config. "
            "This must stay distinct from the server policy horizon."
        ),
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Per-episode seed uses seed_base + episode_index.",
    )
    return p


def _default_n_action_steps_for_env(env_name: str) -> int | None:
    env = str(env_name).strip()
    if env.startswith("gr00tlocomanip_g1_sim/"):
        return int(DEFAULT_G1_EXECUTION_N_ACTION_STEPS)
    return None


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.n_episodes) <= 0:
        raise ValueError(f"--n-episodes must be > 0, got {args.n_episodes}")
    if int(args.max_episode_steps) <= 0:
        raise ValueError(
            f"--max-episode-steps must be > 0, got {args.max_episode_steps}"
        )
    if float(args.connect_timeout_s) <= 0.0:
        raise ValueError(
            f"--connect-timeout-s must be > 0, got {args.connect_timeout_s}"
        )
    if float(args.total_timeout_s) <= 0.0:
        raise ValueError(f"--total-timeout-s must be > 0, got {args.total_timeout_s}")


def _require_canonical_preflight_report(raw_report: str) -> Mapping[str, Any]:
    report_text = str(raw_report or "").strip()
    if not report_text:
        raise ValueError(
            "--canonical-preflight-report is required before GR00T eval/server initialization"
        )
    return validate_preflight_report_for_entrypoint(Path(report_text), require_strict=True)


def _append_purity_blocker(blockers: list[str], key: str, detail: str) -> None:
    blockers.append(f"{key}: {detail}")


def _normalized_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def build_saved_telemetry_surface(
    summary_payload: Mapping[str, Any],
) -> dict[str, object]:
    telemetry_enabled = bool(summary_payload.get("telemetry_enabled", False))
    return {
        "telemetry_enabled": telemetry_enabled,
        "step_telemetry_jsonl": _normalized_optional_str(
            summary_payload.get("step_telemetry_jsonl")
        ),
        "episode_telemetry_jsonl": _normalized_optional_str(
            summary_payload.get("episode_telemetry_jsonl")
        ),
        "telemetry_step_count": int(
            summary_payload.get("telemetry_step_count", 0) or 0
        ),
        "telemetry_episode_count": int(
            summary_payload.get("telemetry_episode_count", 0) or 0
        ),
    }


def _build_public_episode_result(episode_record: Mapping[str, Any]) -> dict[str, object]:
    return {
        "episode_index": int(episode_record.get("episode_index", 0) or 0),
        "seed": int(episode_record.get("seed", 0) or 0),
        "success": bool(episode_record.get("success", False)),
        "episode_elapsed_seconds": float(
            episode_record.get("episode_elapsed_seconds", 0.0) or 0.0
        ),
        "done": bool(episode_record.get("done", False)),
        "terminated": bool(episode_record.get("terminated", False)),
        "truncated": bool(episode_record.get("truncated", False)),
        "outer_steps": int(episode_record.get("outer_steps", 0) or 0),
        "failure_reason": episode_record.get("failure_reason"),
    }


def _normalized_overlay_exclude_regex(value: Any) -> str | None:
    text = _normalized_optional_str(value)
    if text in (None, r"$^"):
        return None
    return text


def _require_mainline_server_purity(provenance: dict[str, Any] | None) -> str:
    if not isinstance(provenance, dict):
        raise ValueError(
            "server_provenance_missing: eval requires server get_provenance/get_server_info.provenance payload"
        )
    blockers: list[str] = []
    purity_mode = "mainline_no_overlay"

    contract_version = provenance.get("advantage_contract_version")
    if contract_version in (None, ""):
        _append_purity_blocker(
            blockers,
            "contract_version_missing",
            "server provenance is missing advantage_contract_version",
        )
    elif str(contract_version) != str(ADVANTAGE_CONTRACT_VERSION):
        _append_purity_blocker(
            blockers,
            "contract_version_mismatch",
            (f"expected {ADVANTAGE_CONTRACT_VERSION!r} got {contract_version!r}"),
        )

    injection_rule = provenance.get("advantage_injection_rule")
    if str(injection_rule) != "sign_consistent":
        _append_purity_blocker(
            blockers,
            "sign_consistent_required",
            f"server provenance advantage_injection_rule={injection_rule!r}",
        )

    if provenance.get("require_advantage_embedding") is not True:
        _append_purity_blocker(
            blockers,
            "advantage_embedding_required",
            (
                "server provenance require_advantage_embedding must be true under Phase A mainline, got "
                f"{provenance.get('require_advantage_embedding')!r}"
            ),
        )

    if provenance.get("legacy_negate_enabled") is not False:
        _append_purity_blocker(
            blockers,
            "legacy_negate_forbidden",
            (
                "server provenance legacy_negate_enabled must be false under Phase A mainline, got "
                f"{provenance.get('legacy_negate_enabled')!r}"
            ),
        )

    overlay_from = _normalized_optional_str(provenance.get("overlay_from"))
    if overlay_from is not None:
        overlay_include_regex = _normalized_optional_str(
            provenance.get("overlay_include_regex")
        )
        overlay_exclude_regex = _normalized_overlay_exclude_regex(
            provenance.get("overlay_exclude_regex")
        )
        base_model_path = _normalized_optional_str(provenance.get("base_model_path"))

        if base_model_path is None:
            _append_purity_blocker(
                blockers,
                "overlay_base_model_missing",
                "server provenance must declare base_model_path when overlay_from is used",
            )

        if overlay_include_regex != r"^action_head\..*":
            _append_purity_blocker(
                blockers,
                "overlay_include_regex_invalid",
                (
                    "server provenance overlay_include_regex must be exactly "
                    f"'^action_head\\..*' for the RECAP action-head lane, got {overlay_include_regex!r}"
                ),
            )

        if overlay_exclude_regex is not None:
            _append_purity_blocker(
                blockers,
                "overlay_exclude_regex_forbidden",
                f"server provenance overlay_exclude_regex={overlay_exclude_regex!r}",
            )

        overlay_path = Path(overlay_from).expanduser()
        if not overlay_path.is_absolute():
            _append_purity_blocker(
                blockers,
                "overlay_path_not_local_absolute",
                (
                    "server provenance overlay_from must be a local absolute checkpoint path, got "
                    f"{overlay_from!r}"
                ),
            )

        if base_model_path is not None and base_model_path == overlay_from:
            _append_purity_blocker(
                blockers,
                "overlay_same_as_base_model",
                "server provenance overlay_from must differ from base_model_path",
            )

        if not blockers:
            purity_mode = "action_head_overlay"

    if blockers:
        raise ValueError("; ".join(blockers))
    return purity_mode


def _scalarize_bool(x: Any) -> bool:
    import numpy as np

    return bool(np.asarray(x).reshape(-1)[0])


def _scalarize_float(x: Any) -> float:
    import numpy as np

    return float(np.asarray(x).reshape(-1)[0])


def _reduce_success_value(x: Any) -> bool:
    import numpy as np

    def _reduce(v: Any) -> bool:
        if isinstance(v, np.ndarray):
            if v.size <= 0:
                return False
            if v.dtype == object:
                return any(_reduce(e) for e in v.reshape(-1).tolist())
            return bool(np.any(v))
        if isinstance(v, (list, tuple)):
            return any(_reduce(e) for e in v)
        if isinstance(v, (bool, int, float)):
            return bool(v)
        try:
            return bool(v)
        except Exception:
            return False

    return _reduce(x)


def _first_env_item(x: Any) -> Any:
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            if x.size <= 0:
                return None
            return x.reshape(-1)[0]
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else None
    return x


def _extract_success_step(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    for key in SUCCESS_KEYS:
        if key in info and _reduce_success_value(info.get(key)):
            return True
    final_info = info.get("final_info")
    if final_info is None:
        return False
    final0 = _first_env_item(final_info)
    if final0 is None:
        return False
    if isinstance(final0, dict):
        for key in SUCCESS_KEYS:
            if key in final0 and _reduce_success_value(final0.get(key)):
                return True
        for nested_key in ("info", "metrics", "stats"):
            if nested_key not in final0:
                continue
            nested = final0.get(nested_key)
            if isinstance(nested, dict):
                for key in SUCCESS_KEYS:
                    if key in nested and _reduce_success_value(nested.get(key)):
                        return True
    return False


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=True, sort_keys=True)
        f.write("\n")


def _iter_env_chain(root: Any) -> Iterator[Any]:
    seen: set[int] = set()
    queue: list[Any] = [root]
    while queue:
        cur = queue.pop(0)
        if cur is None:
            continue
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)
        yield cur
        for attr in ("env", "unwrapped", "base_env"):
            if not hasattr(cur, attr):
                continue
            try:
                nxt = getattr(cur, attr)
            except Exception:
                continue
            if nxt is not None:
                queue.append(nxt)


def _single_env_root(env: Any) -> Any:
    envs = getattr(env, "envs", None)
    if isinstance(envs, list) and len(envs) > 0:
        return envs[0]
    return env


def _safe_vec3(x: Any) -> list[float] | None:
    try:
        import numpy as np

        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if int(arr.size) < 3:
            return None
        return [float(arr[0]), float(arr[1]), float(arr[2])]
    except Exception:
        return None


def _distance_l2(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    try:
        import numpy as np

        av = np.asarray(a, dtype=np.float64).reshape(3)
        bv = np.asarray(b, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(av - bv))
    except Exception:
        return None


def _find_named_body_id(obj_body_id: dict[str, Any] | None, needle: str) -> int | None:
    if not isinstance(obj_body_id, dict):
        return None
    lowered = str(needle).lower()
    for key, value in obj_body_id.items():
        if str(key).lower() == lowered:
            try:
                return int(value)
            except Exception:
                return None
    for key, value in obj_body_id.items():
        if lowered in str(key).lower():
            try:
                return int(value)
            except Exception:
                return None
    return None


def _find_sim(root: Any) -> Any | None:
    for cur in _iter_env_chain(root):
        sim = getattr(cur, "sim", None)
        if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
            return sim
    return None


def _find_obj_body_id(root: Any) -> dict[str, Any] | None:
    for cur in _iter_env_chain(root):
        obj_body_id = getattr(cur, "obj_body_id", None)
        if isinstance(obj_body_id, dict):
            return obj_body_id
    return None


def _find_robot_eef_site_id(root: Any, side: str) -> int | None:
    side_key = str(side).lower()
    for cur in _iter_env_chain(root):
        robots = getattr(cur, "robots", None)
        if not isinstance(robots, (list, tuple)) or len(robots) <= 0:
            continue
        robot0 = robots[0]
        eef_site_id = getattr(robot0, "eef_site_id", None)
        if not isinstance(eef_site_id, dict):
            continue
        if side_key in eef_site_id:
            try:
                return int(eef_site_id[side_key])
            except Exception:
                return None
        for key, value in eef_site_id.items():
            if side_key in str(key).lower():
                try:
                    return int(value)
                except Exception:
                    return None
    return None


def _body_pos_from_sim(sim: Any, body_id: int | None) -> list[float] | None:
    if sim is None or body_id is None:
        return None
    try:
        return _safe_vec3(sim.data.body_xpos[int(body_id)])
    except Exception:
        return None


def _site_pos_from_sim(sim: Any, site_id: int | None) -> list[float] | None:
    if sim is None or site_id is None:
        return None
    try:
        return _safe_vec3(sim.data.site_xpos[int(site_id)])
    except Exception:
        return None


def _fallback_body_id_from_model(sim: Any, needle: str) -> int | None:
    if sim is None:
        return None
    try:
        names = [str(x) for x in getattr(sim.model, "body_names", [])]
    except Exception:
        return None
    lowered = str(needle).lower()
    for name in names:
        if lowered in name.lower():
            try:
                return int(sim.model.body_name2id(name))
            except Exception:
                return None
    return None


def _extract_intermediate_signals(info: Any) -> dict[str, Any] | None:
    if not isinstance(info, dict):
        return None
    raw = info.get("intermediate_signals")
    payload = raw if isinstance(raw, dict) else _first_env_item(raw)
    if not isinstance(payload, dict):
        return None
    out: dict[str, Any] = {}
    for key, value in payload.items():
        try:
            import numpy as np

            arr = np.asarray(value)
            if arr.dtype == np.bool_:
                out[str(key)] = [bool(v) for v in arr.reshape(-1).tolist()]
            elif np.issubdtype(arr.dtype, np.number):
                out[str(key)] = [float(v) for v in arr.reshape(-1).tolist()]
            else:
                out[str(key)] = arr.reshape(-1).tolist()
        except Exception:
            out[str(key)] = repr(value)
    return out


def _summarize_action_chunk(action: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception:
        return {k: {"shape": []} for k in action.keys()}

    summary: dict[str, Any] = {}
    for key, value in action.items():
        arr = np.asarray(value, dtype=np.float32)
        signed_flat = arr.reshape(-1) if int(arr.size) > 0 else np.asarray([])
        flat = np.abs(signed_flat) if int(arr.size) > 0 else np.asarray([])
        preview = flat[: min(6, int(flat.size))].tolist() if int(flat.size) > 0 else []
        signed_preview = (
            signed_flat[: min(6, int(signed_flat.size))].tolist()
            if int(signed_flat.size) > 0
            else []
        )
        summary[str(key)] = {
            "shape": [int(x) for x in arr.shape],
            "mean": float(np.mean(signed_flat)) if int(signed_flat.size) > 0 else 0.0,
            "sum": float(np.sum(signed_flat)) if int(signed_flat.size) > 0 else 0.0,
            "l2": float(np.linalg.norm(signed_flat)) if int(signed_flat.size) > 0 else 0.0,
            "mean_abs": float(np.mean(flat)) if int(flat.size) > 0 else 0.0,
            "sum_abs": float(np.sum(flat)) if int(flat.size) > 0 else 0.0,
            "max_abs": float(np.max(flat)) if int(flat.size) > 0 else 0.0,
            "p95_abs": float(np.quantile(flat, 0.95)) if int(flat.size) > 0 else 0.0,
            "q99_abs": float(np.quantile(flat, 0.99)) if int(flat.size) > 0 else 0.0,
            "abs_preview": [float(v) for v in preview],
            "signed_preview": [float(v) for v in signed_preview],
        }
    return summary


def _collect_env_snapshot(env: Any) -> dict[str, Any]:
    root = _single_env_root(env)
    sim = _find_sim(root)
    obj_body_id = _find_obj_body_id(root)

    apple_body_id = _find_named_body_id(obj_body_id, "apple")
    if apple_body_id is None:
        apple_body_id = _fallback_body_id_from_model(sim, "apple")
    plate_body_id = _find_named_body_id(obj_body_id, "plate")
    if plate_body_id is None:
        plate_body_id = _fallback_body_id_from_model(sim, "plate")

    right_eef_site_id = _find_robot_eef_site_id(root, "right")
    left_eef_site_id = _find_robot_eef_site_id(root, "left")

    apple_pos = _body_pos_from_sim(sim, apple_body_id)
    plate_pos = _body_pos_from_sim(sim, plate_body_id)
    right_eef_pos = _site_pos_from_sim(sim, right_eef_site_id)
    left_eef_pos = _site_pos_from_sim(sim, left_eef_site_id)

    sim_time = None
    if sim is not None:
        try:
            sim_time = float(sim.data.time)
        except Exception:
            sim_time = None

    apple_to_right_eef = _distance_l2(apple_pos, right_eef_pos)
    apple_to_left_eef = _distance_l2(apple_pos, left_eef_pos)
    apple_to_plate = _distance_l2(apple_pos, plate_pos)
    right_eef_to_plate = _distance_l2(right_eef_pos, plate_pos)

    return {
        "sim_time_s": sim_time,
        "apple_body_id": apple_body_id,
        "plate_body_id": plate_body_id,
        "right_eef_site_id": right_eef_site_id,
        "left_eef_site_id": left_eef_site_id,
        "apple_pos_xyz": apple_pos,
        "plate_pos_xyz": plate_pos,
        "right_eef_pos_xyz": right_eef_pos,
        "left_eef_pos_xyz": left_eef_pos,
        "apple_to_right_eef_l2": apple_to_right_eef,
        "apple_to_left_eef_l2": apple_to_left_eef,
        "apple_to_plate_l2": apple_to_plate,
        "right_eef_to_plate_l2": right_eef_to_plate,
        "apple_height_z": None if apple_pos is None else float(apple_pos[2]),
        "plate_height_z": None if plate_pos is None else float(plate_pos[2]),
        "right_eef_height_z": None
        if right_eef_pos is None
        else float(right_eef_pos[2]),
    }


def _episode_failure_reason(
    *,
    success: bool,
    done: bool,
    terminated: bool,
    truncated: bool,
    outer_steps: int,
    outer_max_steps: int,
) -> str | None:
    if success:
        return None
    if truncated:
        return "truncated_without_success"
    if terminated:
        return "terminated_without_success"
    if done:
        return "done_without_success"
    if int(outer_steps) >= int(outer_max_steps):
        return "outer_step_budget_exhausted"
    return "episode_incomplete_without_success"


def _failure_stage_guess(step_records: list[dict[str, Any]]) -> dict[str, Any]:
    hand_thresh_m = 0.10
    lift_thresh_m = 0.03
    plate_thresh_m = 0.12

    hand_dists = [
        float(v)
        for rec in step_records
        for v in [rec.get("apple_to_right_eef_l2")]
        if isinstance(v, (int, float))
    ]
    plate_dists = [
        float(v)
        for rec in step_records
        for v in [rec.get("apple_to_plate_l2")]
        if isinstance(v, (int, float))
    ]
    apple_heights = [
        float(v)
        for rec in step_records
        for v in [rec.get("apple_height_z")]
        if isinstance(v, (int, float))
    ]

    initial_height = apple_heights[0] if apple_heights else None
    max_lift = None
    if apple_heights and initial_height is not None:
        max_lift = float(max(apple_heights) - float(initial_height))

    min_hand = min(hand_dists) if hand_dists else None
    min_plate = min(plate_dists) if plate_dists else None
    ever_near_apple = bool(min_hand is not None and float(min_hand) <= hand_thresh_m)
    ever_lifted = bool(max_lift is not None and float(max_lift) >= lift_thresh_m)
    ever_near_plate = bool(min_plate is not None and float(min_plate) <= plate_thresh_m)

    if not ever_near_apple:
        label = "never_reached_apple"
    elif not ever_lifted:
        label = "reached_apple_not_lifted"
    elif not ever_near_plate:
        label = "lifted_not_brought_to_plate"
    else:
        label = "near_plate_but_not_success"

    return {
        "label": label,
        "ever_near_apple": ever_near_apple,
        "ever_lifted_apple": ever_lifted,
        "ever_near_plate": ever_near_plate,
        "min_apple_to_right_eef_l2": min_hand,
        "min_apple_to_plate_l2": min_plate,
        "max_apple_lift_z": max_lift,
        "thresholds": {
            "near_apple_m": float(hand_thresh_m),
            "lift_m": float(lift_thresh_m),
            "near_plate_m": float(plate_thresh_m),
        },
    }


def _install_alarm_timeout(timeout_s: float) -> None:
    if not hasattr(signal, "SIGALRM"):
        return
    t = int(float(timeout_s))
    if t <= 0:
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Timed out after {t}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(t)


def _clear_alarm_timeout() -> None:
    if hasattr(signal, "SIGALRM"):
        try:
            signal.alarm(0)
        except Exception:
            pass


def _install_robocasa_import_shims() -> None:
    import sys
    import types
    from collections.abc import Callable
    from pathlib import Path
    from typing import cast

    def _install_gradient_ctor_compat(module_obj: Any) -> None:
        import inspect

        gradient_cls = getattr(module_obj, "Gradient", None)
        if gradient_cls is None:
            return
        needs_patch = False
        try:
            sig = inspect.signature(gradient_cls)
            params = tuple(sig.parameters.values())
            has_varargs = any(
                p.kind
                in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for p in params
            )
            positional_capacity = sum(
                1
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            needs_patch = (not has_varargs) and (positional_capacity < 2)
        except Exception:
            needs_patch = True
        if not needs_patch:
            return

        class _GradientCompat:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs
                self.rgba_a = kwargs.get("rgba_a", args[0] if len(args) > 0 else None)
                self.rgba_b = kwargs.get("rgba_b", args[1] if len(args) > 1 else None)

        setattr(module_obj, "Gradient", _GradientCompat)
        print(
            "[WARN] installed robocasa shim: patched Gradient constructor compatibility"
        )

    try:
        obj_utils = importlib.import_module("robocasa.utils.object_utils")
        if not hasattr(obj_utils, "check_obj_upright"):
            obj_cos_fn: Any = getattr(obj_utils, "obj_cos", None)

            def check_obj_upright(
                env: Any,
                obj_name: str,
                threshold: float = 0.8,
                symmetric: bool = False,
            ) -> bool:
                if not callable(obj_cos_fn):
                    return False
                try:
                    raw_alignment: Any = obj_cos_fn(
                        env, obj_name=obj_name, ref=(0, 0, 1)
                    )
                    z_alignment = float(raw_alignment)
                except Exception:
                    return False
                if bool(symmetric):
                    z_alignment = abs(z_alignment)
                return bool(z_alignment > float(threshold))

            setattr(obj_utils, "check_obj_upright", check_obj_upright)
            print("[WARN] installed check_obj_upright compatibility shim")
    except Exception:
        pass

    try:
        robots_mod = importlib.import_module("robocasa.models.robots")
        if not hasattr(robots_mod, "GR00T_LOCOMANIP_ENVS_ROBOTS"):
            setattr(robots_mod, "GR00T_LOCOMANIP_ENVS_ROBOTS", {"G1": "g1_sim"})
            print(
                "[WARN] installed robocasa shim: GR00T_LOCOMANIP_ENVS_ROBOTS={'G1':'g1_sim'}"
            )
        if not hasattr(robots_mod, "remove_mimic_joints"):

            def remove_mimic_joints(_gripper: object, action: Any) -> Any:
                return action

            setattr(robots_mod, "remove_mimic_joints", remove_mimic_joints)
            print("[WARN] installed robocasa shim: remove_mimic_joints passthrough")
    except Exception:
        pass

    try:
        bases_mod = importlib.import_module("robosuite.models.bases")
        base_mapping = getattr(bases_mod, "BASE_MAPPING", None)
        if isinstance(base_mapping, dict) and "NullBase" not in base_mapping:
            if "NoActuationBase" in base_mapping:
                base_mapping["NullBase"] = base_mapping["NoActuationBase"]
                print(
                    "[WARN] installed robosuite shim: BASE_MAPPING['NullBase'] -> NoActuationBase"
                )
            elif "NullMobileBase" in base_mapping:
                base_mapping["NullBase"] = base_mapping["NullMobileBase"]
                print(
                    "[WARN] installed robosuite shim: BASE_MAPPING['NullBase'] -> NullMobileBase"
                )
    except Exception:
        pass

    try:
        ctrl_mod = importlib.import_module("robosuite.controllers.parts.controller")
        Controller = getattr(ctrl_mod, "Controller", None)
        did_patch = False
        if Controller is not None and not hasattr(
            Controller, "use_external_torque_compensation"
        ):
            setattr(Controller, "use_external_torque_compensation", False)
            did_patch = True
        if Controller is not None and not hasattr(
            Controller, "external_torque_compensation"
        ):
            setattr(Controller, "external_torque_compensation", None)
            did_patch = True
        if did_patch:
            print(
                "[WARN] installed robosuite shim: default external torque compensation flags"
            )
    except Exception:
        pass

    try:
        visuals_utls_mod = importlib.import_module("robocasa.utils.visuals_utls")
        _install_gradient_ctor_compat(visuals_utls_mod)
    except ModuleNotFoundError:
        m = types.ModuleType("robocasa.utils.visuals_utls")

        class Gradient:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                return None

        def randomize_materials_rgba(*_args: object, **_kwargs: object) -> None:
            return None

        setattr(m, "Gradient", Gradient)
        setattr(m, "randomize_materials_rgba", randomize_materials_rgba)
        sys.modules["robocasa.utils.visuals_utls"] = m
        _install_gradient_ctor_compat(m)
        print("[WARN] installed robocasa shim: created visuals_utls stub module")
    except Exception:
        return

    try:
        importlib.import_module("robocasa.wrappers.ik_wrapper")
    except ModuleNotFoundError:
        wrappers_mod = types.ModuleType("robocasa.wrappers")
        ik_mod = types.ModuleType("robocasa.wrappers.ik_wrapper")

        class IKWrapper:
            def __init__(self, env: object, **_kwargs: object):
                self.env = env

            def __getattr__(self, name: str) -> object:
                return getattr(self.env, name)

        setattr(ik_mod, "IKWrapper", IKWrapper)
        sys.modules.setdefault("robocasa.wrappers", wrappers_mod)
        sys.modules["robocasa.wrappers.ik_wrapper"] = ik_mod
        print("[WARN] installed robocasa shim: created wrappers.ik_wrapper stub")
    except Exception:
        return

    try:
        sync_env_mod = importlib.import_module(
            "gr00t_wbc.control.envs.robocasa.sync_env"
        )
        G1SyncEnv = getattr(sync_env_mod, "G1SyncEnv", None)
        orig_init = (
            getattr(G1SyncEnv, "__init__", None) if G1SyncEnv is not None else None
        )
        if callable(orig_init) and G1SyncEnv is not None:
            orig_init_fn = cast(Callable[..., Any], orig_init)

            def _patched_init(self, env_name: str, robot_name: str, **kwargs: Any):
                renderer = kwargs.get("renderer", "mjviewer")
                if renderer == "mjviewer":
                    render_camera = kwargs.get("render_camera")
                    if render_camera is None:
                        kwargs["render_camera"] = "robot0_oak_egoview"
                    elif isinstance(render_camera, list):
                        kwargs["render_camera"] = (
                            str(render_camera[0])
                            if len(render_camera) > 0
                            else "robot0_oak_egoview"
                        )
                return orig_init_fn(
                    self, env_name=env_name, robot_name=robot_name, **kwargs
                )

            setattr(G1SyncEnv, "__init__", _patched_init)
            print(
                "[WARN] installed gr00t_wbc shim: coerce G1SyncEnv.render_camera to str"
            )
    except Exception:
        pass

    try:
        env_mod = importlib.import_module(
            "gr00t_wbc.control.envs.robocasa.utils.robocasa_env"
        )
        EnvCls = getattr(env_mod, "Gr00tLocomanipRoboCasaEnv", None)
        orig_get_obs = (
            getattr(EnvCls, "get_gr00t_observation", None) if EnvCls else None
        )
        if callable(orig_get_obs) and EnvCls is not None:
            orig_get_obs_fn = cast(Callable[..., Any], orig_get_obs)

            def _patched_get_gr00t_observation(self, raw_obs: Any) -> Any:
                if not isinstance(raw_obs, dict):
                    return orig_get_obs_fn(self, raw_obs)
                import numpy as np

                if "robot0_torso_link_imu_quat" not in raw_obs:
                    raw_obs["robot0_torso_link_imu_quat"] = np.zeros(
                        (4,), dtype=np.float32
                    )
                if "robot0_torso_link_imu_vel" not in raw_obs:
                    raw_obs["robot0_torso_link_imu_vel"] = np.zeros(
                        (6,), dtype=np.float32
                    )
                if "robot0_joint_acc" not in raw_obs and "robot0_joint_vel" in raw_obs:
                    raw_obs["robot0_joint_acc"] = np.zeros_like(
                        raw_obs["robot0_joint_vel"]
                    )
                if (
                    "robot0_left_gripper_qacc" not in raw_obs
                    and "robot0_left_gripper_qvel" in raw_obs
                ):
                    raw_obs["robot0_left_gripper_qacc"] = np.zeros_like(
                        raw_obs["robot0_left_gripper_qvel"]
                    )
                if (
                    "robot0_right_gripper_qacc" not in raw_obs
                    and "robot0_right_gripper_qvel" in raw_obs
                ):
                    raw_obs["robot0_right_gripper_qacc"] = np.zeros_like(
                        raw_obs["robot0_right_gripper_qvel"]
                    )
                return orig_get_obs_fn(self, raw_obs)

            setattr(EnvCls, "get_gr00t_observation", _patched_get_gr00t_observation)
            print(
                "[WARN] installed gr00t_wbc shim: fill missing robot0_*_acc keys with zeros"
            )
    except Exception:
        pass

    try:
        controller_utils = importlib.import_module(
            "gr00t_wbc.control.envs.robocasa.utils.controller_utils"
        )
        orig_update_controller_cfg = getattr(
            controller_utils, "update_robosuite_controller_configs", None
        )
        if callable(orig_update_controller_cfg):

            def _patched_update_robosuite_controller_configs(
                robot: str,
                wbc_version: str | None = None,
                enable_gravity_compensation: bool = False,
            ) -> Any:
                cfg = orig_update_controller_cfg(
                    robot=robot,
                    wbc_version=wbc_version,
                    enable_gravity_compensation=enable_gravity_compensation,
                )
                target_name = "default_mink_ik_g1_gear_wbc.json"
                if not str(cfg).endswith(target_name):
                    return cfg
                cfg_path = Path(str(cfg))
                if not cfg_path.is_absolute():
                    try:
                        robocasa_mod = importlib.import_module("robocasa")
                        cfg_path = (
                            Path(str(getattr(robocasa_mod, "__file__", "")))
                            .resolve()
                            .parent
                            / ".."
                            / str(cfg)
                        ).resolve()
                    except Exception:
                        cfg_path = Path(str(cfg))
                if cfg_path.is_file():
                    return cfg

                module_file = Path(
                    str(getattr(controller_utils, "__file__", ""))
                ).resolve()
                gr00t_wbc_root = None
                for p in (module_file.parent, *module_file.parents):
                    if p.name == "gr00t_wbc":
                        gr00t_wbc_root = p
                        break
                if gr00t_wbc_root is None:
                    return cfg

                fallback_cfg = (
                    gr00t_wbc_root
                    / "dexmg"
                    / "gr00trobosuite"
                    / "robosuite"
                    / "examples"
                    / "third_party_controller"
                    / "default_mink_ik_gr1.json"
                )
                if not fallback_cfg.is_file():
                    return cfg

                print("[WARN] installed controller config fallback: g1_gear_wbc -> gr1")
                return str(fallback_cfg)

            setattr(
                controller_utils,
                "update_robosuite_controller_configs",
                _patched_update_robosuite_controller_configs,
            )
    except Exception:
        return

    try:
        env_utils = importlib.import_module("gr00t.eval.sim.env_utils")
        tags_mod = importlib.import_module("gr00t.data.embodiment_tags")
        EmbodimentTag = getattr(tags_mod, "EmbodimentTag")
        orig_fn = getattr(env_utils, "get_embodiment_tag_from_env_name")

        def _patched_get_embodiment_tag_from_env_name(env_name: str):
            if str(env_name).split("/")[0] == "gr00tlocomanip_G1":
                return EmbodimentTag.UNITREE_G1
            return orig_fn(env_name)

        setattr(
            env_utils,
            "get_embodiment_tag_from_env_name",
            _patched_get_embodiment_tag_from_env_name,
        )
        print("[WARN] installed gr00t env_utils shim: gr00tlocomanip_G1 -> UNITREE_G1")

        try:
            js_mod = importlib.import_module(
                "gr00t_wbc.control.envs.g1.utils.joint_safety"
            )
            JointSafetyMonitor = getattr(js_mod, "JointSafetyMonitor", None)
            if JointSafetyMonitor is not None and callable(
                getattr(JointSafetyMonitor, "trigger_system_shutdown", None)
            ):

                def _no_safety(self, obs, action):
                    return {
                        "safe_to_continue": True,
                        "action": action,
                        "shutdown_required": False,
                    }

                setattr(JointSafetyMonitor, "handle_violations", _no_safety)

                def _no_shutdown(self):
                    if not bool(getattr(self, "_shutdown_suppressed_once", False)):
                        setattr(self, "_shutdown_suppressed_once", True)
                        print(
                            "[WARN] joint safety shutdown suppressed (sim): trigger_system_shutdown() called"
                        )
                    return None

                setattr(JointSafetyMonitor, "trigger_system_shutdown", _no_shutdown)
                print(
                    "[WARN] installed g1 joint safety shim: suppress trigger_system_shutdown sys.exit"
                )
        except Exception:
            pass
    except Exception:
        return


def _ensure_explicit_g1_env_registration(gym_module: Any) -> dict[str, Any]:
    sync_env_module_name = "gr00t_wbc.control.envs.robocasa.sync_env"
    before_ids = state_conditioned_env_resolution.registered_g1_env_ids(gym_module)
    print("[INFO] registered_g1_env_count_before_explicit_import:", len(before_ids))
    try:
        sync_env_mod = importlib.import_module(sync_env_module_name)
    except Exception as e:
        raise RuntimeError(
            "explicit G1 env registration failed: could not import "
            f"{sync_env_module_name!r} "
            f"(registered_env_count_before_import={len(before_ids)}): "
            f"{type(e).__name__}: {e}"
        ) from e

    after_ids = state_conditioned_env_resolution.registered_g1_env_ids(gym_module)
    module_file = str(getattr(sync_env_mod, "__file__", "<unknown>"))
    print("[INFO] explicit_sync_env_import:", module_file)
    print("[INFO] registered_g1_env_count_before_resolution:", len(after_ids))
    if after_ids:
        print(
            "[INFO] registered_g1_env_ids_sample:", after_ids[: min(5, len(after_ids))]
        )
    else:
        raise RuntimeError(
            "explicit G1 env registration imported "
            f"{sync_env_module_name!r} from {module_file!r} but left "
            "registered_env_count=0 for prefix "
            f"{state_conditioned_env_resolution.ENV_REGISTRY_PREFIX!r}"
        )
    return {
        "sync_env_module": sync_env_module_name,
        "sync_env_module_file": module_file,
        "registered_env_count_before_import": int(len(before_ids)),
        "registered_env_count_before_resolution": int(len(after_ids)),
        "registered_env_ids_sample": after_ids[: min(5, len(after_ids))],
    }


def main() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        try:
            _ = _build_parser().parse_args()
        except SystemExit as e:
            return int(getattr(e, "code", 0) or 0)
        return 0

    args = _build_parser().parse_args()
    _validate_args(args)
    canonical_preflight = _require_canonical_preflight_report(
        str(args.canonical_preflight_report)
    )
    if bool(args.preflight_only):
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "canonical_preflight_report": str(args.canonical_preflight_report),
                    "canonical_preflight_reason_code": canonical_preflight.get("reason_code"),
                },
                sort_keys=True,
            )
        )
        return 0

    repo_root = _repo_root()
    _add_import_roots(repo_root)
    _install_robocasa_import_shims()
    runtime_log_dir = _resolve_path(repo_root, str(args.runtime_log_dir))
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    telemetry_dir = _resolve_path(repo_root, str(args.telemetry_dir))
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    log_path = runtime_log_dir / f"3D_recap_eval_{ts}.log"
    summary_path = (
        _resolve_path(repo_root, str(args.summary_json))
        if str(args.summary_json).strip()
        else (artifact_dir / f"recap_3D_eval_summary_{ts}.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_enabled = bool(args.save_telemetry)
    step_telemetry_path = telemetry_dir / f"{summary_path.stem}_steps.jsonl"
    episode_telemetry_path = telemetry_dir / f"{summary_path.stem}_episodes.jsonl"
    if telemetry_enabled:
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        step_telemetry_path.unlink(missing_ok=True)
        episode_telemetry_path.unlink(missing_ok=True)

    with _tee_stdio(log_path):
        _install_alarm_timeout(float(args.total_timeout_s))
        env: Any = None

        def _jsonable(x: Any) -> Any:
            try:
                json.dumps(x, ensure_ascii=True)
                return x
            except Exception:
                return {"type": type(x).__name__, "repr": repr(x)}

        success_count = 0
        episodes_completed = 0
        ping_payload: Any = None
        server_info_payload: Any = None
        server_provenance_payload: Any = None
        n_action_steps: int | None = None
        n_action_steps_source: str | None = None
        action_horizon: int | None = None
        error_info: dict[str, Any] | None = None
        env_resolution: dict[str, Any] | None = None
        env_registration_info: dict[str, Any] | None = None
        telemetry_step_count = 0
        telemetry_episode_count = 0
        episode_results: list[dict[str, object]] = []

        try:
            print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
            print("[INFO] repo_root:", repo_root)
            print("[INFO] env_name:", str(args.env_name))
            print("[INFO] host:", str(args.host), "port:", int(args.port))
            print("[INFO] advantage:", args.advantage)
            print("[INFO] n_episodes:", int(args.n_episodes))
            print("[INFO] max_episode_steps:", int(args.max_episode_steps))
            print("[INFO] runtime_log:", log_path)
            print(
                "[INFO] canonical_preflight_report:",
                str(args.canonical_preflight_report),
                "reason_code:",
                canonical_preflight.get("reason_code"),
            )
            print("[INFO] save_telemetry:", bool(telemetry_enabled))
            if telemetry_enabled:
                print("[INFO] telemetry_dir:", telemetry_dir)
                print("[INFO] step_telemetry_jsonl:", step_telemetry_path)
                print("[INFO] episode_telemetry_jsonl:", episode_telemetry_path)

            gym = importlib.import_module("gymnasium")
            rollout_mod = importlib.import_module("gr00t.eval.rollout_policy")
            sc_mod = importlib.import_module("gr00t.policy.server_client")
            env_registration_info = _ensure_explicit_g1_env_registration(gym)

            WrapperConfigs = getattr(rollout_mod, "WrapperConfigs")
            VideoConfig = getattr(rollout_mod, "VideoConfig")
            MultiStepConfig = getattr(rollout_mod, "MultiStepConfig")
            create_eval_env = getattr(rollout_mod, "create_eval_env")
            PolicyClient = getattr(sc_mod, "PolicyClient")

            client = PolicyClient(
                host=str(args.host), port=int(args.port), strict=False
            )

            def _configure_client_socket(timeout_ms: int) -> None:
                try:
                    zmq = importlib.import_module("zmq")
                    client.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
                    client.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
                    client.socket.setsockopt(zmq.LINGER, 0)
                except Exception:
                    return

            _configure_client_socket(timeout_ms=1000)

            t0 = time.monotonic()
            modality_cfg: Any = None
            last_err: str | None = None
            while True:
                try:
                    ping_payload = client.call_endpoint("ping", requires_input=False)
                    modality_cfg = client.get_modality_config()
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if time.monotonic() - t0 > float(args.connect_timeout_s):
                        raise TimeoutError(
                            "Timed out waiting for server ping after "
                            f"{args.connect_timeout_s}s (last_error={last_err})"
                        )
                    try:
                        client.socket.close(0)
                    except Exception:
                        pass
                    try:
                        client = PolicyClient(
                            host=str(args.host), port=int(args.port), strict=False
                        )
                        _configure_client_socket(timeout_ms=1000)
                    except Exception:
                        pass
                    time.sleep(1.0)

            _configure_client_socket(
                timeout_ms=int(float(args.total_timeout_s) * 1000.0)
            )

            try:
                server_info_payload = client.call_endpoint(
                    "get_server_info", requires_input=False
                )
            except Exception as e:
                server_info_payload = {"error": f"{type(e).__name__}: {e}"}
            try:
                server_provenance_payload = client.call_endpoint(
                    "get_provenance", requires_input=False
                )
            except Exception as e:
                server_provenance_payload = {"error": f"{type(e).__name__}: {e}"}

            print("[INFO] policy server ready")
            if ping_payload is not None:
                print("[INFO] server_ping:", _jsonable(ping_payload))
            if server_info_payload is not None:
                print("[INFO] server_info:", _jsonable(server_info_payload))
            if server_provenance_payload is not None:
                print("[INFO] server_provenance:", _jsonable(server_provenance_payload))

            runtime_provenance: dict[str, Any] | None = None
            if (
                isinstance(server_provenance_payload, dict)
                and "error" not in server_provenance_payload
            ):
                runtime_provenance = server_provenance_payload
            elif isinstance(server_info_payload, dict):
                info_provenance = server_info_payload.get("provenance")
                if isinstance(info_provenance, dict):
                    runtime_provenance = info_provenance
            purity_mode = _require_mainline_server_purity(runtime_provenance)
            print("[INFO] purity_gate_accepted:", purity_mode)
            if purity_mode == "action_head_overlay" and isinstance(
                runtime_provenance, dict
            ):
                print(
                    "[INFO] purity_gate_overlay_lane: action_head_only",
                    f"overlay_from={runtime_provenance.get('overlay_from')!r}",
                    f"overlay_include_regex={runtime_provenance.get('overlay_include_regex')!r}",
                )

            action_horizon = None
            if isinstance(modality_cfg, dict) and "action" in modality_cfg:
                delta = list(getattr(modality_cfg["action"], "delta_indices", []) or [])
                action_horizon = int(len(delta)) if len(delta) > 0 else None
            print("[INFO] server_action_horizon:", action_horizon)

            env_resolution = (
                state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
                    gym,
                    requested_env_name=str(args.env_name),
                )
            )
            resolved_env_name = str(env_resolution["resolved_env_name"])
            env_resolution_summary = {
                "logical_task": env_resolution["logical_task"],
                "requested_env_name": env_resolution["requested_env_name"],
                "resolved_env_name": env_resolution["resolved_env_name"],
                "alias_applied": env_resolution["alias_applied"],
                "available_close_matches": env_resolution["available_close_matches"],
            }
            print(
                "[INFO] env_resolution:",
                _jsonable(env_resolution_summary),
            )

            n_action_steps = int(args.n_action_steps)
            if n_action_steps <= 0:
                official_default = _default_n_action_steps_for_env(resolved_env_name)
                if official_default is not None:
                    n_action_steps = int(official_default)
                    n_action_steps_source = "g1_execution_surface_default"
                    print(
                        "[INFO] overriding n_action_steps from official env default:",
                        int(n_action_steps),
                    )
                else:
                    if action_horizon is not None:
                        n_action_steps = int(action_horizon)
                        n_action_steps_source = "server_action_horizon_fallback"
                    else:
                        n_action_steps = 20
                        n_action_steps_source = "generic_default_20"
            else:
                n_action_steps_source = "cli_override"
            print("[INFO] n_action_steps:", n_action_steps)
            print("[INFO] n_action_steps_source:", n_action_steps_source)

            wrapper_configs = WrapperConfigs(
                video=VideoConfig(
                    video_dir=None,
                    max_episode_steps=int(args.max_episode_steps),
                    overlay_text=False,
                ),
                multistep=MultiStepConfig(
                    n_action_steps=int(n_action_steps),
                    max_episode_steps=int(args.max_episode_steps),
                    terminate_on_success=True,
                ),
            )

            def env_fn() -> Any:
                return create_eval_env(
                    env_name=resolved_env_name,
                    env_idx=0,
                    total_n_envs=1,
                    wrapper_configs=wrapper_configs,
                )

            env = gym.vector.SyncVectorEnv([env_fn])

            base_options: dict[str, Any] = {}
            if args.advantage is not None:
                base_options["advantage"] = float(args.advantage)
            outer_max_steps = max(
                1,
                (int(args.max_episode_steps) + int(n_action_steps) - 1)
                // int(n_action_steps),
            )
            print("[INFO] policy_options_base:", base_options)
            print(
                "[INFO] policy_options_note: per-episode options include seed=seed_base+episode_index"
            )
            print("[INFO] outer_max_steps_per_episode:", outer_max_steps)

            for ep_idx in range(int(args.n_episodes)):
                seed_i = int(args.seed_base) + int(ep_idx)
                episode_started_wall = _dt.datetime.now().isoformat(timespec="seconds")
                print(
                    "[EPISODE_START]",
                    f"index={ep_idx + 1}/{int(args.n_episodes)}",
                    f"seed={seed_i}",
                    f"started_at={episode_started_wall}",
                )
                obs, _info = env.reset(seed=seed_i)
                options_ep: dict[str, Any] = dict(base_options)
                options_ep["seed"] = int(seed_i)
                client.reset(options=options_ep)

                if bool(args.debug_success) and ep_idx == 0 and isinstance(obs, dict):
                    keys = sorted(list(obs.keys()))
                    print("[DEBUG] reset_obs_key_count:", len(keys))
                    print("[DEBUG] reset_obs_keys_head:", keys[: min(40, len(keys))])

                    try:
                        asp = getattr(env, "single_action_space", None)
                        spaces = (
                            getattr(asp, "spaces", None) if asp is not None else None
                        )
                        if isinstance(spaces, dict):
                            print(
                                "[DEBUG] env_action_keys:",
                                sorted(list(spaces.keys()))[: min(40, len(spaces))],
                            )
                    except Exception:
                        pass

                    anno_keys = [k for k in keys if "annotation" in k or k == "task"]
                    if anno_keys:
                        print("[DEBUG] reset_obs_annotation_keys:", anno_keys)

                    td_key = "annotation.human.task_description"
                    if td_key in obs:
                        td_val = obs.get(td_key)
                        sample = None
                        try:
                            sample = (
                                td_val[0]
                                if isinstance(td_val, (list, tuple))
                                else td_val
                            )
                        except Exception:
                            sample = None
                        print(
                            "[DEBUG] task_description_type:",
                            type(td_val).__name__,
                            "sample=",
                            sample,
                        )

                done = False
                episode_success = False
                outer_steps = 0
                last_step_info: Any = None
                last_terminated = False
                last_truncated = False
                last_reward = 0.0
                episode_started = time.monotonic()
                reset_snapshot = _collect_env_snapshot(env)
                episode_step_records: list[dict[str, Any]] = []
                while (not done) and outer_steps < int(outer_max_steps):
                    action, _action_info = client.get_action(obs, options=options_ep)
                    if not isinstance(action, dict):
                        raise TypeError(
                            "PolicyClient.get_action must return a dict action, got "
                            f"{type(action).__name__}"
                        )

                    if bool(args.debug_success) and outer_steps == 0:
                        print(
                            "[DEBUG] action_keys_head:",
                            sorted(list(action.keys()))[: min(40, len(action))],
                        )

                    obs, reward, term, trunc, step_info = env.step(action)
                    last_step_info = step_info
                    reward_scalar = _scalarize_float(reward)
                    last_reward = float(reward_scalar)
                    last_terminated = bool(_scalarize_bool(term))
                    last_truncated = bool(_scalarize_bool(trunc))
                    done = bool(last_terminated or last_truncated)
                    success_step = _extract_success_step(step_info)
                    episode_success = bool(episode_success or success_step)
                    outer_steps += 1

                    step_snapshot = _collect_env_snapshot(env)
                    step_record: dict[str, Any] = {
                        "episode_index": int(ep_idx + 1),
                        "seed": int(seed_i),
                        "advantage": args.advantage,
                        "advantage_mode": _advantage_mode(args.advantage),
                        "outer_step": int(outer_steps),
                        "reward": float(reward_scalar),
                        "terminated": bool(last_terminated),
                        "truncated": bool(last_truncated),
                        "done": bool(done),
                        "success_step": bool(success_step),
                        "episode_success_so_far": bool(episode_success),
                        "action_summary": _summarize_action_chunk(action),
                    }
                    step_record.update(step_snapshot)
                    intermediate_signals = _extract_intermediate_signals(step_info)
                    if intermediate_signals is not None:
                        step_record["intermediate_signals"] = intermediate_signals
                    episode_step_records.append(step_record)
                    if telemetry_enabled:
                        _append_jsonl(step_telemetry_path, step_record)
                        telemetry_step_count += 1

                success_count += 1 if episode_success else 0

                final_snapshot = (
                    {
                        k: episode_step_records[-1].get(k)
                        for k in (
                            "sim_time_s",
                            "apple_pos_xyz",
                            "plate_pos_xyz",
                            "right_eef_pos_xyz",
                            "left_eef_pos_xyz",
                            "apple_to_right_eef_l2",
                            "apple_to_left_eef_l2",
                            "apple_to_plate_l2",
                            "right_eef_to_plate_l2",
                            "apple_height_z",
                            "plate_height_z",
                            "right_eef_height_z",
                        )
                    }
                    if episode_step_records
                    else reset_snapshot
                )
                failure_reason = _episode_failure_reason(
                    success=bool(episode_success),
                    done=bool(done),
                    terminated=bool(last_terminated),
                    truncated=bool(last_truncated),
                    outer_steps=int(outer_steps),
                    outer_max_steps=int(outer_max_steps),
                )
                failure_stage_guess = (
                    None
                    if episode_success
                    else _failure_stage_guess(episode_step_records)
                )
                episode_elapsed_seconds = float(time.monotonic() - episode_started)
                episode_finished_wall = _dt.datetime.now().isoformat(timespec="seconds")
                episode_record: dict[str, Any] = {
                    "episode_index": int(ep_idx + 1),
                    "seed": int(seed_i),
                    "advantage": args.advantage,
                    "advantage_mode": _advantage_mode(args.advantage),
                    "success": bool(episode_success),
                    "done": bool(done),
                    "terminated": bool(last_terminated),
                    "truncated": bool(last_truncated),
                    "outer_steps": int(outer_steps),
                    "episode_elapsed_seconds": episode_elapsed_seconds,
                    "final_reward": float(last_reward),
                    "failure_reason": failure_reason,
                    "failure_stage_guess": failure_stage_guess,
                    "reset_snapshot": reset_snapshot,
                    "final_snapshot": final_snapshot,
                    "n_success_steps": int(
                        sum(
                            1
                            for rec in episode_step_records
                            if bool(rec.get("success_step"))
                        )
                    ),
                    "step_telemetry_records": int(len(episode_step_records)),
                }
                if telemetry_enabled:
                    _append_jsonl(episode_telemetry_path, episode_record)
                    telemetry_episode_count += 1
                episode_results.append(_build_public_episode_result(episode_record))

                if bool(args.debug_success) and isinstance(last_step_info, dict):
                    present = [k for k in SUCCESS_KEYS if k in last_step_info]
                    print("[DEBUG] success_keys_present:", present)
                    try:
                        import numpy as np

                        for k in present:
                            v = np.asarray(last_step_info.get(k))
                            flat = v.reshape(-1)
                            preview = (
                                flat[: min(10, int(flat.size))].tolist()
                                if flat.size > 0
                                else []
                            )
                            any_true = bool(np.any(v)) if v.size > 0 else False
                            print(
                                f"[DEBUG] {k}:",
                                f"any={any_true}",
                                f"dtype={v.dtype}",
                                f"shape={tuple(v.shape)}",
                                "preview=",
                                preview,
                            )
                    except Exception as e:
                        print(
                            "[DEBUG] failed to print success previews:",
                            type(e).__name__,
                            str(e),
                        )
                print(
                    "[EPISODE_END]",
                    f"index={ep_idx + 1}/{int(args.n_episodes)}",
                    f"seed={seed_i}",
                    f"ended_at={episode_finished_wall}",
                    f"elapsed_s={episode_elapsed_seconds:.6f}",
                    f"done={bool(done)}",
                    f"outer_steps={int(outer_steps)}",
                    f"success={bool(episode_success)}",
                    f"failure_reason={failure_reason or 'none'}",
                )
                episodes_completed += 1

        except Exception as e:
            if isinstance(
                e, state_conditioned_env_resolution.StateConditionedEnvResolutionError
            ):
                error_info = {
                    **e.to_machine_payload(),
                    "type": type(e).__name__,
                    "message": str(e),
                }
            else:
                error_info = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": "".join(traceback.format_exception(e)),
                }
            print("[ERROR] eval_failed:", type(e).__name__, str(e))
        finally:
            _clear_alarm_timeout()
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        requested_episodes = int(args.n_episodes)
        effective_episodes = episodes_completed or 0
        if effective_episodes > 0:
            success_rate = float(success_count) / float(effective_episodes)
        else:
            success_rate = 0.0

        summary: dict[str, Any] = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "advantage": args.advantage,
            "advantage_mode": _advantage_mode(args.advantage),
            "advantage_contract_version": str(ADVANTAGE_CONTRACT_VERSION),
            "task_text_field_expected": str(MAINLINE_TASK_TEXT_FIELD),
            "advantage_none_semantics": "unconditional_baseline",
            "advantage_zero_semantics": "explicit_numeric_neutral_conditioning",
            "advantage_positive_semantics": "positive_numeric_conditioning",
            "episodes": int(effective_episodes),
            "requested_episodes": requested_episodes,
            "success_count": int(success_count),
            "success_rate": float(success_rate),
            "log_path": str(log_path),
            "env_name": str(args.env_name),
            "host": str(args.host),
            "port": int(args.port),
            "max_episode_steps": int(args.max_episode_steps),
            "server_action_horizon": int(action_horizon)
            if action_horizon is not None
            else None,
            "n_action_steps": int(n_action_steps)
            if n_action_steps is not None
            else None,
            "n_action_steps_source": n_action_steps_source,
            "seed_base": int(args.seed_base),
            "canonical_preflight_report": str(args.canonical_preflight_report),
            "canonical_preflight_reason_code": canonical_preflight.get("reason_code"),
            "episode_results": episode_results,
            "telemetry_enabled": bool(telemetry_enabled),
            "step_telemetry_jsonl": str(step_telemetry_path)
            if telemetry_enabled
            else None,
            "episode_telemetry_jsonl": str(episode_telemetry_path)
            if telemetry_enabled
            else None,
            "telemetry_step_count": int(telemetry_step_count),
            "telemetry_episode_count": int(telemetry_episode_count),
            "env_registration": _jsonable(env_registration_info),
            "server_ping": _jsonable(ping_payload),
            "server_info": _jsonable(server_info_payload),
            "server_provenance": _jsonable(server_provenance_payload),
            "execution_surface_contract": {
                "policy_horizon_expected": int(action_horizon)
                if action_horizon is not None
                else None,
                "n_action_steps": int(n_action_steps)
                if n_action_steps is not None
                else None,
                "g1_default_execution_steps": int(DEFAULT_G1_EXECUTION_N_ACTION_STEPS),
                "must_not_conflate_horizon_and_execution": True,
            },
        }
        summary.update(
            build_diagnostic_surface_metadata(
                surface_route=NUMERIC_ADVANTAGE_EVAL_DIAGNOSTIC_ROUTE,
                authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
                compatibility_fields=GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS,
                surface_kind="numeric_advantage_eval_summary",
            )
        )
        if isinstance(server_info_payload, dict):
            summary["server_uuid"] = server_info_payload.get("server_uuid")
        if isinstance(server_provenance_payload, dict):
            summary["task_text_field"] = server_provenance_payload.get(
                "task_text_field"
            )
            summary["advantage_injection_rule"] = server_provenance_payload.get(
                "advantage_injection_rule"
            )
        if env_resolution is not None:
            summary["env_resolution"] = {
                "logical_task": env_resolution["logical_task"],
                "requested_env_name": env_resolution["requested_env_name"],
                "resolved_env_name": env_resolution["resolved_env_name"],
                "alias_applied": env_resolution["alias_applied"],
                "available_close_matches": env_resolution["available_close_matches"],
            }
        if error_info is not None:
            summary["error"] = error_info

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")

        if error_info is None:
            print(
                "[INFO] eval_done",
                f"success_count={int(success_count)}",
                f"episodes={int(effective_episodes)}",
                f"success_rate={float(success_rate):.6f}",
            )
            print("[INFO] summary_json:", summary_path)
            return 0

        print(
            "[INFO] eval_failed_summary_written",
            f"episodes_completed={int(effective_episodes)}",
            f"success_count={int(success_count)}",
            "exit_code=1",
        )
        print("[INFO] summary_json:", summary_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
