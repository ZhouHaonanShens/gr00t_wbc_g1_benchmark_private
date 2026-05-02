from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EPISODE_REQUIRED_KEYS = (
    "schema_version",
    "code_version",
    "iter_tag",
    "episode_id",
    "env_name",
    "model_path",
    "embodiment_tag",
    "server_host",
    "server_port",
    "seed",
    "gamma",
    "C_fail",
    "prompt_raw",
    "prompt_conditioned",
    "n_action_steps_config",
    "terminated",
    "truncated",
    "done",
    "success_episode",
    "episode_return_online",
    "episode_return_wrapper",
    "n_policy_steps",
    "video_dir_tmp",
    "video_dir_archived",
    "arrays_saved",
    "npz_path",
)


TRANSITION_REQUIRED_KEYS = (
    "schema_version",
    "code_version",
    "iter_tag",
    "episode_id",
    "t",
    "T_action",
    "n_action_steps_config",
    "n_action_steps_executed",
    "reward_online",
    "reward_wrapper",
    "terminated",
    "truncated",
    "done",
    "obs_keys",
    "obs_summary",
    "action_keys",
    "action_summary",
    "inner_rewards",
    "inner_dones",
    "success_step",
)


LOCAL_RECOVERY_PSEUDODEMO_REQUIRED_KEYS = (
    "episode_id",
    "producer",
    "teacher_version",
    "teacher_trigger_reason",
    "teacher_trigger_success_rate",
    "teacher_trigger_threshold",
    "source_snapshot_id",
    "source_snapshot_family",
    "source_snapshot_history_k",
    "failure_prefix_step_count",
    "failure_prefix_source_episode_id",
    "failure_prefix_source_t_range",
    "recovery_suffix_step_count",
    "recovery_suffix_source_episode_id",
    "recovery_suffix_source_t_range",
)

LOCAL_RECOVERY_PSEUDODEMO_PRODUCER_VALUES = (
    "base_policy",
    "scripted_teacher",
)


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _require_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _require_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _normalize_t_range(value: Any, *, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"{field_name} must have length 2, got {len(value)}")
    start_t = _require_int(value[0], field_name=f"{field_name}[0]")
    end_t = _require_int(value[1], field_name=f"{field_name}[1]")
    if end_t < start_t:
        raise ValueError(f"{field_name} must be [start_t, end_t] with end_t >= start_t")
    return [int(start_t), int(end_t)]


def validate_local_recovery_pseudodemo_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    _require_keys(
        record,
        LOCAL_RECOVERY_PSEUDODEMO_REQUIRED_KEYS,
        kind="local_recovery_pseudodemo",
    )
    normalized = dict(record)
    normalized["episode_id"] = _require_non_empty_string(
        record.get("episode_id"),
        field_name="episode_id",
    )
    normalized["producer"] = _require_non_empty_string(
        record.get("producer"),
        field_name="producer",
    )
    if normalized["producer"] not in LOCAL_RECOVERY_PSEUDODEMO_PRODUCER_VALUES:
        raise ValueError(
            "producer must be one of "
            + f"{LOCAL_RECOVERY_PSEUDODEMO_PRODUCER_VALUES!r}, got {normalized['producer']!r}"
        )
    normalized["teacher_version"] = _require_non_empty_string(
        record.get("teacher_version"),
        field_name="teacher_version",
    )
    normalized["teacher_trigger_reason"] = _require_non_empty_string(
        record.get("teacher_trigger_reason"),
        field_name="teacher_trigger_reason",
    )
    normalized["teacher_trigger_success_rate"] = _require_number(
        record.get("teacher_trigger_success_rate"),
        field_name="teacher_trigger_success_rate",
    )
    normalized["teacher_trigger_threshold"] = _require_number(
        record.get("teacher_trigger_threshold"),
        field_name="teacher_trigger_threshold",
    )
    normalized["source_snapshot_id"] = _require_non_empty_string(
        record.get("source_snapshot_id"),
        field_name="source_snapshot_id",
    )
    normalized["source_snapshot_family"] = _require_non_empty_string(
        record.get("source_snapshot_family"),
        field_name="source_snapshot_family",
    )
    normalized["source_snapshot_history_k"] = _require_int(
        record.get("source_snapshot_history_k"),
        field_name="source_snapshot_history_k",
    )
    normalized["failure_prefix_step_count"] = _require_int(
        record.get("failure_prefix_step_count"),
        field_name="failure_prefix_step_count",
    )
    if normalized["failure_prefix_step_count"] <= 0:
        raise ValueError("failure_prefix_step_count must be > 0")
    normalized["failure_prefix_source_episode_id"] = _require_non_empty_string(
        record.get("failure_prefix_source_episode_id"),
        field_name="failure_prefix_source_episode_id",
    )
    normalized["failure_prefix_source_t_range"] = _normalize_t_range(
        record.get("failure_prefix_source_t_range"),
        field_name="failure_prefix_source_t_range",
    )
    normalized["recovery_suffix_step_count"] = _require_int(
        record.get("recovery_suffix_step_count"),
        field_name="recovery_suffix_step_count",
    )
    if normalized["recovery_suffix_step_count"] <= 0:
        raise ValueError("recovery_suffix_step_count must be > 0")
    normalized["recovery_suffix_source_episode_id"] = _require_non_empty_string(
        record.get("recovery_suffix_source_episode_id"),
        field_name="recovery_suffix_source_episode_id",
    )
    normalized["recovery_suffix_source_t_range"] = _normalize_t_range(
        record.get("recovery_suffix_source_t_range"),
        field_name="recovery_suffix_source_t_range",
    )
    return normalized


def summarize_value(v: Any) -> str:
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return f"ndarray dtype={v.dtype} shape={list(v.shape)}"
    except Exception:
        pass

    return type(v).__name__


def to_jsonable_list(x: Any) -> list[Any] | None:
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "tolist"):
        try:
            y = x.tolist()
            return list(y) if isinstance(y, (list, tuple)) else [y]
        except Exception:
            pass
    return [x]


def _is_video_key(k: str) -> bool:
    return k.startswith("video.") or k.startswith("video/")


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if s.startswith(prefix) else s


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_jsonl_appendable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= 0:
        return

    try:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with path.open("ab") as f:
                f.write(b"\n")
                f.flush()
    except OSError:
        return


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _append_jsonl_line(path: Path, record: Mapping[str, Any]) -> None:
    _ensure_jsonl_appendable(path)
    line = json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        default=_json_default,
    )
    if "\n" in line:
        line = line.replace("\n", "\\n")

    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def _require_keys(
    record: Mapping[str, Any], required: tuple[str, ...], *, kind: str
) -> None:
    missing = [k for k in required if k not in record]
    if missing:
        raise KeyError(f"{kind} record missing required keys: {missing}")


@dataclass(frozen=True)
class EpisodeWriter:
    iter_tag: str
    schema_version: str = "recap-v0"
    code_version: str = "unknown"
    arrays_saved: bool = True
    dataset_root: Path | None = None

    def __post_init__(self) -> None:
        base = self.dataset_root
        if base is None:
            base = _repo_root_from_here() / "agent" / "artifacts" / "recap_datasets"

        object.__setattr__(self, "dataset_root", Path(base))

        self.iter_dir.mkdir(parents=True, exist_ok=True)
        if self.arrays_saved:
            (self.iter_dir / "arrays").mkdir(parents=True, exist_ok=True)

    @property
    def iter_dir(self) -> Path:
        assert self.dataset_root is not None
        return self.dataset_root / self.iter_tag

    @property
    def episodes_path(self) -> Path:
        return self.iter_dir / "episodes.jsonl"

    @property
    def transitions_path(self) -> Path:
        return self.iter_dir / "transitions.jsonl"

    def append_episode(self, record: Mapping[str, Any]) -> None:
        _require_keys(record, EPISODE_REQUIRED_KEYS, kind="episode")
        _append_jsonl_line(self.episodes_path, record)

    def append_transition(self, record: Mapping[str, Any]) -> None:
        _require_keys(record, TRANSITION_REQUIRED_KEYS, kind="transition")
        _append_jsonl_line(self.transitions_path, record)

    def write_episode_npz(
        self,
        episode_id: str,
        *,
        state_arrays: Mapping[str, Any],
        action_arrays: Mapping[str, Any],
    ) -> str | None:
        if not self.arrays_saved:
            return None

        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError("numpy is required for NPZ writing") from e

        arrays_dir = self.iter_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        rel = Path("arrays") / f"{episode_id}.npz"
        out_path = self.iter_dir / rel

        payload: dict[str, Any] = {}
        for k, v in state_arrays.items():
            ks = str(k)
            if not ks.startswith("state."):
                continue
            inner = _strip_prefix(ks, "state.")
            if _is_video_key(inner):
                continue
            payload[f"state/{inner}"] = np.asarray(v)
        for k, v in action_arrays.items():
            ks = str(k)
            if not ks.startswith("action."):
                continue
            inner = _strip_prefix(ks, "action.")
            if _is_video_key(inner):
                continue
            payload[f"action/{inner}"] = np.asarray(v)

        if not payload:
            raise ValueError("Refusing to write empty NPZ payload")

        np.savez_compressed(out_path, **payload)
        return rel.as_posix()
