from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


from . import policy as recap_policy


def normalize_prompt(obs: Mapping[str, Any]) -> str:
    raw = obs.get("annotation.human.task_description", "")
    try:
        import numpy as np

        if isinstance(raw, (list, tuple)):
            item = raw[0] if len(raw) > 0 else ""
            return str(item)

        if isinstance(raw, np.ndarray):
            if raw.size <= 0:
                return ""
            item = raw.reshape(-1)[0]
            return str(item)
    except Exception:
        pass

    return str(raw)


def extract_T_action(action_chunk: Mapping[str, Any]) -> int:
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy is required for action shape extraction") from e

    t_action: int | None = None
    for k, v in action_chunk.items():
        arr = np.asarray(v)
        if arr.ndim < 2:
            raise ValueError(
                f"action value for key {k!r} must have ndim>=2 to read shape[-2], got shape={arr.shape}"
            )
        cur = int(arr.shape[-2])
        if t_action is None:
            t_action = cur
        elif cur != t_action:
            raise ValueError(
                f"Inconsistent T_action across action keys: expected {t_action}, key {k!r} has {cur}"
            )

    if t_action is None:
        raise ValueError("Empty action_chunk; cannot extract T_action")
    return t_action


def _scalarize_reward(reward: Any) -> float:
    import numpy as np

    return float(np.asarray(reward).reshape(-1)[0])


def _scalarize_bool(x: Any) -> bool:
    import numpy as np

    return bool(np.asarray(x).reshape(-1)[0])


def _squeeze_env_seq(x: Any) -> Any:
    import numpy as np

    cur: Any = x
    for _ in range(32):
        if isinstance(cur, (list, tuple)) and len(cur) == 1:
            cur = cur[0]
            continue

        arr = np.asarray(cur)

        if arr.dtype == object and arr.size == 1:
            cur = arr.reshape(-1)[0]
            continue

        if arr.ndim >= 2 and arr.shape[0] == 1:
            cur = arr[0]
            continue

        return arr

    return np.asarray(cur)


def _as_list_float(x: Any) -> list[float]:
    import numpy as np

    arr = np.asarray(x).reshape(-1)
    return [float(np.asarray(v).reshape(-1)[0]) for v in arr]


def _as_list_bool(x: Any) -> list[bool]:
    import numpy as np

    arr = np.asarray(x).reshape(-1)
    return [bool(np.asarray(v).reshape(-1)[0]) for v in arr]


def _reduce_success_value(x: Any) -> bool:
    import numpy as np

    v = x
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) > 0 else False
    elif isinstance(v, np.ndarray):
        if v.size <= 0:
            return False
        v = v[0]

    if isinstance(v, list):
        return bool(any(bool(e) for e in v))
    if isinstance(v, np.ndarray):
        return bool(np.any(v))
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)

    try:
        return bool(v)
    except Exception:
        return False


def _extract_explicit_success_step(info: Any) -> bool | None:
    if not isinstance(info, dict):
        return None

    if "success" in info:
        return _reduce_success_value(info.get("success"))

    final_info = info.get("final_info")
    if final_info is None:
        return None

    try:
        import numpy as np

        if isinstance(final_info, np.ndarray):
            if final_info.size <= 0:
                return None
            final0 = final_info.reshape(-1)[0]
        elif isinstance(final_info, (list, tuple)):
            final0 = final_info[0] if len(final_info) > 0 else None
        else:
            final0 = final_info
    except Exception:
        if isinstance(final_info, (list, tuple)):
            final0 = final_info[0] if len(final_info) > 0 else None
        else:
            final0 = final_info

    if final0 is None:
        return None

    if hasattr(final0, "success"):
        return _reduce_success_value(getattr(final0, "success"))
    if isinstance(final0, dict) and "success" in final0:
        return _reduce_success_value(final0.get("success"))
    return None


def _reward_success_fallback(
    *, reward_wrapper: Any = None, inner_rewards: Any = None
) -> bool:
    if reward_wrapper is not None:
        try:
            if float(_scalarize_reward(reward_wrapper)) > 0.0:
                return True
        except Exception:
            pass

    if inner_rewards is None:
        return False

    try:
        return any(float(v) > 0.0 for v in _as_list_float(inner_rewards))
    except Exception:
        return False


def infer_success_step(
    info: Any,
    *,
    reward_wrapper: Any = None,
    inner_rewards: Any = None,
) -> bool:
    explicit = _extract_explicit_success_step(info)
    if explicit is not None:
        return bool(explicit)
    return _reward_success_fallback(
        reward_wrapper=reward_wrapper,
        inner_rewards=inner_rewards,
    )


def _extract_success_step(
    info: Any,
    *,
    reward_wrapper: Any = None,
    inner_rewards: Any = None,
) -> bool:
    return infer_success_step(
        info,
        reward_wrapper=reward_wrapper,
        inner_rewards=inner_rewards,
    )


def _read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{idx}, got {type(obj).__name__}"
                )
            records.append(dict(obj))
    return records


def recompute_success_episode_from_records(
    episode_record: Mapping[str, Any],
    transition_records: Sequence[Mapping[str, Any]],
) -> tuple[bool, str]:
    if bool(episode_record.get("success_episode", False)):
        return True, "episode_record"

    for tr in transition_records:
        if bool(tr.get("success_step", False)):
            return True, "transition_success_step"

    for tr in transition_records:
        if _reward_success_fallback(
            reward_wrapper=tr.get("reward_wrapper"),
            inner_rewards=tr.get("inner_rewards"),
        ):
            return True, "reward_fallback"

    if _reward_success_fallback(
        reward_wrapper=episode_record.get("episode_return_wrapper"),
    ):
        return True, "episode_reward_fallback"

    return False, "none"


def summarize_existing_dataset_success(
    episodes_jsonl: str | Path,
    transitions_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    episodes_path = Path(episodes_jsonl)
    transitions_path = None if transitions_jsonl is None else Path(transitions_jsonl)

    episodes = _read_jsonl_records(episodes_path)
    transitions = (
        _read_jsonl_records(transitions_path)
        if transitions_path is not None and transitions_path.is_file()
        else []
    )

    transitions_by_episode: dict[str, list[dict[str, Any]]] = {}
    for tr in transitions:
        episode_id = str(tr.get("episode_id", ""))
        if not episode_id:
            raise ValueError(
                f"Missing episode_id in transitions file: {transitions_path}"
            )
        transitions_by_episode.setdefault(episode_id, []).append(tr)

    recorded_success_count = 0
    recomputed_success_count = 0
    reward_fallback_episode_ids: list[str] = []
    repaired_episode_ids: list[str] = []

    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        if not episode_id:
            raise ValueError(f"Missing episode_id in episodes file: {episodes_path}")

        if bool(episode.get("success_episode", False)):
            recorded_success_count += 1

        recomputed_success, reason = recompute_success_episode_from_records(
            episode,
            transitions_by_episode.get(episode_id, []),
        )
        if recomputed_success:
            recomputed_success_count += 1
        if recomputed_success and not bool(episode.get("success_episode", False)):
            repaired_episode_ids.append(episode_id)
        if reason in {"reward_fallback", "episode_reward_fallback"}:
            reward_fallback_episode_ids.append(episode_id)

    episodes_count = len(episodes)
    success_rate = (
        float(recomputed_success_count) / float(episodes_count)
        if episodes_count > 0
        else 0.0
    )
    return {
        "episodes": int(episodes_count),
        "success_count": int(recomputed_success_count),
        "success_rate": float(success_rate),
        "success_count_recorded": int(recorded_success_count),
        "success_count_recomputed": int(recomputed_success_count),
        "repaired_episode_count": int(len(repaired_episode_ids)),
        "repaired_episode_ids": repaired_episode_ids,
        "reward_fallback_episode_count": int(len(reward_fallback_episode_ids)),
        "reward_fallback_episode_ids": reward_fallback_episode_ids,
        "episodes_jsonl": str(episodes_path),
        "transitions_jsonl": (
            str(transitions_path) if transitions_path is not None else None
        ),
        "success_inference": (
            "info.success -> final_info.success -> positive reward fallback"
        ),
    }
    return False


def _normalize_transition_records_for_episode(
    transition_records: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_record in enumerate(transition_records):
        record = dict(raw_record)
        record_episode_id = str(record.get("episode_id", "")).strip()
        if record_episode_id != episode_id:
            raise ValueError(
                f"transition_records[{index}] episode_id mismatch: {record_episode_id!r} != {episode_id!r}"
            )
        t_value = record.get("t")
        if isinstance(t_value, bool) or not isinstance(t_value, int):
            raise TypeError(f"transition_records[{index}].t must be an int")
        normalized.append(record)
    normalized.sort(key=lambda item: int(item["t"]))
    return normalized


def _recovery_start_index_from_episode_metadata(
    episode_record: Mapping[str, Any],
) -> int | None:
    metadata = episode_record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    analysis_only = metadata.get("analysis_only")
    if not isinstance(analysis_only, Mapping):
        return None
    raw_entry_step = analysis_only.get("recovery_entry_step")
    if raw_entry_step is None:
        return None
    if isinstance(raw_entry_step, bool) or not isinstance(raw_entry_step, int):
        raise TypeError(
            "episode.metadata.analysis_only.recovery_entry_step must be an int or null"
        )
    return int(raw_entry_step)


def _infer_recovery_start_index(
    *,
    episode_record: Mapping[str, Any],
    transition_records: Sequence[Mapping[str, Any]],
) -> tuple[int, str]:
    if len(transition_records) < 2:
        raise ValueError(
            "successful local rollout must include at least 2 steps to preserve failure prefix and recovery suffix"
        )

    explicit_entry_step = _recovery_start_index_from_episode_metadata(episode_record)
    if explicit_entry_step is not None:
        if explicit_entry_step <= 0:
            raise ValueError(
                "recovery_entry_step must be > 0 to preserve a non-empty failure prefix"
            )
        if explicit_entry_step >= len(transition_records):
            raise ValueError(
                "recovery_entry_step must be < transition_count to preserve a non-empty recovery suffix"
            )
        return int(
            explicit_entry_step
        ), "episode.metadata.analysis_only.recovery_entry_step"

    for index, record in enumerate(transition_records):
        if bool(record.get("success_step", False)):
            if index <= 0:
                raise ValueError(
                    "success_step at t=0 would drop the required failure prefix provenance"
                )
            return int(index), "transition.success_step"

    return int(len(transition_records) - 1), "tail_fallback_for_success_episode"


def summarize_local_recovery_rollout_for_pseudodemo(
    *,
    episode_record: Mapping[str, Any],
    transition_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    episode_id = str(episode_record.get("episode_id", "")).strip()
    if not episode_id:
        raise ValueError("episode_record is missing non-empty episode_id")
    normalized_transitions = _normalize_transition_records_for_episode(
        transition_records,
        episode_id=episode_id,
    )
    success_episode, success_reason = recompute_success_episode_from_records(
        episode_record,
        normalized_transitions,
    )
    summary: dict[str, Any] = {
        "episode_id": episode_id,
        "transition_count": int(len(normalized_transitions)),
        "success_episode": bool(success_episode),
        "success_inference_reason": str(success_reason),
        "included_in_pseudodemo_manifest": False,
    }
    if not success_episode:
        return summary

    recovery_start_index, split_source = _infer_recovery_start_index(
        episode_record=episode_record,
        transition_records=normalized_transitions,
    )
    failure_prefix = normalized_transitions[:recovery_start_index]
    recovery_suffix = normalized_transitions[recovery_start_index:]
    if not failure_prefix or not recovery_suffix:
        raise ValueError(
            "successful local rollout must preserve a non-empty failure prefix and recovery suffix"
        )
    summary.update(
        {
            "included_in_pseudodemo_manifest": True,
            "split_source": str(split_source),
            "failure_prefix_step_count": int(len(failure_prefix)),
            "failure_prefix_source_episode_id": episode_id,
            "failure_prefix_source_t_range": [
                int(failure_prefix[0]["t"]),
                int(failure_prefix[-1]["t"]),
            ],
            "recovery_suffix_step_count": int(len(recovery_suffix)),
            "recovery_suffix_source_episode_id": episode_id,
            "recovery_suffix_source_t_range": [
                int(recovery_suffix[0]["t"]),
                int(recovery_suffix[-1]["t"]),
            ],
        }
    )
    return summary


def _select_inner_info_dict(info: Any) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(info, dict):
        return None, False
    if "rewards" in info and "dones" in info:
        return info, False

    final_info = info.get("final_info")
    if final_info is None:
        return None, False
    try:
        import numpy as np

        fi = np.asarray(final_info)
        if fi.size <= 0:
            return None, False
        v0 = fi.reshape(-1)[0]
    except Exception:
        if isinstance(final_info, (list, tuple)) and len(final_info) > 0:
            v0 = final_info[0]
        else:
            return None, False

    if isinstance(v0, dict):
        return v0, True
    return None, False


def _summarize_mapping(d: Mapping[str, Any]) -> dict[str, str]:
    from work.recap.episode_writer import summarize_value

    return {str(k): summarize_value(v) for k, v in d.items()}


def _filter_prefix(d: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        ks = str(k)
        if not ks.startswith(prefix):
            continue
        inner = ks[len(prefix) :]
        if inner.startswith("video.") or inner.startswith("video/"):
            continue
        out[ks] = v
    return out


def _stack_per_key(per_step: list[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not per_step:
        return {}

    keys = sorted({str(k) for d in per_step for k in d.keys()})
    out: dict[str, Any] = {}
    for k in keys:
        seq = []
        for d in per_step:
            if k not in d:
                raise KeyError(f"Missing key {k!r} in per-step arrays")
            seq.append(np.asarray(d[k]))
        out[k] = np.stack(seq, axis=0)
    return out


def _dtype_shape(x: Any) -> str:
    try:
        import numpy as np

        arr = np.asarray(x)
        return f"dtype={arr.dtype} shape={tuple(arr.shape)}"
    except Exception:
        try:
            shape = getattr(x, "shape", None)
            dtype = getattr(x, "dtype", None)
            if shape is not None or dtype is not None:
                return f"dtype={dtype} shape={shape}"
        except Exception:
            pass

    return f"type={type(x).__name__}"


def _select_obs_debug_keys(obs: Mapping[str, Any]) -> list[str]:
    keys = sorted([str(k) for k in obs.keys()])
    selected: list[str] = []

    def _add(k: str) -> None:
        if k in obs and k not in selected:
            selected.append(k)

    _add("annotation.human.task_description")

    video_keys = [k for k in keys if k.startswith("video.") or k.startswith("video/")]
    for k in video_keys[:2]:
        _add(k)

    state_keys = [k for k in keys if k.startswith("state.")]
    for k in state_keys[:2]:
        _add(k)

    if len(selected) < 5:
        for k in keys:
            if k not in selected:
                selected.append(k)
            if len(selected) >= 5:
                break

    return selected


def collect_episode(
    *,
    env: Any,
    client: Any,
    iter_tag: str,
    episode_id: str,
    env_name: str,
    model_path: str,
    embodiment_tag: str,
    server_host: str,
    server_port: int,
    seed: int,
    max_policy_steps: int,
    code_version: str = "unknown",
    video_dir_tmp: str | None = None,
    video_dir_archived: str | None = None,
    arrays_saved: bool = True,
    policy_prompt_prefix: str = "",
    debug_print: Callable[[str], None] | None = None,
    debug_step0_full: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if max_policy_steps <= 0:
        raise ValueError(f"max_policy_steps must be > 0, got {max_policy_steps}")

    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy is required for collection") from e

    gamma = 1.0
    C_fail = int(max_policy_steps)

    try:
        obs, _reset_info = env.reset(seed=seed)
    except TypeError:
        obs, _reset_info = env.reset()

    if hasattr(client, "reset"):
        client.reset()

    prompt_raw = normalize_prompt(obs)

    prefix = str(policy_prompt_prefix or "")
    if prefix and not prefix.endswith(" "):
        prefix = prefix + " "
    if prefix:
        raise ValueError(
            "policy_prompt_prefix is forbidden for authoritative collection; runtime request text must remain canonical"
        )

    prompt_conditioned = (prefix + prompt_raw) if prefix else prompt_raw

    transitions: list[dict[str, Any]] = []
    success_episode = False
    episode_return_online = 0.0
    last_reward_wrapper = 0.0
    last_terminated = False
    last_truncated = False
    last_done = False

    n_action_steps_config: int | None = None

    per_step_state: list[Mapping[str, Any]] = []
    per_step_action: list[Mapping[str, Any]] = []

    def _dbg(line: str) -> None:
        if debug_print is None:
            return
        debug_print(line)

    for t in range(int(max_policy_steps)):
        if not isinstance(obs, Mapping):
            raise TypeError(
                f"env observation must be mapping-like before policy call, got {type(obs).__name__}"
            )
        obs_for_policy = recap_policy.filter_canonical_serving_observation(
            obs,
            field_name=f"obs[t={t}]",
        )

        action_result = client.get_action(obs_for_policy)
        if isinstance(action_result, tuple) and len(action_result) >= 1:
            action_chunk = action_result[0]
        else:
            action_chunk = action_result

        if not isinstance(action_chunk, dict):
            raise TypeError(
                f"client.get_action(obs) must return dict, got {type(action_chunk).__name__}"
            )

        T_action = extract_T_action(action_chunk)
        if n_action_steps_config is None:
            n_action_steps_config = int(T_action)
        elif int(T_action) != int(n_action_steps_config):
            raise ValueError(
                f"T_action changed across steps: expected {n_action_steps_config}, got {T_action} at t={t}"
            )

        next_obs, reward, terminated, truncated, info = env.step(action_chunk)

        reward_wrapper = _scalarize_reward(reward)
        terminated0 = _scalarize_bool(terminated)
        truncated0 = _scalarize_bool(truncated)
        done0 = bool(terminated0 or truncated0)

        inner_info, used_final_info = _select_inner_info_dict(info)
        if (
            inner_info is None
            or ("rewards" not in inner_info)
            or ("dones" not in inner_info)
        ):
            keys = (
                sorted([str(k) for k in info.keys()]) if isinstance(info, dict) else []
            )
            raise KeyError(
                "missing inner-step sequences in info: "
                + f"need keys ['rewards','dones']; got keys={keys}"
            )

        inner_rewards_raw = np.asarray(inner_info["rewards"])
        inner_dones_raw = np.asarray(inner_info["dones"])
        inner_rewards = _squeeze_env_seq(inner_rewards_raw)
        inner_dones = _squeeze_env_seq(inner_dones_raw)
        inner_rewards_list = _as_list_float(inner_rewards)
        inner_dones_list = _as_list_bool(inner_dones)

        n_action_steps_executed = int(len(inner_rewards_list))
        if n_action_steps_executed != int(len(inner_dones_list)):
            raise ValueError(
                "inner-step invariant violated: "
                f"len(inner_rewards)={len(inner_rewards_list)} != len(inner_dones)={len(inner_dones_list)}"
            )
        if n_action_steps_executed <= 0:
            raise ValueError("n_action_steps_executed must be > 0")
        if n_action_steps_executed > int(n_action_steps_config):
            raise ValueError(
                f"n_action_steps_executed={n_action_steps_executed} exceeds n_action_steps_config={n_action_steps_config}"
            )

        success_step = _extract_success_step(info)
        success_episode = bool(success_episode or success_step)

        if debug_print is not None:
            obs_keys_dbg = sorted([str(k) for k in obs.keys()])
            action_keys_dbg = sorted([str(k) for k in action_chunk.keys()])

            obs_debug_keys = _select_obs_debug_keys(obs)
            obs_debug_parts = [
                f"{k}({_dtype_shape(obs.get(k))})" for k in obs_debug_keys if k in obs
            ]
            action_parts = [
                f"{k}({_dtype_shape(action_chunk.get(k))})" for k in action_keys_dbg
            ]

            obs_debug_str = "; ".join(obs_debug_parts) if obs_debug_parts else "(none)"
            action_str = "; ".join(action_parts) if action_parts else "(none)"
            rewards_raw_str = _dtype_shape(inner_rewards_raw)
            dones_raw_str = _dtype_shape(inner_dones_raw)
            if used_final_info:
                _dbg(
                    "[recap.collect] note: used info['final_info'] for inner_rewards/inner_dones"
                )

            if int(t) == 0 and bool(debug_step0_full):
                _dbg(
                    f"[recap.collect] t={t} FULL episode_id={episode_id} iter_tag={iter_tag}"
                )
                _dbg(f"[recap.collect] obs_keys={obs_keys_dbg}")
                _dbg(f"[recap.collect] obs_key_samples={obs_debug_str}")
                _dbg(f"[recap.collect] action_keys={action_keys_dbg}")
                _dbg(f"[recap.collect] action_chunk={action_str}")
                _dbg(
                    "[recap.collect] "
                    + " ".join(
                        [
                            f"T_action={int(T_action)}",
                            f"n_action_steps_config={int(n_action_steps_config)}",
                            f"n_action_steps_executed={int(n_action_steps_executed)}",
                        ]
                    )
                )
                _dbg(
                    f"[recap.collect] info.rewards_raw({rewards_raw_str}) info.dones_raw({dones_raw_str})"
                )
                _dbg(
                    "[recap.collect] "
                    + " ".join(
                        [
                            f"len(inner_rewards)={len(inner_rewards_list)}",
                            f"len(inner_dones)={len(inner_dones_list)}",
                            f"terminated={bool(terminated0)}",
                            f"truncated={bool(truncated0)}",
                            f"done={bool(done0)}",
                            f"success_step={bool(success_step)}",
                        ]
                    )
                )
            else:
                _dbg(
                    "[recap.collect] "
                    + " ".join(
                        [
                            f"t={t}",
                            f"episode_id={episode_id}",
                            f"iter_tag={iter_tag}",
                            f"obs_keys={obs_keys_dbg}",
                            f"obs_key_samples={obs_debug_str}",
                            f"action_chunk={action_str}",
                            f"T_action={int(T_action)}",
                            f"n_action_steps_config={int(n_action_steps_config)}",
                            f"n_action_steps_executed={int(n_action_steps_executed)}",
                            f"info.rewards_raw({rewards_raw_str})",
                            f"info.dones_raw({dones_raw_str})",
                            f"len(inner_rewards)={len(inner_rewards_list)}",
                            f"len(inner_dones)={len(inner_dones_list)}",
                            f"terminated={bool(terminated0)}",
                            f"truncated={bool(truncated0)}",
                            f"done={bool(done0)}",
                            f"success_step={bool(success_step)}",
                        ]
                    )
                )

        if not done0:
            reward_online = -1.0
        else:
            reward_online = 0.0 if success_episode else float(-C_fail)

        episode_return_online += float(reward_online)

        obs_keys = sorted([str(k) for k in obs.keys()])
        action_keys = sorted([str(k) for k in action_chunk.keys()])

        transition = {
            "schema_version": "recap-v0",
            "code_version": str(code_version),
            "iter_tag": str(iter_tag),
            "episode_id": str(episode_id),
            "t": int(t),
            "T_action": int(T_action),
            "n_action_steps_config": int(n_action_steps_config),
            "n_action_steps_executed": int(n_action_steps_executed),
            "reward_online": float(reward_online),
            "reward_wrapper": float(reward_wrapper),
            "terminated": bool(terminated0),
            "truncated": bool(truncated0),
            "done": bool(done0),
            "obs_keys": obs_keys,
            "obs_summary": _summarize_mapping(obs),
            "action_keys": action_keys,
            "action_summary": _summarize_mapping(action_chunk),
            "inner_rewards": inner_rewards_list,
            "inner_dones": inner_dones_list,
            "success_step": bool(success_step),
        }
        from work.recap.episode_writer import TRANSITION_REQUIRED_KEYS

        if set(transition.keys()) != set(TRANSITION_REQUIRED_KEYS):
            missing = [k for k in TRANSITION_REQUIRED_KEYS if k not in transition]
            extra = [k for k in transition.keys() if k not in TRANSITION_REQUIRED_KEYS]
            raise KeyError(
                f"transition record schema mismatch: missing={missing} extra={extra}"
            )

        transitions.append(transition)

        per_step_state.append(_filter_prefix(obs, "state."))
        per_step_action.append(_filter_prefix(action_chunk, "action."))

        last_reward_wrapper = float(reward_wrapper)
        last_terminated = bool(terminated0)
        last_truncated = bool(truncated0)
        last_done = bool(done0)

        obs = next_obs
        if done0:
            break

    if n_action_steps_config is None:
        raise RuntimeError("No steps executed; cannot finalize episode")

    state_arrays = _stack_per_key(per_step_state) if arrays_saved else {}
    action_arrays = _stack_per_key(per_step_action) if arrays_saved else {}

    arrays_blob = {
        "state_arrays": state_arrays,
        "action_arrays": action_arrays,
    }

    episode_record = {
        "schema_version": "recap-v0",
        "code_version": str(code_version),
        "iter_tag": str(iter_tag),
        "episode_id": str(episode_id),
        "env_name": str(env_name),
        "model_path": str(model_path),
        "embodiment_tag": str(embodiment_tag),
        "server_host": str(server_host),
        "server_port": int(server_port),
        "seed": int(seed),
        "gamma": float(gamma),
        "C_fail": int(C_fail),
        "prompt_raw": str(prompt_raw),
        "prompt_conditioned": str(prompt_conditioned),
        "n_action_steps_config": int(n_action_steps_config),
        "terminated": bool(last_terminated),
        "truncated": bool(last_truncated),
        "done": bool(last_done),
        "success_episode": bool(success_episode),
        "episode_return_online": float(episode_return_online),
        "episode_return_wrapper": float(last_reward_wrapper),
        "n_policy_steps": int(len(transitions)),
        "video_dir_tmp": video_dir_tmp,
        "video_dir_archived": video_dir_archived,
        "arrays_saved": bool(arrays_saved),
        "npz_path": None,
    }

    from work.recap.episode_writer import EPISODE_REQUIRED_KEYS

    if set(episode_record.keys()) != set(EPISODE_REQUIRED_KEYS):
        missing = [k for k in EPISODE_REQUIRED_KEYS if k not in episode_record]
        extra = [k for k in episode_record.keys() if k not in EPISODE_REQUIRED_KEYS]
        raise KeyError(
            f"episode record schema mismatch: missing={missing} extra={extra}"
        )

    return episode_record, transitions, arrays_blob


__all__ = [
    "collect_episode",
    "extract_T_action",
    "normalize_prompt",
]
