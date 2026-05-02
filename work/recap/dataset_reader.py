from __future__ import annotations

import json
from pathlib import Path


JsonRecord = dict[str, object]


def _read_jsonl(path: Path) -> list[JsonRecord]:
    if not path.exists():
        raise ValueError(f"Missing file: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    out: list[JsonRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj_raw: object = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {lineno}: {e}") from e
            if not isinstance(obj_raw, dict):
                raise ValueError(
                    f"Invalid record type in {path} at line {lineno}: expected object, got {type(obj_raw).__name__}"
                )

            obj: JsonRecord = {}
            for k, v in obj_raw.items():
                if not isinstance(k, str):
                    raise ValueError(
                        f"Invalid key type in {path} at line {lineno}: expected str key, got {type(k).__name__}"
                    )
                obj[k] = v
            out.append(obj)
    return out


def _require_str_field(record: JsonRecord, field: str, *, context: str) -> str:
    val = record.get(field)
    if val is None:
        raise ValueError(f"Missing {field} ({context})")
    if not isinstance(val, str) or not val:
        raise ValueError(
            f"Invalid {field} ({context}): expected non-empty str, got {val!r}"
        )
    return val


def _coerce_int_field(record: JsonRecord, field: str, *, context: str) -> int:
    val = record.get(field)
    if val is None:
        raise ValueError(f"Missing {field} ({context})")
    if isinstance(val, bool):
        raise ValueError(f"Invalid {field} ({context}): expected int, got bool")

    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not val.is_integer():
            raise ValueError(
                f"Invalid {field} ({context}): expected integer-valued number, got {val!r}"
            )
        return int(val)
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError as e:
            raise ValueError(
                f"Invalid {field} ({context}): expected int-like str, got {val!r}"
            ) from e

    raise ValueError(f"Invalid {field} ({context}): expected int-like, got {val!r}")


def _validate_transition_invariant(
    transition: JsonRecord, *, episode_id: str, t: int
) -> None:
    context = f"episode_id={episode_id} t={t}"

    n_exec = _coerce_int_field(transition, "n_action_steps_executed", context=context)
    inner_rewards = transition.get("inner_rewards")
    inner_dones = transition.get("inner_dones")

    if not isinstance(inner_rewards, list):
        raise ValueError(
            f"Invalid inner_rewards ({context}): expected list, got {type(inner_rewards).__name__}"
        )
    if not isinstance(inner_dones, list):
        raise ValueError(
            f"Invalid inner_dones ({context}): expected list, got {type(inner_dones).__name__}"
        )

    if n_exec != len(inner_rewards) or n_exec != len(inner_dones):
        raise ValueError(
            f"M1 invariant violation ({context}): n_action_steps_executed={n_exec} "
            f"len(inner_rewards)={len(inner_rewards)} len(inner_dones)={len(inner_dones)}"
        )


def _assert_npz_has_no_video_keys(npz_path: Path, *, episode_id: str) -> None:
    try:
        import numpy as np
    except Exception as e:
        raise ValueError(
            f"episode_id={episode_id} requires numpy for NPZ validation but import failed: {e}"
        ) from e

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            keys = list(getattr(data, "files", []))
    except Exception as e:
        raise ValueError(
            f"episode_id={episode_id} failed to read npz: {npz_path}: {e}"
        ) from e

    bad = [
        k
        for k in keys
        if isinstance(k, str) and (k.startswith("video/") or k.startswith("video."))
    ]
    if bad:
        bad_preview = ", ".join(bad[:5])
        more = "" if len(bad) <= 5 else f" (+{len(bad) - 5} more)"
        raise ValueError(
            f"episode_id={episode_id} NPZ contains forbidden video keys: {bad_preview}{more} (file={npz_path})"
        )


def read_m1_dataset(
    dataset_dir: str | Path, *, check_npz_keys: bool = True
) -> dict[str, object]:
    dataset_dir_path = Path(dataset_dir)
    episodes_path = dataset_dir_path / "episodes.jsonl"
    transitions_path = dataset_dir_path / "transitions.jsonl"

    episodes = _read_jsonl(episodes_path)
    episode_by_id: dict[str, JsonRecord] = {}
    episodes_out: list[JsonRecord] = []
    for idx, ep in enumerate(episodes):
        context = f"episodes.jsonl record#{idx + 1}"
        episode_id = _require_str_field(ep, "episode_id", context=context)
        if episode_id in episode_by_id:
            raise ValueError(f"Duplicate episode_id in episodes.jsonl: {episode_id}")

        if "prompt_raw" not in ep:
            ep["prompt_raw"] = None
        if "prompt_conditioned" not in ep:
            ep["prompt_conditioned"] = None
        if "npz_path" not in ep:
            ep["npz_path"] = None

        episode_by_id[episode_id] = ep
        episodes_out.append(ep)

    transitions = _read_jsonl(transitions_path)
    transitions_by_episode: dict[str, list[JsonRecord]] = {
        eid: [] for eid in episode_by_id
    }

    for idx, tr in enumerate(transitions):
        context = f"transitions.jsonl record#{idx + 1}"
        episode_id = _require_str_field(tr, "episode_id", context=context)
        if episode_id not in episode_by_id:
            raise ValueError(
                f"Unknown episode_id in transitions.jsonl ({context}): {episode_id} (missing in episodes.jsonl)"
            )

        t = _coerce_int_field(tr, "t", context=f"episode_id={episode_id}")
        tr["t"] = t

        _validate_transition_invariant(tr, episode_id=episode_id, t=t)

        ep = episode_by_id[episode_id]
        for k in ("prompt_raw", "prompt_conditioned", "npz_path"):
            if k not in tr:
                tr[k] = ep.get(k)

        transitions_by_episode[episode_id].append(tr)

    for episode_id, trs in transitions_by_episode.items():

        def _t_key(tr: JsonRecord) -> int:
            t_val = tr.get("t")
            if not isinstance(t_val, int):
                raise ValueError(
                    f"Invalid t type after parsing: episode_id={episode_id} t={t_val!r}"
                )
            return t_val

        trs.sort(key=_t_key)
        if not trs:
            continue

        expected_t = 0
        for tr in trs:
            t_obj = tr.get("t")
            if not isinstance(t_obj, int):
                raise ValueError(
                    f"Invalid t type after parsing: episode_id={episode_id} t={t_obj!r}"
                )
            t = t_obj
            if t != expected_t:
                raise ValueError(
                    f"Invalid t sequence: episode_id={episode_id} expected t={expected_t} but got t={t}"
                )
            expected_t += 1

    if check_npz_keys:
        for episode_id, ep in episode_by_id.items():
            npz_path_val = ep.get("npz_path")
            npz_candidate: Path | None = None

            if isinstance(npz_path_val, str) and npz_path_val:
                p = Path(npz_path_val)
                npz_candidate = p if p.is_absolute() else (dataset_dir_path / p)
                if not npz_candidate.exists():
                    raise ValueError(
                        f"episode_id={episode_id} npz_path does not exist: {npz_candidate} (npz_path={npz_path_val!r})"
                    )
            else:
                p = dataset_dir_path / "arrays" / f"{episode_id}.npz"
                if p.exists():
                    npz_candidate = p

            if npz_candidate is not None:
                _assert_npz_has_no_video_keys(npz_candidate, episode_id=episode_id)

    return {
        "episodes": episodes_out,
        "transitions_by_episode": transitions_by_episode,
        "n_episodes": len(episodes_out),
        "n_transitions": len(transitions),
        "dataset_dir": str(dataset_dir_path),
    }
