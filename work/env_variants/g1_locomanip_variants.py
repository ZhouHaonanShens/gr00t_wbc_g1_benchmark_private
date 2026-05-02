from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Callable, Iterable, Sequence

from typing import Protocol, cast


class _ArrayLike(Protocol):
    def __getitem__(self, key: object) -> object: ...

    def __setitem__(self, key: object, value: object) -> None: ...


class _DataModelLike(Protocol):
    qpos0: _ArrayLike


class _ModelLike(Protocol):
    body_names: Sequence[str]
    joint_names: Sequence[str]
    _model: object

    def body_name2id(self, name: str) -> int: ...

    def body_id2name(self, id: int) -> str: ...

    def get_joint_qpos_addr(self, name: str) -> int | tuple[int, int]: ...

    def get_joint_qvel_addr(self, name: str) -> int | tuple[int, int]: ...


class _DataLike(Protocol):
    body_xpos: Sequence[Sequence[float]]
    body_xquat: Sequence[Sequence[float]]
    model: _DataModelLike
    _data: object

    def get_joint_qpos(self, name: str) -> object: ...

    def set_joint_qpos(self, name: str, qpos: object) -> None: ...


class _SimLike(Protocol):
    model: _ModelLike
    data: _DataLike


@dataclasses.dataclass(frozen=True, slots=True)
class VariantSpec:
    name: str
    apple_pos_mode: str
    apple_pos_xy: tuple[float, float] | None = None
    lr_offset_y_m: float = 0.22
    farther_scale: float = 1.7
    debug_dump: bool = False


def make_variant_spec(
    *,
    name: str,
    apple_x: float | None = None,
    apple_y: float | None = None,
    debug_dump: bool = False,
) -> VariantSpec:
    v = str(name or "").strip()
    if not v:
        v = "official"

    if v == "official":
        return VariantSpec(
            name="official", apple_pos_mode="official", debug_dump=bool(debug_dump)
        )

    if v in ("apple_left_of_plate", "apple_right_of_plate"):
        return VariantSpec(
            name=v,
            apple_pos_mode=v,
            lr_offset_y_m=0.22,
            debug_dump=bool(debug_dump),
        )

    if v == "apple_farther":
        return VariantSpec(
            name=v,
            apple_pos_mode=v,
            farther_scale=1.7,
            debug_dump=bool(debug_dump),
        )

    if v == "apple_xy":
        if apple_x is None or apple_y is None:
            raise ValueError("apple_xy requires --apple-x and --apple-y")
        return VariantSpec(
            name=v,
            apple_pos_mode=v,
            apple_pos_xy=(float(apple_x), float(apple_y)),
            debug_dump=bool(debug_dump),
        )

    raise ValueError(
        f"Unknown env variant: {v!r} (supported: official, apple_left_of_plate, apple_right_of_plate, apple_farther, apple_xy)"
    )


def wrap_with_variant(
    env: object,
    spec: VariantSpec,
    *,
    print_fn: Callable[[str], None] | None = None,
) -> object:
    if spec.apple_pos_mode == "official":
        return env

    gym = importlib.import_module("gymnasium")

    def _print(s: str) -> None:
        if print_fn is None:
            print(str(s))
        else:
            print_fn(str(s))

    class _VariantResetWrapper(gym.Wrapper):
        def __init__(self, env: object):
            super().__init__(env)
            self._variant_spec = spec
            self._printed_header = False

        def reset(self, **kwargs):  # type: ignore[override]
            obs, info = super().reset(**kwargs)
            try:
                if not self._printed_header and bool(self._variant_spec.debug_dump):
                    _print(
                        "[VARIANT] enabled: %s (left/right defined as +/- world-Y relative to plate)"
                        % (self._variant_spec.name,)
                    )
                    self._printed_header = True

                changed = apply_variant(self.env, self._variant_spec, log_fn=_print)
                if changed:
                    obs = (
                        _refresh_obs_after_sim_mutation(self.env, log_fn=_print) or obs
                    )
            except Exception as e:
                _print(f"[VARIANT][WARN] apply_variant failed: {type(e).__name__}: {e}")
            return obs, info

    return _VariantResetWrapper(env)


def apply_variant(
    env: object, spec: VariantSpec, *, log_fn: Callable[[str], None] = print
) -> bool:
    if spec.apple_pos_mode == "official":
        return False

    np = importlib.import_module("numpy")

    sim = _find_sim(env)
    if sim is None:
        log_fn("[VARIANT][WARN] cannot find MuJoCo sim on env")
        return False

    base_env = _find_base_env_with_obj_body_id(env)

    apple_joint, candidates = _find_apple_free_joint(sim)
    if bool(spec.debug_dump):
        _dump_body_candidates(sim, needle="apple", log_fn=log_fn)
        _dump_candidates(candidates, log_fn=log_fn)

    if apple_joint is None:
        log_fn("[VARIANT][WARN] cannot find apple free joint (qpos size==7)")
        return False

    try:
        qpos_old = np.asarray(
            sim.data.get_joint_qpos(apple_joint), dtype=float
        ).reshape(-1)
    except Exception as e:
        log_fn(
            f"[VARIANT][WARN] get_joint_qpos failed for {apple_joint!r}: {type(e).__name__}: {e}"
        )
        return False

    if int(qpos_old.size) != 7:
        log_fn(
            f"[VARIANT][WARN] apple joint {apple_joint!r} qpos size != 7: {int(qpos_old.size)}"
        )
        return False

    pos_old = qpos_old[:3].copy()
    quat_old = qpos_old[3:7].copy()

    apple_body = _maybe_body_info(base_env, sim, "apple")
    plate_body = _maybe_body_info(base_env, sim, "plate")

    plate_pos = _get_plate_pos_xyz(base_env, sim)

    pos_new = pos_old.copy()
    if spec.apple_pos_mode == "apple_left_of_plate":
        if plate_pos is not None:
            pos_new[0] = float(plate_pos[0])
            pos_new[1] = float(plate_pos[1]) - abs(float(spec.lr_offset_y_m))
        else:
            pos_new[1] = float(pos_old[1]) - abs(float(spec.lr_offset_y_m))
    elif spec.apple_pos_mode == "apple_right_of_plate":
        if plate_pos is not None:
            pos_new[0] = float(plate_pos[0])
            pos_new[1] = float(plate_pos[1]) + abs(float(spec.lr_offset_y_m))
        else:
            pos_new[1] = float(pos_old[1]) + abs(float(spec.lr_offset_y_m))
    elif spec.apple_pos_mode == "apple_farther":
        if plate_pos is not None:
            v = pos_old - np.asarray(plate_pos, dtype=float).reshape(3)
            s = float(spec.farther_scale)
            s = max(1.0, min(3.0, s))
            pos_new = np.asarray(plate_pos, dtype=float).reshape(3) + v * s
        else:
            pos_new[0] = float(pos_old[0]) + 0.25
    elif spec.apple_pos_mode == "apple_xy":
        assert spec.apple_pos_xy is not None
        pos_new[0] = float(spec.apple_pos_xy[0])
        pos_new[1] = float(spec.apple_pos_xy[1])
    else:
        log_fn(f"[VARIANT][WARN] unsupported apple_pos_mode: {spec.apple_pos_mode!r}")
        return False

    pos_new[2] = float(pos_old[2])

    qpos_new = np.concatenate(
        [np.asarray(pos_new, dtype=float).reshape(3), quat_old.reshape(4)]
    )

    if bool(spec.debug_dump):
        qpos_addr = _maybe_joint_addr(sim, apple_joint, "get_joint_qpos_addr")
        qvel_addr = _maybe_joint_addr(sim, apple_joint, "get_joint_qvel_addr")

        apple_body_s = _fmt_body_info(apple_body)
        plate_body_s = _fmt_body_info(plate_body)

        log_fn(
            "[VARIANT] apple_joint=%s qpos_addr=%s qvel_addr=%s apple_body=%s plate_body=%s"
            % (
                apple_joint,
                qpos_addr,
                qvel_addr,
                apple_body_s,
                plate_body_s,
            )
        )
        log_fn(
            "[VARIANT] pos_old=%s yaw_old=%.3f quat_old(wxyz)=%s"
            % (
                _fmt_vec3(pos_old),
                _yaw_from_quat_wxyz(quat_old),
                _fmt_vec4(quat_old),
            )
        )
        if plate_pos is not None:
            log_fn("[VARIANT] plate_pos=%s" % (_fmt_vec3(plate_pos),))
        log_fn(
            "[VARIANT] pos_new=%s yaw_new=%.3f quat_new(wxyz)=%s"
            % (
                _fmt_vec3(pos_new),
                _yaw_from_quat_wxyz(quat_old),
                _fmt_vec4(quat_old),
            )
        )

    _set_free_joint_qpos_and_qpos0(sim, apple_joint, qpos_new)
    _forward_sim(sim)

    if bool(spec.debug_dump):
        try:
            qpos_after = np.asarray(
                sim.data.get_joint_qpos(apple_joint), dtype=float
            ).reshape(-1)
            log_fn(
                "[VARIANT] pos_after=%s yaw_after=%.3f quat_after(wxyz)=%s"
                % (
                    _fmt_vec3(qpos_after[:3]),
                    _yaw_from_quat_wxyz(qpos_after[3:7]),
                    _fmt_vec4(qpos_after[3:7]),
                )
            )
        except Exception as e:
            log_fn(
                f"[VARIANT][WARN] failed to read qpos_after: {type(e).__name__}: {e}"
            )
        if apple_body is not None:
            try:
                bid = int(apple_body[0])
                bx = sim.data.body_xpos[bid]
                bq = sim.data.body_xquat[bid]
                log_fn(
                    "[VARIANT] apple_body_xpos_after=%s apple_body_xquat_after(wxyz)=%s"
                    % (_fmt_vec3(bx), _fmt_vec4(bq))
                )
            except Exception:
                pass
    return True


def _find_sim(env: object) -> _SimLike | None:
    seen: set[int] = set()

    def _enqueue(x: object, q: list[object]) -> None:
        if x is None:
            return
        xid = id(x)
        if xid in seen:
            return
        seen.add(xid)
        q.append(x)

    q: list[object] = []
    _enqueue(env, q)

    for _ in range(50):
        if not q:
            break
        cur = q.pop(0)
        if hasattr(cur, "sim"):
            sim = getattr(cur, "sim")
            if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
                return cast(_SimLike, sim)
        if hasattr(cur, "base_env"):
            _enqueue(getattr(cur, "base_env"), q)
            be = getattr(cur, "base_env")
            if be is not None and hasattr(be, "sim"):
                sim = getattr(be, "sim")
                if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
                    return cast(_SimLike, sim)
        for k in ("env", "unwrapped"):
            if hasattr(cur, k):
                _enqueue(getattr(cur, k), q)
    return None


def _find_base_env_with_obj_body_id(env: object) -> object | None:
    seen: set[int] = set()
    q: list[object] = [env]
    for _ in range(60):
        if not q:
            break
        cur = q.pop(0)
        if cur is None:
            continue
        xid = id(cur)
        if xid in seen:
            continue
        seen.add(xid)

        if hasattr(cur, "obj_body_id"):
            ob = getattr(cur, "obj_body_id")
            if isinstance(ob, dict):
                return cur
        for k in ("base_env", "env", "unwrapped"):
            if hasattr(cur, k):
                q.append(getattr(cur, k))
    return None


def _get_plate_pos_xyz(
    base_env: object | None, sim: _SimLike
) -> Sequence[float] | None:
    try:
        if base_env is not None:
            ob = getattr(base_env, "obj_body_id", None)
            if isinstance(ob, dict) and "plate" in ob:
                bid = int(ob["plate"])
                return list(sim.data.body_xpos[bid])
    except Exception:
        pass

    try:
        names = [str(x) for x in getattr(sim.model, "body_names", [])]
        cand = [n for n in names if "plate" in n.lower()]
        if cand:
            bid = int(sim.model.body_name2id(cand[0]))
            return list(sim.data.body_xpos[bid])
    except Exception:
        pass
    return None


def _find_apple_free_joint(sim: _SimLike) -> tuple[str | None, list[tuple[str, int]]]:
    np = importlib.import_module("numpy")
    candidates: list[tuple[str, int]] = []
    chosen: str | None = None

    joint_names = [str(x) for x in getattr(sim.model, "joint_names", [])]
    apple_like = [j for j in joint_names if "apple" in j.lower()]
    if not apple_like:
        apple_like = joint_names

    for j in apple_like:
        try:
            q = sim.data.get_joint_qpos(j)
            sz = int(np.size(q))
        except Exception:
            continue
        candidates.append((j, sz))
        if chosen is None and sz == 7 and "apple" in j.lower():
            chosen = j

    if chosen is None:
        for j, sz in candidates:
            if sz == 7 and "apple" in j.lower():
                chosen = j
                break

    if chosen is None:
        for j, sz in candidates:
            if sz == 7:
                chosen = j
                break

    return chosen, candidates


def _dump_candidates(
    candidates: Iterable[tuple[str, int]], *, log_fn: Callable[[str], None]
) -> None:
    c = list(candidates)
    c_sorted = sorted(c, key=lambda x: (0 if "apple" in x[0].lower() else 1, x[0]))
    head = c_sorted[:40]
    log_fn("[VARIANT][DEBUG] apple joint candidates (name -> qpos_size):")
    for name, sz in head:
        log_fn(f"  - {name}: {sz}")
    if len(c_sorted) > len(head):
        log_fn(f"  ... ({len(c_sorted) - len(head)} more)")


def _dump_body_candidates(
    sim: object, *, needle: str, log_fn: Callable[[str], None]
) -> None:
    sim2 = cast(_SimLike, sim)
    names = [str(x) for x in getattr(sim2.model, "body_names", [])]
    n = str(needle or "").lower()
    hits = [x for x in names if n in x.lower()]
    hits = sorted(hits)
    log_fn(f"[VARIANT][DEBUG] body candidates containing {needle!r}:")
    for b in hits[:60]:
        log_fn(f"  - {b}")
    if len(hits) > 60:
        log_fn(f"  ... ({len(hits) - 60} more)")


def _maybe_body_info(
    base_env: object | None, sim: object, key: str
) -> tuple[int, str] | None:
    sim2 = cast(_SimLike, sim)
    if base_env is not None:
        ob = getattr(base_env, "obj_body_id", None)
        if isinstance(ob, dict) and key in ob:
            try:
                bid = int(ob[key])
                name = str(sim2.model.body_id2name(bid))
                return bid, name
            except Exception:
                return None
    try:
        names = [str(x) for x in getattr(sim2.model, "body_names", [])]
        hits = [x for x in names if str(key).lower() in x.lower()]
        if hits:
            bid = int(sim2.model.body_name2id(hits[0]))
            return bid, str(hits[0])
    except Exception:
        return None
    return None


def _fmt_body_info(body: tuple[int, str] | None) -> str:
    if body is None:
        return "-"
    return "%s#%d" % (body[1], int(body[0]))


def _maybe_joint_addr(sim: _SimLike, joint_name: str, attr: str) -> str:
    try:
        fn = getattr(sim.model, attr, None)
        if callable(fn):
            v = fn(joint_name)
            return str(v)
    except Exception:
        return "-"
    return "-"


def _set_free_joint_qpos_and_qpos0(
    sim: _SimLike, joint_name: str, qpos7: object
) -> None:
    np = importlib.import_module("numpy")
    q = np.asarray(qpos7, dtype=float).reshape(-1)
    if int(q.size) != 7:
        raise ValueError(f"expected 7-d qpos, got {int(q.size)}")

    try:
        addr = sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, tuple) and len(addr) == 2:
            start_i, end_i = int(addr[0]), int(addr[1])
            sim.data.model.qpos0[start_i:end_i] = q
    except Exception:
        pass

    sim.data.set_joint_qpos(joint_name, q)


def _forward_sim(sim: object) -> None:
    sim2 = cast(_SimLike, sim)
    try:
        fn = getattr(sim2, "forward", None)
        if callable(fn):
            fn()
            return
    except Exception:
        pass

    try:
        mujoco = importlib.import_module("mujoco")
        model = getattr(sim2.model, "_model", None)
        data = getattr(sim2.data, "_data", None)
        if model is not None and data is not None:
            mujoco.mj_forward(model, data)
    except Exception:
        return


def _refresh_obs_after_sim_mutation(
    env: object, *, log_fn: Callable[[str], None]
) -> object | None:
    def _accept(x: object) -> object | None:
        if isinstance(x, dict) and "q" in x:
            return x
        return None

    try:
        if hasattr(env, "env") and hasattr(
            getattr(env, "env"), "force_update_observation"
        ):
            raw = getattr(env, "env").force_update_observation(timestep=0)
            cache = getattr(env, "cache", None)
            if isinstance(cache, dict):
                cache["obs"] = raw
            obs_fn = getattr(env, "observe", None)
            if callable(obs_fn):
                return _accept(obs_fn())
            return _accept(raw)
    except Exception as e:
        log_fn(
            f"[VARIANT][WARN] force_update_observation failed: {type(e).__name__}: {e}"
        )

    for name in ("get_observation", "_get_observations"):
        try:
            fn = getattr(env, name, None)
            if callable(fn):
                return _accept(fn())
        except Exception:
            pass
    return None


def _fmt_vec3(x: Sequence[float]) -> str:
    return "[%.3f, %.3f, %.3f]" % (float(x[0]), float(x[1]), float(x[2]))


def _fmt_vec4(x: Sequence[float]) -> str:
    return "[%.3f, %.3f, %.3f, %.3f]" % (
        float(x[0]),
        float(x[1]),
        float(x[2]),
        float(x[3]),
    )


def _yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
