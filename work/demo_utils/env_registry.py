from __future__ import annotations

import datetime as _dt
import importlib
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast


def _extract_registry_ids(registry: object) -> list[str]:
    if isinstance(registry, dict):
        return [str(k) for k in registry.keys()]

    env_specs = getattr(registry, "env_specs", None)
    if isinstance(env_specs, dict):
        return [str(k) for k in env_specs.keys()]

    keys = getattr(registry, "keys", None)
    if callable(keys):
        try:
            out: list[str] = []
            for k in cast(Iterable[object], keys()):
                out.append(str(k))
            return out
        except Exception:
            pass

    values = getattr(registry, "values", None)
    if callable(values):
        try:
            ids_from_values: list[str] = []
            for spec in cast(Iterable[object], values()):
                env_id = getattr(spec, "id", "")
                if env_id:
                    ids_from_values.append(str(env_id))
            return ids_from_values
        except Exception:
            pass

    all_fn = getattr(registry, "all", None)
    if callable(all_fn):
        try:
            ids_from_all: list[str] = []
            for spec in cast(Iterable[object], all_fn()):
                env_id = getattr(spec, "id", "")
                if env_id:
                    ids_from_all.append(str(env_id))
            return ids_from_all
        except Exception:
            pass

    return []


def list_registered_env_ids(
    *,
    prefix: str,
    log_path: Path,
    register_modules: Sequence[str] = (),
) -> list[str]:
    for mod in register_modules:
        _ = importlib.import_module(str(mod))

    reg_mod = None
    try:
        reg_mod = importlib.import_module("gymnasium.envs.registration")
    except Exception:
        reg_mod = importlib.import_module("gym.envs.registration")

    registry = getattr(reg_mod, "registry", None)
    ids = _extract_registry_ids(registry)
    wanted = sorted([i for i in ids if i.startswith(prefix)])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    with open(log_path, "a", encoding="utf-8", buffering=1) as f:
        _ = f.write(
            f"\n===== env registry {ts} pid={os.getpid()} prefix={prefix!r} =====\n"
        )
        for env_id in wanted:
            _ = f.write(env_id + "\n")

    for env_id in wanted:
        print(env_id)
    print(f"[INFO] listed {len(wanted)} env ids (prefix={prefix!r}); wrote: {log_path}")

    if not wanted:
        msg = (
            "[WARN] no matching envs found. Ensure you are running under the WBC venv and "
            + "that importing gr00t_wbc.control.envs.robocasa.sync_env succeeds."
        )
        print(msg)

    return wanted
