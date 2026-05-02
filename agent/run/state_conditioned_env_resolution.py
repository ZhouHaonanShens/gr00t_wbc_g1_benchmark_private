from __future__ import annotations

from collections.abc import Iterable
import difflib
from typing import Any, cast


LOGICAL_TASK_APPLE_TO_PLATE_G1 = "apple_to_plate_g1"
ENV_REGISTRY_PREFIX = "gr00tlocomanip_g1_sim/"
DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME = (
    "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
)

APPLE_TO_PLATE_G1_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME: (
        "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_g1_sim_gear_wbc",
    ),
}


def _extract_registry_ids(registry: object) -> list[str]:
    if isinstance(registry, dict):
        return [str(key) for key in registry.keys()]

    env_specs = getattr(registry, "env_specs", None)
    if isinstance(env_specs, dict):
        return [str(key) for key in env_specs.keys()]

    keys = getattr(registry, "keys", None)
    if callable(keys):
        try:
            return [str(key) for key in cast(Iterable[object], keys())]
        except Exception:
            pass

    values = getattr(registry, "values", None)
    if callable(values):
        try:
            ids: list[str] = []
            for spec in cast(Iterable[object], values()):
                env_id = getattr(spec, "id", "")
                if env_id:
                    ids.append(str(env_id))
            return ids
        except Exception:
            pass

    all_fn = getattr(registry, "all", None)
    if callable(all_fn):
        try:
            ids = []
            for spec in cast(Iterable[object], all_fn()):
                env_id = getattr(spec, "id", "")
                if env_id:
                    ids.append(str(env_id))
            return ids
        except Exception:
            pass

    return []


def registered_g1_env_ids(gym_module: object) -> list[str]:
    registry = getattr(getattr(gym_module, "envs", None), "registry", None)
    return sorted(
        env_id
        for env_id in _extract_registry_ids(registry)
        if env_id.startswith(ENV_REGISTRY_PREFIX)
    )


def _close_matches(
    requested_env_name: str, registered_env_ids: Iterable[str]
) -> list[str]:
    registered = [str(env_id) for env_id in registered_env_ids]
    apple_matches = [
        env_id
        for env_id in registered
        if "AppleToPlate" in env_id or ("Apple" in env_id and "Plate" in env_id)
    ]
    approx_matches = difflib.get_close_matches(
        str(requested_env_name),
        registered,
        n=5,
        cutoff=0.35,
    )
    close_matches: list[str] = []
    for candidate in [*apple_matches, *approx_matches]:
        if candidate not in close_matches:
            close_matches.append(candidate)
    return close_matches[:5]


class StateConditionedEnvResolutionError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        logical_task: str,
        requested_env_name: str,
        alias_candidates: Iterable[str],
        available_close_matches: Iterable[str],
        registered_env_ids: Iterable[str],
    ):
        self.code = str(code)
        self.logical_task = str(logical_task)
        self.requested_env_name = str(requested_env_name)
        self.alias_candidates = [str(value) for value in alias_candidates]
        self.available_close_matches = [str(value) for value in available_close_matches]
        self.registered_env_ids = [str(value) for value in registered_env_ids]
        super().__init__(
            "env resolution failed"
            + f": code={self.code} logical_task={self.logical_task}"
            + f" requested_env_name={self.requested_env_name!r}"
            + f" available_close_matches={self.available_close_matches!r}"
        )

    def to_machine_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "logical_task": self.logical_task,
            "requested_env_name": self.requested_env_name,
            "alias_candidates": list(self.alias_candidates),
            "available_close_matches": list(self.available_close_matches),
            "registered_env_prefix": ENV_REGISTRY_PREFIX,
            "registered_env_count": int(len(self.registered_env_ids)),
        }


def resolve_apple_to_plate_g1_env_name(
    gym_module: object,
    *,
    requested_env_name: str | None = None,
) -> dict[str, Any]:
    requested = str(
        requested_env_name
        if requested_env_name is not None
        else DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
    )
    registered = registered_g1_env_ids(gym_module)
    alias_candidates = [
        requested,
        *APPLE_TO_PLATE_G1_ENV_ALIASES.get(requested, ()),
    ]
    for index, candidate in enumerate(alias_candidates):
        if candidate in registered:
            return {
                "logical_task": LOGICAL_TASK_APPLE_TO_PLATE_G1,
                "requested_env_name": requested,
                "resolved_env_name": candidate,
                "alias_applied": bool(index > 0),
                "alias_candidates": alias_candidates,
                "registered_env_ids": registered,
                "available_close_matches": _close_matches(requested, registered),
            }

    raise StateConditionedEnvResolutionError(
        code="state_conditioned_env_unavailable",
        logical_task=LOGICAL_TASK_APPLE_TO_PLATE_G1,
        requested_env_name=requested,
        alias_candidates=alias_candidates,
        available_close_matches=_close_matches(requested, registered),
        registered_env_ids=registered,
    )


__all__ = [
    "APPLE_TO_PLATE_G1_ENV_ALIASES",
    "DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME",
    "ENV_REGISTRY_PREFIX",
    "LOGICAL_TASK_APPLE_TO_PLATE_G1",
    "StateConditionedEnvResolutionError",
    "registered_g1_env_ids",
    "resolve_apple_to_plate_g1_env_name",
]
