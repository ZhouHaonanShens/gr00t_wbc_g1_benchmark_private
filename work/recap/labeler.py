from __future__ import annotations

import importlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

from work.recap import text_indicator
from work.recap.label_writer import REQUIRED_LABEL_KEYS, validate_label_record
from work.recap.lerobot_export.dataset_export import STATE_KEY_ORDER_LOCK
from work.recap.phase_thresholds import (
    DEFAULT_EPSILON_THRESHOLD_PHASE,
    resolve_epsilon_quantile,
)


JsonRecord = dict[str, object]

ValueBaseline = Literal["mean_return", "t_mean_return"]
EpsilonStrategy = Literal["const", "quantile"]

_MULTIMODAL_CRITIC_TYPE = "multimodal_distributional_v1"
_MULTIMODAL_BACKEND_QWEN3_VL_LATE_FUSION_V1 = "qwen3_vl_late_fusion_v1"

REQUIRED_PRELABEL_KEYS = [
    "schema_version",
    "code_version",
    "iter_tag",
    "episode_id",
    "t",
    "return_G",
    "value_V",
    "advantage_A",
    "is_correction",
    "prompt_raw",
]

_REQUIRED_PRELABEL_KEYS_SET = frozenset(REQUIRED_PRELABEL_KEYS)


def _to_float(x: object, *, context: str) -> float:
    if isinstance(x, bool):
        raise ValueError(f"Expected float-like, got bool ({context})")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError as e:
            raise ValueError(f"Invalid float-like str {x!r} ({context})") from e
    raise ValueError(f"Expected float-like, got {type(x).__name__} ({context})")


def _to_int(x: object, *, context: str) -> int:
    if isinstance(x, bool):
        raise ValueError(f"Expected int-like, got bool ({context})")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not x.is_integer():
            raise ValueError(f"Expected integer-valued number, got {x!r} ({context})")
        return int(x)
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError as e:
            raise ValueError(f"Invalid int-like str {x!r} ({context})") from e
    raise ValueError(f"Expected int-like, got {type(x).__name__} ({context})")


def _to_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def validate_m2_prelabel_record(obj: object) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"prelabel record must be a dict, got: {type(obj).__name__}")

    record = cast(dict[str, object], obj)
    keys = set(record.keys())
    missing = _REQUIRED_PRELABEL_KEYS_SET - keys
    extra = keys - _REQUIRED_PRELABEL_KEYS_SET
    if missing or extra:
        missing_s = ", ".join(sorted(missing)) if missing else "(none)"
        extra_s = ", ".join(sorted(extra)) if extra else "(none)"
        raise ValueError(
            f"prelabel record keys mismatch; missing=[{missing_s}] extra=[{extra_s}]"
        )

    episode_id = record.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(
            f"prelabel episode_id must be a non-empty str, got: {episode_id!r}"
        )

    for field in ("schema_version", "code_version", "iter_tag"):
        value = record.get(field)
        if not isinstance(value, str):
            raise ValueError(
                f"prelabel {field} must be a str, got: {type(value).__name__}"
            )

    _ = _to_int(record.get("t"), context="prelabel.t")
    _ = _to_float(record.get("return_G"), context="prelabel.return_G")
    _ = _to_float(record.get("value_V"), context="prelabel.value_V")
    _ = _to_float(record.get("advantage_A"), context="prelabel.advantage_A")

    prompt_raw = record.get("prompt_raw")
    if not isinstance(prompt_raw, str):
        raise ValueError(
            f"prelabel prompt_raw must be a str, got: {type(prompt_raw).__name__}"
        )


def _quantile_linear(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"q must be in [0,1], got {q!r}")

    s = sorted(float(v) for v in values)
    n = len(s)
    if n == 1:
        return float(s[0])

    pos = float(q) * float(n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    w = pos - float(lo)
    return float((1.0 - w) * float(s[lo]) + w * float(s[hi]))


def _quantile(values: list[float], q: float) -> float:
    return _quantile_linear(values, q)


def _compute_returns_mc_gamma1(rewards: list[float]) -> list[float]:
    out = [0.0] * len(rewards)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc += float(rewards[i])
        out[i] = float(acc)
    return out


def _compute_value_baseline(
    return_G_by_record: list[float],
    t_by_record: list[int],
    *,
    value_baseline: ValueBaseline,
) -> list[float]:
    if not return_G_by_record:
        return []

    if value_baseline == "mean_return":
        mean = float(sum(return_G_by_record) / float(len(return_G_by_record)))
        return [mean for _ in return_G_by_record]

    if value_baseline == "t_mean_return":
        sums: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for g, t in zip(return_G_by_record, t_by_record, strict=True):
            sums[int(t)] += float(g)
            counts[int(t)] += 1

        global_mean = float(sum(return_G_by_record) / float(len(return_G_by_record)))
        means_by_t = {
            int(t): (float(sums[int(t)]) / float(counts[int(t)])) for t in counts
        }
        return [float(means_by_t.get(int(t), global_mean)) for t in t_by_record]

    raise ValueError(f"Unknown value_baseline: {value_baseline!r}")


def _import_numpy() -> Any:
    try:
        import importlib

        return importlib.import_module("numpy")
    except Exception as e:
        raise RuntimeError(f"generate_m2_labels requires numpy at runtime: {e}") from e


def _resolve_episode_npz_path(
    dataset: Mapping[str, object], *, episode_id: str, episode: JsonRecord
) -> Path:
    dataset_dir_raw = dataset.get("dataset_dir")
    dataset_dir: Path | None = None
    if isinstance(dataset_dir_raw, str) and dataset_dir_raw:
        dataset_dir = Path(dataset_dir_raw).expanduser().resolve()

    npz_path_val = episode.get("npz_path")
    if isinstance(npz_path_val, str) and npz_path_val:
        p = Path(npz_path_val).expanduser()
        if not p.is_absolute() and dataset_dir is not None:
            p = dataset_dir / p
        npz_path = p.resolve()
    else:
        if dataset_dir is None:
            raise ValueError(
                f"episode_id={episode_id} missing dataset['dataset_dir']; cannot resolve default arrays/{episode_id}.npz"
            )
        npz_path = (dataset_dir / "arrays" / f"{episode_id}.npz").resolve()

    if not npz_path.exists() or not npz_path.is_file():
        raise ValueError(f"episode_id={episode_id} missing npz file: {npz_path}")
    return npz_path


def _load_state_by_step_from_npz(
    npz_path: Path, *, episode_id: str, n_policy_steps: int, np: Any
) -> Any:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            keys = list(getattr(data, "files", []))
            state_keys = sorted(
                [k for k in keys if isinstance(k, str) and k.startswith("state/")]
            )
            if state_keys != STATE_KEY_ORDER_LOCK:
                raise ValueError(
                    f"episode_id={episode_id} state key order mismatch in npz: expected {STATE_KEY_ORDER_LOCK} but got {state_keys} (file={npz_path})"
                )

            parts: list[object] = []
            for k in STATE_KEY_ORDER_LOCK:
                if k not in data:
                    raise ValueError(
                        f"episode_id={episode_id} missing key in npz: {k!r} (file={npz_path})"
                    )
                arr = np.asarray(data[k])
                if getattr(arr, "ndim", None) != 4:
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} expected ndim=4, got shape={getattr(arr, 'shape', None)} (file={npz_path})"
                    )
                if int(arr.shape[0]) != int(n_policy_steps):
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} n_policy_steps mismatch: transitions={n_policy_steps} but npz has {arr.shape[0]} (file={npz_path})"
                    )
                if int(arr.shape[1]) != 1 or int(arr.shape[2]) != 1:
                    raise ValueError(
                        f"episode_id={episode_id} key={k!r} expected shape[1:3]=(1,1), got shape={arr.shape} (file={npz_path})"
                    )
                parts.append(arr[:, 0, 0, :].astype(np.float32, copy=False))

            state_by_step = np.concatenate(parts, axis=-1)
            if getattr(state_by_step, "ndim", None) != 2 or int(
                state_by_step.shape[0]
            ) != int(n_policy_steps):
                raise ValueError(
                    f"episode_id={episode_id} invalid state_by_step shape: {getattr(state_by_step, 'shape', None)} (file={npz_path})"
                )
            return state_by_step
    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError, KeyError)):
            raise
        raise ValueError(
            f"episode_id={episode_id} failed to read/parse npz: {npz_path}: {e}"
        ) from e


def _load_json_object(path: Path) -> dict[str, object]:
    import json

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    out: dict[str, object] = {}
    for key, value in obj.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid non-str JSON key in {path}: {key!r}")
        out[key] = value
    return out


def _resolve_critic_backend_kind(
    critic_dir: str | Path,
) -> tuple[str, Path, dict[str, object]]:
    critic_path = Path(critic_dir).expanduser().resolve()
    if not critic_path.exists() or not critic_path.is_dir():
        raise FileNotFoundError(f"critic_dir is not a directory: {critic_path}")

    config_path = critic_path / "config.json"
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(
            f"unknown_critic_backend: missing config.json in critic_dir={critic_path}"
        )
    config = _load_json_object(config_path)
    artifact_version = config.get("artifact_version")
    critic_type = config.get("critic_type")
    backend_name = config.get("smoke_backend")

    if (
        artifact_version == _MULTIMODAL_CRITIC_TYPE
        and critic_type == _MULTIMODAL_CRITIC_TYPE
    ):
        if backend_name != _MULTIMODAL_BACKEND_QWEN3_VL_LATE_FUSION_V1:
            raise ValueError(
                "unknown_critic_backend: "
                f"artifact_version={artifact_version!r} critic_type={critic_type!r} "
                f"backend_name={backend_name!r} critic_dir={critic_path}"
            )
        return _MULTIMODAL_CRITIC_TYPE, critic_path, config

    if all(key in config for key in ("state_dim", "include_t", "bin_centers")):
        return "state_only_dist_bins", critic_path, config

    raise ValueError(
        "unknown_critic_backend: "
        f"artifact_version={artifact_version!r} critic_type={critic_type!r} critic_dir={critic_path}"
    )


def _compute_value_from_state_only_critic(
    dataset: Mapping[str, object],
    *,
    base_records: list[JsonRecord],
    transitions_by_episode: dict[str, list[JsonRecord]],
    episode_by_id: dict[str, JsonRecord],
    critic_dir: str | Path,
) -> list[float]:
    from agent.archive.recap_legacy_state_only_critic.critic_dist_bins import (
        load_critic,
        predict_value_V,
    )

    np = _import_numpy()
    predictor = load_critic(critic_dir)
    include_t = bool(getattr(predictor.config, "include_t", False))
    expected_state_dim = int(getattr(predictor.config, "state_dim"))

    cache_state_by_episode: dict[str, tuple[Any, Path]] = {}
    expected_t_by_episode: dict[str, int] = {}

    out: list[float] = []
    for rec in base_records:
        episode_id = _to_str(rec.get("episode_id"))
        if not episode_id:
            raise ValueError("internal error: missing episode_id in base record")
        t = _to_int(rec.get("t"), context=f"episode_id={episode_id} label.t")

        if episode_id not in cache_state_by_episode:
            ep = episode_by_id.get(episode_id)
            if not isinstance(ep, dict):
                raise ValueError(f"episode_id={episode_id} missing episode record")
            trs = transitions_by_episode.get(episode_id)
            if not isinstance(trs, list):
                raise ValueError(f"episode_id={episode_id} missing transitions list")

            npz_path = _resolve_episode_npz_path(
                dataset, episode_id=episode_id, episode=ep
            )
            state_by_step: Any = _load_state_by_step_from_npz(
                npz_path, episode_id=episode_id, n_policy_steps=len(trs), np=np
            )
            state_dim = int(getattr(state_by_step, "shape")[1])
            if int(state_dim) != int(expected_state_dim):
                raise ValueError(
                    f"episode_id={episode_id} state_dim mismatch: critic expects {expected_state_dim} but npz has {state_dim} (file={npz_path})"
                )
            cache_state_by_episode[episode_id] = (state_by_step, npz_path)
            expected_t_by_episode[episode_id] = 0

        state_by_step, npz_path = cache_state_by_episode[episode_id]
        exp_t = int(expected_t_by_episode[episode_id])
        if int(t) != int(exp_t):
            raise ValueError(
                f"episode_id={episode_id} invalid transition order for critic: expected t={exp_t} but got t={t} (file={npz_path})"
            )
        expected_t_by_episode[episode_id] = int(exp_t + 1)

        n_steps = int(getattr(state_by_step, "shape")[0])
        if int(t) < 0 or int(t) >= int(n_steps):
            raise ValueError(
                f"episode_id={episode_id} t out of range for npz state_by_step: t={t} n_policy_steps={n_steps} (file={npz_path})"
            )
        state_vec = state_by_step[int(t)]
        if int(getattr(state_vec, "shape")[0]) != int(expected_state_dim):
            raise ValueError(
                f"episode_id={episode_id} t={t} state_vec shape mismatch: expected ({expected_state_dim},) got {getattr(state_vec, 'shape', None)} (file={npz_path})"
            )

        v = (
            predict_value_V(predictor, state_vec, int(t))
            if include_t
            else predict_value_V(predictor, state_vec, None)
        )
        if not isinstance(v, float):
            raise RuntimeError(
                f"episode_id={episode_id} t={t} critic returned non-scalar value: {type(v).__name__}"
            )
        out.append(float(v))

    if len(out) != len(base_records):
        raise RuntimeError("internal error: critic value_V size mismatch")
    return out


def _compute_value_from_critic(
    dataset: Mapping[str, object],
    *,
    base_records: list[JsonRecord],
    transitions_by_episode: dict[str, list[JsonRecord]],
    episode_by_id: dict[str, JsonRecord],
    critic_dir: str | Path,
) -> list[float]:
    backend_kind, critic_path, _config = _resolve_critic_backend_kind(critic_dir)
    print(f"[INFO] critic_backend={backend_kind}")

    if backend_kind == "state_only_dist_bins":
        return _compute_value_from_state_only_critic(
            dataset,
            base_records=base_records,
            transitions_by_episode=transitions_by_episode,
            episode_by_id=episode_by_id,
            critic_dir=critic_path,
        )
    if backend_kind == "multimodal_distributional_v1":
        backend_mod = importlib.import_module("work.recap.critic_vlm.backend")
        predict_labeler_values = getattr(backend_mod, "predict_labeler_values")

        return predict_labeler_values(
            dataset,
            base_records=base_records,
            transitions_by_episode=transitions_by_episode,
            episode_by_id=episode_by_id,
            critic_dir=critic_path,
        )
    raise ValueError(
        f"unknown_critic_backend: backend_kind={backend_kind!r} critic_dir={critic_path}"
    )


def _build_base_label_records(
    dataset: Mapping[str, object],
    *,
    schema_version_default: str,
    code_version_default: str,
) -> tuple[
    list[JsonRecord],
    list[float],
    list[int],
    dict[str, list[JsonRecord]],
    dict[str, JsonRecord],
]:
    episodes_raw = dataset.get("episodes")
    transitions_by_episode_raw = dataset.get("transitions_by_episode")
    if not isinstance(episodes_raw, list):
        raise ValueError(
            f"dataset['episodes'] must be a list, got: {type(episodes_raw).__name__}"
        )
    if not isinstance(transitions_by_episode_raw, dict):
        raise ValueError(
            f"dataset['transitions_by_episode'] must be a dict, got: {type(transitions_by_episode_raw).__name__}"
        )

    episodes = cast(list[JsonRecord], episodes_raw)
    transitions_by_episode = cast(
        dict[str, list[JsonRecord]], transitions_by_episode_raw
    )
    episode_by_id: dict[str, JsonRecord] = {}
    for ep in episodes:
        episode_id = ep.get("episode_id")
        if isinstance(episode_id, str) and episode_id:
            episode_by_id[episode_id] = ep

    base_records: list[JsonRecord] = []
    return_Gs: list[float] = []
    ts: list[int] = []

    episode_ids_in_order = [
        cast(str, ep["episode_id"])
        for ep in episodes
        if isinstance(ep.get("episode_id"), str)
    ]
    for episode_id in episode_ids_in_order:
        trs = transitions_by_episode.get(episode_id, [])
        if not trs:
            continue

        rewards = [
            _to_float(
                tr.get("reward_online"),
                context=f"episode_id={episode_id} t={tr.get('t')!r} reward_online",
            )
            for tr in trs
        ]
        returns = _compute_returns_mc_gamma1(rewards)
        ep = episode_by_id.get(episode_id, {})
        for tr, g in zip(trs, returns, strict=True):
            t = _to_int(
                tr.get("t"),
                context=f"episode_id={episode_id} transition.t",
            )

            schema_version = _to_str(
                tr.get(
                    "schema_version", ep.get("schema_version", schema_version_default)
                ),
            )
            code_version = _to_str(
                tr.get("code_version", ep.get("code_version", code_version_default)),
            )
            iter_tag = _to_str(
                tr.get("iter_tag", ep.get("iter_tag", "")),
            )

            prompt_raw = _to_str(
                tr.get("prompt_raw", ep.get("prompt_raw")),
            )
            is_correction = bool(tr.get("is_correction", False))

            base: JsonRecord = {
                "schema_version": schema_version,
                "code_version": code_version,
                "iter_tag": iter_tag,
                "episode_id": str(episode_id),
                "t": int(t),
                "return_G": float(g),
                "is_correction": bool(is_correction),
                "prompt_raw": str(prompt_raw),
            }
            base_records.append(base)
            return_Gs.append(float(g))
            ts.append(int(t))

    return base_records, return_Gs, ts, transitions_by_episode, episode_by_id


def build_m2_prelabels(
    dataset: Mapping[str, object],
    *,
    value_baseline: ValueBaseline = "t_mean_return",
    value_source: Literal["baseline", "critic"] = "baseline",
    critic_dir: str | Path | None = None,
    schema_version_default: str = "recap-v0",
    code_version_default: str = "unknown",
) -> list[JsonRecord]:
    base_records, return_Gs, ts, transitions_by_episode, episode_by_id = (
        _build_base_label_records(
            dataset,
            schema_version_default=schema_version_default,
            code_version_default=code_version_default,
        )
    )

    if value_source == "baseline":
        value_Vs = _compute_value_baseline(return_Gs, ts, value_baseline=value_baseline)
    elif value_source == "critic":
        if critic_dir is None:
            raise ValueError(
                "value_source='critic' requires critic_dir containing versioned critic metadata (critic_dir is None)"
            )

        value_Vs = _compute_value_from_critic(
            dataset,
            base_records=base_records,
            transitions_by_episode=transitions_by_episode,
            episode_by_id=episode_by_id,
            critic_dir=critic_dir,
        )
    else:
        raise ValueError(f"Unknown value_source: {value_source!r}")
    if len(value_Vs) != len(base_records):
        raise RuntimeError("internal error: value_V size mismatch")

    out: list[JsonRecord] = []
    for rec, v in zip(base_records, value_Vs, strict=True):
        g = _to_float(rec.get("return_G"), context="label.return_G")
        a = float(g - float(v))
        prelabel: JsonRecord = {
            "schema_version": rec["schema_version"],
            "code_version": rec["code_version"],
            "iter_tag": rec["iter_tag"],
            "episode_id": rec["episode_id"],
            "t": rec["t"],
            "return_G": rec["return_G"],
            "value_V": float(v),
            "advantage_A": float(a),
            "is_correction": bool(rec.get("is_correction")),
            "prompt_raw": _to_str(rec.get("prompt_raw")),
        }
        validate_m2_prelabel_record(prelabel)
        out.append(prelabel)

    return out


def compute_m2_epsilon_l(
    prelabels: Iterable[Mapping[str, object]],
    *,
    epsilon_strategy: EpsilonStrategy = "quantile",
    epsilon_value: float = 0.0,
    epsilon_quantile: float | None = None,
    threshold_phase: str = DEFAULT_EPSILON_THRESHOLD_PHASE,
) -> float:
    advantages: list[float] = []
    for rec in prelabels:
        validate_m2_prelabel_record(rec)
        advantages.append(
            _to_float(rec.get("advantage_A"), context="prelabel.advantage_A")
        )

    if epsilon_strategy == "const":
        return float(epsilon_value)
    if epsilon_strategy == "quantile":
        effective_quantile = resolve_epsilon_quantile(
            threshold_phase=threshold_phase,
            epsilon_quantile=epsilon_quantile,
        )
        return float(_quantile(advantages, float(effective_quantile)))
    raise ValueError(f"Unknown epsilon_strategy: {epsilon_strategy!r}")


def finalize_m2_prelabels(
    prelabels: Iterable[Mapping[str, object]],
    *,
    epsilon_l: float,
) -> list[JsonRecord]:
    epsilon_l_value = float(epsilon_l)

    out: list[JsonRecord] = []
    for rec in prelabels:
        validate_m2_prelabel_record(rec)
        a = _to_float(rec.get("advantage_A"), context="label.advantage_A")
        is_correction = bool(rec.get("is_correction"))
        indicator = 1 if float(a) > epsilon_l_value else 0
        if is_correction:
            indicator = 1

        prompt_raw = _to_str(rec.get("prompt_raw"))
        prefix = "advantage positive " if indicator == 1 else "advantage negative "
        prompt_conditioned = prefix + prompt_raw
        carrier_text_v1 = text_indicator.build_authoritative_carrier_text_v1(
            prompt_raw,
            text_indicator.indicator_mode_from_indicator_value(
                indicator,
                field_name="indicator_I",
            ),
        )

        label: JsonRecord = {
            "schema_version": rec["schema_version"],
            "code_version": rec["code_version"],
            "iter_tag": rec["iter_tag"],
            "episode_id": rec["episode_id"],
            "t": rec["t"],
            "return_G": rec["return_G"],
            "value_V": rec["value_V"],
            "advantage_A": rec["advantage_A"],
            "epsilon_l": float(epsilon_l_value),
            "indicator_I": int(indicator),
            "is_correction": bool(is_correction),
            "prompt_raw": str(prompt_raw),
            "prompt_conditioned": str(prompt_conditioned),
            text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD: str(carrier_text_v1),
        }

        if set(label.keys()) != set(REQUIRED_LABEL_KEYS):
            missing = [k for k in REQUIRED_LABEL_KEYS if k not in label]
            extra = [k for k in label.keys() if k not in REQUIRED_LABEL_KEYS]
            raise KeyError(f"label schema mismatch: missing={missing} extra={extra}")
        validate_label_record(label)
        out.append(label)

    return out


def generate_m2_labels(
    dataset: Mapping[str, object],
    *,
    value_baseline: ValueBaseline = "t_mean_return",
    value_source: Literal["baseline", "critic"] = "baseline",
    critic_dir: str | Path | None = None,
    epsilon_strategy: EpsilonStrategy = "quantile",
    epsilon_value: float = 0.0,
    epsilon_quantile: float | None = None,
    threshold_phase: str = DEFAULT_EPSILON_THRESHOLD_PHASE,
    schema_version_default: str = "recap-v0",
    code_version_default: str = "unknown",
) -> list[JsonRecord]:
    prelabels = build_m2_prelabels(
        dataset,
        value_baseline=value_baseline,
        value_source=value_source,
        critic_dir=critic_dir,
        schema_version_default=schema_version_default,
        code_version_default=code_version_default,
    )
    epsilon_l = compute_m2_epsilon_l(
        prelabels,
        epsilon_strategy=epsilon_strategy,
        epsilon_value=epsilon_value,
        epsilon_quantile=epsilon_quantile,
        threshold_phase=threshold_phase,
    )
    return finalize_m2_prelabels(prelabels, epsilon_l=epsilon_l)


__all__ = [
    "EpsilonStrategy",
    "JsonRecord",
    "REQUIRED_PRELABEL_KEYS",
    "ValueBaseline",
    "build_m2_prelabels",
    "compute_m2_epsilon_l",
    "finalize_m2_prelabels",
    "generate_m2_labels",
    "validate_m2_prelabel_record",
]
