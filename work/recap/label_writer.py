from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import cast

from work.recap import text_indicator


M2_LABELS_SUBDIR_NAME = "m2_labels"
LABELS_JSONL_NAME = "labels.jsonl"
STATS_JSON_NAME = "stats.json"
PRIVATE_PRELABELS_SUBDIR_NAME = "_private_prelabels"

REQUIRED_LABEL_KEYS = [
    "schema_version",
    "code_version",
    "iter_tag",
    "episode_id",
    "t",
    "return_G",
    "value_V",
    "advantage_A",
    "epsilon_l",
    "indicator_I",
    "is_correction",
    "prompt_raw",
    "prompt_conditioned",
    text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD,
]


_REQUIRED_LABEL_KEYS_SET = frozenset(REQUIRED_LABEL_KEYS)

JsonRecord = dict[str, object]


def validate_label_record(obj: object) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"label record must be a dict, got: {type(obj).__name__}")

    record = cast(dict[str, object], obj)

    keys = set(record.keys())
    missing = _REQUIRED_LABEL_KEYS_SET - keys
    extra = keys - _REQUIRED_LABEL_KEYS_SET
    if missing or extra:
        missing_s = ", ".join(sorted(missing)) if missing else "(none)"
        extra_s = ", ".join(sorted(extra)) if extra else "(none)"
        raise ValueError(
            f"label record keys mismatch; missing=[{missing_s}] extra=[{extra_s}]"
        )

    indicator = record["indicator_I"]
    if indicator not in (0, 1):
        raise ValueError(f"indicator_I must be 0 or 1, got: {indicator!r}")

    if record.get("is_correction") and indicator != 1:
        raise ValueError(
            f"invalid correction label: is_correction is truthy but indicator_I != 1 (indicator_I={indicator!r})"
        )

    prompt_raw = text_indicator.require_prompt_raw(
        record.get("prompt_raw"),
        field_name="prompt_raw",
    )
    carrier_text_v1 = text_indicator.require_prompt_raw(
        record.get(text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD),
        field_name=text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD,
    )
    indicator_mode = text_indicator.indicator_mode_from_indicator_value(
        indicator,
        field_name="indicator_I",
    )
    _ = text_indicator.require_authoritative_carrier_text_v1(
        carrier_text_v1,
        prompt_raw=prompt_raw,
        indicator_mode=indicator_mode,
    )


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


def _require_existing_dir(path: Path, *, kind: str) -> None:
    if not path.exists():
        raise ValueError(f"Missing {kind} directory: {path}")
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path} ({kind})")


def _atomic_write_json(path: Path, obj: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        _ = f.write("\n")
    _ = tmp.replace(path)
    return path


def _atomic_write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, object]],
    *,
    validator: Callable[[object], None] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            if validator is not None:
                validator(rec)
            line = json.dumps(
                rec,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            _ = f.write(line + "\n")
    _ = tmp.replace(path)
    return path


def _read_jsonl_records(path: Path) -> list[JsonRecord]:
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
            for key, value in obj_raw.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"Invalid key type in {path} at line {lineno}: expected str key, got {type(key).__name__}"
                    )
                obj[key] = value
            out.append(obj)
    return out


def _normalize_private_shard_name(shard_name: str) -> str:
    candidate = shard_name.strip()
    if not candidate:
        raise ValueError("shard_name must be a non-empty string")
    p = Path(candidate)
    if p.is_absolute() or len(p.parts) != 1 or candidate in {".", ".."}:
        raise ValueError(f"Invalid shard_name: {shard_name!r}")
    if candidate.endswith(".jsonl"):
        return candidate
    return f"{candidate}.jsonl"


def m2_labels_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    _require_existing_dir(out, kind="output")
    d = out / M2_LABELS_SUBDIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def m2_private_prelabels_dir(output_dir: str | Path) -> Path:
    d = m2_labels_dir(output_dir) / PRIVATE_PRELABELS_SUBDIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def m2_private_prelabel_shard_path(output_dir: str | Path, *, shard_name: str) -> Path:
    return m2_private_prelabels_dir(output_dir) / _normalize_private_shard_name(
        shard_name
    )


def write_m2_private_prelabel_shard_jsonl(
    output_dir: str | Path,
    *,
    shard_name: str,
    records: Iterable[Mapping[str, object]],
) -> Path:
    out_path = m2_private_prelabel_shard_path(output_dir, shard_name=shard_name)
    return _atomic_write_jsonl(out_path, records)


def list_m2_private_prelabel_shard_paths(output_dir: str | Path) -> list[Path]:
    shards_dir = m2_private_prelabels_dir(output_dir)
    return sorted(p for p in shards_dir.glob("*.jsonl") if p.is_file())


def merge_m2_private_prelabel_shards(output_dir: str | Path) -> list[JsonRecord]:
    out: list[JsonRecord] = []
    for shard_path in list_m2_private_prelabel_shard_paths(output_dir):
        out.extend(_read_jsonl_records(shard_path))
    return out


def compute_m2_label_stats(
    labels: Iterable[Mapping[str, object]],
    *,
    epsilon_strategy: str | None = None,
    epsilon_value: float | None = None,
) -> dict[str, object]:
    labels_list = labels if isinstance(labels, list) else list(labels)
    if not labels_list:
        raise ValueError("No labels provided")

    episode_ids: set[str] = set()
    epsilons: set[float] = set()
    advantages: list[float] = []
    pos = 0

    for idx, rec in enumerate(labels_list, start=1):
        validate_label_record(rec)
        episode_ids.add(str(rec.get("episode_id", "")))
        indicator_raw = rec.get("indicator_I")

        indicator_i: int | None
        if isinstance(indicator_raw, bool):
            indicator_i = 1 if indicator_raw else 0
        elif isinstance(indicator_raw, int):
            indicator_i = indicator_raw
        elif isinstance(indicator_raw, float):
            indicator_i = int(indicator_raw) if indicator_raw.is_integer() else None
        elif isinstance(indicator_raw, str):
            try:
                indicator_i = int(indicator_raw)
            except ValueError:
                indicator_i = None
        else:
            indicator_i = None

        if indicator_i not in (0, 1):
            raise ValueError(
                f"Invalid indicator_I at record#{idx}: expected 0/1, got {indicator_raw!r}"
            )
        pos += int(indicator_i)

        advantages.append(
            _to_float(rec.get("advantage_A"), context=f"record#{idx} advantage_A")
        )
        epsilons.add(_to_float(rec.get("epsilon_l"), context=f"record#{idx} epsilon_l"))

    n_transitions = int(len(labels_list))
    n_episodes = int(len(episode_ids))
    pos_ratio = float(pos) / float(n_transitions)

    advantage_min = float(min(advantages))
    advantage_max = float(max(advantages))
    advantage_mean = float(sum(advantages) / float(len(advantages)))

    if epsilon_value is None:
        if len(epsilons) != 1:
            eps_preview = ", ".join(str(e) for e in sorted(epsilons)[:5])
            more = "" if len(epsilons) <= 5 else f" (+{len(epsilons) - 5} more)"
            raise ValueError(
                f"Cannot infer epsilon_value: labels contain multiple epsilon_l values: {eps_preview}{more}"
            )
        epsilon_value = float(next(iter(epsilons)))

    return {
        "n_transitions": int(n_transitions),
        "n_episodes": int(n_episodes),
        "epsilon_strategy": str(epsilon_strategy)
        if epsilon_strategy is not None
        else "unknown",
        "epsilon_value": float(epsilon_value),
        "pos_ratio": float(pos_ratio),
        "advantage_mean": float(advantage_mean),
        "advantage_min": float(advantage_min),
        "advantage_max": float(advantage_max),
    }


def write_m2_labels_jsonl(
    output_dir: str | Path,
    labels: Iterable[Mapping[str, object]],
) -> Path:
    labels_dir = m2_labels_dir(output_dir)
    out_path = labels_dir / LABELS_JSONL_NAME
    return _atomic_write_jsonl(out_path, labels, validator=validate_label_record)


def write_m2_stats_json(output_dir: str | Path, stats: Mapping[str, object]) -> Path:
    labels_dir = m2_labels_dir(output_dir)
    out_path = labels_dir / STATS_JSON_NAME
    return _atomic_write_json(out_path, stats)


def write_m2_label_outputs(
    output_dir: str | Path,
    labels: Iterable[Mapping[str, object]],
    *,
    epsilon_strategy: str | None = None,
    epsilon_value: float | None = None,
) -> dict[str, object]:
    labels_list = labels if isinstance(labels, list) else list(labels)
    stats = compute_m2_label_stats(
        labels_list, epsilon_strategy=epsilon_strategy, epsilon_value=epsilon_value
    )
    _ = write_m2_labels_jsonl(output_dir, labels_list)
    _ = write_m2_stats_json(output_dir, stats)
    return stats


__all__ = [
    "LABELS_JSONL_NAME",
    "M2_LABELS_SUBDIR_NAME",
    "PRIVATE_PRELABELS_SUBDIR_NAME",
    "REQUIRED_LABEL_KEYS",
    "STATS_JSON_NAME",
    "compute_m2_label_stats",
    "list_m2_private_prelabel_shard_paths",
    "m2_private_prelabel_shard_path",
    "m2_private_prelabels_dir",
    "merge_m2_private_prelabel_shards",
    "m2_labels_dir",
    "validate_label_record",
    "write_m2_label_outputs",
    "write_m2_labels_jsonl",
    "write_m2_private_prelabel_shard_jsonl",
    "write_m2_stats_json",
]
