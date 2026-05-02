from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from .json_store import (
    load_rollout_eval_v21_authority_bundle,
    load_rollout_eval_v2_authority_bundle,
    write_json,
    write_jsonl,
    write_markdown,
)


@dataclass(frozen=True)
class AuthorityBundleLoader:
    def load_rollout_eval(self, authority_dir: str | Path, *, trace_capable: bool) -> dict[str, object]:
        if trace_capable:
            return load_rollout_eval_v21_authority_bundle(authority_dir)
        return load_rollout_eval_v2_authority_bundle(authority_dir)


@dataclass(frozen=True)
class AuthorityBundleWriter:
    def write_json(self, path: Path, payload: object) -> None:
        write_json(path, payload)

    def write_jsonl(self, path: Path, rows: Sequence[Mapping[str, object]], *, sort_keys: bool = False) -> None:
        write_jsonl(path, rows, sort_keys=sort_keys)

    def write_markdown(self, path: Path, text: str) -> None:
        write_markdown(path, text)
