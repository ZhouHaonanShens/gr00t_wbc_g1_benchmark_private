from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .json_store import read_json, read_jsonl


@dataclass(frozen=True)
class RolloutSourceLoader:
    """Read rollout-source rows and summaries from the canonical JSON surfaces.

    The loader intentionally stays thin: it performs structured I/O only and
    leaves semantic validation to the workflow or protocol layer above it.
    """

    def read_rows(self, path: Path) -> list[dict[str, object]]:
        return read_jsonl(path)

    def read_summary(self, path: Path) -> dict[str, object]:
        return read_json(path)
