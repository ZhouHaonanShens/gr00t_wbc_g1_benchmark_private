#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Protocol, cast


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_JSON_REL = (
    "agent/artifacts/vlm_critic_manifests/task5_artifact_smoke.json"
)
DEFAULT_SAMPLE_INDEX = 0
PASS_SENTINEL = "ARTIFACT_SMOKE_OK"
FAIL_SENTINEL = "ARTIFACT_SMOKE_FAIL"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_vlm_critic_module = importlib.import_module("work.recap.critic_vlm")


class _ArtifactSmokeLike(Protocol):
    def to_json(self) -> dict[str, object]: ...


run_artifact_smoke = cast(
    Callable[..., _ArtifactSmokeLike],
    getattr(_vlm_critic_module, "run_artifact_smoke"),
)
write_json = cast(
    Callable[[Path, Mapping[str, object]], None],
    getattr(_vlm_critic_module, "write_json"),
)


def _resolve_path(repo_root: Path, raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel)
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def _emit_result(
    *,
    sentinel: str,
    output_json: Path | None,
    payload: Mapping[str, object],
) -> None:
    if output_json is not None:
        write_json(output_json, payload)
        print(f"[INFO] wrote_json: {output_json}")
    error_text = payload.get("error")
    if isinstance(error_text, str) and error_text.strip():
        print(f"[ERROR] {error_text}", file=sys.stderr)
    print(f"SENTINEL:{sentinel}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="43b_vlm_critic_artifact_smoke.py",
        description="Run the versioned multimodal critic artifact smoke API on a single dataset sample.",
    )
    _ = parser.add_argument("--critic-dir", type=str, required=True)
    _ = parser.add_argument("--dataset-path", type=str, required=True)
    _ = parser.add_argument(
        "--sample-index", type=int, default=int(DEFAULT_SAMPLE_INDEX)
    )
    _ = parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    critic_dir = _resolve_path(REPO_ROOT, str(args.critic_dir), default_rel="")
    dataset_path = _resolve_path(REPO_ROOT, str(args.dataset_path), default_rel="")
    output_json_raw = str(args.output_json or "").strip()
    output_json = (
        _resolve_path(REPO_ROOT, output_json_raw, default_rel=DEFAULT_OUTPUT_JSON_REL)
        if output_json_raw
        else None
    )

    try:
        smoke = run_artifact_smoke(
            critic_dir=critic_dir,
            dataset_path=dataset_path,
            sample_index=int(args.sample_index),
        )
        smoke_json = smoke.to_json()
        result = {"pass": True, **smoke_json}
        _emit_result(sentinel=PASS_SENTINEL, output_json=output_json, payload=result)
        return 0
    except Exception as exc:
        failure = {
            "pass": False,
            "critic_dir": str(critic_dir),
            "dataset_path": str(dataset_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _emit_result(sentinel=FAIL_SENTINEL, output_json=output_json, payload=failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
