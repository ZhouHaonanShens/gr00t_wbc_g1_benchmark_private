from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_complete_variant(variant_dir: Path, *, episode_count: int = 2) -> None:
    rows = [
        json.dumps(
            {
                "schema_version": "v22_formal_eval_per_episode_trace_v1",
                "episode_index": index,
                "success": index % 2 == 0,
            },
            sort_keys=True,
        )
        for index in range(episode_count)
    ]
    _write_text(variant_dir / "per_episode_trace.jsonl", "\n".join(rows) + "\n")
    _write_json(
        variant_dir / "summary.json",
        {
            "schema_version": "v22_formal_eval_variant_summary_v1",
            "status": "PASS",
            "episode_count": episode_count,
        },
    )
    _write_json(variant_dir / "metric_ladder_summary.json", {"success_rate": 0.5})
    _write_json(variant_dir / "bootstrap_ci.json", {"computed": False})
    _write_text(variant_dir / "SHA256SUMS", "0" * 64 + "  summary.json\n")


def test_resume_index_skips_completed_variant_when_requested(tmp_path: Path) -> None:
    from work.openpi.eval.v22_formal_eval_runner import build_resume_index

    _write_complete_variant(tmp_path / "A", episode_count=2)

    resume_index = build_resume_index(
        tmp_path,
        ("A", "B"),
        expected_episode_count=2,
        skip_completed=True,
    )

    assert resume_index["schema_version"] == "v22_formal_eval_resume_index_v1"
    assert resume_index["completed_variants"] == ["A"]
    assert resume_index["skipped_variants"] == ["A"]
    assert resume_index["incomplete_variants"] == ["B"]
    assert resume_index["rerun_required_variants"] == ["B"]


def test_resume_index_requires_full_episode_count(tmp_path: Path) -> None:
    from work.openpi.eval.v22_formal_eval_runner import build_resume_index

    _write_complete_variant(tmp_path / "A", episode_count=1)

    resume_index = build_resume_index(
        tmp_path,
        ("A",),
        expected_episode_count=2,
        skip_completed=True,
    )

    assert resume_index["completed_variants"] == []
    assert resume_index["incomplete_variants"] == ["A"]

