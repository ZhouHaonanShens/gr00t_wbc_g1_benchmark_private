from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.stage_b import precheck_eval_protocol as p0  # noqa: E402


def _write_stage_a_fixture(root: Path) -> Path:
    stage_a = root / "stage_A"
    stage_a.mkdir(parents=True)
    with (stage_a / "pre_registration_seed_table_v1.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_role",
                "seed_index",
                "seed_value",
                "formal_lite",
                "formal_30",
                "high_variance_50",
                "notes",
            ],
        )
        writer.writeheader()
        for index in range(50):
            writer.writerow(
                {
                    "seed_role": "base",
                    "seed_index": index,
                    "seed_value": 20000 + index,
                    "formal_lite": str(index < 10).lower(),
                    "formal_30": str(index < 30).lower(),
                    "high_variance_50": "true",
                    "notes": "fixture",
                }
            )
    (stage_a / "baseline_manifest_v1.json").write_text(
        json.dumps(
            {
                "internal_baseline": {
                    "checkpoint_abs_path": "/tmp/checkpoint-6600",
                    "checkpoint_exists": True,
                    "env_name": p0.DEFAULT_ENV_NAME,
                },
                "public_baseline": {"model_repo": "nvidia/public-model"},
                "public_reproduction": {"status": "failed"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return stage_a


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_read_formal_30_seed_rows_freezes_exact_documented_set(tmp_path: Path) -> None:
    stage_a = _write_stage_a_fixture(tmp_path)

    rows, source = p0.read_formal_30_seed_rows(stage_a)

    assert source == stage_a / "pre_registration_seed_table_v1.csv"
    assert len(rows) == 30
    assert [row.seed_value for row in rows[:3]] == [20000, 20001, 20002]
    assert rows[-1].seed_value == 20029
    assert all(row.formal_30 for row in rows)


def test_build_p0_matrix_records_gated_ladder_and_runner_semantics(tmp_path: Path) -> None:
    stage_a = _write_stage_a_fixture(tmp_path)
    stage_b = tmp_path / "stage_B"

    matrix = p0.build_p0_eval_matrix(
        stage_a_dir=stage_a,
        stage_b_dir=stage_b,
        p1_status="P1_PENDING",
    )

    assert matrix["artifact_kind"] == p0.P0_ARTIFACT_KIND
    assert matrix["p1_status"] == "P1_PENDING"
    assert matrix["seed_count"] == 30
    assert len(matrix["cells"]) == 8
    by_id = {cell["cell_id"]: cell for cell in matrix["cells"]}
    assert by_id["P0a_post_recap_nenvs_1"]["status"] == "PENDING_P1_GATE"
    assert by_id["P0d_post_recap_nenvs_50"]["effective_min_episode_count"] == 50
    command = " ".join(by_id["P0c_post_recap_nenvs_5"]["representative_command"])
    assert "rollout_policy.py" in command
    assert "--n_envs 5" in command
    assert (
        by_id["P0c_post_recap_nenvs_5"]["runner_semantics"][
            "does_not_assume_gr00t_g3_formal_eval_supports_n_envs"
        ]
        is True
    )
    assert (
        by_id["P0c_post_recap_nenvs_5"]["runner_semantics"][
            "rollout_policy_effective_episode_rule"
        ]
        == "n_episodes=max(requested_n_episodes,n_envs)"
    )


def test_p0_gate_stops_on_post_recap_nenvs1_recovery() -> None:
    gate = p0.classify_p0_gate(
        p1_status="P1_PASS",
        cell_results=[
            {"cell_id": "P0a_post_recap_nenvs_1", "success_count": 9},
        ],
    )

    assert gate["decision"] == "STOP_EVAL_PROTOCOL"
    assert gate["continue_to_p2"] is False
    assert gate["continue_to_runtime_probes"] is False


def test_p0_gate_blocks_until_p1_passes() -> None:
    gate = p0.classify_p0_gate(p1_status="P1_BLOCKED")

    assert gate["decision"] == "P0_BLOCKED"
    assert gate["blocked_by"] == "P1_loader_audit"


def test_write_p0_artifacts_creates_seed_matrix_and_gate(tmp_path: Path) -> None:
    stage_a = _write_stage_a_fixture(tmp_path)
    stage_b = tmp_path / "stage_B"

    paths = p0.write_p0_artifacts(
        stage_a_dir=stage_a,
        stage_b_dir=stage_b,
        p1_status="P1_PENDING",
    )

    assert paths["seed_table"].is_file()
    assert paths["shared_seed_table"].is_file()
    assert paths["matrix_json"].is_file()
    assert paths["matrix_md"].is_file()
    assert paths["gate_json"].is_file()
    assert paths["gate_md"].is_file()
    assert paths["static_log"].is_file()

    matrix = _read_json(paths["matrix_json"])
    gate = _read_json(paths["gate_json"])
    assert matrix["seed_values"] == list(range(20000, 20030))
    assert gate["decision"] == "P0_BLOCKED"
    assert "execution_started=false" in paths["static_log"].read_text(encoding="utf-8")
