from __future__ import annotations

from pathlib import Path
import sys
from typing import cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap import dataset_aggregation  # noqa: E402
import work.openpi.pipelines.recap.iteration as iteration_script  # noqa: E402
from tests.openpi.test_recap_collection_schema import (  # noqa: E402
    patch_rollout_eval,
    write_demo_source,
    write_policy_checkpoint,
)


def test_iteration_script_emits_machine_readable_iteration_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_dir = write_demo_source(tmp_path)
    critic_checkpoint_ref = str((tmp_path / "critic" / "best").resolve())
    policy_checkpoint = write_policy_checkpoint(
        tmp_path,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    patch_rollout_eval(
        monkeypatch=monkeypatch,
        critic_checkpoint_ref=critic_checkpoint_ref,
        success_pattern=(True, False, True, False),
    )
    output_dir = tmp_path / "iter0"

    manifest_payload = iteration_script.run_iteration(
        iteration_script.IterationConfig(
            iter_id="iter0",
            seed_policy_checkpoint=policy_checkpoint,
            critic_checkpoint=None,
            indicator_mode="positive",
            task_suite_name="libero_spatial",
            task_ids="0,1",
            episodes=4,
            output_dir=output_dir,
            demo_dir=demo_dir,
            correction_dir=None,
            critic_config=None,
            repaired_matrix_summary_path=(
                REPO_ROOT / "agent" / "artifacts" / "openpi_recap_v1" / "repaired_matrix_summary.json"
            ),
            tracked_summary_path=(
                REPO_ROOT / "agent" / "exchange" / "openpi_recap_iteration_smoke_summary_v1.md"
            ),
        )
    )

    assert manifest_payload["iter_id"] == "iter0"
    manifest = dataset_aggregation.read_json(
        output_dir / dataset_aggregation.ITERATION_MANIFEST_NAME
    )
    dataset_mix = cast(dict[str, object], manifest["dataset_mix"])
    policy_lineage = cast(dict[str, object], manifest["policy_lineage"])
    stage_lineage = cast(dict[str, object], manifest["stage_lineage"])
    canonical_source = cast(
        dict[str, object], manifest["canonical_demo_source_root_proof"]
    )

    assert (
        manifest["schema_version"]
        == dataset_aggregation.ITERATION_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["iter_id"] == "iter0"
    assert manifest["policy_checkpoint_ref"] == str(policy_checkpoint)
    assert manifest["critic_checkpoint_ref"] == critic_checkpoint_ref
    assert manifest["episodes_added"] == 4
    assert manifest["corrections_added"] == 0
    assert dataset_mix["autonomous"] == {"episodes": 4, "successes": 2, "failures": 2}
    assert canonical_source["status"] == "ready"
    assert policy_lineage["policy_stage"] == "sft_fixed_positive"
    assert stage_lineage["collect_route_id"] == dataset_aggregation.COLLECTION_ROUTE_ID
    assert (
        stage_lineage["merge_route_id"] == dataset_aggregation.MERGED_DATASET_ROUTE_ID
    )
