from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import cast

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import work.openpi.sources.libero_official.relabels as relabels_module  # noqa: E402
from work.openpi.sources.libero_official.relabels import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    EPSILON_QUANTILE,
    REQUIRED_OUTPUT_LABEL_COLUMNS,
    ROUTE_ID,
    SUCCESS_EXIT_CODE,
    build_label_plan,
    main,
    materialize_dataset,
)


DOC = REPO_ROOT / "agent/exchange/openpi_libero_official_8d_relabel_contract.md"
OFFICIAL_DIR = (
    REPO_ROOT
    / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mapping(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError(f"expected dict, got {type(raw).__name__}")
    return cast(dict[str, object], raw)


def _episode_lengths(dataset_dir: str | Path) -> dict[int, int]:
    episodes_path = Path(dataset_dir) / "meta" / "episodes.jsonl"
    out: dict[int, int] = {}
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = _mapping(json.loads(line))
        out[int(cast(int, row["episode_index"]))] = int(cast(int, row["length"]))
    return out


def _deterministic_critic_values(
    *,
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    episode_indices: list[int] | tuple[int, ...],
) -> dict[int, list[float]]:
    del checkpoint_dir
    lengths = _episode_lengths(dataset_dir)
    values_by_episode: dict[int, list[float]] = {}
    for episode_index in episode_indices:
        length = lengths[int(episode_index)]
        returns = [float(t - (length - 1)) for t in range(length)]
        values_by_episode[int(episode_index)] = [
            return_g - (1.0 if step_index % 2 == 0 else -1.0)
            for step_index, return_g in enumerate(returns)
        ]
    return values_by_episode


@pytest.fixture
def deterministic_critic_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    critic_dir = tmp_path / "deterministic_critic_checkpoint"
    processor_dir = critic_dir / "processor"
    processor_dir.mkdir(parents=True)
    _write_json(
        critic_dir / "config.json",
        {
            "artifact_version": "multimodal_distributional_v1",
            "critic_type": "multimodal_distributional_v1",
            "base_model": "repo-local deterministic official-8d test critic",
            "value_scale": "raw_return",
            "upgrade_pending": "none_test_fixture_only",
            "smoke_backend": "synthetic_checker_v1",
        },
    )
    _write_json(
        processor_dir / "processor_config.json",
        {
            "task_text_field": "prompt_raw",
            "frame_policy": "current_step_index",
            "allow_future_frames": False,
            "side_channels": [],
        },
    )
    _write_json(
        critic_dir / "model.pt",
        {
            "bias": 0.0,
            "text_scale": 0.0,
            "step_scale": 0.0,
            "frame_scale": 0.0,
            "temperature": 1.0,
        },
    )
    _write_json(critic_dir / "bin_centers.json", {"bin_centers": [-1.0, 0.0, 1.0]})
    _write_json(critic_dir / "provenance.json", {"fixture": "official_8d_cpu_only"})
    _write_json(critic_dir / "metrics.json", {"fixture": "official_8d_cpu_only"})
    _write_json(
        critic_dir / "split_manifest_ref.json", {"fixture": "official_8d_cpu_only"}
    )
    monkeypatch.setattr(
        relabels_module,
        "build_episode_value_predictions",
        _deterministic_critic_values,
    )
    return critic_dir


def test_relabel_contract_doc_freezes_direct_official_8d_scope() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO official/native 8D recap relabel 合同",
        "route_id=official_native_8d_recap_relabels_v1",
        "official/native 8D direct relabel",
        "not state-token training",
        "no 43D WBC state import",
        "no 35D action import",
        "no weak-key cross-dataset join",
        "reward scheme = success_demo_terminal_only_v1",
        "return_G = Monte Carlo gamma=1 over derived rewards",
        "value_V = t_mean_return baseline over the same official/native 8D population",
        "epsilon_l = quantile(advantage_A, q=0.7)",
        "advantage_input uses sign_aware_quantile_by_sign_v1",
        "observation.state.shape=[8]",
        "action.shape=[7]",
    ]
    for item in required:
        assert item in text, f"missing relabel contract item: {item}"


def test_build_label_plan_keeps_official_shapes_and_generates_recap_scales(
    deterministic_critic_checkpoint: Path,
) -> None:
    plan = build_label_plan(
        OFFICIAL_DIR,
        episode_limit=4,
        critic_checkpoint_dir=deterministic_critic_checkpoint,
    )

    source = plan["source"]
    assert getattr(source, "state_dim") == 8
    assert getattr(source, "action_dim") == 7
    assert plan["selected_episode_count"] == 4
    assert plan["selected_frame_count"] == 1128
    threshold_estimates = _mapping(plan["threshold_estimates"])
    epsilon_l = max(
        float(_mapping(raw_estimate)["epsilon_l"])
        for raw_estimate in threshold_estimates.values()
    )
    raw_summary = _mapping(plan["raw_summary"])
    assert float(cast(float, raw_summary["min"])) <= epsilon_l
    assert epsilon_l <= float(cast(float, raw_summary["max"]))
    scale_summary = _mapping(plan["scale_metadata"])
    assert scale_summary["positive_scale"] is not None
    assert scale_summary["negative_scale_abs"] is not None
    contract = _mapping(plan["advantage_contract"])
    assert contract["task_text_field"] == "prompt_raw"
    assert contract["value_source"] == "critic"
    assert contract["critic_checkpoint_ref"] == str(
        deterministic_critic_checkpoint.resolve()
    )


def test_materializer_creates_canonical_recap_columns_on_subset(
    tmp_path: Path,
    deterministic_critic_checkpoint: Path,
) -> None:
    output_dir = tmp_path / "physical_intelligence_libero_official_8d_recap_relabels_v1"

    report = materialize_dataset(
        OFFICIAL_DIR,
        output_dir,
        episode_limit=4,
        critic_checkpoint_dir=deterministic_critic_checkpoint,
    )

    report_mapping = _mapping(report)
    assert report_mapping["route_id"] == ROUTE_ID
    assert report_mapping["final_status"] == "materialized"
    assert report_mapping["state_dim"] == 8
    assert report_mapping["action_dim"] == 7
    assert report_mapping["epsilon_quantile"] == EPSILON_QUANTILE
    assert set(cast(list[str], report_mapping["required_output_label_columns"])) == set(
        REQUIRED_OUTPUT_LABEL_COLUMNS
    )

    info = json.loads((output_dir / "meta/info.json").read_text(encoding="utf-8"))
    assert info["schema_version"] == DATASET_SCHEMA_VERSION
    assert info["route_id"] == ROUTE_ID
    assert info["features"]["observation.state"]["shape"] == [8]
    assert info["features"]["action"]["shape"] == [7]
    assert (
        info["recap_label_recipe"]["reward_scheme"] == "success_demo_terminal_only_v1"
    )
    assert (
        info["recap_label_recipe"]["value_baseline"]
        == "critic_raw_return_adapter_v1"
    )

    df = pd.read_parquet(output_dir / "data/chunk-000/episode_000000.parquet")
    required_columns = {
        "observation.images.ego_view",
        "observation.images.wrist_view",
        "observation.state",
        "action",
        "annotation.human.task_description",
        "annotation.human.action.task_description",
        *REQUIRED_OUTPUT_LABEL_COLUMNS,
    }
    assert required_columns.issubset(df.columns)
    assert tuple(df.iloc[0]["observation.state"].shape) == (8,)
    assert tuple(df.iloc[0]["action"].shape) == (7,)
    assert df["recap_m2.t"].tolist() == list(range(len(df)))
    assert float(df.iloc[0]["recap_m2.return_G"]) == -213.0
    assert float(df.iloc[-1]["recap_m2.return_G"]) == 0.0
    assert bool(df["recap_m2.indicator_I"].isin([0, 1]).all())
    assert bool(df["recap_m2.advantage_input"].between(-1.0, 1.0).all())
    assert all(
        isinstance(value, str) and value for value in df["recap_m2.prompt_raw"].tolist()
    )
    assert all(
        isinstance(value, str)
        and value.startswith(raw_prompt)
        and value.endswith(("\nAdvantage: positive", "\nAdvantage: negative"))
        for value, raw_prompt in zip(
            df["recap_m2.prompt_conditioned"].tolist(),
            df["recap_m2.prompt_raw"].tolist(),
            strict=True,
        )
    )


def test_cli_materializer_returns_zero_on_subset_independent_smoke(
    tmp_path: Path,
    deterministic_critic_checkpoint: Path,
) -> None:
    output_dir = tmp_path / "cli_dataset"
    rc = main(
        [
            "--official-dataset-dir",
            str(OFFICIAL_DIR),
            "--output-dir",
            str(output_dir),
            "--episode-limit",
            "4",
            "--critic-checkpoint-dir",
            str(deterministic_critic_checkpoint),
        ]
    )
    assert rc == SUCCESS_EXIT_CODE
    report = _mapping(
        json.loads(
            (output_dir / "materialization_report.json").read_text(encoding="utf-8")
        )
    )
    assert report["route_id"] == ROUTE_ID
    assert report["final_status"] == "materialized"


def test_cli_materializer_requires_explicit_critic_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli_dataset"
    with pytest.raises(SystemExit) as exc_info:
        _ = main(
            [
                "--official-dataset-dir",
                str(OFFICIAL_DIR),
                "--output-dir",
                str(output_dir),
                "--episode-limit",
                "1",
            ]
        )
    assert exc_info.value.code == 2
