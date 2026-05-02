from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_ROOT = REPO_ROOT / "tests"
OPENPI_TESTS_ROOT = TESTS_ROOT / "openpi"


def _ensure_namespace_package_path(
    module_name: str, package_path: Path
) -> types.ModuleType:
    module = sys.modules.get(module_name)
    if not isinstance(module, types.ModuleType):
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
    path_attr = getattr(module, "__path__", None)
    normalized_package_path = str(package_path)
    normalized_paths = list(path_attr) if isinstance(path_attr, list) else []
    if normalized_package_path not in normalized_paths:
        normalized_paths.insert(0, normalized_package_path)
    module.__path__ = normalized_paths  # type: ignore[attr-defined]
    return module


tests_pkg = _ensure_namespace_package_path("tests", TESTS_ROOT)
openpi_tests_pkg = _ensure_namespace_package_path("tests.openpi", OPENPI_TESTS_ROOT)
setattr(tests_pkg, "openpi", openpi_tests_pkg)


import work.openpi.pipelines.recap.policy_training as recap_train_script  # noqa: E402
from work.openpi.pipelines.recap.variant_training import (  # noqa: E402
    _require_local_orbax_checkpoint_assets,
)
from work.openpi.recap.checkpoint_provenance import (  # noqa: E402
    REQUIRED_STAGE_PROVENANCE_FIELDS,
)
from work.openpi.recap.checkpoint import (  # noqa: E402
    SERVEABLE_ARTIFACT_HARDLINK_MODE,
)
from work.openpi.recap import data_transforms  # noqa: E402
from work.openpi.recap.train_config import (  # noqa: E402
    RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL,
    RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS,
)
from tests.openpi.test_recap_train_manifest import (  # noqa: E402
    patch_stage_train_dependencies,
    run_stage_train,
    write_gate_eval_manifest,
    write_minimal_recap_ready_dataset,
)


def test_checkpoint_and_export_provenance_record_task7_machine_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, export_dir = run_stage_train(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stage="recap_informative",
    )
    checkpoint_provenance = json.loads(
        (output_dir / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    variant_derivation = checkpoint_provenance["variant_derivation"]
    export_manifest = json.loads(
        (export_dir / "export_manifest.json").read_text(encoding="utf-8")
    )
    duplicates_per_positive_episode = (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    )
    expected_total_episodes = 1 + duplicates_per_positive_episode
    expected_total_rows = 2 * expected_total_episodes

    assert checkpoint_provenance["stage"] == "recap_informative"
    for field_name in REQUIRED_STAGE_PROVENANCE_FIELDS:
        assert field_name in checkpoint_provenance
        assert field_name in variant_derivation
        assert field_name in export_manifest
    assert checkpoint_provenance["critic_checkpoint_ref"].endswith("/critic/best")
    assert checkpoint_provenance["indicator_mode_train"] == "informative"
    assert checkpoint_provenance["indicator_dropout_p"] == 0.3
    assert (
        checkpoint_provenance["epsilon_source"] == "per_task_quantile:prompt_raw:q=0.7"
    )
    assert checkpoint_provenance["human_correction_override"] is True
    assert (
        checkpoint_provenance["default_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert (
        checkpoint_provenance["effective_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert checkpoint_provenance["num_train_steps_source"] == "stage_default"
    assert (
        checkpoint_provenance["default_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert (
        checkpoint_provenance["effective_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert checkpoint_provenance["save_interval_source"] == "stage_default"
    assert variant_derivation["consumer_mode"] == "informative"
    assert variant_derivation["prepared_dataset_dir"].endswith("/prepared_dataset")
    assert (
        variant_derivation["default_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert (
        variant_derivation["effective_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert variant_derivation["num_train_steps_source"] == "stage_default"
    assert (
        variant_derivation["default_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert (
        variant_derivation["effective_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert variant_derivation["save_interval_source"] == "stage_default"
    assert export_manifest["stage_provenance"]["indicator_mode_train"] == "informative"
    informative_policy = checkpoint_provenance[
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY
    ]
    assert (
        informative_policy["policy_name"]
        == data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME
    )
    assert informative_policy["applies_to_stage"] == "recap_informative"
    assert informative_policy["duplication_unit"] == "episode"
    assert (
        informative_policy["duplicates_per_positive_episode"]
        == duplicates_per_positive_episode
    )
    assert informative_policy["source_total_episodes"] == 1
    assert informative_policy["source_positive_episode_count"] == 1
    assert informative_policy["effective_total_episodes"] == expected_total_episodes
    assert (
        informative_policy["effective_positive_episode_count"]
        == expected_total_episodes
    )
    assert informative_policy["effective_total_rows"] == expected_total_rows
    assert (
        informative_policy["effective_positive_indicator_count"]
        == expected_total_episodes
    )
    assert informative_policy["effective_positive_indicator_fraction"] == 0.5
    assert (
        variant_derivation[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY]
        == informative_policy
    )
    assert (
        export_manifest["stage_provenance"][
            data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY
        ]
        == informative_policy
    )
    assert (
        export_manifest["default_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert (
        export_manifest["num_train_steps"] == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert (
        export_manifest["effective_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert export_manifest["num_train_steps_source"] == "stage_default"
    assert (
        export_manifest["default_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert export_manifest["save_interval"] == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    assert (
        export_manifest["effective_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert export_manifest["save_interval_source"] == "stage_default"
    assert export_manifest["save_interval"] == export_manifest["num_train_steps"]


def test_omit_control_export_provenance_stays_distinct_from_fixed_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    omit_output_dir, omit_export_dir = run_stage_train(
        tmp_path=tmp_path / "omit_case",
        monkeypatch=monkeypatch,
        stage="omit_control",
    )
    fixed_output_dir, fixed_export_dir = run_stage_train(
        tmp_path=tmp_path / "fixed_case",
        monkeypatch=monkeypatch,
        stage="sft_fixed_positive",
    )
    omit_checkpoint = json.loads(
        (omit_output_dir / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    fixed_checkpoint = json.loads(
        (fixed_output_dir / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    omit_export = json.loads(
        (omit_export_dir / "export_manifest.json").read_text(encoding="utf-8")
    )
    fixed_export = json.loads(
        (fixed_export_dir / "export_manifest.json").read_text(encoding="utf-8")
    )

    assert omit_checkpoint["indicator_mode_train"] == "omit"
    assert fixed_checkpoint["indicator_mode_train"] == "fixed_positive"
    assert omit_export["fixed_indicator_mode"] == "omit"
    assert fixed_export["fixed_indicator_mode"] == "positive"
    assert (
        omit_checkpoint["indicator_mode_train"]
        != fixed_checkpoint["indicator_mode_train"]
    )
    assert data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY not in omit_checkpoint
    assert data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY not in fixed_checkpoint


def test_checkpoint_and_export_provenance_record_correction_aware_reweight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, export_dir = run_stage_train(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stage="recap_informative",
        correction_segments=3,
    )
    checkpoint_provenance = json.loads(
        (output_dir / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    variant_derivation = checkpoint_provenance["variant_derivation"]
    export_manifest = json.loads(
        (export_dir / "export_manifest.json").read_text(encoding="utf-8")
    )

    informative_policy = checkpoint_provenance[
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY
    ]
    assert checkpoint_provenance["stage"] == "recap_informative"
    assert informative_policy["policy_name"] == (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME
    )
    assert informative_policy["applies_to_stage"] == "recap_informative"
    duplicates_per_positive_episode = (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    )
    assert informative_policy["enabled"] is True
    assert informative_policy["applied"] is True
    assert informative_policy["correction_aware"] is True
    assert informative_policy["duplication_unit"] == "episode"
    assert (
        informative_policy["duplicates_per_positive_episode"]
        == duplicates_per_positive_episode
    )
    assert informative_policy["source_dataset_dir"].endswith("/prepared_dataset")
    correction_signal = informative_policy["correction_signal"]
    assert correction_signal["corrections_added"] == 3
    assert correction_signal["dataset_mix_correction_segments"] == 3
    assert correction_signal["episode_level_correction_rows_present"] is True
    assert informative_policy["source_positive_correction_episode_count"] == 1
    assert (
        informative_policy["duplicated_correction_episode_count"]
        == duplicates_per_positive_episode
    )
    assert (
        variant_derivation[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY]
        == informative_policy
    )
    assert (
        export_manifest["stage_provenance"][
            data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY
        ]
        == informative_policy
    )


def _write_symlinked_real_export(root: Path) -> tuple[Path, Path]:
    runtime_root = root / "runtime"
    source_checkpoint_dir = (
        runtime_root
        / "upstream_train_checkpoints"
        / "pi05_libero"
        / "recap_informative"
        / "23"
    )
    params_dir = source_checkpoint_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    _ = (params_dir / "_METADATA").write_text('{"tree":"fixture"}\n', encoding="utf-8")
    _ = (params_dir / "manifest.ocdbt").write_text(
        "fixture-ocdbt-manifest\n",
        encoding="utf-8",
    )
    assets_dir = source_checkpoint_dir / "assets" / "physical-intelligence" / "libero"
    assets_dir.mkdir(parents=True, exist_ok=True)
    _ = (assets_dir / "norm_stats.json").write_text(
        json.dumps(
            {
                "state": {
                    "mean": [0.0] * 8,
                    "std": [1.0] * 8,
                    "q01": [0.0] * 8,
                    "q99": [1.0] * 8,
                },
                "actions": {
                    "mean": [0.0] * 7,
                    "std": [1.0] * 7,
                    "q01": [0.0] * 7,
                    "q99": [1.0] * 7,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    export_dir = runtime_root / "real_variant_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "params").symlink_to(params_dir.resolve(), target_is_directory=True)
    export_assets_parent = export_dir / "assets" / "physical-intelligence"
    export_assets_parent.mkdir(parents=True, exist_ok=True)
    (export_assets_parent / "libero").symlink_to(
        assets_dir.resolve(),
        target_is_directory=True,
    )
    _ = (export_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "openpi_real_variant_export_v1",
                "source_checkpoint_dir": str(source_checkpoint_dir),
                "export_dir": str(export_dir),
                "artifact_mirror_mode": "directory_symlink",
                "default_num_train_steps": RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS,
                "num_train_steps": RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS,
                "num_train_steps_source": "stage_default",
                "default_save_interval": RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL,
                "save_interval": RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL,
                "save_interval_source": "stage_default",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return export_dir, runtime_root


def test_repaired_checkpoint_stays_servable_after_runtime_checkpoint_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    critic_checkpoint = tmp_path / "critic" / "best"
    critic_checkpoint.mkdir(parents=True, exist_ok=True)
    prepared_dataset_dir = write_minimal_recap_ready_dataset(
        tmp_path,
        critic_checkpoint_ref=str(critic_checkpoint),
    )
    export_dir, runtime_root = _write_symlinked_real_export(tmp_path / "source")
    patch_stage_train_dependencies(
        monkeypatch=monkeypatch,
        prepared_dataset_dir=prepared_dataset_dir,
        export_dir=export_dir,
        critic_checkpoint_ref=str(critic_checkpoint),
    )
    gate_eval_manifest = write_gate_eval_manifest(tmp_path)
    output_dir = tmp_path / "recap_informative"

    rc = recap_train_script.main(
        [
            "--stage",
            "recap_informative",
            "--dataset-dir",
            str(tmp_path / "official_source"),
            "--critic-checkpoint",
            str(critic_checkpoint),
            "--output-dir",
            str(output_dir),
            "--gate-eval-manifest",
            str(gate_eval_manifest),
            "--prepared-dataset-dir",
            str(prepared_dataset_dir),
        ]
    )

    assert rc == 0
    best_dir = output_dir / "best"
    params_dir = best_dir / "params"
    assets_dir = best_dir / "assets" / "physical-intelligence" / "libero"
    source_params_metadata = (
        runtime_root
        / "upstream_train_checkpoints"
        / "pi05_libero"
        / "recap_informative"
        / "23"
        / "params"
        / "_METADATA"
    )
    source_norm_stats = (
        runtime_root
        / "upstream_train_checkpoints"
        / "pi05_libero"
        / "recap_informative"
        / "23"
        / "assets"
        / "physical-intelligence"
        / "libero"
        / "norm_stats.json"
    )
    assert params_dir.is_dir()
    assert assets_dir.is_dir()
    assert not params_dir.is_symlink()
    assert not assets_dir.is_symlink()
    assert (
        source_params_metadata.stat().st_ino == (params_dir / "_METADATA").stat().st_ino
    )
    assert (
        source_norm_stats.stat().st_ino
        == (assets_dir / "norm_stats.json").stat().st_ino
    )
    assert (params_dir / "_METADATA").stat().st_nlink >= 2
    assert (assets_dir / "norm_stats.json").stat().st_nlink >= 2

    shutil.rmtree(runtime_root / "upstream_train_checkpoints")

    _require_local_orbax_checkpoint_assets(best_dir)
    assert (params_dir / "_METADATA").is_file()
    assert (assets_dir / "norm_stats.json").is_file()

    checkpoint_payload = json.loads(
        (best_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    checkpoint_provenance = json.loads(
        (output_dir / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    expected_source_checkpoint_dir = str(export_dir.resolve())

    for payload in (checkpoint_payload, checkpoint_provenance):
        assert payload["artifact_mirror_mode"] == SERVEABLE_ARTIFACT_HARDLINK_MODE
        source_layout = payload["servable_source_layout"]
        assert source_layout["source_checkpoint_dir"] == expected_source_checkpoint_dir
        assert source_layout["source_layout_mode"] == "directory_symlink"
        assert source_layout["bundle_layout_mode"] == SERVEABLE_ARTIFACT_HARDLINK_MODE
        assert source_layout["source_params_layout"]["is_symlink"] is True
        assert source_layout["source_libero_assets_layout"]["is_symlink"] is True
        assert source_layout["bundle_params_layout"]["is_symlink"] is False
        assert source_layout["bundle_libero_assets_layout"]["is_symlink"] is False
        assert (
            source_layout["bundle_params_layout"]["materialization_mode"]
            == SERVEABLE_ARTIFACT_HARDLINK_MODE
        )
        assert (
            source_layout["bundle_libero_assets_layout"]["materialization_mode"]
            == SERVEABLE_ARTIFACT_HARDLINK_MODE
        )
        assert (
            source_layout["bundle_params_layout"]["same_filesystem_as_source"] is True
        )
        assert (
            source_layout["bundle_libero_assets_layout"]["same_filesystem_as_source"]
            is True
        )
