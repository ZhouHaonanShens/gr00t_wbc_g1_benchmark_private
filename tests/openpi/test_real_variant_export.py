from __future__ import annotations

import errno
import pickle
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.real_variant_export import (
    DEFAULT_LOCAL_SAFE_CUDA_VISIBLE_DEVICES,
    DEFAULT_LOCAL_SAFE_BATCH_SIZE,
    DEFAULT_LOCAL_SAFE_FSDP_DEVICES,
    DEFAULT_LOCAL_SAFE_JAX_PLATFORM,
    DEFAULT_LOCAL_SAFE_NUM_TRAIN_STEPS,
    DEFAULT_LOCAL_SAFE_NUM_WORKERS,
    DEFAULT_LOCAL_SAFE_SAVE_INTERVAL,
    DEFAULT_LOCAL_SAFE_XLA_ALLOCATOR,
    DEFAULT_LOCAL_SAFE_XLA_PREALLOCATE,
    SUBPROCESS_CACHE_ROOT_DIRNAME,
    TRAIN_DEFAULT_SAVE_INTERVAL_ENV,
    TRAIN_DEFAULT_NUM_STEPS_ENV,
    TRAIN_SAVE_INTERVAL_ENV,
    TRAIN_SAVE_INTERVAL_SOURCE_ENV,
    TRAIN_NUM_STEPS_ENV,
    TRAIN_NUM_STEPS_SOURCE_ENV,
    RealVariantExportRequest,
    ResumeCheckpointPlan,
    _install_resume_bootstrap_step_hook,
    _resolve_resume_checkpoint_plan,
    VariantPromptTransform,
    _validate_resume_target_step,
    _build_subprocess_env,
    _copytree,
    _resolve_train_resource_settings,
)
import work.openpi.recap.checkpoint as checkpoint_mod
from work.openpi.recap.checkpoint import (
    SERVEABLE_ARTIFACT_COPY_MODE,
    SERVEABLE_ARTIFACT_HARDLINK_MODE,
    _replace_tree_with_durable_links_or_copy,
)
from work.openpi.prompting.routes import (
    FIXEDADV_CONSTANT_CONSUMER_MODE,
    RECAP_RELABEL_CONSUMER_MODE,
    SHUFFLED_ADV_DIAG_CONSUMER_MODE,
)


def _assert_expected_subprocess_cache_env(
    env: dict[str, str], runtime_dir: Path
) -> None:
    cache_root = runtime_dir.resolve() / SUBPROCESS_CACHE_ROOT_DIRNAME
    assert env["HF_HOME"] == str(cache_root / "hf_home")
    assert env["HF_DATASETS_CACHE"] == str(cache_root / "hf_home" / "datasets")
    assert env["TRANSFORMERS_CACHE"] == str(cache_root / "hf_home" / "transformers")
    assert env["TMPDIR"] == str(cache_root / "tmp")
    assert Path(env["HF_HOME"]).is_dir()
    assert Path(env["HF_DATASETS_CACHE"]).is_dir()
    assert Path(env["TRANSFORMERS_CACHE"]).is_dir()
    assert Path(env["TMPDIR"]).is_dir()


def _prompt_text(transform: VariantPromptTransform, *, indicator: int) -> str:
    payload = transform(
        {
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": indicator,
        }
    )
    prompt = payload["prompt"]
    return str(prompt.item())


def test_variant_prompt_transform_is_picklable_for_fixedadv_omit() -> None:
    transform = VariantPromptTransform(
        consumer_mode=FIXEDADV_CONSTANT_CONSUMER_MODE,
        fixed_indicator_mode="omit",
    )

    restored = pickle.loads(pickle.dumps(transform))

    assert _prompt_text(restored, indicator=1) == "put the bowl on the plate"
    assert _prompt_text(restored, indicator=0) == "put the bowl on the plate"


def test_variant_prompt_transform_preserves_informative_adv_indicator_carrier() -> None:
    transform = VariantPromptTransform(
        consumer_mode=RECAP_RELABEL_CONSUMER_MODE,
        fixed_indicator_mode=None,
    )

    restored = pickle.loads(pickle.dumps(transform))

    positive_prompt = _prompt_text(restored, indicator=1)
    negative_prompt = _prompt_text(restored, indicator=0)

    assert positive_prompt != negative_prompt
    assert positive_prompt.startswith("put the bowl on the plate")
    assert negative_prompt.startswith("put the bowl on the plate")
    assert positive_prompt.endswith("Advantage: positive")
    assert negative_prompt.endswith("Advantage: negative")


def test_variant_prompt_transform_accepts_scalar_tensor_indicator_value() -> None:
    torch = pytest.importorskip("torch")
    transform = VariantPromptTransform(
        consumer_mode=RECAP_RELABEL_CONSUMER_MODE,
        fixed_indicator_mode=None,
    )

    payload = transform(
        {
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": torch.tensor(1),
        }
    )

    assert str(payload["prompt"].item()).endswith("Advantage: positive")


def test_variant_prompt_transform_shuffled_diag_ignores_raw_indicator_for_same_sample() -> (
    None
):
    transform = VariantPromptTransform(
        consumer_mode=SHUFFLED_ADV_DIAG_CONSUMER_MODE,
        fixed_indicator_mode=None,
    )

    prompt_with_positive_label = transform(
        {
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": 1,
            "observation/state": [0.1, 0.2, 0.3],
        }
    )
    prompt_with_negative_label = transform(
        {
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": 0,
            "observation/state": [0.1, 0.2, 0.3],
        }
    )

    assert str(prompt_with_positive_label["prompt"].item()) == str(
        prompt_with_negative_label["prompt"].item()
    )


def test_resolve_train_resource_settings_uses_local_safe_defaults() -> None:
    resource_settings = _resolve_train_resource_settings(
        SimpleNamespace(fsdp_devices=8)
    )

    assert resource_settings == {
        "num_train_steps": DEFAULT_LOCAL_SAFE_NUM_TRAIN_STEPS,
        "batch_size": DEFAULT_LOCAL_SAFE_BATCH_SIZE,
        "save_interval": DEFAULT_LOCAL_SAFE_SAVE_INTERVAL,
        "num_workers": DEFAULT_LOCAL_SAFE_NUM_WORKERS,
        "fsdp_devices": DEFAULT_LOCAL_SAFE_FSDP_DEVICES,
    }


@pytest.mark.parametrize(
    ("env_name", "env_value", "field_name"),
    [
        ("OPENPI_VARIANT_TRAIN_NUM_STEPS", "7", "num_train_steps"),
        ("OPENPI_VARIANT_TRAIN_BATCH_SIZE", "3", "batch_size"),
        ("OPENPI_VARIANT_TRAIN_SAVE_INTERVAL", "5", "save_interval"),
        ("OPENPI_VARIANT_TRAIN_NUM_WORKERS", "2", "num_workers"),
        ("OPENPI_VARIANT_TRAIN_FSDP_DEVICES", "4", "fsdp_devices"),
    ],
)
def test_resolve_train_resource_settings_honors_env_overrides(
    monkeypatch,
    env_name: str,
    env_value: str,
    field_name: str,
) -> None:
    monkeypatch.setenv(env_name, env_value)

    resource_settings = _resolve_train_resource_settings(
        SimpleNamespace(fsdp_devices=1)
    )

    assert resource_settings[field_name] == int(env_value)


def test_build_subprocess_env_defaults_to_cpu_safe_runtime(monkeypatch, tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    for env_name in (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "CUDA_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
    ):
        monkeypatch.delenv(env_name, raising=False)

    env = _build_subprocess_env(
        RealVariantExportRequest(
            variant="fixedadv_control",
            variant_name="fixedadv_relabel8d_control_v1",
            dataset_dir=tmp_path / "dataset",
            runtime_dir=runtime_dir,
            consumer_mode=FIXEDADV_CONSTANT_CONSUMER_MODE,
            fixed_indicator_mode="omit",
            default_num_train_steps=5,
            default_save_interval=5,
        )
    )

    assert env["JAX_PLATFORMS"] == DEFAULT_LOCAL_SAFE_JAX_PLATFORM
    assert env["JAX_PLATFORM_NAME"] == DEFAULT_LOCAL_SAFE_JAX_PLATFORM
    assert env["CUDA_VISIBLE_DEVICES"] == DEFAULT_LOCAL_SAFE_CUDA_VISIBLE_DEVICES
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == DEFAULT_LOCAL_SAFE_XLA_PREALLOCATE
    assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == DEFAULT_LOCAL_SAFE_XLA_ALLOCATOR
    assert env[TRAIN_NUM_STEPS_ENV] == "5"
    assert env[TRAIN_DEFAULT_NUM_STEPS_ENV] == "5"
    assert env[TRAIN_NUM_STEPS_SOURCE_ENV] == "stage_default"
    assert env[TRAIN_SAVE_INTERVAL_ENV] == "5"
    assert env[TRAIN_DEFAULT_SAVE_INTERVAL_ENV] == "5"
    assert env[TRAIN_SAVE_INTERVAL_SOURCE_ENV] == "stage_default"
    _assert_expected_subprocess_cache_env(env, runtime_dir)


def test_build_subprocess_env_preserves_explicit_runtime_overrides(monkeypatch, tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    monkeypatch.setenv(TRAIN_NUM_STEPS_ENV, "11")
    monkeypatch.setenv(TRAIN_SAVE_INTERVAL_ENV, "13")

    env = _build_subprocess_env(
        RealVariantExportRequest(
            variant="fixedadv_control",
            variant_name="fixedadv_relabel8d_control_v1",
            dataset_dir=tmp_path / "dataset",
            runtime_dir=runtime_dir,
            consumer_mode=FIXEDADV_CONSTANT_CONSUMER_MODE,
            fixed_indicator_mode="omit",
            default_num_train_steps=5,
            default_save_interval=5,
        )
    )

    assert env["JAX_PLATFORMS"] == "gpu"
    assert env["JAX_PLATFORM_NAME"] == "gpu"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
    assert env[TRAIN_NUM_STEPS_ENV] == "11"
    assert env[TRAIN_DEFAULT_NUM_STEPS_ENV] == "5"
    assert env[TRAIN_NUM_STEPS_SOURCE_ENV] == "env_override"
    assert env[TRAIN_SAVE_INTERVAL_ENV] == "13"
    assert env[TRAIN_DEFAULT_SAVE_INTERVAL_ENV] == "5"
    assert env[TRAIN_SAVE_INTERVAL_SOURCE_ENV] == "env_override"
    _assert_expected_subprocess_cache_env(env, runtime_dir)


def test_resolve_resume_checkpoint_plan_prefers_native_resume_when_train_state_exists(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "runtime" / "upstream_train_checkpoints" / "pi0_libero" / "variant"
    train_state_dir = checkpoint_dir / "200" / "train_state"
    params_dir = checkpoint_dir / "200" / "params"
    train_state_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    plan = _resolve_resume_checkpoint_plan(
        checkpoint_dir=checkpoint_dir,
        runtime_dir=tmp_path / "runtime",
        requested_resume=True,
    )

    assert plan.mode == "native_resume"
    assert plan.latest_step == 200
    assert plan.resume_state_step == 201
    assert plan.latest_checkpoint_dir == checkpoint_dir / "200"
    assert plan.seed_params_path is None


def test_resolve_resume_checkpoint_plan_rejects_train_state_without_params(
    tmp_path: Path,
) -> None:
    import pytest

    checkpoint_dir = tmp_path / "runtime" / "upstream_train_checkpoints" / "pi0_libero" / "variant"
    train_state_dir = checkpoint_dir / "200" / "train_state"
    train_state_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="missing params required for restore/bootstrap"):
        _resolve_resume_checkpoint_plan(
            checkpoint_dir=checkpoint_dir,
            runtime_dir=tmp_path / "runtime",
            requested_resume=True,
        )


def test_resolve_resume_checkpoint_plan_bootstraps_from_params_only_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "runtime" / "upstream_train_checkpoints" / "pi0_libero" / "variant"
    params_dir = checkpoint_dir / "200" / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    _ = (params_dir / "_METADATA").write_text("fixture\n", encoding="utf-8")

    plan = _resolve_resume_checkpoint_plan(
        checkpoint_dir=checkpoint_dir,
        runtime_dir=tmp_path / "runtime",
        requested_resume=True,
    )

    assert plan.mode == "params_bootstrap"
    assert plan.latest_step == 200
    assert plan.resume_state_step == 201
    assert plan.latest_checkpoint_dir == checkpoint_dir / "200"
    assert plan.seed_params_path is not None
    assert plan.seed_params_path.is_dir()
    assert (plan.seed_params_path / "_METADATA").read_text(encoding="utf-8") == "fixture\n"
    assert (plan.seed_params_path / "_METADATA").stat().st_ino == (
        params_dir / "_METADATA"
    ).stat().st_ino


def test_install_resume_bootstrap_step_hook_carries_forward_latest_step() -> None:
    class _FakeState:
        def __init__(self, step: int) -> None:
            self.step = step

    class _FakeDataclasses:
        @staticmethod
        def replace(state: _FakeState, *, step: int) -> _FakeState:
            return _FakeState(step)

    def _original_init_train_state(*_args: object, **_kwargs: object):
        return _FakeState(0), "sharding"

    train_main_mod = SimpleNamespace(
        init_train_state=_original_init_train_state,
        dataclasses=_FakeDataclasses(),
    )

    finalize = _install_resume_bootstrap_step_hook(train_main_mod, resume_step=201)
    state, sharding = train_main_mod.init_train_state(None, None, None, resume=False)

    assert state.step == 201
    assert sharding == "sharding"

    finalize()
    restored_state, _ = train_main_mod.init_train_state(None, None, None, resume=False)
    assert restored_state.step == 0


def test_validate_resume_target_step_blocks_params_bootstrap_when_target_not_ahead() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_train_steps greater than restored state step"):
        _validate_resume_target_step(
            ResumeCheckpointPlan(
                mode="params_bootstrap",
                latest_step=200,
                resume_state_step=201,
            ),
            num_train_steps=201,
        )


def test_copytree_materializes_real_directory_even_when_source_is_symlink(
    tmp_path: Path,
) -> None:
    upstream_dir = tmp_path / "runtime" / "upstream_train_checkpoints" / "23" / "params"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    _ = (upstream_dir / "_METADATA").write_text(
        '{"tree":"fixture"}\n', encoding="utf-8"
    )
    _ = (upstream_dir / "manifest.ocdbt").write_text(
        "fixture-ocdbt-manifest\n",
        encoding="utf-8",
    )
    symlinked_source = tmp_path / "runtime" / "real_variant_export" / "params"
    symlinked_source.parent.mkdir(parents=True, exist_ok=True)
    symlinked_source.symlink_to(upstream_dir.resolve(), target_is_directory=True)
    materialized_dir = tmp_path / "artifact" / "best" / "params"

    _copytree(symlinked_source, materialized_dir)

    assert materialized_dir.is_dir()
    assert not materialized_dir.is_symlink()
    assert (materialized_dir / "_METADATA").is_file()
    assert (materialized_dir / "manifest.ocdbt").read_text(encoding="utf-8") == (
        "fixture-ocdbt-manifest\n"
    )

    shutil.rmtree(upstream_dir.parent.parent.parent)

    assert (materialized_dir / "_METADATA").is_file()
    assert (materialized_dir / "manifest.ocdbt").read_text(encoding="utf-8") == (
        "fixture-ocdbt-manifest\n"
    )


def test_replace_tree_with_durable_links_or_copy_prefers_hardlinks_on_same_filesystem(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source" / "params"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = source_dir / "_METADATA"
    source_manifest = nested_dir / "manifest.ocdbt"
    _ = source_metadata.write_text('{"tree":"fixture"}\n', encoding="utf-8")
    _ = source_manifest.write_text("fixture-ocdbt-manifest\n", encoding="utf-8")
    materialized_dir = tmp_path / "artifact" / "best" / "params"

    result = _replace_tree_with_durable_links_or_copy(source_dir, materialized_dir)

    materialized_metadata = materialized_dir / "_METADATA"
    materialized_manifest = materialized_dir / "nested" / "manifest.ocdbt"
    assert result.mode == SERVEABLE_ARTIFACT_HARDLINK_MODE
    assert result.same_filesystem is True
    assert materialized_dir.is_dir()
    assert not materialized_dir.is_symlink()
    assert materialized_metadata.is_file()
    assert materialized_manifest.is_file()
    assert source_metadata.stat().st_ino == materialized_metadata.stat().st_ino
    assert source_manifest.stat().st_ino == materialized_manifest.stat().st_ino
    assert materialized_metadata.stat().st_nlink >= 2
    assert materialized_manifest.stat().st_nlink >= 2

    shutil.rmtree(source_dir.parent)

    assert materialized_metadata.read_text(encoding="utf-8") == '{"tree":"fixture"}\n'
    assert materialized_manifest.read_text(encoding="utf-8") == (
        "fixture-ocdbt-manifest\n"
    )


def test_replace_tree_with_durable_links_or_copy_falls_back_to_copy_when_hardlinking_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source" / "params"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = source_dir / "_METADATA"
    _ = source_metadata.write_text('{"tree":"fixture"}\n', encoding="utf-8")
    materialized_dir = tmp_path / "artifact" / "best" / "params"

    def _raise_cross_device_link(_: Path, __: Path) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(checkpoint_mod.os, "link", _raise_cross_device_link)

    result = _replace_tree_with_durable_links_or_copy(source_dir, materialized_dir)

    materialized_metadata = materialized_dir / "_METADATA"
    assert result.mode == SERVEABLE_ARTIFACT_COPY_MODE
    assert result.same_filesystem is True
    assert materialized_metadata.is_file()
    assert source_metadata.read_text(
        encoding="utf-8"
    ) == materialized_metadata.read_text(encoding="utf-8")
    assert source_metadata.stat().st_ino != materialized_metadata.stat().st_ino
