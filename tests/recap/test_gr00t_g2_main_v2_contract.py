from __future__ import annotations

import importlib
import json
import shutil
from argparse import Namespace
from pathlib import Path

from work.recap import gr00t_main_recap
from work.recap import hf_snapshot_patch
from work.recap import text_indicator


def test_text_indicator_dropout_is_deterministic_and_preserves_omit() -> None:
    assert (
        text_indicator.apply_indicator_dropout(
            "positive", dropout_p=0.0, seed=7, sample_key="a"
        )
        == "positive"
    )
    assert (
        text_indicator.apply_indicator_dropout(
            "negative", dropout_p=1.0, seed=7, sample_key="a"
        )
        == "omit"
    )
    assert (
        text_indicator.apply_indicator_dropout(
            "omit", dropout_p=1.0, seed=7, sample_key="a"
        )
        == "omit"
    )
    first = text_indicator.apply_indicator_dropout(
        "positive", dropout_p=0.3, seed=20260429, sample_key="episode=1|step=2"
    )
    second = text_indicator.apply_indicator_dropout(
        "positive", dropout_p=0.3, seed=20260429, sample_key="episode=1|step=2"
    )
    assert first == second


def test_canonical_text_indicator_places_indicator_after_prompt() -> None:
    assert (
        text_indicator.build_canonical_text_indicator("pick apple", "positive")
        == "pick apple\nAdvantage: positive"
    )
    assert (
        text_indicator.build_canonical_text_indicator("pick apple", "negative")
        == "pick apple\nAdvantage: negative"
    )
    assert text_indicator.build_canonical_text_indicator("pick apple", "omit") == "pick apple"


def test_critic_vlm_assessment_blocks_old_constant_query_artifact(tmp_path: Path) -> None:
    critic_dir = tmp_path / "critic"
    dataset = tmp_path / "dataset"
    critic_dir.mkdir()
    dataset.mkdir()
    (critic_dir / "config.json").write_text(
        json.dumps(
            {
                "artifact_version": "multimodal_distributional_v1",
                "bin_centers": list(range(201)),
                "value_scale": "raw_return",
                "prompt_text_mode": "constant_query_only",
                "smoke_backend": "qwen3_vl_late_fusion_v1",
            }
        ),
        encoding="utf-8",
    )
    (critic_dir / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_path": str(tmp_path / "other_dataset"),
                "train_manifest_summary": {"input_mode": {"use_prompt": False}},
            }
        ),
        encoding="utf-8",
    )
    (critic_dir / "metrics.json").write_text("{}", encoding="utf-8")
    assessment = gr00t_main_recap.assess_critic_vlm_for_g2_main_v2(
        critic_dir=critic_dir,
        target_dataset_path=dataset,
    )
    assert assessment["usable_as_authoritative_critic"] is False
    assert "VALUE_SCALE_NOT_TASK_NORMALIZED_RETURN" in assessment["blocking_reasons"]
    assert "PROMPT_NOT_LANGUAGE_CONDITIONED" in assessment["blocking_reasons"]
    assert "CRITIC_DATASET_DOES_NOT_MATCH_G2_MAIN_V2_DATASET" in assessment["blocking_reasons"]


def test_34b_text_indicator_route_freezes_mainline_policy_spec() -> None:
    smoke = importlib.import_module("work.recap.scripts.34b_recap_numeric_adv_smoke")
    args = Namespace(
        conditioning_route="text_indicator_v1",
        runtime_indicator_mode="positive",
        recap_train_scope="strict_full",
        condition_focused_continuation=False,
        condition_hot_lr_scale=3.0,
        diffusion_trunk_lr_scale=1.0,
    )
    scope_summary = smoke._scope_summary_for_args(args)
    authority = smoke._build_trainability_authority_from_args(
        args,
        scope_summary=scope_summary,
    )
    assert scope_summary["method_faithfulness"]["recap_method_contract"] == "binary_text_indicator_v1"
    assert authority["route_freeze"]["route"] == "carrier_text_v1"
    assert authority["route_freeze"]["mainline_authority"] is True
    assert authority["route_freeze"]["diagnostic_only"] is False


def test_identity_hf_patch_accepts_existing_nonzero_top_llm_config(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text(
        json.dumps({"model_type": "Gr00tN1d6", "tune_top_llm_layers": 4}),
        encoding="utf-8",
    )
    out_root = Path("agent/runtime_logs/pytest_hf_identity_patch")
    shutil.rmtree(out_root, ignore_errors=True)
    try:
        first = hf_snapshot_patch.make_patched_base_model_dir(
            repo_id="local/test-model",
            snapshot_dir=snapshot_dir,
            out_root=out_root,
            force_tune_top_llm_layers_zero=False,
            emit_evidence=False,
        )
        second = hf_snapshot_patch.make_patched_base_model_dir(
            repo_id="local/test-model",
            snapshot_dir=snapshot_dir,
            out_root=out_root,
            force_tune_top_llm_layers_zero=False,
            emit_evidence=False,
        )
        assert second == first
        patched = json.loads((second / "config.json").read_text(encoding="utf-8"))
        assert patched["tune_top_llm_layers"] == 4
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_text_indicator_hf_patch_updates_processor_formalize_language(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text(
        json.dumps({"formalize_language": True, "tune_top_llm_layers": 4}),
        encoding="utf-8",
    )
    (snapshot_dir / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gr00tN1d6Processor",
                "processor_kwargs": {"formalize_language": True},
            }
        ),
        encoding="utf-8",
    )
    out_root = Path("agent/runtime_logs/pytest_hf_text_indicator_patch")
    shutil.rmtree(out_root, ignore_errors=True)
    try:
        patched_dir = hf_snapshot_patch.make_patched_base_model_dir(
            repo_id="local/test-model",
            snapshot_dir=snapshot_dir,
            out_root=out_root,
            overrides={"formalize_language": False},
            force_tune_top_llm_layers_zero=False,
            emit_evidence=False,
        )
        model_config = json.loads((patched_dir / "config.json").read_text(encoding="utf-8"))
        processor_config = json.loads(
            (patched_dir / "processor_config.json").read_text(encoding="utf-8")
        )
        assert model_config["formalize_language"] is False
        assert processor_config["processor_kwargs"]["formalize_language"] is False
    finally:
        shutil.rmtree(out_root, ignore_errors=True)
