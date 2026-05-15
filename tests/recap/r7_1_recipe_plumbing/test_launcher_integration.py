from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from work.recap import finetune_full
from work.recap import launch_finetune_use_ddp
from work.recap.r7_1_recipe_plumbing.cli import split_recipe_args
from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags


def _base_cli_args(tmp_path: Path) -> list[str]:
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    return [
        "--dataset-path",
        str(dataset),
        "--output-dir",
        "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/r7_1_parser_test",
        "--python",
        ".envs/wbc/bin/python",
    ]


def _delegate_cmd(tmp_path: Path, extra_args: list[str], monkeypatch=None) -> list[str]:
    parser = finetune_full._build_parser()
    args, forwarded = parser.parse_known_args([*_base_cli_args(tmp_path), *extra_args])
    recipe_flags = RecipeFlags.from_argparse(args)
    forwarded = [*finetune_full.recipe_flags_to_cli_args(recipe_flags), *list(forwarded)]
    if monkeypatch is not None:
        contract = {
            "launcher_python": ".envs/wbc/bin/python",
            "delegate_runtime_python": ".envs/wbc/bin/python",
            "contract_manifest_path": "test",
        }
        monkeypatch.setattr(
            finetune_full,
            "_resolve_two_layer_python_contract",
            lambda repo_root, delegate_runtime_python_flag: contract,
        )
    cmd, _resolved = finetune_full._build_delegate_cmd(repo_root=Path.cwd(), args=args, forwarded=list(forwarded))
    return cmd


def test_finetune_full_help_contains_recipe_group() -> None:
    help_text = finetune_full._build_parser().format_help()
    assert "r7.1_recipe_plumbing" in help_text
    assert "--enable-dual-loss" in help_text


def test_finetune_full_parses_six_new_flags(tmp_path: Path) -> None:
    parser = finetune_full._build_parser()
    args, forwarded = parser.parse_known_args(
        [
            *_base_cli_args(tmp_path),
            "--enable-dual-loss",
            "--dual-loss-alpha",
            "0.5",
            "--indicator-dropout-p",
            "0.15",
            "--indicator-dropout-seed",
            "3",
            "--dual-loss-uses-carrier-text",
            "--carrier-text-field",
            "carrier_text_v1",
        ]
    )
    assert forwarded == []
    assert RecipeFlags.from_argparse(args) == RecipeFlags(True, 0.5, 0.15, 3, True, "carrier_text_v1")


def test_unset_flags_are_default(tmp_path: Path) -> None:
    parser = finetune_full._build_parser()
    args, _forwarded = parser.parse_known_args(_base_cli_args(tmp_path))
    assert RecipeFlags.from_argparse(args).is_default()


def test_single_flag_keeps_other_recipe_defaults(tmp_path: Path) -> None:
    parser = finetune_full._build_parser()
    args, _forwarded = parser.parse_known_args([*_base_cli_args(tmp_path), "--dual-loss-uses-carrier-text"])
    flags = RecipeFlags.from_argparse(args)
    assert flags.dual_loss_uses_carrier_text is True
    assert flags.enable_dual_loss is False
    assert flags.indicator_dropout_p == 0.0


def test_default_equivalent_flags_do_not_change_delegate_cmd(monkeypatch, tmp_path: Path) -> None:
    no_flags = _delegate_cmd(tmp_path, [], monkeypatch)
    explicit_defaults = _delegate_cmd(
        tmp_path,
        [
            "--dual-loss-alpha",
            "0.0",
            "--indicator-dropout-p=0.0",
            "--indicator-dropout-seed",
            "0",
            "--carrier-text-field",
            "prompt_raw",
        ],
        monkeypatch,
    )
    assert explicit_defaults == no_flags


def test_non_default_flags_are_forwarded_to_delegate_cmd(monkeypatch, tmp_path: Path) -> None:
    cmd = _delegate_cmd(
        tmp_path,
        [
            "--enable-dual-loss",
            "--dual-loss-alpha",
            "0.5",
            "--indicator-dropout-p",
            "0.15",
            "--dual-loss-uses-carrier-text",
            "--carrier-text-field",
            "carrier_text_v1",
        ],
        monkeypatch,
    )
    assert "--enable-dual-loss" in cmd
    assert "--dual-loss-alpha" in cmd
    assert "--carrier-text-field" in cmd


def test_launcher_split_recipe_args_keeps_tyro_args_order() -> None:
    flags, remaining, explicit = split_recipe_args(
        [
            "--dataset-path",
            "data",
            "--enable-dual-loss",
            "--max-steps",
            "1",
            "--carrier-text-field=carrier_text_v1",
        ]
    )
    assert flags.enable_dual_loss is True
    assert remaining == ["--dataset-path", "data", "--max-steps", "1"]
    assert explicit == ["--enable-dual-loss", "--carrier-text-field", "carrier_text_v1"]


def test_apply_r7_1_recipe_flags_attaches_only_non_default() -> None:
    config = SimpleNamespace(model=SimpleNamespace(), data=SimpleNamespace(), training=SimpleNamespace())
    ft_config = SimpleNamespace()
    launch_finetune_use_ddp.apply_r7_1_recipe_flags(config=config, ft_config=ft_config)
    assert not hasattr(config.training, "r7_1_recipe_flags")
    flags = RecipeFlags(enable_dual_loss=True, dual_loss_alpha=0.5, carrier_text_field="carrier_text_v1")
    setattr(ft_config, launch_finetune_use_ddp.R7_1_RECIPE_FLAGS_FIELD, flags)
    setattr(ft_config, launch_finetune_use_ddp.R7_1_RECIPE_CLI_ARGS_FIELD, ["--enable-dual-loss"])
    launch_finetune_use_ddp.apply_r7_1_recipe_flags(config=config, ft_config=ft_config)
    assert config.model.r7_1_recipe_flags == flags
    assert config.data.task_text_field == "carrier_text_v1"
