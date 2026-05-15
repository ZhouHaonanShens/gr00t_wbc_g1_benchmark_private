from __future__ import annotations

from pathlib import Path

import pytest

from work.recap.r7_2_uplift_probe import cli
from work.recap.r7_2_uplift_probe.contract import R7UpliftError

TOKEN = "b" * 64


def _args(tmp_path: Path, *extra: str) -> list[str]:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    return [
        "trial",
        "--trial-id",
        "trial-1",
        "--base-ckpt",
        str(ckpt),
        "--recipe-preset",
        "full_C1_C2_C5",
        "--output-root",
        str(tmp_path / "out"),
        "--leader-approval-token",
        TOKEN,
        "--gpu",
        "1",
        *extra,
    ]


@pytest.mark.parametrize("preset", ["full_C1_C2_C5", "subset_C1_C5_no_dropout", "subset_C1_only"])
def test_three_recipe_presets_are_accepted(tmp_path: Path, preset: str) -> None:
    argv = _args(tmp_path)
    argv[argv.index("--recipe-preset") + 1] = preset
    request = cli.request_from_args(cli.build_parser().parse_args(argv))
    assert request.recipe_preset == preset


def test_fourth_recipe_preset_is_rejected_by_argparse(tmp_path: Path) -> None:
    argv = _args(tmp_path)
    argv[argv.index("--recipe-preset") + 1] = "ad_hoc"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


def test_trial_id_rejects_ad_hoc(tmp_path: Path) -> None:
    argv = _args(tmp_path)
    argv[argv.index("--trial-id") + 1] = "trial-x"
    with pytest.raises(R7UpliftError):
        cli.request_from_args(cli.build_parser().parse_args(argv))


def test_output_root_must_not_exist(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(R7UpliftError):
        cli.request_from_args(cli.build_parser().parse_args(_args(tmp_path)))


@pytest.mark.parametrize("gpu", ["0", "3"])
def test_gpu_0_and_3_rejected(tmp_path: Path, gpu: str) -> None:
    argv = _args(tmp_path)
    argv[argv.index("--gpu") + 1] = gpu
    with pytest.raises(R7UpliftError):
        cli.request_from_args(cli.build_parser().parse_args(argv))


def test_trial_1_rejects_gpu_2(tmp_path: Path) -> None:
    argv = _args(tmp_path)
    argv[argv.index("--gpu") + 1] = "2"
    with pytest.raises(R7UpliftError):
        cli.request_from_args(cli.build_parser().parse_args(argv))


def test_token_and_lora_gates(tmp_path: Path) -> None:
    argv = _args(tmp_path, "--lora-rank", "17")
    with pytest.raises(R7UpliftError):
        cli.request_from_args(cli.build_parser().parse_args(argv))
    argv = _args(tmp_path)
    argv[argv.index("--leader-approval-token") + 1] = "bad"
    with pytest.raises(R7UpliftError):
        cli.request_from_args(cli.build_parser().parse_args(argv))
