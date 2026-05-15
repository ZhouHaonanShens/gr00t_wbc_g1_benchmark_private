from __future__ import annotations

import argparse
from pathlib import Path

from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags, build_argparse_group, recipe_flags_to_cli_args

_VALUE_FLAGS = ("--dual-loss-alpha", "--indicator-dropout-p", "--indicator-dropout-seed", "--carrier-text-field")
_BOOL_FLAGS = ("--enable-dual-loss", "--dual-loss-uses-carrier-text")


def split_recipe_args(argv: list[str]) -> tuple[RecipeFlags, list[str], list[str]]:
    remaining_args: list[str] = []
    recipe_args: list[str] = []
    index = 0
    while index < len(argv):
        current = str(argv[index])
        flag_name = current.split("=", maxsplit=1)[0]
        if current in _BOOL_FLAGS:
            recipe_args.append(current)
        elif current in _VALUE_FLAGS or flag_name in _VALUE_FLAGS:
            recipe_args.append(current)
            if "=" not in current:
                index += 1
                if index >= len(argv):
                    raise ValueError(f"missing value after {current}")
                recipe_args.append(str(argv[index]))
        else:
            remaining_args.append(current)
        index += 1
    flags = _parse_recipe_args(recipe_args)
    return flags, remaining_args, recipe_flags_to_cli_args(flags)


def _parse_recipe_args(args: list[str]) -> RecipeFlags:
    parser = argparse.ArgumentParser(add_help=False)
    build_argparse_group(parser)
    namespace = parser.parse_args(list(args))
    return RecipeFlags.from_argparse(namespace)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(argv)
    if namespace.command == "dryrun":
        return _run_dryrun_command(namespace)
    if namespace.command == "dryrun-child":
        return _run_child_command(namespace)
    parser.error(f"unknown command {namespace.command!r}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m work.recap.r7_1_recipe_plumbing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_dryrun_parser(subparsers)
    _add_child_parser(subparsers)
    return parser


def _add_dryrun_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dryrun_parser = subparsers.add_parser("dryrun")
    dryrun_parser.add_argument("--ckpt", required=True)
    dryrun_parser.add_argument("--output-root", required=True)
    dryrun_parser.add_argument("--leader-approval-token", required=True)
    dryrun_parser.add_argument("--gpu", type=int, required=True)
    dryrun_parser.add_argument("--budget-minutes", type=int, default=2)
    build_argparse_group(dryrun_parser)


def _add_child_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    child_parser = subparsers.add_parser("dryrun-child")
    child_parser.add_argument("--ckpt", required=True)
    child_parser.add_argument("--output-root", required=True)
    build_argparse_group(child_parser)


def _run_dryrun_command(namespace: argparse.Namespace) -> int:
    from work.recap.r7_1_recipe_plumbing.dryrun import DryrunRequest, run_dryrun
    flags = RecipeFlags.from_argparse(namespace)
    request = DryrunRequest(
        ckpt_abs_path=str(Path(namespace.ckpt).expanduser()),
        flags=flags,
        output_root=str(Path(namespace.output_root).expanduser()),
        gpu_id=int(namespace.gpu),
        leader_approval_token=str(namespace.leader_approval_token),
        budget_minutes=int(namespace.budget_minutes),
    )
    report = run_dryrun(request)
    if report.loss_finite:
        return int(report.subprocess_returncode)
    return 1


def _run_child_command(namespace: argparse.Namespace) -> int:
    from work.recap.r7_1_recipe_plumbing.dryrun import run_child_smoke
    flags = RecipeFlags.from_argparse(namespace)
    output_root = str(Path(namespace.output_root).expanduser())
    ckpt_path = str(Path(namespace.ckpt).expanduser())
    return run_child_smoke(ckpt_path, output_root, flags)
