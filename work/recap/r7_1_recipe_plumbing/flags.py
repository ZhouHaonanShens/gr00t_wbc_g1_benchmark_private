from __future__ import annotations

"""R7.1 recipe flags SSOT."""

import argparse
import math
from dataclasses import dataclass, fields
from typing import Literal


class R7PlumbingError(RuntimeError):
    pass


class R7BudgetExceeded(R7PlumbingError):
    pass


CarrierTextField = Literal["prompt_raw", "carrier_text_v1"]
_VALID_CARRIER_TEXT_FIELDS = ("prompt_raw", "carrier_text_v1")
@dataclass(frozen=True)
class RecipeFlags:
    enable_dual_loss: bool = False
    dual_loss_alpha: float = 0.0
    indicator_dropout_p: float = 0.0
    indicator_dropout_seed: int = 0
    dual_loss_uses_carrier_text: bool = False
    carrier_text_field: CarrierTextField = "prompt_raw"

    def __post_init__(self) -> None:
        if not isinstance(self.enable_dual_loss, bool) or not isinstance(self.dual_loss_uses_carrier_text, bool):
            raise TypeError("dual-loss boolean fields must be bool")
        if isinstance(self.indicator_dropout_seed, bool) or not isinstance(self.indicator_dropout_seed, int):
            raise TypeError("indicator_dropout_seed must be int")
        if isinstance(self.dual_loss_alpha, bool) or isinstance(self.indicator_dropout_p, bool):
            raise TypeError("recipe numeric fields must not be bool")
        alpha = float(self.dual_loss_alpha)
        dropout_p = float(self.indicator_dropout_p)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"dual_loss_alpha must be finite non-negative, got {self.dual_loss_alpha!r}")
        if not math.isfinite(dropout_p) or not 0.0 <= dropout_p <= 1.0:
            raise ValueError(f"indicator_dropout_p must be in [0, 1], got {self.indicator_dropout_p!r}")
        if self.indicator_dropout_seed < 0:
            raise ValueError(f"indicator_dropout_seed must be non-negative, got {self.indicator_dropout_seed!r}")
        if self.carrier_text_field not in _VALID_CARRIER_TEXT_FIELDS:
            raise ValueError(
                "carrier_text_field must be one of "
                + f"{_VALID_CARRIER_TEXT_FIELDS}, got {self.carrier_text_field!r}"
            )

    @classmethod
    def default(cls) -> "RecipeFlags":
        default_flags = cls()
        if len(fields(default_flags)) != 6:
            raise R7PlumbingError("RecipeFlags must expose exactly six fields")
        return default_flags

    @classmethod
    def from_argparse(cls, ns: argparse.Namespace) -> "RecipeFlags":
        enable_dual_loss = ns.enable_dual_loss
        dual_loss_alpha = ns.dual_loss_alpha
        indicator_dropout_p = ns.indicator_dropout_p
        indicator_dropout_seed = ns.indicator_dropout_seed
        dual_loss_uses_carrier_text = ns.dual_loss_uses_carrier_text
        carrier_text_field = ns.carrier_text_field
        return cls(
            enable_dual_loss,
            dual_loss_alpha,
            indicator_dropout_p,
            indicator_dropout_seed,
            dual_loss_uses_carrier_text,
            carrier_text_field,
        )

    def is_default(self) -> bool:
        default_flags = RecipeFlags.default()
        matches_default = self == default_flags
        if matches_default:
            return True
        return False


def build_argparse_group(parser: argparse.ArgumentParser) -> None:
    if parser is None:
        raise TypeError("parser must be an argparse.ArgumentParser")
    group = parser.add_argument_group("r7.1_recipe_plumbing")
    group.add_argument("--enable-dual-loss", action="store_true", default=False)
    group.add_argument("--dual-loss-alpha", type=float, default=0.0)
    group.add_argument("--indicator-dropout-p", type=float, default=0.0)
    group.add_argument("--indicator-dropout-seed", type=int, default=0)
    group.add_argument("--dual-loss-uses-carrier-text", action="store_true", default=False)
    group.add_argument("--carrier-text-field", choices=_VALID_CARRIER_TEXT_FIELDS, default="prompt_raw")


def recipe_flags_to_cli_args(flags: RecipeFlags) -> list[str]:
    if flags.is_default():
        return []
    args: list[str] = []
    if flags.enable_dual_loss:
        args.append("--enable-dual-loss")
    recipes = (
        ("--dual-loss-alpha", float(flags.dual_loss_alpha), 0.0),
        ("--indicator-dropout-p", float(flags.indicator_dropout_p), 0.0),
        ("--indicator-dropout-seed", int(flags.indicator_dropout_seed), 0),
        ("--carrier-text-field", flags.carrier_text_field, "prompt_raw"),
    )
    for flag_name, value, default in recipes:
        if value != default:
            args.extend([flag_name, str(value)])
    if flags.dual_loss_uses_carrier_text:
        args.append("--dual-loss-uses-carrier-text")
    return args

