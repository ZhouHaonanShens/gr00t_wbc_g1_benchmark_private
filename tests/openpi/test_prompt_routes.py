from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Mapping
from pathlib import Path
import sys
from typing import Callable, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "work/openpi/prompting/routes.py"
SPEC = importlib.util.spec_from_file_location("openpi_prompt_routes", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load prompt routes module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["openpi_prompt_routes"] = MODULE
SPEC.loader.exec_module(MODULE)

BuildPromptRoute: TypeAlias = Callable[..., object]
BuildPromptProvenance: TypeAlias = Callable[[object], dict[str, str]]
BuildPromptText: TypeAlias = Callable[[object, object], str]

CONDITIONING_MODE = cast(str, getattr(MODULE, "CONDITIONING_MODE"))
PHASE1_PROMPT_ROUTE = cast(str, getattr(MODULE, "PHASE1_PROMPT_ROUTE"))
FIXEDADV_CONSTANT_CONSUMER_MODE = cast(
    str, getattr(MODULE, "FIXEDADV_CONSTANT_CONSUMER_MODE")
)
RECAP_RELABEL_CONSUMER_MODE = cast(str, getattr(MODULE, "RECAP_RELABEL_CONSUMER_MODE"))
SHUFFLED_ADV_DIAG_CONSUMER_MODE = cast(
    str, getattr(MODULE, "SHUFFLED_ADV_DIAG_CONSUMER_MODE")
)
build_shuffled_adv_diag_indicator_mode = cast(
    Callable[[Mapping[str, object]], str],
    getattr(MODULE, "build_shuffled_adv_diag_indicator_mode"),
)
build_phase1_prompt_route = cast(
    BuildPromptRoute, getattr(MODULE, "build_phase1_prompt_route")
)
build_phase1_prompt_provenance = cast(
    BuildPromptProvenance, getattr(MODULE, "build_phase1_prompt_provenance")
)
build_phase1_prompt_text = cast(
    BuildPromptText, getattr(MODULE, "build_phase1_prompt_text")
)


class GuardedLabelRow(Mapping[str, object]):
    _data: dict[str, object]
    _forbidden_keys: set[str]

    def __init__(
        self,
        data: dict[str, object],
        *,
        forbidden_keys: tuple[str, ...] = (),
    ):
        self._data = data
        self._forbidden_keys = set(forbidden_keys)

    def __getitem__(self, key: str) -> object:
        if key in self._forbidden_keys:
            raise AssertionError(f"unexpected access to forbidden key: {key}")
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def test_phase1_prompt_route_happy_path() -> None:
    spec = build_phase1_prompt_route(
        {
            "prompt_raw": "pick up the apple, walk left and place the apple on the plate.",
            "recap_m2.indicator_I": 1,
        }
    )
    provenance = build_phase1_prompt_provenance(spec)

    assert getattr(spec, "prompt_route") == PHASE1_PROMPT_ROUTE
    assert getattr(spec, "conditioning_mode") == CONDITIONING_MODE
    assert getattr(spec, "indicator_mode") == "positive"
    assert getattr(spec, "source_prompt_field") == "prompt_raw"
    assert getattr(spec, "consumer_mode") == RECAP_RELABEL_CONSUMER_MODE
    assert "Advantage: positive" in getattr(spec, "prompt_text")
    assert getattr(spec, "authoritative_carrier_text") == getattr(spec, "prompt_text")
    assert provenance["prompt_route"] == PHASE1_PROMPT_ROUTE
    assert provenance["conditioning_mode"] == CONDITIONING_MODE
    assert provenance["consumer_mode"] == RECAP_RELABEL_CONSUMER_MODE
    assert provenance["authoritative_carrier_field"] == "carrier_text_v1"
    assert provenance["authoritative_carrier_matches_prompt_text"] == "true"


def test_phase1_prompt_route_accepts_legacy_recap_relabel_alias() -> None:
    spec = build_phase1_prompt_route(
        {
            "prompt_raw": "pick up the apple.",
            "recap_m2.indicator_I": 1,
        },
        consumer_mode="recap_relabel",
    )

    assert getattr(spec, "consumer_mode") == RECAP_RELABEL_CONSUMER_MODE


def test_phase1_prompt_text_negative_indicator_uses_canonical_line() -> None:
    prompt = build_phase1_prompt_text("pick up the apple.", 0)
    assert prompt.endswith("Advantage: negative")


def test_phase1_prompt_route_rejects_mixed_prompt_conditioned() -> None:
    try:
        _ = build_phase1_prompt_route(
            {
                "prompt_raw": "pick up the apple.",
                "prompt_conditioned": "advantage positive pick up the apple.",
                "recap_m2.indicator_I": 1,
            }
        )
    except ValueError as exc:
        assert "prompt_conditioned" in str(exc)
    else:
        raise AssertionError("expected mixed prompt semantics to fail")


def test_phase1_prompt_route_rejects_mismatched_carrier_text_v1() -> None:
    try:
        _ = build_phase1_prompt_route(
            {
                "prompt_raw": "pick up the apple.",
                "carrier_text_v1": "pick up the apple.\nAdvantage: negative",
                "recap_m2.indicator_I": 1,
            }
        )
    except ValueError as exc:
        assert "carrier_text_v1" in str(exc)
    else:
        raise AssertionError("expected mismatched carrier_text_v1 to fail")


def test_phase1_prompt_route_rejects_dual_task_text_and_numeric_passthrough() -> None:
    payloads: tuple[dict[str, object], ...] = (
        {
            "prompt_raw": "pick up the apple.",
            "recap_m2.indicator_I": 1,
            "dual_task_text": True,
        },
        {
            "prompt_raw": "pick up the apple.",
            "recap_m2.indicator_I": 1,
            "advantage_input": 0.5,
        },
    )
    for payload in payloads:
        try:
            _ = build_phase1_prompt_route(payload)
        except ValueError as exc:
            message = str(exc)
            assert "Phase 1 prompt route forbids" in message
        else:
            raise AssertionError("expected forbidden prompt route payload to fail")


def test_fixedadv_prompt_route_uses_prompt_raw_only_without_sample_level_reads() -> (
    None
):
    label_row = GuardedLabelRow(
        {
            "prompt_raw": "pick up the apple, walk left and place the apple on the plate."
        },
        forbidden_keys=(
            "recap_m2.indicator_I",
            "prompt_conditioned",
            "advantage_input",
        ),
    )

    spec = build_phase1_prompt_route(
        label_row,
        consumer_mode=FIXEDADV_CONSTANT_CONSUMER_MODE,
    )
    provenance = build_phase1_prompt_provenance(spec)

    assert getattr(spec, "prompt_text") == label_row["prompt_raw"]
    assert getattr(spec, "indicator_mode") == "omit"
    assert getattr(spec, "fixed_indicator_mode") == "omit"
    assert getattr(spec, "consumer_mode") == FIXEDADV_CONSTANT_CONSUMER_MODE
    assert provenance["consumer_mode"] == FIXEDADV_CONSTANT_CONSUMER_MODE
    assert provenance["fixed_indicator_mode"] == "omit"
    assert provenance["indicator_source"] == "fixed_indicator_mode"
    assert provenance["prompt_text_surface"] == "prompt_raw_only"
    assert provenance["per_sample_indicator_consumption"] == "false"
    assert provenance["prompt_conditioned_dependency"] == "false"
    assert provenance["advantage_input_dependency"] == "false"


def test_fixedadv_prompt_route_rejects_non_omit_fixed_indicator_mode() -> None:
    try:
        _ = build_phase1_prompt_route(
            {"prompt_raw": "pick up the apple."},
            consumer_mode=FIXEDADV_CONSTANT_CONSUMER_MODE,
            fixed_indicator_mode="positive",
        )
    except ValueError as exc:
        assert "fixedadv_constant" in str(exc)
    else:
        raise AssertionError(
            "expected fixedadv to reject non-omit fixed_indicator_mode"
        )


def test_shuffled_adv_diag_route_uses_deterministic_sample_hash_not_raw_indicator() -> (
    None
):
    label_row = {
        "prompt_raw": "pick up the apple.",
        "observation.state": [0.1, 0.2, 0.3],
        "episode_index": 5,
    }

    positive_mode = build_shuffled_adv_diag_indicator_mode(
        {
            **label_row,
            "recap_m2.indicator_I": 1,
        }
    )
    negative_mode = build_shuffled_adv_diag_indicator_mode(
        {
            **label_row,
            "recap_m2.indicator_I": 0,
        }
    )
    spec = build_phase1_prompt_route(
        {
            **label_row,
            "recap_m2.indicator_I": 1,
        },
        consumer_mode=SHUFFLED_ADV_DIAG_CONSUMER_MODE,
    )
    provenance = build_phase1_prompt_provenance(spec)

    assert positive_mode == negative_mode
    assert getattr(spec, "consumer_mode") == SHUFFLED_ADV_DIAG_CONSUMER_MODE
    assert getattr(spec, "indicator_mode") == positive_mode
    assert provenance["consumer_mode"] == SHUFFLED_ADV_DIAG_CONSUMER_MODE
    assert provenance["indicator_source"] == "deterministic_shuffled_sample_key"
    assert provenance["per_sample_indicator_consumption"] == "true"
    assert provenance["prompt_text_surface"] == "canonical_text_indicator"
