from __future__ import annotations

from work.recap.r7_recipe_diff.analyzer import analyze_all
from work.recap.r7_recipe_diff.reports import (
    COMPONENT_KEYS,
    DELTA_KEYS,
    SCHEMA_VERSION,
    TOP_LEVEL_KEYS,
    recipe_diff_payload,
    render_recipe_diff_markdown,
)


def _payload() -> dict[str, object]:
    return recipe_diff_payload(
        base_cell="A.2",
        generated_at_utc="20260513T235244Z",
        source_openpi_report="agent/exchange/openpi_recap_fidelity_fact_report_v1.md",
        deltas=analyze_all("A.2"),
    )


def test_recipe_diff_json_schema_is_exact_and_ordered() -> None:
    payload = _payload()
    assert tuple(payload) == TOP_LEVEL_KEYS
    assert payload["schema_version"] == SCHEMA_VERSION
    deltas = payload["deltas"]
    assert isinstance(deltas, list)
    for delta in deltas:
        assert tuple(delta) == DELTA_KEYS
        component = delta["component"]
        assert isinstance(component, dict)
        assert tuple(component) == COMPONENT_KEYS
        for pair in delta["config_path_diff"]:
            assert isinstance(pair, list)
            assert len(pair) == 2


def test_composable_cli_args_preserves_component_order() -> None:
    payload = _payload()
    assert payload["composable_cli_args"] == (
        "--enable-dual-loss --dual-loss-alpha=0.5 "
        "--indicator-dropout-p=0.15 --indicator-dropout-seed=0 "
        "--enable-learned-value-head --value-loss-alpha=0.1 "
        "--action-head-advantage-input=enabled --advantage-embedding-dim=16 "
        "--dual-loss-uses-carrier-text --carrier-text-field=carrier_text_v1"
    )


def test_markdown_contains_summary_and_future_args_label() -> None:
    report = render_recipe_diff_markdown(_payload())
    assert "| component_id | current | paper-prescribed | diff_action | cli_addition |" in report
    assert "## Composable training recipe" in report
    assert "required future args" in report
    assert "R7.0 does not execute them" in report
