from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tests.recap import test_state_conditioned_training_fairness as training_fairness
from work.recap import policy as recap_policy
from work.recap import text_indicator
from work.recap import run_manifest
from work.recap.scripts import state_conditioned_train


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_comparable_run_spec_records_mainline_carrier_route_and_runtime_policy(
    tmp_path: Path,
) -> None:
    training_root = training_fairness._build_training_set_root(tmp_path / "fixtures")
    output_root = tmp_path / "training_runs"
    args, forwarded = training_fairness._build_args(
        "--training-set-root",
        str(training_root),
        "--output-dir",
        str(output_root),
    )

    _ = state_conditioned_train.materialize_state_conditioned_training(
        args=args,
        forwarded=forwarded,
        kernel_runner=training_fairness._fake_runner_factory(),
    )

    for variant_key in ("c0", "c1"):
        metadata = _read_json(
            output_root
            / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT[variant_key]
        )
        comparable_run_spec = dict(metadata["comparable_run_spec"])
        training_route = dict(metadata["training_route"])

        assert comparable_run_spec["carrier_schema_version"] == (
            state_conditioned_train.MAINLINE_CARRIER_SCHEMA_VERSION
        )
        assert comparable_run_spec["carrier_schema_version"] == (
            run_manifest.TEXT_CARRIER_SCHEMA_VERSION
        )
        assert comparable_run_spec["carrier_route"] == (
            state_conditioned_train.MAINLINE_CARRIER_ROUTE
        )
        assert comparable_run_spec["carrier_route"] == run_manifest.TEXT_CARRIER_ROUTE
        assert comparable_run_spec["carrier_route"] == (
            text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
        )
        assert comparable_run_spec["prompt_source_field"] == (
            state_conditioned_train.MAINLINE_PROMPT_SOURCE_FIELD
        )
        assert (
            comparable_run_spec["prompt_source_field"]
            == run_manifest.PROMPT_SOURCE_FIELD
        )
        assert comparable_run_spec["prompt_source_field"] == (
            text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD
        )
        assert (
            comparable_run_spec["indicator_source"]
            == run_manifest.INDICATOR_SOURCE_FIELD
        )

        assert training_route == comparable_run_spec["training_route"]
        assert training_route["carrier_route"] == run_manifest.TEXT_CARRIER_ROUTE
        assert training_route["carrier_schema_version"] == (
            run_manifest.TEXT_CARRIER_SCHEMA_VERSION
        )
        assert training_route["prompt_source_field"] == run_manifest.PROMPT_SOURCE_FIELD
        assert training_route["indicator_source"] == run_manifest.INDICATOR_SOURCE_FIELD
        assert training_route["runtime_route"] == recap_policy.MAINLINE_RUNTIME_ROUTE
        assert training_route["runtime_policy_class"] == (
            recap_policy.MAINLINE_RUNTIME_POLICY_CLASS_NAME
        )
        assert training_route["runtime_indicator_mode_required"] is True
        assert training_route["mainline_authority"] is True
        assert training_route["diagnostic_only"] is False
        assert set(training_route["runtime_supported_indicator_modes"]) == set(
            recap_policy.MAINLINE_RUNTIME_INDICATOR_MODES
        )
        assert set(training_route["runtime_supported_indicator_modes"]) == {
            text_indicator.TEXT_INDICATOR_OMIT,
            text_indicator.TEXT_INDICATOR_POSITIVE,
            text_indicator.TEXT_INDICATOR_NEGATIVE,
        }
