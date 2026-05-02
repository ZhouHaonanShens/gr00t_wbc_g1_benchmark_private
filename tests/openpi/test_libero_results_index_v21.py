from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Callable, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DOC = REPO_ROOT / "agent/exchange/openpi_libero_v21_results.md"
ENTRY_DOC = REPO_ROOT / "agent/exchange/openpi_libero_v22_entry_prereqs.md"
LIVE_PAIRED_SUMMARY_JSON = (
    REPO_ROOT / "agent/artifacts/openpi_libero_v21/paired_summary_abcx_v21.json"
)
LIVE_GO_NO_GO_JSON = REPO_ROOT / "agent/artifacts/openpi_libero_v21/go_no_go_v21.json"
CANONICAL_PAIRED_SUMMARY_PATH = (
    "agent/artifacts/openpi_libero_v21/paired_summary_abcx_v21.json"
)
CANONICAL_GO_NO_GO_PATH = "agent/artifacts/openpi_libero_v21/go_no_go_v21.json"
CANONICAL_SELECTION_SOURCE = (
    "agent/artifacts/openpi_libero_v21/stock_seed_scan_v21/stock_seed_scan_summary.json"
)
CANONICAL_SELECTION_SOURCE_HASH = "fixture-stock-scan-hash"
FIXTURE_SELECTED_SEEDS = [7, 17, 27, 37, 47, 57]

PUBLISHER_MODULE_PATH = REPO_ROOT / "work/openpi/scripts/libero_publish_v21_results.py"
PUBLISHER_SPEC = importlib.util.spec_from_file_location(
    "openpi_libero_publish_v21_results", PUBLISHER_MODULE_PATH
)
if PUBLISHER_SPEC is None or PUBLISHER_SPEC.loader is None:
    raise RuntimeError(f"unable to load publisher module from {PUBLISHER_MODULE_PATH}")
PUBLISHER_MODULE = importlib.util.module_from_spec(PUBLISHER_SPEC)
sys.modules["openpi_libero_publish_v21_results"] = PUBLISHER_MODULE
PUBLISHER_SPEC.loader.exec_module(PUBLISHER_MODULE)

GO_NO_GO_MODULE_PATH = REPO_ROOT / "work/openpi/eval/libero_go_no_go_v21.py"
GO_NO_GO_SPEC = importlib.util.spec_from_file_location(
    "openpi_libero_go_no_go_eval_v21_for_docs", GO_NO_GO_MODULE_PATH
)
if GO_NO_GO_SPEC is None or GO_NO_GO_SPEC.loader is None:
    raise RuntimeError(
        f"unable to load v21 go/no-go module from {GO_NO_GO_MODULE_PATH}"
    )
GO_NO_GO_MODULE = importlib.util.module_from_spec(GO_NO_GO_SPEC)
sys.modules["openpi_libero_go_no_go_eval_v21_for_docs"] = GO_NO_GO_MODULE
GO_NO_GO_SPEC.loader.exec_module(GO_NO_GO_MODULE)

PublishV21Results: TypeAlias = Callable[..., tuple[str, str]]
BuildMetricLadderSummaryV21: TypeAlias = Callable[..., object]
BuildBootstrapCiV21: TypeAlias = Callable[..., object]
BuildLiberoAbcxGateArtifactsV21: TypeAlias = Callable[..., object]

publish_v21_results = cast(
    PublishV21Results, getattr(PUBLISHER_MODULE, "publish_v21_results")
)
build_metric_ladder_summary_v21 = cast(
    BuildMetricLadderSummaryV21,
    getattr(GO_NO_GO_MODULE, "build_metric_ladder_summary_v21"),
)
build_bootstrap_ci_v21 = cast(
    BuildBootstrapCiV21, getattr(GO_NO_GO_MODULE, "build_bootstrap_ci_v21")
)
build_libero_abcx_gate_artifacts_v21 = cast(
    BuildLiberoAbcxGateArtifactsV21,
    getattr(GO_NO_GO_MODULE, "build_libero_abcx_gate_artifacts_v21"),
)
DEFAULT_PRIMARY_METRIC_ID = cast(
    str, getattr(GO_NO_GO_MODULE, "DEFAULT_PRIMARY_METRIC_ID")
)
VARIANT_CODE_TO_ID = cast(
    Mapping[str, str], getattr(GO_NO_GO_MODULE, "VARIANT_CODE_TO_ID")
)


def _mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise AssertionError(f"expected mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _sequence(raw: object) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise AssertionError(f"expected sequence, got {type(raw).__name__}")
    return raw


def _metric_payload(
    *, pairs: Mapping[str, object], pair_label: str, metric_id: str
) -> Mapping[str, object]:
    pair_payload = _mapping(pairs[pair_label])
    metrics = _mapping(pair_payload["metrics"])
    return _mapping(metrics[metric_id])


def _load_json(path: Path) -> Mapping[str, object]:
    return _mapping(cast(object, json.loads(path.read_text(encoding="utf-8"))))


def _trace_row(
    *,
    variant: str,
    task_id: int,
    seed: int,
    trial_idx: int,
    executed_steps: int,
    max_steps_resolved: int,
    first_success_step: int | None,
    timeout_flag: bool,
) -> dict[str, object]:
    return {
        "variant": variant,
        "task_id": task_id,
        "seed": seed,
        "trial_idx": trial_idx,
        "success": first_success_step is not None,
        "first_success_step": first_success_step,
        "executed_steps": executed_steps,
        "max_steps_resolved": max_steps_resolved,
        "success_within_50pct_budget": first_success_step is not None
        and first_success_step <= 10,
        "success_within_75pct_budget": first_success_step is not None
        and first_success_step <= 15,
        "timeout_flag": timeout_flag,
        "deviation_notes": [],
    }


def _variant_trace_rows(
    variant: str,
    episodes: Sequence[tuple[int, int | None, int, bool]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (seed, first_success_step, executed_steps, timeout_flag) in enumerate(
        episodes
    ):
        rows.append(
            _trace_row(
                variant=variant,
                task_id=0 if index < 2 else 1,
                seed=seed,
                trial_idx=0,
                executed_steps=executed_steps,
                max_steps_resolved=20,
                first_success_step=first_success_step,
                timeout_flag=timeout_flag,
            )
        )
    return rows


def _summary_fixture(
    trace_rows: Sequence[Mapping[str, object]],
    metric_ladder_summary: Mapping[str, object],
    *,
    variant_id: str,
) -> dict[str, object]:
    success_count = sum(
        1 for row in trace_rows if row["first_success_step"] is not None
    )
    failure_count = len(trace_rows) - success_count
    timeout_count = sum(1 for row in trace_rows if bool(row["timeout_flag"]))
    return {
        "variant": variant_id,
        "primary_metric_id": metric_ladder_summary["primary_metric_id"],
        "scope_audit": {
            "observed_episode_count": len(trace_rows),
            "success_count": success_count,
            "failure_count": failure_count,
            "timeout_count": timeout_count,
        },
        "rollout_summary": {
            "success_rate": success_count / len(trace_rows),
            "success_count": success_count,
            "failure_count": failure_count,
        },
    }


def _seed_record(task_id: int, seed: int, rank: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "seed": seed,
        "ranking_key": {
            "success_rate@0.50_budget": float(rank),
            "success_rate@0.75_budget": float(rank),
            "median_first_success_step_fraction_null_as_one": 1.0 - float(rank) * 0.01,
            "timeout_rate": float(rank) * 0.01,
            "seed": seed,
        },
    }


def _fixture_stock_scan_summary() -> dict[str, object]:
    seed_records: list[dict[str, object]] = []
    for task_id in (0, 1):
        for rank, seed in enumerate(FIXTURE_SELECTED_SEEDS, start=1):
            seed_records.append(_seed_record(task_id, seed, rank))
    return {
        "schema_version": "openpi_libero_stock_seed_scan_summary_v21",
        "task_ids": [0, 1],
        "seed_records": seed_records,
    }


def _materialize_fixture_selection_source(tmp_path: Path) -> tuple[str, str]:
    selection_source_path = tmp_path / "stock_seed_scan_summary.json"
    _write_json(selection_source_path, _fixture_stock_scan_summary())
    return str(selection_source_path), CANONICAL_SELECTION_SOURCE_HASH


def _manifest_payload(
    *,
    authority_id: str,
    selection_source: str,
    selection_source_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "openpi_libero_fresh_rollout_manifest_v21",
        "authority_id": authority_id,
        "manifest_name": "hard_seed_strong_v21",
        "task_suite_name": "libero_spatial",
        "task_ids": [0, 1],
        "seed_manifest": list(FIXTURE_SELECTED_SEEDS),
        "num_trials_per_task": 2,
        "variant_scope": list(VARIANT_CODE_TO_ID.values()),
        "budget_fractions": [0.5, 0.75, 1.0],
        "metric_profile": "budget_ladder_v1",
        "episode_budget_mode": "inherit_from_protocol",
        "selection_policy": "stock_only_hard_seed_v1",
        "selection_source": selection_source,
        "selection_source_hash": selection_source_hash,
    }


def _authority_bundle(
    *,
    variant_code: str,
    authority_dir: str,
    authority_id: str,
    selection_source: str,
    selection_source_hash: str,
    episodes: Sequence[tuple[int, int | None, int, bool]],
) -> dict[str, object]:
    variant_id = VARIANT_CODE_TO_ID[variant_code]
    trace_rows = _variant_trace_rows(variant_id, episodes)
    metric_ladder_summary = cast(
        Mapping[str, object],
        build_metric_ladder_summary_v21(
            trace_rows=trace_rows,
            authority_id=authority_id,
            variant=variant_id,
            checkpoint_ref=f"fixture://{variant_code.lower()}",
            metric_profile="budget_ladder_v1",
            primary_metric_id=DEFAULT_PRIMARY_METRIC_ID,
        ),
    )
    bootstrap_ci = cast(
        Mapping[str, object],
        build_bootstrap_ci_v21(
            trace_rows=trace_rows,
            deterministic_seed_material=f"docs-fixture:{variant_code}",
            variant=variant_id,
        ),
    )
    return {
        "authority_dir": authority_dir,
        "eval_manifest": _manifest_payload(
            authority_id=authority_id,
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
        ),
        "summary": _summary_fixture(
            trace_rows,
            metric_ladder_summary,
            variant_id=variant_id,
        ),
        "metric_ladder_summary": metric_ladder_summary,
        "bootstrap_ci": bootstrap_ci,
        "pairwise_delta": {"schema_version": "fixture_pairwise_delta_placeholder"},
        "per_episode_trace": trace_rows,
    }


def _lite_selection_bundles(
    *, selection_source: str, selection_source_hash: str
) -> dict[str, dict[str, object]]:
    return {
        "A": _authority_bundle(
            variant_code="A",
            authority_id="fresh_rollout_v21_lite",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/a_hard_seed_lite_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 5, 5, False),
                (17, 12, 12, False),
                (27, None, 20, True),
                (37, None, 20, True),
            ],
        )
    }


def _strong_not_decision_capable_bundles(
    *, selection_source: str, selection_source_hash: str
) -> dict[str, dict[str, object]]:
    return {
        "A": _authority_bundle(
            variant_code="A",
            authority_id="fresh_rollout_v21_strong",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/a_hard_seed_strong_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 5, 5, False),
                (17, 5, 5, False),
                (27, 5, 5, False),
                (37, 5, 5, False),
            ],
        ),
        "B": _authority_bundle(
            variant_code="B",
            authority_id="fresh_rollout_v21_strong",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/b_hard_seed_strong_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 10, 10, False),
                (17, 10, 10, False),
                (27, 10, 10, False),
                (37, 10, 10, False),
            ],
        ),
        "C": _authority_bundle(
            variant_code="C",
            authority_id="fresh_rollout_v21_strong",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/c_hard_seed_strong_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 5, 5, False),
                (17, 5, 5, False),
                (27, 5, 5, False),
                (37, 5, 5, False),
            ],
        ),
        "X": _authority_bundle(
            variant_code="X",
            authority_id="fresh_rollout_v21_strong",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/x_hard_seed_strong_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 14, 14, False),
                (17, 14, 14, False),
                (27, 14, 14, False),
                (37, 14, 14, False),
            ],
        ),
    }


def _strong_headroom_recovered_bundles(
    *, selection_source: str, selection_source_hash: str
) -> dict[str, dict[str, object]]:
    return {
        "A": _authority_bundle(
            variant_code="A",
            authority_id="fresh_rollout_v21_strong",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/a_hard_seed_strong_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 12, 12, False),
                (17, 12, 12, False),
                (27, 12, 12, False),
                (37, 12, 12, False),
            ],
        ),
        "B": _authority_bundle(
            variant_code="B",
            authority_id="fresh_rollout_v21_strong",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/b_hard_seed_strong_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 14, 14, False),
                (17, 14, 14, False),
                (27, 14, 14, False),
                (37, 14, 14, False),
            ],
        ),
        "C": _authority_bundle(
            variant_code="C",
            authority_id="fresh_rollout_v21_strong",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/c_hard_seed_strong_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 5, 5, False),
                (17, 5, 5, False),
                (27, 5, 5, False),
                (37, 5, 5, False),
            ],
        ),
        "X": _authority_bundle(
            variant_code="X",
            authority_id="fresh_rollout_v21_strong",
            authority_dir="agent/artifacts/openpi_libero_v21/runs/x_hard_seed_strong_v21",
            selection_source=selection_source,
            selection_source_hash=selection_source_hash,
            episodes=[
                (7, 14, 14, False),
                (17, 14, 14, False),
                (27, 14, 14, False),
                (37, 14, 14, False),
            ],
        ),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _materialize_fixture_json(
    tmp_path: Path,
    *,
    strong_bundles: Mapping[str, Mapping[str, object]],
) -> tuple[Path, Path, Mapping[str, object], Mapping[str, object]]:
    selection_source, selection_source_hash = _materialize_fixture_selection_source(
        tmp_path
    )
    lite_selection_bundles = _lite_selection_bundles(
        selection_source=selection_source,
        selection_source_hash=selection_source_hash,
    )
    payload = _mapping(
        build_libero_abcx_gate_artifacts_v21(
            variant_authorities=strong_bundles,
            authority_mode="strong",
            selection_variant_authorities=lite_selection_bundles,
        )
    )
    paired_summary = dict(_mapping(payload["paired_summary"]))
    report = dict(_mapping(payload["go_no_go_report"]))
    paired_summary["output_path"] = CANONICAL_PAIRED_SUMMARY_PATH
    report["paired_summary_path"] = CANONICAL_PAIRED_SUMMARY_PATH
    report["output_path"] = CANONICAL_GO_NO_GO_PATH
    paired_summary_path = tmp_path / "paired_summary_abcx_v21.json"
    go_no_go_path = tmp_path / "go_no_go_v21.json"
    _write_json(paired_summary_path, paired_summary)
    _write_json(go_no_go_path, report)
    return paired_summary_path, go_no_go_path, paired_summary, report


def test_publisher_renders_machine_checkable_not_decision_capable_fixture(
    tmp_path: Path,
) -> None:
    paired_summary_path, go_no_go_path, paired_summary, report = (
        _materialize_fixture_json(
            tmp_path,
            strong_bundles=_strong_not_decision_capable_bundles(
                selection_source=str(tmp_path / "stock_seed_scan_summary.json"),
                selection_source_hash=CANONICAL_SELECTION_SOURCE_HASH,
            ),
        )
    )
    generated_results_doc = tmp_path / "openpi_libero_v21_results.md"
    generated_entry_doc = tmp_path / "openpi_libero_v22_entry_prereqs.md"

    _ = publish_v21_results(
        paired_summary_json_path=paired_summary_path,
        go_no_go_json_path=go_no_go_path,
        results_doc_path=generated_results_doc,
        entry_doc_path=generated_entry_doc,
    )

    generated_results_text = generated_results_doc.read_text(encoding="utf-8")
    generated_entry_text = generated_entry_doc.read_text(encoding="utf-8")
    gates = _mapping(report["gates"])
    variants = _mapping(paired_summary["variants"])
    pairwise_delta = _mapping(paired_summary["pairwise_delta"])
    pairs = _mapping(pairwise_delta["pairs"])
    primary_metric_id = str(paired_summary["primary_metric_id"])

    required_results_items = [
        "openpi LIBERO v21 desaturation 结果",
        CANONICAL_GO_NO_GO_PATH,
        CANONICAL_PAIRED_SUMMARY_PATH,
        f"authority_mode={report['authority_mode']}",
        "eval_authority=fresh_rollout_v21_strong",
        "task_suite_name=libero_spatial",
        "task_ids=[0,1]",
        "headline_variant_codes=[A,B,C]",
        "diagnostic_variant_codes=[X]",
        f"primary_metric_id={primary_metric_id}",
        f"headroom_gate_decision={_mapping(gates['H2'])['decision_text']}",
        "state_side_publication_status=STATE_SIDE_STILL_FROZEN",
        f"headroom_recovered={str(report['headroom_recovered']).lower()}",
        f"recap_validated_on_desaturated_eval={str(report['recap_validated_on_desaturated_eval']).lower()}",
        f"informativeness_validated={str(report['informativeness_validated']).lower()}",
        f"eligible_for_state_side_v22={str(report['eligible_for_state_side_v22']).lower()}",
        "X.diagnostic_only=true",
        "D.not_executed=true",
        "- H2=FAIL",
        "- H3=PASS",
        "- H4=PASS",
        "- H5=FAIL",
        "- H7=FAIL",
        "A.authority_dir=agent/artifacts/openpi_libero_v21/runs/a_hard_seed_strong_v21",
        "B.authority_dir=agent/artifacts/openpi_libero_v21/runs/b_hard_seed_strong_v21",
        "C.authority_dir=agent/artifacts/openpi_libero_v21/runs/c_hard_seed_strong_v21",
        "X.authority_dir=agent/artifacts/openpi_libero_v21/runs/x_hard_seed_strong_v21",
        f"A.selected_metric_point_estimate={_mapping(variants['A'])['selected_metric_point_estimate']}",
        f"B.selected_metric_point_estimate={_mapping(variants['B'])['selected_metric_point_estimate']}",
        f"C.selected_metric_point_estimate={_mapping(variants['C'])['selected_metric_point_estimate']}",
        f"X.selected_metric_point_estimate={_mapping(variants['X'])['selected_metric_point_estimate']}",
        "A.throughput_ci_non_degenerate=false",
        "B.throughput_ci_non_degenerate=false",
        "C.throughput_ci_non_degenerate=false",
        "X.throughput_ci_non_degenerate=false",
        f"A.selection_source={str(tmp_path / 'stock_seed_scan_summary.json')}",
        f"C-B.delta={_metric_payload(pairs=pairs, pair_label='C-B', metric_id=primary_metric_id)['delta']}",
        f"C-X.delta={_metric_payload(pairs=pairs, pair_label='C-X', metric_id=primary_metric_id)['delta']}",
        f"C-A.compatibility_delta@1.00_budget={_metric_payload(pairs=pairs, pair_label='C-A', metric_id='success_rate@1.00_budget')['delta']}",
        f"H2.decision_text={_mapping(gates['H2'])['decision_text']}",
        f"H5.decision_text={_mapping(gates['H5'])['decision_text']}",
        f"H7.decision_text={_mapping(gates['H7'])['decision_text']}",
    ]
    for item in required_results_items:
        assert item in generated_results_text, (
            f"missing required fixture v21 results item: {item}"
        )

    required_entry_items = [
        "openpi LIBERO v22 state-side 入口前提",
        "本文只回答一个问题：`eligible_for_state_side_v22=true/false`。",
        CANONICAL_GO_NO_GO_PATH,
        "H2=FAIL",
        "H4=PASS",
        "H5=FAIL",
        "H6=PASS",
        "H7=FAIL",
        f"headroom_recovered={str(report['headroom_recovered']).lower()}",
        f"recap_validated_on_desaturated_eval={str(report['recap_validated_on_desaturated_eval']).lower()}",
        f"informativeness_validated={str(report['informativeness_validated']).lower()}",
        f"eligible_for_state_side_v22={str(report['eligible_for_state_side_v22']).lower()}",
        "D_not_executed_in_v21=true",
        "state_side_publication_status=STATE_SIDE_STILL_FROZEN",
        f"headroom_gate_decision={_mapping(gates['H2'])['decision_text']}",
        f"decision_text={_mapping(gates['H2'])['decision_text']}",
        f"decision_text={_mapping(gates['H5'])['decision_text']}",
        f"decision_text={_mapping(gates['H7'])['decision_text']}",
    ]
    for item in required_entry_items:
        assert item in generated_entry_text, (
            f"missing required fixture v22 entry item: {item}"
        )

    forbidden_results_items = [
        "STATE_SIDE_ENTRY_PREREQS_MET",
        "eligible_for_state_side_v22=true",
    ]
    for item in forbidden_results_items:
        assert item not in generated_results_text, (
            f"unexpected fixture item in generated results doc: {item}"
        )


def test_repo_docs_match_live_strong_artifacts_and_use_repo_relative_paths(
    tmp_path: Path,
) -> None:
    generated_results_doc = tmp_path / "openpi_libero_v21_results.md"
    generated_entry_doc = tmp_path / "openpi_libero_v22_entry_prereqs.md"
    paired_summary = _load_json(LIVE_PAIRED_SUMMARY_JSON)
    report = _load_json(LIVE_GO_NO_GO_JSON)

    _ = publish_v21_results(
        paired_summary_json_path=LIVE_PAIRED_SUMMARY_JSON,
        go_no_go_json_path=LIVE_GO_NO_GO_JSON,
        results_doc_path=generated_results_doc,
        entry_doc_path=generated_entry_doc,
    )

    repo_results_text = RESULTS_DOC.read_text(encoding="utf-8")
    repo_entry_text = ENTRY_DOC.read_text(encoding="utf-8")
    generated_results_text = generated_results_doc.read_text(encoding="utf-8")
    generated_entry_text = generated_entry_doc.read_text(encoding="utf-8")
    gates = _mapping(report["gates"])
    variants = _mapping(paired_summary["variants"])
    pairwise_delta = _mapping(paired_summary["pairwise_delta"])
    pairs = _mapping(pairwise_delta["pairs"])
    primary_metric_id = str(paired_summary["primary_metric_id"])
    selection_source_hash = str(_mapping(variants["A"])["selection_source_hash"])

    assert repo_results_text == generated_results_text
    assert repo_entry_text == generated_entry_text

    required_results_items = [
        "openpi LIBERO v21 desaturation 结果",
        CANONICAL_GO_NO_GO_PATH,
        CANONICAL_PAIRED_SUMMARY_PATH,
        f"authority_mode={report['authority_mode']}",
        "eval_authority=fresh_rollout_v21_strong",
        "task_suite_name=libero_spatial",
        "task_ids=[0,1]",
        f"primary_metric_id={primary_metric_id}",
        f"headroom_gate_decision={_mapping(gates['H2'])['decision_text']}",
        f"headroom_recovered={str(report['headroom_recovered']).lower()}",
        f"recap_validated_on_desaturated_eval={str(report['recap_validated_on_desaturated_eval']).lower()}",
        f"informativeness_validated={str(report['informativeness_validated']).lower()}",
        f"eligible_for_state_side_v22={str(report['eligible_for_state_side_v22']).lower()}",
        "state_side_publication_status=STATE_SIDE_STILL_FROZEN",
        "X.diagnostic_only=true",
        "D.not_executed=true",
        f"- H2={_mapping(gates['H2'])['status']}",
        f"- H4={_mapping(gates['H4'])['status']}",
        f"- H5={_mapping(gates['H5'])['status']}",
        f"- H7={_mapping(gates['H7'])['status']}",
        "A.authority_dir=agent/artifacts/openpi_libero_v21/runs/a_hard_seed_strong_v21",
        "B.authority_dir=agent/artifacts/openpi_libero_v21/runs/b_hard_seed_strong_v21",
        "C.authority_dir=agent/artifacts/openpi_libero_v21/runs/c_hard_seed_strong_v21",
        "X.authority_dir=agent/artifacts/openpi_libero_v21/runs/x_hard_seed_strong_v21",
        f"A.selected_metric_point_estimate={_mapping(variants['A'])['selected_metric_point_estimate']}",
        f"B.selected_metric_point_estimate={_mapping(variants['B'])['selected_metric_point_estimate']}",
        f"C.selected_metric_point_estimate={_mapping(variants['C'])['selected_metric_point_estimate']}",
        f"X.selected_metric_point_estimate={_mapping(variants['X'])['selected_metric_point_estimate']}",
        f"A.throughput_ci_non_degenerate={str(_mapping(variants['A'])['throughput_ci_non_degenerate']).lower()}",
        f"B.throughput_ci_non_degenerate={str(_mapping(variants['B'])['throughput_ci_non_degenerate']).lower()}",
        f"C.throughput_ci_non_degenerate={str(_mapping(variants['C'])['throughput_ci_non_degenerate']).lower()}",
        f"X.throughput_ci_non_degenerate={str(_mapping(variants['X'])['throughput_ci_non_degenerate']).lower()}",
        f"A.selection_source_hash={selection_source_hash}",
        f"C-B.delta={_metric_payload(pairs=pairs, pair_label='C-B', metric_id=primary_metric_id)['delta']}",
        f"C-X.delta={_metric_payload(pairs=pairs, pair_label='C-X', metric_id=primary_metric_id)['delta']}",
        f"C-A.compatibility_delta@1.00_budget={_metric_payload(pairs=pairs, pair_label='C-A', metric_id='success_rate@1.00_budget')['delta']}",
        f"H4.decision_text={_mapping(gates['H4'])['decision_text']}",
        f"H5.decision_text={_mapping(gates['H5'])['decision_text']}",
        f"H7.decision_text={_mapping(gates['H7'])['decision_text']}",
    ]
    for item in required_results_items:
        assert item in repo_results_text, f"missing live strong results item: {item}"

    required_entry_items = [
        "openpi LIBERO v22 state-side 入口前提",
        CANONICAL_GO_NO_GO_PATH,
        f"H2={_mapping(gates['H2'])['status']}",
        f"H4={_mapping(gates['H4'])['status']}",
        f"H5={_mapping(gates['H5'])['status']}",
        f"H6={_mapping(gates['H6'])['status']}",
        f"H7={_mapping(gates['H7'])['status']}",
        f"headroom_recovered={str(report['headroom_recovered']).lower()}",
        f"recap_validated_on_desaturated_eval={str(report['recap_validated_on_desaturated_eval']).lower()}",
        f"informativeness_validated={str(report['informativeness_validated']).lower()}",
        f"eligible_for_state_side_v22={str(report['eligible_for_state_side_v22']).lower()}",
        "D_not_executed_in_v21=true",
        "state_side_publication_status=STATE_SIDE_STILL_FROZEN",
        f"headroom_gate_decision={_mapping(gates['H2'])['decision_text']}",
        f"decision_text={_mapping(gates['H4'])['decision_text']}",
        f"decision_text={_mapping(gates['H5'])['decision_text']}",
        f"decision_text={_mapping(gates['H7'])['decision_text']}",
    ]
    for item in required_entry_items:
        assert item in repo_entry_text, f"missing live strong entry item: {item}"

    forbidden_absolute_prefix = str(REPO_ROOT)
    assert forbidden_absolute_prefix not in repo_results_text
    assert forbidden_absolute_prefix not in repo_entry_text
    assert forbidden_absolute_prefix not in generated_results_text
    assert forbidden_absolute_prefix not in generated_entry_text


def test_publisher_handles_headroom_recovered_and_state_side_ready(
    tmp_path: Path,
) -> None:
    paired_summary_path, go_no_go_path, _, report = _materialize_fixture_json(
        tmp_path,
        strong_bundles=_strong_headroom_recovered_bundles(
            selection_source=str(tmp_path / "stock_seed_scan_summary.json"),
            selection_source_hash=CANONICAL_SELECTION_SOURCE_HASH,
        ),
    )
    generated_results_doc = tmp_path / "openpi_libero_v21_results.md"
    generated_entry_doc = tmp_path / "openpi_libero_v22_entry_prereqs.md"

    _ = publish_v21_results(
        paired_summary_json_path=paired_summary_path,
        go_no_go_json_path=go_no_go_path,
        results_doc_path=generated_results_doc,
        entry_doc_path=generated_entry_doc,
    )

    results_text = generated_results_doc.read_text(encoding="utf-8")
    entry_text = generated_entry_doc.read_text(encoding="utf-8")
    gates = _mapping(report["gates"])
    gate_statuses: dict[str, str] = {}
    for name in _sequence(report["gate_order"]):
        gate_name = str(name)
        gate_statuses[gate_name] = str(_mapping(gates[gate_name])["status"])

    assert report["headroom_recovered"] is True
    assert report["eligible_for_state_side_v22"] is True
    assert gate_statuses["H2"] == "PASS"
    assert gate_statuses["H4"] == "PASS"
    assert gate_statuses["H5"] == "PASS"
    assert gate_statuses["H6"] == "PASS"
    assert gate_statuses["H7"] == "PASS"
    assert (
        f"headroom_gate_decision={_mapping(gates['H2'])['decision_text']}"
        in results_text
    )
    assert "state_side_publication_status=STATE_SIDE_ENTRY_PREREQS_MET" in results_text
    assert "eligible_for_state_side_v22=true" in results_text
    assert "state_side_publication_status=STATE_SIDE_ENTRY_PREREQS_MET" in entry_text
    assert "eligible_for_state_side_v22=true" in entry_text
    assert "EVAL_SLICE_NOT_DECISION_CAPABLE_ON_TASKS_0_1" not in results_text
    assert "STATE_SIDE_STILL_FROZEN" not in entry_text
