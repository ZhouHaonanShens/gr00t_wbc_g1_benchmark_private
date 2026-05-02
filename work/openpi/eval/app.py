from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import datetime as dt
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from work.openpi.checkpoint import (
    CheckpointProvenanceLoader,
    load_checkpoint_provenance_pair,
    load_provenance_pair,
)
from work.openpi.contracts import (
    ExecutedRuntimeBinding,
    RequestedRuntimeBinding,
    TraceBuildOutputs,
    V2AuthorityPaths,
    V2EvalRequest,
    V21AuthorityPaths,
    V21EvalRequest,
)
from work.openpi.dataloader import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_markdown,
)
from work.openpi.eval.scenarios import (
    RolloutEvaluationScenario,
    StockSmokeScenario,
    TrackedGateEvaluationScenario,
)
from work.openpi.eval.workflows.tracked_gate import (
    derive_go_no_go_core_from_authority_bundle,
)
from work.openpi.metrics import (
    AggregationValidationError,
    COMPATIBILITY_ONLY_METRICS,
    DEFAULT_PRIMARY_METRIC_ID,
    HEADLINE_METRIC_ORDER,
    assert_variant_aggregate_conservation_v21,
    build_bootstrap_ci_v21,
    build_v21_metric_ladder_summary,
)
from work.openpi.model import (
    effective_runtime_spec_from_runtime_prompting,
    effective_runtime_spec_hash,
    effective_runtime_surface_signature,
    normalize_effective_runtime_spec,
    normalize_runtime_prompting_payload,
)


def _tracked_gate_impl() -> Any:
    return importlib.import_module("work.openpi.eval.workflows.tracked_gate")


def _rollout_support() -> Any:
    return importlib.import_module("work.openpi.eval.workflows.rollout_support")


@dataclass
class TrackedGateEvaluationWorkflow:
    """Materialize tracked-gate authority artifacts from a canonical scenario.

    This workflow keeps the v2 tracked-gate path thin at the surface and pushes
    the actual contract checks, rollout source resolution, and authority-bundle
    writes through one canonical orchestration point.
    """

    scenario: TrackedGateEvaluationScenario

    def run(self) -> int:
        impl = _tracked_gate_impl()
        args = self._build_namespace()
        request = self._resolve_request(args, impl)
        source_dir = self._ensure_rollout_source_dir(args, request, impl)
        bundle = self._build_authority_bundle(request, source_dir, impl)
        self._write_authority_bundle(request, bundle, impl)
        return 0

    def _build_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            variant=self.scenario.checkpoint.variant,
            checkpoint_dir=self.scenario.checkpoint.checkpoint_ref,
            eval_manifest=str(self.scenario.eval_manifest_path),
            baseline_variant=self.scenario.checkpoint.baseline_variant,
            output_root=str(self.scenario.artifacts.output_root),
        )

    def _resolve_request(self, args: SimpleNamespace, impl: Any) -> V2EvalRequest:
        variant = impl._normalize_variant(str(args.variant))
        baseline_variant = impl._normalize_variant(str(args.baseline_variant))
        raw_checkpoint_dir = str(args.checkpoint_dir)
        impl._reject_legacy_authority_input(raw_checkpoint_dir)
        checkpoint_ref = impl._normalize_checkpoint_ref(raw_checkpoint_dir)
        eval_manifest = impl.manifest_payload_v2(
            impl.load_rollout_eval_manifest_v2(str(args.eval_manifest))
        )
        eval_manifest_hash = impl.compute_rollout_eval_manifest_hash(eval_manifest)
        eval_manifest_id = impl.build_eval_manifest_id(eval_manifest)
        raw_paths = impl._build_artifact_paths(
            output_root=str(args.output_root),
            variant=variant,
            eval_manifest_id=eval_manifest_id,
            baseline_variant=baseline_variant,
        )
        paths = V2AuthorityPaths.from_mapping(raw_paths)
        paths.artifact_dir.mkdir(parents=True, exist_ok=True)
        paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        impl._log(
            f"[{dt.datetime.now().isoformat(timespec='seconds')}] rollout_eval_v2 variant={variant} manifest={eval_manifest_id}",
            log_path=paths.log_path,
        )
        return V2EvalRequest(
            variant=variant,
            baseline_variant=baseline_variant,
            checkpoint_ref=checkpoint_ref,
            eval_manifest=eval_manifest,
            eval_manifest_hash=eval_manifest_hash,
            eval_manifest_id=eval_manifest_id,
            paths=paths,
        )

    def _ensure_rollout_source_dir(
        self,
        args: SimpleNamespace,
        request: V2EvalRequest,
        impl: Any,
    ) -> Path:
        source_dir = impl._ensure_rollout_source_dir(
            checkpoint_ref=request.checkpoint_ref,
            output_root=str(args.output_root),
            variant=request.variant,
            eval_manifest_id=request.eval_manifest_id,
            eval_manifest=request.eval_manifest,
            artifact_dir=request.paths.artifact_dir,
            runtime_dir=request.paths.runtime_dir,
            log_path=request.paths.log_path,
        )
        impl._log(f"source_rollout_dir={source_dir}", log_path=request.paths.log_path)
        return source_dir

    def _build_authority_bundle(
        self,
        request: V2EvalRequest,
        source_dir: Path,
        impl: Any,
    ) -> dict[str, object]:
        input_rows = read_jsonl(source_dir / impl.PER_EPISODE_NAME)
        canonical_rows, scope_audit = impl._validate_and_canonicalize_episode_rollouts(
            rows=input_rows,
            eval_manifest=request.eval_manifest,
            variant=request.variant,
            checkpoint_ref=request.checkpoint_ref,
        )
        train_manifest, checkpoint_provenance = load_provenance_pair(source_dir)
        train_provenance = impl._extract_train_provenance(
            variant=request.variant,
            train_manifest=train_manifest,
            checkpoint_provenance=checkpoint_provenance,
        )
        deviation_notes = impl._build_deviation_notes(
            variant=request.variant,
            baseline_variant=request.baseline_variant,
            source_dir=source_dir,
            train_provenance=train_provenance,
        )
        bootstrap_ci = impl._bootstrap_success_rate(
            canonical_rows=canonical_rows,
            eval_manifest=request.eval_manifest,
            variant=request.variant,
        )
        video_index = impl._build_video_index(
            canonical_rows=canonical_rows,
            eval_manifest_id=request.eval_manifest_id,
            variant=request.variant,
        )
        current_summary_row = impl._summary_row(
            variant=request.variant,
            checkpoint_ref=request.checkpoint_ref,
            scope_audit=scope_audit,
        )
        baseline_summary = impl._load_baseline_summary(
            paths=request.paths.as_impl_mapping(),
            variant=request.variant,
            baseline_variant=request.baseline_variant,
            current_summary={
                "schema_version": impl.SUMMARY_SCHEMA_VERSION,
                "eval_authority": impl.EXPECTED_EVAL_AUTHORITY,
                "checkpoint_dir": request.checkpoint_ref,
                "rollout_summary": current_summary_row,
            },
        )
        paired_delta = impl._build_paired_delta(
            variant=request.variant,
            baseline_variant=request.baseline_variant,
            eval_manifest_id=request.eval_manifest_id,
            checkpoint_ref=request.checkpoint_ref,
            current_summary_row=current_summary_row,
            baseline_summary_payload=baseline_summary,
        )
        eval_manifest_payload = {
            **request.eval_manifest,
            "eval_manifest_hash": request.eval_manifest_hash,
            "eval_manifest_id": request.eval_manifest_id,
        }
        summary = {
            "schema_version": impl.SUMMARY_SCHEMA_VERSION,
            "eval_authority": impl.EXPECTED_EVAL_AUTHORITY,
            "variant": request.variant,
            "baseline_variant": request.baseline_variant,
            "checkpoint_dir": request.checkpoint_ref,
            "output_dir": str(request.paths.artifact_dir),
            "runtime_dir": str(request.paths.runtime_dir),
            "source_rollout_dir": str(source_dir),
            "eval_manifest_id": request.eval_manifest_id,
            "eval_manifest_hash": request.eval_manifest_hash,
            "manifest_name": str(request.eval_manifest["manifest_name"]),
            "task_suite_name": str(request.eval_manifest["task_suite_name"]),
            "rollout_summary": {
                **current_summary_row,
                "episode_count": impl._coerce_int(
                    scope_audit.get("observed_episode_count"),
                    context="scope_audit.observed_episode_count",
                ),
            },
            "scope_audit": scope_audit,
            "train_provenance": train_provenance,
            "required_outputs": request.paths.required_outputs(),
            "deviation_notes": list(deviation_notes),
        }
        authority_bundle = {
            "authority_dir": str(request.paths.artifact_dir),
            "summary": summary,
            "eval_manifest": eval_manifest_payload,
            "per_episode_rollouts": canonical_rows,
            "video_index": video_index,
            "bootstrap_ci": bootstrap_ci,
            "paired_delta": paired_delta,
            "deviation_notes": impl._deviation_notes_markdown(deviation_notes),
        }
        return {
            "summary": summary,
            "eval_manifest_payload": eval_manifest_payload,
            "canonical_rows": canonical_rows,
            "video_index": video_index,
            "bootstrap_ci": bootstrap_ci,
            "paired_delta": paired_delta,
            "deviation_notes": deviation_notes,
            "authority_bundle": authority_bundle,
        }

    def _write_authority_bundle(
        self,
        request: V2EvalRequest,
        bundle: Mapping[str, object],
        impl: Any,
    ) -> None:
        write_json(request.paths.eval_manifest, bundle["eval_manifest_payload"])
        write_jsonl(
            request.paths.per_episode,
            cast(list[dict[str, object]], bundle["canonical_rows"]),
            sort_keys=True,
        )
        write_json(request.paths.video_index, bundle["video_index"])
        write_json(request.paths.bootstrap_ci, bundle["bootstrap_ci"])
        write_markdown(
            request.paths.deviation_notes,
            impl._deviation_notes_markdown(bundle["deviation_notes"]),
        )
        write_json(request.paths.paired_delta, bundle["paired_delta"])
        summary_payload = dict(cast(dict[str, object], bundle["summary"]))
        write_json(request.paths.summary, summary_payload)
        authority_bundle = dict(cast(dict[str, object], bundle["authority_bundle"]))
        authority_bundle["summary"] = summary_payload
        summary_payload["go_no_go_core"] = derive_go_no_go_core_from_authority_bundle(
            authority_bundle
        )
        write_json(request.paths.summary, summary_payload)
        impl._log(
            f"summary_json={request.paths.summary}",
            log_path=request.paths.log_path,
        )
        impl._log("LIBERO_ROLLOUT_EVAL_V2_DONE", log_path=request.paths.log_path)


@dataclass
class RolloutEvaluationWorkflow:
    """Run the canonical v21 rollout-evaluation pipeline for one scenario.

    The workflow owns the request-to-artifact path: normalize scenario inputs,
    resolve requested and executed runtime bindings, build trace-derived
    summaries, and write the machine-readable authority bundle consumed by the
    rest of the mainline.
    """

    scenario: RolloutEvaluationScenario
    dependencies: Any
    impl: Any

    def run(self) -> int:
        args = self._build_namespace()
        inputs = self._resolve_inputs(args)
        requested_runtime = self._resolve_requested_runtime_binding(inputs, args)
        source_dir = self._ensure_rollout_source_dir(inputs, requested_runtime)
        self.impl._log(f"source_rollout_dir={source_dir}", log_path=inputs.log_path)
        trace_outputs = self._build_trace_outputs(inputs, source_dir)
        executed_runtime = self._load_executed_runtime_binding(
            inputs,
            source_dir,
            requested_runtime,
        )
        summary = self._build_summary(
            inputs=inputs,
            source_dir=source_dir,
            trace_outputs=trace_outputs,
            requested_runtime=requested_runtime,
            executed_runtime=executed_runtime,
        )
        self._write_authority_bundle(
            inputs=inputs,
            trace_outputs=trace_outputs,
            summary=summary,
        )
        return 0

    def _build_namespace(self) -> SimpleNamespace:
        checkpoint_source = self.scenario.checkpoint.checkpoint_source or ""
        checkpoint_dir = self.scenario.checkpoint.checkpoint_ref or ""
        return SimpleNamespace(
            variant=self.scenario.checkpoint.variant,
            checkpoint_source=checkpoint_source,
            checkpoint_dir=checkpoint_dir,
            manifest=str(self.scenario.manifest_path),
            metric_profile=self.scenario.metrics.metric_profile,
            output_dir=str(self.scenario.output_dir),
            indicator_mode=self.scenario.indicator_mode,
            resolved_runtime_indicator_mode=self.scenario.resolved_runtime_indicator_mode,
            resolved_runtime_indicator_source=self.scenario.resolved_runtime_indicator_source,
            resolved_runtime_consumer_mode=self.scenario.resolved_runtime_consumer_mode,
            resolved_runtime_fixed_indicator_mode=self.scenario.resolved_runtime_fixed_indicator_mode,
            resolved_runtime_critic_checkpoint_ref=self.scenario.resolved_runtime_critic_checkpoint_ref,
            canonical_source_dir=(
                str(self.scenario.canonical_source_dir)
                if self.scenario.canonical_source_dir is not None
                else ""
            ),
        )

    def _resolve_inputs(self, args: SimpleNamespace) -> V21EvalRequest:
        variant = self.impl._normalize_variant(str(args.variant))
        eval_manifest = self.impl._normalize_eval_manifest_payload(
            self.impl.load_rollout_manifest_v21(str(args.manifest))
        )
        metric_profile = self.impl._require_non_empty_str(
            args.metric_profile,
            context="scenario.metric_profile",
        )
        manifest_metric_profile = self.impl._require_non_empty_str(
            eval_manifest.get("metric_profile"),
            context="eval_manifest.metric_profile",
        )
        if metric_profile != manifest_metric_profile:
            raise self.impl.FailFastError(
                "metric_profile mismatch: "
                + f"scenario={metric_profile!r} manifest={manifest_metric_profile!r}"
            )
        variant_scope = {
            str(item).strip()
            for item in self.impl._require_sequence(
                eval_manifest.get("variant_scope"),
                context="eval_manifest.variant_scope",
            )
        }
        if variant not in variant_scope:
            raise self.impl.FailFastError(
                f"variant {variant!r} is outside manifest variant_scope={sorted(variant_scope)!r}"
            )
        checkpoint_ref, checkpoint_mode = self.impl._resolve_checkpoint_input(
            args,
            variant,
        )
        raw_checkpoint_dir = (
            str(getattr(args, "checkpoint_dir", "") or "").strip() or None
        )
        output_dir = Path(str(args.output_dir)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_manifest_hash = self.impl.compute_rollout_manifest_hash_v21(eval_manifest)
        eval_manifest_id = self.impl.build_eval_manifest_id(eval_manifest)
        runtime_dir = self.dependencies.build_runtime_dir(variant, eval_manifest_id)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = runtime_dir / "eval.log"
        self.impl._log(
            f"[{dt.datetime.now().isoformat(timespec='seconds')}] rollout_eval variant={variant} manifest={eval_manifest_id}",
            log_path=log_path,
        )
        canonical_source_dir = (
            Path(str(getattr(args, "canonical_source_dir", "") or "")).resolve()
            if str(getattr(args, "canonical_source_dir", "") or "").strip()
            else None
        )
        return V21EvalRequest(
            variant=variant,
            eval_manifest=eval_manifest,
            manifest_metric_profile=manifest_metric_profile,
            checkpoint_ref=checkpoint_ref,
            checkpoint_mode=checkpoint_mode,
            raw_checkpoint_dir=raw_checkpoint_dir,
            output_dir=output_dir,
            eval_manifest_hash=eval_manifest_hash,
            eval_manifest_id=eval_manifest_id,
            runtime_dir=runtime_dir,
            log_path=log_path,
            indicator_mode_requested=str(args.indicator_mode),
            canonical_source_dir=canonical_source_dir,
            output_paths=V21AuthorityPaths.from_output_dir(
                output_dir,
                trace_name=self.impl.TRACE_NAME,
                metric_ladder_name=self.impl.METRIC_LADDER_NAME,
                bootstrap_name=self.impl.BOOTSTRAP_NAME,
                pairwise_delta_name=self.impl.PAIRWISE_DELTA_NAME,
                summary_name=self.impl.SUMMARY_NAME,
                eval_manifest_name=self.impl.EVAL_MANIFEST_NAME,
                deviation_notes_name=self.impl.DEVIATION_NOTES_NAME,
            ),
        )

    def _resolve_requested_runtime_binding(
        self,
        inputs: V21EvalRequest,
        args: SimpleNamespace,
    ) -> RequestedRuntimeBinding:
        train_manifest, checkpoint_provenance = load_checkpoint_provenance_pair(
            checkpoint_ref=inputs.checkpoint_ref,
            raw_checkpoint_dir=inputs.raw_checkpoint_dir,
        )
        runtime_indicator_config = self.dependencies.runtime_indicator_config_from_args(
            args,
            inputs.variant,
            train_manifest,
            checkpoint_provenance,
        )
        prompt_surface_bundle = self.impl.build_runtime_prompt_bundle(
            "runtime prompt surface preview",
            config=runtime_indicator_config,
        )
        return RequestedRuntimeBinding(
            runtime_indicator_config=runtime_indicator_config,
            prompt_surface_bundle=prompt_surface_bundle,
            runtime_prompting=self.impl._expected_runtime_prompting_payload(
                runtime_indicator_config=runtime_indicator_config,
                prompt_surface_bundle=prompt_surface_bundle,
            ),
            effective_runtime_spec=self.impl.build_effective_runtime_spec(
                variant=inputs.variant,
                checkpoint_ref=inputs.checkpoint_ref,
                runtime_indicator_config=runtime_indicator_config,
                prompt_surface_bundle=prompt_surface_bundle,
            ),
        )

    def _ensure_rollout_source_dir(
        self,
        inputs: V21EvalRequest,
        requested_runtime: RequestedRuntimeBinding,
    ) -> Path:
        return self.dependencies.ensure_rollout_source_dir(
            inputs.checkpoint_ref,
            inputs.checkpoint_mode,
            inputs.raw_checkpoint_dir,
            inputs.indicator_mode_requested,
            inputs.variant,
            inputs.eval_manifest,
            inputs.output_dir,
            inputs.runtime_dir,
            inputs.log_path,
            requested_runtime.runtime_indicator_config,
            requested_runtime.prompt_surface_bundle,
            inputs.canonical_source_dir,
            self.scenario.runtime.host,
            self.scenario.runtime.port,
            self.scenario.runtime.server_ready_timeout_s,
            self.scenario.runtime.client_timeout_s,
        )

    def _build_trace_outputs(
        self,
        inputs: V21EvalRequest,
        source_dir: Path,
    ) -> TraceBuildOutputs:
        input_rows = read_jsonl(source_dir / self.impl.V2_INPUT_PER_EPISODE_NAME)
        trace_rows, scope_audit = self.dependencies.validate_and_build_trace_rows(
            input_rows,
            inputs.eval_manifest,
            inputs.variant,
        )
        metric_ladder_summary = build_v21_metric_ladder_summary(
            trace_rows=trace_rows,
            authority_id=str(inputs.eval_manifest["authority_id"]),
            variant=inputs.variant,
            checkpoint_ref=inputs.checkpoint_ref,
            metric_profile=inputs.manifest_metric_profile,
            primary_metric_id=DEFAULT_PRIMARY_METRIC_ID,
        )
        bootstrap_ci = build_bootstrap_ci_v21(
            trace_rows=trace_rows,
            variant=inputs.variant,
            deterministic_seed_material=f"{inputs.variant}:{inputs.eval_manifest_hash}",
        )
        pairwise_delta = self.impl._build_pairwise_delta(
            trace_rows=trace_rows,
            variant=inputs.variant,
            checkpoint_ref=inputs.checkpoint_ref,
        )
        deviation_notes = self.impl._build_deviation_notes(
            variant=inputs.variant,
            checkpoint_mode=inputs.checkpoint_mode,
            checkpoint_ref=inputs.checkpoint_ref,
            source_dir=source_dir,
            trace_rows=trace_rows,
        )
        return TraceBuildOutputs(
            trace_rows=trace_rows,
            scope_audit=scope_audit,
            metric_ladder_summary=metric_ladder_summary,
            bootstrap_ci=bootstrap_ci,
            pairwise_delta=pairwise_delta,
            deviation_notes=deviation_notes,
        )

    def _load_executed_runtime_binding(
        self,
        inputs: V21EvalRequest,
        source_dir: Path,
        requested_runtime: RequestedRuntimeBinding,
    ) -> ExecutedRuntimeBinding:
        source_rollout_input_summary_path = (
            source_dir / self.impl.ROLLOUT_INPUT_SUMMARY_NAME
        )
        if not source_rollout_input_summary_path.is_file():
            return ExecutedRuntimeBinding(
                runtime_prompting=requested_runtime.runtime_prompting,
                effective_runtime_spec=requested_runtime.effective_runtime_spec,
            )
        source_rollout_input_summary = read_json(source_rollout_input_summary_path)
        try:
            executed_runtime_prompting = normalize_runtime_prompting_payload(
                self.impl._require_mapping(
                    source_rollout_input_summary.get("runtime_prompting"),
                    context="source_rollout_input_summary.runtime_prompting",
                ),
                context="source_rollout_input_summary.runtime_prompting",
            )
            raw_executed_effective_runtime_spec = source_rollout_input_summary.get(
                "effective_runtime_spec"
            )
            if isinstance(raw_executed_effective_runtime_spec, Mapping):
                executed_effective_runtime_spec = normalize_effective_runtime_spec(
                    raw_executed_effective_runtime_spec,
                    context="source_rollout_input_summary.effective_runtime_spec",
                    runtime_spec_schema_version=self.impl.EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
                )
            else:
                executed_effective_runtime_spec = effective_runtime_spec_from_runtime_prompting(
                    executed_runtime_prompting,
                    variant=inputs.variant,
                    checkpoint_ref=inputs.checkpoint_ref,
                    key_files=self.impl.CHECKPOINT_INSTANCE_BINDING_KEY_FILES,
                    binding_schema_version=self.impl.CHECKPOINT_INSTANCE_BINDING_SCHEMA_VERSION,
                    runtime_spec_schema_version=self.impl.EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
                    context="source_rollout_input_summary.runtime_prompting",
                )
        except ValueError as exc:
            raise self.impl.FailFastError(str(exc)) from exc
        return ExecutedRuntimeBinding(
            runtime_prompting=executed_runtime_prompting,
            effective_runtime_spec=executed_effective_runtime_spec,
        )

    def _build_summary(
        self,
        *,
        inputs: V21EvalRequest,
        source_dir: Path,
        trace_outputs: TraceBuildOutputs,
        requested_runtime: RequestedRuntimeBinding,
        executed_runtime: ExecutedRuntimeBinding,
    ) -> dict[str, object]:
        summary = {
            "schema_version": self.impl.SUMMARY_SCHEMA_VERSION,
            "authority_id": str(inputs.eval_manifest["authority_id"]),
            "variant": inputs.variant,
            "checkpoint_ref": inputs.checkpoint_ref,
            "checkpoint_mode": inputs.checkpoint_mode,
            "output_dir": str(inputs.output_dir),
            "runtime_dir": str(inputs.runtime_dir),
            "source_rollout_dir": str(source_dir),
            "eval_manifest_id": inputs.eval_manifest_id,
            "eval_manifest_hash": inputs.eval_manifest_hash,
            "manifest_name": str(inputs.eval_manifest["manifest_name"]),
            "task_suite_name": str(inputs.eval_manifest["task_suite_name"]),
            "metric_profile": inputs.manifest_metric_profile,
            "primary_metric_id": trace_outputs.metric_ladder_summary[
                "primary_metric_id"
            ],
            "headline_metric_order": list(HEADLINE_METRIC_ORDER),
            "compatibility_only_metrics": list(COMPATIBILITY_ONLY_METRICS),
            "scope_audit": trace_outputs.scope_audit,
            "metric_ladder_summary": trace_outputs.metric_ladder_summary,
            "runtime_prompting": executed_runtime.runtime_prompting,
            "requested_runtime_prompting": requested_runtime.runtime_prompting,
            "effective_runtime_spec": executed_runtime.effective_runtime_spec,
            "requested_effective_runtime_spec": requested_runtime.effective_runtime_spec,
            "rollout_source_binding": {
                "source_selection_mode": (
                    "explicit_canonical_source_dir"
                    if inputs.canonical_source_dir is not None
                    else "variant_output_staging"
                ),
                "requested_runtime_prompting_matches_executed": (
                    requested_runtime.runtime_prompting
                    == executed_runtime.runtime_prompting
                ),
                "effective_runtime_spec_matches_requested": (
                    effective_runtime_surface_signature(
                        executed_runtime.effective_runtime_spec,
                        runtime_spec_schema_version=self.impl.EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
                    )
                    == effective_runtime_surface_signature(
                        requested_runtime.effective_runtime_spec,
                        runtime_spec_schema_version=self.impl.EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
                    )
                ),
                "effective_runtime_spec_hash": effective_runtime_spec_hash(
                    executed_runtime.effective_runtime_spec
                ),
            },
            "required_outputs": inputs.output_paths.required_outputs(),
            "deviation_notes": list(trace_outputs.deviation_notes),
        }
        try:
            assert_variant_aggregate_conservation_v21(
                trace_rows=trace_outputs.trace_rows,
                metric_ladder_summary=trace_outputs.metric_ladder_summary,
                bootstrap_ci=trace_outputs.bootstrap_ci,
                summary=summary,
            )
        except AggregationValidationError as exc:
            raise self.impl.FailFastError(str(exc)) from exc
        return summary

    def _write_authority_bundle(
        self,
        *,
        inputs: V21EvalRequest,
        trace_outputs: TraceBuildOutputs,
        summary: Mapping[str, object],
    ) -> None:
        eval_manifest_payload = {
            **inputs.eval_manifest,
            "eval_manifest_hash": inputs.eval_manifest_hash,
            "eval_manifest_id": inputs.eval_manifest_id,
        }
        write_json(inputs.output_paths.eval_manifest, eval_manifest_payload)
        write_jsonl(inputs.output_paths.per_episode_trace, trace_outputs.trace_rows)
        write_json(
            inputs.output_paths.metric_ladder_summary,
            trace_outputs.metric_ladder_summary,
        )
        write_json(inputs.output_paths.bootstrap_ci, trace_outputs.bootstrap_ci)
        write_json(inputs.output_paths.pairwise_delta, trace_outputs.pairwise_delta)
        write_markdown(
            inputs.output_paths.deviation_notes,
            self.impl._deviation_notes_markdown(trace_outputs.deviation_notes),
        )
        write_json(inputs.output_paths.summary, summary)
        self.impl._log(
            f"summary_json={inputs.output_paths.summary}",
            log_path=inputs.log_path,
        )
        self.impl._log("LIBERO_ROLLOUT_EVAL_DONE", log_path=inputs.log_path)


@dataclass(frozen=True)
class OpenPIEvalApp:
    """Route canonical eval scenarios to their workflow implementations.

    Callers depend on this app-level facade so the public entry path stays at
    Scenario -> App -> Workflow rather than importing internal workflow wiring
    or dependency assembly details directly.
    """

    def run_tracked_gate_evaluation(
        self,
        scenario: TrackedGateEvaluationScenario,
    ) -> int:
        return TrackedGateEvaluationWorkflow(scenario).run()

    def run_rollout_evaluation(
        self,
        scenario: RolloutEvaluationScenario,
        *,
        dependencies: Any | None = None,
    ) -> int:
        impl = _rollout_support()
        resolved_dependencies = (
            impl._default_dependencies() if dependencies is None else dependencies
        )
        if scenario.runtime.runtime_root is not None:
            runtime_root = scenario.runtime.runtime_root.resolve()

            def _build_runtime_dir(variant: str, eval_manifest_id: str) -> Path:
                return runtime_root / "rollouts" / variant / eval_manifest_id

            resolved_dependencies = replace(
                resolved_dependencies,
                build_runtime_dir=_build_runtime_dir,
            )
        return RolloutEvaluationWorkflow(
            scenario=scenario,
            dependencies=resolved_dependencies,
            impl=impl,
        ).run()

    def run_stock_smoke(self, scenario: StockSmokeScenario) -> int:
        from work.openpi.eval.workflows.stock_smoke import StockSmokeWorkflow

        return StockSmokeWorkflow(scenario).run()


__all__ = [
    "OpenPIEvalApp",
    "RolloutEvaluationWorkflow",
    "TrackedGateEvaluationWorkflow",
]
