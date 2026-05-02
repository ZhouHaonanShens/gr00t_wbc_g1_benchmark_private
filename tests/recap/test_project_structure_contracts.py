from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DOC_PATH = REPO_ROOT / "agent/exchange/repo_placement_contract.md"
TEST_STATE_CONDITIONED_ENV_RESOLUTION_PATH = (
    REPO_ROOT / "tests/recap/test_state_conditioned_env_resolution.py"
)

SPEC_START_MARKER = "<!-- REPO_PLACEMENT_CONTRACT_SPEC_START -->"
SPEC_END_MARKER = "<!-- REPO_PLACEMENT_CONTRACT_SPEC_END -->"

EXPECTED_AGENT_RUN_ALLOWED_ROLES = (
    "public_cli",
    "thin_wrapper",
    "allowlisted_public_import_surface",
    "prompt_blocked_retained_survivor",
)

ALLOWLISTED_PUBLIC_IMPORT_SURFACES = frozenset(
    {"agent/run/state_conditioned_env_resolution.py"}
)
PROMPT_BLOCKED_RETAINED_SURVIVOR_PATHS = frozenset(
    {"agent/run/state_conditioned_teacher_upper_bound_gate.py"}
)
HISTORICAL_IMMUTABLE_RECORD_PATH_EXAMPLES = [
    "agent/logs/*.md",
    "agent/runtime_logs/**",
    "agent/artifacts/**",
    ".sisyphus/evidence/*",
    "agent/archive/**",
]
CURRENT_FACING_SUMMARY_DOC_PATHS = (
    "agent/exchange/package_migration_report.md",
    "agent/exchange/task16_tuesday_blocker_summary.md",
)

HISTORICAL_SCAN_EXEMPT_PREFIXES = (
    "agent/archive/",
    "agent/logs/",
    "agent/runtime_logs/",
    "agent/artifacts/",
    ".sisyphus/evidence/",
)
IGNORED_SCAN_PREFIXES = (
    ".git/",
    ".pytest_cache/",
    ".venv/",
    "submodules/",
)
IGNORED_PARTS = {"__pycache__"}

SURFACE_ALLOWED_SUFFIXES = {
    "agent/run": {".py", ".sh"},
    "work/**": {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".md"},
    "tests/**": {".py", ".md"},
    "agent/exchange/**": {".md", ".json"},
    "agent/logs/**": {".md"},
}


@dataclass(frozen=True)
class AgentRunImportViolation:
    source_path: str
    lineno: int
    import_kind: str
    target_module: str

    def render(self) -> str:
        return (
            f"{self.source_path}:{self.lineno}: forbidden {self.import_kind} "
            f"implementation import -> {self.target_module}"
        )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _repo_rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _extract_embedded_spec(markdown_text: str) -> dict[str, Any]:
    pattern = re.compile(
        re.escape(SPEC_START_MARKER)
        + r"\s*```json\s*(\{.*?\})\s*```\s*"
        + re.escape(SPEC_END_MARKER),
        re.DOTALL,
    )
    match = pattern.search(markdown_text)
    assert match is not None, (
        "repo placement contract is missing the embedded JSON spec block"
    )
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict), (
        "repo placement contract spec must be a JSON object"
    )
    return dict(payload)


def _load_repo_placement_contract_spec() -> tuple[str, dict[str, Any]]:
    markdown_text = _read_text(CONTRACT_DOC_PATH)
    return markdown_text, _extract_embedded_spec(markdown_text)


def _iter_live_files() -> list[Path]:
    live_files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        rel = _repo_rel(path)
        if rel.startswith(IGNORED_SCAN_PREFIXES):
            continue
        live_files.append(path)
    return sorted(live_files)


def _iter_live_python_files() -> list[Path]:
    return [path for path in _iter_live_files() if path.suffix == ".py"]


def _is_historical_scan_exempt(rel_path: str) -> bool:
    return rel_path.startswith(HISTORICAL_SCAN_EXEMPT_PREFIXES)


def _surface_key_for_live_path(rel_path: str) -> str | None:
    if rel_path.startswith("agent/run/"):
        return "agent/run"
    if rel_path.startswith("work/"):
        return "work/**"
    if rel_path.startswith("tests/"):
        return "tests/**"
    if rel_path.startswith("agent/exchange/"):
        return "agent/exchange/**"
    if rel_path.startswith("agent/logs/"):
        return "agent/logs/**"
    if rel_path.startswith("agent/runtime_logs/"):
        return "agent/runtime_logs/**"
    if rel_path.startswith("agent/artifacts/"):
        return "agent/artifacts/**"
    return None


def _classify_agent_run_role(rel_path: str) -> str:
    assert rel_path.startswith("agent/run/")
    if rel_path in ALLOWLISTED_PUBLIC_IMPORT_SURFACES:
        return "allowlisted_public_import_surface"
    if rel_path in PROMPT_BLOCKED_RETAINED_SURVIVOR_PATHS:
        return "prompt_blocked_retained_survivor"
    if rel_path.endswith(".sh"):
        return "public_cli"
    return "thin_wrapper"


def _module_name_to_agent_run_path(module_name: str) -> str | None:
    if module_name == "agent.run":
        return None
    if not module_name.startswith("agent.run."):
        return None
    module_bits = module_name.removeprefix("agent.run.").split(".")
    return "agent/run/" + "/".join(module_bits) + ".py"


def _import_from_target_modules(node: ast.ImportFrom) -> list[str]:
    module_name = str(node.module or "")
    if module_name == "agent.run":
        target_modules: list[str] = []
        for alias in node.names:
            alias_name = str(alias.name)
            if alias_name == "*":
                target_modules.append("agent.run.*")
                continue
            target_modules.append(f"agent.run.{alias_name}")
        return target_modules
    if module_name.startswith("agent.run."):
        return [module_name]
    return []


def _scan_for_forbidden_agent_run_implementation_imports(
    source_text: str,
    *,
    source_path: str,
) -> list[AgentRunImportViolation]:
    tree = ast.parse(source_text, filename=source_path)
    violations: list[AgentRunImportViolation] = []
    source_is_test = source_path.startswith("tests/")
    source_is_prompt_blocked_survivor = (
        source_path in PROMPT_BLOCKED_RETAINED_SURVIVOR_PATHS
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for target_module in _import_from_target_modules(node):
                target_path = _module_name_to_agent_run_path(target_module)
                if target_path is None:
                    continue
                if source_is_test or source_is_prompt_blocked_survivor:
                    continue
                if target_path in ALLOWLISTED_PUBLIC_IMPORT_SURFACES:
                    continue
                violations.append(
                    AgentRunImportViolation(
                        source_path=source_path,
                        lineno=int(node.lineno),
                        import_kind="from",
                        target_module=str(target_module),
                    )
                )
            continue

        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            target_path = _module_name_to_agent_run_path(alias.name)
            if target_path is None:
                continue
            if source_is_test or source_is_prompt_blocked_survivor:
                continue
            if target_path in ALLOWLISTED_PUBLIC_IMPORT_SURFACES:
                continue
            violations.append(
                AgentRunImportViolation(
                    source_path=source_path,
                    lineno=int(node.lineno),
                    import_kind="import",
                    target_module=str(alias.name),
                )
            )
    return violations


def test_repo_placement_contract_freezes_required_categories_roles_and_guards() -> None:
    markdown_text, spec = _load_repo_placement_contract_spec()

    assert spec["contract_key"] == "repo_placement_contract"
    assert spec["schema_version"] == "repo_placement_contract_v1"
    assert spec["contract_status"] == "frozen_current_and_future_placement_rules"
    assert spec["authority_statement"]["single_source_of_truth"] == (
        "agent/exchange/repo_placement_contract.md"
    )
    assert spec["authority_statement"]["historical_docs_are_not_live_truth"] is True
    assert spec["scope"]["covers"] == [
        "agent/run",
        "work/**",
        "tests/**",
        "agent/exchange/**",
        "agent/logs/**",
        "agent/runtime_logs/**",
        "agent/artifacts/**",
        "historical_immutable_records",
    ]
    assert set(spec["placement_categories"].keys()) == {
        "agent/run",
        "work/**",
        "tests/**",
        "agent/exchange/**",
        "agent/logs/**",
        "agent/runtime_logs/**",
        "agent/artifacts/**",
        "historical_immutable_records",
    }
    assert spec["placement_categories"]["agent/run"]["allowed_roles"] == list(
        EXPECTED_AGENT_RUN_ALLOWED_ROLES
    )
    assert spec["agent_run_role_policy"]["allowed_role_enum"] == list(
        EXPECTED_AGENT_RUN_ALLOWED_ROLES
    )
    assert spec["agent_run_role_policy"]["new_business_logic_default_destination"] == (
        "work/**"
    )
    assert spec["negative_guard_targets"] == {
        "forbidden_run_to_run_imports": True,
        "forbidden_new_business_logic_in_agent_run": True,
        "forbidden_agent_run_as_default_landing_zone": True,
        "forbidden_manual_implementation_in_runtime_logs": True,
        "forbidden_manual_implementation_in_artifacts": True,
        "forbidden_retroactive_history_rewrite_requirement": True,
        "forbidden_interpreting_historical_mentions_as_live_consumer": True,
    }
    assert (
        spec["placement_categories"]["historical_immutable_records"][
            "rewrite_obligation"
        ]
        == "exempt"
    )
    assert (
        spec["placement_categories"]["historical_immutable_records"]["path_examples"]
        == HISTORICAL_IMMUTABLE_RECORD_PATH_EXAMPLES
    )
    assert "current-facing summary docs" in markdown_text
    for rel_path in CURRENT_FACING_SUMMARY_DOC_PATHS:
        assert (
            rel_path
            not in spec["placement_categories"]["historical_immutable_records"][
                "path_examples"
            ]
        )
        assert f"`{rel_path}`" in markdown_text
    assert spec["placement_categories"]["agent/run"]["backpointer_authorities"] == [
        "agent/exchange/strict_run_entrypoint_matrix.md",
        "agent/exchange/agent_run_wrapper_allowlist.md",
    ]
    assert "prompt_blocked_retained_survivor" in markdown_text
    assert "historical_immutable_records" in markdown_text


def test_live_tree_files_fit_frozen_repo_placement_surfaces() -> None:
    seen_surfaces: set[str] = set()
    for path in _iter_live_files():
        rel_path = _repo_rel(path)
        surface_key = _surface_key_for_live_path(rel_path)
        if surface_key is None:
            continue
        seen_surfaces.add(surface_key)
        if surface_key in {"agent/runtime_logs/**", "agent/artifacts/**"}:
            continue
        allowed_suffixes = SURFACE_ALLOWED_SUFFIXES.get(surface_key)
        if allowed_suffixes is None:
            continue
        assert path.suffix in allowed_suffixes, (
            f"{rel_path} does not fit the frozen {surface_key} placement suffix set "
            f"{sorted(allowed_suffixes)}"
        )

    assert {"agent/run", "work/**", "tests/**", "agent/exchange/**"}.issubset(
        seen_surfaces
    )


def test_agent_run_retained_boundary_roles_stay_distinct_and_script_only() -> None:
    agent_run_root = REPO_ROOT / "agent/run"
    unexpected_nested_entries: list[str] = []
    seen_roles: set[str] = set()

    for path in sorted(agent_run_root.rglob("*")):
        rel_path = _repo_rel(path)
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.is_dir():
            unexpected_nested_entries.append(rel_path)
            continue
        assert path.suffix in {".py", ".sh"}, (
            f"agent/run non-script payload: {rel_path}"
        )
        role = _classify_agent_run_role(rel_path)
        seen_roles.add(role)
        assert role in EXPECTED_AGENT_RUN_ALLOWED_ROLES

    assert unexpected_nested_entries == []
    assert seen_roles == set(EXPECTED_AGENT_RUN_ALLOWED_ROLES)
    assert _classify_agent_run_role("agent/run/11b_sm120_run_once.sh") == "public_cli"
    assert (
        _classify_agent_run_role("agent/run/pseudodemo_label_contract_checker.py")
        == "thin_wrapper"
    )
    assert (
        _classify_agent_run_role("agent/run/state_conditioned_env_resolution.py")
        == "allowlisted_public_import_surface"
    )
    assert (
        _classify_agent_run_role(
            "agent/run/state_conditioned_teacher_upper_bound_gate.py"
        )
        == "prompt_blocked_retained_survivor"
    )


def test_prompt_blocked_retained_survivors_are_explicitly_listed_and_live() -> None:
    assert PROMPT_BLOCKED_RETAINED_SURVIVOR_PATHS
    for rel_path in sorted(PROMPT_BLOCKED_RETAINED_SURVIVOR_PATHS):
        assert rel_path.startswith("agent/run/")
        assert (REPO_ROOT / rel_path).is_file(), (
            f"missing retained survivor: {rel_path}"
        )
        assert rel_path not in ALLOWLISTED_PUBLIC_IMPORT_SURFACES


def test_detector_flags_known_forbidden_agent_run_implementation_import_pattern() -> (
    None
):
    sample_source = (
        "from agent.run.state_conditioned_teacher_upper_bound_sanity "
        "import _overall_gate\n"
    )

    violations = _scan_for_forbidden_agent_run_implementation_imports(
        sample_source,
        source_path="agent/run/example_gate.py",
    )

    assert violations == [
        AgentRunImportViolation(
            source_path="agent/run/example_gate.py",
            lineno=1,
            import_kind="from",
            target_module="agent.run.state_conditioned_teacher_upper_bound_sanity",
        )
    ]


def test_detector_flags_known_forbidden_direct_agent_run_module_import_pattern() -> (
    None
):
    sample_source = "from agent.run import interface_localization_contract\n"

    violations = _scan_for_forbidden_agent_run_implementation_imports(
        sample_source,
        source_path="agent/run/example_trace.py",
    )

    assert violations == [
        AgentRunImportViolation(
            source_path="agent/run/example_trace.py",
            lineno=1,
            import_kind="from",
            target_module="agent.run.interface_localization_contract",
        )
    ]


def test_tests_importing_agent_run_module_surfaces_are_not_misclassified() -> None:
    violations = _scan_for_forbidden_agent_run_implementation_imports(
        _read_text(TEST_STATE_CONDITIONED_ENV_RESOLUTION_PATH),
        source_path="tests/recap/test_state_conditioned_env_resolution.py",
    )

    assert violations == []


def test_repo_wide_live_python_has_no_non_allowlisted_agent_run_implementation_imports() -> (
    None
):
    violations: list[AgentRunImportViolation] = []

    for path in _iter_live_python_files():
        rel_path = _repo_rel(path)
        if rel_path.startswith(IGNORED_SCAN_PREFIXES):
            continue
        if _is_historical_scan_exempt(rel_path):
            continue
        violations.extend(
            _scan_for_forbidden_agent_run_implementation_imports(
                _read_text(path),
                source_path=rel_path,
            )
        )

    assert violations == [], "\n".join(v.render() for v in violations)
