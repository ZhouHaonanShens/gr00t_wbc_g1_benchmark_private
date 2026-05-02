from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_module(filename: str, module_name: str):
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_interface_localization_scripts_remain_spec_loadable() -> None:
    pack = _load_module(
        "interface_localization_pack.py",
        "interface_localization_pack_wrapper_test",
    )
    text_map = _load_module(
        "interface_localization_text_rewrite_map.py",
        "interface_localization_text_rewrite_map_wrapper_test",
    )

    assert pack.build_parser().prog == "interface_localization_pack.py"
    assert text_map.build_parser().prog == "interface_localization_text_rewrite_map.py"
    assert callable(pack.main)
    assert callable(text_map.main)


def test_pack_wrapper_help_and_bad_input_remain_clean(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack = _load_module(
        "interface_localization_pack.py",
        "interface_localization_pack_wrapper_bad_input_test",
    )

    with pytest.raises(SystemExit) as exc_info:
        pack.main(["--help"])
    assert exc_info.value.code == 0

    exit_code = pack.main(
        [
            "--input-dir",
            str(tmp_path / "missing_inputs"),
            "--output-dir",
            str(tmp_path / "out"),
            "--runtime-log-dir",
            str(tmp_path / "runtime"),
            "--evidence-json",
            str(tmp_path / "task-9-interface-localization-pack.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing required input file" in captured.err
    assert "Traceback" not in captured.err


def test_text_rewrite_map_wrapper_main_writes_artifact(tmp_path: Path) -> None:
    text_map = _load_module(
        "interface_localization_text_rewrite_map.py",
        "interface_localization_text_rewrite_map_wrapper_main_test",
    )
    output_dir = tmp_path / "interface_localization_sprint"

    exit_code = text_map.main(["--output-dir", str(output_dir)])

    assert exit_code == 0
    payload = _read_json(output_dir / text_map.TEXT_REWRITE_MAP_JSON_NAME)
    backpointer = payload["backpointer"]
    assert isinstance(backpointer, dict)
    assert payload["schema_version"] == text_map.TEXT_REWRITE_MAP_SCHEMA_VERSION
    assert payload["artifact_kind"] == text_map.TEXT_REWRITE_MAP_ARTIFACT_KIND
    assert backpointer["writer_script"] == (
        "work/recap/scripts/interface_localization_text_rewrite_map.py"
    )
