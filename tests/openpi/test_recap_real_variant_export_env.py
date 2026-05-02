from __future__ import annotations

import importlib
import os
from pathlib import Path


def test_subprocess_env_includes_openpi_client_src_for_override_root(
    tmp_path: Path, monkeypatch
) -> None:
    openpi_root = tmp_path / "openpi"
    openpi_src = openpi_root / "src"
    openpi_client_src = openpi_root / "packages" / "openpi-client" / "src"
    openpi_src.mkdir(parents=True)
    openpi_client_src.mkdir(parents=True)
    (openpi_root / ".venv" / "bin").mkdir(parents=True)
    (openpi_root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")

    import work.openpi.recap.real_variant_export as real_variant_export

    monkeypatch.setenv("OPENPI_ROOT_OVERRIDE", str(openpi_root))
    try:
        module = importlib.reload(real_variant_export)
        request = module.RealVariantExportRequest(
            variant="one_step_probe",
            variant_name="env_probe",
            dataset_dir=tmp_path / "datasets" / "formal_dataset",
            runtime_dir=tmp_path / "runtime",
            consumer_mode="prompt_text_with_recap_indicator",
            fixed_indicator_mode=None,
        )

        env = module._build_subprocess_env(request)

        pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)
        assert str(openpi_src.resolve()) in pythonpath_entries
        assert str(openpi_client_src.resolve()) in pythonpath_entries
        assert module.OPENPI_VENV_PYTHON == openpi_root / ".venv" / "bin" / "python"
    finally:
        monkeypatch.delenv("OPENPI_ROOT_OVERRIDE", raising=False)
        importlib.reload(real_variant_export)
