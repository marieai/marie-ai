from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_engine(
    tmp_path: Path, code: str, catalog: str | None = None, *, mounted: bool = False
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("MARIE_MODEL_CATALOG", None)
    env.pop("MARIE_DEFAULT_MOUNT", None)
    if catalog is not None:
        path = tmp_path / "config" / "models.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text("schema_version: 1\n" + catalog, encoding="utf-8")
        env["MARIE_DEFAULT_MOUNT" if mounted else "MARIE_MODEL_CATALOG"] = (
            str(tmp_path) if mounted else str(path)
        )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_engine_import_does_not_require_catalog(tmp_path: Path) -> None:
    result = _run_engine(
        tmp_path,
        "import marie.engine; from marie.engine import EngineLM; "
        "from marie.engine.completion_contract import CompletionCallParams; "
        "assert EngineLM.__module__ == 'marie.engine.base'",
    )
    assert result.returncode == 0, result.stderr


def test_lookup_without_configuration_reports_how_to_configure(tmp_path: Path) -> None:
    result = _run_engine(
        tmp_path,
        "from marie.engine import check_if_multimodal; check_if_multimodal('custom')",
    )
    assert result.returncode != 0
    assert "MARIE_MODEL_CATALOG" in result.stderr
    assert "MARIE_DEFAULT_MOUNT" in result.stderr


def test_missing_file_does_not_break_import_but_lookup_fails(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"
    code = (
        "import os; import marie.engine; "
        f"os.environ['MARIE_MODEL_CATALOG'] = {str(missing)!r}; "
        "from marie.engine import EngineLM; print('imported', flush=True); "
        "marie.engine.check_if_multimodal('custom')"
    )
    result = _run_engine(tmp_path, code)
    assert result.returncode != 0
    assert "imported" in result.stdout
    assert str(missing) in result.stderr


@pytest.mark.parametrize("mounted", [False, True])
def test_external_catalog_is_authoritative(tmp_path: Path, mounted: bool) -> None:
    result = _run_engine(
        tmp_path,
        "from marie.engine import MODEL_NAME_MAP, REVERSE_MODEL_NAME_MAP, "
        "__MULTIMODAL_MODELS__, check_if_multimodal; "
        "assert MODEL_NAME_MAP == {'vision': 'acme/vision', 'text': 'acme/text'}; "
        "assert REVERSE_MODEL_NAME_MAP == {'acme/vision': 'vision', 'acme/text': 'text'}; "
        "assert __MULTIMODAL_MODELS__ == ['acme/vision']; "
        "assert check_if_multimodal('vision'); assert check_if_multimodal('acme/vision'); "
        "assert not check_if_multimodal('text'); assert not check_if_multimodal('acme/text')",
        "models:\n  vision: {name: acme/vision, multimodal: true}\n"
        "  text: {name: acme/text, multimodal: false}\n",
        mounted=mounted,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_path_takes_precedence_over_mount(tmp_path: Path) -> None:
    result = _run_engine(
        tmp_path,
        "import os; os.environ['MARIE_DEFAULT_MOUNT'] = '/absent-mount'; "
        "from marie.engine import MODEL_NAME_MAP; assert MODEL_NAME_MAP == {'text': 'acme/text'}",
        "models:\n  text: {name: acme/text, multimodal: false}\n",
    )
    assert result.returncode == 0, result.stderr


def test_configuration_can_be_set_after_engine_import(tmp_path: Path) -> None:
    path = tmp_path / "later.yaml"
    path.write_text(
        "schema_version: 1\nmodels:\n  vision: {name: acme/vision, multimodal: true}\n"
    )
    result = _run_engine(
        tmp_path,
        "import os; from marie.engine import check_if_multimodal; "
        f"os.environ['MARIE_MODEL_CATALOG'] = {str(path)!r}; "
        "assert check_if_multimodal('vision')",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"model_type": "new_vision", "vision_config": {"hidden_size": 32}}, True),
        ({"model_type": "custom", "img_processor": {"image_dim_out": 32}}, True),
        ({"model_type": "qwen2_5_vl"}, True),
        ({"model_type": "qwen2"}, False),
    ],
)
def test_model_metadata_determines_capability(
    tmp_path: Path, metadata: dict, expected: bool
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(metadata))
    result = _run_engine(
        tmp_path,
        "from marie.engine import check_if_multimodal; "
        f"assert check_if_multimodal('local') is {expected}; "
        f"assert check_if_multimodal({str(model_dir)!r}) is {expected}",
        f"models:\n  local:\n    name: {model_dir}\n",
    )
    assert result.returncode == 0, result.stderr


def test_direct_model_path_does_not_need_catalog_entry(tmp_path: Path) -> None:
    model_dir = tmp_path / "unlisted-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"vision_config": {"hidden_size": 32}}')
    result = _run_engine(
        tmp_path,
        "from marie.engine import check_if_multimodal; "
        f"assert check_if_multimodal({str(model_dir)!r})",
        "models: {}\n",
    )
    assert result.returncode == 0, result.stderr


def test_explicit_capability_skips_metadata_resolution(tmp_path: Path) -> None:
    result = _run_engine(
        tmp_path,
        "from marie.engine import check_if_multimodal; "
        "assert check_if_multimodal('custom'); "
        "assert check_if_multimodal('acme/does-not-exist')",
        "models:\n  custom: {name: acme/does-not-exist, multimodal: true}\n",
    )
    assert result.returncode == 0, result.stderr


def test_unknown_metadata_requires_override(tmp_path: Path) -> None:
    model_dir = tmp_path / "unknown-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "unrecognized"}')
    result = _run_engine(
        tmp_path,
        "from marie.engine import check_if_multimodal; check_if_multimodal('custom')",
        f"models:\n  custom:\n    name: {model_dir}\n",
    )
    assert result.returncode != 0
    assert "multimodal" in result.stderr
    assert str(tmp_path / "config/models.yaml") in result.stderr


@pytest.mark.parametrize(
    "catalog",
    [
        "models: [broken",
        "models: []",
        "models:\n  custom: {name: '', multimodal: true}",
        "models:\n  custom: {name: acme/vision, multimodal: 'false'}",
        "models:\n  custom: {name: acme/vision, multi_modal: true}",
        "models:\n  '': {name: acme/vision}",
        "models:\n  custom: {name: acme/vision}\nunknown: true",
        "models:\n  one: {name: acme/model, multimodal: true}\n"
        "  two: {name: acme/model, multimodal: false}",
        "models:\n  one: {name: acme/model, multimodal: true}\n"
        "  acme/model: {name: acme/other, multimodal: true}",
    ],
)
def test_invalid_catalog_fails_with_source_path(tmp_path: Path, catalog: str) -> None:
    result = _run_engine(
        tmp_path,
        "from marie.engine import MODEL_NAME_MAP",
        catalog,
    )
    assert result.returncode != 0
    assert str(tmp_path / "config/models.yaml") in result.stderr


def test_same_model_aliases_share_explicit_capability(tmp_path: Path) -> None:
    result = _run_engine(
        tmp_path,
        "from marie.engine import check_if_multimodal; "
        "assert check_if_multimodal('one'); assert check_if_multimodal('two'); "
        "assert check_if_multimodal('acme/vision')",
        "models:\n  one: {name: acme/vision, multimodal: true}\n"
        "  two: {name: acme/vision}\n",
    )
    assert result.returncode == 0, result.stderr
