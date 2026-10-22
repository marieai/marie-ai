from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_complete_registry_is_owned_by_registry_package(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("MARIE_DEFAULT_MOUNT", None)
    env.pop("MARIE_MODEL_CATALOG", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from marie.registry.model_registry import ModelRegistry; "
            "import marie.registry.model_registry as module; "
            "assert 'packages/marie-registry/src/' in module.__file__; "
            "assert 'marie.constants' not in sys.modules; "
            "assert 'marie.engine' not in sys.modules; "
            "assert 'vllm' not in sys.modules",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_configured_native_root_preserves_model_uri_resolution(tmp_path: Path) -> None:
    from marie.registry.model_registry import ModelRegistry

    model = tmp_path / "model_zoo" / "private-model"
    model.mkdir(parents=True)
    (model / "marie.json").write_text('{"_name_or_path": "private-model"}')
    ModelRegistry.configure(
        config_dir=tmp_path / "config", model_root=tmp_path / "model_zoo"
    )
    assert ModelRegistry.get("zoo://private-model") == str(model)
    assert ModelRegistry.get("model://private-model") == str(model)
    assert ModelRegistry.get(f"file://{model}") == str(model)


def test_changing_native_root_invalidates_catalog_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    from marie.registry.model_catalog import load_model_catalog
    from marie.registry.model_registry import ModelRegistry

    monkeypatch.delenv("MARIE_MODEL_CATALOG", raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        "schema_version: 1\nmodels:\n  private: {name: zoo://private}\n"
    )
    for root, body in [
        ("a", '{"vision_config": {}}'),
        ("b", '{"model_type": "qwen2"}'),
    ]:
        model = tmp_path / root / "private"
        model.mkdir(parents=True)
        (model / "marie.json").write_text('{"_name_or_path": "private"}')
        (model / "config.json").write_text(body)
    ModelRegistry.configure(config_dir=config_dir, model_root=tmp_path / "a")
    assert load_model_catalog().is_multimodal("private") is True
    ModelRegistry.configure(model_root=tmp_path / "b")
    assert load_model_catalog().is_multimodal("private") is False
