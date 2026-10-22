from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from marie.registry.model_catalog import ModelCatalog
from marie.registry.model_registry import ModelRegistry


@pytest.mark.parametrize("uri", ["hf://acme/vision", "transformer://acme/vision"])
def test_hugging_face_metadata_downloads_only_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uri: str
) -> None:
    import huggingface_hub

    config = tmp_path / "config.json"
    config.write_text('{"vision_config": {"hidden_size": 32}}')

    def download(
        *, repo_id: str, filename: str, revision: str, local_files_only: bool
    ) -> str:
        from huggingface_hub.errors import EntryNotFoundError

        assert repo_id == "acme/vision"
        assert revision == "pinned-revision"
        assert local_files_only is True
        if filename != "config.json":
            raise EntryNotFoundError("Optional metadata file absent")
        return str(config)

    def download_weights(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Capability detection must not download a model snapshot")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download_weights)
    metadata = ModelRegistry.metadata(uri, "pinned-revision", local_files_only=True)
    assert metadata.config == {"vision_config": {"hidden_size": 32}}


def test_native_model_metadata_uses_existing_uri_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "private-model"
    model.mkdir()
    (model / "marie.json").write_text('{"_name_or_path": "private-model"}')
    (model / "config.json").write_text('{"vision_config": {"hidden_size": 32}}')
    handler = ModelRegistry.get_handler("zoo://private-model")
    monkeypatch.setattr(handler, "discovered", False)
    monkeypatch.setattr(handler, "resolved_models", {})

    catalog_path = tmp_path / "models.yaml"
    catalog_path.write_text(
        "schema_version: 1\nmodels:\n  private:\n    name: zoo://private-model\n"
    )
    catalog = ModelCatalog(
        catalog_path, partial(ModelRegistry.metadata, __model_path__=str(tmp_path))
    )
    assert catalog.is_multimodal("private") is True
    assert catalog.is_multimodal("zoo://private-model") is True

    (model / "config.json").write_text('{"model_type": "qwen2"}')
    assert catalog.is_multimodal("private") is True
    fresh_catalog = ModelCatalog(
        catalog_path, partial(ModelRegistry.metadata, __model_path__=str(tmp_path))
    )
    assert fresh_catalog.is_multimodal("private") is False


@pytest.mark.parametrize("prefix", ["", "file://"])
def test_local_metadata_can_use_marie_definition(tmp_path: Path, prefix: str) -> None:
    (tmp_path / "marie.json").write_text('{"model_type": "qwen2"}')
    assert ModelRegistry.metadata(f"{prefix}{tmp_path}").config == {
        "model_type": "qwen2"
    }


def test_non_object_metadata_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(["invalid"]))
    with pytest.raises(ValueError, match="must be an object"):
        ModelRegistry.metadata(str(tmp_path))
