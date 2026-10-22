from __future__ import annotations

from pathlib import Path

import pytest


def test_local_model_metadata_includes_processor_and_generation(tmp_path: Path) -> None:
    from marie.registry.model_registry import ModelRegistry

    (tmp_path / "config.json").write_text('{"vision_config": {"hidden_size": 32}}')
    (tmp_path / "preprocessor_config.json").write_text('{"min_pixels": 784}')
    (tmp_path / "tokenizer_config.json").write_text(
        '{"chat_template": "native template"}'
    )
    (tmp_path / "generation_config.json").write_text('{"max_new_tokens": 256}')
    data = ModelRegistry.metadata(str(tmp_path))
    assert data.config == {"vision_config": {"hidden_size": 32}}
    assert data.processor == {"min_pixels": 784}
    assert data.tokenizer == {"chat_template": "native template"}
    assert data.generation == {"max_new_tokens": 256}


def test_metadata_uses_registered_handler_instead_of_parallel_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    from marie.registry.model_config import ModelMetadata
    from marie.registry.model_registry import ModelRegistry, ModelRegistryHandler

    class MetadataOnlyHandler(ModelRegistryHandler):
        def _get_supported_prefixes(self):
            return ["custom-metadata://"]

        def get_local_path(self, *args, **kwargs):
            raise AssertionError(
                "Metadata must use the existing handler metadata operation"
            )

        def get_metadata(self, name, version=None, **kwargs):
            assert name == "custom-metadata://model"
            assert version == "selected-revision"
            return ModelMetadata(config={"vision_config": {}}, revision=version)

    monkeypatch.setattr(
        ModelRegistry, "_PATH_HANDLERS", ModelRegistry._PATH_HANDLERS.copy()
    )
    ModelRegistry.register_handler(MetadataOnlyHandler())
    metadata = ModelRegistry.metadata("custom-metadata://model", "selected-revision")
    assert metadata.config == {"vision_config": {}}
    assert metadata.revision == "selected-revision"


def test_reference_uses_registered_handler(monkeypatch) -> None:
    from marie.registry.model_registry import ModelRegistry, ModelRegistryHandler

    class ReferenceOnlyHandler(ModelRegistryHandler):
        def _get_supported_prefixes(self):
            return ["custom-reference://"]

        def get_local_path(self, *args, **kwargs):
            raise AssertionError("Inference reference must use the registered handler")

        def get_reference(self, name, version=None):
            return "provider-resolved-reference"

    monkeypatch.setattr(
        ModelRegistry, "_PATH_HANDLERS", ModelRegistry._PATH_HANDLERS.copy()
    )
    ModelRegistry.register_handler(ReferenceOnlyHandler())
    assert (
        ModelRegistry.resolve_reference("custom-reference://model")
        == "provider-resolved-reference"
    )


def test_hugging_face_handler_pins_auxiliary_files_to_config_revision(
    tmp_path, monkeypatch
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError
    from marie.registry.model_registry import ModelRegistry

    commit = "a" * 40
    snapshot = tmp_path / commit
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"vision_config": {}}')
    (snapshot / "tokenizer_config.json").write_text('{"chat_template": "template"}')
    seen = []

    def download(*, repo_id, filename, revision, local_files_only):
        seen.append((repo_id, filename, revision, local_files_only))
        path = snapshot / filename
        if not path.is_file():
            raise EntryNotFoundError("Missing optional metadata")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kw: pytest.fail("Must not fetch weights"),
    )
    metadata = ModelRegistry.metadata("hf://acme/vision", "main", local_files_only=True)
    assert metadata.revision == commit
    assert metadata.tokenizer == {"chat_template": "template"}
    assert seen[0] == ("acme/vision", "config.json", "main", True)
    assert all(call[2] == commit for call in seen[1:])


def test_unqualified_hugging_face_id_routes_through_registered_handler(
    monkeypatch,
) -> None:
    from marie.registry.model_config import ModelMetadata
    from marie.registry.model_registry import ModelRegistry

    hf = ModelRegistry.get_handler("hf://acme/vision")
    seen = []
    monkeypatch.setattr(
        ModelRegistry._NATIVE_PATH_HANDLER, "get_metadata", lambda *a, **kw: None
    )

    def metadata(name, version=None, **kwargs):
        seen.append((name, version, kwargs))
        return ModelMetadata(config={"vision_config": {}})

    monkeypatch.setattr(hf, "get_metadata", metadata)
    assert ModelRegistry.metadata("acme/vision", local_files_only=True).config == {
        "vision_config": {}
    }
    assert seen == [("acme/vision", None, {"local_files_only": True})]


@pytest.mark.parametrize(
    "name", ["file:///missing/model", "zoo://missing", "/missing/model", "./missing"]
)
def test_missing_local_models_never_fall_back_to_hugging_face(
    name, monkeypatch
) -> None:
    from marie.registry.model_registry import ModelRegistry

    hf = ModelRegistry.get_handler("hf://acme/vision")
    monkeypatch.setattr(
        hf,
        "get_metadata",
        lambda *a, **kw: pytest.fail("Local references must remain local"),
    )
    with pytest.raises(FileNotFoundError):
        ModelRegistry.metadata(name)


@pytest.mark.parametrize("offline", [False, True])
def test_optional_uncached_metadata_only_tolerated_in_explicit_offline_mode(
    tmp_path, monkeypatch, offline
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from marie.registry.model_registry import ModelRegistry

    config = tmp_path / "config.json"
    config.write_text('{"vision_config": {}}')

    def download(*, filename, **kwargs):
        if filename == "config.json":
            return str(config)
        raise LocalEntryNotFoundError("Network unavailable and file not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    if offline:
        assert (
            ModelRegistry.metadata("hf://acme/vision", local_files_only=True).processor
            == {}
        )
    else:
        with pytest.raises(LocalEntryNotFoundError):
            ModelRegistry.metadata("hf://acme/vision")
