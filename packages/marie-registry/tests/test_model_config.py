from __future__ import annotations

from pathlib import Path

import pytest
from marie.registry.model_catalog import ModelCatalog
from marie.registry.model_config import ModelMetadata


def _catalog(path: Path) -> ModelCatalog:
    return ModelCatalog(
        path, lambda name, revision: ModelMetadata(config={"vision_config": {}})
    )


def test_provider_sections_merge_with_explicit_values(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("""schema_version: 1
defaults:
  vllm:
    loading: {dtype: bfloat16}
    runtime: {max_model_len: 8192, tensor_parallel_size: 1, enforce_eager: true}
    processor: {min_pixels: 784, max_pixels: 1003520}
    generation: {temperature: 0.8, top_p: 1.0, max_tokens: 4096, stop_token_ids: [1]}
    prompt: {system: deployment prompt, template: original template}
profiles:
  vision:
    vllm:
      runtime: {max_model_len: 16384}
      processor: {fps: 1}
models:
  private:
    name: acme/private
    multimodal: true
    profile: vision
    providers:
      vllm:
        runtime: {tensor_parallel_size: 2, enforce_eager: false}
        generation: {temperature: 0, stop_token_ids: []}
        prompt: {template: null}
""")
    catalog = _catalog(path)
    result = catalog.resolve_config("private", "vllm")
    assert result.name == "acme/private"
    assert result.multimodal is True
    assert result.runtime == {
        "max_model_len": 16384,
        "tensor_parallel_size": 2,
        "enforce_eager": False,
    }
    assert result.processor == {"min_pixels": 784, "max_pixels": 1003520, "fps": 1}
    assert result.generation == {
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "stop_token_ids": [],
    }
    assert result.prompt.template is None
    assert result.prompt.system == "deployment prompt"
    result.runtime["max_model_len"] = 1
    assert catalog.resolve_config("private", "vllm").runtime["max_model_len"] == 16384


def test_constructor_overrides_are_section_scoped(tmp_path: Path) -> None:
    from marie.registry.model_config import ModelConfigOverrides

    path = tmp_path / "models.yaml"
    path.write_text("""schema_version: 1
defaults:
  vllm:
    runtime: {max_model_len: 8192}
    processor: {min_pixels: 784}
    generation: {max_tokens: 4096}
models:
  private: {name: acme/private, multimodal: false}
""")
    config = _catalog(path).resolve_config(
        "private",
        "vllm",
        overrides=ModelConfigOverrides(
            runtime={"max_model_len": 2048},
            processor={"min_pixels": 16},
            generation={"max_tokens": 32},
        ),
    )
    assert config.runtime == {"max_model_len": 2048}
    assert config.processor == {"min_pixels": 16}
    assert config.generation == {"max_tokens": 32}


@pytest.mark.parametrize(
    "body",
    [
        "schema_version: 2\nmodels: {}",
        "schema_version: 1\nmodels:\n  a: {name: acme/private, profile: missing}",
        "schema_version: 1\nmodels:\n  a: {name: acme/private}\n  a: {name: acme/other}",
        "schema_version: 1\ndefaults:\n  vllm:\n    runtime: {max_model_len: 0}\nmodels: {}",
        "schema_version: 1\nmodels:\n  a: {name: acme/private, revision: first}\n  b: {name: acme/private, revision: second}",
    ],
)
def test_invalid_profiles_report_catalog_path(tmp_path: Path, body: str) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=str(path)):
        ModelCatalog(path)


def test_disabled_provider_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("""schema_version: 1
models:
  private:
    name: acme/private
    multimodal: true
    providers:
      vllm: {enabled: false}
""")
    result = _catalog(path).resolve_config("private", "vllm")
    assert result.enabled is False


def test_resolved_metadata_is_independent_from_cache(tmp_path: Path) -> None:
    from marie.registry.model_config import ModelMetadata

    path = tmp_path / "models.yaml"
    path.write_text("schema_version: 1\nmodels:\n  private: {name: acme/private}\n")
    source = ModelMetadata(
        config={"vision_config": {}}, generation={"temperature": 0.4}
    )
    catalog = ModelCatalog(path, lambda name, revision: source)
    first = catalog.resolve_config("private", "vllm")
    first.metadata.generation["temperature"] = 0.9
    second = catalog.resolve_config("private", "vllm")
    assert second.generation["temperature"] == 0.4
    assert second.metadata.generation["temperature"] == 0.4


def test_image_defaults_and_processor_overrides_have_one_path(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("""schema_version: 1
defaults:
  vllm:
    image: {min_pixels: 784, max_pixels: 1003520}
profiles:
  vision:
    vllm:
      processor: {fps: 1}
models:
  private:
    name: acme/private
    multimodal: true
    profile: vision
    providers:
      vllm:
        processor: {max_pixels: 4096}
""")
    config = _catalog(path).resolve_config("private", "vllm")
    assert config.processor == {"fps": 1, "max_pixels": 4096}
    assert config.image == {"min_pixels": 784, "max_pixels": 4096, "fps": 1}


def test_resolved_revision_prefers_commit_observed_by_handler(tmp_path: Path) -> None:
    from marie.registry.model_config import ModelMetadata

    path = tmp_path / "models.yaml"
    path.write_text(
        "schema_version: 1\nmodels:\n  private: {name: hf://acme/private, revision: main}\n"
    )
    commit = "a" * 40
    catalog = ModelCatalog(
        path,
        lambda name, revision: ModelMetadata(
            config={"vision_config": {}}, revision=commit
        ),
    )
    assert catalog.resolve_config("private", "vllm").revision == commit


@pytest.mark.parametrize(
    "section",
    [
        "runtime: {model: another-model}",
        "runtime: {revision: mutable-tag}",
        "runtime: {mm_processor_kwargs: {fps: 1}}",
        "loading: {requires: [bitsandbytes]}",
        "loading: {fallback: invalid}",
        "generation: {max_new_tokens: 32}",
        "generation: {structured_outputs: invalid}",
    ],
)
def test_invalid_section_settings_fail_with_catalog_context(
    tmp_path: Path, section: str
) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        f"schema_version: 1\ndefaults:\n  vllm:\n    {section}\nmodels: {{}}\n"
    )
    with pytest.raises(ValueError, match=str(path)):
        ModelCatalog(path)


def test_explicit_capability_still_resolves_metadata_defaults_and_revision(
    tmp_path: Path,
) -> None:
    from marie.registry.model_config import ModelMetadata

    path = tmp_path / "models.yaml"
    path.write_text(
        "schema_version: 1\nmodels:\n  private: {name: hf://acme/private, multimodal: true, revision: main}\n"
    )
    calls = []

    def resolve(name, revision):
        calls.append((name, revision))
        return ModelMetadata(
            config={"vision_config": {}},
            generation={"max_new_tokens": 256},
            revision="a" * 40,
        )

    config = ModelCatalog(path, resolve).resolve_config("private", "vllm")
    assert config.generation["max_tokens"] == 256
    assert config.revision == "a" * 40
    assert calls == [("hf://acme/private", "main")]


def test_artifact_reference_uses_metadata_revision(tmp_path, monkeypatch) -> None:
    from marie.registry.model_registry import ModelRegistry, ModelRegistryHandler

    class Handler(ModelRegistryHandler):
        def _get_supported_prefixes(self):
            return ["pinned-artifact://"]

        def get_metadata(self, name, version=None, **kwargs):
            return ModelMetadata(config={"vision_config": {}}, revision="a" * 40)

        def get_reference(self, name, version=None):
            return f"/artifacts/{version}"

    monkeypatch.setattr(
        ModelRegistry, "_PATH_HANDLERS", ModelRegistry._PATH_HANDLERS.copy()
    )
    ModelRegistry.register_handler(Handler())
    path = tmp_path / "models.yaml"
    path.write_text(
        "schema_version: 1\nmodels:\n  private: {name: pinned-artifact://model, revision: main}\n"
    )
    config = ModelCatalog(path).resolve_config("private", "vllm")
    assert config.name == f"/artifacts/{config.revision}"


@pytest.mark.parametrize("alias", ["hf://acme/model", "transformer://acme/model"])
def test_prefixed_alias_cannot_shadow_canonical_model(tmp_path, alias) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        f"schema_version: 1\nmodels:\n  original: {{name: hf://acme/model}}\n  {alias}: {{name: acme/other}}\n"
    )
    with pytest.raises(ValueError, match="shadows"):
        ModelCatalog(path)
