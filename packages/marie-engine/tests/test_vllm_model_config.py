from __future__ import annotations

import json
from pathlib import Path

import pytest
from marie.registry.model_catalog import ModelCatalog
from marie.registry.model_config import ModelMetadata

_FIXTURES = Path(__file__).resolve().parents[3] / "tests/unit/registry/fixtures"
_LEGACY = json.loads((_FIXTURES / "legacy-vllm-settings.json").read_text())["models"]


def _catalog() -> ModelCatalog:
    def metadata(name: str, revision: str | None) -> ModelMetadata:
        record = next(item for item in _LEGACY.values() if item["name"] == name)
        config = (
            {"vision_config": {}} if record["multimodal"] else {"model_type": "qwen2"}
        )
        return ModelMetadata(config=config, revision=revision)

    return ModelCatalog(_FIXTURES / "migrated-models.yaml", metadata)


@pytest.mark.parametrize(
    "alias", [key for key, item in _LEGACY.items() if item["factory"]]
)
@pytest.mark.parametrize("installed", [False, True])
def test_all_migrated_profiles_preserve_legacy_llm_arguments(
    alias: str, installed: bool
) -> None:
    from marie.engine.vllm_config import build_llm_kwargs

    config = _catalog().resolve_config(alias, "vllm")
    expected = _LEGACY[alias]["variants"]["bnb_present" if installed else "bnb_absent"]
    assert (
        build_llm_kwargs(config, dependency_available=lambda name: installed)
        == expected["llm_kwargs"]
    )
    assert config.prompt.template == expected["prompt"]
    assert config.processor == (expected["processor"] or {})
    assert config.generation["max_tokens"] == 4096
    assert config.generation["temperature"] == 0
    assert config.generation["stop_token_ids"] == expected["stop_token_ids"]


def test_new_model_runtime_and_processor_overrides_reach_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marie.engine.vllm_config as adapter

    path = tmp_path / "models.yaml"
    path.write_text("""schema_version: 1
defaults:
  vllm:
    loading: {quantization: null, load_format: auto, dtype: bfloat16}
    runtime: {max_model_len: 2048, gpu_memory_utilization: 0.6, tensor_parallel_size: 1}
    processor: {min_pixels: 784, max_pixels: 1003520, fps: 2}
models:
  new_model: {name: acme/never-listed-in-code, multimodal: true}
""")
    config = ModelCatalog(
        path, lambda name, revision: ModelMetadata(config={"vision_config": {}})
    ).resolve_config("new_model", "vllm")

    class RecordedLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(adapter, "LLM", RecordedLLM)
    llm = adapter.create_llm_instance(config)
    assert llm.kwargs["model"] == "acme/never-listed-in-code"
    assert llm.kwargs["max_model_len"] == 2048
    assert llm.kwargs["gpu_memory_utilization"] == 0.6
    assert llm.kwargs["mm_processor_kwargs"] == {
        "min_pixels": 784,
        "max_pixels": 1003520,
        "fps": 2,
    }


def test_disabled_profile_fails_before_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marie.engine.vllm_config as adapter

    def unexpected(**kwargs):
        pytest.fail("Disabled provider must not initialize a model")

    monkeypatch.setattr(adapter, "LLM", unexpected)
    config = _catalog().resolve_config("pixtral_12b", "vllm")
    with pytest.raises(ValueError, match="disabled"):
        adapter.create_llm_instance(config)


@pytest.mark.parametrize(
    "alias, field",
    [("mistral_7b", "dtype"), ("deepseek_r1_qwen_14b", "enable_reasoning")],
)
def test_legacy_runtime_incompatibility_is_explicit(alias: str, field: str) -> None:
    from marie.engine.vllm_config import create_llm_instance

    config = _catalog().resolve_config(alias, "vllm")
    with pytest.raises(ValueError, match=field):
        create_llm_instance(config)


@pytest.mark.parametrize(
    "alias", [key for key, item in _LEGACY.items() if item["factory"]]
)
def test_migrated_templates_format_exact_legacy_prompt(alias: str) -> None:
    from types import SimpleNamespace

    from marie.engine.vllm_engine import VLLMEngine

    expected_template = _LEGACY[alias]["variants"]["bnb_present"]["prompt"]
    engine = object.__new__(VLLMEngine)
    config = _catalog().resolve_config(alias, "vllm")
    engine.prompt = config.prompt.template
    engine.is_multimodal = config.multimodal
    engine.tokenizer = SimpleNamespace(
        apply_chat_template=lambda **kwargs: "native tokenizer template"
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "document question"}]}
    ]
    expected = (
        expected_template.replace(
            "SYSTEM_PROMPT_PLACEHOLDER", "document system"
        ).replace("QUESTION_PLACEHOLDER", "document question")
        if expected_template
        else "native tokenizer template"
    )
    assert engine._generate_prompt(messages, "document system") == expected
